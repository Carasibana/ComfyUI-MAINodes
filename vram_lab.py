# SPDX-License-Identifier: GPL-3.0-or-later
"""VRAM Lab (alpha, 2026-08-18): exact low-memory execution of MiniMax-H3 blocks.

The problem, measured 2026-08-17/18 on an RTX PRO 6000 (numbers below): a full-length H3 forward materialises, per DiT block, the fused
QKV projection ``[N, 3*7168]`` and the SwiGLU pre-activation ``[N, 2*14336]``
for the whole packed sequence. At ~217k tokens (an 8 s clip de-roped at
d_max 4) those two tensors are 8.7 GiB and 15.4 GiB, and the MLP-phase peak is
~24 GiB before weights. That is what OOMs 24 GB cards on the de-rope.

What this module does instead, per block, with the SAME math:

    phase 1  for each token chunk: h = mod(norm1(x_c)); qkv = qkv_proj(h);
             keep K, V (RMS-normed + RoPE'd) into full-sequence buffers,
             discard Q.
    phase 2  for each token chunk: recompute h and qkv, keep Q only;
             attention(Q_c, K, V) is exact per query row (flash / SDPA);
             out_proj; gated residual written in place into x[c].
    phase 3  for each token chunk: mod(norm2(x_c)) -> mlp -> gated residual.

Peak per block falls from ~N x 118 KB (MLP phase) to ~N x 39 KB (x + K + V)
plus chunk-sized transients. The projection is run twice (phase 1 and 2); at
217k tokens that is ~4% of the block's FLOPs, because attention is N^2 and
projections are N. This costs no exactness: the int8 kernel quantises
activations per row, so chunking rows is bit-identical (measured
2026-08-18 on comfy_kitchen TensorWiseINT8), and every other weight format
goes through the module's own forward, weight streaming included.

``kv_block > 0`` (experimental): phase 2 attends K/V in blocks and combines
with the flash kernel's log-sum-exp (online softmax), measured to one bf16
ulp against a single call. AS BUILT it does NOT lower memory: K and V are
still materialised in full by phase 1, and the blockwise combine adds
buffers (measured +13.0 GiB forward peak vs +11.9 plain at 216k tokens,
~7% slower). A real "turtle" needs K/V staged in host RAM and streamed per
block; until then leave it at 0.

Composition: MLP chunking here is optional (``mlp_chunk = 0`` leaves the
model's own ``mlp.forward`` alone, so KJNodes' MiniMax H3 Chunk FeedForward
keeps working). Attention goes through ``optimized_attention`` so Sol-Attn
style overrides still apply; head-group chunking from KJNodes' Low VRAM
Attention is read from ``transformer_options["minimax_head_chunks"]``.

Not a quality dial. If it changes the picture, it is a bug; the gate is a
same-seed difference clip against the stock block.
"""
import logging
import math
import os
import time

import torch

import comfy.model_management
import comfy.quant_ops
from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention

log = logging.getLogger("MAINodes.vram_lab")


# --------------------------------------------------------------------------- helpers

def _ranges(n, size, min_tail=512):
    """Row chunks of `size`; a tail shorter than `min_tail` is folded into the
    previous chunk. Row-wise ops are exact at any chunk size, but comfy_kitchen
    picks a different int8 GEMM kernel for m <= 128 rows (`_prefer_turing_fused_int8`)
    and its fp32 epilogue is not proven identical across kernels, so never emit
    a tiny tail (2026-08-18)."""
    if size <= 0 or size >= n:
        return [(0, n)]
    out = [(a, min(a + size, n)) for a in range(0, n, size)]
    if len(out) > 1 and out[-1][1] - out[-1][0] < min_tail:
        a, _ = out.pop()
        out[-1] = (out[-1][0], n)
    return out


_SM_CACHE = {}


def _sm_count(device):
    key = str(device)
    if key not in _SM_CACHE:
        try:
            _SM_CACHE[key] = torch.cuda.get_device_properties(device).multi_processor_count
        except Exception:
            _SM_CACHE[key] = 128
    return _SM_CACHE[key]


def _min_query_chunk(device, heads_per_call, q_block=128, margin=2):
    """Smallest query chunk that keeps the flash kernel on its non-split path.

    Measured 2026-08-18 on an RTX PRO 6000 (188 SMs, 56 heads): a 259-query
    tail (168 query blocks < 188 SMs) made SDPA take its split-KV path and the
    result stopped being bit-equal to the full-length call (43% of elements
    off by ~2e-4, which diffusion amplified into a visibly different clip);
    512 queries (224 blocks) was bit-equal. Query chunking is exact only while
    every chunk has at least ~SMs query blocks, so we require margin x SMs.

    Measured boundary (same card, 215k K/V, 2026-08-18): flash leaves its split-KV path exactly when
    heads_per_call x ceil(L/64) >= 0.8 x 2 x SMs (PyTorch flash_api.cpp
    set_params_splitkv / num_splits_heuristic): L >= 321 / 641 / 1345 for
    56 / 28 / 14 heads. This function returns 896 / 1792 / 3456 there, a
    2.6-2.8x margin over the measured line, so it is not the constraint on
    chunk size in practice. The heuristic constants are PyTorch's and can
    move; self_check stays the authority. cuDNN and mem-efficient SDPA are
    chunk-invariant at every length (and 1.1-2x slower); no two backends are
    bit-equal to each other, and stock renders run flash, so we stay on flash.
    """
    return q_block * math.ceil(margin * _sm_count(device) / max(1, heads_per_call))


def _balanced_ranges(n, size, min_size):
    """Split [0, n) into chunks of at most `size` tokens, all of length >= min_size
    (the tail is folded into its neighbour rather than left short). One chunk if
    n < 2 * min_size."""
    if size <= 0 or size >= n or n < 2 * min_size:
        return [(0, n)]
    parts = max(1, math.ceil(n / size))
    while parts > 1 and n / parts < min_size:
        parts -= 1
    base, extra = divmod(n, parts)
    out, a = [], 0
    for i in range(parts):
        b = a + base + (1 if i < extra else 0)
        out.append((a, b))
        a = b
    return out


def _mod_scale_shift_range(h, shift, scale, segments, c0, c1):
    """h is norm(x[c0:c1]); apply the per-segment affine restricted to [c0, c1)."""
    for a, b, row in segments:
        lo, hi = max(a, c0), min(b, c1)
        if lo < hi:
            h[lo - c0:hi - c0].mul_(1.0 + scale[row].to(h.dtype)).add_(shift[row].to(h.dtype))
    return h


def _mod_gate_range(x, gate, other, segments, c0, c1):
    """x[c0:c1] += gate[row] * other, per segment, in place on the full residual."""
    for a, b, row in segments:
        lo, hi = max(a, c0), min(b, c1)
        if lo < hi:
            x[lo:hi].addcmul_(other[lo - c0:hi - c0], gate[row].to(x.dtype))
    return x


def _norm_rope(attn, q, k, rope_c, s):
    """Mirror of Attention.forward's fused RMSNorm + split-half RoPE on a chunk.

    q, k: [s, heads*hd] views into the chunk's qkv buffer (split on last dim,
    viewable as [1, s, heads, hd] exactly as core does). Returns [s, heads, hd].
    """
    heads, hd = attn.heads, attn.head_dim
    if rope_c is not None:
        q = q.view(1, s, heads, hd)
        k = k.view(1, s, heads, hd)
        qw = comfy.model_management.cast_to(attn.q_norm.weight, device=q.device)
        kw = comfy.model_management.cast_to(attn.k_norm.weight, device=q.device)
        rot = rope_c.shape[-3] * 2
        comfy.quant_ops.ck.rms_rope_split_half_(q, k, rope_c, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot)
        return q[0], k[0]
    return attn.q_norm(q.view(s, heads, hd)), attn.k_norm(k.view(s, heads, hd))


def _blockwise_attention(qc, K, V, block, out_dtype):
    """Exact attention of qc [1,H,c,hd] against K,V [1,H,S,hd] in K/V blocks.

    Uses the flash kernel's per-row logsumexp for the online-softmax combine.
    fp32 accumulator; result cast to out_dtype. Slow on purpose.
    """
    op = torch.ops.aten._scaled_dot_product_flash_attention.default
    S = K.shape[2]
    acc = None
    lse_acc = None
    for a, b in _ranges(S, block):
        r = op(qc, K[:, :, a:b].contiguous(), V[:, :, a:b].contiguous(), 0.0, False, False)
        o, lse = r[0], r[1]                       # o [1,H,c,hd], lse [1,H,c] fp32
        if acc is None:
            acc, lse_acc = o.float(), lse
        else:
            new = torch.logaddexp(lse_acc, lse)
            acc = acc * torch.exp(lse_acc - new).unsqueeze(-1) + o.float() * torch.exp(lse - new).unsqueeze(-1)
            lse_acc = new
        del r, o, lse
    return acc.to(out_dtype)


def _attend(qc, K, V, heads, head_chunks, transformer_options):
    """qc [1,H,c,hd]; K,V [1,H,S,hd] -> [1, c, H*hd]. Honors KJ-style head groups."""
    hd = qc.shape[-1]
    n = max(1, min(int(head_chunks or 1), heads))
    if n <= 1:
        return optimized_attention(AttentionTensorContainer(qc), AttentionTensorContainer(K),
                                   AttentionTensorContainer(V), heads, mask=None, skip_reshape=True,
                                   transformer_options=transformer_options)
    c = qc.shape[2]
    out = torch.empty((1, c, heads * hd), dtype=qc.dtype, device=qc.device)
    hs = 0
    sizes = [heads // n + (1 if i < heads % n else 0) for i in range(n)]
    for size in sizes:
        he = hs + size
        o = optimized_attention(AttentionTensorContainer(qc[:, hs:he]), AttentionTensorContainer(K[:, hs:he]),
                                AttentionTensorContainer(V[:, hs:he]), size, mask=None, skip_reshape=True,
                                transformer_options=transformer_options)
        out[:, :, hs * hd:he * hd] = o
        hs = he
    return out


# --------------------------------------------------------------------------- the block

def _phase1_kv(block, x, shift_msa, scale_msa, mod_segments, rope_freqs, kv_chunk, heads, hd, inner):
    """K, V for the whole sequence, chunk by chunk (Q computed and dropped)."""
    S = x.shape[0]
    attn = block.attn
    K = torch.empty((1, heads, S, hd), dtype=x.dtype, device=x.device)
    V = torch.empty((1, heads, S, hd), dtype=x.dtype, device=x.device)
    for a, b in _ranges(S, kv_chunk):
        s = b - a
        h = _mod_scale_shift_range(block.norm1(x[a:b]), shift_msa, scale_msa, mod_segments, a, b)
        qkv = attn.qkv_proj(h)
        q, k, v = qkv.split(inner, dim=-1)
        rope_c = rope_freqs[:, a:b].contiguous() if rope_freqs is not None else None
        _, k = _norm_rope(attn, q, k, rope_c, s)
        K[0, :, a:b] = k.transpose(0, 1)
        V[0, :, a:b] = v.view(s, heads, hd).transpose(0, 1)
        del h, qkv, q, k, v, rope_c
    return K, V


def _phase2_q_attn(block, x, K, V, shift_msa, scale_msa, gate_msa, mod_segments, rope_freqs,
                   transformer_options, q_chunk, kv_block, heads, hd, inner, head_chunks):
    """Q per chunk, attention against full K/V, out_proj, gated residual in place."""
    S = x.shape[0]
    attn = block.attn
    heads_per_call = max(1, heads // max(1, min(int(head_chunks or 1), heads)))
    q_ranges = _balanced_ranges(S, q_chunk, _min_query_chunk(x.device, heads_per_call))
    for a, b in q_ranges:
        s = b - a
        h = _mod_scale_shift_range(block.norm1(x[a:b]), shift_msa, scale_msa, mod_segments, a, b)
        qkv = attn.qkv_proj(h)
        q, k, _v = qkv.split(inner, dim=-1)
        rope_c = rope_freqs[:, a:b].contiguous() if rope_freqs is not None else None
        q, _ = _norm_rope(attn, q, k, rope_c, s)
        qc = q.transpose(0, 1).unsqueeze(0).contiguous()      # [1, heads, s, hd]
        del h, qkv, q, k, _v, rope_c
        if kv_block and kv_block > 0:
            o = _blockwise_attention(qc, K, V, kv_block, x.dtype)          # [1, heads, s, hd]
            o = o.transpose(1, 2).reshape(1, s, inner)
        else:
            o = _attend(qc, K, V, heads, head_chunks, transformer_options)  # [1, s, inner]
        del qc
        o = attn.out_proj(o.squeeze(0))
        _mod_gate_range(x, gate_msa, o, mod_segments, a, b)
        del o


def _phase3_mlp(block, x, shift_mlp, scale_mlp, gate_mlp, mod_segments, mlp_chunk):
    """MLP per chunk, gated residual in place."""
    S = x.shape[0]
    for a, b in _ranges(S, mlp_chunk):
        h = _mod_scale_shift_range(block.norm2(x[a:b]), shift_mlp, scale_mlp, mod_segments, a, b)
        o = block.mlp(h)
        _mod_gate_range(x, gate_mlp, o, mod_segments, a, b)
        del h, o


def streamed_block_forward(block, x, t_emb, mod_segments, rope_freqs, transformer_options,
                           q_chunk=16384, kv_chunk=16384, mlp_chunk=16384, kv_block=0, probe=None, index=None):
    """Exact replacement for DiTBlock.forward with chunk-bounded transients.

    The three phases are separate named functions so an allocator trace
    (torch.cuda.memory._record_memory_history / memory_viz, see H3MemoryProbe)
    labels every band by phase from the Python stack alone; `probe` (an
    H3MemoryProbe ledger, read from transformer_options["h3_memprobe"]) gets a
    zero-sync mark after each phase.
    """
    attn = block.attn
    heads, hd = attn.heads, attn.head_dim
    inner = heads * hd
    head_chunks = transformer_options.get("minimax_head_chunks", 1) if isinstance(transformer_options, dict) else 1

    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaln_proj(t_emb)

    K, V = _phase1_kv(block, x, shift_msa, scale_msa, mod_segments, rope_freqs, kv_chunk, heads, hd, inner)
    if probe is not None:
        probe.mark(index, "kv")
    _phase2_q_attn(block, x, K, V, shift_msa, scale_msa, gate_msa, mod_segments, rope_freqs,
                   transformer_options, q_chunk, kv_block, heads, hd, inner, head_chunks)
    del K, V
    if probe is not None:
        probe.mark(index, "attn")
    _phase3_mlp(block, x, shift_mlp, scale_mlp, gate_mlp, mod_segments, mlp_chunk)
    if probe is not None:
        probe.mark(index, "mlp")
    return x


def _self_check(block, args, extra, cfg, tag):
    """Run stock and streamed on the same input and log per-phase divergence. Diagnostic only."""
    x = args["img"]
    S = x.shape[0]
    ref = extra["original_block"](dict(args, img=x.clone()))["img"]
    scale = ref.float().abs().max().item()
    ulp = 2.0 ** (math.floor(math.log2(max(scale, 1e-30))) - 7)
    variants = {"full": cfg,
                "q_only": dict(cfg, kv_chunk=S, mlp_chunk=S),
                "kv_only": dict(cfg, q_chunk=S, mlp_chunk=S),
                "mlp_only": dict(cfg, q_chunk=S, kv_chunk=S),
                "none": dict(cfg, q_chunk=S, kv_chunk=S, mlp_chunk=S)}
    parts = []
    for name, c in variants.items():
        out = streamed_block_forward(block, x.clone(), args["t_emb"], args["mod_segments"], args["rope_freqs"],
                                     args["transformer_options"], q_chunk=c["q_chunk"], kv_chunk=c["kv_chunk"],
                                     mlp_chunk=c["mlp_chunk"], kv_block=c["kv_block"])
        d = (out.float() - ref.float()).abs()
        parts.append(f"{name}: max {d.max().item():.3e} ({d.max().item() / ulp:.2f} ulp) mean {d.mean().item():.2e}")
        del out, d
    to = args["transformer_options"]
    keys = sorted(k for k in to.keys()) if isinstance(to, dict) else type(to).__name__
    log.warning("H3StreamedBlocks self_check[%s] S=%d segs=%d rope=%s x=%s/%s to_keys=%s | %s",
                tag, S, len(args["mod_segments"]), tuple(args["rope_freqs"].shape) if args["rope_freqs"] is not None else None,
                x.dtype, x.is_contiguous(), keys, " | ".join(parts))
    del ref


def _make_replacement(block, cfg, index=0):
    state = {"checked": False}

    def fn(args, extra):
        x = args["img"]
        if x.shape[0] < cfg["min_tokens"]:
            return extra["original_block"](args)
        if cfg.get("self_check") and index == 0 and not state["checked"]:
            state["checked"] = True
            _self_check(block, args, extra, cfg, f"block{index}")
        to = args["transformer_options"]
        probe = to.get("h3_memprobe") if isinstance(to, dict) else None
        x = streamed_block_forward(block, x, args["t_emb"], args["mod_segments"], args["rope_freqs"],
                                   to, q_chunk=cfg["q_chunk"], kv_chunk=cfg["kv_chunk"],
                                   mlp_chunk=cfg["mlp_chunk"], kv_block=cfg["kv_block"], probe=probe, index=index)
        return {"img": x}
    return _named(fn, f"block{index:02d}")


def _named(fn, name):
    """Return `fn` under a new code-object name so it shows as `name` in Python
    stacks (allocator traces label bands by frame name; there is no other
    per-call label channel)."""
    import types
    code = fn.__code__.replace(co_name=name)
    g = types.FunctionType(code, fn.__globals__, name, fn.__defaults__, fn.__closure__)
    g.__qualname__ = name
    return g



# --------------------------------------------------------------------------- final layer

def streamed_final_layer_forward(fl, x, t_emb, video_seg, audio_seg, chunk=16384, probe=None, exact_gemm=True):
    """Row-chunked FinalLayer.forward. Stock (comfy/ldm/minimax/model.py:295)
    builds `norm(x[span]) * (1 + scale) + shift` for the whole target span and
    the fp32 modulation promotes it to fp32 twice: 2 x 4.28 GiB at 216k tokens,
    measured as the forward's peak on a 216k-token de-rope (2026-08-18). Every op
    is per row (RMSNorm, per-element mod, per-row fp32 linear), so the same
    math per chunk yields the same [rows, out] result with ~chunk-sized
    transients. Whether the fp32 cuBLAS GEMM stays bit-equal under a different
    M is a kernel property: gated, not assumed."""
    shift, scale = fl.adaln_proj(t_emb)

    def head(a, b, row, out_mod):
        n = b - a
        if exact_gemm:
            # exact tier: chunk only the norm/mod/fp32 promotion into ONE fp32 buffer,
            # then run the head GEMM at the stock M (same cuBLAS kernel -> bit-equal).
            # Transient: one [n, hidden] fp32 (4.28 GiB at 213k rows) instead of ~10.7.
            hbuf = torch.empty((n, x.shape[1]), dtype=torch.float32, device=x.device)
            for c0, c1 in _ranges(n, chunk):
                hbuf[c0:c1] = fl.norm(x[a + c0:a + c1]) * (1.0 + scale[row]) + shift[row]
            out = out_mod(hbuf)
            del hbuf
            return out
        # numerically-equivalent tier: chunk the GEMM too (fp32 cuBLAS picks kernels by M;
        # measured max |d| ~5e-6 vs stock on random weights). Transient ~chunk-sized.
        parts = []
        for c0, c1 in _ranges(n, chunk):
            h = (fl.norm(x[a + c0:a + c1]) * (1.0 + scale[row]) + shift[row]).to(torch.float32)
            parts.append(out_mod(h))
            del h
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)

    va, vb, vrow = video_seg
    aa, ab, arow = audio_seg
    v = head(va, vb, vrow, fl.video_out)
    a = head(aa, ab, arow, fl.audio_out)
    if probe is not None:
        probe.mark(None, "final")
    return v, a


# --------------------------------------------------------------------------- node

class H3StreamedBlocks:
    """Run every H3 DiT block in token chunks: exact, chunk-bounded VRAM."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "q_chunk": ("INT", {"default": 16384, "min": 1024, "max": 262144, "step": 1024,
                                    "tooltip": "Query tokens per attention call. Smaller = lower transient VRAM, more calls. Exact at any value."}),
                "kv_chunk": ("INT", {"default": 16384, "min": 1024, "max": 262144, "step": 1024,
                                     "tooltip": "Tokens per K/V projection chunk while building the full K/V buffers."}),
                "mlp_chunk": ("INT", {"default": 16384, "min": 0, "max": 262144, "step": 1024,
                                      "tooltip": "Tokens per MLP chunk. 0 = leave the model's mlp.forward alone (e.g. KJNodes' chunk node)."}),
                "min_tokens": ("INT", {"default": 32768, "min": 0, "max": 1048576, "step": 1024,
                                       "tooltip": "Below this packed sequence length the stock block runs (short clips gain nothing)."}),
                "kv_block": ("INT", {"default": 0, "min": 0, "max": 262144, "step": 1024,
                                     "tooltip": "EXPERIMENTAL, leave at 0. Attends K/V in blocks with a log-sum-exp combine (1 bf16 ulp). As built it does not lower memory (K/V are still fully built; measured +1.1 GiB and ~7% slower at 216k tokens); kept for the host-staged K/V design to come."}),
            },
            "optional": {
                "final_layer_chunk": ("INT", {"default": 16384, "min": 0, "max": 262144, "step": 1024,
                                              "tooltip": "Rows per chunk through the output head's norm -> mod -> fp32 promotion. Stock promotes the whole span to fp32 twice (~10 GiB at 216k tokens, the forward's peak). 0 = stock."}),
                "final_layer_gemm": (["exact (whole GEMM, one fp32 buffer)", "streamed (chunked GEMM, ~1e-6 fp32 diffs)"],
                                     {"default": "exact (whole GEMM, one fp32 buffer)",
                                      "tooltip": "exact: same head GEMM as stock, transient = one fp32 [rows, hidden] (bit-equal). streamed: GEMM per chunk, transient ~chunk-sized, but fp32 cuBLAS is not chunk-invariant (numerically-equivalent tier)."}),
                "self_check": ("BOOLEAN", {"default": False,
                                           "tooltip": "Diagnostic: on block 0's first call, run stock and streamed on the same input and log per-phase divergence. Costs one extra block forward."}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "MAINodes/VRAM Lab"
    DESCRIPTION = ("Exact low-VRAM execution of MiniMax-H3: never materialises the full-sequence "
                   "fused QKV or SwiGLU tensors. Same math as the stock block; costs ~4% extra "
                   "projection work at long lengths. See vram_lab.py for the ledger.")

    def patch(self, model, q_chunk, kv_chunk, mlp_chunk, min_tokens, kv_block, self_check=False, final_layer_chunk=16384,
              final_layer_gemm="exact (whole GEMM, one fp32 buffer)"):
        dm = getattr(getattr(model, "model", None), "diffusion_model", None)
        blocks = getattr(dm, "blocks", None)
        if not blocks or not hasattr(blocks[0], "attn") or not hasattr(blocks[0].attn, "qkv_proj"):
            log.warning("H3StreamedBlocks: model does not look like MiniMax H3 (no blocks[*].attn.qkv_proj); unchanged")
            return (model,)
        cfg = {"q_chunk": q_chunk, "kv_chunk": kv_chunk, "mlp_chunk": mlp_chunk,
               "min_tokens": min_tokens, "kv_block": kv_block, "self_check": bool(self_check)}
        m = model.clone()
        for i, block in enumerate(blocks):
            m.set_model_patch_replace(_make_replacement(block, cfg, i), "dit", "double_block", i)
        fl = getattr(dm, "final_layer", None)
        if final_layer_chunk and fl is not None and hasattr(fl, "video_out") and hasattr(fl, "audio_out"):
            _exact = str(final_layer_gemm).startswith("exact")

            def _fl_forward(x, t_emb, video_seg, audio_seg, _fl=fl, _c=int(final_layer_chunk), _e=_exact):
                if x.shape[0] < cfg["min_tokens"]:
                    return type(_fl).forward(_fl, x, t_emb, video_seg, audio_seg)
                return streamed_final_layer_forward(_fl, x, t_emb, video_seg, audio_seg, chunk=_c,
                                                    probe=getattr(dm, "_h3_memprobe", None), exact_gemm=_e)
            m.add_object_patch("diffusion_model.final_layer.forward", _fl_forward)
        log.info("H3StreamedBlocks: %d blocks patched (q %d, kv %d, mlp %d, min %d, kv_block %d, final_layer_chunk %d, %s)",
                 len(blocks), q_chunk, kv_chunk, mlp_chunk, min_tokens, kv_block, final_layer_chunk, final_layer_gemm)
        return (m,)



# --------------------------------------------------------------------------- memory probe

def _rss():
    """(RssAnon, RssFile) of this process in bytes from /proc/self/status; ~µs.
    RssAnon is the host RAM the process really holds (mirrors, retained
    allocator arenas, pinned buffers); RssFile is mmap'd model files."""
    anon = file = 0
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("RssAnon:"):
                    anon = int(line.split()[1]) * 1024
                elif line.startswith("RssFile:"):
                    file = int(line.split()[1]) * 1024
    except OSError:
        pass
    return anon, file


class _MemLedger:
    """Zero-sync memory ledger: reads the caching allocator's host-side counters
    (allocated, peak since last mark, reserved) at phase boundaries. No device
    sync: allocations happen at launch time, so the counters are exact on the
    host timeline; the timestamps are launch times, not completion times."""

    def __init__(self, path, dev):
        self.path = path
        self.dev = dev
        self.fwd = -1
        self.rows = []
        self.t0 = time.perf_counter()

    def begin_forward(self, shape):
        self.fwd += 1
        torch.cuda.reset_peak_memory_stats(self.dev)
        self._base = torch.cuda.memory_allocated(self.dev)
        self.rows.append({"fwd": self.fwd, "block": None, "phase": "start", "t": time.perf_counter() - self.t0,
                          "alloc": self._base, "peak": self._base,
                          "reserved": torch.cuda.memory_reserved(self.dev), "shape": list(shape),
                          "rss_anon": _rss()[0], "rss_file": _rss()[1]})

    def mark(self, block, phase):
        st = torch.cuda.memory_stats(self.dev)
        self.rows.append({"fwd": self.fwd, "block": block, "phase": phase, "t": time.perf_counter() - self.t0,
                          "alloc": st.get("allocated_bytes.all.current", 0),
                          "peak": st.get("allocated_bytes.all.peak", 0),
                          "reserved": st.get("reserved_bytes.all.current", 0),
                          "rss_anon": _rss()[0], "rss_file": _rss()[1]})
        torch.cuda.reset_peak_memory_stats(self.dev)

    def end_forward(self):
        st = torch.cuda.memory_stats(self.dev)
        self.rows.append({"fwd": self.fwd, "block": None, "phase": "end", "t": time.perf_counter() - self.t0,
                          "alloc": st.get("allocated_bytes.all.current", 0),
                          "peak": st.get("allocated_bytes.all.peak", 0),
                          "reserved": st.get("reserved_bytes.all.current", 0),
                          "rss_anon": _rss()[0], "rss_file": _rss()[1]})
        self.flush()

    def flush(self):
        import json
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            for r in self.rows:
                f.write(json.dumps(r) + "\n")
        try:
            with open(os.path.splitext(self.path)[0] + ".html", "w") as f:
                f.write(_ledger_html(self.rows))
        except Exception as e:  # noqa: BLE001
            log.debug("ledger html failed: %s", e)

    def summary(self):
        rows = [r for r in self.rows if r["fwd"] == self.fwd]
        if not rows:
            return ""
        peak = max(r["peak"] for r in rows)
        top = max(rows, key=lambda r: r["peak"])
        return (f"fwd {self.fwd}: base {rows[0]['alloc'] / 2**30:.1f} GiB, peak {peak / 2**30:.1f} GiB "
                f"(at block {top['block']} {top['phase']}), reserved {rows[-1]['reserved'] / 2**30:.1f} GiB, "
                f"{rows[-1]['t'] - rows[0]['t']:.1f} s; RSS anon {rows[0]['rss_anon'] / 2**30:.1f} -> {rows[-1]['rss_anon'] / 2**30:.1f} GiB "
                f"(max {max(r['rss_anon'] for r in rows) / 2**30:.1f})")



_PHASE_COLOR = {"start": "#888", "kv": "#4c78a8", "attn": "#f58518", "mlp": "#54a24b", "end": "#888"}


def _ledger_html(rows):
    """Self-contained SVG timeline of the ledger (no external assets): allocated,
    per-phase peak, reserved and process RSS in GiB against wall time, one
    forward per band; hover a mark for its numbers. Deep dive = trace.html."""
    if not rows:
        return "<p>empty ledger</p>"
    G = 2.0 ** 30
    W, H, L, T, B = 1400, 520, 70, 30, 60
    t0 = rows[0]["t"]
    tmax = max(r["t"] for r in rows) - t0 or 1.0
    ymax = max(max(r["peak"], r["reserved"], r.get("rss_anon", 0)) for r in rows) / G * 1.05 or 1.0
    xs = lambda t: L + (t - t0) / tmax * (W - L - 20)
    ys = lambda v: T + (H - T - B) * (1 - v / ymax)

    def path(key):
        return " ".join(f"{'M' if i == 0 else 'L'}{xs(r['t']):.1f},{ys(r[key] / G):.1f}" for i, r in enumerate(rows))

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" style="max-width:100%;font:12px sans-serif;background:#fff">']
    for g in range(0, int(ymax) + 1, max(1, int(ymax // 10) or 1)):
        out.append(f'<line x1="{L}" x2="{W-20}" y1="{ys(g):.1f}" y2="{ys(g):.1f}" stroke="#eee"/>'
                   f'<text x="{L-6}" y="{ys(g)+4:.1f}" text-anchor="end" fill="#666">{g} GiB</text>')
    # forward bands
    fwds = sorted({r["fwd"] for r in rows})
    for fw in fwds:
        rr = [r for r in rows if r["fwd"] == fw]
        x0, x1 = xs(rr[0]["t"]), xs(rr[-1]["t"])
        out.append(f'<rect x="{x0:.1f}" y="{T}" width="{max(1.0, x1-x0):.1f}" height="{H-T-B}" fill="{"#f7f7f7" if fw % 2 else "#fff"}"/>'
                   f'<text x="{x0+3:.1f}" y="{T+12}" fill="#999">fwd {fw}</text>')
    out.append(f'<path d="{path("reserved")}" fill="none" stroke="#bbb" stroke-width="1.5"/>')
    out.append(f'<path d="{path("peak")}" fill="none" stroke="#e45756" stroke-width="1" stroke-dasharray="3,2"/>')
    out.append(f'<path d="{path("alloc")}" fill="none" stroke="#222" stroke-width="1.5"/>')
    if any(r.get("rss_anon") for r in rows):
        out.append(f'<path d="{path("rss_anon")}" fill="none" stroke="#9467bd" stroke-width="1.5"/>')
    for r in rows:
        c = _PHASE_COLOR.get(r["phase"], "#333")
        tip = (f"fwd {r['fwd']}  block {r['block']}  {r['phase']}\\nt = {r['t']-t0:.1f} s\\nallocated {r['alloc']/G:.2f} GiB\\n"
               f"peak since last mark {r['peak']/G:.2f} GiB\\nreserved {r['reserved']/G:.2f} GiB\\nRSS anon {r.get('rss_anon',0)/G:.2f} GiB, file {r.get('rss_file',0)/G:.2f} GiB")
        out.append(f'<circle cx="{xs(r["t"]):.1f}" cy="{ys(r["peak"]/G):.1f}" r="3" fill="{c}"><title>{tip}</title></circle>')
    lg = [("#222", "allocated"), ("#e45756", "peak since last mark"), ("#bbb", "reserved"), ("#9467bd", "process RSS (anon)"),
          ("#4c78a8", "mark: kv"), ("#f58518", "mark: attn"), ("#54a24b", "mark: mlp")]
    for i, (col, name) in enumerate(lg):
        x = L + i * 190
        out.append(f'<rect x="{x}" y="{H-28}" width="14" height="10" fill="{col}"/><text x="{x+18}" y="{H-19}" fill="#333">{name}</text>')
    out.append(f'<text x="{W/2:.0f}" y="{H-2}" text-anchor="middle" fill="#666">wall time, {tmax:.0f} s span; hover a mark</text></svg>')
    return ("<!doctype html><meta charset=utf-8><title>H3 memory ledger</title>"
            "<style>body{margin:12px;font-family:sans-serif}</style><h3>H3MemoryProbe ledger</h3>" + "".join(out))

class H3MemoryProbe:
    """See what holds VRAM, per block and phase, and optionally record the
    allocator trace for a hoverable timeline (torch memory_viz)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "tag": ("STRING", {"default": "probe", "tooltip": "Run label; files land in out_dir/<tag>_<time>/"}),
                "ledger": ("BOOLEAN", {"default": True,
                                       "tooltip": "Per-forward JSONL of allocator counters at every H3StreamedBlocks phase boundary (start, block i kv/attn/mlp, end). No device sync, negligible cost. Stock blocks contribute start/end only."}),
                "record_history_forwards": ("INT", {"default": 0, "min": 0, "max": 64,
                                                    "tooltip": "Record the caching allocator's alloc/free trace (with Python stacks) for this many model forwards, then dump snapshot.pickle and trace.html (standalone; hover a band for the stack that allocated it). 0 = off. ~20k events per H3 forward at 200k tokens; a few percent while recording."}),
                "max_entries": ("INT", {"default": 300000, "min": 10000, "max": 5000000, "step": 10000,
                                        "tooltip": "Ring size for the allocator trace."}),
                "out_dir": ("STRING", {"default": "output/h3_memprobe",
                                       "tooltip": "Relative to the ComfyUI working directory. Not /tmp."}),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "report")
    FUNCTION = "patch"
    CATEGORY = "MAINodes/VRAM Lab"
    DESCRIPTION = ("Memory instrument for the H3 diffusion model: a per-block/per-phase ledger of PyTorch's "
                   "allocator counters (with H3StreamedBlocks upstream), and an optional allocator trace "
                   "rendered to a hoverable HTML timeline. Off = no cost; not installed at all.")

    def patch(self, model, tag, ledger, record_history_forwards, max_entries, out_dir):
        import folder_paths
        import comfy.patcher_extension as pe
        dev = comfy.model_management.get_torch_device()
        base = out_dir if os.path.isabs(out_dir) else os.path.join(os.path.dirname(folder_paths.get_output_directory()), out_dir)
        run_dir = os.path.join(base, f"{tag}_{time.strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(run_dir, exist_ok=True)
        led = _MemLedger(os.path.join(run_dir, "ledger.jsonl"), dev) if ledger else None
        state = {"fwd": 0, "recording": False, "done": False, "n": int(record_history_forwards)}

        m = model.clone()
        dm = getattr(getattr(m, "model", None), "diffusion_model", None)
        if led is not None:
            to = m.model_options.setdefault("transformer_options", {})
            to["h3_memprobe"] = led

        def wrapper(executor, *args, **kwargs):
            if led is not None and dm is not None:
                dm._h3_memprobe = led   # for patched pieces that do not receive transformer_options (final layer)
            x = args[0] if args else None
            shape = tuple(x.shape) if hasattr(x, "shape") else ()
            if state["n"] > 0 and not state["done"] and not state["recording"]:
                try:
                    torch.cuda.memory._record_memory_history(enabled="all", context="all", stacks="python",
                                                             max_entries=int(max_entries), device=dev,
                                                             record_pinned_host_memory=True)
                    state["recording"] = True
                    log.info("H3MemoryProbe[%s]: allocator trace ON (fwd %d)", tag, state["fwd"])
                except RuntimeError as e:
                    # ComfyUI enables cudaMallocAsync by default; torch's recorder needs the
                    # native caching allocator. Ledger still runs.
                    state["done"] = True
                    log.warning("H3MemoryProbe[%s]: allocator trace unavailable (%s). Start ComfyUI with "
                                "--disable-cuda-malloc to record it; the ledger still runs.", tag, str(e)[:120])
            if led is not None:
                led.begin_forward(shape)
            try:
                return executor(*args, **kwargs)
            finally:
                if dm is not None and hasattr(dm, "_h3_memprobe"):
                    del dm._h3_memprobe
                if led is not None:
                    led.end_forward()
                    log.info("H3MemoryProbe[%s]: %s", tag, led.summary())
                state["fwd"] += 1
                if state["recording"] and state["fwd"] >= state["n"]:
                    _dump_trace(run_dir, dev, tag)
                    torch.cuda.memory._record_memory_history(enabled=None, device=dev)
                    state["recording"] = False
                    state["done"] = True

        m.add_wrapper_with_key(pe.WrappersMP.DIFFUSION_MODEL, "h3_memprobe", wrapper)
        rep = f"H3MemoryProbe: {run_dir} (ledger {'on' if ledger else 'off'}, trace forwards {record_history_forwards})"
        log.info(rep)
        return (m, rep)


def _dump_trace(run_dir, dev, tag):
    try:
        os.makedirs(run_dir, exist_ok=True)
        snap = torch.cuda.memory._snapshot(device=dev)
        import pickle
        with open(os.path.join(run_dir, "snapshot.pickle"), "wb") as f:
            pickle.dump(snap, f)
        try:
            from torch.cuda._memory_viz import trace_plot
            html = trace_plot(snap, device=None)
            with open(os.path.join(run_dir, "trace.html"), "w") as f:
                f.write(html)
        except Exception as e:  # noqa: BLE001
            log.warning("H3MemoryProbe[%s]: trace_plot failed (%s); snapshot.pickle kept for pytorch.org/memory_viz", tag, e)
        log.info("H3MemoryProbe[%s]: allocator trace written to %s", tag, run_dir)
    except Exception as e:  # noqa: BLE001
        log.warning("H3MemoryProbe[%s]: snapshot failed: %s", tag, e)




class H3FreeCache:
    """Passthrough that returns the allocator's cached-but-free VRAM to the
    driver before the next stage. Measured motivation (2026-08-18): the
    VAE decode after a long H3 pass grew the pool 69.6 -> 77.9 GiB while live
    tensors were LOWER than during sampling; decode-shaped blocks could not
    reuse sampling's freed ones. Costs a few ms; changes no math."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"samples": ("LATENT",)},
                "optional": {"also_gc": ("BOOLEAN", {"default": True, "tooltip": "gc.collect() first so dead Python refs release their tensors too."})}}

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("samples", "report")
    FUNCTION = "free"
    CATEGORY = "MAINodes/VRAM Lab"
    DESCRIPTION = "Empty the CUDA caching allocator (torch.cuda.empty_cache) between stages; passthrough for a LATENT so it can sit right before VAE Decode."

    def free(self, samples, also_gc=True):
        import gc
        dev = comfy.model_management.get_torch_device()
        before = torch.cuda.memory_reserved(dev)
        if also_gc:
            gc.collect()
        comfy.model_management.soft_empty_cache(force=True)
        after = torch.cuda.memory_reserved(dev)
        rep = f"H3FreeCache: reserved {before / 2**30:.2f} -> {after / 2**30:.2f} GiB (live {torch.cuda.memory_allocated(dev) / 2**30:.2f})"
        log.info(rep)
        return (samples, rep)



class H3EvictTextEncoder:
    """Passthrough for CONDITIONING that unloads the text encoder (the CLIP
    patcher and its clones) the moment encoding is done - the same call
    ckinpdx's MMH3Tools makes inside its nodes. Under --gpu-only it is a no-op
    (offload device is the GPU); in normal mode it frees the TE's VRAM before
    the DiT loads instead of letting the planner evict it on demand. Exists to
    MEASURE whether explicit eviction matters on a small card (2026-08-18)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"conditioning": ("CONDITIONING",), "clip": ("CLIP",)}}

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "report")
    FUNCTION = "evict"
    CATEGORY = "MAINodes/VRAM Lab"
    DESCRIPTION = "Unload the text encoder right after encoding (passthrough for the conditioning)."

    def evict(self, conditioning, clip):
        dev = comfy.model_management.get_torch_device()
        before = comfy.model_management.get_free_memory(dev)
        try:
            comfy.model_management.unload_model_and_clones(clip.patcher)
        except Exception as e:  # noqa: BLE001
            log.warning("H3EvictTextEncoder: unload failed: %s", e)
        comfy.model_management.soft_empty_cache()
        after = comfy.model_management.get_free_memory(dev)
        rep = f"H3EvictTextEncoder: device free {before / 2**30:.1f} -> {after / 2**30:.1f} GiB"
        log.info(rep)
        return (conditioning, rep)

NODE_CLASS_MAPPINGS = {"H3StreamedBlocks": H3StreamedBlocks, "H3MemoryProbe": H3MemoryProbe, "H3FreeCache": H3FreeCache, "H3EvictTextEncoder": H3EvictTextEncoder}
NODE_DISPLAY_NAME_MAPPINGS = {"H3StreamedBlocks": "H3 Streamed Blocks (exact low-VRAM, alpha)",
                              "H3MemoryProbe": "H3 Memory Probe (ledger + allocator trace, alpha)",
                              "H3FreeCache": "H3 Free Cache (empty allocator between stages)",
                              "H3EvictTextEncoder": "H3 Evict Text Encoder (unload after encode)"}
