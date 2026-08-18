# SPDX-License-Identifier: GPL-3.0-or-later
"""VRAM Lab (alpha, 2026-08-18): exact low-memory execution of MiniMax-H3 blocks.

The problem, measured (ModelCatalog docs/SPATIAL_WALK_AND_MEMORY_2026-08-17.md
s.13, s.17): a full-length H3 forward materialises, per DiT block, the fused
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

Turtle mode (``kv_block > 0``): phase 2 attends K/V in blocks and combines
with the flash kernel's log-sum-exp (online softmax). Measured to one bf16
ulp against a single call. It exists so a card that cannot hold full K/V
can still finish; it is slow by design.

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

import torch

import comfy.model_management
import comfy.quant_ops
from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention

log = logging.getLogger("MAINodes.vram_lab")


# --------------------------------------------------------------------------- helpers

def _ranges(n, size):
    if size <= 0 or size >= n:
        return [(0, n)]
    return [(a, min(a + size, n)) for a in range(0, n, size)]


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

def streamed_block_forward(block, x, t_emb, mod_segments, rope_freqs, transformer_options,
                           q_chunk=16384, kv_chunk=16384, mlp_chunk=16384, kv_block=0):
    """Exact replacement for DiTBlock.forward with chunk-bounded transients."""
    S = x.shape[0]
    attn = block.attn
    heads, hd = attn.heads, attn.head_dim
    inner = heads * hd
    head_chunks = transformer_options.get("minimax_head_chunks", 1) if isinstance(transformer_options, dict) else 1

    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaln_proj(t_emb)

    # ---- phase 1: K, V for the whole sequence, chunk by chunk
    K = torch.empty((1, heads, S, hd), dtype=x.dtype, device=x.device)
    V = torch.empty((1, heads, S, hd), dtype=x.dtype, device=x.device)
    for a, b in _ranges(S, kv_chunk):
        s = b - a
        h = _mod_scale_shift_range(block.norm1(x[a:b]), shift_msa, scale_msa, mod_segments, a, b)
        qkv = attn.qkv_proj(h)
        q, k, v = qkv.split(inner, dim=-1)
        rope_c = rope_freqs[:, a:b].contiguous() if rope_freqs is not None else None
        _, k = _norm_rope(attn, q, k, rope_c, s)              # q computed and dropped
        K[0, :, a:b] = k.transpose(0, 1)
        V[0, :, a:b] = v.view(s, heads, hd).transpose(0, 1)
        del h, qkv, q, k, v, rope_c

    # ---- phase 2: Q per chunk, attention against full K/V, out_proj, gated residual in place
    for a, b in _ranges(S, q_chunk):
        s = b - a
        h = _mod_scale_shift_range(block.norm1(x[a:b]), shift_msa, scale_msa, mod_segments, a, b)
        qkv = attn.qkv_proj(h)
        q, k, _v = qkv.split(inner, dim=-1)
        rope_c = rope_freqs[:, a:b].contiguous() if rope_freqs is not None else None
        q, _ = _norm_rope(attn, q, k, rope_c, s)              # k recomputed and dropped
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
    del K, V

    # ---- phase 3: MLP per chunk
    for a, b in _ranges(S, mlp_chunk):
        h = _mod_scale_shift_range(block.norm2(x[a:b]), shift_mlp, scale_mlp, mod_segments, a, b)
        o = block.mlp(h)
        _mod_gate_range(x, gate_mlp, o, mod_segments, a, b)
        del h, o
    return x


def _make_replacement(block, cfg):
    def fn(args, extra):
        x = args["img"]
        if x.shape[0] < cfg["min_tokens"]:
            return extra["original_block"](args)
        x = streamed_block_forward(block, x, args["t_emb"], args["mod_segments"], args["rope_freqs"],
                                   args["transformer_options"], q_chunk=cfg["q_chunk"], kv_chunk=cfg["kv_chunk"],
                                   mlp_chunk=cfg["mlp_chunk"], kv_block=cfg["kv_block"])
        return {"img": x}
    return fn


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
                                     "tooltip": "Turtle mode: attend K/V in blocks of this many tokens with an exact log-sum-exp combine. 0 = off. Slow by design; for cards that cannot hold full K/V."}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "MAINodes/VRAM Lab"
    DESCRIPTION = ("Exact low-VRAM execution of MiniMax-H3: never materialises the full-sequence "
                   "fused QKV or SwiGLU tensors. Same math as the stock block; costs ~4% extra "
                   "projection work at long lengths. See vram_lab.py for the ledger.")

    def patch(self, model, q_chunk, kv_chunk, mlp_chunk, min_tokens, kv_block):
        dm = getattr(getattr(model, "model", None), "diffusion_model", None)
        blocks = getattr(dm, "blocks", None)
        if not blocks or not hasattr(blocks[0], "attn") or not hasattr(blocks[0].attn, "qkv_proj"):
            log.warning("H3StreamedBlocks: model does not look like MiniMax H3 (no blocks[*].attn.qkv_proj); unchanged")
            return (model,)
        cfg = {"q_chunk": q_chunk, "kv_chunk": kv_chunk, "mlp_chunk": mlp_chunk,
               "min_tokens": min_tokens, "kv_block": kv_block}
        m = model.clone()
        for i, block in enumerate(blocks):
            m.set_model_patch_replace(_make_replacement(block, cfg), "dit", "double_block", i)
        log.info("H3StreamedBlocks: %d blocks patched (q %d, kv %d, mlp %d, min %d, kv_block %d)",
                 len(blocks), q_chunk, kv_chunk, mlp_chunk, min_tokens, kv_block)
        return (m,)


NODE_CLASS_MAPPINGS = {"H3StreamedBlocks": H3StreamedBlocks}
NODE_DISPLAY_NAME_MAPPINGS = {"H3StreamedBlocks": "H3 Streamed Blocks (exact low-VRAM, alpha)"}
