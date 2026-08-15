"""The H3 block tap: pass-through instrumentation for ONE armed capture.

This is the only file in `concept_lab` that touches torch and ComfyUI, and
it exists because a capture cannot be made from outside the process: the
states we want live for microseconds inside a DiT block loop, and the
launch that computed them is part of the condition (DEC-CL-0015).

The seam is the sanctioned one, read off the installed ComfyUI on
2026-08-15 rather than remembered:

  comfy/ldm/minimax/model.py:522-526  the model's `forward` runs `_forward`
        through a DIFFUSION_MODEL WrapperExecutor, passing x, timestep,
        context and transformer_options POSITIONALLY and `minimax_payload=`
        as a keyword. That is the only place the packed layout is reachable
        from outside, so the wrapper's whole job is to stash it.
  comfy/ldm/minimax/model.py:643-655  the block loop calls
        `patches_replace["dit"][("double_block", i)](args, {"original_block":
        block_wrap})` and takes `["img"]` off the result. `args["img"]` is
        2-D `[seq_len, hidden]`: batch is always 1 here (:538-539).
  comfy/ldm/minimax/model.py:404-420  PackedLayout exposes `.seq_len`,
        `.segments` as `(start, stop, kind)` triples, `.position_ids`, and
        `.signature = (text_len, latent_t, latent_h, latent_w, audio_t)`.
  comfy/ldm/minimax/model.py:70-81  `_frame_grid` builds one latent frame's
        rows from `dim // patch` per axis with patch=2, so one latent frame
        is `(latent_h//2) * (latent_w//2)` rows, and video rows are
        frame-major (`patchify_video`'s einsum puts t outermost, :42-49).
  comfy/model_patcher.py:430 clone, :667 set_model_patch_replace,
        :1343 add_wrapper_with_key. Installing on a clone is what keeps the
        caller's MODEL unpatched.
  comfy/samplers.py:510-513  transformer_options carries `cond_or_uncond`
        (a list; 0 is the cond entry), `sigmas` (this call's sigma) and
        `sample_sigmas` (the whole schedule).

TWO RULES THIS FILE IS BUILT AROUND.

Pass-through is not a goal, it is the gate. The block patch calls the
original block, records from a detached view of its output, and returns the
ORIGINAL dict object. Nothing is written into `args` or into the returned
dict, so with `enabled=False` the tap is a function call and a branch.

Nothing is recorded that cannot be re-identified. The capture id is
computable before the first tensor is written (identity is stack + variant
+ tap + seed + sampler + replicate + launch, and tensors are not identity),
tensors are written as they are captured rather than accumulated in VRAM,
and every file lands with its sha256 in a TensorRef.

The variant half of that identity is MEASURED here rather than declared by
the caller (DEC-CL-0019): the first wrapper call hashes the text-encoder
output and every ref and keyframe latent the model was handed, and those
hashes are the variant's sources. The refusal that protects the bank is
armed at the same moment, before the first byte, and tensors stream into
`<id>.partial` until the manifest is filed.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess

import torch
from safetensors.torch import save_file

from concept_lab import api
from concept_lab import types as T
from concept_lab.space import open_space

try:                                    # comfy at run time, absent in tests
    from comfy.patcher_extension import WrappersMP
    DIFFUSION_MODEL = WrappersMP.DIFFUSION_MODEL
except Exception:                       # verified equal upstream:
    DIFFUSION_MODEL = "diffusion_model"  # patcher_extension.py:50-57

PATCH_NAME = "dit"
PATCH_BLOCK = "double_block"

# dtype names as they go into a TensorRef. Short, and the same words the
# rest of the program uses.
_DTYPE_NAME = {torch.float32: "fp32", torch.float16: "fp16",
               torch.bfloat16: "bf16", torch.float64: "fp64"}


class TapError(RuntimeError):
    """The tap refused. Always names the numbers that disagreed."""


def _tensor_bytes(t: torch.Tensor) -> bytes:
    return t.detach().cpu().contiguous().numpy().tobytes()


def _sha256_tensor(t: torch.Tensor) -> str:
    return "sha256:" + hashlib.sha256(_tensor_bytes(t)).hexdigest()


def _cond_sha256(*tensors: torch.Tensor) -> str:
    """sha256 over conditioning tensors, in the order given, through fp32.

    Float tensors go through float32 rather than their own dtype for two
    reasons: bf16 has no numpy dtype at all (`.numpy()` raises on it, and
    conditioning arrives bf16 whenever the stack is), and the same latent
    arriving bf16 in one launch and fp32 in the next is one condition rather
    than two. Integer tensors (token tags) are hashed as they are.

    Separate from `_sha256_tensor` on purpose: that one hashes position ids
    and the sketch matrix, whose digests are already in manifests on disk,
    and widening it would move ids that exist.
    """
    h = hashlib.sha256()
    for t in tensors:
        t = t.detach().to("cpu")
        if t.is_floating_point():
            t = t.to(torch.float32)
        h.update(t.contiguous().numpy().tobytes())
    return "sha256:" + h.hexdigest()


# payload ref `kind` -> the ConditioningSource kind it becomes.
# comfy_extras/nodes_minimax_h3.py:290-355 emits exactly these four.
_REF_SOURCE_KIND = {"image": "image", "video": "video",
                    "video_audio": "video", "audio": "audio"}


def render_fingerprint(after) -> "dict | None":
    """Evidence about what the render DECODED to, for a finished capture.

    Not identity (DEC-CL-0006 stands: tensors and outputs are not identity)
    and deliberately so: this is how E0's pass-through gate stops being a
    human comparing two videos by eye. Two arms that claim to be the same
    render either carry the same frame hashes or one of them is lying.

    `after` is whatever was wired into the flush node for ordering. If it is
    not an image-shaped tensor this returns None, because a fingerprint of
    something that is not a render is worse than no fingerprint.
    """
    if not isinstance(after, torch.Tensor) or after.ndim < 3:
        return None
    x = after.detach().to("cpu", torch.float32)
    u8 = (x.clamp(0, 1) * 255).round().to(torch.uint8).contiguous()
    return {
        "frame0_sha256": "sha256:" + hashlib.sha256(
            u8[0].numpy().tobytes()).hexdigest(),
        "frames_sha256": "sha256:" + hashlib.sha256(
            u8.numpy().tobytes()).hexdigest(),
        "n_frames": int(u8.shape[0]), "h": int(u8.shape[-3]),
        "w": int(u8.shape[-2]),
    }


def _git_short(path: str):
    """The commit of the tree at `path`, or None. Never a guess."""
    try:
        r = subprocess.run(["git", "-C", path, "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=5)
        return r.stdout.strip() or None if r.returncode == 0 else None
    except Exception:
        return None


def fingerprint_from_patcher(model_patcher, stack_label: str,
                             backend: str = "h3") -> T.ModelStackFingerprint:
    """What is honestly readable off a live ModelPatcher, and nothing else.

    The checkpoint NAME is not among the readable things: a ModelPatcher
    carries a model, not the filename it was loaded from, so `checkpoint` is
    the caller's `stack_label` and says so in `model_options`. Guessing it
    from a class name would put a plausible wrong condition on every capture
    in the bank, which is the exact failure ModelStackFingerprint exists to
    prevent.

    What IS readable and does belong to identity: the model class, the
    config class, the parameter dtype, and how many weight patches (LoRAs,
    merges) are stacked on top with which key prefixes.
    """
    model = getattr(model_patcher, "model", None)
    opts = {
        "model_class": type(model).__name__ if model is not None else None,
        "checkpoint_name_source":
            "caller-supplied; the ModelPatcher does not carry the UNET "
            "filename",
    }
    cfg = getattr(model, "model_config", None) if model is not None else None
    if cfg is not None:
        opts["model_config_class"] = type(cfg).__name__

    dtype = None
    try:
        for p in model.parameters():
            dtype = str(p.dtype).replace("torch.", "")
            break
    except Exception:
        pass
    opts["param_dtype"] = dtype

    patches = getattr(model_patcher, "patches", {}) or {}
    opts["n_weight_patches"] = len(patches)
    prefixes = sorted({str(k).split(".")[0] for k in patches})
    opts["patch_key_prefixes"] = prefixes[:16]

    comfy_root = None
    try:
        import folder_paths                                   # noqa: PLC0415
        comfy_root = folder_paths.base_path
    except Exception:
        pass
    pack_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))

    comfy_version = None
    try:
        from comfyui_version import __version__ as comfy_version  # noqa
    except Exception:
        pass

    return T.ModelStackFingerprint(
        checkpoint=stack_label,
        model_options=opts,
        backend=backend,
        comfy_version=comfy_version,
        comfy_commit=_git_short(comfy_root) if comfy_root else None,
        mainodes_commit=_git_short(pack_root),
    )


class TapSession:
    """One armed capture. Holds the TapSpec, the manifest-in-progress, the
    layout stashed by the wrapper, the step selection, and the sink.

    One session is one arm of one comparison in one launch. It refuses a
    second packed layout because a capture whose rows mean different things
    at different steps cannot be subtracted from anything.
    """

    def __init__(self, tap: T.TapSpec, arm: str, variant_name: str, seed: int,
                 replicate: int, stack: T.ModelStackFingerprint,
                 index_root=None, tensor_root=None, corpus_row_id=None,
                 enabled=True, cond_only=True, notes="", sampler_label=""):
        self.tap = tap
        self.arm = arm
        self.variant_name = variant_name
        self.seed = int(seed)
        self.replicate = int(replicate)
        self.stack = stack
        self.corpus_row_id = corpus_row_id or None
        self.enabled = bool(enabled)
        self.cond_only = bool(cond_only)
        self.notes = notes
        self.sampler_label = sampler_label

        # A capture writes to disk from inside a block loop, so the space
        # has to exist before the first forward rather than after it.
        self.space = open_space(index_root, tensor_root, create=True,
                                label="concept_lab")
        self.index_root = self.space.index_root
        self.tensor_root = self.space.tensor_root

        self.process = dict(T.process_fingerprint(),
                            torch=torch.__version__,
                            cuda=getattr(torch.version, "cuda", None))

        self._layout = None
        self._layout_sig = None
        self._timestep = None
        self._sched = None
        self._n_steps = None
        self._selected = None
        self._sampler = None
        self._capture_id = None
        self._sketch_R = None
        self._sketch_sha = None
        # measured at the model boundary on the first wrapper call, then
        # frozen: the variant is what the model SAW, not what a node said
        # it was armed with (DEC-CL-0019)
        self._variant = None
        self._context_shape = None
        self._render_fingerprint = None

        self.tensors = []
        self.bytes_written = 0
        self.calls_seen = 0
        self.calls_recorded = 0
        self.cond_or_uncond_seen = []
        self.finalized = False

    # ------------------------------------------------------- installation
    def install(self, model_patcher):
        """Clone the patcher, hang the wrapper and the block patches on the
        clone, hand it back. The caller's MODEL is untouched."""
        m = model_patcher.clone()
        m.add_wrapper_with_key(DIFFUSION_MODEL, "concept_lab_tap",
                               self._dm_wrapper)
        for i in self.tap.blocks:
            m.set_model_patch_replace(self._make_block_patch(int(i)),
                                      PATCH_NAME, PATCH_BLOCK, int(i))
        return m

    # ------------------------------------------------------- the wrapper
    def _dm_wrapper(self, executor, x, timestep, context,
                    transformer_options=None, minimax_payload=None, **kw):
        """Stash the packed layout, change nothing, call on.

        The layout is the only thing here that cannot be recovered later: it
        is what turns a row index into a named segment, and without it a
        captured tensor is a rectangle of numbers with no coordinates.
        """
        self.calls_seen += 1
        layout = (minimax_payload or {}).get("layout")
        if layout is None:
            raise TapError(
                f"concept_lab tap: diffusion-model call #{self.calls_seen} "
                f"carried no packed layout in minimax_payload (keys: "
                f"{sorted((minimax_payload or {}).keys())}). A capture "
                f"without a layout is not a capture: row indices would have "
                f"no segment names. Upstream normally prebuilds the layout "
                f"in extra_conds (model.py:548); if this graph does not, the "
                f"tap refuses rather than inventing one")
        sig = self._layout_signature(layout)
        if self._layout_sig is None:
            self._layout_sig = sig
        elif sig.identity() != self._layout_sig.identity():
            raise TapError(
                f"concept_lab tap: the packed layout CHANGED during one "
                f"capture (call #{self.calls_seen}). First signature "
                f"{self._layout_sig.signature_id} with "
                f"{self._layout_sig.total_rows} rows, now "
                f"{sig.signature_id} with {sig.total_rows} rows. One capture "
                f"spans one layout; two layouts are two conditions")
        if self._variant is None:
            self._freeze_variant(context, minimax_payload or {})
        self._layout = layout
        self._timestep = timestep
        return executor(x, timestep, context, transformer_options,
                        minimax_payload=minimax_payload, **kw)

    # ------------------------------------------- what the model actually saw
    def _freeze_variant(self, context, payload) -> None:
        """Derive the variant's identity from the conditioning, once.

        Called on the first diffusion-model call, which is before any block
        patch can fire and therefore before a byte can be written. The
        caller's `variant_name` is a LABEL and stays out of identity: two
        arms named differently that fed the model the same bytes are one
        condition, and two arms with one name that fed it different bytes
        are two (DEC-CL-0019; the defect it replaces gave every R cell in a
        15-cell grid one id).

        `prompt` stays None. The human text is not reachable at this seam,
        and a hash of the encoder output the model consumed is the stronger
        identity anyway.
        """
        # packed order: text, then keyframe (cond) rows, then ref rows
        # (model.py:322-345), so sources mirror the rows the model saw
        sources = [self._text_source(context, payload)]
        for i, blk in enumerate(payload.get("keyframes") or ()):
            sources.append(self._block_source(blk, i, "keyframe"))
        for i, blk in enumerate(payload.get("refs") or ()):
            sources.append(self._block_source(blk, i, "ref"))
        self._variant = T.ConditionVariant(
            name=self.variant_name, arm=self.arm, sources=tuple(sources),
            prompt=None)

    def _text_source(self, context, payload) -> T.ConditioningSource:
        if not isinstance(context, torch.Tensor):
            raise TapError(
                f"concept_lab tap: the diffusion-model call carried "
                f"context={type(context).__name__}, not a tensor. The text "
                f"conditioning IS the identity of a text-only arm "
                f"(DEC-CL-0019); without it two different prompts would file "
                f"one capture id")
        self._context_shape = [int(s) for s in context.shape]
        seed = payload.get("seed")
        if seed is not None and int(seed) != self.seed:
            raise TapError(
                f"concept_lab tap: the sampler's payload seed is {int(seed)} "
                f"but this arm was told seed={self.seed}. The node cannot "
                f"read the sampler's seed (it lives in the sampler) so it is "
                f"copied by hand, and a mismatch mislabels the capture. "
                f"Fix the Arm node's seed to {int(seed)} and rerun")
        tags = payload.get("text_token_tags")
        if tags is None:
            tag_hash, tag_n = None, None
        elif isinstance(tags, torch.Tensor):
            tag_hash, tag_n = _cond_sha256(tags), int(tags.numel())
        else:
            tag_hash, tag_n = T.digest(list(tags)), len(list(tags))
        return T.ConditioningSource(
            kind="text", content_hash=_cond_sha256(context),
            preprocess={
                # the tags are a per-token modality vector (model.py:574-583)
                # and can be thousands long, so identity carries their digest
                # rather than the vector itself
                "text_token_tags": tag_hash,
                "text_token_tags_len": tag_n,
                "visual_cond_noise_aug": payload.get("visual_cond_noise_aug"),
                "audio_cond_noise_aug": payload.get("audio_cond_noise_aug"),
                "audio_scale": payload.get("audio_scale"),
                "payload_seed": None if seed is None else int(seed)},
            notes="text-encoder output as the model consumed it")

    def _block_source(self, blk, idx: int, family: str) -> T.ConditioningSource:
        """One ref or keyframe block, hashed as the model got it.

        Ref blocks carry `kind` in {image, video, video_audio, audio}
        (nodes_minimax_h3.py:290-355); keyframe blocks carry no kind at all
        and are recorded as kind "keyframe".
        """
        latent, audio = blk.get("latent"), blk.get("audio_latent")
        if family == "keyframe":
            payload_kind, source_kind = "keyframe", "keyframe"
            parts = [t for t in (latent, audio) if t is not None]
        else:
            payload_kind = str(blk.get("kind"))
            source_kind = _REF_SOURCE_KIND.get(payload_kind)
            if source_kind is None:
                raise TapError(
                    f"concept_lab tap: ref block {idx} has kind "
                    f"{payload_kind!r}, not one of "
                    f"{sorted(_REF_SOURCE_KIND)}. An unknown block kind means "
                    f"the packed order this capture records is not the order "
                    f"the model used")
            if payload_kind == "audio":
                parts = [audio]
            elif payload_kind == "video_audio":
                parts = [latent, audio]
            else:
                parts = [latent]
        if not parts or any(p is None for p in parts):
            raise TapError(
                f"concept_lab tap: {family} block {idx} (kind "
                f"{payload_kind!r}) carries no latent to hash; its keys are "
                f"{sorted(blk)}. A reference whose bytes cannot be read "
                f"cannot be part of an identity")
        pre = {"payload_kind": payload_kind,
               "latent_shape": None if latent is None
               else [int(s) for s in latent.shape],
               "audio_latent_shape": None if audio is None
               else [int(s) for s in audio.shape],
               "ref_audio_t": None if blk.get("ref_audio_t") is None
               else int(blk["ref_audio_t"])}
        if family == "keyframe":
            # upstream names it resolved_frame_index (nodes_minimax_h3.py:145,
            # 150, 209); frame_idx is read too in case a caller sets it
            fi = blk.get("frame_idx", blk.get("resolved_frame_index"))
            pre["frame_idx"] = None if fi is None else int(fi)
        return T.ConditioningSource(kind=source_kind,
                                    content_hash=_cond_sha256(*parts),
                                    preprocess=pre)

    @staticmethod
    def _layout_signature(layout) -> T.LayoutSignature:
        # upstream packs (start, stop, kind); LayoutSignature holds
        # (kind, start, stop), and the reorder is deliberate: the contract
        # aligns by NAME, so the name goes first.
        segs = tuple((str(k), int(a), int(b)) for a, b, k in layout.segments)
        pos = getattr(layout, "position_ids", None)
        phash = _sha256_tensor(pos) if pos is not None else None
        return T.LayoutSignature(segments=segs, total_rows=int(layout.seq_len),
                                 position_hash=phash)

    # --------------------------------------------------- the block patch
    def _make_block_patch(self, i: int):
        def fn(args, extra):
            out = extra["original_block"](args)
            if self.enabled:
                self._record(i, out["img"], args.get("transformer_options"))
            # the ORIGINAL dict, not a copy and not a rebuild: pass-through
            # has to be an identity, not an equality
            return out
        return fn

    # ----------------------------------------------------- the selection
    def _selection(self, transformer_options):
        """(step index, sigma) for this call, and the selection itself once.

        The step index is read back from the schedule rather than counted:
        a sampler may call the model more than once per step, and a counter
        would drift on the first one that does.
        """
        to = transformer_options or {}
        sched_t = to.get("sample_sigmas")
        sig_t = to.get("sigmas")
        if sched_t is None or sig_t is None:
            raise TapError(
                "concept_lab tap: transformer_options carries no "
                f"{'sample_sigmas' if sched_t is None else 'sigmas'}; "
                "step selection would be a guess (comfy/samplers.py:510-513 "
                "sets both)")
        if self._sched is None:
            sched = [float(v) for v in torch.as_tensor(sched_t).flatten()
                     .tolist()]
            n_steps = len(sched) - 1
            if n_steps < 1:
                raise TapError(
                    f"concept_lab tap: sample_sigmas has {len(sched)} value(s), "
                    f"so there are {n_steps} steps to select from")
            if self.tap.steps:
                sel = sorted({int(s) for s in self.tap.steps})
            else:
                # q in [0, 1] maps onto the step INDEX range [0, n_steps-1],
                # so q=0 is the first step and q=1 is the last one.
                sel = sorted({int(round(float(q) * (n_steps - 1)))
                              for q in self.tap.step_quantiles})
            bad = [s for s in sel if not 0 <= s < n_steps]
            if bad:
                raise TapError(
                    f"concept_lab tap: step(s) {bad} are outside this "
                    f"schedule's 0..{n_steps - 1}")
            self._sched, self._n_steps, self._selected = sched, n_steps, sel
            self._sampler = {"n_steps": n_steps, "sigma_first": sched[0],
                             "sigma_last": sched[-1],
                             "label": self.sampler_label}
        sigma = float(torch.as_tensor(sig_t).flatten()[0])
        step = min(range(self._n_steps),
                   key=lambda k: abs(self._sched[k] - sigma))
        return step, sigma

    # --------------------------------------------------------- recording
    def _record(self, block: int, img, transformer_options):
        to = transformer_options or {}
        cou = to.get("cond_or_uncond")
        if isinstance(cou, (list, tuple)):
            tag = ",".join(str(int(c)) for c in cou)
            if tag not in self.cond_or_uncond_seen:
                self.cond_or_uncond_seen.append(tag)
            if self.cond_only and 0 not in [int(c) for c in cou]:
                return
        step, sigma = self._selection(to)
        if step not in self._selected:
            return
        if self._layout is None:
            raise TapError("concept_lab tap: a block fired before the "
                           "diffusion-model wrapper stashed a layout")

        h = img.detach()
        if h.shape[0] != self._layout_sig.total_rows:
            raise TapError(
                f"concept_lab tap: block {block} state has {h.shape[0]} rows "
                f"but the stashed layout has {self._layout_sig.total_rows}. "
                f"Upstream rebuilds the layout when the signature moves "
                f"(model.py:549-552), so this capture's segment names do not "
                f"describe these rows")
        self._ensure_capture_id()

        kinds = self.tap.store_kinds()
        for seg_kind in self.tap.segments:
            spans = [(a, b) for (k, a, b) in self._layout_sig.segments
                     if k == seg_kind]
            for idx, (a, b) in enumerate(spans):
                suffix = f"_{idx}" if len(spans) > 1 else ""
                seg_name = f"{seg_kind}#{idx}" if len(spans) > 1 else seg_kind
                rows = h[a:b]
                for store in kinds:
                    self._write(block, step, sigma, seg_kind, seg_name,
                                suffix, store, rows)
        self.calls_recorded += 1

    def _write(self, block, step, sigma, seg_kind, seg_name, suffix, store,
               rows):
        n_rows, hidden = int(rows.shape[0]), int(rows.shape[1])
        payload = {}
        if store == "stats":
            r = rows.to(torch.float32)
            payload["stats"] = torch.stack(
                [r.mean(dim=0), r.std(dim=0, unbiased=False),
                 (r * r).mean(dim=0)], dim=0)
        elif store == "frame_mean":
            if seg_kind != "video":
                return                      # only the target stream has time
            latent_t = int(self._layout.signature[1])
            lat_h, lat_w = int(self._layout.signature[2]), \
                int(self._layout.signature[3])
            frame_rows = (lat_h // 2) * (lat_w // 2)
            if latent_t * frame_rows != n_rows:
                raise TapError(
                    f"concept_lab tap: frame-row mismatch on segment "
                    f"{seg_name!r}: latent_t={latent_t} x "
                    f"frame_rows={frame_rows} = {latent_t * frame_rows}, but "
                    f"the segment has {n_rows} rows. The layout signature "
                    f"(text_len, latent_t, latent_h, latent_w, audio_t) = "
                    f"{tuple(self._layout.signature)} does not describe these "
                    f"rows (model.py:77-81, 397-402)")
            g = rows.to(torch.float32).reshape(latent_t, frame_rows, hidden)
            payload["frame_mean"] = g.mean(dim=1)
            payload["patch_mean"] = g.mean(dim=0).to(torch.float16)
        elif store == "sketch":
            R = self._sketch(hidden, rows.device)
            payload["sketch"] = (rows.to(torch.float32) @ R).to(torch.float16)
        elif store == "full":
            payload["full"] = rows.to(torch.bfloat16)
        else:
            raise TapError(f"concept_lab tap: unknown store {store!r}")

        name = (f"b{int(block):02d}_s{int(step):02d}_{seg_kind}{suffix}"
                f"_{store}.safetensors")
        # the ref records the FINAL path; the bytes land in the partial dir
        # until finalize files the manifest and promotes it (DEC-CL-0019)
        rel = os.path.join(self._capture_id, name)
        path = os.path.join(self.partial_dir(), name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_file({k: v.detach().cpu().contiguous()
                   for k, v in payload.items()}, path)
        sha = T.file_sha256(path)
        self.bytes_written += os.path.getsize(path)
        for k, v in payload.items():
            self.tensors.append(T.TensorRef(
                key=f"{seg_name}/b{int(block)}/s{int(step)}/{k}",
                path=rel, shape=tuple(int(s) for s in v.shape),
                dtype=_DTYPE_NAME.get(v.dtype, str(v.dtype)), sha256=sha,
                segment=seg_name, block=int(block), step=int(step),
                sigma=sigma, kind=store))

    def _sketch(self, hidden: int, device):
        """The random projection, drawn ONCE per session on the CPU.

        CPU because a CUDA generator's stream is not reproducible across
        devices, and an unreproducible projection makes every sketch in the
        bank incomparable to the next run's.
        """
        dim = int(self.tap.sketch_dim or 0)
        if dim <= 0:
            raise TapError("concept_lab tap: store 'sketch' needs sketch_dim")
        if self._sketch_R is None:
            g = torch.Generator().manual_seed(int(self.tap.sketch_seed or 0))
            R = torch.randn(hidden, dim, generator=g, dtype=torch.float32)
            R /= math.sqrt(dim)
            self._sketch_R = R
            self._sketch_sha = _sha256_tensor(R)
        if self._sketch_R.shape[0] != hidden:
            raise TapError(
                f"concept_lab tap: sketch matrix is [{self._sketch_R.shape[0]}"
                f", {dim}] but this segment's width is {hidden}")
        return self._sketch_R.to(device=device, dtype=torch.float32)

    # ---------------------------------------------------------- identity
    def capture_dir(self) -> str:
        return os.path.join(self.space.tensor_dir(), self._capture_id or "")

    def partial_dir(self) -> str:
        return self.capture_dir() + ".partial"

    def _ensure_capture_id(self) -> str:
        """The id, computed before the first byte is written, then frozen.

        Tensors are not identity (DEC-CL-0006), so the id is knowable as
        soon as the sampler's schedule is; freezing it here is what lets the
        tap write into `<tensor_dir>/<capture_id>.partial/` as it goes
        instead of buffering a capture's worth of state in VRAM.

        The collision check lives HERE rather than in `capture_record`,
        which is the last thing a capture does. An id already on disk means
        the bytes about to be streamed would overwrite an earlier capture's
        before anything noticed, and that is not a hypothetical: E0
        replicate 0's tensors were destroyed exactly this way (DEC-CL-0019).
        """
        cid = self._manifest(()).capture_id
        if self._capture_id is None:
            if self._variant is None:
                raise TapError(
                    "concept_lab tap: the capture id was asked for before "
                    "the diffusion-model wrapper ran. Identity is measured "
                    "from what the model saw (DEC-CL-0019), so there is "
                    "nothing to hash yet")
            taken = api.capture_exists(cid, index_root=self.index_root,
                                       tensor_root=self.tensor_root)
            if taken:
                raise TapError(
                    f"concept_lab tap: capture id {cid} is already taken by "
                    f"variant {taken.get('variant_name')!r} (arm "
                    f"{taken.get('arm')!r}, replicate {taken.get('replicate')}"
                    f", launch {taken.get('process_launch_id')}, manifest "
                    f"{taken.get('manifest_path')}, tensors "
                    f"{taken.get('tensor_dir')}). This arm is armed as "
                    f"{self.variant_name!r} replicate {self.replicate}: same "
                    f"stack, same conditioning, same seed, same launch. "
                    f"increment `replicate` for an honest rerun; the tap "
                    f"wrote nothing")
            self._capture_id = cid
        elif cid != self._capture_id:
            raise TapError(
                f"concept_lab tap: the capture id moved mid-capture "
                f"({self._capture_id} -> {cid}). Tensors are already on disk "
                f"under the old one")
        return self._capture_id

    def _manifest(self, tensors) -> T.FunctionalCaptureManifest:
        variant = self._variant or T.ConditionVariant(name=self.variant_name,
                                                      arm=self.arm)
        return T.FunctionalCaptureManifest(
            stack=self.stack, variant=variant, tap=self.tap, seed=self.seed,
            sampler=dict(self._sampler or {}),
            sigmas=tuple(self._sched or ()),
            layout=self._layout_sig, tensors=tuple(tensors),
            corpus_row_id=self.corpus_row_id, replicate=self.replicate,
            process=dict(self.process), origin=self.space.space_id,
            status=T.Status.MEASURED, notes=self._notes())

    def _notes(self) -> str:
        return json.dumps({
            "notes": self.notes,
            "enabled": self.enabled,
            "cond_only": self.cond_only,
            "cond_or_uncond_seen": list(self.cond_or_uncond_seen),
            "calls_seen": self.calls_seen,
            "calls_recorded": self.calls_recorded,
            "steps_selected": list(self._selected or ()),
            "sketch_R_sha256": self._sketch_sha,
            "context_shape": self._context_shape,
            "render_fingerprint": self._render_fingerprint,
        }, sort_keys=True)

    # ---------------------------------------------------------- the sink
    def finalize(self, render_fingerprint=None) -> dict:
        """File the manifest, THEN promote the tensors. Nothing re-read.

        With `enabled=False` this writes NOTHING, on purpose: the disabled
        arm is the pass-through gate's control, and a control that leaves a
        record behind is not a control.

        The order is the rule: the manifest is filed first and the
        `<id>.partial` directory becomes `<id>` only once it landed. Renaming
        first would leave orphaned final bytes behind a refusal, which is
        indistinguishable on disk from a capture that succeeded.
        """
        self._render_fingerprint = render_fingerprint
        if not self.enabled:
            self.finalized = True
            return {"enabled": False, "calls_seen": self.calls_seen,
                    "note": "pass-through only; nothing recorded"}
        if self._capture_id is None:
            raise TapError(
                f"concept_lab tap: nothing was recorded in {self.calls_seen} "
                f"diffusion-model call(s), so there is no capture to file. "
                f"Either the tap's blocks {list(self.tap.blocks)} never fired "
                f"or no call landed on a selected step")
        man = self._manifest(self.tensors)
        if man.capture_id != self._capture_id:
            raise TapError(
                f"concept_lab tap: the capture id moved between the first "
                f"tensor and finalize ({self._capture_id} -> "
                f"{man.capture_id})")
        try:
            out = api.capture_record(man.to_dict(), index_root=self.index_root,
                                     tensor_root=self.tensor_root)
        except Exception as e:
            raise TapError(
                f"concept_lab tap: the manifest for {self._capture_id} was "
                f"refused, so its tensors were NOT promoted and are still at "
                f"{self.partial_dir()} (evidence; `space doctor` reports it "
                f"as an unfinished capture and it is yours to delete). "
                f"{type(e).__name__}: {e}") from e
        if os.path.isdir(self.partial_dir()):
            os.rename(self.partial_dir(), self.capture_dir())
        self.finalized = True
        out.update({"enabled": True, "n_tensors": len(self.tensors),
                    "bytes_written": self.bytes_written,
                    "capture_id": self._capture_id,
                    "calls_seen": self.calls_seen,
                    "calls_recorded": self.calls_recorded,
                    "render_fingerprint": self._render_fingerprint,
                    "tensor_dir": self.capture_dir(),
                    "steps_selected": list(self._selected or ()),
                    "sigmas_selected": [self._sched[s]
                                        for s in (self._selected or ())]})
        return out


def install(model_patcher, session: TapSession):
    """Module-level door, so `backends/h3.py` can delegate in one line."""
    return session.install(model_patcher)
