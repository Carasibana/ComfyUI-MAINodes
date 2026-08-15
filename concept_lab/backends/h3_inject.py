"""The H3 delta injection: ADD a stored delta at the seam the tap reads.

The mirror of `h3_tap`. The tap reads block states out of a running forward
and files them; this reads a delta back in and adds it to the target-video
rows of the same block output, at chosen blocks and steps. It is the
instrument for the premise test ("take the delta overlap reference and
generate something with it"), not a compiler and not a factor library:
nothing here decides what a delta MEANS.

Same seam, read off the installed ComfyUI on 2026-08-15 and unchanged from
the tap's reading (h3_tap's module docstring has the line numbers):
comfy/ldm/minimax/model.py:643-655 runs each tapped block through
`patches_replace["dit"][("double_block", i)]`, and `args["img"]` is 2-D
`[seq_len, hidden]`. The wrapper at model.py:522-526 is still the only place
the packed layout is reachable, so the wrapper's whole job is to stash it and
to refuse a run whose video geometry does not match the pack's.

THREE RULES THIS FILE IS BUILT AROUND.

The write is in place and the dict is the block's own. `out["img"]` is
modified through a view of its video rows and the ORIGINAL dict object comes
back, so an injection is arithmetic on the state rather than a rebuilt
pipeline: with `enabled=False`, or with `control="zero"`, the code path runs
and the numbers do not move.

A delta is only addable where its rows mean the same thing. A pack carries
`latent_t` and `frame_rows` (the manifests do not carry the packed
signature tuple, only segments and a position hash), and the first wrapper
call recomputes both off the live layout exactly as `TapSession._write`'s
frame_mean branch does. A mismatch refuses BEFORE any block fires, printing
both pairs. text_len and audio_t are free to differ: a different prompt is
allowed, because the delta lives on video rows.

Controls are part of the instrument, not of the analysis. `zero`,
`time_shuffle` and `sign_flip` are applied to the delta ONCE at load, so
every arm of the first read runs identical code with different numbers.
"""
from __future__ import annotations

import json
import os
import re

import torch
from safetensors.torch import load_file

from concept_lab.backends.h3_tap import (DIFFUSION_MODEL, PATCH_BLOCK,
                                         PATCH_NAME, TapSession,
                                         schedule_and_sigma, step_from_sigma)

WRAPPER_KEY = "concept_lab_inject"
PACK_TYPE = "concept_delta_pack"

STEP_MODES = ("captured_only", "nearest_all")
MODES = ("frame", "separable")
CONTROLS = ("none", "zero", "time_shuffle", "sign_flip")

# b03_s12/frame_mean — block, step, which map
_KEY_RE = re.compile(r"^b(\d+)_s(\d+)/(frame_mean|patch_mean)$")


class InjectError(RuntimeError):
    """The injector refused. Always names the numbers that disagreed."""


# ------------------------------------------------------------- the pack
class DeltaPack:
    """A `.safetensors` of per-(block, step) deltas plus its sidecar JSON.

    Loaded once, on the CPU, in float32, with the control already applied.
    The tensors are small by construction (a frame map is `[latent_t,
    hidden]` and a patch map `[frame_rows, hidden]`, not a state), which is
    the whole reason the first injection is built on reductions.
    """

    def __init__(self, path: str, meta: dict, entries: dict):
        self.path = path
        self.meta = meta
        self.entries = entries                     # (block, step) -> dict
        self.latent_t = int(meta["latent_t"])
        self.frame_rows = int(meta["frame_rows"])
        self.hidden = int(meta["hidden"])
        self.label = str(meta.get("label") or "")
        self.control = str(meta.get("_control") or "none")
        self.control_seed = int(meta.get("_control_seed") or 0)
        self.blocks = sorted({b for b, _s in entries})
        self.steps = sorted({s for _b, s in entries})
        self.pack_n_steps = (meta.get("steps") or {}).get("n_steps")

    # ----------------------------------------------------------- loading
    @classmethod
    def load(cls, pack_path: str, control: str = "none",
             control_seed: int = 0) -> "DeltaPack":
        if control not in CONTROLS:
            raise InjectError(
                f"concept_lab inject: control {control!r} is not one of "
                f"{list(CONTROLS)}")
        path = os.path.abspath(os.path.expanduser(str(pack_path or "")))
        if not os.path.isfile(path):
            raise InjectError(
                f"concept_lab inject: no delta pack at {path!r}. pack_path is "
                f"an absolute path to the .safetensors; its sidecar JSON sits "
                f"beside it")
        # both spellings are accepted because both are natural to write:
        # <pack>.safetensors.json and <pack>.json
        cands = [path + ".json", os.path.splitext(path)[0] + ".json"]
        side = next((c for c in cands if os.path.isfile(c)), None)
        if side is None:
            raise InjectError(
                f"concept_lab inject: {path} has no sidecar; looked for "
                f"{cands}. A pack without its sidecar is a bag of tensors "
                f"with no geometry and no provenance")
        with open(side) as fh:
            meta = json.load(fh)
        if meta.get("_type") != PACK_TYPE:
            raise InjectError(
                f"concept_lab inject: {side} has _type "
                f"{meta.get('_type')!r}, not {PACK_TYPE!r}")
        for field in ("latent_t", "frame_rows", "hidden"):
            if meta.get(field) is None:
                raise InjectError(
                    f"concept_lab inject: {side} carries no {field!r}. The "
                    f"guard that keeps a delta off rows it does not describe "
                    f"is (latent_t, frame_rows) and it cannot be inferred "
                    f"from a run")
        latent_t, frame_rows = int(meta["latent_t"]), int(meta["frame_rows"])
        hidden = int(meta["hidden"])
        sig = meta.get("layout_signature")
        if sig and int(sig[1]) != latent_t:
            raise InjectError(
                f"concept_lab inject: {side} disagrees with itself: "
                f"layout_signature {list(sig)} says latent_t={int(sig[1])} but "
                f"latent_t={latent_t}")

        raw = load_file(path)
        entries: dict = {}
        for key, t in raw.items():
            m = _KEY_RE.match(key)
            if not m:
                raise InjectError(
                    f"concept_lab inject: {path} carries key {key!r}, which is "
                    f"not b<block>_s<step>/frame_mean or .../patch_mean. An "
                    f"unreadable key is a delta nobody can place")
            block, step, which = int(m.group(1)), int(m.group(2)), m.group(3)
            want = (latent_t if which == "frame_mean" else frame_rows, hidden)
            if tuple(int(s) for s in t.shape) != want:
                raise InjectError(
                    f"concept_lab inject: {path} key {key!r} is "
                    f"{tuple(int(s) for s in t.shape)} but the sidecar says it "
                    f"must be {want} (latent_t={latent_t}, "
                    f"frame_rows={frame_rows}, hidden={hidden})")
            entries.setdefault((block, step), {})[which] = t.to(torch.float32)
        if not entries:
            raise InjectError(
                f"concept_lab inject: {path} carries no b<block>_s<step> "
                f"entries, so there is nothing to add")
        missing = sorted(k for k, v in entries.items()
                         if "frame_mean" not in v)
        if missing:
            raise InjectError(
                f"concept_lab inject: entries {missing} have a patch_mean but "
                f"no frame_mean. frame_mean is required for every (block, "
                f"step); patch_mean is the optional spatial half")

        meta = dict(meta, _control=control, _control_seed=int(control_seed),
                    _sidecar=side)
        pack = cls(path, meta, cls._apply_control(entries, control,
                                                  int(control_seed)))
        return pack

    # ---------------------------------------------------------- controls
    @staticmethod
    def _apply_control(entries: dict, control: str, seed: int) -> dict:
        """The arms of the first read, applied once, to the numbers only.

        `zero` is the pass-through control that still runs every line of the
        patch; `time_shuffle` keeps the delta's energy and destroys its
        timing; `sign_flip` points it the other way. The permutation is drawn
        on the CPU from `seed` so an arm is re-runnable from its label.
        """
        if control == "none":
            return entries
        out = {}
        for key in sorted(entries):
            v = dict(entries[key])
            if control == "zero":
                v = {k: torch.zeros_like(t) for k, t in v.items()}
            elif control == "sign_flip":
                v = {k: -t for k, t in v.items()}
            elif control == "time_shuffle":
                f = v["frame_mean"]
                g = torch.Generator().manual_seed(int(seed))
                perm = torch.randperm(f.shape[0], generator=g)
                v["frame_mean"] = f[perm].contiguous()
            out[key] = v
        return out

    # ------------------------------------------------------------ access
    def steps_for(self, block: int):
        return sorted(s for b, s in self.entries if b == block)

    def summary(self) -> dict:
        return {"path": self.path, "sidecar": self.meta.get("_sidecar"),
                "label": self.label, "latent_t": self.latent_t,
                "frame_rows": self.frame_rows, "hidden": self.hidden,
                "blocks": list(self.blocks), "steps": list(self.steps),
                "n_entries": len(self.entries),
                "pack_n_steps": self.pack_n_steps,
                "sources": list(self.meta.get("sources") or ()),
                "n_subjects": self.meta.get("n_subjects"),
                "control": self.control, "control_seed": self.control_seed}


# ---------------------------------------------------------- the session
class InjectSession:
    """One installed injection. Holds the pack, the layout stash, the cast
    cache and the tally.

    It composes `TapSession`'s pieces rather than subclassing it: the step
    arithmetic is the shared `schedule_and_sigma` / `step_from_sigma` pair
    and the layout signature is `TapSession._layout_signature`, so a capture
    of an injected run and the injection agree about which step and which
    rows they are talking about by construction.
    """

    def __init__(self, pack: DeltaPack, blocks=(), alpha: float = 1.0,
                 step_mode: str = "captured_only", mode: str = "frame",
                 cond_only: bool = True, enabled: bool = True):
        if step_mode not in STEP_MODES:
            raise InjectError(
                f"concept_lab inject: step_mode {step_mode!r} is not one of "
                f"{list(STEP_MODES)}")
        if mode not in MODES:
            raise InjectError(
                f"concept_lab inject: mode {mode!r} is not one of "
                f"{list(MODES)}")
        self.pack = pack
        self.alpha = float(alpha)
        self.step_mode = step_mode
        self.mode = mode
        self.cond_only = bool(cond_only)
        self.enabled = bool(enabled)

        blocks = tuple(int(b) for b in (blocks or ()))
        self.blocks = tuple(sorted(set(blocks))) if blocks \
            else tuple(pack.blocks)
        unknown = [b for b in self.blocks if b not in pack.blocks]
        if unknown:
            raise InjectError(
                f"concept_lab inject: block(s) {unknown} are not in the pack, "
                f"which carries {list(pack.blocks)}")
        if mode == "separable":
            no_patch = sorted(k for k in pack.entries
                              if k[0] in self.blocks
                              and "patch_mean" not in pack.entries[k])
            if no_patch:
                raise InjectError(
                    f"concept_lab inject: mode 'separable' needs patch_mean, "
                    f"and entries {no_patch} carry only frame_mean. Either "
                    f"build the pack with the spatial half or run mode "
                    f"'frame'")

        self._layout = None
        self._layout_sig = None
        self._video_span = None
        self._latent_t = None
        self._frame_rows = None
        self._sched = None
        self._n_steps = None
        self._cast: dict = {}
        self._logged = False

        self.dm_calls = 0
        self.calls_seen = 0
        self.calls_injected = 0
        self.injected: dict = {}                  # (block, cap_step) -> count
        self.norms: dict = {}                     # (block, cap_step) -> dict
        self.cond_or_uncond_seen = []

    # ------------------------------------------------------- installation
    def install(self, model_patcher):
        """Clone, refuse a block another patch already claims, hang the
        wrapper and the block patches on the clone."""
        m = model_patcher.clone()
        self._refuse_collisions(m)
        m.add_wrapper_with_key(DIFFUSION_MODEL, WRAPPER_KEY, self._dm_wrapper)
        for i in self.blocks:
            m.set_model_patch_replace(self._make_block_patch(int(i)),
                                      PATCH_NAME, PATCH_BLOCK, int(i))
        return m

    def _refuse_collisions(self, m) -> None:
        """Two patches on one block are one patch, silently.

        `set_model_options_patch_replace` (comfy/model_patcher.py:93-112)
        stores ONE callable per (name, block_name, index), so installing an
        inject on a block a tap already claims would drop the tap without a
        word. A separate patch NAME does not help: the block loop reads only
        `patches_replace["dit"]` (model.py:643-655), so a patch filed under
        any other name never fires at all. Hence: refuse, and say where to
        put it instead.
        """
        opts = getattr(m, "model_options", None) or {}
        existing = ((opts.get("transformer_options") or {})
                    .get("patches_replace") or {}).get(PATCH_NAME) or {}
        clash = [b for b in self.blocks if (PATCH_BLOCK, b) in existing]
        if clash:
            raise InjectError(
                f"concept_lab inject: block(s) {clash} already carry a "
                f"{PATCH_NAME}/{PATCH_BLOCK} patch on this model (the tap "
                f"installs one per tapped block). ComfyUI keeps one patch per "
                f"block and the second silently replaces the first, and a "
                f"patch under a different name is never called, so this "
                f"refuses instead. Tap and inject on DIFFERENT blocks: an "
                f"inject at block {clash[0]} is visible to any tap at a later "
                f"block, which is how you capture an injected run")

    # -------------------------------------------------------- the wrapper
    def _dm_wrapper(self, executor, x, timestep, context,
                    transformer_options=None, minimax_payload=None, **kw):
        """Stash the layout, check the geometry, change nothing, call on."""
        self.dm_calls += 1
        layout = (minimax_payload or {}).get("layout")
        if layout is None:
            raise InjectError(
                f"concept_lab inject: diffusion-model call #{self.dm_calls} "
                f"carried no packed layout in minimax_payload (keys: "
                f"{sorted((minimax_payload or {}).keys())}). Without a layout "
                f"the video rows cannot be found, and adding a delta to the "
                f"wrong rows is worse than not adding it")
        sig = TapSession._layout_signature(layout)
        if self._layout_sig is None:
            self._check_geometry(layout, sig)
            self._layout_sig = sig
        elif sig.identity() != self._layout_sig.identity():
            raise InjectError(
                f"concept_lab inject: the packed layout CHANGED during one "
                f"injection (call #{self.dm_calls}): {self._layout_sig.total_rows}"
                f" rows -> {sig.total_rows} rows. The delta was checked against "
                f"the first layout only")
        self._layout = layout
        out = executor(x, timestep, context, transformer_options,
                       minimax_payload=minimax_payload, **kw)
        self._maybe_log(transformer_options)
        return out

    def _check_geometry(self, layout, sig) -> None:
        """(latent_t, frame_rows) of the RUN, computed the way the tap
        computes them, against the pack's. Before any block fires."""
        span = sig.segment("video")
        if span is None:
            raise InjectError(
                f"concept_lab inject: this layout has no 'video' segment "
                f"(segments: {[s[0] for s in sig.segments]}). The delta lives "
                f"on the target-video rows")
        latent_t = int(layout.signature[1])
        lat_h, lat_w = int(layout.signature[2]), int(layout.signature[3])
        frame_rows = (lat_h // 2) * (lat_w // 2)      # model.py:70-81
        n_rows = span[1] - span[0]
        if latent_t * frame_rows != n_rows:
            raise InjectError(
                f"concept_lab inject: frame-row mismatch: latent_t={latent_t} "
                f"x frame_rows={frame_rows} = {latent_t * frame_rows}, but the "
                f"video segment has {n_rows} rows. The layout signature "
                f"{tuple(layout.signature)} does not describe these rows "
                f"(model.py:77-81, 397-402)")
        if (latent_t, frame_rows) != (self.pack.latent_t,
                                      self.pack.frame_rows):
            raise InjectError(
                f"concept_lab inject: this run is (latent_t, frame_rows) = "
                f"{(latent_t, frame_rows)} and the pack is "
                f"{(self.pack.latent_t, self.pack.frame_rows)} "
                f"({self.pack.path}). A delta added to rows that mean "
                f"something else is noise with a provenance; nothing was "
                f"injected. text_len and audio_t may differ, this may not")
        self._video_span = (int(span[0]), int(span[1]))
        self._latent_t, self._frame_rows = latent_t, frame_rows

    def _maybe_log(self, transformer_options) -> None:
        """One line on the ComfyUI console at the last step, once.

        There is no report OUTPUT on the node (one output, so a graph cannot
        wire the injection into a place it changes execution order), and a
        tally nobody can read is a tally nobody trusts.
        """
        if self._logged or self._n_steps is None:
            return
        to = transformer_options or {}
        if to.get("sigmas") is None or to.get("sample_sigmas") is None:
            return
        sched, n_steps, sigma = schedule_and_sigma(to, who="inject",
                                                   error=InjectError)
        if step_from_sigma(sched, n_steps, sigma) >= n_steps - 1:
            self._logged = True
            print("[concept_lab inject] " + json.dumps(self.report(),
                                                       sort_keys=True))

    # ----------------------------------------------------- the block patch
    def _make_block_patch(self, i: int):
        def fn(args, extra):
            out = extra["original_block"](args)
            if self.enabled:
                self._inject(i, out, args.get("transformer_options"))
            # the ORIGINAL dict, modified in place on its video rows: an
            # injection is arithmetic on the state, not a rebuilt pipeline
            return out
        return fn

    # ---------------------------------------------------------- selection
    def _pick_step(self, block: int, step: int):
        """Which captured step of this block answers for this run step."""
        caps = self.pack.steps_for(block)
        if not caps:
            return None
        if self.step_mode == "captured_only":
            return step if step in caps else None
        return min(caps, key=lambda c: (abs(c - step), c))   # ties -> earlier

    # ---------------------------------------------------------- the write
    def _inject(self, block: int, out, transformer_options) -> None:
        to = transformer_options or {}
        self.calls_seen += 1
        cou = to.get("cond_or_uncond")
        if isinstance(cou, (list, tuple)):
            tag = ",".join(str(int(c)) for c in cou)
            if tag not in self.cond_or_uncond_seen:
                self.cond_or_uncond_seen.append(tag)
            if self.cond_only and 0 not in [int(c) for c in cou]:
                return
        sched, n_steps, sigma = schedule_and_sigma(to, who="inject",
                                                   error=InjectError)
        if self._sched is None:
            self._sched, self._n_steps = sched, n_steps
        step = step_from_sigma(self._sched, self._n_steps, sigma)
        cap = self._pick_step(block, step)
        if cap is None:
            return
        if self._layout_sig is None:
            raise InjectError("concept_lab inject: a block fired before the "
                              "diffusion-model wrapper stashed a layout")
        h = out["img"]
        if int(h.shape[0]) != self._layout_sig.total_rows:
            raise InjectError(
                f"concept_lab inject: block {block} state has {h.shape[0]} "
                f"rows but the stashed layout has "
                f"{self._layout_sig.total_rows}; the video span "
                f"{self._video_span} does not point at video rows")
        if int(h.shape[1]) != self.pack.hidden:
            raise InjectError(
                f"concept_lab inject: this model's hidden width is "
                f"{int(h.shape[1])} and the pack's is {self.pack.hidden} "
                f"({self.pack.path})")
        a, b = self._video_span
        rows = h[a:b]
        if not rows.is_contiguous():
            raise InjectError(
                f"concept_lab inject: the video rows [{a}:{b}] of this block "
                f"state are not contiguous (shape {tuple(h.shape)}, strides "
                f"{tuple(h.stride())}), so the in-place write would not land "
                f"in the state the model carries on")
        g = rows.view(self._latent_t, self._frame_rows, int(h.shape[1]))

        f = self._cast_delta(block, cap, "frame", h.device, h.dtype)
        g.add_(f, alpha=self.alpha)
        if self.mode == "separable":
            p = self._cast_delta(block, cap, "patch", h.device, h.dtype)
            g.add_(p, alpha=self.alpha)
        self.calls_injected += 1
        self.injected[(block, cap)] = self.injected.get((block, cap), 0) + 1

    def _cast_delta(self, block: int, cap: int, which: str, device, dtype):
        """The delta as this state's dtype and device, cast once and kept.

        `frame` is `[latent_t, 1, hidden]` and broadcasts over the patch
        rows. `patch` is the separable mode's second term, `[1, frame_rows,
        hidden]`, already centred: the rank-2 approximation is frame + patch
        - grand mean, and the grand mean rides inside frame_mean, so it is
        subtracted from the patch map once here rather than counted twice.
        """
        key = (block, cap, which, str(device), str(dtype))
        hit = self._cast.get(key)
        if hit is not None:
            return hit
        e = self.pack.entries[(block, cap)]
        if which == "frame":
            d = e["frame_mean"].unsqueeze(1)
        else:
            grand = e["frame_mean"].mean(dim=0, keepdim=True)
            d = (e["patch_mean"] - grand).unsqueeze(0)
        n = self.norms.setdefault((block, cap), {})
        n[which] = float(torch.linalg.vector_norm(d).item()) * abs(self.alpha)
        d = d.to(device=device, dtype=dtype)
        self._cast[key] = d
        return d

    # ------------------------------------------------------------ report
    def report(self) -> dict:
        return {
            "enabled": self.enabled, "alpha": self.alpha, "mode": self.mode,
            "step_mode": self.step_mode, "cond_only": self.cond_only,
            "blocks": list(self.blocks),
            "pack": self.pack.summary(),
            "dm_calls": self.dm_calls, "calls_seen": self.calls_seen,
            "calls_injected": self.calls_injected,
            "cond_or_uncond_seen": list(self.cond_or_uncond_seen),
            "n_steps_run": self._n_steps,
            "layout": {"latent_t": self._latent_t,
                       "frame_rows": self._frame_rows,
                       "video_span": list(self._video_span or ())},
            "injected": {f"b{b:02d}_s{s:02d}": c
                         for (b, s), c in sorted(self.injected.items())},
            "delta_norms": {f"b{b:02d}_s{s:02d}": v
                            for (b, s), v in sorted(self.norms.items())},
        }


def install(model_patcher, session: InjectSession):
    """Module-level door, matching h3_tap.install."""
    return session.install(model_patcher)
