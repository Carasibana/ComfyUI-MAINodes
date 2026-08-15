"""The ComfyUI surface of concept_lab: three thin nodes, no rules.

There is exactly one reason these exist this early (DEC-CL-0013 said nodes
wait for T12, DEC-CL-0018 amends its TIMING): a capture has to happen inside
the ComfyUI process, so the seam needs a node before anything can be
measured. They hold no policy. The Arm node builds a TapSpec, reads a stack
fingerprint off the live ModelPatcher, opens a TapSession and installs it on
a CLONE of the model; the Flush node calls finalize(). Every rule is in
concept_lab/, and every refusal comes from there too.

    [MODEL] -> MAI Concept Capture Arm -> [MODEL] -> KSampler -> ... -> decode
                        |                                                 |
                        +--> CONCEPT_CAPTURE -> MAI Concept Capture Flush <+
                                                        (after)

The `after` input is what orders the flush behind the decode. ComfyUI runs
the graph by data dependency, and a flush that is not downstream of anything
can execute before the sampler has taken a single step.

The third node is the mirror of the first (DEC-CL-0020): where the tap READS
block states, `MAI Concept Inject Delta` ADDS a stored one back, so the same
seam serves the capture and the premise test that spends it. It holds no
rules either; the pack, the geometry guard and the controls live in
`concept_lab/backends/h3_inject.py`.

ALPHA. The pass-through gate (E0: tap-installed-disabled renders must be
pixel-identical to the same seed unpatched) has NOT been run live yet. If
these nodes fail it, they get pulled rather than patched around.
"""
from __future__ import annotations

import json
import os
import sys

# ComfyUI loads a pack by file location and does NOT put the pack directory
# on sys.path (nodes.py:2243-2265), while every module under concept_lab/
# imports itself absolutely (`from concept_lab import ...`) so that the CLI,
# the tests and the MCP server all address one module identity. Same idiom
# as cli.py:33-34.
_PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PACK not in sys.path:
    sys.path.insert(0, _PACK)

from concept_lab import types as T                              # noqa: E402
from concept_lab.backends import h3_inject                      # noqa: E402
from concept_lab.backends import h3_tap                         # noqa: E402

CATEGORY = "MAI/concept (alpha)"


class _AnyType(str):
    def __ne__(self, other):
        return False


ANY = _AnyType("*")


def _ints(s, field, who="MAI Concept Capture Arm"):
    out = []
    for part in str(s or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            raise ValueError(f"{who}: {field} wants "
                             f"comma-separated integers, got {part!r}")
    return tuple(out)


def _floats(s, field):
    out = []
    for part in str(s or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(float(part))
        except ValueError:
            raise ValueError(f"MAI Concept Capture Arm: {field} wants "
                             f"comma-separated numbers, got {part!r}")
    return tuple(out)


def _words(s):
    return tuple(p.strip() for p in str(s or "").split(",") if p.strip())


class MAIConceptCaptureArm:
    """Install a capture tap on one arm of a comparison."""

    DESCRIPTION = (
        "EXPERIMENTAL (alpha), new 2026-08-15. Installs the concept_lab "
        "block tap on a CLONE of the model and hands the clone on, so the "
        "sampler downstream renders exactly as it would have while selected "
        "DiT block states are written to the concept space.\n\n"
        "One node is ONE ARM of one comparison (R = the actual reference, "
        "N = a shape-matched control, 0 = no reference). All arms of a "
        "comparison have to run in ONE ComfyUI process: this pipeline is "
        "bit-identical within a launch and only close across launches, so a "
        "restart makes a new replicate rather than the same condition "
        "(DEC-CL-0015).\n\n"
        "enabled=False is the pass-through control: the tap is installed and "
        "records nothing, which is how you prove the instrumentation is free."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL", {"tooltip": "the model this arm renders with; "
                                "a patched CLONE comes out, the input is "
                                "untouched"}),
            "arm": (list(T.ARMS), {"default": "R",
                    "tooltip": "R actual reference / N shape-matched control "
                               "/ 0 no reference / free (uncontrolled)"}),
            "variant_name": ("STRING", {"default": "",
                             "tooltip": "a label; identity comes from what "
                                        "the model actually saw (text "
                                        "context, reference latents), "
                                        "measured by the tap"}),
            "seed": ("INT", {"default": 0, "min": 0,
                     "max": 0xffffffffffffffff,
                     "tooltip": "MUST equal the sampler's noise seed. This "
                                "node cannot read it (the seed lives in the "
                                "sampler, not in the model), so it is copied "
                                "by hand and a mismatch silently mislabels "
                                "the capture; the tap now refuses on "
                                "mismatch"}),
            "replicate": ("INT", {"default": 0, "min": 0, "max": 999,
                          "tooltip": "which repeat of this condition, inside "
                                     "this launch"}),
            "blocks": ("STRING", {"default": "0,5,10,20,30,40,49",
                       "tooltip": "DiT block indices to tap. An experiment "
                                  "config, not a format fact"}),
            "step_quantiles": ("STRING", {"default": "0.15,0.5,0.85",
                               "tooltip": "schedule-relative positions; 0 is "
                                          "the first step, 1 the last"}),
            "steps": ("STRING", {"default": "",
                      "tooltip": "explicit step indices instead; giving both "
                                 "this and step_quantiles is refused"}),
            "segments": ("STRING", {"default": "video,ref_img",
                         "tooltip": "packed-layout segment names: text, cond, "
                                    "cond_audio, ref_img, ref_audio, audio, "
                                    "video"}),
            "store": ("STRING", {"default": "stats,frame_mean,sketch",
                      "tooltip": "comma-joined subset of stats, frame_mean, "
                                 "sketch, full. 'full' is the whole state and "
                                 "costs accordingly"}),
            "sketch_dim": ("INT", {"default": 256, "min": 8, "max": 4096,
                           "tooltip": "width of the seeded random projection"}),
            "sketch_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff,
                            "tooltip": "recorded in the manifest; an "
                                       "unrecorded projection is an "
                                       "unreproducible capture"}),
            "stack_label": ("STRING", {"default": "",
                            "tooltip": "the checkpoint name, from you. A "
                                       "ModelPatcher does not carry the UNET "
                                       "filename and this node will not guess "
                                       "one"}),
        }, "optional": {
            "sampler_label": ("STRING", {"default": "",
                              "tooltip": "sampler/scheduler/shift as you ran "
                                         "them; it joins the capture's "
                                         "identity, so two arms must agree"}),
            "corpus_row_id": ("STRING", {"default": "",
                              "tooltip": "the corpus row this arm renders, if "
                                         "there is one"}),
            "enabled": ("BOOLEAN", {"default": True,
                        "tooltip": "False installs the tap and records "
                                   "NOTHING: the pass-through control"}),
            "cond_only": ("BOOLEAN", {"default": True,
                          "tooltip": "skip the unconditional forward; CFG "
                                     "makes two forwards per step and they "
                                     "are two different conditions"}),
            "index_root": ("STRING", {"default": "",
                           "tooltip": "manifest root; empty = the space "
                                      "default (H3_CONCEPT_INDEX_DIR)"}),
            "tensor_root": ("STRING", {"default": "",
                            "tooltip": "tensor root; empty = the space "
                                       "default (H3_CONCEPT_TENSOR_DIR)"}),
            "notes": ("STRING", {"default": "", "multiline": True}),
        }}

    RETURN_TYPES = ("MODEL", "CONCEPT_CAPTURE")
    RETURN_NAMES = ("model", "capture")
    FUNCTION = "arm"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # A capture is a measurement, and a cached measurement is not one:
        # re-running the same graph must build a NEW session and record
        # again, so this node never reports itself unchanged.
        return float("nan")

    def arm(self, model, arm, variant_name, seed, replicate, blocks,
            step_quantiles, steps, segments, store, sketch_dim, sketch_seed,
            stack_label, sampler_label="", corpus_row_id="", enabled=True,
            cond_only=True, index_root="", tensor_root="", notes=""):
        step_idx = _ints(steps, "steps")
        tap = T.TapSpec(
            blocks=_ints(blocks, "blocks"),
            steps=step_idx,
            step_quantiles=() if step_idx else _floats(step_quantiles,
                                                       "step_quantiles"),
            segments=_words(segments), store=store,
            sketch_dim=int(sketch_dim), sketch_seed=int(sketch_seed))

        from concept_lab.backends import get_backend
        get_backend("h3").validate_tap(tap)

        stack = h3_tap.fingerprint_from_patcher(
            model, stack_label or "UNNAMED (stack_label was left empty)")
        session = h3_tap.TapSession(
            tap=tap, arm=arm, variant_name=variant_name or "unnamed",
            seed=int(seed), replicate=int(replicate), stack=stack,
            index_root=index_root or None, tensor_root=tensor_root or None,
            corpus_row_id=corpus_row_id or None, enabled=bool(enabled),
            cond_only=bool(cond_only), notes=notes,
            sampler_label=sampler_label)
        return (session.install(model), session)


class MAIConceptCaptureFlush:
    """File the capture. Wire `after` to something the decode produced."""

    DESCRIPTION = (
        "EXPERIMENTAL (alpha), new 2026-08-15. Finalizes a capture: writes "
        "the manifest into the concept space and reports what landed.\n\n"
        "`after` accepts anything and is used ONLY to order execution. "
        "ComfyUI runs a graph by data dependency, so a flush that is not "
        "downstream of the render can fire before the sampler has taken a "
        "step; wire the decoded images (or whatever the last node emits) "
        "into `after` and the flush happens when the render is done.\n\n"
        "The tensors are already on disk by then: the tap writes as it "
        "captures rather than holding a run's worth of state in VRAM.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "capture": ("CONCEPT_CAPTURE", {}),
            "after": (ANY, {"tooltip": "execution ordering only; wire the "
                            "decode's output so the flush runs last"}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "flush"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def flush(self, capture, after=None):
        # `after` is the decode's IMAGE batch in every graph that wires this
        # correctly, so it is also free evidence about what the render came
        # out as: hashed, recorded in the notes, never part of identity
        # (DEC-CL-0006). Anything that is not an image tensor records null.
        return (json.dumps(
            capture.finalize(
                render_fingerprint=h3_tap.render_fingerprint(after)),
            indent=2, sort_keys=True),)


class MAIConceptInjectDelta:
    """Add a stored delta to the target-video rows at chosen blocks/steps."""

    DESCRIPTION = (
        "EXPERIMENTAL (alpha), new 2026-08-15. The mirror of the capture tap: "
        "loads a delta pack (a .safetensors of per-(block, step) frame_mean "
        "maps plus its sidecar JSON) and ADDS it, times alpha, to the "
        "target-video rows of the block output, on a CLONE of the model.\n\n"
        "It answers one question and holds no theory: does a delta measured "
        "at the tap do anything when it is put back? Run it against its own "
        "controls. control='zero' runs every line of the patch with zeroes "
        "(if that render differs from the unpatched one, the plumbing is the "
        "effect); control='time_shuffle' keeps the delta's energy and "
        "destroys its timing (if that renders the same as 'none', the timing "
        "is not what is doing the work).\n\n"
        "The pack's (latent_t, frame_rows) must match the run's: the node "
        "refuses before the first block otherwise. A different prompt or a "
        "different audio length is fine, the delta lives on video rows.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL", {"tooltip": "a patched CLONE comes out; the "
                                "input is untouched"}),
            "pack_path": ("STRING", {"default": "",
                          "tooltip": "absolute path to the pack "
                                     ".safetensors; its sidecar <pack>.json "
                                     "(or <pack>.safetensors.json) must sit "
                                     "beside it"}),
            "blocks": ("STRING", {"default": "",
                       "tooltip": "comma-separated DiT blocks to inject at; "
                                  "empty = every block the pack carries. Must "
                                  "not collide with a tap's blocks on the "
                                  "same model"}),
            "alpha": ("FLOAT", {"default": 1.0, "min": -8.0, "max": 8.0,
                      "step": 0.05,
                      "tooltip": "scale on the delta; negative subtracts"}),
            "step_mode": (list(h3_inject.STEP_MODES),
                          {"default": "captured_only",
                           "tooltip": "captured_only fires only on the steps "
                                      "the pack was captured at; nearest_all "
                                      "fires every step using the nearest "
                                      "captured entry"}),
            "mode": (list(h3_inject.MODES), {"default": "frame",
                     "tooltip": "frame = one vector per latent frame; "
                                "separable = frame + patch - grand mean, the "
                                "rank-2 map (needs patch_mean in the pack)"}),
            "control": (list(h3_inject.CONTROLS), {"default": "none",
                        "tooltip": "the arm: none / zero (plumbing control) / "
                                   "time_shuffle (same energy, wrong timing) "
                                   "/ sign_flip"}),
            "control_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff,
                             "tooltip": "seeds time_shuffle's permutation, on "
                                        "the CPU, so an arm is re-runnable "
                                        "from its label"}),
        }, "optional": {
            "cond_only": ("BOOLEAN", {"default": True,
                          "tooltip": "inject only on the conditional forward; "
                                     "CFG makes two per step and they are two "
                                     "different conditions"}),
            "enabled": ("BOOLEAN", {"default": True,
                        "tooltip": "False installs the patches and adds "
                                   "NOTHING: the pass-through control"}),
        }}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "inject"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # same reason as the Arm node: an injection is an experiment, and a
        # cached experiment is not one
        return float("nan")

    def inject(self, model, pack_path, blocks, alpha, step_mode, mode,
               control, control_seed, cond_only=True, enabled=True):
        pack = h3_inject.DeltaPack.load(pack_path, control=control,
                                        control_seed=int(control_seed))
        session = h3_inject.InjectSession(
            pack=pack,
            blocks=_ints(blocks, "blocks", "MAI Concept Inject Delta"),
            alpha=float(alpha),
            step_mode=step_mode, mode=mode, cond_only=bool(cond_only),
            enabled=bool(enabled))
        return (session.install(model),)


NODE_CLASS_MAPPINGS = {
    "MAIConceptCaptureArm": MAIConceptCaptureArm,
    "MAIConceptCaptureFlush": MAIConceptCaptureFlush,
    "MAIConceptInjectDelta": MAIConceptInjectDelta,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MAIConceptCaptureArm": "MAI Concept Capture Arm (alpha)",
    "MAIConceptCaptureFlush": "MAI Concept Capture Flush (alpha)",
    "MAIConceptInjectDelta": "MAI Concept Inject Delta (alpha)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
