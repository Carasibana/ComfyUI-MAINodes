"""MiniMax-H3, the first backend.

Everything here was read off the installed ComfyUI on 2026-08-15 rather
than taken from a document, and the anchors are cited so the next person
can re-check them in one grep. Upstream moves; a stale anchor should be
findable, not archaeological.

VERIFIED ANCHORS (ComfyUI 7fe8a613, comfy/ldm/minimax/model.py)

  :300  class PackedLayout builds a contiguous named segment table. The
        packed order is
            [ text | cond | cond_audio | ref_img | ref_audio | audio | video ]
        with target audio then target video always last (:390-398). Segments
        are exposed as (start, stop, kind) triples on layout.segments (:420).
        This is the alignment surface: named segments, never row index.

  :643-656  the DiT block loop reads
            transformer_options["patches_replace"]["dit"][("double_block", i)]
        and calls it with {"img", "t_emb", "mod_segments", "rope_freqs",
        "transformer_options"} plus an original_block callable. That is the
        instrumentation seam, and it means a capture can observe block state
        without editing core.

  :424  defaults hidden_size=5376, num_layers=50. Read the loaded config
        rather than trusting these; they are here to make a stale assumption
        visible, not to be depended on.

  comfy/model_patcher.py:667  set_model_patch_replace(patch, name,
        block_name, number) writes exactly that key, so the tap is installed
        through the public ModelPatcher API on a clone.

THE CONFOUND THIS BACKEND EXISTS TO CONTROL. PackedLayout advances its
cursor past every reference block before it places the target streams, so
h(with reference) - h(no reference) mixes three things at once: the
reference's semantic content, the fact that a reference is present at all
(including its Qwen3-VL presentation tokens), and a moved time origin for
the target stream. The shape-matched control arm is not a nicety here. It
is the difference between measuring content and measuring presence.
"""
from __future__ import annotations

from concept_lab import types as T
from concept_lab.backends.base import Backend, BackendError

# The packed layout's own words, in packed order. Read from PackedLayout's
# segment appends (model.py:307-398).
H3_SEGMENTS = ("text", "cond", "cond_audio", "ref_img", "ref_audio",
               "audio", "video")

# What a v0 capture keeps. "video" is the target stream and carries the
# question everyone actually has: how did the reference change what gets
# generated. "ref_img" is diagnostic secondary evidence about how the
# reference is represented, which is a different and lesser question.
H3_DEFAULT_SEGMENTS = ("video", "ref_img")

# A first ladder over 50 blocks. This is an experiment config and is
# expected to be replaced by a measured depth profile. It is deliberately
# not written into any serialized format.
H3_DEFAULT_BLOCKS = (0, 5, 10, 20, 30, 40, 49)

PATCH_NAME = "dit"
PATCH_BLOCK = "double_block"


class H3Backend(Backend):
    name = "h3"

    def capabilities(self) -> dict:
        return {
            "backend": self.name,
            "reference_modalities": ["image", "video", "audio"],
            "segments": list(H3_SEGMENTS),
            "target_segments": ["video", "audio"],
            "tap_seam": f'patches_replace["{PATCH_NAME}"][("{PATCH_BLOCK}", i)]',
            "tap_installed_via": "ModelPatcher.set_model_patch_replace",
            "expected_blocks": 50,
            "expected_width": 5376,
            # The tap exists (T4) and runs INSIDE the ComfyUI process through
            # the two alpha nodes; there is no out-of-process capture verb to
            # call, and there is not going to be one.
            "supports_capture": "via_node",
            "supports_compile": False,      # T9
            "notes": [
                "references advance the target cursor, so a two-arm "
                "reference on/off delta is confounded by layout and "
                "presentation; use the shape-matched control arm",
                "reference rows carry their own timestep class, not the "
                "target video timestep, so segment role has to travel with "
                "every captured tensor",
                "keyframe conditioning combined with references has known "
                "upstream interactions; keep keyframes out of the first "
                "capture",
            ],
        }

    def cost_model(self) -> dict:
        """What a capture actually costs here, which is not forwards.

        A video reference block adds roughly a second video's worth of
        tokens, and per-step cost is superlinear in the packed sequence, so
        the R arm and the 0 arm are not two units of the same thing.
        """
        return {
            "video_reference_sequence_multiplier": 2.0,
            "step_cost_exponent_range": [1.5, 1.7],
            "note": "per-step cost is superlinear in packed sequence length; "
                    "a video reference block adds roughly a second video's "
                    "worth of tokens (measured exponent 1.52 over 27-47 "
                    "temporal tokens at 0.74 MP, ~1.7 at larger sequences; "
                    "h3-latent-mechanics SKILL §4)",
        }

    def default_blocks(self) -> tuple:
        return H3_DEFAULT_BLOCKS

    def default_segments(self) -> tuple:
        return H3_DEFAULT_SEGMENTS

    def segment_vocabulary(self) -> tuple:
        return H3_SEGMENTS

    def validate_tap(self, tap: T.TapSpec) -> None:
        super().validate_tap(tap)
        bad = [b for b in tap.blocks if not isinstance(b, int) or b < 0]
        if bad:
            raise BackendError(f"h3: block indices must be non-negative "
                               f"ints, got {bad}")
        if tap.blocks and max(tap.blocks) > 63:
            raise BackendError(
                f"h3: block {max(tap.blocks)} is past the 50-block default "
                f"stack. If a checkpoint really is deeper, read its config "
                f"and say so in the capture manifest rather than widening "
                f"this check")

    def plan_arms(self, source: T.ConditioningSource) -> dict:
        """What a shape-matched control has to match for this source.

        Returned as data rather than performed, because the control asset is
        the operator's choice and several strata should be tried: a neutral
        image with matched global statistics, an unrelated same-class image,
        a blurred or averaged source, a shuffled latent preserving norm and
        spectrum. A blank grey frame is the tempting one and is probably too
        far out of distribution to be a fair control.

        `assert_before_subtracting` is a checklist to enforce, not advice.
        Two of its entries are read off the installed model:
        `comfy/ldm/minimax/model.py:484-491` applies visual_cond_noise_aug to
        the reference rows, and `:562-566` draws that noise from a generator
        restarted from the payload seed. So an R and an N that disagree on
        either the augmentation level or the seed differ by a noise draw as
        well as by content, and R-N silently prices both.
        """
        pre = dict(source.preprocess or {})
        must_match = ["modality", "reference_count", "width", "height"]
        if source.kind == "video":
            must_match += ["frames", "fps"]
        return {
            "arm_R": {"role": "actual reference", "source": source.to_dict()},
            "arm_N": {
                "role": "shape-matched control",
                "must_match": must_match,
                "preprocess": pre,
                "strata_to_try": [
                    "neutral image with matched global statistics",
                    "unrelated image from the same broad class",
                    "blurred or averaged version of the source",
                    "shuffled latent preserving norm and spectrum",
                ],
                "why": "R and N must produce identical packed layouts and "
                       "identical presentation token counts, so that R-N is "
                       "semantic content and nothing else",
            },
            "arm_0": {
                "role": "no reference",
                "why": "N-0 measures what mere presence costs. It is a "
                       "diagnostic, and on this model it is not expected to "
                       "be small",
            },
            "assert_before_subtracting": [
                "layout signatures of R and N are equal",
                "position ids of the target video segment are equal",
                "presentation token counts are equal",
                "visual_cond_noise_aug equal on R and N",
                "seed equal on R and N (condition-row noise is drawn from a "
                "generator restarted from the payload seed)",
            ],
        }

    def fingerprint(self, **kw) -> T.ModelStackFingerprint:
        raise BackendError(
            "h3.fingerprint needs a live ModelPatcher, which does not cross "
            "the api boundary (DEC-CL-0011). Inside the process, call "
            "concept_lab.backends.h3_tap.fingerprint_from_patcher(patcher, "
            "stack_label); it reads the model class, config class, parameter "
            "dtype and weight-patch stack, and takes the checkpoint NAME "
            "from the caller because a ModelPatcher does not carry it.")

    def tap_install(self, model_patcher, session):
        """Hang a TapSession on a clone of `model_patcher`.

        The import is local because everything above this line stays
        importable in a process with no torch and no ComfyUI in it, and
        `h3_tap` is the file where both arrive.
        """
        from concept_lab.backends import h3_tap                 # noqa: PLC0415
        return h3_tap.install(model_patcher, session)

    def capture(self, *a, **k):
        raise BackendError(
            "h3.capture stays refusing on purpose. Capture runs INSIDE the "
            "ComfyUI process (the states live for microseconds inside the "
            "block loop and the launch is part of the condition), so the "
            "path is the MAIConceptCaptureArm / MAIConceptCaptureFlush node "
            "pair over backends/h3_tap.py, not a verb called from outside. "
            "The gate it still has to clear is E0: a pass-through tap must "
            "be numerically identical to the unpatched model at the same "
            "seed and settings, and capture disabled must be a no-op.")

    def compile(self, *a, **k):
        raise BackendError("h3.compile is T9 and blocked on a factor that "
                           "has cleared its held-out gates.")
