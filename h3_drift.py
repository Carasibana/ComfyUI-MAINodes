"""H3 Drift Control (alpha): schedule-matched noise on the carried video prefix.

A chained segment's carried prefix is bit-clean conditioning. One seam loves
that; a CHAIN does not: repeated clean conditioning accumulates contrast and
texture error, and our seam probe measures the decay directly (join1 ~0.9,
join2 ~0.65 across every variant tested 2026-08-24). The fix is to give the
disposable carried video prefix a small, scheduler-matched amount of the
sampler's own noise at every model evaluation, tapering to exact at the seam
end, so the model never treats the prefix as impossibly clean.

Mechanism and recipe ADAPTED from ComfyUI-MiniMaxH3-Contex-Loop's
drift_control.py (NikoDemon80 / ethanfel, GPL-3.0) - same license as this
pack. Their doctrine, kept: the dynamic mask must be installed in TWO places
that agree (ComfyUI's denoise-mask function AND an apply-model wrapper),
because without the second path H3 labels the prefix clean while receiving a
noisy input. Their sigma-split sampler handoff wrapper is deliberately NOT
ported: our graphs run one SamplerCustomAdvanced per pass. Audio is
deliberately untouched - the frozen per-tick audio prefix stays hard.

Wire: H3 V2V Init latent -> here (for shape + mask presence), the SAME model
your pass-1 guider uses -> here, guider takes THIS node's model out.
prefix_frames must equal the plan's handle (39 frames = 12 video steps).
"""

import math

import torch

NODE_NAME = "H3DriftControl"
MASK_QUANTIZATION = 256.0


# ------------------------------------------------------------- pure functions

def _schedule_values(sigmas):
    """Finite, non-negative schedule values, descending, deduplicated."""
    if torch.is_tensor(sigmas):
        values = sigmas.detach().float().reshape(-1).cpu().tolist()
    else:
        values = list(sigmas or ())
    out = set()
    for v in values:
        v = float(v)
        if math.isfinite(v) and v >= 0.0:
            out.add(v)
    return tuple(sorted(out, reverse=True))


def next_schedule_sigma(current_sigma, sigmas):
    """The next strictly lower sigma in the schedule, else 0."""
    current = float(current_sigma)
    if not math.isfinite(current) or current <= 0.0:
        return 0.0
    tol = max(1e-7, abs(current) * 1e-6)
    for candidate in _schedule_values(sigmas):
        if candidate < current - tol:
            return candidate
    return 0.0


def matched_noise_ratio(current_sigma, sigmas):
    """next_sigma / current_sigma clamped to [0, 1]."""
    current = float(current_sigma)
    if not math.isfinite(current) or current <= 0.0:
        return 0.0
    return max(0.0, min(1.0, next_schedule_sigma(current, sigmas) / current))


def temporal_prefix_weights(prefix_steps, taper_steps):
    """Matched weights 1.0 over the old prefix, tapering to 0 at the seam end.

    taper 4 gives .75 .50 .25 .00: the LAST carried latent step stays exact at
    the generated-future boundary."""
    count, taper = int(prefix_steps), int(taper_steps)
    if count < 1:
        raise ValueError("drift control: prefix_steps must be positive")
    if taper < 1 or taper > count:
        raise ValueError("drift control: taper_steps must be in [1, prefix_steps]")
    weights = [1.0] * (count - taper)
    weights.extend(float(taper - i - 1) / float(taper) for i in range(taper))
    return tuple(weights)


def apply_dynamic_prefix_mask(packed_mask, video_shape, prefix_steps, ratio,
                              taper_steps):
    """Rewrite the video-prefix region of a packed [B,1,N] denoise mask.

    Returns (packed mask, one-channel H3 video mask). The H3 mask is
    ceil-quantized to 1/256 like core's merged-mask path, so both install
    points agree on native and compatibility cores."""
    if not torch.is_tensor(packed_mask) or packed_mask.ndim != 3:
        raise ValueError("drift control: sampler denoise mask must be [B,1,N]")
    shape = tuple(int(v) for v in video_shape)
    if len(shape) != 5 or shape[0] != int(packed_mask.shape[0]):
        raise ValueError("drift control: expected video latent [B,C,T,H,W], got %s"
                         % (shape,))
    steps = int(prefix_steps)
    if steps < 1 or steps > shape[2]:
        raise ValueError("drift control: prefix %d steps outside the %d-step latent"
                         % (steps, shape[2]))
    video_elements = math.prod(shape[1:])
    if int(packed_mask.shape[-1]) < video_elements:
        raise ValueError("drift control: packed mask shorter than its video stream")

    out = packed_mask.clone()
    video = out[..., :video_elements].reshape(shape)
    weights = torch.tensor(temporal_prefix_weights(steps, taper_steps),
                           device=video.device, dtype=video.dtype)
    weights = weights.mul_(float(max(0.0, min(1.0, ratio))))
    video[:, :, :steps] = weights.view(1, 1, steps, 1, 1)
    h3_video_mask = torch.ceil(video[:, :1].float() * MASK_QUANTIZATION) / MASK_QUANTIZATION
    return out, h3_video_mask


def frames_to_steps(frames):
    """17k+5 pixel frames -> 5k+2 video latent steps (39 -> 12)."""
    f = int(frames)
    if f < 5 or (f - 5) % 17:
        raise ValueError("drift control: prefix_frames must sit on the 17k+5 grid "
                         "(39, 56, 90, ...); got %d" % f)
    return 5 * ((f - 5) // 17) + 2


# ---------------------------------------------------------------- hook state

class _DriftState:
    def __init__(self, video_shape, prefix_steps, matched_steps, taper_steps,
                 schedule_override=None):
        if int(prefix_steps) != int(matched_steps) + int(taper_steps):
            raise ValueError(
                "drift control: prefix has %d video steps but matched %d + "
                "taper %d were asked for - they must sum exactly"
                % (prefix_steps, matched_steps, taper_steps))
        self.video_shape = tuple(int(v) for v in video_shape)
        self.prefix_steps = int(prefix_steps)
        self.taper_steps = int(taper_steps)
        self.schedule_override = _schedule_values(schedule_override)
        self.current_video_mask = None
        self.last_sigma = None
        self.last_ratio = None

    def denoise_mask_function(self, sigma, denoise_mask, extra_options=None):
        current = float(torch.as_tensor(sigma).detach().float().reshape(-1)[0])
        schedule = self.schedule_override or (extra_options or {}).get("sigmas", ())
        ratio = matched_noise_ratio(current, schedule)
        out, video_mask = apply_dynamic_prefix_mask(
            denoise_mask, self.video_shape, self.prefix_steps, ratio,
            self.taper_steps)
        self.current_video_mask = video_mask
        self.last_sigma, self.last_ratio = current, ratio
        return out

    def apply_model_wrapper(self, executor, *args, **kwargs):
        # H3's per-row timestep conditioning must see the SAME dynamic video
        # mask the sampler mixes with, or the prefix is labeled clean while
        # receiving noisy input (the two-install-points doctrine).
        if self.current_video_mask is not None:
            kwargs["denoise_mask"] = self.current_video_mask
        return executor(*args, **kwargs)


# --------------------------------------------------------------------- node

class H3DriftControl:
    """Clone the model and install the coupled dynamic-prefix-mask hooks."""

    DESCRIPTION = (
        "EXPERIMENTAL (alpha). Schedule-matched noise on the carried VIDEO "
        "prefix of a chained segment, tapering to exact at the seam end - "
        "stops clean-conditioning error accumulating down a chain (measured "
        "join decay 0.9 -> 0.65 by link two, 2026-08-24). Wire the H3 V2V "
        "Init latent here for shape, feed the SAME base model your pass-1 "
        "guider uses, and give the guider THIS model output. prefix_frames "
        "must equal the plan's handle (39 f = 12 steps = matched 8 + taper "
        "4, the field-validated recipe). Audio is untouched: the frozen "
        "per-tick audio prefix stays hard. Adapted from Contex-Loop's "
        "Drift-Control AV (GPL-3.0). Refuses to stack on another dynamic "
        "denoise-mask patch, and does not support sigma-split samplers.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL", {"tooltip": "the base H3 model your pass-1 guider would use"}),
            "latent": ("LATENT", {"tooltip": "H3 V2V Init output: supplies the video shape and must carry the static noise_mask"}),
        }, "optional": {
            "prefix_frames": ("INT", {"default": 39, "min": 5, "max": 3600,
                "tooltip": "carried handle in pixel frames, on the 17k+5 grid; 39 -> 12 latent steps"}),
            "matched_steps": ("INT", {"default": 8, "min": 0, "max": 400,
                "tooltip": "older prefix steps held at full matched noise"}),
            "taper_steps": ("INT", {"default": 4, "min": 1, "max": 400,
                "tooltip": "steps ramping matched -> exact toward the seam end; matched+taper must equal the prefix's latent steps"}),
            "full_sigmas": ("SIGMAS", {"tooltip": "optional: the complete schedule if the sampler only sees a partial one; usually leave unwired"}),
        }}

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "report")
    FUNCTION = "install"
    CATEGORY = "MAINodes/alpha"

    def install(self, model, latent, prefix_frames=39, matched_steps=8,
                taper_steps=4, full_sigmas=None):
        samples = latent.get("samples") if isinstance(latent, dict) else None
        if hasattr(samples, "is_nested") and samples.is_nested:
            video = samples.tensors[0]
        elif torch.is_tensor(samples):
            video = samples
        else:
            raise ValueError("drift control: latent has no video tensor")
        if "noise_mask" not in latent:
            raise ValueError(
                "drift control: the latent carries no noise_mask - wire H3 V2V "
                "Init (masked prefix) first; drift control rewrites that mask, "
                "it does not create one")
        steps = frames_to_steps(prefix_frames)

        patched = model.clone()
        options = getattr(patched, "model_options", None)
        if not isinstance(options, dict):
            raise ValueError("drift control: model has no model_options")
        if callable(options.get("denoise_mask_function")):
            raise ValueError(
                "drift control: another dynamic denoise-mask patch is already "
                "installed on this model - they cannot stack")

        from comfy.patcher_extension import WrappersMP
        state = _DriftState(tuple(video.shape), steps, matched_steps,
                            taper_steps, schedule_override=full_sigmas)
        patched.set_model_denoise_mask_function(state.denoise_mask_function)
        patched.add_wrapper_with_key(WrappersMP.APPLY_MODEL,
                                     "mainodes_h3_drift_control",
                                     state.apply_model_wrapper)
        patched.model_options["mainodes_h3_drift_control_state"] = state
        text = ("H3 Drift Control: %d-step video prefix = %d matched + %d "
                "taper, schedule-matched ratio per step; audio untouched"
                % (steps, matched_steps, taper_steps))
        print("[MAINodes] " + text)
        return (patched, text)


NODE_CLASS_MAPPINGS = {NODE_NAME: H3DriftControl}
NODE_DISPLAY_NAME_MAPPINGS = {NODE_NAME: "H3 Drift Control (alpha)"}
