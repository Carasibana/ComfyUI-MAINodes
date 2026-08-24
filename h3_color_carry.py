"""H3 Delta Color Carry (alpha): cancel VAE round-trip color bias on a prefix.

Every carried handle is a VAE round-trip, and each round-trip darkens ~2.4%
(measured 2026-08-23); chains accumulate it. Seam Normalize fits gains AFTER
the fact; this module fixes the prefix BEFORE sampling, and does it without
replacing the sampled latent: decode the prefix once, apply a weak scene-one
exposure/saturation correction in RGB, encode BOTH the original and corrected
frames, and add only E(corrected) - E(original) to the latent. The encode
bias appears in both terms and cancels; only the intended grade survives.
The delta is spatially low-passed and tapered from zero at the old edge to
full strength beside the generated future. Audio is never touched.

ADAPTED from ComfyUI-MiniMaxH3-Contex-Loop's latent_color_carry.py
(NikoDemon80 / ethanfel, GPL-3.0 - same license as this pack), reshaped for
our masked-prefix graphs: the correction applies in place to the first
prefix_frames of the TARGET latent (the extend graph's VAEEncode output)
instead of a standalone 12-step prefix latent, and the step count derives
from prefix_frames rather than being pinned to the recipe.

Wire: H3 Tail Context images (scene one) -> H3 Scene Color Stats -> anchor;
current predecessor tail -> another Stats -> current; target latent from the
extend graph's VAEEncode -> here -> H3 V2V Init. For link one, anchor and
current are the same tail and the transform is identity (early exit).
"""

import math
import statistics

import torch
import torch.nn.functional as functional

STATS_VERSION = "h3_latent_color_stats_v1"   # same schema as the upstream
DEFAULTS = {
    "strength": 0.50,
    "max_luma_shift_code_values": 6.0,
    "max_saturation_change": 0.06,
    "spatial_lowpass_kernel": 3,
}


# ------------------------------------------------------------- pure functions

def tensor_scene_color_stats(frames):
    """Robust center-weighted luma/saturation percentiles from IMAGE [N,H,W,C]."""
    if not torch.is_tensor(frames) or frames.ndim != 4:
        raise ValueError("color carry expected IMAGE [frames,H,W,C], got %s"
                         % (tuple(getattr(frames, "shape", ())),))
    if int(frames.shape[0]) < 1 or int(frames.shape[-1]) < 3:
        raise ValueError("color carry received an empty RGB image")
    count, height, width = (int(frames.shape[i]) for i in range(3))
    frame_step = max(1, count // 24)
    y0, y1 = int(height * 0.12), max(int(height * 0.92), 1)
    x0, x1 = int(width * 0.20), max(int(width * 0.80), 1)
    sample = frames[::frame_step, y0:y1:4, x0:x1:4, :3]
    if not int(sample.numel()):
        sample = frames[::frame_step, ::4, ::4, :3]
    rgb = torch.clamp(sample.detach().float(), 0.0, 1.0).mul_(255.0)
    luma = rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722
    mx = torch.amax(rgb, dim=-1)
    mn = torch.amin(rgb, dim=-1)
    sat = torch.where(mx > 0.0, (mx - mn) / torch.clamp(mx, min=1.0) * 255.0,
                      torch.zeros_like(mx))
    valid = (luma >= 20.0) & (luma <= 235.0)
    if int(torch.count_nonzero(valid).item()) >= 64:
        luma, sat = luma[valid], sat[valid]
    lq = torch.quantile(luma.reshape(-1),
                        torch.tensor((0.10, 0.50, 0.90), dtype=torch.float32))
    sq = torch.quantile(sat.reshape(-1),
                        torch.tensor((0.25, 0.50, 0.75), dtype=torch.float32))
    return {"version": STATS_VERSION,
            "luma_percentiles": [float(v) for v in lq.cpu().tolist()],
            "saturation_percentiles": [float(v) for v in sq.cpu().tolist()],
            "sampled_frames": len(range(0, count, frame_step))}


def _validated_stats(value, usage):
    if not isinstance(value, dict) or value.get("version") != STATS_VERSION:
        raise ValueError("%s has no compatible scene-color statistics" % usage)
    luma = tuple(float(v) for v in value.get("luma_percentiles", ()))
    sat = tuple(float(v) for v in value.get("saturation_percentiles", ()))
    if len(luma) != 3 or len(sat) != 3:
        raise ValueError("%s scene-color statistics are malformed" % usage)
    if not all(math.isfinite(v) for v in (*luma, *sat)):
        raise ValueError("%s scene-color statistics are not finite" % usage)
    return {"luma_percentiles": luma, "saturation_percentiles": sat}


def _coherent_delta(values, minimum):
    """Median of same-signed deltas; 0 unless at least two agree in sign."""
    pos = tuple(v for v in values if v > 0.0)
    neg = tuple(v for v in values if v < 0.0)
    sel = pos if len(pos) >= 2 else (neg if len(neg) >= 2 else ())
    if not sel:
        return 0.0
    v = float(statistics.median(sel))
    return v if abs(v) >= float(minimum) else 0.0


def scene_color_transform(anchor, current, strength=DEFAULTS["strength"],
                          max_luma=DEFAULTS["max_luma_shift_code_values"],
                          max_sat=DEFAULTS["max_saturation_change"]):
    """Weak normalized brightness offset + saturation multiplier, clamped."""
    target = _validated_stats(anchor, "color carry anchor")
    source = _validated_stats(current, "color carry source")
    luma_shift = _coherent_delta(tuple(
        t - s for t, s in zip(target["luma_percentiles"],
                              source["luma_percentiles"])), 1.0)
    luma_shift = max(-max_luma, min(max_luma, luma_shift * float(strength)))
    ratios = tuple(t / max(s, 1.0) - 1.0 for t, s in
                   zip(target["saturation_percentiles"],
                       source["saturation_percentiles"]))
    sat_delta = _coherent_delta(ratios, 0.02) * float(strength)
    sat_delta = max(-max_sat, min(max_sat, sat_delta))
    return luma_shift / 255.0, 1.0 + sat_delta


def apply_rgb_color_transform(frames, brightness, saturation):
    """Rec.709 luma-preserving saturation + exposure on IMAGE [N,H,W,C]."""
    if not torch.is_tensor(frames) or frames.ndim != 4:
        raise ValueError("color carry VAE returned invalid IMAGE")
    result = frames.detach().contiguous().clone()
    rgb = torch.clamp(result[..., :3].float(), 0.0, 1.0)
    luma = (rgb[..., 0:1] * 0.2126 + rgb[..., 1:2] * 0.7152 +
            rgb[..., 2:3] * 0.0722)
    corrected = luma + (rgb - luma) * float(saturation)
    corrected = torch.clamp(corrected + float(brightness), 0.0, 1.0)
    result[..., :3] = corrected.to(device=result.device, dtype=result.dtype)
    return result


def temporal_delta_weights(steps):
    """Zero-to-one smoothstep over the prefix's latent time."""
    count = int(steps)
    if count < 2:
        raise ValueError("color carry needs at least two video steps")
    t = torch.linspace(0.0, 1.0, count, dtype=torch.float32)
    return t.square().mul_(3.0 - 2.0 * t)


def spatial_lowpass(delta, kernel_size):
    kernel = int(kernel_size)
    if kernel <= 1:
        return delta
    if kernel % 2 != 1:
        raise ValueError("color carry spatial kernel must be odd")
    radius = kernel // 2
    padded = functional.pad(delta.float(),
                            (radius, radius, radius, radius, 0, 0),
                            mode="replicate")
    return functional.avg_pool3d(padded, kernel_size=(1, kernel, kernel),
                                 stride=1).to(device=delta.device,
                                              dtype=delta.dtype)


def frames_to_steps(frames):
    f = int(frames)
    if f < 5 or (f - 5) % 17:
        raise ValueError("color carry: prefix_frames must sit on the 17k+5 "
                         "grid (39, 56, 90, ...); got %d" % f)
    return 5 * ((f - 5) // 17) + 2


def correct_prefix_in_place(target_video, prefix_steps, delta, kernel):
    """Add the low-passed, tapered delta to the target's prefix region."""
    steps = int(prefix_steps)
    if not torch.is_tensor(target_video) or target_video.ndim != 5:
        raise ValueError("color carry expected video latent [B,C,T,H,W]")
    if steps < 2 or steps > int(target_video.shape[2]):
        raise ValueError("color carry prefix outside the target latent")
    if tuple(delta.shape[2:3]) != (steps,):
        raise ValueError("color carry delta step count mismatch")
    delta = spatial_lowpass(delta, kernel)
    weights = temporal_delta_weights(steps).to(device=target_video.device,
                                               dtype=target_video.dtype)
    out = target_video.detach().contiguous().clone()
    out[:, :, :steps] += delta.to(device=out.device, dtype=out.dtype) \
        * weights.view(1, 1, steps, 1, 1)
    return out


# -------------------------------------------------------------------- nodes

import json as _json


class H3SceneColorStats:
    DESCRIPTION = ("EXPERIMENTAL (alpha). Robust scene color statistics "
                   "(center-weighted luma/saturation percentiles) from a "
                   "frame batch, as a JSON string for H3 Delta Color Carry. "
                   "Feed it a delivered tail (H3 Tail Context images).")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",)}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("stats",)
    FUNCTION = "measure"
    CATEGORY = "MAINodes/alpha"

    def measure(self, images):
        return (_json.dumps(tensor_scene_color_stats(images)),)


class H3DeltaColorCarry:
    DESCRIPTION = (
        "EXPERIMENTAL (alpha). Cancels the VAE round-trip color bias on a "
        "carried prefix by adding only E(corrected)-E(original) to the "
        "target latent's prefix region - the encode bias cancels, the weak "
        "scene-one grade survives. Spatially low-passed, tapered zero at "
        "the old edge to full beside the generated future; audio untouched. "
        "anchor_stats = scene one's delivered tail, current_stats = the "
        "current predecessor's tail; identical stats = identity (early "
        "exit, latent passes through untouched). Adapted from Contex-Loop's "
        "Color-Stable Drift AV (GPL-3.0). Wire between the extend graph's "
        "VAEEncode and H3 V2V Init.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "samples": ("LATENT", {"tooltip": "the target video latent (extend graph's VAEEncode output)"}),
            "vae": ("VAE",),
            "anchor_stats": ("STRING", {"forceInput": True, "tooltip": "H3 Scene Color Stats of scene ONE's delivered tail"}),
            "current_stats": ("STRING", {"forceInput": True, "tooltip": "H3 Scene Color Stats of the CURRENT predecessor's tail"}),
        }, "optional": {
            "prefix_frames": ("INT", {"default": 39, "min": 22, "max": 3600,
                "tooltip": "carried handle in pixel frames on the 17k+5 grid"}),
            "strength": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.05}),
            "max_luma_shift": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 32.0,
                "tooltip": "clamp, in 8-bit code values"}),
            "max_saturation_change": ("FLOAT", {"default": 0.06, "min": 0.0, "max": 0.5}),
            "spatial_kernel": ("INT", {"default": 3, "min": 1, "max": 15,
                "tooltip": "odd; low-pass on the latent delta"}),
        }}

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("samples", "report")
    FUNCTION = "carry"
    CATEGORY = "MAINodes/alpha"

    def carry(self, samples, vae, anchor_stats, current_stats,
              prefix_frames=39, strength=0.50, max_luma_shift=6.0,
              max_saturation_change=0.06, spatial_kernel=3):
        video = samples["samples"]
        if hasattr(video, "is_nested") and video.is_nested:
            raise ValueError("color carry runs on the VIDEO latent before "
                             "H3 V2V Init packs audio - wire it earlier")
        steps = frames_to_steps(prefix_frames)
        anchor = _json.loads(anchor_stats)
        current = _json.loads(current_stats)
        brightness, saturation = scene_color_transform(
            anchor, current, strength, max_luma_shift, max_saturation_change)
        if abs(brightness) < (0.5 / 255.0) and abs(saturation - 1.0) < 0.005:
            text = "H3 Delta Color Carry: transform within deadband, latent untouched"
            print("[MAINodes] " + text)
            return (samples, text)

        prefix = video[:, :, :steps]
        decoded = vae.decode(prefix)
        if decoded.ndim == 5:
            decoded = decoded.reshape(-1, decoded.shape[-3],
                                      decoded.shape[-2], decoded.shape[-1])
        baseline = vae.encode(decoded)
        corrected_rgb = apply_rgb_color_transform(decoded, brightness, saturation)
        corrected = vae.encode(corrected_rgb)
        if hasattr(baseline, "is_nested"):
            raise ValueError("color carry VAE returned a nested latent")
        delta = corrected.to(prefix) - baseline.to(prefix)
        out_video = correct_prefix_in_place(video, steps, delta, spatial_kernel)
        out = dict(samples)
        out["samples"] = out_video
        text = ("H3 Delta Color Carry: brightness %+.4f, saturation x%.3f "
                "over %d steps (delta rms %.5f), taper 0->1, audio untouched"
                % (brightness, saturation, steps,
                   float(delta.float().square().mean().sqrt())))
        print("[MAINodes] " + text)
        return (out, text)


NODE_CLASS_MAPPINGS = {
    "H3SceneColorStats": H3SceneColorStats,
    "H3DeltaColorCarry": H3DeltaColorCarry,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3SceneColorStats": "H3 Scene Color Stats (alpha)",
    "H3DeltaColorCarry": "H3 Delta Color Carry (alpha)",
}
