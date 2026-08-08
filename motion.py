"""v2v time-smear nodes — the validated fast-motion recipe as knobs.

Pipeline (all defaults = measured best values, 2026-08-08):
  H3JerkOracle      latent -> jerk profile, window, LocalRate segments,
                    per-frame integer hold map (C1 ramps)
  H3TimeSmear       frames + hold map -> smeared frames on a 17k+5 grid
  (VAEEncode)       smeared frames -> video latent
  H3V2VInit         video latent -> nested AV latent for injection
  H3InjectSchedule  model -> truncated SIGMAS (inject 0.70 default)
  (SamplerCustomAdvanced) -> generated latent -> decode
  H3ExactRecover    frames + hold map -> world clock (exact selection)
  H3JerkHeatmap     frames + latent -> oracle overlay + jerk strip (demo tile)
"""
import json

import numpy as np
import torch

LEGAL_STEP = 17  # legal pixel lengths are 17k+5


def _video_component(samples):
    z = samples["samples"]
    if hasattr(z, "is_nested") and z.is_nested:
        z = z.tensors[0]
    return z  # (1, 24, t_lat, h, w)


def _tok_start_frame(t):
    c, i = divmod(t, 5)
    return c * 17 + (0 if i == 0 else 4 * (i - 1) + 1)


def _frame_token(f, t_lat):
    for t in range(t_lat - 1, -1, -1):
        if _tok_start_frame(t) <= f:
            return t
    return 0


def _jerk_profile(z):
    """Phase-normalized per-token |Δ³| profile from a video latent."""
    v = z.detach().float().cpu().numpy()          # (1, 24, T, h, w)
    j = np.abs(np.diff(v, n=3, axis=2)).mean(axis=(0, 1, 3, 4))
    prof = np.pad(j, (1, v.shape[2] - len(j) - 1), mode="edge")
    for ph in range(5):                            # (1,4,4,4,4) grid bias
        m = prof[ph::5].mean()
        if m > 0:
            prof[ph::5] /= m
    return prof                                    # (t_lat,)


def _legal_ceil(n):
    k = max(2, -(-(n - 5) // LEGAL_STEP))
    return LEGAL_STEP * k + 5


class H3JerkOracle:
    """Read the jerk oracle from a final latent. Emits everything downstream
    knobs consume: LocalRate segment string, detected window, and the
    per-frame integer hold map (with C1 ramp shoulders) for H3TimeSmear."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "samples": ("LATENT",),
            "length": ("INT", {"default": 124, "min": 5, "max": 3600, "step": 17}),
            "q": ("FLOAT", {"default": 0.75, "min": 0.5, "max": 0.99, "step": 0.01,
                            "tooltip": "jerk quantile that counts as hot; higher = tighter span"}),
            "d_max": ("INT", {"default": 4, "min": 2, "max": 8,
                              "tooltip": "peak hold count / divisor on the hottest tokens"}),
            "ramp": ("BOOLEAN", {"default": True,
                                 "tooltip": "C1 ramp shoulders (1,2,..,d_max,..,2,1) instead of hard steps"}),
        }}

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT", "STRING")
    RETURN_NAMES = ("hold_map", "segments", "window_start", "window_len", "profile")
    FUNCTION = "read"
    CATEGORY = "latent/minimax/motion"

    def read(self, samples, length, q, d_max, ramp):
        z = _video_component(samples)
        t_lat = z.shape[2]
        prof = _jerk_profile(z)

        thr = np.quantile(prof, q)
        tok_d = np.where(prof >= thr, d_max, 1).astype(int)
        if ramp:
            for _ in range(d_max - 1):            # relax until |Δd| <= 1
                left = np.concatenate([[1], tok_d[:-1]])
                right = np.concatenate([tok_d[1:], [1]])
                tok_d = np.maximum(tok_d, np.maximum(left, right) - 1)

        holds = [int(tok_d[_frame_token(f, t_lat)]) for f in range(length)]

        segs, t0 = [], 0
        for t in range(1, t_lat + 1):
            if t == t_lat or tok_d[t] != tok_d[t0]:
                if tok_d[t0] > 1:
                    segs.append(f"{t0}:{t}:{int(tok_d[t0])}")
                t0 = t
        hot = np.where(tok_d > 1)[0]
        if len(hot):
            w0 = (_tok_start_frame(int(hot.min())) // 17) * 17
            w1 = min(length, _tok_start_frame(min(int(hot.max()) + 1, t_lat - 1)) + 4)
            wlen = _legal_ceil(w1 - w0)
            wlen = min(wlen, length - w0) if w0 + wlen > length else wlen
        else:
            w0, wlen = 0, length

        hold_map = json.dumps({"holds": holds, "world_len": length})
        profile = " ".join(f"{v:.2f}" for v in prof)
        return (hold_map, ",".join(segs), int(w0), int(wlen), profile)


class H3TimeSmear:
    """Retime frames onto a longer uniform grid by integer holds — the
    nonuniform (oracle) or uniform (dilation) smear that seeds v2v
    injection. Output length is snapped up to the 17k+5 grid by extending
    the final hold; the emitted hold_map records exactly what happened so
    H3ExactRecover can invert it losslessly."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "dilation": ("INT", {"default": 4, "min": 1, "max": 8,
                                 "tooltip": "uniform hold count; ignored when hold_map is wired"}),
        }, "optional": {
            "hold_map": ("STRING", {"default": "",
                                    "tooltip": "from H3JerkOracle — per-frame integer holds"}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "hold_map_used")
    FUNCTION = "smear"
    CATEGORY = "image/minimax/motion"

    def smear(self, images, dilation, hold_map=""):
        n = images.shape[0]
        holds = (json.loads(hold_map)["holds"] if hold_map.strip()
                 else [dilation] * n)
        assert len(holds) == n, f"hold map covers {len(holds)} frames, batch has {n}"
        target = _legal_ceil(sum(holds))
        holds = list(holds)
        holds[-1] += target - sum(holds)          # tail pad lives in the last hold
        idx = torch.tensor([i for i, h in enumerate(holds) for _ in range(h)])
        used = json.dumps({"holds": holds, "world_len": n})
        return (images[idx], used)


class H3ExactRecover:
    """Invert H3TimeSmear: keep the first frame of every hold group —
    exact 24fps real-time recovery by frame selection (never resampling)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "hold_map": ("STRING", {"default": ""}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "recover"
    CATEGORY = "image/minimax/motion"

    def recover(self, images, hold_map):
        holds = json.loads(hold_map)["holds"]
        starts, cur = [], 0
        for h in holds:
            starts.append(cur)
            cur += h
        assert cur == images.shape[0], (cur, images.shape[0])
        return (images[torch.tensor(starts)],)


class H3V2VInit:
    """Wrap a VAE-encoded video latent as the nested AV latent that
    SamplerCustomAdvanced expects for H3, ready for partial-denoise
    injection (pair with H3InjectSchedule). Audio starts from zeros and
    regenerates on the truncated schedule (jointly with the video —
    the operator-preferred audio source for regenerated content)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "samples": ("LATENT", {"tooltip": "video latent from VAEEncode of the smeared frames"}),
            "length": ("INT", {"default": 294, "min": 5, "max": 3600, "step": 17,
                               "tooltip": "pixel frame count of the smeared clip"}),
        }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "build"
    CATEGORY = "latent/minimax/motion"

    def build(self, samples, length):
        import comfy.nested_tensor
        from comfy_extras.nodes_minimax_h3 import temporal_shape

        _, t_lat, audio_t = temporal_shape(length)
        video = _video_component(samples)
        assert video.shape[2] == t_lat, (
            f"latent has {video.shape[2]} tokens, length {length} needs {t_lat}")
        audio = torch.zeros(video.shape[0], 32, 2, audio_t,
                            device=video.device, dtype=video.dtype)
        return ({"samples": comfy.nested_tensor.NestedTensor((video, audio))},)


class H3InjectSchedule:
    """Truncated sigma schedule for v2v injection. inject=0.70 (the
    measured sweet spot) keeps the init's coarse choreography and re-rolls
    rendering; lower inherits more artifact risk, higher drifts toward
    free generation (invented-physics regime)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "scheduler": (["simple", "normal", "beta", "sgm_uniform", "karras",
                           "exponential"], {"default": "simple"}),
            "total_steps": ("INT", {"default": 25, "min": 4, "max": 100}),
            "inject": ("FLOAT", {"default": 0.70, "min": 0.05, "max": 1.0,
                                 "step": 0.05}),
        }}

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "sigmas"
    CATEGORY = "sampling/custom_sampling/schedulers"

    def sigmas(self, model, scheduler, total_steps, inject):
        import comfy.samplers
        full = comfy.samplers.calculate_sigmas(
            model.get_model_object("model_sampling"), scheduler, total_steps)
        run = max(1, int(round(total_steps * inject)))
        return (full[total_steps - run:],)


class H3JerkHeatmap:
    """The oracle made visible (demo tile as a node): jerk-heat overlay on
    the frames + a per-token jerk strip with playhead along the bottom."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "samples": ("LATENT",),
            "alpha": ("FLOAT", {"default": 0.55, "min": 0.0, "max": 1.0, "step": 0.05}),
            "strip_height": ("INT", {"default": 96, "min": 0, "max": 256}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "overlay"
    CATEGORY = "image/minimax/motion"

    def overlay(self, images, samples, alpha, strip_height):
        z = _video_component(samples)
        t_lat = z.shape[2]
        n, H, W, _ = images.shape

        v = z.detach().float().cpu().numpy()
        jmap = np.abs(np.diff(v, n=3, axis=2)).mean(axis=(0, 1))   # (T-3, h, w)
        tok = np.stack([jmap[min(max(k - 1, 0), jmap.shape[0] - 1)]
                        for k in range(t_lat)])
        for ph in range(5):
            m = tok[ph::5].mean()
            if m > 0:
                tok[ph::5] /= m
        lo, hi = np.quantile(tok, 0.05), np.quantile(tok, 0.995)
        tok = np.clip((tok - lo) / (hi - lo + 1e-9), 0, 1)
        heat = torch.nn.functional.interpolate(
            torch.from_numpy(tok).float()[None], size=(H, W),
            mode="bilinear", align_corners=False)[0]               # (T, H, W)

        prof = _jerk_profile(z)
        pn = (prof - prof.min()) / (prof.max() - prof.min() + 1e-9)

        out = []
        bar_w = max(W // t_lat, 1)
        for f in range(n):
            k = _frame_token(min(f, 3600), t_lat) if f < 3600 else 0
            k = min(k, t_lat - 1)
            hm = heat[k]
            a = (hm * alpha)[..., None]
            color = torch.stack([torch.ones_like(hm),
                                 0.3 + 0.7 * (1 - hm),
                                 torch.zeros_like(hm)], -1)
            img = images[f] * (1 - a) + color * a
            if strip_height:
                strip = torch.full((strip_height, W, 3), 0.09)
                for t in range(t_lat):
                    bh = int(pn[t] * (strip_height - 14)) + 4
                    x0, x1 = t * bar_w + 1, min((t + 1) * bar_w - 1, W)
                    c = (torch.tensor([1.0, 0.3 + 0.7 * (1 - pn[t]), 0.0]) if t == k
                         else torch.tensor([0.35, 0.35 + 0.45 * (1 - pn[t]), 0.63]))
                    strip[strip_height - bh:, x0:x1] = c
                px = int(f / max(n - 1, 1) * (W - 1))
                strip[:, max(px - 1, 0):px + 1] = 1.0
                img = torch.cat([img, strip.to(img.dtype)], 0)
            out.append(img)
        return (torch.stack(out),)


TIMESMEAR_CLASS_MAPPINGS = {
    "H3JerkOracle": H3JerkOracle,
    "H3TimeSmear": H3TimeSmear,
    "H3ExactRecover": H3ExactRecover,
    "H3V2VInit": H3V2VInit,
    "H3InjectSchedule": H3InjectSchedule,
    "H3JerkHeatmap": H3JerkHeatmap,
}
TIMESMEAR_DISPLAY_MAPPINGS = {
    "H3JerkOracle": "H3 Jerk Oracle (profile / window / hold map)",
    "H3TimeSmear": "H3 Time Smear (integer holds)",
    "H3ExactRecover": "H3 Exact Recover (24fps frame selection)",
    "H3V2VInit": "H3 V2V Init (nested AV latent)",
    "H3InjectSchedule": "H3 Inject Schedule (v2v sigmas, 0.70)",
    "H3JerkHeatmap": "H3 Jerk Heatmap (oracle overlay tile)",
}
