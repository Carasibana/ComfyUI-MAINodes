"""v2v time-smear nodes — the validated fast-motion recipe as knobs.

Pipeline (all defaults = measured best values, 2026-08-08):
  H3VideoFit        (source-video path only) arbitrary frames -> the 17k+5
                    grid, audio cut to match, length for the oracle
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

# Per-step cost does NOT scale linearly with token count: attention dominates,
# so time goes as tokens**COST_EXP. Measured 2026-08-09, one clip one card,
# 37 -> 92 tokens took 13.35 -> 65.85 s/step (4.93x for a 2.49x token ratio,
# exponent 1.75). An independent field report on other hardware (16 -> 72
# s/step) lands near 1.64. 1.7 splits them. Expect it to climb toward 2.0 at
# higher resolution, where attention takes a larger share of the step.
COST_EXP = 1.7


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


def _phase_norm(prof):
    """Divide out the (1,4,4,4,4) grid bias so tokens are comparable."""
    for ph in range(5):
        m = prof[ph::5].mean()
        if m > 0:
            prof[ph::5] /= m
    return prof


def _value_profile(v, order):
    """Per-token |Δ^order| of the latent VALUES, averaged over c/h/w."""
    j = np.abs(np.diff(v, n=order, axis=2)).mean(axis=(0, 1, 3, 4))
    lead = order // 2
    return np.pad(j, (lead, v.shape[2] - len(j) - lead), mode="edge")


def _trajectory_profile(v, order=3):
    """Per-token |Δ^order| of the energy CENTROID's path.

    The value-domain score is contaminated by motion energy: a textured
    object passing a location makes the value there pulse, and a pulse has
    large differences of every order even at constant velocity (measured
    corr(|d1|,|d3|) = 0.96-0.98 on real clips). Differentiating the centroid
    TRAJECTORY instead measures how abruptly the motion changes, which is
    what the method actually cares about, and on real textured clips it
    separated jerk from velocity where the value domain did not (top-decile
    share 0.29 vs 0.22, r = 0.46).

    Caveat measured 2026-08-10: on a SMOOTH synthetic blob the centroid path
    is nearly noise-free, so peak-to-mean contrast there is not comparable to
    the value domain's (1.21x vs 1.88x separation on such a toy). This mode
    is provided for ablation; it is not established as the better detector.
    """
    e = np.abs(v).mean(axis=(0, 1))                # (T, h, w) energy per token
    e = e - e.min(axis=(1, 2), keepdims=True)
    tot = e.sum(axis=(1, 2)) + 1e-8
    ys = np.arange(e.shape[1], dtype=np.float64)[None, :, None]
    xs = np.arange(e.shape[2], dtype=np.float64)[None, None, :]
    cy = (e * ys).sum(axis=(1, 2)) / tot
    cx = (e * xs).sum(axis=(1, 2)) / tot
    path = np.stack([cy, cx], axis=1)              # (T, 2)
    j = np.linalg.norm(np.diff(path, n=order, axis=0), axis=1)
    lead = order // 2
    return np.pad(j, (lead, path.shape[0] - len(j) - lead), mode="edge")


PROFILE_MODES = {
    "value |d3| (default)": ("value", 3),
    "value |d1| (energy baseline)": ("value", 1),
    "trajectory centroid |d3|": ("traj", 3),
}


def _jerk_profile(z, mode="value |d3| (default)", phase_norm=True):
    """Per-token motion-overload profile from a video latent.

    Default reproduces the original phase-normalized |Δ³| exactly. The other
    modes exist so the detector can be ablated against a cheap baseline
    rather than assumed: see ROADMAP.md section 1.
    """
    v = z.detach().float().cpu().numpy()          # (1, 24, T, h, w)
    kind, order = PROFILE_MODES.get(mode, ("value", 3))
    prof = _value_profile(v, order) if kind == "value" else _trajectory_profile(v, order)
    return _phase_norm(prof) if phase_norm else prof   # (t_lat,)


def _legal_ceil(n):
    k = max(2, -(-(n - 5) // LEGAL_STEP))
    return LEGAL_STEP * k + 5


# The two exact neighbours on the 17k+5 grid. _legal_ceil above clamps to
# k>=2 (never below 39) because a smear target that short is not worth
# generating; a FIT must not invent 27 frames out of a 12-frame clip, so it
# uses these instead.
def _grid_floor(n):
    return 5 + LEGAL_STEP * max(0, (n - 5) // LEGAL_STEP)


def _grid_ceil(n):
    return 5 + LEGAL_STEP * max(0, -(-(n - 5) // LEGAL_STEP))


def _token_count(frames):
    """Latent tokens for a pixel length, snapped up to the 17k+5 grid."""
    return (_legal_ceil(frames) - 5) // LEGAL_STEP * 5 + 2


# Fixed per-render cost that is NOT sampling: model/LoRA setup, VAE encode and
# decode, node overhead. Measured 2026-08-12, 1.5 MP pass 2 on GPU1, warm:
# 124.8 s at 1 step, 209.9 / 212.8 at 2, 293.6 at 3 — a straight line of
# ~84 s per step on ~41 s of intercept. Without this term a 1-step estimate
# is 33% low.
#
# That 41 s was measured on an already-dilated pass, so it is NOT a constant:
# divided by that run's time multiplier it is ~6.7 s, and a separate fit over
# 266 corpus runs found the step-independent cost scaling at t**1.66 against
# the step's t**1.70. So overhead is scaled by time_x below, the same as the
# steps are. The two measurements agree once the dilation is divided out
# (~0.5 of one step for pass 2; the corpus fit reads ~1.1 for turbo pass 1,
# which is a different workload, not a contradiction).
#
# The whole minutes estimate is a ballpark. It is here so nobody discovers
# a 30 minute pass by waiting through it, not to be quotable.
OVERHEAD_S = 6.7


def _cost_report(world_len, dilated, fps=24, s_per_step=0.0, est_steps=18,
                 overhead_s=OVERHEAD_S, tail=""):
    """The price tag every targeting node shows before the expensive pass.

    Same sentence everywhere: world length in, effective regeneration length
    out, then the TIME multiplier. The frame ratio is stated too but labelled,
    because reading it as a time ratio understates the bill badly: per-step
    cost goes as tokens**COST_EXP, so 2.5x the frames is about 4.9x the time.
    """
    world_len = max(1, int(world_len))
    dilated = max(world_len, int(dilated))
    fps = max(1, int(fps))
    t_world = _token_count(world_len)
    t_dil = _token_count(dilated)
    time_x = (t_dil / t_world) ** COST_EXP
    report = (f"{world_len}f ({world_len / fps:.1f}s) -> {dilated}f "
              f"({dilated / fps:.1f}s) effective regen, "
              f"{dilated / world_len:.2f}x frames / {time_x:.1f}x time per "
              f"step; tokens {t_world} -> {t_dil}")
    if tail:
        report += f"; {tail}"
    if s_per_step > 0:
        secs = time_x * (overhead_s + s_per_step * max(1, int(est_steps)))
        report += (f"; roughly {secs / 60:.0f} min at {s_per_step:g} s/step x "
                   f"{int(est_steps)} steps (+{overhead_s:g}s encode/decode, "
                   f"all x{time_x:.1f})")
    return report


def _cost_widgets(with_fps=False):
    """Optional widgets that turn the report's multiplier into minutes.
    Shared so every node that prices a pass asks for the same three numbers."""
    w = {}
    if with_fps:
        w["fps"] = ("INT", {"default": 24, "min": 1, "max": 120,
                    "tooltip": "only used to phrase the report in seconds"})
    w["s_per_step"] = ("FLOAT", {"default": 0.0, "min": 0.0, "max": 120.0, "step": 0.05,
                       "tooltip": "seconds per step from a baseline render of this clip; 0 skips the minutes estimate"})
    w["est_steps"] = ("INT", {"default": 18, "min": 1, "max": 100,
                      "tooltip": "steps the regen pass will actually run (total_steps x inject)"})
    w["overhead_s"] = ("FLOAT", {"default": OVERHEAD_S, "min": 0.0, "max": 600.0, "step": 1.0,
                       "tooltip": "fixed non-sampling seconds per render (setup, VAE encode/decode). "
                                  "40 measured at 1.5 MP on a warm instance; take it from the gap "
                                  "between your own 1-step and 2-step wall times"})
    return w


# --- window / segment geometry -------------------------------------------
# A frame index is a token START iff it is one of these offsets inside a
# 17-frame chunk: _tok_start_frame(5c+i) = 17c + (0, 1, 5, 9, 13)[i].
TOK_OFFSETS = (0, 1, 5, 9, 13)


def _is_tok_start(f):
    return f % LEGAL_STEP in TOK_OFFSETS


def _seg_contrib(holds, f, c0, c1, hot_lo, hot_hi):
    """Smeared-frame contribution of world frame f inside a window whose
    regenerated core is [c0, c1]. Handle frames are hold 1 at a COLD cut
    (real-time baseline context, the shipped behaviour), but inherit the
    world hold at a HOT cut so both sides of the seam are repaired at the
    same temporal rate (seam policy 4)."""
    if c0 <= f <= c1:
        return int(holds[f])
    hot = hot_lo if f < c0 else hot_hi
    return int(holds[f]) if hot else 1


def _seg_holds(holds, a, b, c0, c1, hot_lo=False, hot_hi=False):
    return [_seg_contrib(holds, f, c0, c1, hot_lo, hot_hi)
            for f in range(a, b + 1)]


def _grid_grow(holds, a, b, c0, c1, n, hot_lo=False, hot_hi=False):
    """Widen [a, b] outward until sum(seg_holds) sits exactly on the 17k+5
    grid, and return (a, b, residual).

    Why: H3TimeSmear lands on the grid with `holds[-1] += target - sum`,
    which parks a multi-frame freeze on the window's last frame. Handle
    frames cost exactly 1 smeared frame each, so growing a handle by the
    deficit reaches the same grid point with no freeze at all. residual is
    what is still left for H3TimeSmear to absorb (0 unless the clip runs
    out of frames on both sides).
    """
    def contrib(f):
        return _seg_contrib(holds, f, c0, c1, hot_lo, hot_hi)

    s = sum(contrib(f) for f in range(a, b + 1))
    need = _legal_ceil(s) - s
    # each step subtracts exactly the frame's own contribution and never
    # overshoots, so _legal_ceil(sum) is invariant through the loop
    while need > 0 and b < n - 1 and contrib(b + 1) <= need:
        b += 1
        need -= contrib(b)
    while need > 0 and a > 0 and contrib(a - 1) <= need:
        a -= 1
        need -= contrib(a)
    return a, b, need


def _soft_edge(mask, feather, profile="linear", direction="centered"):
    """Feather a (T, H, W) 0/1 mask. Separable box or gaussian blur;
    direction pre-shifts the boundary so the ramp eats into the masked
    side (inward) or the kept side (outward) instead of straddling it."""
    import torch.nn.functional as F
    if feather <= 0:
        return mask
    m = mask[:, None]
    s = feather // 2
    if s and direction != "centered":
        k = s * 2 + 1
        if direction == "inward":
            m = 1 - F.max_pool2d(1 - m, k, stride=1, padding=k // 2)
        else:  # outward
            m = F.max_pool2d(m, k, stride=1, padding=k // 2)
    k = feather // 2 * 2 + 1
    if profile == "gaussian":
        x = torch.arange(k, dtype=torch.float32) - k // 2
        w = torch.exp(-(x ** 2) / (2 * (max(feather, 1) / 4.0) ** 2))
    else:
        w = torch.ones(k)
    w = w / w.sum()
    # replicate padding: masks that bleed off the image edge (inverted
    # background lassos) must not erode at the border
    m = F.pad(m, (k // 2, k // 2, k // 2, k // 2), mode="replicate")
    m = F.conv2d(m, w.view(1, 1, k, 1))
    m = F.conv2d(m, w.view(1, 1, 1, k))
    m = m[:, 0].clamp(0, 1)
    if profile == "smoothstep":
        m = m * m * (3 - 2 * m)
    return m


class H3VideoFit:
    """Snap an arbitrary clip's frame count onto H3's 17k+5 grid before it
    enters the pipeline: trim or pad the frames, cut the audio to match,
    and emit the resulting length so nothing downstream has to be told it
    by hand."""

    DESCRIPTION = (
        "EXPERIMENTAL (alpha), new 2026-08-12; the classic pipeline nodes are unchanged.\n\n"
        "The doorway for footage you already have. H3 works in 17-frame "
        "chunks, so legal clip lengths are 5, 22, 39, 56 ... 17k+5; an "
        "arbitrary MP4 lands between them. This trims (or pads) the batch to "
        "the nearest legal count, cuts the source audio by the same amount, "
        "and outputs 'length' — wire it into H3 Jerk Oracle's length instead "
        "of typing a number or wiring the duration expression the generated "
        "examples use.\n\n"
        "Why it matters even though nothing errors: the H3 VAE encoder pads a "
        "short final chunk by REPEATING THE LAST FRAME, silently. A 312-frame "
        "clip is encoded as 323 frames, the last 11 of them frozen, so the "
        "final tokens read as calm and the oracle under-dilates the end of "
        "your clip. Trimming to 311 removes the invented stillness.\n\n"
        "Default 'trim tail': never invents content, cuts at most 16 frames "
        "(0.67s at 24fps), and keeps frame 0, which is the anchor every "
        "keyframe/FLF path and the audio clock reference. 'trim head' if the "
        "action you care about is at the end. 'pad tail' repeats the last "
        "frame up to the next legal length (and pads the audio with silence) "
        "— it makes the encoder's hidden padding explicit and visible rather "
        "than removing it, so prefer it only when you cannot lose the tail.\n\n"
        "max_frames is the cost lever: per-step time is superlinear in token "
        "count, so capping a long source before fitting is the cheapest knob "
        "in the graph. 0 = use the whole clip.\n\n"
        "The report output is the receipt: frames in, frames out, which end "
        "lost them, the token count, and a warning if the source is not 24fps "
        "(H3 has one frame rate; a 30fps source will play back 1.25x slower).")

    MODES = ["trim tail (default)", "trim head",
             "pad tail (freeze last frame)", "nearest (trim or pad)"]

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE", {"tooltip": "source frames, e.g. LoadVideo -> GetVideoComponents"}),
            "mode": (cls.MODES, {"default": "trim tail (default)",
                     "tooltip": "how to reach the nearest 17k+5 length; trimming never invents frames"}),
        }, "optional": {
            "max_frames": ("INT", {"default": 0, "min": 0, "max": 3600,
                           "tooltip": "cap the clip before fitting (drops the tail); 0 = whole clip. "
                                      "The cheapest cost knob there is: time per step goes as tokens**1.7"}),
            "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.01,
                    "tooltip": "source frame rate, for the audio cut and the 24fps warning; "
                               "wire GetVideoComponents' fps output"}),
            "audio": ("AUDIO", {"tooltip": "the source's own track; cut to the same span. "
                                "Unwired -> the audio output carries nothing"}),
        }}

    RETURN_TYPES = ("IMAGE", "INT", "AUDIO", "STRING")
    RETURN_NAMES = ("images", "length", "audio", "report")
    FUNCTION = "fit"
    CATEGORY = "image/minimax/motion"

    def fit(self, images, mode, max_frames=0, fps=24.0, audio=None):
        images = images.detach().cpu()
        src = int(images.shape[0])
        assert src > 0, "no frames on the images input"
        n = min(src, int(max_frames)) if max_frames else src
        capped = n < src

        lo, hi = _grid_floor(n), _grid_ceil(n)
        if n < 5:                                  # cannot trim below the grid
            out_n, how = 5, "pad"
        elif n == lo:
            out_n, how = n, "none"
        elif mode.startswith("pad"):
            out_n, how = hi, "pad"
        elif mode.startswith("nearest"):
            out_n = lo if (n - lo) <= (hi - n) else hi
            how = "trim" if out_n < n else "pad"
        else:
            out_n, how = lo, "trim"

        head = mode == "trim head" and how == "trim"
        start = n - out_n if head else 0           # first kept source frame
        clip = images[:n]
        if how == "trim":
            out = clip[start:start + out_n]
        elif how == "pad":
            pad = out_n - n
            out = torch.cat([clip, clip[-1:].repeat(pad, 1, 1, 1)], dim=0)
        else:
            out = clip

        fps = float(fps) or 24.0
        cut = None
        if audio is not None:
            wav = audio["waveform"].detach().float().cpu()   # [B, C, N]
            sr = int(audio["sample_rate"])
            spf = sr / fps
            a = int(round(start * spf))
            b = int(round((start + out_n) * spf))
            seg = wav[..., a:min(b, wav.shape[-1])]
            if seg.shape[-1] < b - a:                        # pad tail = silence
                seg = torch.nn.functional.pad(seg, (0, b - a - seg.shape[-1]))
            cut = {"waveform": seg.contiguous(), "sample_rate": sr}

        parts = [f"{src} frames in, {out_n} out"]
        if capped:
            parts.append(f"capped at max_frames={max_frames}, {src - n} off the tail")
        if how == "trim":
            parts.append(f"dropped {n - out_n} from the "
                         f"{'head' if head else 'tail'}")
        elif how == "pad":
            parts.append(f"padded {out_n - n} by repeating the last frame "
                         f"(audio padded with silence)")
        else:
            parts.append("already on the 17k+5 grid, passed through")
        # tokens straight from the grid index, not _token_count: that helper
        # goes through _legal_ceil's k>=2 clamp and would price a legal
        # 5-frame clip at 39 frames / 12 tokens.
        k = (out_n - 5) // LEGAL_STEP
        report = ("; ".join(parts) +
                  f". {out_n} = 17x{k}+5, {5 * k + 2} tokens, "
                  f"{out_n / fps:.2f}s at {fps:g} fps")
        if cut is not None:
            report += f"; audio {cut['sample_rate']} Hz cut to match"
        if how == "trim" and (n - out_n) > 0.25 * n:
            # only fires under ~64 frames, where one grid step is a big share
            # of the clip. Flag it, do not override the user's mode.
            report += (f". WARNING: the trim dropped {(n - out_n) / n:.0%} of "
                       f"the clip; 'pad tail' would have kept all {n} frames")
        if abs(fps - 24.0) > 0.1:
            report += (f". WARNING: H3 is a 24fps model and these frames go in "
                       f"as 24fps, so the result plays {fps / 24.0:.2f}x "
                       f"slower than the source")
        return (out, int(out_n), cut, report)


class H3JerkOracle:
    """Read the jerk oracle from a final latent. Emits everything downstream
    knobs consume: LocalRate segment string, detected window, and the
    per-frame integer hold map (with C1 ramp shoulders) for H3TimeSmear."""

    DESCRIPTION = (
        "Reads WHERE and WHEN a clip's motion is too fast for the model from "
        "the clip's own latent (per-token jerk, |Δ³| over time). Outputs: "
        "hold_map → wire into H3 Time Smear for ADAPTIVE dilation; segments → "
        "H3 Local Rate; window/profile for inspection.\n\n"
        "Knobs: q = jerk quantile that counts as 'hot' (default 0.75; raise "
        "toward 0.85 for tighter spans and lower cost, lower toward 0.7 to "
        "catch more of the burst). d_max = peak hold count (default 4; the "
        "measured sweet spot — 2-3 saves time but starts to rope again). "
        "ramp = C1 shoulders on the hold curve (keep ON; hard steps jitter).\n\n"
        "ADAPTIVE MODE NOTE: the oracle hold map dilates only the hot spans, "
        "which saves significant render time (~2.4-3x total budget instead of "
        "uniform 4x) and follows the clip's intended pacing/attention more "
        "closely — quiet spans keep their native beat contrast. Trade-off: it "
        "can artifact slightly more than uniform dilation if the hold plateau "
        "dips inside a burst — the bridge knob (default 8) closes such valleys automatically per our measured production rule; if you still see hiccups mid-burst, lower q or raise "
        "d_max so the whole burst sits at the plateau.\n\n"
        "EFFECTIVE SIZE: this node decides how long the regeneration pass "
        "really is, and the report output states it before you pay — world "
        "length in, effective regen length out, then the multipliers. Expect "
        "a 5 s action clip to run as 11 to 13 s of frame data, so on a card "
        "that fits 10 s you can de-rope about a third of that. The frame and "
        "TIME multipliers are not the same number: per-step cost goes as "
        "tokens^1.7, so 2.5x the frames is roughly 4.9x the time per step. "
        "q and d_max are the two knobs that move it. Set s_per_step from a "
        "baseline render of this clip on this card for a minutes estimate.")

    PRESETS = {
        "balanced (default)": {"q": 0.75, "d_max": 4, "ramp": True},
        "max quality (wide plateau)": {"q": 0.70, "d_max": 4, "ramp": True},
        "economy (tight spans)": {"q": 0.85, "d_max": 3, "ramp": True},
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "samples": ("LATENT",),
            "length": ("INT", {"default": 124, "min": 5, "max": 3600, "step": 17}),
            "q": ("FLOAT", {"default": 0.75, "min": 0.5, "max": 0.99, "step": 0.01,
                            "tooltip": "jerk quantile that counts as hot; higher = tighter span, lower cost"}),
            "d_max": ("INT", {"default": 4, "min": 2, "max": 8,
                              "tooltip": "peak hold count on the hottest tokens; 4 = measured sweet spot"}),
            "ramp": ("BOOLEAN", {"default": True,
                                 "tooltip": "C1 ramp shoulders (1,2,..,d_max,..,2,1) instead of hard steps — keep ON"}),
        }, "optional": {
            "preset": (["custom"] + list(cls.PRESETS), {"default": "balanced (default)",
                       "tooltip": "any choice but 'custom' overrides the knobs above"}),
            "bridge": ("INT", {"default": 8, "min": 0, "max": 20,
                       "tooltip": "bridge inter-peak valleys within a burst at d_max "
                                  "(measured production rule: a plateau dip between peaks "
                                  "of the same burst causes mid-burst artifacts). Max gap "
                                  "in tokens to fill; 0 disables."}),
            "profile_mode": (list(PROFILE_MODES), {"default": "value |d3| (default)",
                       "tooltip": "(alpha) which signal to threshold. The default is the "
                                  "shipped one. |d1| is the honest cheap baseline (on real "
                                  "clips it correlates 0.96-0.98 with |d3|, so the default "
                                  "is closer to motion ENERGY than to jerk). 'trajectory' "
                                  "differentiates the energy CENTROID's path instead, which "
                                  "is closer to the physical quantity; on real textured clips "
                                  "it gave a narrower profile than velocity. EXPERIMENTAL: on "
                                  "SMOOTH synthetic content the centroid is nearly noise-free, "
                                  "so its contrast is not comparable to the value domain's "
                                  "there. Ablate it, do not assume it."}),
            "abstain_below": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1,
                       "tooltip": "(alpha) ABSOLUTE gate, 0 = off (shipped behaviour). "
                                  "q is a quantile, so the oracle always dilates the top "
                                  "(1-q) of tokens even on a clip that needs nothing: a "
                                  "synthetic clip with EXACTLY zero trajectory jerk still "
                                  "got 3.06x. If the profile's peak-to-mean contrast is "
                                  "below this, the oracle abstains and returns a flat "
                                  "hold map (no dilation, no cost). Measured contrast ran "
                                  "about 2x higher on a jerky clip than a smooth one, so "
                                  "try 1.5-2.5 and check against a clip you know is calm."}),
            **_cost_widgets(with_fps=True),
        }}

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT", "STRING", "STRING")
    RETURN_NAMES = ("hold_map", "segments", "window_start", "window_len",
                    "profile", "report")
    FUNCTION = "read"
    CATEGORY = "latent/minimax/motion"

    def read(self, samples, length, q, d_max, ramp, preset="custom", bridge=8,
             profile_mode="value |d3| (default)", abstain_below=0.0,
             fps=24, s_per_step=0.0, est_steps=18, overhead_s=OVERHEAD_S):
        if preset in self.PRESETS:
            p = self.PRESETS[preset]
            q, d_max, ramp = p["q"], p["d_max"], p["ramp"]
        z = _video_component(samples)
        t_lat = z.shape[2]
        prof = _jerk_profile(z, profile_mode)

        # Absolute gate. The quantile can rank but cannot abstain, so without
        # this a calm clip still pays for its own fastest quarter.
        contrast = float(prof.max() / max(prof.mean(), 1e-8))
        if abstain_below > 0.0 and contrast < abstain_below:
            flat = json.dumps({"holds": [1] * length, "world_len": length})
            return (flat, "", 0, int(length),
                    " ".join(f"{v:.2f}" for v in prof),
                    _cost_report(length, _legal_ceil(length), fps, s_per_step,
                                 est_steps, overhead_s,
                                 tail=f"abstained, profile contrast "
                                      f"{contrast:.2f} < {abstain_below:g}"))

        thr = np.quantile(prof, q)
        tok_d = np.where(prof >= thr, d_max, 1).astype(int)
        if bridge:
            # production rule (measured): never let the plateau dip between
            # peaks of the same burst — the dip is where mid-burst artifacts
            # come back (4 of 5 in the v1 map). Fill short valleys at d_max.
            hot = np.where(tok_d == d_max)[0]
            for a, b in zip(hot[:-1], hot[1:]):
                if 1 < b - a <= bridge:
                    tok_d[a:b + 1] = d_max
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
        n_held = sum(1 for h in holds if h > 1)
        report = _cost_report(
            length, _legal_ceil(sum(holds)), fps, s_per_step, est_steps,
            overhead_s,
            tail=f"{n_held} of {length} frames held, peak x{int(tok_d.max())}")
        return (hold_map, ",".join(segs), int(w0), int(wlen), profile, report)


def _compile_hold_map(frame_holds, length, ramp, bridge):
    """Frame-domain holds -> token-snapped holds + segments string.
    Shared by H3ManualHoldMap and H3MotionEditor; same bridge/ramp rules
    as the oracle."""
    t_lat = 0
    while _tok_start_frame(t_lat) < length:
        t_lat += 1
    tok_d = np.ones(t_lat, int)
    for t in range(t_lat):
        f0 = _tok_start_frame(t)
        f1 = min(_tok_start_frame(t + 1), length)
        if f1 > f0:
            tok_d[t] = int(np.max(frame_holds[f0:f1]))

    d_peak = int(tok_d.max())
    if bridge and d_peak > 1:
        hot = np.where(tok_d == d_peak)[0]
        for a, b in zip(hot[:-1], hot[1:]):
            if 1 < b - a <= bridge:
                tok_d[a:b + 1] = d_peak
    if ramp and d_peak > 1:
        for _ in range(d_peak - 1):
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
    return holds, ",".join(segs), t_lat


def _env_value(auto, param, frame, default):
    """Evaluate a breakpoint envelope [[frame, value], ...] at a frame."""
    pts = (auto or {}).get(param)
    if not pts:
        return default
    pts = sorted(pts, key=lambda p: p[0])
    if frame <= pts[0][0]:
        return float(pts[0][1])
    if frame >= pts[-1][0]:
        return float(pts[-1][1])
    for (f0, v0), (f1, v1) in zip(pts[:-1], pts[1:]):
        if f0 <= frame <= f1:
            if f1 == f0:
                return float(v1)
            a = (frame - f0) / (f1 - f0)
            return float(v0) + a * (float(v1) - float(v0))
    return default


def _rasterize_strokes(strokes, h, w):
    """Vector strokes (normalized coords, disc brush) -> (h, w) 0/1 mask.
    Brush and erase apply in stroke order."""
    import math
    m = torch.zeros(h, w)
    for s in strokes or []:
        r = max(1.0, float(s.get("r", 0.03)) * w)
        pts = s.get("pts") or []
        stamped = []
        for i, (x1, y1) in enumerate(pts):
            if i == 0:
                stamped.append((x1, y1))
                continue
            x0, y0 = pts[i - 1]
            d = math.hypot((x1 - x0) * w, (y1 - y0) * h)
            nsub = max(1, int(d / max(1.0, r * 0.5)))
            for k in range(1, nsub + 1):
                stamped.append((x0 + (x1 - x0) * k / nsub,
                                y0 + (y1 - y0) * k / nsub))
        val = 0.0 if s.get("t") == "erase" else 1.0
        for px, py in stamped:
            cx, cy = px * w, py * h
            x0i, x1i = int(max(0, cx - r - 1)), int(min(w, cx + r + 2))
            y0i, y1i = int(max(0, cy - r - 1)), int(min(h, cy + r + 2))
            if x1i <= x0i or y1i <= y0i:
                continue
            ys = torch.arange(y0i, y1i).float()[:, None]
            xs = torch.arange(x0i, x1i).float()[None, :]
            hit = (ys - cy) ** 2 + (xs - cx) ** 2 <= r * r
            patch = m[y0i:y1i, x0i:x1i]
            patch[hit] = val
    return m


class H3ManualHoldMap:
    """Author the hold map by hand: time ranges in, oracle-format hold
    map out. Solo mode replaces the oracle; gate mode keeps the oracle's
    holds only inside your ranges (the oracle proposes, you dispose)."""

    DESCRIPTION = (
        "EXPERIMENTAL (alpha), new 2026-08-09; the classic pipeline nodes are unchanged.\n\n"
        "Manual targeting: turns user-chosen time ranges into the same "
        "hold-map JSON the H3 Jerk Oracle emits, so H3 Time Smear, H3 "
        "Exact Recover and H3 Audio Recover work unmodified.\n\n"
        "ranges syntax: comma-separated start-end pairs, in frames or "
        "seconds, with an optional per-range hold count: '36-60, "
        "88-102:3' or '1.5s-2.4s:4'. Ends inclusive. Ranges snap "
        "outward to the model's token grid (one token spans ~4 frames); "
        "the segments output echoes what actually got held, so trust it "
        "over your typed numbers.\n\n"
        "GATE mode: wire the oracle's hold_map into oracle_hold_map and "
        "the oracle's holds survive only inside your ranges — the fix "
        "for an overzealous oracle. Leave it unwired to author holds "
        "directly at 'hold' per range.\n\n"
        "The report output is the price tag: world length vs effective "
        "regen length. Show it before committing to the expensive pass; "
        "set s_per_step (measured on a baseline run of YOUR clip on "
        "YOUR card) for a minutes estimate.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "length": ("INT", {"default": 124, "min": 5, "max": 3600,
                       "tooltip": "world-clock frame count of the clip"}),
            "fps": ("INT", {"default": 24, "min": 1, "max": 120}),
            "ranges": ("STRING", {"default": "", "multiline": True,
                       "tooltip": "start-end[:hold], comma-separated; frames or seconds ('1.5s'), ends inclusive"}),
            "hold": ("INT", {"default": 4, "min": 2, "max": 8,
                     "tooltip": "hold count for ranges without an explicit :hold"}),
            "ramp": ("BOOLEAN", {"default": True,
                     "tooltip": "C1 ramp shoulders, same as the oracle — keep ON"}),
            "bridge": ("INT", {"default": 8, "min": 0, "max": 20,
                       "tooltip": "fill short valleys between peak spans, same rule as the oracle"}),
        }, "optional": {
            "oracle_hold_map": ("STRING", {"default": "", "forceInput": True,
                                "tooltip": "wire H3 Jerk Oracle's hold_map to gate it by your ranges"}),
            **_cost_widgets(),
        }}

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("hold_map", "segments", "report")
    FUNCTION = "build"
    CATEGORY = "latent/minimax/motion"

    def build(self, length, fps, ranges, hold, ramp, bridge,
              oracle_hold_map="", s_per_step=0.0, est_steps=18,
              overhead_s=OVERHEAD_S):
        spans = []
        for part in ranges.split(","):
            part = part.strip()
            if not part:
                continue
            h = hold
            if ":" in part:
                part, hs = part.rsplit(":", 1)
                h = max(1, int(hs))
            a_s, b_s = part.split("-")

            def to_frame(v):
                v = v.strip().lower()
                return (int(round(float(v[:-1]) * fps)) if v.endswith("s")
                        else int(v))

            a, b = max(0, to_frame(a_s)), min(length - 1, to_frame(b_s))
            assert a <= b, f"empty range '{part}' after clamping to the clip"
            spans.append((a, b, h))
        assert spans, "give at least one range, e.g. '36-60' or '1.5s-2.4s:4'"

        frame_holds = np.ones(length, int)
        if oracle_hold_map.strip():
            oracle = json.loads(oracle_hold_map)["holds"]
            assert len(oracle) == length, (
                f"oracle map covers {len(oracle)} frames, length is {length}")
            for a, b, _ in spans:                 # gate: oracle inside, 1 outside
                frame_holds[a:b + 1] = oracle[a:b + 1]
        else:
            for a, b, h in spans:
                frame_holds[a:b + 1] = h

        holds, segments, t_lat = _compile_hold_map(frame_holds, length,
                                                   ramp, bridge)
        report = _cost_report(length, _legal_ceil(sum(holds)), fps,
                              s_per_step, est_steps, overhead_s)
        hold_map = json.dumps({"holds": holds, "world_len": length})
        return (hold_map, segments, report)


class H3TimeSmear:
    """Retime frames onto a longer uniform grid by integer holds — the
    nonuniform (oracle) or uniform (dilation) smear that seeds v2v
    injection. Output length is snapped up to the 17k+5 grid by extending
    the final hold; the emitted hold_map records exactly what happened so
    H3ExactRecover can invert it losslessly."""

    DESCRIPTION = (
        "Retimes frames onto a longer uniform grid by integer frame holds — "
        "the seed material for v2v regeneration.\n\n"
        "Two modes: UNIFORM (nothing wired to hold_map): every frame held "
        "'dilation' times (default 4 — the zero-artifact reference point; "
        "highest cost). ADAPTIVE (wire H3 Jerk Oracle's hold_map): only "
        "jerk-hot spans get held, quiet spans stay real-time — cheaper and "
        "preserves the clip's natural beat contrast ('motion beauty'), at a "
        "small artifact risk where the hold curve dips inside a burst.\n\n"
        "Output length is snapped up to the H3-legal 17k+5 grid by extending "
        "the final hold. ALWAYS pass hold_map_used to H3 Exact Recover — it "
        "records exactly what happened so recovery is lossless.\n\n"
        "EFFECTIVE SIZE: the report output is the price tag for everything "
        "downstream of here — a 5 s action clip regenerates as 11 to 13 s of "
        "frame data, and that dilated length, not the runtime, is what sets "
        "the bill and the VRAM peak. Read the TIME multiplier, not the frame "
        "one: per-step cost is superlinear in tokens, so 2.5x the frames is "
        "about 4.9x the time per step. Set s_per_step from a baseline render "
        "of this clip on this card for a minutes estimate.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "dilation": ("INT", {"default": 4, "min": 1, "max": 8,
                                 "tooltip": "uniform hold count; ignored when hold_map is wired"}),
        }, "optional": {
            "hold_map": ("STRING", {"default": "",
                                    "tooltip": "from H3JerkOracle — per-frame integer holds"}),
            **_cost_widgets(with_fps=True),
        }}

    RETURN_TYPES = ("IMAGE", "STRING", "INT", "STRING")
    RETURN_NAMES = ("images", "hold_map_used", "length", "report")
    FUNCTION = "smear"
    CATEGORY = "image/minimax/motion"

    def smear(self, images, dilation, hold_map="", fps=24, s_per_step=0.0,
              est_steps=18, overhead_s=OVERHEAD_S):
        images = images.detach().cpu()  # keep the (possibly huge) held batch off VRAM
        n = images.shape[0]
        holds = (json.loads(hold_map)["holds"] if hold_map.strip()
                 else [dilation] * n)
        assert len(holds) == n, f"hold map covers {len(holds)} frames, batch has {n}"
        target = _legal_ceil(sum(holds))
        n_held = sum(1 for h in holds if h > 1)   # count before the tail pad
        holds = list(holds)
        holds[-1] += target - sum(holds)          # tail pad lives in the last hold
        idx = torch.tensor([i for i, h in enumerate(holds) for _ in range(h)])
        used = json.dumps({"holds": holds, "world_len": n})
        mode = ("uniform x{}".format(dilation) if not hold_map.strip()
                else "adaptive, {} of {} frames held".format(n_held, n))
        report = _cost_report(n, target, fps, s_per_step, est_steps,
                              overhead_s, tail=mode)
        return (images[idx], used, int(target), report)


class H3ExactRecover:
    """Invert H3TimeSmear: keep the first frame of every hold group —
    exact 24fps real-time recovery by frame selection (never resampling)."""

    DESCRIPTION = (
        "Inverts H3 Time Smear: keeps the first frame of every hold group, "
        "giving exact 24fps real-time recovery by pure frame selection — "
        "never interpolation or resampling, so recovered frames are pixel-"
        "identical to generated ones. Wire hold_map from the SAME H3 Time "
        "Smear node that produced the frames (hold_map_used output).")

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

    DESCRIPTION = (
        "Wraps a VAE-encoded video latent (from VAEEncode of the smeared "
        "frames) as the nested audio+video latent H3's SamplerCustomAdvanced "
        "expects, ready for partial-denoise injection. Audio starts empty and "
        "generates jointly with the video on the truncated schedule — "
        "causally synced foley, the preferred audio source for regenerated "
        "content. length=0 (default) derives the frame count from the latent "
        "itself; wire H3 Time Smear's length output or set it only to assert "
        "a specific grid.\n\n"
        "Background freeze: wire the BASELINE latent into oracle_samples and "
        "set freeze_threshold above 0 to keep everything outside the "
        "oracle's motion region frozen to the smeared init during "
        "generation. Frozen background is held baseline content, so after "
        "exact recovery its timing is exactly the baseline's: background "
        "agents (birds, crowds) cannot speed up. The mask is static over "
        "time, so nothing pops at its boundary. Effects that fly far from "
        "the subject may be clipped by the freeze; lower the threshold or "
        "raise freeze_grow to give them room.\n\n"
        "Manual freeze: wire a MASK instead and YOU choose the boundary "
        "(overrides the oracle path). mask = the region to REGENERATE; "
        "invert_mask flips it so you can paint the background/birds to "
        "freeze directly. The mask is unioned over time (static boundary, "
        "never pops) and snapped to HARD latent cells by default "
        "(mask_feather 0): every ~16 px cell is fully frozen or fully "
        "live, no half-frozen blend cells, and the decode smooths the "
        "edge. Set mask_feather above 0 to get the old pixel-space ramp "
        "pooled to fractional cells if a hard seam ever shows. Prefer "
        "this over the composite node when background and subject share "
        "lighting, shadows or water contact.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "samples": ("LATENT", {"tooltip": "video latent from VAEEncode of the smeared frames"}),
        }, "optional": {
            "length": ("INT", {"default": 0, "min": 0, "max": 3600,
                               "tooltip": "0 = derive from the latent (recommended); nonzero asserts this exact 17k+5 length"}),
            "oracle_samples": ("LATENT", {"tooltip": "baseline latent; enables background freezing"}),
            "freeze_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                "tooltip": "0 = off. Above 0: freeze background latent to the smeared init so its "
                           "timing stays exactly the baseline's (fixes background agents speeding up). "
                           "The subject mask is the oracle heat unioned over time, so the boundary never moves. "
                           "0.35 is a sane start."}),
            "freeze_grow": ("INT", {"default": 2, "min": 0, "max": 16,
                "tooltip": "latent-pixels of mask dilation (16 image px each); applies to both mask sources"}),
            "mask": ("MASK", {"tooltip": "(alpha) manual region to REGENERATE (1) vs freeze to baseline timing (0). "
                     "Overrides the oracle path. Union over time: the boundary never moves"}),
            "mask_feather": ("INT", {"default": 0, "min": 0, "max": 256,
                "tooltip": "0 (default): hard latent cells, every cell fully frozen or fully live; "
                           "the decode smooths the edge. >0: pixel-space ramp pooled to fractional "
                           "cells (~16 px quanta) if a hard seam ever shows"}),
            "invert_mask": ("BOOLEAN", {"default": False,
                "tooltip": "on: the mask marks the FREEZE region instead (paint the background/birds directly)"}),
        }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "build"
    CATEGORY = "latent/minimax/motion"

    def build(self, samples, length=0, oracle_samples=None, freeze_threshold=0.0,
              freeze_grow=2, mask=None, mask_feather=32, invert_mask=False):
        import torch.nn.functional as F

        import comfy.nested_tensor
        from comfy_extras.nodes_minimax_h3 import temporal_shape

        video = _video_component(samples)
        if not length:
            length = (video.shape[2] - 2) // 5 * 17 + 5  # invert t_lat
        _, t_lat, audio_t = temporal_shape(length)
        assert video.shape[2] == t_lat, (
            f"latent has {video.shape[2]} tokens, length {length} needs {t_lat}")
        audio = torch.zeros(video.shape[0], 32, 2, audio_t,
                            device=video.device, dtype=video.dtype)
        out = {"samples": comfy.nested_tensor.NestedTensor((video, audio))}

        if mask is not None:
            h, w = video.shape[3], video.shape[4]
            m = mask.detach().float().cpu()
            if m.dim() == 2:
                m = m[None]
            if invert_mask:
                m = 1.0 - m
            m = (m.max(dim=0).values >= 0.5).float()[None]  # union: static boundary
            m = _soft_edge(m, mask_feather)                 # pixel-space ramp first
            m = F.interpolate(m[None], size=(h, w), mode="area")[0]  # fractional cells
            if freeze_grow:
                k = freeze_grow * 2 + 1
                m = F.max_pool2d(m[None], k, stride=1, padding=k // 2)[0]
            if mask_feather <= 0:
                # hard cells: area pooling leaves boundary cells fractional
                # (half-frozen blends denoise off-manifold); snap to 0/1 and
                # let the decode's receptive-field overlap smooth the edge
                m = (m >= 0.5).float()
            vid_mask = m[0].clamp(0, 1).expand(t_lat, h, w)[None, None].to(video.device)
            aud_mask = torch.ones(1, 32, 2, audio_t)
            out["noise_mask"] = comfy.nested_tensor.NestedTensor(
                (vid_mask.contiguous(), aud_mask))
        elif oracle_samples is not None and freeze_threshold > 0:
            z = _video_component(oracle_samples)
            v = z.detach().float().cpu().numpy()
            jmap = np.abs(np.diff(v, n=3, axis=2)).mean(axis=(0, 1))  # (T-3, h, w)
            for ph in range(5):
                m = jmap[ph::5].mean()
                if m > 0:
                    jmap[ph::5] /= m
            heat = jmap.max(axis=0)                       # union over time: static boundary
            lo, hi = np.quantile(heat, 0.05), np.quantile(heat, 0.995)
            heat = np.clip((heat - lo) / (hi - lo + 1e-9), 0, 1)
            m = torch.from_numpy(heat >= freeze_threshold).float()[None, None]
            if freeze_grow:
                k = freeze_grow * 2 + 1
                m = F.max_pool2d(m, k, stride=1, padding=k // 2)
            h, w = video.shape[3], video.shape[4]
            if m.shape[-2:] != (h, w):
                m = F.interpolate(m, size=(h, w), mode="nearest")
            vid_mask = m[0, 0].expand(t_lat, h, w)[None, None].to(video.device)
            aud_mask = torch.ones(1, 32, 2, audio_t)
            out["noise_mask"] = comfy.nested_tensor.NestedTensor(
                (vid_mask.contiguous(), aud_mask))
        return (out,)


class H3InjectSchedule:
    """Truncated sigma schedule for v2v injection. inject=0.70 (the
    measured sweet spot) keeps the init's coarse choreography and re-rolls
    rendering; lower inherits more artifact risk, higher drifts toward
    free generation (invented-physics regime)."""

    DESCRIPTION = (
        "Truncated sigma schedule for v2v injection — THE quality dial of "
        "the pipeline. inject = how much of the denoise trajectory actually "
        "runs on top of your smeared init.\n\n"
        "Recommended range 0.5-0.8. Default 0.70 (playback-ratified). 0.5 "
        "measured as the metric quality point in our A/B (sharpest AND "
        "closest choreography tracking) — try both on your content. Below "
        "~0.5 the init's own artifacts start surviving into the output; "
        "above ~0.8 the model increasingly ignores your baseline and invents "
        "its own choreography. total_steps 25 for the base model; drop to "
        "the distilled step count if you stack a turbo LoRA (measure first — "
        "injection under heavy distillation is still experimental).")

    PRESETS = {
        "balanced 0.70 (default)": 0.70,
        "faithful detail 0.50 (metric best)": 0.50,
        "loose / creative 0.80": 0.80,
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "scheduler": (["simple", "normal", "beta", "sgm_uniform", "karras",
                           "exponential"], {"default": "simple"}),
            "total_steps": ("INT", {"default": 25, "min": 4, "max": 100}),
            "inject": ("FLOAT", {"default": 0.70, "min": 0.05, "max": 1.0,
                                 "step": 0.05,
                                 "tooltip": "0.5-0.8 recommended; lower keeps init artifacts, higher invents choreography"}),
        }, "optional": {
            "preset": (["custom"] + list(cls.PRESETS), {"default": "balanced 0.70 (default)",
                       "tooltip": "any choice but 'custom' overrides the inject knob"}),
        }}

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "sigmas"
    CATEGORY = "sampling/custom_sampling/schedulers"

    def sigmas(self, model, scheduler, total_steps, inject, preset="custom"):
        if preset in self.PRESETS:
            inject = self.PRESETS[preset]
        import comfy.samplers
        full = comfy.samplers.calculate_sigmas(
            model.get_model_object("model_sampling"), scheduler, total_steps)
        run = max(1, int(round(total_steps * inject)))
        return (full[total_steps - run:],)


class H3JerkHeatmap:
    """The oracle made visible (demo tile as a node): jerk-heat overlay on
    the frames + a per-token jerk strip with playhead along the bottom."""

    DESCRIPTION = (
        "The oracle made visible: overlays the jerk heat map on your frames "
        "(red-yellow pools where motion is too fast per latent token) and "
        "draws the per-token jerk profile as a bar strip with a playhead — "
        "watch the burst light up as playback reaches it. With show_drift "
        "on, regions that move steadily WITHOUT burst jerk (birds, crowds, "
        "traffic: velocity-high, jerk-low) glow blue: the drifter class "
        "that time warping mishandles and background freezing protects. "
        "Purely diagnostic/"
        "presentational; wire the same latent you'd give H3 Jerk Oracle. "
        "alpha 0.4-0.7 reads well; strip_height 0 hides the bar strip.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "samples": ("LATENT",),
            "alpha": ("FLOAT", {"default": 0.55, "min": 0.0, "max": 1.0, "step": 0.05}),
            "strip_height": ("INT", {"default": 96, "min": 0, "max": 256}),
        }, "optional": {
            "show_drift": ("BOOLEAN", {"default": True,
                "tooltip": "blue overlay on steady movers (velocity-high, jerk-low): the birds"}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "overlay"
    CATEGORY = "image/minimax/motion"

    def overlay(self, images, samples, alpha, strip_height, show_drift=True):
        images = images.detach().float().cpu()  # --gpu-only hands us cuda tensors
        z = _video_component(samples)
        t_lat = z.shape[2]
        n, H, W, _ = images.shape

        v = z.detach().float().cpu().numpy()
        jmap = np.abs(np.diff(v, n=3, axis=2)).mean(axis=(0, 1))   # (T-3, h, w)
        vmap = np.abs(np.diff(v, n=1, axis=2)).mean(axis=(0, 1))   # (T-1, h, w)
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

        drift = None
        if show_drift:
            vtok = np.stack([vmap[min(max(k - 1, 0), vmap.shape[0] - 1)]
                             for k in range(t_lat)])
            for ph in range(5):
                m = vtok[ph::5].mean()
                if m > 0:
                    vtok[ph::5] /= m
            lo2, hi2 = np.quantile(vtok, 0.05), np.quantile(vtok, 0.995)
            vtok = np.clip((vtok - lo2) / (hi2 - lo2 + 1e-9), 0, 1)
            dmap = np.clip(vtok - tok, 0, 1)          # moving, but not bursting
            drift = torch.nn.functional.interpolate(
                torch.from_numpy(dmap).float()[None], size=(H, W),
                mode="bilinear", align_corners=False)[0]

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
            if drift is not None:
                dm = drift[k]
                da = (dm * alpha * 0.8)[..., None]
                dcolor = torch.stack([torch.zeros_like(dm),
                                      0.45 + 0.35 * (1 - dm),
                                      torch.ones_like(dm)], -1)
                img = img * (1 - da) + dcolor * da
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


class H3AudioRecover:
    """Retime the regenerated clip's jointly-generated audio back to the
    original clock, using the same hold map as the video."""

    DESCRIPTION = (
        "Retimes the regenerated clip's own audio back to the original "
        "clock, using the same hold map as the video. Each hold segment is "
        "compressed with a phase vocoder, so pitch is preserved while "
        "duration shrinks. Wire audio from VAEDecodeAudio of the "
        "regenerated latent and hold_map from the same H3 Time Smear that "
        "built the init; the result lines up with H3 Exact Recover's video "
        "frame for frame. fps is the video frame rate the holds count in.\n\n"
        "Thickness dial: the regenerated foley is scored for the slowed "
        "performance, so it comes back leaner than a native-speed mix "
        "(often a more realistic feel). Wire the baseline clip's audio "
        "into reference and raise reference_mix to blend its denser "
        "full-speed track back in: 0 keeps the lean regenerated foley, "
        "1 is the baseline track alone. Note: with an adaptive hold map, "
        "audio in unheld spans passes through untouched, so dialog "
        "outside the bursts is unaffected either way.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "audio": ("AUDIO",),
            "hold_map": ("STRING", {"default": ""}),
            "fps": ("INT", {"default": 24, "min": 1, "max": 120}),
        }, "optional": {
            "reference": ("AUDIO", {"tooltip": "baseline clip audio (already real-time)"}),
            "reference_mix": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0,
                                        "step": 0.05,
                                        "tooltip": "0 = regenerated foley only (lean), 1 = reference only (dense)"}),
        }}

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "recover"
    CATEGORY = "audio/minimax/motion"

    def recover(self, audio, hold_map, fps=24, reference=None, reference_mix=0.0):
        import math

        import torchaudio  # noqa: F401  (phase_vocoder)

        holds = json.loads(hold_map)["holds"]
        wav = audio["waveform"].detach().float().cpu()   # [B, C, N]
        sr = audio["sample_rate"]
        b, c, n = wav.shape
        x = wav.reshape(b * c, n)

        runs = []                                        # consecutive equal holds
        for h in holds:
            if runs and runs[-1][0] == h:
                runs[-1][1] += 1
            else:
                runs.append([h, 1])

        n_fft, hop = 2048, 512
        window = torch.hann_window(n_fft)
        phase_adv = torch.linspace(0, math.pi * hop, n_fft // 2 + 1)[..., None]
        spf = sr / float(fps)                            # samples per frame
        segs, cursor = [], 0.0
        for h, count in runs:
            src = h * count * spf
            s0, s1 = int(round(cursor)), int(round(cursor + src))
            cursor += src
            seg = x[:, s0:min(s1, n)]
            if seg.shape[1] == 0:
                continue
            if h == 1:
                segs.append(seg)
                continue
            spec = torch.stft(seg, n_fft, hop, window=window, return_complex=True)
            spec = torchaudio.functional.phase_vocoder(spec, float(h), phase_adv)
            segs.append(torch.istft(spec, n_fft, hop, window=window))
        y = torch.cat(segs, dim=1)
        if reference is not None and reference_mix > 0:
            ref = reference["waveform"].detach().float().cpu().reshape(-1, reference["waveform"].shape[-1])
            if reference["sample_rate"] != sr:
                import torchaudio as _ta
                ref = _ta.functional.resample(ref, reference["sample_rate"], sr)
            n_out = min(y.shape[1], ref.shape[1])
            if ref.shape[0] != y.shape[0]:
                ref = ref[:1].expand(y.shape[0], -1)
            y = (1 - reference_mix) * y[:, :n_out] + reference_mix * ref[:, :n_out]
        return ({"waveform": y.reshape(b, c, -1).contiguous(), "sample_rate": sr},)


class H3ProbeSchedule:
    """Head-only schedule for oracle probing: skip most of the first pass."""

    DESCRIPTION = (
        "Runs only the head of the baseline schedule. Wire the sampler's "
        "denoised_output (the x0 estimate) into H3 Jerk Oracle and into the "
        "decode that feeds H3 Time Smear: in our measurements the burst "
        "timing is readable by step 4-5 of 25, and injection destroys fine "
        "detail anyway, so the coarse early estimate is a workable init. "
        "probe_steps is the dial: 6 of 25 skips ~75% of the first pass; "
        "raise it if the init loses too much choreography on your content. "
        "Trade-off: no finished baseline means no finished baseline audio "
        "(H3 Audio Recover's reference input has nothing full-speed to "
        "blend, and the probe's own audio estimate is rough).")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "scheduler": (["simple", "normal", "beta", "sgm_uniform", "karras",
                           "exponential"], {"default": "simple"}),
            "total_steps": ("INT", {"default": 25, "min": 4, "max": 100}),
            "probe_steps": ("INT", {"default": 6, "min": 2, "max": 100,
                            "tooltip": "how much of the schedule to actually run"}),
        }}

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "sigmas"
    CATEGORY = "sampling/custom_sampling/schedulers"

    def sigmas(self, model, scheduler, total_steps, probe_steps):
        import comfy.samplers
        full = comfy.samplers.calculate_sigmas(
            model.get_model_object("model_sampling"), scheduler, total_steps)
        return (full[:min(probe_steps, total_steps) + 1],)


class H3ExpertSchedule:
    """Split the injected schedule: base-model head, turbo tail."""

    DESCRIPTION = (
        "Expert split for the regeneration pass: the first base_head steps "
        "run on the base model (structure forms on the least-distilled "
        "weights), the remaining steps run on the turbo LoRA (refinement, "
        "where distilled models are comfortable). Outputs head and tail "
        "sigma slices of one continuous schedule. Wire: head into a "
        "SamplerCustomAdvanced on the plain model with your RandomNoise; "
        "tail into a second SamplerCustomAdvanced on the LoRA-patched "
        "model with DisableNoise, continuing the head's output latent. "
        "Defaults: total 8, inject 0.70 (6 steps run), base_head 2, so the "
        "turbo tail gets 4 steps, its native count.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "scheduler": (["simple", "normal", "beta", "sgm_uniform", "karras",
                           "exponential"], {"default": "beta"}),
            "total_steps": ("INT", {"default": 8, "min": 4, "max": 100}),
            "inject": ("FLOAT", {"default": 0.70, "min": 0.05, "max": 1.0,
                                 "step": 0.05}),
            "base_head": ("INT", {"default": 2, "min": 0, "max": 20,
                          "tooltip": "steps run on the base model before the turbo tail"}),
        }}

    RETURN_TYPES = ("SIGMAS", "SIGMAS")
    RETURN_NAMES = ("head_sigmas", "tail_sigmas")
    FUNCTION = "sigmas"
    CATEGORY = "sampling/custom_sampling/schedulers"

    def sigmas(self, model, scheduler, total_steps, inject, base_head):
        import comfy.samplers
        full = comfy.samplers.calculate_sigmas(
            model.get_model_object("model_sampling"), scheduler, total_steps)
        run = max(1, int(round(total_steps * inject)))
        s = full[total_steps - run:]
        h = min(base_head, run - 1)
        return (s[:h + 1], s[h:])


class _TrajBankSampler:
    def __init__(self, inner, dump_dir, every_n):
        self.inner = inner
        self.dump_dir = dump_dir
        self.every_n = max(1, every_n)

    def max_denoise(self, model_wrap, sigmas):
        return self.inner.max_denoise(model_wrap, sigmas)

    def sample(self, model_wrap, sigmas, extra_args, callback, noise,
               latent_image=None, denoise_mask=None, disable_pbar=False):
        import os
        os.makedirs(self.dump_dir, exist_ok=True)
        torch.save(sigmas.detach().cpu(), os.path.join(self.dump_dir, "sigmas.pt"))

        def parts(t):
            if hasattr(t, "is_nested") and t.is_nested:
                return list(t.tensors)
            return [t]

        def cb(i, denoised, x, total):
            if i % self.every_n == 0 or i == total - 1:
                comps = parts(x)
                payload = {"step": i, "total_steps": total,
                           "video": comps[0].detach().to(torch.float16).cpu()}
                if len(comps) > 1:
                    payload["audio"] = comps[1].detach().to(torch.float16).cpu()
                torch.save(payload, os.path.join(self.dump_dir, f"x_step{i:03d}.pt"))
            if callback is not None:
                callback(i, denoised, x, total)

        return self.inner.sample(model_wrap, sigmas, extra_args, cb, noise,
                                 latent_image, denoise_mask, disable_pbar)


class H3TrajectoryBank:
    """SAMPLER wrapper that checkpoints the noisy latent at every step."""

    DESCRIPTION = (
        "Wraps a sampler and saves the trajectory latent (x_t, the noisy "
        "state the sampler actually carries) after each step, plus the "
        "sigma schedule. About 7 MB per step for a 5 s 1024 clip, so a "
        "full 25-step run banks under 200 MB. Pair with H3 Trajectory "
        "Load to branch from any step without recomputing the head: swap "
        "the model, LoRA, guider, or remaining schedule and continue.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "sampler": ("SAMPLER",),
            "dump_dir": ("STRING", {"default": "/tmp/h3_trajectory"}),
            "every_n": ("INT", {"default": 1, "min": 1, "max": 25,
                                "tooltip": "save every Nth step (last step always saved)"}),
        }}

    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "wrap"
    CATEGORY = "sampling/custom_sampling/samplers"

    def wrap(self, sampler, dump_dir, every_n):
        return (_TrajBankSampler(sampler, dump_dir, every_n),)


class H3TrajectoryLoad:
    """Resume a banked trajectory from any saved step."""

    DESCRIPTION = (
        "Loads a step checkpoint saved by H3 Trajectory Bank and the "
        "matching remaining sigma schedule. Wire the LATENT into a "
        "SamplerCustomAdvanced with DisableNoise and the SIGMAS output as "
        "its schedule: sampling continues exactly where the banked run "
        "stopped, under whatever model, LoRA, or guider you attach. "
        "Changing anything downstream of the loaded step is the point.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "dump_dir": ("STRING", {"default": "/tmp/h3_trajectory"}),
            "step": ("INT", {"default": 5, "min": 0, "max": 200,
                             "tooltip": "resume after this saved step (0-based)"}),
        }}

    RETURN_TYPES = ("LATENT", "SIGMAS", "INT")
    RETURN_NAMES = ("samples", "remaining_sigmas", "loaded_step")
    FUNCTION = "load"
    CATEGORY = "latent/minimax/motion"

    @classmethod
    def IS_CHANGED(cls, dump_dir, step):
        import os
        p = os.path.join(dump_dir, f"x_step{step:03d}.pt")
        try:
            return os.path.getmtime(p)
        except OSError:
            return float("nan")

    def load(self, dump_dir, step):
        import os

        import comfy.nested_tensor
        p = os.path.join(dump_dir, f"x_step{step:03d}.pt")
        d = torch.load(p, weights_only=True)
        sigmas = torch.load(os.path.join(dump_dir, "sigmas.pt"), weights_only=True)
        video = d["video"].float()
        if "audio" in d:
            samples = comfy.nested_tensor.NestedTensor((video, d["audio"].float()))
        else:
            samples = video
        return ({"samples": samples}, sigmas[step + 1:], int(step))


class H3MotionComposite:
    """Spatial recovery: regenerated pixels where the oracle saw motion
    (or where a manual mask says so), baseline pixels everywhere else."""

    DESCRIPTION = (
        "Fixes the sped-up-background side effect: inside dilated spans "
        "the model keeps background agents (birds, crowds, traffic) near "
        "their natural pace instead of full slow motion, so recovery "
        "overcranks them. This node composites per pixel on the shared "
        "world clock: where the subject mask is high it keeps the "
        "regenerated frames; where it is low it keeps the baseline, whose "
        "timing was correct all along.\n\n"
        "Two mask sources. ORACLE mode (wire samples, the BASELINE "
        "latent): spatial jerk heat picks the subject automatically; "
        "threshold sets how much heat counts as subject. MANUAL mode "
        "(wire mask): you decide. A human can hide the seam along a real "
        "edge (rooftop line, horizon) where the oracle cannot; lasso "
        "generously and let feather do the blending. mask=1 keeps "
        "regenerated pixels; invert_mask flips that, so you can lasso "
        "the birds/sky region you want kept at baseline timing instead. "
        "A single mask = static boundary (safe, never pops); a mask "
        "batch = per-frame on the world clock (moving boundaries can "
        "pop at the seam; feather harder).\n\n"
        "grow expands the mask to cover pose drift. feather softens the "
        "seam: profile linear (box), smoothstep or gaussian; direction "
        "centered straddles the boundary, inward eats into the masked "
        "side, outward eats into the kept side (trace a rooftop tight, "
        "then feather outward into the sky).")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "regenerated": ("IMAGE",),
            "baseline": ("IMAGE",),
            "threshold": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05,
                          "tooltip": "oracle mode: how much heat counts as subject; manual mode: binarization level for soft masks"}),
            "grow": ("INT", {"default": 32, "min": 0, "max": 256,
                             "tooltip": "pixels of mask dilation, covers pose drift"}),
            "feather": ("INT", {"default": 48, "min": 0, "max": 256}),
        }, "optional": {
            "samples": ("LATENT", {"tooltip": "BASELINE latent (oracle mode); optional when mask is wired"}),
            "mask": ("MASK", {"tooltip": "(alpha) manual subject mask, overrides the oracle. 1 = keep regenerated, 0 = keep baseline. One mask = static boundary; a batch = per-frame"}),
            "invert_mask": ("BOOLEAN", {"default": False,
                            "tooltip": "on: the mask marks the KEEP-BASELINE region instead (lasso the birds directly)"}),
            "feather_profile": (["linear", "smoothstep", "gaussian"], {"default": "linear"}),
            "feather_direction": (["centered", "inward", "outward"], {"default": "centered",
                                  "tooltip": "where the ramp lives relative to the mask boundary"}),
            "mask_is_soft": ("BOOLEAN", {"default": False,
                             "tooltip": "(alpha) mask values are final alphas (e.g. from H3 Motion Editor): skip threshold/grow/feather"}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "composite"
    CATEGORY = "image/minimax/motion"

    def composite(self, regenerated, baseline, threshold, grow, feather,
                  samples=None, mask=None, invert_mask=False,
                  feather_profile="linear", feather_direction="centered",
                  mask_is_soft=False):
        import torch.nn.functional as F

        regenerated = regenerated.detach().float().cpu()
        baseline = baseline.detach().float().cpu()
        n = min(regenerated.shape[0], baseline.shape[0])
        H, W = baseline.shape[1], baseline.shape[2]

        if mask is not None:
            m = mask.detach().float().cpu()
            if m.dim() == 2:
                m = m[None]
            if invert_mask:
                m = 1.0 - m
            if m.shape[-2:] != (H, W):
                m = F.interpolate(m[:, None], size=(H, W), mode="bilinear",
                                  align_corners=False)[:, 0]
            alpha = m.clamp(0, 1) if mask_is_soft else (m >= threshold).float()
            per_frame = alpha.shape[0]
            token_indexed = False
        else:
            assert samples is not None, \
                "wire samples (oracle mode) or mask (manual mode)"
            z = _video_component(samples)
            t_lat = z.shape[2]
            v = z.detach().float().cpu().numpy()
            jmap = np.abs(np.diff(v, n=3, axis=2)).mean(axis=(0, 1))
            tok = np.stack([jmap[min(max(k - 1, 0), jmap.shape[0] - 1)]
                            for k in range(t_lat)])
            for ph in range(5):
                m = tok[ph::5].mean()
                if m > 0:
                    tok[ph::5] /= m
            lo, hi = np.quantile(tok, 0.05), np.quantile(tok, 0.995)
            tok = np.clip((tok - lo) / (hi - lo + 1e-9), 0, 1)
            heat = torch.from_numpy(tok).float()[None]          # (1, T, h, w)
            heat = F.interpolate(heat, size=(H, W), mode="bilinear",
                                 align_corners=False)[0]        # (T, H, W)
            alpha = (heat >= threshold).float()
            per_frame = 0
            token_indexed = True

        if not (mask is not None and mask_is_soft):
            if grow:
                k = grow // 2 * 2 + 1
                alpha = F.max_pool2d(alpha[:, None], k, stride=1,
                                     padding=k // 2)[:, 0]
            alpha = _soft_edge(alpha, feather, feather_profile,
                               feather_direction)

        out = []
        for f in range(n):
            if token_indexed:
                a = alpha[min(_frame_token(f, t_lat), t_lat - 1)][..., None]
            elif per_frame == 1:
                a = alpha[0][..., None]
            else:
                idx = int(round(f * (per_frame - 1) / max(n - 1, 1)))
                a = alpha[min(idx, per_frame - 1)][..., None]
            out.append(baseline[f] * (1 - a) + regenerated[f] * a)
        return (torch.stack(out),)


class H3MotionEditor:
    """DAW-style timeline + mask editor. The JS widget (web/motion_editor.js)
    edits a serialized state; this node compiles it into a hold map, a soft
    per-frame mask, and envelope data. Agents can author the same state JSON
    directly, no GUI needed.

    editor_state contract (v1):
      {"v": 1, "blocks": [{
          "id": str, "start": int, "end": int,     # frames, inclusive
          "hold": int,                              # 0 = use oracle here
          "dials": {"feather": 48, "profile": "smoothstep",
                     "direction": "centered", "grow": 0, "fade": 6,
                     "strength": 1.0},
          "auto": {"hold": [[f,v],...], "feather": [[f,v],...],
                    "strength": [[f,v],...]},       # breakpoint envelopes
          "strokes": {"<frame>": [{"t": "brush"|"erase", "r": 0.03,
                                    "pts": [[x,y],...]}, ...]},  # normalized
          "static_strokes": [ ...same, applies to every frame of the block ]
      }, ...]}

    Mask semantics: a block with no strokes regenerates the whole frame for
    its time span; strokes narrow that to the painted problem areas. Frames
    outside every block follow outside_blocks. No blocks at all = mask is
    all ones (composite becomes a no-op passthrough of the regenerated clip)
    and the oracle hold map (if wired) passes through untouched."""

    DESCRIPTION = (
        "EXPERIMENTAL (alpha), new 2026-08-09; the classic pipeline nodes are unchanged.\n\n"
        "The Motion Lab editor node. Wire the baseline frames (and latent) "
        "in, queue once to load the filmstrip, then edit right on the node: "
        "drag time blocks on the timeline (DAW-style brackets, snapped to "
        "the model's token grid), click a block and paint problem areas "
        "frame by frame, dial feather/grow/fade per block, and draw "
        "automation envelopes for hold, feather and strength. Outputs are "
        "drop-ins: hold_map feeds H3 Time Smear, mask feeds H3 Motion "
        "Composite (enable mask_is_soft there: the mask comes out already "
        "feathered and envelope-scaled), report prices the pass before you "
        "run it. Everything upstream stays cached between edits, so "
        "re-queueing after an edit only re-runs the regeneration side.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE", {"tooltip": "baseline frames (world clock)"}),
            "editor_state": ("STRING", {"default": "", "multiline": True,
                             "tooltip": "serialized editor state; the GUI widget maintains this"}),
        }, "optional": {
            "samples": ("LATENT", {"tooltip": "baseline latent, for the jerk profile strip"}),
            "oracle_hold_map": ("STRING", {"default": "", "forceInput": True,
                                "tooltip": "wire the oracle to gate it; blocks with hold=0 use oracle holds"}),
            "fps": ("INT", {"default": 24, "min": 1, "max": 120}),
            "ramp": ("BOOLEAN", {"default": True}),
            "bridge": ("INT", {"default": 8, "min": 0, "max": 20}),
            "invert_mask": ("BOOLEAN", {"default": False,
                            "tooltip": "flip the final mask (paint keep-baseline regions instead)"}),
            "outside_blocks": (["baseline", "regenerated"], {"default": "baseline",
                               "tooltip": "what the composite shows on frames no block covers"}),
            "paint_res": ("INT", {"default": 512, "min": 128, "max": 1024, "step": 64,
                          "tooltip": "mask compile width; the composite rescales to full res"}),
        }}

    RETURN_TYPES = ("STRING", "MASK", "STRING", "STRING")
    RETURN_NAMES = ("hold_map", "mask", "envelopes", "report")
    OUTPUT_NODE = True
    FUNCTION = "compile"
    CATEGORY = "latent/minimax/motion"

    def _thumbs(self, images, paint_w):
        """Save filmstrip + paint-res frames to the temp dir for the widget.
        Returns the ui payload lists; empty when not running inside ComfyUI."""
        try:
            import folder_paths
            from PIL import Image
        except ImportError:
            return [], []
        import os
        import uuid
        sub = "h3_editor"
        root = os.path.join(folder_paths.get_temp_directory(), sub)
        os.makedirs(root, exist_ok=True)
        tag = uuid.uuid4().hex[:8]
        n, H, W, _ = images.shape
        pw = min(paint_w, W)
        ph = max(1, round(H * pw / W))
        sw = 96
        sh = max(1, round(H * sw / W))
        paint, strip = [], []
        arr = (images.clamp(0, 1) * 255).byte().cpu().numpy()
        for f in range(n):
            im = Image.fromarray(arr[f])
            name_p = f"{tag}_p{f:04d}.jpg"
            im.resize((pw, ph)).save(os.path.join(root, name_p), quality=88)
            paint.append({"filename": name_p, "subfolder": sub, "type": "temp"})
            name_s = f"{tag}_s{f:04d}.jpg"
            im.resize((sw, sh)).save(os.path.join(root, name_s), quality=80)
            strip.append({"filename": name_s, "subfolder": sub, "type": "temp"})
        return paint, strip

    def compile(self, images, editor_state, samples=None, oracle_hold_map="",
                fps=24, ramp=True, bridge=8, invert_mask=False,
                outside_blocks="baseline", paint_res=512):
        import torch.nn.functional as F

        images = images.detach().float().cpu()
        n, H, W, _ = images.shape
        pw = min(paint_res, W)
        ph = max(1, round(H * pw / W))

        state = {}
        if editor_state.strip():
            state = json.loads(editor_state)
        blocks = state.get("blocks") or []

        oracle = None
        if oracle_hold_map.strip():
            oracle = json.loads(oracle_hold_map)["holds"]
            assert len(oracle) == n, (
                f"oracle map covers {len(oracle)} frames, clip has {n}")

        # ---- hold map ----
        if not blocks:
            holds = list(oracle) if oracle else [1] * n
            segments = ""
            if oracle:
                _, segments, _ = _compile_hold_map(
                    np.asarray(holds, int), n, False, 0)
        else:
            frame_holds = np.ones(n, int)
            for b in blocks:
                a = max(0, int(b.get("start", 0)))
                z = min(n - 1, int(b.get("end", a)))
                base_hold = int(b.get("hold", 0))
                for f in range(a, z + 1):
                    h = _env_value(b.get("auto"), "hold", f,
                                   float(base_hold))
                    h = int(round(h))
                    if h <= 0:
                        h = oracle[f] if oracle else 4
                    frame_holds[f] = max(frame_holds[f], h)
            holds, segments, _ = _compile_hold_map(frame_holds, n, ramp, bridge)

        # ---- mask ----
        if not blocks:
            mask = torch.ones(n, ph, pw)
        else:
            outside = 0.0 if outside_blocks == "baseline" else 1.0
            mask = torch.full((n, ph, pw), outside)
            for b in blocks:
                a = max(0, int(b.get("start", 0)))
                z = min(n - 1, int(b.get("end", a)))
                dials = b.get("dials") or {}
                fade = max(0, int(dials.get("fade", 6)))
                grow = max(0, int(dials.get("grow", 0)))
                profile = dials.get("profile", "smoothstep")
                direction = dials.get("direction", "centered")
                static = b.get("static_strokes") or []
                per_frame = b.get("strokes") or {}
                base_m = (_rasterize_strokes(static, ph, pw)
                          if static else None)
                for f in range(a, z + 1):
                    fs = per_frame.get(str(f)) or []
                    if fs or static:
                        m = base_m.clone() if base_m is not None \
                            else torch.zeros(ph, pw)
                        if fs:
                            mm = _rasterize_strokes(fs, ph, pw)
                            m = torch.maximum(m, mm)
                    else:
                        m = torch.ones(ph, pw)   # bare block: whole frame
                    if grow:
                        k = grow // 2 * 2 + 1
                        m = F.max_pool2d(m[None, None], k, stride=1,
                                         padding=k // 2)[0, 0]
                    feather = int(round(_env_value(
                        b.get("auto"), "feather", f,
                        float(dials.get("feather", 48)))))
                    feather = int(feather * pw / max(W, 1))  # px are image px
                    if feather > 0:
                        m = _soft_edge(m[None], feather, profile,
                                       direction)[0]
                    strength = _env_value(b.get("auto"), "strength", f,
                                          float(dials.get("strength", 1.0)))
                    if fade:
                        edge = min(f - a + 1, z - f + 1)
                        if edge <= fade:
                            strength *= edge / (fade + 1)
                    m = m * max(0.0, min(1.0, strength))
                    mask[f] = torch.maximum(mask[f], m)
        if invert_mask:
            mask = 1.0 - mask

        # ---- report / envelopes ----
        dilated = _legal_ceil(sum(holds)) if holds else n
        t_world = (_legal_ceil(n) - 5) // 17 * 5 + 2
        t_dil = (dilated - 5) // 17 * 5 + 2
        report = (f"{n}f ({n / fps:.1f}s) -> {dilated}f ({dilated / fps:.1f}s) "
                  f"effective regen, {dilated / max(n, 1):.2f}x frames / "
                  f"{(t_dil / max(t_world, 1)) ** COST_EXP:.1f}x time per step; "
                  f"{len(blocks)} block(s)")
        if segments:
            report += f"; held segments {segments}"
        envelopes = json.dumps({
            "fps": fps, "length": n,
            "blocks": [{"id": b.get("id"), "start": b.get("start"),
                        "end": b.get("end"), "auto": b.get("auto") or {}}
                       for b in blocks]})
        hold_map = json.dumps({"holds": holds, "world_len": n})

        paint, strip = self._thumbs(images, paint_res)
        prof = []
        if samples is not None:
            prof = [round(float(v), 3)
                    for v in _jerk_profile(_video_component(samples))]
        ui = {"h3_paint": paint, "h3_strip": strip,
              "h3_profile": prof, "h3_length": [n], "h3_fps": [fps],
              "h3_report": [report]}
        return {"ui": ui,
                "result": (hold_map, mask, envelopes, report)}


class H3SegmentCrop:
    """Cut the world down to the held window plus real-time handles, so the
    expensive regeneration pass only pays for the frames the user targeted."""

    DESCRIPTION = (
        "EXPERIMENTAL (alpha), new 2026-08-09; the classic pipeline nodes are unchanged.\n\n"
        "The compute lever for targeted de-roping: crops the clip to the "
        "hold map's held span plus handle_frames of untouched context on "
        "each side. Wire the cropped images into H3 Time Smear together "
        "with the emitted segment hold_map, and the whole regeneration "
        "chain (encode, v2v, sample, recover, audio) runs on the segment "
        "only. Cost scales with dilated token count, so a one-burst window "
        "in a long clip regenerates several times faster than the full "
        "world. splice_map goes to H3 Segment Splice for reassembly. "
        "first/last frame outputs are the handle endpoints for FLF "
        "pinning on the regeneration's conditioning if you run an FL2VA "
        "checkpoint (recommended: it anchors the seam poses).\n\n"
        "The handles are generated at hold 1 and injected at your inject "
        "value, so they come back close to the baseline; H3 Segment "
        "Splice crossfades inside them. Multiple separate bursts are "
        "covered by one window spanning first to last held frame in this "
        "version.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE", {"tooltip": "baseline frames, world clock"}),
            "hold_map": ("STRING", {"default": "", "forceInput": True,
                         "tooltip": "from the oracle, H3 Manual Hold Map, or H3 Motion Editor"}),
            "handle_frames": ("INT", {"default": 12, "min": 2, "max": 48,
                              "tooltip": "real-time context frames kept on each side of the held span"}),
        }, "optional": {
            "grid_align": ("BOOLEAN", {"default": False,
                           "tooltip": "(alpha, OFF = shipped behaviour, bit-identical) H3 Time "
                                      "Smear reaches the 17k+5 grid by extending the LAST hold, "
                                      "which parks a multi-frame freeze on the window's final "
                                      "frame (measured: hold 8, a third of a second frozen) and "
                                      "a frozen tail is where H3 invents extra beats. Handle "
                                      "frames cost exactly 1 smeared frame each, so turning this "
                                      "ON grows a hold-1 handle by the deficit instead and the "
                                      "sum lands on the grid exactly, no freeze. Changes the "
                                      "crop bounds, so it changes the render: leave it off to "
                                      "reproduce an earlier result."}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("images", "hold_map", "splice_map", "first_frame",
                    "last_frame", "report")
    FUNCTION = "crop"
    CATEGORY = "image/minimax/motion"

    def crop(self, images, hold_map, handle_frames=12, grid_align=False):
        images = images.detach().cpu()
        holds = json.loads(hold_map)["holds"]
        n = images.shape[0]
        assert len(holds) == n, f"hold map covers {len(holds)}, clip has {n}"
        held = [i for i, h in enumerate(holds) if h > 1]
        assert held, "hold map has no held span; nothing to crop"
        a = max(0, held[0] - handle_frames)
        b = min(n - 1, held[-1] + handle_frames)
        raw = sum(_seg_holds(holds, a, b, held[0], held[-1]))
        pad = _legal_ceil(raw) - raw          # what H3TimeSmear would freeze
        if grid_align:
            a, b, pad = _grid_grow(holds, a, b, held[0], held[-1], n)
        seg = images[a:b + 1]
        seg_holds = [holds[f] if held[0] <= f <= held[-1] else 1
                     for f in range(a, b + 1)]
        seg_len = b - a + 1
        splice = json.dumps({
            "start": a, "end": b, "world_len": n,
            "handle_in": held[0] - a, "handle_out": b - held[-1]})
        dil_full = _legal_ceil(sum(holds))
        dil_seg = _legal_ceil(sum(seg_holds))
        t_full = (dil_full - 5) // 17 * 5 + 2
        t_seg = (dil_seg - 5) // 17 * 5 + 2
        report = (f"window f{a}-f{b} ({seg_len}f of {n}f); regen "
                  f"{dil_seg}f instead of {dil_full}f full-clip "
                  f"({t_seg} vs {t_full} tokens): about "
                  f"{(t_full / t_seg) ** COST_EXP:.1f}x faster per step "
                  f"(cost is superlinear in tokens, so the time saved beats "
                  f"the {t_full / t_seg:.1f}x token cut)")
        if grid_align:
            report += (f"; grid_align grew the handles to f{a}-f{b}, "
                       f"tail freeze {pad}f")
        elif pad:
            report += (f"; NOTE H3 Time Smear will reach the grid by holding "
                       f"the last frame {pad + 1}x ({pad}f of freeze); "
                       f"grid_align removes it")
        return (seg, json.dumps({"holds": seg_holds, "world_len": seg_len}),
                splice, seg[:1].clone(), seg[-1:].clone(), report)


class H3SegmentSplice:
    """Reassemble: baseline outside the window, regenerated segment inside,
    crossfaded across the handles. Audio spliced sample-accurately."""

    DESCRIPTION = (
        "EXPERIMENTAL (alpha), new 2026-08-09; the classic pipeline nodes are unchanged.\n\n"
        "Inverts H3 Segment Crop after recovery: the world is baseline "
        "outside the window and the recovered segment inside, with a "
        "video crossfade over feather_frames inside each handle zone "
        "(handles regenerate close to baseline, so the blend hides the "
        "residual tone drift). Wire segment = H3 Exact Recover's output "
        "for the segment. Audio: baseline_audio is the full clip track, "
        "segment_audio the retimed segment track from H3 Audio Recover; "
        "the splice is sample-accurate with an equal-power crossfade in "
        "the same handle zones. Omit the audio inputs to splice video "
        "only.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "baseline": ("IMAGE",),
            "segment": ("IMAGE", {"tooltip": "recovered segment, world clock"}),
            "splice_map": ("STRING", {"default": "", "forceInput": True}),
            "feather_frames": ("INT", {"default": 6, "min": 0, "max": 24,
                               "tooltip": "crossfade width inside each handle"}),
        }, "optional": {
            "baseline_audio": ("AUDIO",),
            "segment_audio": ("AUDIO",),
            "fps": ("INT", {"default": 24, "min": 1, "max": 120}),
        }}

    RETURN_TYPES = ("IMAGE", "AUDIO")
    FUNCTION = "splice"
    CATEGORY = "image/minimax/motion"

    def splice(self, baseline, segment, splice_map, feather_frames=6,
               baseline_audio=None, segment_audio=None, fps=24):
        baseline = baseline.detach().float().cpu()
        segment = segment.detach().float().cpu()
        sp = json.loads(splice_map)
        a, b, n = sp["start"], sp["end"], sp["world_len"]
        assert baseline.shape[0] >= n, (baseline.shape[0], n)
        seg_len = b - a + 1
        assert segment.shape[0] == seg_len, (segment.shape[0], seg_len)
        fade_in = min(feather_frames, sp.get("handle_in", feather_frames))
        fade_out = min(feather_frames, sp.get("handle_out", feather_frames))

        out = baseline.clone()
        for i in range(seg_len):
            w = 1.0
            if fade_in and i < fade_in:
                w = (i + 1) / (fade_in + 1)
            j = seg_len - 1 - i
            if fade_out and j < fade_out:
                w = min(w, (j + 1) / (fade_out + 1))
            f = a + i
            out[f] = baseline[f] * (1 - w) + segment[i] * w

        audio = baseline_audio
        if baseline_audio is not None and segment_audio is not None:
            import math
            sr = baseline_audio["sample_rate"]
            bw = baseline_audio["waveform"].detach().float().cpu()
            swav = segment_audio["waveform"].detach().float().cpu()
            if segment_audio["sample_rate"] != sr:
                import torchaudio
                shp = swav.shape
                swav = torchaudio.functional.resample(
                    swav.reshape(-1, shp[-1]), segment_audio["sample_rate"],
                    sr).reshape(shp[0], shp[1], -1)
            y = bw.clone()
            s0 = int(round(a / fps * sr))
            s1 = min(int(round((b + 1) / fps * sr)), y.shape[-1])
            need = s1 - s0
            seg_a = swav[..., :need]
            if seg_a.shape[-1] < need:   # pad with baseline tail if short
                pad = y[..., s0 + seg_a.shape[-1]:s1]
                seg_a = torch.cat([seg_a, pad], dim=-1)
            xf = int(round(max(fade_in, fade_out) / fps * sr))
            xf = min(xf, need // 2)
            mixed = seg_a.clone()
            if xf > 0:
                t = torch.linspace(0, math.pi / 2, xf)
                up, down = torch.sin(t) ** 2, torch.cos(t) ** 2
                mixed[..., :xf] = (y[..., s0:s0 + xf] * down +
                                   seg_a[..., :xf] * up)
                mixed[..., -xf:] = (y[..., s1 - xf:s1] * up.flip(0) +
                                    seg_a[..., -xf:] * down.flip(0))
            y[..., s0:s1] = mixed
            audio = {"waveform": y.contiguous(), "sample_rate": sr}
        # 32 kHz, not 48: H3 emits 32 kHz and every other audio path in this
        # file assumes it. A downstream node that trusts a wrong rate here
        # mis-times everything after it.
        return (out, audio if audio is not None else
                {"waveform": torch.zeros(1, 2, 1), "sample_rate": 32000})


class _AnyType(str):
    def __ne__(self, other):
        return False


ANY = _AnyType("*")


class H3ModeSwitch:
    """Lazy two-way switch: only the selected branch executes."""

    DESCRIPTION = (
        "Routes one of two inputs through, and only the selected branch "
        "executes (lazy evaluation), so a single workflow can carry both a "
        "fast turbo preview path and the full pipeline with a mode dropdown "
        "deciding which one actually runs. Wire any matching pair: VIDEO to "
        "VIDEO, IMAGE to IMAGE. Recommended use: mode 'preview' while "
        "iterating prompts and seeds, 'final' for the keeper.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "mode": (["preview", "final"], {"default": "preview"}),
        }, "optional": {
            "preview": (ANY, {"lazy": True}),
            "final": (ANY, {"lazy": True}),
        }}

    RETURN_TYPES = (ANY,)
    FUNCTION = "pick"
    CATEGORY = "utils/minimax"

    def check_lazy_status(self, mode, preview=None, final=None):
        return ["preview"] if mode == "preview" else ["final"]

    def pick(self, mode, preview=None, final=None):
        out = preview if mode == "preview" else final
        assert out is not None, f"selected branch '{mode}' is not wired"
        return (out,)


TIMESMEAR_CLASS_MAPPINGS = {
    "H3VideoFit": H3VideoFit,
    "H3JerkOracle": H3JerkOracle,
    "H3ManualHoldMap": H3ManualHoldMap,
    "H3TimeSmear": H3TimeSmear,
    "H3ExactRecover": H3ExactRecover,
    "H3V2VInit": H3V2VInit,
    "H3InjectSchedule": H3InjectSchedule,
    "H3JerkHeatmap": H3JerkHeatmap,
    "H3AudioRecover": H3AudioRecover,
    "H3ProbeSchedule": H3ProbeSchedule,
    "H3ExpertSchedule": H3ExpertSchedule,
    "H3TrajectoryBank": H3TrajectoryBank,
    "H3TrajectoryLoad": H3TrajectoryLoad,
    "H3MotionComposite": H3MotionComposite,
    "H3ModeSwitch": H3ModeSwitch,
    "H3MotionEditor": H3MotionEditor,
    "H3SegmentCrop": H3SegmentCrop,
    "H3SegmentSplice": H3SegmentSplice,
}
TIMESMEAR_DISPLAY_MAPPINGS = {
    "H3VideoFit": "H3 Video Fit (source clip -> 17k+5 frames) [alpha]",
    "H3JerkOracle": "H3 Jerk Oracle (profile / window / hold map)",
    "H3ManualHoldMap": "H3 Manual Hold Map (ranges to holds, gate) [alpha]",
    "H3TimeSmear": "H3 Time Smear (integer holds)",
    "H3ExactRecover": "H3 Exact Recover (24fps frame selection)",
    "H3V2VInit": "H3 V2V Init (nested AV latent)",
    "H3InjectSchedule": "H3 Inject Schedule (v2v sigmas, 0.70)",
    "H3JerkHeatmap": "H3 Jerk Heatmap (oracle overlay tile)",
    "H3AudioRecover": "H3 Audio Recover (hold-map atempo, pitch kept)",
    "H3ProbeSchedule": "H3 Probe Schedule (early-oracle head)",
    "H3ExpertSchedule": "H3 Expert Schedule (base head, turbo tail)",
    "H3TrajectoryBank": "H3 Trajectory Bank (checkpoint every step)",
    "H3TrajectoryLoad": "H3 Trajectory Load (branch from a step)",
    "H3MotionComposite": "H3 Motion Composite (subject regen, background baseline)",
    "H3ModeSwitch": "H3 Mode Switch (preview / final, lazy)",
    "H3MotionEditor": "H3 Motion Editor (timeline, masks, automation) [alpha]",
    "H3SegmentCrop": "H3 Segment Crop (regen only the window) [alpha]",
    "H3SegmentSplice": "H3 Segment Splice (crossfade reassembly) [alpha]",
}
