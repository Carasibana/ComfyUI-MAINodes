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
        "d_max so the whole burst sits at the plateau.")

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
        }}

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT", "STRING")
    RETURN_NAMES = ("hold_map", "segments", "window_start", "window_len", "profile")
    FUNCTION = "read"
    CATEGORY = "latent/minimax/motion"

    def read(self, samples, length, q, d_max, ramp, preset="custom", bridge=8):
        if preset in self.PRESETS:
            p = self.PRESETS[preset]
            q, d_max, ramp = p["q"], p["d_max"], p["ramp"]
        z = _video_component(samples)
        t_lat = z.shape[2]
        prof = _jerk_profile(z)

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
        return (hold_map, ",".join(segs), int(w0), int(wlen), profile)


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
            "s_per_step": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 120.0, "step": 0.05,
                           "tooltip": "seconds per step from a baseline render of this clip; 0 skips the minutes estimate"}),
            "est_steps": ("INT", {"default": 18, "min": 1, "max": 100,
                          "tooltip": "steps the regen pass will actually run (total_steps x inject)"}),
        }}

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("hold_map", "segments", "report")
    FUNCTION = "build"
    CATEGORY = "latent/minimax/motion"

    def build(self, length, fps, ranges, hold, ramp, bridge,
              oracle_hold_map="", s_per_step=0.0, est_steps=18):
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
        dilated = _legal_ceil(sum(holds))
        t_lat_d = (dilated - 5) // 17 * 5 + 2
        report = (f"{length}f ({length / fps:.1f}s) -> {dilated}f "
                  f"({dilated / fps:.1f}s) effective regen, "
                  f"{dilated / length:.2f}x; tokens {t_lat} -> {t_lat_d}")
        if s_per_step > 0:
            est = s_per_step * (t_lat_d / t_lat) * est_steps / 60
            report += (f"; ~{est:.1f} min at {s_per_step:g} s/step x "
                       f"{est_steps} steps")
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
        "records exactly what happened so recovery is lossless.")

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

    RETURN_TYPES = ("IMAGE", "STRING", "INT")
    RETURN_NAMES = ("images", "hold_map_used", "length")
    FUNCTION = "smear"
    CATEGORY = "image/minimax/motion"

    def smear(self, images, dilation, hold_map=""):
        images = images.detach().cpu()  # keep the (possibly huge) held batch off VRAM
        n = images.shape[0]
        holds = (json.loads(hold_map)["holds"] if hold_map.strip()
                 else [dilation] * n)
        assert len(holds) == n, f"hold map covers {len(holds)} frames, batch has {n}"
        target = _legal_ceil(sum(holds))
        holds = list(holds)
        holds[-1] += target - sum(holds)          # tail pad lives in the last hold
        idx = torch.tensor([i for i, h in enumerate(holds) for _ in range(h)])
        used = json.dumps({"holds": holds, "world_len": n})
        return (images[idx], used, int(target))


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
        "never pops) and feathered in pixel space BEFORE pooling to the "
        "latent grid, so edge cells carry fractional freeze strength: a "
        "smooth ramp, quantized to ~16 px cells, minimum one cell wide. "
        "Prefer this over the composite node when background and subject "
        "share lighting, shadows or water contact.")

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
            "mask": ("MASK", {"tooltip": "manual region to REGENERATE (1) vs freeze to baseline timing (0). "
                     "Overrides the oracle path. Union over time: the boundary never moves"}),
            "mask_feather": ("INT", {"default": 32, "min": 0, "max": 256,
                "tooltip": "feather width in image pixels; pooled to fractional latent cells (~16 px quanta, smooth ramp)"}),
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
            "mask": ("MASK", {"tooltip": "manual subject mask, overrides the oracle. 1 = keep regenerated, 0 = keep baseline. One mask = static boundary; a batch = per-frame"}),
            "invert_mask": ("BOOLEAN", {"default": False,
                            "tooltip": "on: the mask marks the KEEP-BASELINE region instead (lasso the birds directly)"}),
            "feather_profile": (["linear", "smoothstep", "gaussian"], {"default": "linear"}),
            "feather_direction": (["centered", "inward", "outward"], {"default": "centered",
                                  "tooltip": "where the ramp lives relative to the mask boundary"}),
            "mask_is_soft": ("BOOLEAN", {"default": False,
                             "tooltip": "mask values are final alphas (e.g. from H3 Motion Editor): skip threshold/grow/feather"}),
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
        report = (f"{n}f ({n / fps:.1f}s) -> {dilated}f ({dilated / fps:.1f}s) "
                  f"effective regen, {dilated / max(n, 1):.2f}x; "
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
}
TIMESMEAR_DISPLAY_MAPPINGS = {
    "H3JerkOracle": "H3 Jerk Oracle (profile / window / hold map)",
    "H3ManualHoldMap": "H3 Manual Hold Map (ranges to holds, gate)",
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
    "H3MotionEditor": "H3 Motion Editor (timeline, masks, automation)",
}
