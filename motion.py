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
        "a specific grid.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "samples": ("LATENT", {"tooltip": "video latent from VAEEncode of the smeared frames"}),
        }, "optional": {
            "length": ("INT", {"default": 0, "min": 0, "max": 3600,
                               "tooltip": "0 = derive from the latent (recommended); nonzero asserts this exact 17k+5 length"}),
        }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "build"
    CATEGORY = "latent/minimax/motion"

    def build(self, samples, length=0):
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
        return ({"samples": comfy.nested_tensor.NestedTensor((video, audio))},)


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
        "watch the burst light up as playback reaches it. Purely diagnostic/"
        "presentational; wire the same latent you'd give H3 Jerk Oracle. "
        "alpha 0.4-0.7 reads well; strip_height 0 hides the bar strip.")

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
        images = images.detach().float().cpu()  # --gpu-only hands us cuda tensors
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


TIMESMEAR_CLASS_MAPPINGS = {
    "H3JerkOracle": H3JerkOracle,
    "H3TimeSmear": H3TimeSmear,
    "H3ExactRecover": H3ExactRecover,
    "H3V2VInit": H3V2VInit,
    "H3InjectSchedule": H3InjectSchedule,
    "H3JerkHeatmap": H3JerkHeatmap,
    "H3AudioRecover": H3AudioRecover,
    "H3ProbeSchedule": H3ProbeSchedule,
    "H3ExpertSchedule": H3ExpertSchedule,
}
TIMESMEAR_DISPLAY_MAPPINGS = {
    "H3JerkOracle": "H3 Jerk Oracle (profile / window / hold map)",
    "H3TimeSmear": "H3 Time Smear (integer holds)",
    "H3ExactRecover": "H3 Exact Recover (24fps frame selection)",
    "H3V2VInit": "H3 V2V Init (nested AV latent)",
    "H3InjectSchedule": "H3 Inject Schedule (v2v sigmas, 0.70)",
    "H3JerkHeatmap": "H3 Jerk Heatmap (oracle overlay tile)",
    "H3AudioRecover": "H3 Audio Recover (hold-map atempo, pitch kept)",
    "H3ProbeSchedule": "H3 Probe Schedule (early-oracle head)",
    "H3ExpertSchedule": "H3 Expert Schedule (base head, turbo tail)",
}
