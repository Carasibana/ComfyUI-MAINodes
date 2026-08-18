"""H3 Window Loop — rolling-window regeneration as ONE node that iterates.

The "fat node" variant. It takes the whole regeneration chain's ingredients
(MODEL, CONDITIONING, VAE, audio VAE, SAMPLER, SIGMAS, seed) and runs the
chain once per window internally:

    plan -> [ crop -> pin -> smear -> (scale) -> encode -> v2v init ->
              sample -> decode -> exact recover -> audio recover -> trim ->
              splice ] x N -> world

It reimplements what SamplerCustomAdvanced does for you. That is the point of
this shape and also its price; see agent-window-fat.md in the ops repo for the
measured consequences (progress, interrupt, extensibility, VRAM).

Nothing here edits motion.py: every pipeline stage is the shipped class or
helper, imported and called. The only new logic is the PLANNER and the SEAM
POLICY, which are pure functions at the top of the file so they can be
exercised without a model.

Seam policy (shared with the other two window-driver variants so the
comparison is fair):
  * budget is counted in DILATED frames, not world frames;
  * window boundaries snap to cold cuts (hold == 1 on both sides) and to
    token boundaries;
  * a boundary never lands inside a ramp shoulder;
  * a hot cut (boundary forced inside a burst) gets pin-and-trim, never a
    crossfade: the next window regenerates an overlap that is seeded from the
    already-recovered world, uses it as context, then throws it away, leaving
    a hard cut at a pinned frame;
  * hot-cut handles inherit the world hold value, so both sides of the seam
    are repaired at the same temporal rate;
  * hot-cut audio is a hard cut placed at the local RMS minimum near the
    nominal boundary.
"""
import json
import logging
import math

import torch

try:                                    # normal import, as a package member
    from .motion import (COST_EXP, H3AudioRecover, H3ExactRecover,
                         H3SegmentSplice, H3TimeSmear, H3V2VInit, _legal_ceil,
                         _tok_start_frame, _token_count, _torchaudio)
except ImportError:                     # standalone (test harness) import
    from motion import (COST_EXP, H3AudioRecover, H3ExactRecover,
                        H3SegmentSplice, H3TimeSmear, H3V2VInit, _legal_ceil,
                        _tok_start_frame, _token_count, _torchaudio)

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# planner: pure, no torch, no comfy. Exercised offline in the report harness.
# --------------------------------------------------------------------------

def _bursts(holds):
    """Maximal runs of held (hold > 1) frames, as inclusive (start, end)."""
    out, s = [], None
    for f, h in enumerate(holds):
        if h > 1 and s is None:
            s = f
        elif h <= 1 and s is not None:
            out.append((s, f - 1))
            s = None
    if s is not None:
        out.append((s, len(holds) - 1))
    return out


def _token_starts(n):
    """Frame indices that begin a latent token, within a clip of n frames."""
    out, t = [], 0
    while True:
        f = _tok_start_frame(t)
        if f >= n:
            return out
        out.append(f)
        t += 1


def _seg_hold_list(holds, a, b, p, q, hot_in, hot_out):
    """Per-frame holds for the cropped window.

    Inside the held span the world holds are used verbatim. A COLD handle is
    forced to 1 (real-time baseline context, exactly what H3SegmentCrop
    ships). A HOT handle inherits the world hold value instead, so the frames
    either side of the seam are repaired at the same temporal rate — without
    this the window regenerates the seam region at real time, which is the
    roping the pass exists to remove.
    """
    out = []
    for f in range(a, b + 1):
        if p <= f <= q:
            out.append(holds[f])
        elif f < p and hot_in:
            out.append(holds[f])
        elif f > q and hot_out:
            out.append(holds[f])
        else:
            out.append(1)
    return out


def _grow_to_grid(holds, a, b, p, q, hot_in, hot_out, n):
    """Land sum(holds) exactly on the 17k+5 grid by growing a COLD handle.

    H3TimeSmear absorbs the grid remainder into the FINAL hold
    (`holds[-1] += target - sum(holds)`), which on an interior window parks a
    multi-frame freeze mid-clip. Handle frames are hold 1, so each extra
    handle frame contributes exactly 1 and the sum can be walked onto the
    grid instead. Growth only eats frames whose WORLD hold is 1, so a grown
    handle stays genuinely cold and never steals a neighbouring burst.
    _legal_ceil is unchanged by the growth (it only fills the pad), so the
    dilated cost does not move.
    """
    def total():
        return sum(_seg_hold_list(holds, a, b, p, q, hot_in, hot_out))

    need = _legal_ceil(total()) - total()
    while need > 0 and not hot_out and b + 1 <= n - 1 and holds[b + 1] == 1:
        b += 1
        need -= 1
    while need > 0 and not hot_in and a - 1 >= 0 and holds[a - 1] == 1:
        a -= 1
        need -= 1
    return a, b, need


def _window(holds, p, q, handle_frames, pin_frames, hot_in, hot_out):
    """Build one window record for held span [p, q]."""
    n = len(holds)
    in_w = pin_frames if hot_in else handle_frames
    out_w = pin_frames if hot_out else handle_frames
    a = max(0, p - in_w)
    b = min(n - 1, q + out_w)
    a, b, pad_left = _grow_to_grid(holds, a, b, p, q, hot_in, hot_out, n)
    seg = _seg_hold_list(holds, a, b, p, q, hot_in, hot_out)
    dil = _legal_ceil(sum(seg))
    return {
        "a": a, "b": b, "p": p, "q": q,
        "hot_in": hot_in, "hot_out": hot_out,
        "seg_holds": seg,
        "seg_len": b - a + 1,
        "dilated": dil,
        "tokens": _token_count(dil),
        "tail_pad": dil - sum(seg),      # frames the smear would freeze
        "trim_in": (p - a) if hot_in else 0,
        "trim_out": (b - q) if hot_out else 0,
    }


def _snap_cut(holds, lo, hi, n):
    """Pick a legal boundary frame in [lo, hi]; the next window starts at +1.

    Preference order, highest first: a COLD frame (hold 1 either side of the
    boundary), then a TOKEN boundary that is not inside a ramp shoulder, then
    any token boundary, then the raw index. Returns (cut, why).
    """
    starts = [f for f in _token_starts(n) if lo + 1 <= f <= hi + 1]
    for f in sorted(starts, reverse=True):        # cold + token, best case
        c = f - 1
        if holds[c] == 1 and holds[min(c + 1, n - 1)] == 1:
            return c, "cold cut on a token boundary"
    for c in range(hi, lo - 1, -1):               # cold, off the token grid
        if holds[c] == 1 and holds[min(c + 1, n - 1)] == 1:
            return c, "cold cut (no token boundary in the band)"
    for f in sorted(starts, reverse=True):        # hot, but on a plateau
        c = f - 1
        nxt = holds[min(c + 1, n - 1)]
        prv = holds[max(c - 1, 0)]
        if holds[c] == nxt and holds[c] == prv:
            return c, "HOT cut on a token boundary, on the hold plateau"
    if starts:
        return max(starts) - 1, ("HOT cut on a token boundary, INSIDE a ramp "
                                 "shoulder (no plateau boundary fit)")
    return hi, "HOT cut off the token grid (no token boundary fit the budget)"


def plan_windows(holds, budget_dilated, handle_frames=12, pin_frames=6):
    """Slice the world into windows whose DILATED length fits the budget.

    Returns (windows, notes). Greedy: pack as many whole bursts into one
    window as fit, and only split a burst (a hot cut) when the burst alone
    cannot fit.
    """
    n = len(holds)
    notes = []
    bursts = _bursts(holds)
    if not bursts:
        return [], ["hold map has no held frames: nothing to regenerate"]

    def fits(p, q, hot_in, hot_out):
        return _window(holds, p, q, handle_frames, pin_frames,
                       hot_in, hot_out)["dilated"] <= budget_dilated

    windows, i, hot_in = [], 0, False
    guard = 0
    while i < len(bursts):
        guard += 1
        if guard > 4 * n + 16:
            notes.append("planner guard tripped; emitting what exists")
            break
        p, bq = bursts[i]
        if not fits(p, bq, hot_in, False):
            # This burst alone busts the budget: find the furthest end that
            # fits, then snap that boundary to a legal cut.
            e = p
            while e + 1 <= bq and fits(p, e + 1, hot_in, True):
                e += 1
            lo = p + max(1, (e - p) // 4)         # keep the cut meaningful
            cut, why = _snap_cut(holds, lo, e, n)
            cut = max(p, min(cut, bq - 1))
            w = _window(holds, p, cut, handle_frames, pin_frames, hot_in, True)
            w["cut_note"] = why
            if w["dilated"] > budget_dilated:
                notes.append(
                    f"window {len(windows)}: snapped cut at f{cut} costs "
                    f"{w['dilated']} dilated frames, over the {budget_dilated} "
                    f"budget; the nearest legal boundary is the binding "
                    f"constraint, not the budget")
            windows.append(w)
            bursts[i] = (cut + 1, bq)
            hot_in = True
            continue
        j = i
        while j + 1 < len(bursts) and fits(p, bursts[j + 1][1], hot_in, False):
            j += 1
        windows.append(_window(holds, p, bursts[j][1], handle_frames,
                               pin_frames, hot_in, False))
        i, hot_in = j + 1, False

    for k, w in enumerate(windows):
        w["k"] = k
        if w["tail_pad"] and w["hot_out"]:
            # The pad rides on the LAST hold, i.e. on world frame b, and at a
            # hot-out edge frame b is inside the pin overlap that gets
            # trimmed. So pin-and-trim eats the freeze for free.
            notes.append(
                f"window {k}: {w['tail_pad']} frame(s) of grid pad freeze the "
                f"last frame, but that frame is inside the trimmed pin "
                f"overlap, so the freeze is discarded")
        elif w["tail_pad"]:
            notes.append(
                f"window {k}: {w['tail_pad']} frame(s) of grid pad could not "
                f"be absorbed by a cold handle; H3 Time Smear will freeze the "
                f"last frame that many times")
    hot = sum(1 for w in windows if w["hot_in"] or w["hot_out"])
    if hot:
        notes.append(f"{hot} window(s) carry a HOT edge: the boundary sits "
                     f"inside a burst, handled by pin-and-trim (no crossfade)")
    return windows, notes


def plan_report(windows, notes, world_len, holds, budget_dilated, fps=24,
                s_per_step=0.0, est_steps=18):
    """The price tag, same shape as H3ManualHoldMap's report."""
    if not windows:
        return "no windows: " + ("; ".join(notes) if notes else "empty plan")
    full = _legal_ceil(sum(holds))
    t_full = _token_count(full)
    lines = [f"{len(windows)} window(s) over {world_len}f "
             f"({world_len / max(fps, 1):.1f}s); one-shot would be {full}f "
             f"dilated / {t_full} tokens"]
    one_shot = t_full ** COST_EXP
    split = 0.0
    for w in windows:
        split += w["tokens"] ** COST_EXP
        edges = ("hot" if w["hot_in"] else "cold") + "/" + \
                ("hot" if w["hot_out"] else "cold")
        lines.append(
            f"  w{w['k']}: world f{w['a']}-f{w['b']} ({w['seg_len']}f, held "
            f"f{w['p']}-f{w['q']}) -> {w['dilated']}f dilated / {w['tokens']} "
            f"tokens, edges {edges}"
            + (f", trim {w['trim_in']}/{w['trim_out']}"
               if w["trim_in"] or w["trim_out"] else "")
            + (f" [{w['cut_note']}]" if w.get("cut_note") else ""))
    peak = max(w["tokens"] for w in windows)
    lines.append(
        f"  peak window {peak} tokens vs {t_full} one-shot "
        f"({(t_full / peak) ** COST_EXP:.2f}x less per-step work at the peak); "
        f"total sampling work {split / one_shot:.2f}x the one-shot pass "
        f"(sum tokens^{COST_EXP}, handles counted)")
    if s_per_step > 0:
        secs = sum((w["tokens"] ** COST_EXP) for w in windows) / \
            (_token_count(world_len) ** COST_EXP) * \
            s_per_step * max(1, int(est_steps))
        lines.append(f"  roughly {secs / 60:.0f} min of sampling at "
                     f"{s_per_step:g} s/step x {int(est_steps)} steps scaled "
                     f"from the world-length step cost")
    for note in notes:
        lines.append(f"  note: {note}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# audio seam: asymmetric fades, and a hard cut at the local RMS minimum
# --------------------------------------------------------------------------

def _rms_min_shift(base, seg, centre, search):
    """Sample offset within +/- search that minimises local energy.

    A hot audio boundary is a hard cut between two independent foley takes of
    the same moment. Crossfading doubles or smears a percussive hit at any
    width, so the policy is a hard cut placed where both tracks are quietest.
    """
    if search <= 0:
        return 0
    win = max(64, search // 4)
    best, best_v = 0, None
    for d in range(-search, search + 1, max(1, win // 8)):
        c = centre + d
        if c - win < 0 or c + win > min(base.shape[-1], seg.shape[-1]):
            continue
        v = float(base[..., c - win:c + win].pow(2).mean() +
                  seg[..., c - win:c + win].pow(2).mean())
        if best_v is None or v < best_v:
            best, best_v = d, v
    return best


def _splice_audio(base, seg, a, b, fps, xf_in, xf_out, rms_search=0):
    """Write seg over base for world frames [a, b].

    xf_in / xf_out are equal-power crossfade widths in FRAMES; 0 means a hard
    cut, and a hard cut gets snapped to the local RMS minimum. Written here
    rather than reused from H3SegmentSplice because that node applies
    max(fade_in, fade_out) to BOTH ends, which would put a crossfade on a hot
    seam whenever the other end of the same window is cold.
    """
    sr = base["sample_rate"]
    y = base["waveform"].detach().float().cpu().clone()
    s = seg["waveform"].detach().float().cpu()
    if seg["sample_rate"] != sr:
        torchaudio = _torchaudio("resample")
        shp = s.shape
        s = torchaudio.functional.resample(
            s.reshape(-1, shp[-1]), seg["sample_rate"], sr).reshape(
                shp[0], shp[1], -1)
    if s.shape[1] != y.shape[1]:
        s = s[:, :1].expand(-1, y.shape[1], -1)
    s0 = int(round(a / fps * sr))
    s1 = min(int(round((b + 1) / fps * sr)), y.shape[-1])
    need = s1 - s0
    if need <= 0:
        return base
    part = s[..., :need]
    if part.shape[-1] < need:
        part = torch.cat([part, y[..., s0 + part.shape[-1]:s1]], dim=-1)
    part = part.clone()
    for side, xf in (("in", xf_in), ("out", xf_out)):
        w = int(round(max(0, xf) / fps * sr))
        w = min(w, need // 2)
        if w > 0:
            t = torch.linspace(0, math.pi / 2, w)
            up, down = torch.sin(t) ** 2, torch.cos(t) ** 2
            if side == "in":
                part[..., :w] = y[..., s0:s0 + w] * down + part[..., :w] * up
            else:
                part[..., -w:] = (y[..., s1 - w:s1] * up.flip(0) +
                                  part[..., -w:] * down.flip(0))
        elif rms_search > 0:
            centre = 0 if side == "in" else need
            d = _rms_min_shift(y[..., s0:s1], part, centre, rms_search)
            if side == "in" and d > 0:
                part[..., :d] = y[..., s0:s0 + d]
            elif side == "out" and d < 0:
                part[..., d:] = y[..., s1 + d:s1]
    y[..., s0:s1] = part
    return {"waveform": y.contiguous(), "sample_rate": sr}


def _silence(world_len, fps, sr, channels=2):
    n = int(round(world_len / max(fps, 1) * sr))
    return {"waveform": torch.zeros(1, channels, max(n, 1)), "sample_rate": sr}


# --------------------------------------------------------------------------
# the node
# --------------------------------------------------------------------------

class H3WindowLoop:
    """Rolling-window regeneration in one node: plan, then run the whole
    chain per window and splice the world back together."""

    DESCRIPTION = (
        "EXPERIMENTAL (alpha), new 2026-08-12.\n\n"
        "Regenerates a clip in rolling windows so the peak dilated span "
        "never exceeds a budget you set, then splices the world back "
        "together. One node runs the whole per-window chain internally "
        "(time smear, VAE encode, v2v init, sample, decode, exact recover, "
        "audio recover, splice), so it needs the sampler's ingredients as "
        "inputs: MODEL + positive CONDITIONING (or a GUIDER), video VAE, "
        "audio VAE, SAMPLER, SIGMAS and a seed.\n\n"
        "budget_dilated is the knob and it is counted in DILATED frames, "
        "not world frames: that is the number that sets both the bill and "
        "the VRAM peak. Run with plan_only ON first — the report prices "
        "every window before you pay for any of them.\n\n"
        "SEAMS. A boundary between bursts is a COLD cut (hold 1 on both "
        "sides): handles are real-time baseline context and the splice "
        "crossfades inside them, which is the shipped single-window "
        "behaviour N times over. A boundary forced INSIDE a burst is a HOT "
        "cut, and gets pin-and-trim instead: the next window regenerates an "
        "overlap seeded from the already-recovered world, uses it as "
        "context, then discards it, leaving a hard cut at a pinned frame. "
        "Hot handles inherit the world hold value so both sides of the seam "
        "are repaired at the same rate, and hot audio is a hard cut at the "
        "local RMS minimum, never a crossfade (two independent foley takes "
        "of one moment cannot be dissolved).\n\n"
        "What rides along and what does not: MODEL-domain user nodes "
        "(attention patches, chunked feed-forward, LoRAs) and SAMPLER "
        "wrappers apply because you wire them upstream of this node. "
        "IMAGE-domain nodes inside a loop body cannot; the common one, an "
        "ImageScale between smear and encode, is replicated by the "
        "draft_width/draft_height widgets. Anything else you wanted in the "
        "loop is not reachable from here.\n\n"
        "Progress is one bar across every window's every step, so the node "
        "reports N x steps and previews as usual. Interrupt lands at the "
        "next sampler step, not at the next window; set on_interrupt to "
        "'return partial world' to keep the windows that already landed.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL", {"tooltip": "patched however you like: LoRAs, attention patches and chunked feed-forward all ride along"}),
            "positive": ("CONDITIONING",),
            "vae": ("VAE", {"tooltip": "video VAE"}),
            "audio_vae": ("VAE", {"tooltip": "audio VAE, for the jointly generated foley"}),
            "sampler": ("SAMPLER",),
            "sigmas": ("SIGMAS", {"tooltip": "from H3 Inject Schedule; the whole point is that these are the INJECTED sigmas"}),
            "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                           "control_after_generate": True}),
            "images": ("IMAGE", {"tooltip": "baseline frames, world clock"}),
            "hold_map": ("STRING", {"default": "", "forceInput": True,
                         "tooltip": "from the oracle, H3 Manual Hold Map or H3 Motion Editor"}),
            "budget_dilated": ("INT", {"default": 124, "min": 39, "max": 3600,
                               "tooltip": "max DILATED frames per window; the VRAM and cost dial"}),
        }, "optional": {
            "guider": ("GUIDER", {"tooltip": "overrides model+positive; wire your own guider to keep control of conditioning"}),
            "baseline_audio": ("AUDIO", {"tooltip": "the baseline clip's track; unwired, the world starts as silence"}),
            "handle_frames": ("INT", {"default": 12, "min": 2, "max": 48,
                              "tooltip": "real-time context frames on each COLD edge"}),
            "pin_frames": ("INT", {"default": 6, "min": 2, "max": 48,
                           "tooltip": "overlap regenerated then discarded at each HOT edge (pin-and-trim)"}),
            "feather_frames": ("INT", {"default": 6, "min": 0, "max": 24,
                               "tooltip": "crossfade width inside a COLD handle; hot edges ignore it by policy"}),
            "fps": ("INT", {"default": 24, "min": 1, "max": 120}),
            "seed_stride": ("INT", {"default": 1, "min": 0, "max": 1000,
                            "tooltip": "seed offset per window; 0 = every window on the same seed"}),
            "draft_width": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8,
                            "tooltip": "0 = off. Replicates the ImageScale that community loop bodies put between smear and encode"}),
            "draft_height": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8}),
            "upscale_method": (["lanczos", "bicubic", "bilinear", "area", "nearest-exact"],
                               {"default": "lanczos"}),
            "on_interrupt": (["raise (stock behaviour)", "return partial world"],
                             {"default": "raise (stock behaviour)"}),
            "plan_only": ("BOOLEAN", {"default": False,
                          "tooltip": "price the plan and stop; images pass through untouched"}),
            "audio_rms_search_ms": ("FLOAT", {"default": 12.0, "min": 0.0, "max": 200.0, "step": 1.0,
                                    "tooltip": "how far a hot audio cut may slide to land on a local RMS minimum"}),
            "s_per_step": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 120.0, "step": 0.05}),
        }}

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("images", "audio", "plan", "report")
    FUNCTION = "run"
    CATEGORY = "image/minimax/motion"

    # ---- planning ---------------------------------------------------------

    def _plan(self, holds, budget, handle_frames, pin_frames):
        return plan_windows(holds, budget, handle_frames, pin_frames)

    # ---- one window -------------------------------------------------------

    def _one_window(self, w, world, world_audio, ctx):
        """Crop from the RUNNING world (that is the pin), regenerate, recover,
        trim, splice. Returns the new (world, world_audio)."""
        import comfy.model_management
        import comfy.utils
        from comfy_extras.nodes_audio import vae_decode_audio

        comfy.model_management.throw_exception_if_processing_interrupted()
        a, b = w["a"], w["b"]
        seg = world[a:b + 1]
        seg_map = json.dumps({"holds": w["seg_holds"], "world_len": w["seg_len"]})

        smeared, used_map, length, _rep = H3TimeSmear().smear(
            seg, 4, hold_map=seg_map)
        if ctx["draft"] is not None:
            dw, dh = ctx["draft"]
            smeared = comfy.utils.common_upscale(
                smeared.movedim(-1, 1), dw, dh, ctx["upscale_method"],
                "disabled").movedim(1, -1)

        comfy.model_management.throw_exception_if_processing_interrupted()
        latent = {"samples": ctx["vae"].encode(smeared[:, :, :, :3])}
        init = H3V2VInit().build(latent, length=0)[0]

        noise = _Noise(ctx["seed"] + w["k"] * ctx["seed_stride"])
        samples = ctx["guider"].sample(
            noise.generate_noise(init), init["samples"], ctx["sampler"],
            ctx["sigmas"], denoise_mask=init.get("noise_mask"),
            callback=ctx["make_cb"](w["k"]), disable_pbar=True,
            seed=noise.seed)
        out = {"samples": samples.to(comfy.model_management.intermediate_device())}

        comfy.model_management.throw_exception_if_processing_interrupted()
        video = out["samples"]
        if video.is_nested:
            video = video.unbind()[0]
        frames = ctx["vae"].decode(video)
        if len(frames.shape) == 5:
            frames = frames.reshape(-1, *frames.shape[-3:])
        rec = H3ExactRecover().recover(frames, used_map)[0]
        seg_audio = None
        if ctx["audio_vae"] is not None:
            raw = vae_decode_audio(ctx["audio_vae"], out)
            seg_audio = H3AudioRecover().recover(raw, used_map, ctx["fps"])[0]

        # ---- trim the pinned overlap away, then splice --------------------
        ti, to = w["trim_in"], w["trim_out"]
        if ti or to:
            rec = rec[ti:rec.shape[0] - to if to else rec.shape[0]]
            if seg_audio is not None:
                sr = seg_audio["sample_rate"]
                s0 = int(round(ti / ctx["fps"] * sr))
                s1 = seg_audio["waveform"].shape[-1] - \
                    int(round(to / ctx["fps"] * sr))
                seg_audio = {"waveform": seg_audio["waveform"][..., s0:s1],
                             "sample_rate": sr}
        start, end = a + ti, b - to
        splice = json.dumps({
            "start": start, "end": end, "world_len": world.shape[0],
            "handle_in": 0 if w["hot_in"] else (w["p"] - a),
            "handle_out": 0 if w["hot_out"] else (b - w["q"])})
        if rec.shape[1:] != world.shape[1:]:
            rec = comfy.utils.common_upscale(
                rec.movedim(-1, 1), world.shape[2], world.shape[1],
                ctx["upscale_method"], "disabled").movedim(1, -1)
        world = H3SegmentSplice().splice(world, rec, splice,
                                         ctx["feather_frames"])[0]
        if seg_audio is not None and world_audio is not None:
            world_audio = _splice_audio(
                world_audio, seg_audio, start, end, ctx["fps"],
                0 if w["hot_in"] else ctx["feather_frames"],
                0 if w["hot_out"] else ctx["feather_frames"],
                ctx["rms_search"])
        return world, world_audio

    # ---- driver -----------------------------------------------------------

    def run(self, model, positive, vae, audio_vae, sampler, sigmas, noise_seed,
            images, hold_map, budget_dilated, guider=None, baseline_audio=None,
            handle_frames=12, pin_frames=6, feather_frames=6, fps=24,
            seed_stride=1, draft_width=0, draft_height=0,
            upscale_method="lanczos", on_interrupt="raise (stock behaviour)",
            plan_only=False, audio_rms_search_ms=12.0, s_per_step=0.0):
        import comfy.model_management
        import comfy.utils
        import latent_preview

        images = images.detach().cpu()
        holds = json.loads(hold_map)["holds"]
        n = images.shape[0]
        assert len(holds) == n, (
            f"hold map covers {len(holds)} frames, clip has {n}")

        windows, notes = self._plan(holds, int(budget_dilated),
                                    int(handle_frames), int(pin_frames))
        plan = plan_report(windows, notes, n, holds, int(budget_dilated), fps,
                           s_per_step, max(1, sigmas.shape[-1] - 1))
        if plan_only or not windows:
            return (images, baseline_audio or _silence(n, fps, 32000),
                    plan, "plan_only: nothing was generated" if plan_only
                    else "no windows to run")

        if guider is None:
            from comfy_extras.nodes_custom_sampler import Guider_Basic
            guider = Guider_Basic(model)
            guider.set_conds(positive)

        steps = max(1, sigmas.shape[-1] - 1)
        total = steps * len(windows)
        pbar = comfy.utils.ProgressBar(total)
        previewer = latent_preview.get_previewer(
            guider.model_patcher.load_device,
            guider.model_patcher.model.latent_format)

        def make_cb(k):
            def cb(step, x0, x, total_steps):
                preview = None
                if previewer is not None:
                    z = x0.tensors[0] if getattr(x0, "is_nested", False) else x0
                    preview = previewer.decode_latent_to_preview_image("JPEG", z)
                # ONE bar across the whole run: the hook resolves node_id from
                # the executing context, so this lands on this node, and the
                # hook itself raises InterruptProcessingException, which is
                # what gives per-STEP interrupt granularity.
                pbar.update_absolute(k * steps + step + 1, total, preview)
            return cb

        ctx = {"vae": vae, "audio_vae": audio_vae, "sampler": sampler,
               "sigmas": sigmas, "guider": guider, "seed": int(noise_seed),
               "seed_stride": int(seed_stride), "fps": int(fps),
               "feather_frames": int(feather_frames), "make_cb": make_cb,
               "upscale_method": upscale_method,
               "draft": ((int(draft_width), int(draft_height))
                         if draft_width and draft_height else None),
               "rms_search": int(round(audio_rms_search_ms / 1000.0 * 32000))}

        world = images.clone()
        world_audio = baseline_audio
        done, lines = 0, []
        try:
            for w in windows:
                log.info("H3WindowLoop: window %d/%d, world f%d-f%d, %d "
                         "dilated frames (%d tokens)", w["k"] + 1,
                         len(windows), w["a"], w["b"], w["dilated"],
                         w["tokens"])
                if world_audio is None and audio_vae is not None:
                    world_audio = _silence(n, fps, 32000)
                world, world_audio = self._one_window(w, world, world_audio, ctx)
                done += 1
                lines.append(f"w{w['k']} done: f{w['a']}-f{w['b']}, "
                             f"{w['dilated']}f dilated, {w['tokens']} tokens")
        except comfy.model_management.InterruptProcessingException:
            if not on_interrupt.startswith("return"):
                raise
            lines.append(f"INTERRUPTED after {done} of {len(windows)} "
                         f"windows; the world carries the finished ones and "
                         f"baseline everywhere else")
        report = (f"{done} of {len(windows)} window(s) run, {steps} steps "
                  f"each ({done * steps} of {total} sampler steps)\n"
                  + "\n".join(lines))
        return (world, world_audio or _silence(n, fps, 32000), plan, report)


class _Noise:
    """Same contract as nodes_custom_sampler.Noise_RandomNoise, inlined so the
    loop does not depend on that module's import surface at call time."""

    def __init__(self, seed):
        self.seed = int(seed) & 0xffffffffffffffff

    def generate_noise(self, input_latent):
        import comfy.sample
        return comfy.sample.prepare_noise(
            input_latent["samples"], self.seed,
            input_latent.get("batch_index"))


NODE_CLASS_MAPPINGS = {"H3WindowLoop": H3WindowLoop}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3WindowLoop": "H3 Window Loop (rolling regen, one node) [alpha]"}
