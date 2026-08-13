"""Rolling-window regeneration driven by ComfyUI's node-expansion API.

One node (H3 Window Expand) reads a hold map, plans N windows against a
dilated-frame budget, and returns an EXPANSION: a graph containing the whole
regeneration chain once per window, wired into a chained splice. The user
presses queue once. The sampler stays a real SamplerCustomAdvanced node in
the expanded graph, so per-step progress, previews and interrupt behave the
way they do anywhere else.

Window k's body is the shipped segment recipe, unchanged:

  H3WindowCrop -> H3TimeSmear -> VAEEncode -> H3V2VInit
               -> SamplerCustomAdvanced -> VAEDecode -> H3ExactRecover
               (audio: VAEDecodeAudio -> H3AudioRecover)
               -> H3WindowSeam -> H3SegmentSplice(baseline = window k-1)

H3WindowCrop and H3WindowSeam exist because the planner, not the crop node,
owns the per-frame holds and the per-side seam policy. They are placed by
H3 Window Expand; you can wire them by hand but there is no reason to.

Seam policy (shared with the other rolling-window drivers so the three can
be compared): the budget is counted in DILATED frames; window boundaries
snap to cold cuts (hold == 1 on both sides) and, failing that, to token
boundaries; a boundary is never placed inside a ramp shoulder; a hot cut
(a boundary forced inside a burst) is served by pin-and-trim, not by a
crossfade.

Nothing here edits motion.py. It imports from it.
"""
import json

import torch

from .motion import COST_EXP, _legal_ceil, _tok_start_frame, _token_count


# ---------------------------------------------------------------------------
# planner
# ---------------------------------------------------------------------------

def _bursts(holds):
    """Maximal runs of held frames as inclusive (first, last) pairs."""
    out, start = [], None
    for f, h in enumerate(holds):
        if h > 1 and start is None:
            start = f
        elif h <= 1 and start is not None:
            out.append((start, f - 1))
            start = None
    if start is not None:
        out.append((start, len(holds) - 1))
    return out


def _plateaus(holds):
    """Per-frame plateau value: the maximum hold of the burst a frame is in,
    or 1 outside every burst. A frame whose hold is below its plateau is on a
    ramp shoulder, and the policy refuses to cut there: the two sides would
    disagree about that frame's hold and the same world frame would be
    regenerated at two different rates."""
    plat = [1] * len(holds)
    for a, b in _bursts(holds):
        m = max(holds[a:b + 1])
        for f in range(a, b + 1):
            plat[f] = m
    return plat


def _tok_starts(limit):
    """Dilated-frame indices that begin a latent token, up to `limit`."""
    out, t = set(), 0
    while True:
        f = _tok_start_frame(t)
        if f > limit:
            return out
        out.add(f)
        t += 1


def _seg_holds(holds, a, b, body_a, body_b, hot_in, hot_out):
    """Holds for the cropped segment [a..b].

    Body frames keep their world hold. A handle frame is hold 1 on a COLD
    side (it is real-time baseline context, which is what the shipped
    single-window path does) and INHERITS the world hold on a HOT side, so
    both sides of a mid-burst seam get repaired at the same temporal rate.
    """
    out = []
    for f in range(a, b + 1):
        if body_a <= f <= body_b:
            out.append(holds[f])
        elif f < body_a and hot_in:
            out.append(holds[f])
        elif f > body_b and hot_out:
            out.append(holds[f])
        else:
            out.append(1)
    return out


def _grow_to_grid(holds, a, b, body_a, body_b, hot_in, hot_out, n):
    """Land sum(seg_holds) exactly on the 17k+5 grid by growing a hold-1
    handle rather than letting H3TimeSmear absorb the remainder into the
    final hold. A hold-1 frame adds exactly 1, so the deficit is spendable
    one frame at a time. Falls back to today's tail pad when the clip runs
    out of room on both sides."""
    for _ in range(64):
        sh = _seg_holds(holds, a, b, body_a, body_b, hot_in, hot_out)
        need = _legal_ceil(sum(sh)) - sum(sh)
        if need <= 0:
            return a, b
        if b < n - 1 and not hot_out:
            b += 1
        elif a > 0 and not hot_in:
            a -= 1
        else:
            return a, b
    return a, b


def _window(holds, body_a, body_b, handle, n, hot_in, hot_out, grid_grow=True):
    """Build one window record from its body span. A handle exists on both
    kinds of seam; the difference is that a hot handle inherits the world
    hold instead of being pinned to hold 1."""
    a = max(0, body_a - handle)
    b = min(n - 1, body_b + handle)
    if grid_grow:
        a, b = _grow_to_grid(holds, a, b, body_a, body_b, hot_in, hot_out, n)
    sh = _seg_holds(holds, a, b, body_a, body_b, hot_in, hot_out)
    dil = _legal_ceil(sum(sh))
    return {"a": a, "b": b, "body_a": body_a, "body_b": body_b,
            "holds": sh, "dilated": dil, "tokens": _token_count(dil),
            "hot_in": hot_in, "hot_out": hot_out,
            "handle_in": body_a - a, "handle_out": b - body_b}


def plan_windows(holds, budget, handle=12, snap_search=8, max_windows=12,
                 grid_grow=True):
    """Greedy plan: the largest span of held material whose dilated length
    fits the budget, cut at the coldest legal boundary available.

    Returns (windows, notes). A window dict carries its world span, its
    per-frame segment holds, its dilated length and its seam types.
    """
    n = len(holds)
    notes = []
    bursts = _bursts(holds)
    if not bursts:
        return [], ["hold map has no held frames; nothing to regenerate"]
    plat = _plateaus(holds)
    toks = _tok_starts(sum(holds) + 64)
    cum = [0] * (n + 1)                       # dilated position of world frame
    for f in range(n):
        cum[f + 1] = cum[f] + holds[f]

    # every legal place a window may END (inclusive), in world frames
    cold_ends = {b for _, b in bursts}
    hot_ends = set()
    for a, b in bursts:
        for f in range(a, b):
            # cut between f and f+1: both sides must be at the plateau, so
            # neither side sits on a ramp shoulder
            if holds[f] == plat[f] and holds[f + 1] == plat[f + 1]:
                hot_ends.add(f)

    burst_starts = {a for a, _ in bursts}
    windows, cursor, guard = [], bursts[0][0], 0
    while cursor <= bursts[-1][1] and guard < max_windows:
        guard += 1
        body_a = cursor
        hot_in = body_a not in burst_starts   # we cut into a burst last time
        cands = sorted(e for e in (cold_ends | hot_ends) if e >= body_a)
        fits_cold, fits_hot = [], []
        for e in cands:
            hot_out = e not in cold_ends
            w = _window(holds, body_a, e, handle, n, hot_in, hot_out,
                        grid_grow)
            if w["dilated"] <= budget:
                (fits_hot if hot_out else fits_cold).append((e, w))
            elif fits_cold or fits_hot:
                break                          # cost is monotone in e
        if fits_cold:
            e, w = fits_cold[-1]
        elif fits_hot:
            e, w = fits_hot[-1]
            snapped = None
            # token snap: look back a few frames for a boundary that lands on
            # a token start in the dilated timeline
            for e2, w2 in reversed(fits_hot):
                if e - e2 > snap_search:
                    break
                if cum[e2 + 1] in toks:
                    snapped, e, w = (e - e2), e2, w2
                    break
            notes.append(f"window {len(windows)}: NO cold cut fits the "
                         f"{budget}f budget, forced a hot cut after f{e} "
                         f"(inside a burst)" +
                         (f", snapped back {snapped}f onto the token "
                          f"boundary at dilated f{cum[e + 1]}"
                          if snapped is not None else
                          ", no token boundary within snap_search"))
        else:
            e = cands[0] if cands else bursts[-1][1]
            hot_out = e not in cold_ends
            w = _window(holds, body_a, e, handle, n, hot_in, hot_out,
                        grid_grow)
            notes.append(f"window {len(windows)}: budget {budget}f is below "
                         f"the smallest legal window ({w['dilated']}f); "
                         f"emitting it anyway")
        w["k"] = len(windows)
        windows.append(w)
        cursor = e + 1
        while cursor < n and holds[cursor] <= 1:
            cursor += 1                        # skip to the next held frame
        if cursor >= n:
            break
    if guard >= max_windows and cursor <= bursts[-1][1]:
        notes.append(f"stopped at max_windows={max_windows}; frames "
                     f"f{cursor}-f{bursts[-1][1]} are NOT covered")

    for i, w in enumerate(windows):
        w["seam_in"] = "hot" if w["hot_in"] else ("edge" if i == 0 else "cold")
        w["seam_out"] = "hot" if w["hot_out"] else (
            "edge" if i == len(windows) - 1 else "cold")
    return windows, notes


def plan_report(windows, notes, holds, budget, fps=24, s_per_step=0.0,
                est_steps=18):
    """The price tag, in the shape H3ManualHoldMap uses."""
    n = len(holds)
    full = _legal_ceil(sum(holds))
    t_full = _token_count(full)
    lines = [f"world {n}f, single-pass dilated {full}f ({t_full} tokens); "
             f"budget {budget} dilated f/window -> {len(windows)} window(s)"]
    tot, cov = 0.0, 0
    for w in windows:
        rel = (w["tokens"] / t_full) ** COST_EXP
        tot += rel
        cov += w["body_b"] - w["body_a"] + 1
        pad = w["dilated"] - sum(w["holds"])
        lines.append(
            f"  w{w['k']}: world f{w['a']}-f{w['b']} "
            f"(body f{w['body_a']}-f{w['body_b']}), dilated {w['dilated']}f "
            f"/ {w['tokens']} tok, {rel:.2f}x one full pass, seams "
            f"{w['seam_in']}/{w['seam_out']}" +
            (f", TAIL PAD {pad}f (both handles hot, no hold-1 frame left to "
             f"grow; H3 Time Smear freezes the last frame)" if pad else ""))
    lines.append(f"  total {tot:.2f}x one full pass per step "
                 f"({abs(1 - tot) * 100:.0f}% "
                 f"{'cheaper' if tot < 1 else 'MORE EXPENSIVE'} than one "
                 f"window), "
                 f"{cov}/{sum(1 for h in holds if h > 1)} held frames covered")
    dup = sum(w["b"] - w["a"] + 1 for w in windows) - cov
    lines.append(f"  {dup} handle frames regenerated as context "
                 f"({dup / max(1, sum(w['b'] - w['a'] + 1 for w in windows)) * 100:.0f}% "
                 f"of generated frames are duplication)")
    if s_per_step > 0:
        lines.append(f"  roughly {tot * s_per_step * est_steps / 60:.0f} min "
                     f"of sampling at {s_per_step:g} s/step x {est_steps} "
                     f"steps for a full pass")
    lines.append(f"  RAM: the chained splice holds {len(windows)} full-world "
                 f"image copies in the cache at once")
    lines += ["  " + s for s in notes]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# body nodes placed inside the expansion
# ---------------------------------------------------------------------------

# What H3WindowCrop is told about a window. Deliberately NOT the fade widths
# or the trim: those live on H3WindowSeam, so retuning a seam does not change
# the cache signature of anything that had to be sampled.
CROP_SPEC_KEYS = ("k", "a", "b", "world_len", "holds", "handle_in",
                  "handle_out", "seam_in", "seam_out")

class H3WindowCrop:
    """Cut one planned window out of the world. Placed by H3 Window Expand."""

    DESCRIPTION = (
        "EXPERIMENTAL (alpha), new 2026-08-12; the classic pipeline nodes "
        "are unchanged.\n\n"
        "Internal to H3 Window Expand: crops the world to one planned "
        "window and emits that window's exact per-frame hold map plus a "
        "splice map in world coordinates. Unlike H3 Segment Crop it does "
        "not derive the window from the hold map, because the planner has "
        "already chosen the span, the per-side handle widths and whether "
        "each handle is real-time (cold seam) or hold-inheriting (hot "
        "seam). Wire it by hand only if you are debugging a plan.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE", {"tooltip": "world frames (or the running splice)"}),
            "window_spec": ("STRING", {"default": "", "multiline": True,
                            "tooltip": "JSON from H3 Window Expand's planner"}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("images", "hold_map", "splice_map", "first_frame",
                    "last_frame", "report")
    FUNCTION = "crop"
    CATEGORY = "image/minimax/motion"

    def crop(self, images, window_spec):
        images = images.detach().cpu()
        w = json.loads(window_spec)
        a, b, holds = w["a"], w["b"], w["holds"]
        n = int(images.shape[0])
        assert w["world_len"] <= n, (w["world_len"], n)
        assert len(holds) == b - a + 1, (len(holds), b - a + 1)
        seg = images[a:b + 1]
        # the fade widths are NOT here on purpose: H3 Window Seam writes
        # them, so retuning feather_frames invalidates the seam and the
        # splice and leaves the sampled windows in cache
        splice = json.dumps({"start": a, "end": b, "world_len": w["world_len"],
                             "handle_in": w["handle_in"],
                             "handle_out": w["handle_out"]})
        dil = _legal_ceil(sum(holds))
        report = (f"w{w['k']}: f{a}-f{b} ({b - a + 1}f of {w['world_len']}f), "
                  f"dilated {dil}f / {_token_count(dil)} tok, seams "
                  f"{w['seam_in']}/{w['seam_out']}")
        return (seg, json.dumps({"holds": holds, "world_len": b - a + 1}),
                splice, seg[:1].clone(), seg[-1:].clone(), report)


class H3WindowSeam:
    """Pin-and-trim at a hot seam: discard the leading handle of a recovered
    window instead of crossfading two independent takes of it."""

    DESCRIPTION = (
        "EXPERIMENTAL (alpha), new 2026-08-12; the classic pipeline nodes "
        "are unchanged.\n\n"
        "Internal to H3 Window Expand. At a COLD seam (both sides of the "
        "boundary at hold 1) it is a passthrough that only carries the "
        "planner's per-side crossfade widths into the splice map. At a HOT "
        "seam (a boundary forced inside a burst) it drops trim_head frames "
        "from the front of the recovered window and moves the splice start "
        "forward by the same amount, so the previous window's output is "
        "kept and this window's second take of those frames is thrown away. "
        "Crossfading them instead would dissolve between two takes of the "
        "fastest motion in the clip, and for audio it would double or smear "
        "a percussive hit no matter how wide the fade.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "segment": ("IMAGE", {"tooltip": "recovered window, world clock"}),
            "splice_map": ("STRING", {"default": "", "forceInput": True}),
            "trim_head": ("INT", {"default": 0, "min": 0, "max": 96}),
            "fade_in": ("INT", {"default": 6, "min": 0, "max": 48,
                        "tooltip": "crossfade width the splice will use on the leading seam; 0 = hard cut"}),
            "fade_out": ("INT", {"default": 6, "min": 0, "max": 48}),
        }, "optional": {
            "segment_audio": ("AUDIO",),
            "fps": ("INT", {"default": 24, "min": 1, "max": 120}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING", "AUDIO")
    RETURN_NAMES = ("segment", "splice_map", "segment_audio")
    FUNCTION = "seam"
    CATEGORY = "image/minimax/motion"

    def seam(self, segment, splice_map, trim_head=0, fade_in=6, fade_out=6,
             segment_audio=None, fps=24):
        sp = json.loads(splice_map)
        t = max(0, min(int(trim_head), int(segment.shape[0]) - 1))
        # H3SegmentSplice fades over min(feather_frames, handle_in/out), and
        # the expansion hands it feather_frames=24, so these two numbers are
        # what actually sets each side's crossfade width
        sp["handle_in"], sp["handle_out"] = int(fade_in), int(fade_out)
        if t:
            segment = segment[t:]
            sp["start"] = sp["start"] + t
            sp["handle_in"] = 0
            if segment_audio is not None:
                sr = segment_audio["sample_rate"]
                cut = int(round(t / max(1, fps) * sr))
                w = segment_audio["waveform"]
                segment_audio = {"waveform": w[..., cut:].contiguous(),
                                 "sample_rate": sr}
        if segment_audio is None:
            segment_audio = {"waveform": torch.zeros(1, 2, 1),
                             "sample_rate": 32000}
        return (segment, json.dumps(sp), segment_audio)


class H3WindowPass:
    """Identity on (IMAGE, AUDIO). Placed by H3 Window Expand when there is
    nothing to regenerate or when plan_only is on, because an expansion is
    the only way a node with raw-link inputs can hand its inputs back."""

    DESCRIPTION = (
        "EXPERIMENTAL (alpha), new 2026-08-12. Internal passthrough placed "
        "by H3 Window Expand in plan-only mode and when the hold map has no "
        "held frames.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",)},
                "optional": {"audio": ("AUDIO",)}}

    RETURN_TYPES = ("IMAGE", "AUDIO")
    FUNCTION = "pass_through"
    CATEGORY = "image/minimax/motion"

    def pass_through(self, images, audio=None):
        if audio is None:
            audio = {"waveform": torch.zeros(1, 2, 1), "sample_rate": 32000}
        return (images, audio)


# ---------------------------------------------------------------------------
# the driver
# ---------------------------------------------------------------------------

class H3WindowExpand:
    """Plan N windows, then expand into N regeneration chains and a chained
    splice. One queue press, one node in the user's graph."""

    DESCRIPTION = (
        "EXPERIMENTAL (alpha), new 2026-08-12; the classic pipeline nodes "
        "are unchanged.\n\n"
        "Rolling-window regeneration for clips whose dilated length does "
        "not fit in VRAM. Reads the hold map, splits the held material into "
        "windows no larger than max_dilated_frames DILATED frames each, and "
        "returns an expanded graph containing the whole segment recipe once "
        "per window (crop, smear, encode, v2v init, sample, decode, "
        "recover) plus a chained H3 Segment Splice that writes each "
        "recovered window back over the world. Press queue once.\n\n"
        "The budget is in DILATED frames because that, not the clip length, "
        "is what sets the bill and the VRAM peak: read H3 Time Smear's "
        "report on a full-clip pass and set max_dilated_frames below the "
        "length that fits your card.\n\n"
        "Seams: boundaries snap to cold cuts (hold 1 on both sides), then "
        "to token boundaries, and never land inside a ramp shoulder. When "
        "no cold cut fits the budget the plan says so in the report and "
        "falls back to pin-and-trim: the next window's leading handle is "
        "generated as context at the same hold rate, then discarded, and "
        "the splice makes a hard cut instead of dissolving two takes of the "
        "same fast motion.\n\n"
        "Everything except the hold map is taken as a raw link and rewired "
        "into the expansion, so the model, guider, sampler and VAE are "
        "loaded once and shared by every window, and the per-window nodes "
        "keep normal input-signature caching: requeue after changing one "
        "widget and the windows whose inputs did not change are cache "
        "hits. Read the report output before generating; plan_only returns "
        "the baseline untouched with the plan attached.")

    HOT = ["pin-and-trim (default)", "crossfade", "refuse"]

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE", {"rawLink": True,
                       "tooltip": "baseline world frames"}),
            "hold_map": ("STRING", {"default": "", "forceInput": True,
                         "tooltip": "from the oracle, H3 Manual Hold Map or H3 Motion Editor"}),
            "vae": ("VAE", {"rawLink": True, "tooltip": "video VAE"}),
            "noise": ("NOISE", {"rawLink": True}),
            "guider": ("GUIDER", {"rawLink": True}),
            "sampler": ("SAMPLER", {"rawLink": True}),
            "sigmas": ("SIGMAS", {"rawLink": True,
                       "tooltip": "H3 Inject Schedule; every window uses it"}),
            "max_dilated_frames": ("INT", {"default": 209, "min": 39, "max": 4000,
                                   "tooltip": "budget per window, in DILATED frames"}),
            "handle_frames": ("INT", {"default": 12, "min": 2, "max": 48}),
            "feather_frames": ("INT", {"default": 6, "min": 0, "max": 24,
                               "tooltip": "crossfade width at COLD seams; hot seams are hard cuts"}),
        }, "optional": {
            "audio_vae": ("VAE", {"rawLink": True,
                          "tooltip": "audio VAE; unwired means video-only windows"}),
            "baseline_audio": ("AUDIO", {"rawLink": True}),
            "fps": ("INT", {"default": 24, "min": 1, "max": 120}),
            "reference_mix": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "hot_cut": (cls.HOT, {"default": cls.HOT[0]}),
            "snap_search": ("INT", {"default": 8, "min": 0, "max": 48,
                            "tooltip": "frames to look back for a token-aligned hot cut"}),
            "max_windows": ("INT", {"default": 12, "min": 1, "max": 64}),
            "grid_growth": ("BOOLEAN", {"default": True,
                            "tooltip": "grow a hold-1 handle to land on the 17k+5 grid "
                                       "instead of letting the tail freeze absorb the remainder"}),
            "plan_only": ("BOOLEAN", {"default": False}),
            "s_per_step": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 600.0, "step": 0.05}),
            "est_steps": ("INT", {"default": 18, "min": 1, "max": 100}),
        }}

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("images", "audio", "report")
    FUNCTION = "expand"
    CATEGORY = "image/minimax/motion"

    # ---- planning is separated from graph building so it can be tested
    #      without a running server (see agent-window-expand.md)
    def build_plan(self, hold_map, max_dilated_frames, handle_frames,
                   feather_frames, hot_cut="pin-and-trim (default)",
                   snap_search=8, max_windows=12, grid_growth=True,
                   fps=24, s_per_step=0.0, est_steps=18):
        holds = json.loads(hold_map)["holds"]
        n = len(holds)
        windows, notes = plan_windows(holds, int(max_dilated_frames),
                                      int(handle_frames), int(snap_search),
                                      int(max_windows), bool(grid_growth))
        specs = []
        for w in windows:
            hot_in = w["seam_in"] == "hot"
            hot_out = w["seam_out"] == "hot"
            if hot_in and hot_cut == "refuse":
                notes.append(f"window {w['k']}: hot seam and hot_cut=refuse; "
                             f"raise max_dilated_frames or lower d_max")
            fade_in = 0 if hot_in else min(int(feather_frames), w["handle_in"])
            fade_out = 0 if hot_out else min(int(feather_frames),
                                             w["handle_out"])
            if hot_in and hot_cut == "crossfade":
                fade_in = min(int(feather_frames), w["handle_in"])
            trim = w["handle_in"] if (hot_in and
                                      hot_cut != "crossfade") else 0
            specs.append({"k": w["k"], "a": w["a"], "b": w["b"],
                          "body_a": w["body_a"], "body_b": w["body_b"],
                          "world_len": n, "holds": w["holds"],
                          "handle_in": w["handle_in"],
                          "handle_out": w["handle_out"],
                          "dilated": w["dilated"], "tokens": w["tokens"],
                          "seam_in": w["seam_in"], "seam_out": w["seam_out"],
                          "fade_in": fade_in, "fade_out": fade_out,
                          "trim_head": trim})
        report = plan_report(windows, notes, holds, int(max_dilated_frames),
                             fps, s_per_step, est_steps)
        return specs, report

    def build_graph(self, specs, images, vae, noise, guider, sampler, sigmas,
                    audio_vae=None, baseline_audio=None, fps=24,
                    reference_mix=0.0, plan_only=False):
        """Return (GraphBuilder, image_link, audio_link).

        `images`, `vae`, ... are raw links (["node_id", slot]) into the
        user's graph, so the loaders and the conditioning stay outside the
        expansion and are shared by every window.
        """
        from comfy_execution.graph_utils import GraphBuilder

        g = GraphBuilder()

        def node(cls, nid, **kw):
            # An unwired optional stays absent rather than becoming a null
            # input: expanded nodes are never validated, so a null would only
            # surface later as a TypeError inside the body.
            return g.node(cls, nid, **{k: v for k, v in kw.items()
                                       if v is not None})

        if plan_only or not specs:
            p = node("H3WindowPass", "pass", images=images,
                     audio=baseline_audio)
            return g, p.out(0), p.out(1)

        world, world_audio = images, baseline_audio
        for s in specs:
            k = s["k"]
            # A hot seam has to read its leading handle from the window that
            # already ran, so pin-and-trim pins onto regenerated frames. A
            # cold seam reads the pristine baseline, which lets its crop run
            # before the previous window finishes and keeps one fewer
            # full-world tensor alive.
            src = world if s["seam_in"] == "hot" else images
            crop = node("H3WindowCrop", f"w{k}_crop", images=src,
                        window_spec=json.dumps(
                            {q: s[q] for q in CROP_SPEC_KEYS}))
            smear = node("H3TimeSmear", f"w{k}_smear", images=crop.out(0),
                         dilation=1, hold_map=crop.out(1), fps=fps)
            enc = node("VAEEncode", f"w{k}_enc", pixels=smear.out(0), vae=vae)
            init = node("H3V2VInit", f"w{k}_init", samples=enc.out(0),
                        length=0)
            samp = node("SamplerCustomAdvanced", f"w{k}_sampler", noise=noise,
                        guider=guider, sampler=sampler, sigmas=sigmas,
                        latent_image=init.out(0))
            dec = node("VAEDecode", f"w{k}_dec", samples=samp.out(0), vae=vae)
            rec = node("H3ExactRecover", f"w{k}_rec", images=dec.out(0),
                       hold_map=smear.out(1))
            seg_audio = None
            if audio_vae is not None:
                deca = node("VAEDecodeAudio", f"w{k}_deca",
                            samples=samp.out(0), vae=audio_vae)
                arec = node("H3AudioRecover", f"w{k}_arec", audio=deca.out(0),
                            hold_map=smear.out(1), fps=fps,
                            reference=world_audio,
                            reference_mix=reference_mix)
                seg_audio = arec.out(0)
            seam = node("H3WindowSeam", f"w{k}_seam", segment=rec.out(0),
                        splice_map=crop.out(2), trim_head=s["trim_head"],
                        fade_in=s["fade_in"], fade_out=s["fade_out"],
                        segment_audio=seg_audio, fps=fps)
            spl = node("H3SegmentSplice", f"w{k}_splice", baseline=world,
                       segment=seam.out(0), splice_map=seam.out(1),
                       feather_frames=24, baseline_audio=world_audio,
                       segment_audio=seam.out(2) if seg_audio else None,
                       fps=fps)
            world, world_audio = spl.out(0), spl.out(1)
        return g, world, world_audio

    def expand(self, images, hold_map, vae, noise, guider, sampler, sigmas,
               max_dilated_frames, handle_frames, feather_frames,
               audio_vae=None, baseline_audio=None, fps=24, reference_mix=0.0,
               hot_cut="pin-and-trim (default)", snap_search=8,
               max_windows=12, grid_growth=True, plan_only=False,
               s_per_step=0.0, est_steps=18):
        specs, report = self.build_plan(
            hold_map, max_dilated_frames, handle_frames, feather_frames,
            hot_cut, snap_search, max_windows, grid_growth, fps, s_per_step,
            est_steps)
        g, img, aud = self.build_graph(
            specs, images, vae, noise, guider, sampler, sigmas, audio_vae,
            baseline_audio, fps, reference_mix, plan_only)
        return {"result": (img, aud, report), "expand": g.finalize()}


NODE_CLASS_MAPPINGS = {
    "H3WindowExpand": H3WindowExpand,
    "H3WindowCrop": H3WindowCrop,
    "H3WindowSeam": H3WindowSeam,
    "H3WindowPass": H3WindowPass,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3WindowExpand": "H3 Window Expand (rolling regen)",
    "H3WindowCrop": "H3 Window Crop (internal)",
    "H3WindowSeam": "H3 Window Seam (internal)",
    "H3WindowPass": "H3 Window Pass (internal)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS",
           "plan_windows", "plan_report"]
