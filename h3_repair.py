"""Repair: regenerate a few bad frames of a finished clip and splice them
back, frame-exact (alpha, 2026-08-23).

The bilateral sibling of the extension nodes. Extension asks "what comes
after this clip"; repair asks "these frames are wrong, render them again
and give me back the same clip with only those frames different".

Two nodes, no model patch, no new de-rope:

  H3 Repair Plan     the arithmetic, in integers: snap the operator's bad
                     frame range outward to whole latent time tokens, carry
                     the span through a nearby shot cut if there is one,
                     and emit the three things the graph needs from it (a
                     hold map for H3 Time Smear, a time-varying regenerate
                     mask for H3 V2V Init, and the splice bounds)
  H3 Repair Splice   original frames outside the splice, repaired frames
                     inside, and the seam numbers that say whether it worked

DOCTRINE (minted 2026-08-23 from three measured cells, EXTENSION_V0):
regeneration MASKS snap to latent tokens; SPLICES are free. The mask has
to obey the grid because a latent time token cannot be half regenerated
(H3 V2V Init max-pools any mask onto the token clock), but the assembly
afterwards is pure frame selection and may cut anywhere. So: extend the
mask where the grid forces it, then choose the splice points for picture
reasons - enter in quiet motion, exit ON a shot cut whenever one is near,
and leave every untouched frame as original pixels.

Why the cut rule earns its keep. On the validation cell the token snap
pushed the regenerated span across a hard cut, so the model reinvented the
first frames of the next shot: the exit seam popped at 2.6x the clip's
median frame delta. Extending the mask through the cut did not fix it
(2.3x). Handing back to the original AT its own cut did: entry 0.21x,
exit 130.5 against the original's own 130.9 at that cut, i.e. the seam IS
the cut and reads as the edit the shot always had.

Token geometry (motion.py _token_frame_spans): frames group in 17s, each
group is 5 tokens covering (1, 4, 4, 4, 4) frames, so every 5th token is a
singleton on a 17-multiple; legal clip lengths are 17k+5. A bad frame at 45
lives in the token covering 43..46 and a bad frame at 47 in the token
covering 47..50, which is why "frames 45-47" becomes "regenerate 43..50".
"""
from __future__ import annotations

import json
import logging

import torch

try:                                    # loaded as part of the node pack
    from . import motion as _mo
    from .capsule_types import SCHEMA, Span, Timebase, digest
except ImportError:                     # loaded standalone (tests, tools)
    # by PATH, not by sys.path: putting the pack root on sys.path makes its
    # __init__.py importable as a top-level module named `__init__`, which
    # pytest's parent-module walk then tries to import and dies on.
    import importlib.util as _ilu
    import os as _os
    import sys as _sys

    def _sideload(_name):
        _key = "mainodes_sideload_" + _name
        if _key in _sys.modules:
            return _sys.modules[_key]
        _p = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _name + ".py")
        _s = _ilu.spec_from_file_location(_key, _p)
        _m = _ilu.module_from_spec(_s)
        _sys.modules[_key] = _m         # @dataclass resolves cls.__module__ here
        _s.loader.exec_module(_m)
        return _m

    _mo = _sideload("motion")
    _ct = _sideload("capsule_types")
    SCHEMA, Span, Timebase, digest = _ct.SCHEMA, _ct.Span, _ct.Timebase, _ct.digest

log = logging.getLogger("MAINodes.h3_repair")

LUMA = (0.2126, 0.7152, 0.0722)         # Rec.709, same weights as H3 Seam Normalize


# ------------------------------------------------------------------ helpers

def _token_spans(length):
    """Inclusive (first, last) frame span of every latent time token covering
    a clip of `length` frames, on H3's own grid. The token count comes from
    the length H3 would actually generate (_legal_ceil), so the spans tile
    [0, legal_length) even when the operator's clip is off-grid."""
    legal = _mo._legal_ceil(int(length))
    t_lat = _mo._token_count(int(length))
    return _mo._token_frame_spans(t_lat), legal, t_lat


def _token_of(f, spans):
    for t, (a, b) in enumerate(spans):
        if a <= f <= b:
            return t
    return len(spans) - 1


def _luma_deltas(images):
    """Per-frame mean abs luma delta in 0-255 units; d[0] = 0.

    d[i] is the delta ACROSS the boundary i-1 -> i, so the seam between the
    last kept original frame and the first repaired one is d[splice_lo]."""
    x = images.detach().float().cpu()
    if x.dim() == 4 and x.shape[-1] >= 3:
        w = torch.tensor(LUMA, dtype=x.dtype)
        y = (x[..., :3] * w).sum(-1)
    else:
        y = x.reshape(x.shape[0], -1)
    y = y.reshape(y.shape[0], -1) * 255.0
    d = torch.zeros(y.shape[0], dtype=y.dtype)
    if y.shape[0] > 1:
        d[1:] = (y[1:] - y[:-1]).abs().mean(dim=1)
    return d


def _parse_cuts(s, length):
    cuts, bad = [], []
    for part in str(s).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            c = int(part)
        except ValueError:
            bad.append(part)
            continue
        if 1 <= c <= length - 1:
            cuts.append(c)
        else:
            bad.append(part)
    return sorted(set(cuts)), bad


# ------------------------------------------------------------------- plan

class H3RepairPlan:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "length": ("INT", {"default": 90, "min": 22, "max": 3600,
                               "tooltip": "frames in the SOURCE clip being repaired (world clock, not dilated)"}),
            "bad_start": ("INT", {"default": 0, "min": 0, "max": 3599,
                                  "tooltip": "first bad frame, 0-BASED and inclusive. A file-numbered frame 46 is 45 here"}),
            "bad_end": ("INT", {"default": 0, "min": 0, "max": 3599,
                                "tooltip": "last bad frame, 0-based and inclusive"}),
            "hold": ("INT", {"default": 3, "min": 1, "max": 8,
                             "tooltip": "de-rope hold count on the regenerated span only; everything else stays real time. Set H3 Time Smear's dilation to the same number"}),
            "fps": ("INT", {"default": 24, "min": 1, "max": 120}),
            "shot_cuts": ("STRING", {"default": "", "multiline": False,
                                     "tooltip": "comma list of 0-based indices of the FIRST frame of each new shot, e.g. '53'. A cut near the span becomes the exit: the splice hands back to the original AT the cut"}),
            "cut_reach": ("INT", {"default": 8, "min": 0, "max": 68,
                                  "tooltip": "how many frames past the snapped span a cut still counts as 'near' and gets used as the exit"}),
        }, "optional": {
            "images": ("IMAGE", {"tooltip": "the source clip, for the REPORT only: per-frame luma deltas, so the plan can say whether the entry frame is quiet. Also supplies width/height when those are 0"}),
            "width": ("INT", {"default": 0, "min": 0, "max": 16384,
                              "tooltip": "0 = take it from images if wired, else the mask comes out 1x1 (H3 V2V Init resamples it to the latent grid anyway)"}),
            "height": ("INT", {"default": 0, "min": 0, "max": 16384}),
        }}

    RETURN_TYPES = ("STRING", "STRING", "MASK", "STRING", "STRING")
    RETURN_NAMES = ("hold_map", "ranges", "regen_mask", "plan", "report")
    FUNCTION = "plan"
    CATEGORY = "MAINodes/alpha"
    DESCRIPTION = (
        "EXPERIMENTAL (alpha), new 2026-08-23. Turns 'frames 45 to 47 are wrong' into the whole "
        "arithmetic of a repair pass: the bad range snapped OUTWARD to whole latent time tokens "
        "(a token cannot be half regenerated), the span carried through a nearby shot cut so the "
        "model does not reinvent the next shot's first frames, the hold map for H3 Time Smear, the "
        "time-varying regenerate mask for H3 V2V Init in DILATED coordinates, and the splice bounds "
        "for H3 Repair Splice. Masks snap to tokens, splices are free: the mask obeys the grid, the "
        "assembly exits ON the cut. Wire hold_map to H3 Time Smear, regen_mask to H3 V2V Init "
        "(time_varying ON), plan to H3 Repair Splice.")

    def plan(self, length, bad_start, bad_end, hold, fps, shot_cuts, cut_reach,
             images=None, width=0, height=0):
        length = int(length)
        hold = max(1, int(hold))
        tb = Timebase(fps_num=int(fps), fps_den=1)
        notes = []

        spans, legal, t_lat = _token_spans(length)
        if legal != length:
            notes.append(f"length {length} is not on the 17k+5 grid; H3 generates {legal} "
                         f"and pads {legal - length} frame(s). The token grid below is {legal}'s")

        a = max(0, min(int(bad_start), length - 1))
        b = max(0, min(int(bad_end), length - 1))
        if b < a:
            a, b = b, a
            notes.append("bad_end was before bad_start; swapped")
        if (a, b) != (int(bad_start), int(bad_end)):
            notes.append(f"bad range clamped to the clip: {bad_start}-{bad_end} -> {a}-{b}")

        t_lo, t_hi = _token_of(a, spans), _token_of(b, spans)
        regen_lo, regen_hi = spans[t_lo][0], spans[t_hi][1]
        snap_tokens = [t_lo, t_hi]

        # 2. a cut inside the span, or just past it, becomes the exit
        cuts, bad_cuts = _parse_cuts(shot_cuts, length)
        for s in bad_cuts:
            notes.append(f"ignored shot_cuts entry '{s}': not an integer in 1..{length - 1}")
        reach = max(0, int(cut_reach))
        cands = [c for c in cuts
                 if (regen_lo <= c <= regen_hi) or (regen_hi < c <= regen_hi + reach)]
        rejected = [c for c in cands if c <= regen_lo]
        cands = [c for c in cands if c > regen_lo]
        for c in rejected:
            notes.append(f"cut {c} is at or before the span start {regen_lo}: using it as the exit "
                         f"would leave nothing repaired, so it was ignored")
        cut_used = -1
        if cands:
            cut_used = min(cands, key=lambda c: (abs(c - regen_hi), c))
            t_cut = _token_of(cut_used, spans)
            regen_hi = max(regen_hi, spans[t_cut][1])
            snap_tokens[1] = max(snap_tokens[1], t_cut)
            splice_hi = cut_used - 1
            if len(cands) > 1:
                notes.append(f"cuts {cands} all qualified; took the nearest to the span end, {cut_used}")
        else:
            splice_hi = regen_hi
        splice_lo = regen_lo

        if regen_hi > length - 1:
            notes.append(f"the last regenerated token ends at frame {regen_hi}, past the clip's last "
                         f"frame {length - 1}; the hold map covers real frames only")
            regen_hi = length - 1
            splice_hi = min(splice_hi, length - 1)

        # 3. entry quietness (report only, never moves the splice)
        entry_quiet = None
        entry_delta = median_delta = None
        suggestion = None
        if images is not None and int(images.shape[0]) > 1:
            d = _luma_deltas(images)
            median_delta = float(d.median())
            n = int(d.shape[0])
            if splice_lo < n:
                entry_delta = float(d[splice_lo])
                entry_quiet = entry_delta <= median_delta
                if not entry_quiet:
                    win = [f for f in range(max(1, splice_lo - 4), min(n, splice_lo + 5))]
                    if win:
                        best = min(win, key=lambda f: float(d[f]))
                        if best != splice_lo:
                            suggestion = (best, float(d[best]))
            if n != length:
                notes.append(f"images carry {n} frames but length says {length}; the delta report "
                             f"used the images as given")

        # 4. dilated coordinates
        holds = [1] * length
        for f in range(regen_lo, regen_hi + 1):
            holds[f] = hold
        dilated_lo = sum(holds[:regen_lo])
        dilated_hi = dilated_lo + hold * (regen_hi - regen_lo + 1) - 1
        dilated_len = sum(holds)
        dilated_padded = _mo._legal_ceil(dilated_len)
        pad = dilated_padded - dilated_len

        # H3 Time Smear parks the grid pad in the LAST hold, so the pad frames
        # are extra copies of the last WORLD frame. They belong to the regen
        # span only when the span runs to the end of the clip.
        mask_hi = dilated_hi + (pad if regen_hi == length - 1 else 0)

        _, e2e_note = _mo.expand_hold_map_to_end(list(holds))
        if e2e_note:
            notes.append("H3 Time Smear's expand_to_end WOULD REWRITE this map (" + e2e_note +
                         "), which moves every dilated coordinate below. Set expand_to_end OFF "
                         "on H3 Time Smear for a repair pass")

        # 5. mask
        W = int(width) or (int(images.shape[2]) if images is not None else 0) or 1
        H = int(height) or (int(images.shape[1]) if images is not None else 0) or 1
        if not int(width) and images is None:
            notes.append("no width/height and no images wired: the mask is 1x1. H3 V2V Init "
                         "resamples it to the latent grid, so a whole-frame time mask still works, "
                         "but wire the ResolutionSelector if you want to read it")
        mask = torch.zeros(dilated_padded, H, W)
        mask[dilated_lo:mask_hi + 1] = 1.0

        p = {"regen_lo": regen_lo, "regen_hi": regen_hi,
             "splice_lo": splice_lo, "splice_hi": splice_hi,
             "cut_used": cut_used, "dilated_len": dilated_len,
             "dilated_lo": dilated_lo, "dilated_hi": dilated_hi,
             "hold": hold, "length": length}
        plan = {"schema": SCHEMA, "kind": "repair_plan", "timebase": tb.to_dict(),
                "requested": {"length": length, "bad_start": int(bad_start), "bad_end": int(bad_end),
                              "hold": hold, "shot_cuts": str(shot_cuts), "cut_reach": reach},
                "resolved": dict(p, bad_start=a, bad_end=b,
                                 dilated_padded=dilated_padded, dilated_pad=pad,
                                 mask_len=dilated_padded, mask_hi=mask_hi,
                                 mask_width=W, mask_height=H,
                                 legal_length=legal, t_lat=t_lat,
                                 tokens=snap_tokens,
                                 token_span_lo=list(spans[snap_tokens[0]]),
                                 token_span_hi=list(spans[snap_tokens[1]]),
                                 entry_quiet=entry_quiet,
                                 entry_delta=entry_delta, median_delta=median_delta,
                                 span=Span.make(splice_lo, splice_hi + 1, tb).to_dict()),
                "notes": notes}
        plan.update(p)                     # top-level too: the splice node reads it flat
        plan["digest"] = digest({k: plan[k] for k in ("timebase", "resolved")})

        ranges = f"{regen_lo}-{regen_hi}:{hold}"
        hold_map = json.dumps({"holds": holds, "world_len": length})

        rep = ["H3 Repair Plan",
               f"  clip            {length} frames @ {fps} fps ({length / max(1, int(fps)):.2f} s), "
               f"{t_lat} latent time tokens",
               f"  bad             {a}..{b}  ({b - a + 1} frames, 0-based)",
               f"  regenerate      {regen_lo}..{regen_hi}  ({regen_hi - regen_lo + 1} frames) "
               f"= tokens {snap_tokens[0]}..{snap_tokens[1]}, spans {tuple(spans[snap_tokens[0]])} "
               f"to {tuple(spans[snap_tokens[1]])}   MASK, snapped to tokens",
               f"  keep repaired   {splice_lo}..{splice_hi}  ({splice_hi - splice_lo + 1} frames)"
               f"                       SPLICE, free",
               f"  hold            {hold} on the regen span, 1 everywhere else",
               f"  dilated span    {dilated_lo}..{dilated_hi} of {dilated_len} dilated frames"
               + (f"; H3 Time Smear pads to {dilated_padded} (+{pad} on the last hold)" if pad else
                  "; already on the 17k+5 grid")]
        if cut_used >= 0:
            rep.append(f"  DOCTRINE        exit on cut {cut_used}: the mask runs through the cut to "
                       f"{regen_hi} because the token grid says so, and the splice hands back to the "
                       f"original AT {cut_used}, so the seam is the cut the shot already had")
        else:
            rep.append(f"  DOCTRINE        no cut within {reach} frames of the span; exit at the token "
                       f"end {regen_hi}. The exit seam is a real seam - check it in the splice report")
        if entry_delta is not None:
            rep.append(f"  entry motion    frame {splice_lo} delta {entry_delta:.1f}/255 vs clip median "
                       f"{median_delta:.1f} -> {'QUIET, good place to enter' if entry_quiet else 'BUSY'}")
            if suggestion:
                rep.append(f"                  quieter frame within 4: {suggestion[0]} "
                           f"(delta {suggestion[1]:.1f}). The plan does NOT move the entry - "
                           f"change bad_start if you want it")
        rep.append("  cost            " + _mo._cost_report(length, dilated_padded, int(fps)))
        for n in notes:
            rep.append(f"  NOTE {n}")
        text = "\n".join(rep)
        log.info("\n" + text)
        return (hold_map, ranges, mask, json.dumps(plan), text)


# ------------------------------------------------------------------ splice

class H3RepairSplice:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "original": ("IMAGE", {"tooltip": "the source clip, untouched"}),
            "repaired": ("IMAGE", {"tooltip": "H3 Exact Recover's output: the same clip, regenerated"}),
            "plan": ("STRING", {"forceInput": True, "tooltip": "from H3 Repair Plan"}),
        }, "optional": {
            "audio": ("AUDIO", {"tooltip": "passthrough. A repair does not touch audio: same length, same clock"}),
        }}

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("images", "audio", "report")
    FUNCTION = "splice"
    CATEGORY = "MAINodes/alpha"
    DESCRIPTION = (
        "EXPERIMENTAL (alpha), new 2026-08-23. Frame-exact reassembly: original pixels outside the "
        "splice, repaired pixels inside, nothing resampled and nothing blended. Splices are free - "
        "the plan already chose the bounds (exit on a cut when one was near), this node just does "
        "the selection and measures the seams: entry and exit mean abs luma delta against the clip's "
        "median frame delta, each printed NEXT TO the original clip's own delta at the same frames, "
        "because at a hard cut a big number is the right answer. It also asserts that every frame "
        "outside the splice is bit-identical to the original.")

    def splice(self, original, repaired, plan, audio=None):
        p = json.loads(plan)
        lo = int(p["splice_lo"])
        hi = int(p["splice_hi"])
        o = original.detach().cpu()
        r = repaired.detach().cpu()
        L = int(o.shape[0])
        warn = []
        if int(r.shape[0]) != L:
            warn.append(f"repaired has {r.shape[0]} frames, original has {L}; the splice reads the "
                        f"repaired clip at the SAME indices, so a length mismatch means the plan's "
                        f"hold map and H3 Exact Recover disagree")
        hi = min(hi, L - 1, int(r.shape[0]) - 1)
        lo = max(0, min(lo, hi))
        out = torch.cat([o[:lo], r[lo:hi + 1].to(o.dtype), o[hi + 1:]], dim=0)

        outside = 0.0
        for sl in (slice(0, lo), slice(hi + 1, L)):
            if sl.stop > sl.start:
                outside = max(outside, float((out[sl] - o[sl]).abs().max()))
        assert outside == 0.0, (
            f"frames outside the splice differ from the original by {outside}; reassembly must be "
            f"pure frame selection")

        d_out = _luma_deltas(out)
        d_org = _luma_deltas(o)
        med_out = float(d_out.median())
        med_org = float(d_org.median())

        def seam(i, what):
            if not (1 <= i < L):
                return f"  {what:<12}n/a (frame {i} is outside the clip)"
            a, b = float(d_out[i]), float(d_org[i])
            return (f"  {what:<12}{a:8.2f}/255 = {a / (med_out + 1e-9):.2f}x median   "
                    f"(the original's own delta at {i - 1}->{i}: {b:.2f} = {b / (med_org + 1e-9):.2f}x)")

        rep = [f"H3 Repair Splice: original[0:{lo}] + repaired[{lo}:{hi + 1}] + original[{hi + 1}:{L}] "
               f"= {int(out.shape[0])} frames, {hi - lo + 1} repaired",
               f"  median delta {med_out:.2f}/255 (output), {med_org:.2f}/255 (original)",
               seam(lo, f"entry {lo - 1}->{lo}"),
               seam(hi + 1, f"exit {hi}->{hi + 1}")]
        if int(p.get("cut_used", -1)) >= 0:
            rep.append(f"  exit is ON the shot cut at {p['cut_used']}: the two numbers above should be "
                       f"close to each other, and both large - that is the cut, not a pop")
        rep.append(f"  outside the splice: max abs pixel difference from the original {outside:.0f} "
                   f"(0 = frame-exact, asserted)")
        if audio is not None:
            wf = audio["waveform"]
            rep.append(f"  audio passthrough: {int(wf.shape[-1])} samples @ {int(audio['sample_rate'])} Hz "
                       f"({wf.shape[-1] / max(1, int(audio['sample_rate'])):.3f} s), untouched")
        rep += ["  WARNING " + w for w in warn]
        text = "\n".join(rep)
        log.info("\n" + text)
        return (out, audio, text)


NODE_CLASS_MAPPINGS = {
    "H3RepairPlan": H3RepairPlan,
    "H3RepairSplice": H3RepairSplice,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3RepairPlan": "H3 Repair Plan (alpha)",
    "H3RepairSplice": "H3 Repair Splice (alpha)",
}
