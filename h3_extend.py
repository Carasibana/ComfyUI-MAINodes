"""Extension: make a long clip out of short generations (alpha, 2026-08-23).

The job: "my card only comfortably renders five seconds, I want forty."
Generate a segment, de-rope it, carry its last HANDLE frames (video and
the matching audio) into the next generation as opening context, generate
the next segment, de-rope it with the carried prefix protected, trim the
prefix and the grid surplus, append. Repeat.

Four nodes, no model patch, no new de-rope:

  H3 Extension Plan   the arithmetic, in integers: segment and handle
                      lengths on the 17k+5 grid and the 40 Hz audio clock,
                      requested vs resolved, the anchor modes, one report
  H3 Tail Context     the last HANDLE frames + sample-exact audio of a
                      finished segment, ready for MiniMaxH3AddGuide @ 0
  H3 Protect Prefix   hold_map[:HANDLE] = 1 so the de-rope sees the incoming
                      velocity but may not retime the anchor
  H3 Trim             drop the hidden prefix and the surplus, audio
                      sample-exact, and advance the global timeline

The default atom is 141/39: 141 frames = 235 ticks, handle 39 = 65 ticks,
new material 102 = 170 ticks, the handle starts at 102 = 6x17 in the
source and lands at 0 in the destination. Every append adds exactly 102
frames and 170 ticks, so nothing fractional accumulates. 22 is legal on
the frame grid but not on the audio clock (36.67 ticks); the plan says so
rather than letting the chain drift.

Measured basis (h3-extend-and-heal, 2026-08-14): a 39-frame guide-anchored
join was playback-clean as a hard cut, overlap MAE 2.18/255 vs 16.47
unanchored, one neutral cell at 0.4 MP, base 12-step. 5 and 22 are
unmeasured cells.

v0 anchors with MiniMaxH3AddGuide (core): the handle is the guide clip at
frame 0, full H3 VAE, never noised, never TAE. Per-token mask pinning
(#15375) is the better mechanism once a graph shape for it is validated;
the plan resolves `auto` to `guide` and says why.
"""
from __future__ import annotations

import json
import logging
from fractions import Fraction

import torch

from .capsule_types import (SCHEMA, Handle, Span, Timebase, align_down, align_up,
                            canonical_json, digest)

log = logging.getLogger("MAINodes.h3_extend")

TRANSITIONS = {
    "seamless (39-frame handle)": 39,
    "cheap seamless (5-frame handle, unmeasured)": 5,
    "frame continuity (last frame only)": 1,
    "scene cut (no handle, global refs only)": 0,
    "custom (use handle_frames)": -1,
}


def _probe():
    try:
        from .h3_capabilities import probe_core
        return probe_core()
    except Exception:  # noqa: BLE001
        return {}


class H3ExtensionPlan:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "segment_frames": ("INT", {"default": 141, "min": 5, "max": 3600,
                                       "tooltip": "frames to GENERATE per segment, handle included; rounds UP to 17k+5 like core. 141 = the atom"}),
            "transition": (list(TRANSITIONS), {"default": "seamless (39-frame handle)"}),
            "handle_frames": ("INT", {"default": 39, "min": 0, "max": 600,
                                      "tooltip": "only read for transition = custom; clip guides round DOWN to 17k+5, below 5 becomes one frame"}),
            "new_frames": ("INT", {"default": 0, "min": 0, "max": 3600,
                                   "tooltip": "frames of NEW material wanted from this segment; 0 = everything after the handle (102 on the atom). More than fits raises the generation length"}),
            "visual_anchor": (["auto", "guide", "per_token_mask", "none"], {"default": "auto"}),
            "audio_anchor": (["auto", "guide", "regenerated", "none"], {"default": "auto"}),
            "fps_num": ("INT", {"default": 24, "min": 1, "max": 240}),
            "fps_den": ("INT", {"default": 1, "min": 1, "max": 1001}),
            "previous_length": ("INT", {"default": 141, "min": 0, "max": 3600,
                                        "tooltip": "frames in the FINISHED previous segment (after its own trim, or the first segment's full length); places the handle's source span"}),
        }}

    RETURN_TYPES = ("STRING", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("plan", "length", "handle_frames", "new_frames", "report")
    FUNCTION = "plan"
    CATEGORY = "MAINodes/alpha"
    DESCRIPTION = (
        "EXPERIMENTAL (alpha), new 2026-08-23. Integer arithmetic for one extension step: "
        "how many frames to generate so that HANDLE context frames plus NEW frames fit the 17k+5 grid, "
        "whether each length lands on the 40 Hz audio clock (frames divisible by 3), where the handle "
        "comes from in the previous segment and where it lands, and which anchor mechanism the running "
        "core supports. Feed `length` to the H3 conditioning node, `plan` to H3 Tail Context, "
        "H3 Protect Prefix and H3 Trim.")

    def plan(self, segment_frames, transition, handle_frames, new_frames, visual_anchor, audio_anchor,
             fps_num, fps_den, previous_length):
        tb = Timebase(fps_num=int(fps_num), fps_den=int(fps_den))
        caps = _probe()
        requested = {"segment_frames": int(segment_frames), "transition": transition,
                     "handle_frames": int(handle_frames), "new_frames": int(new_frames),
                     "visual_anchor": visual_anchor, "audio_anchor": audio_anchor}
        reasons = {}
        notes = []

        # handle
        h_req = TRANSITIONS[transition]
        if h_req < 0:
            h_req = int(handle_frames)
        handle = align_down(h_req) if h_req >= 5 else h_req      # 1 and 0 pass through
        if handle != h_req:
            reasons["handle_frames"] = f"clip guides round DOWN to 17k+5: {h_req} -> {handle}"

        # generation length
        if int(new_frames) > 0:
            new = int(new_frames)
            length = align_up(handle + new)
            if length != handle + new:
                reasons["length"] = f"{handle} handle + {new} new = {handle + new}, rounded UP to {length} (core align_frame_count)"
        else:
            length = align_up(int(segment_frames))
            if length != int(segment_frames):
                reasons["length"] = f"segment_frames {segment_frames} rounded UP to {length} (core align_frame_count)"
            new = length - handle
        surplus = length - handle - new

        # anchors
        v = visual_anchor
        if v == "auto":
            v = "guide" if handle > 0 else "none"
            reasons["visual_anchor"] = ("guide: v0 anchors the handle with MiniMaxH3AddGuide at frame 0; "
                                        "per-token masks are present on this core" if caps.get("per_token_masks") else
                                        "guide: per-token masks not found on this core")
            if handle == 0:
                reasons["visual_anchor"] = "none: scene cut carries no handle"
        elif v == "per_token_mask" and not caps.get("per_token_masks"):
            v = "guide"
            reasons["visual_anchor"] = "per_token_mask requested but the running core lacks #15375; fell back to guide"
        a = audio_anchor
        if a == "auto":
            a = "guide" if handle > 0 else "regenerated"
            reasons["audio_anchor"] = "guide rides the same AddGuide call as the video handle" if handle > 0 else "no handle, audio regenerates"
        if a == "guide" and not caps.get("audio_guide", True):
            a = "regenerated"
            reasons["audio_anchor"] = "AddGuide has no audio input on this core; audio regenerates"

        # spans
        src_start = max(0, int(previous_length) - handle)
        source = Span.make(src_start, int(previous_length), tb) if handle else Span.make(0, 0, tb)
        dest = Span.make(0, handle, tb)
        if handle >= 5 and src_start % 17 != 0:
            notes.append(f"handle source starts at {src_start}, not a 17-multiple: token phase differs between source and destination "
                         f"(the atom starts at 102 = 6x17). Legal, but unmeasured")
        for name, n in (("length", length), ("handle", handle), ("new", new)):
            if not tb.clock_aligned(n):
                notes.append(f"{name} = {n} frames is {tb.ticks(n)} audio ticks, not an integer: "
                             f"up to 12.5 ms of audio-length error per segment, accumulates in a chain. Use frames divisible by 3")
        if surplus:
            notes.append(f"{surplus} surplus frame(s) generated beyond the requested new material; H3 Trim drops them")

        hd = Handle(role="extension_prefix", source=source, destination=dest, protected=True,
                    retime_allowed=False, ownership="source", visual_anchor=v, audio_anchor=a, notes=notes)
        resolved = {"length": length, "handle_frames": handle, "new_frames": new, "surplus_frames": surplus,
                    "visual_anchor": v, "audio_anchor": a,
                    "length_ticks": str(tb.ticks(length)), "handle_ticks": str(tb.ticks(handle)),
                    "new_ticks": str(tb.ticks(new))}
        plan = {"schema": SCHEMA, "kind": "extension_plan", "timebase": tb.to_dict(),
                "requested": requested, "resolved": resolved, "reasons": reasons,
                "handle": hd.to_dict(),
                "capabilities": {k: caps.get(k) for k in ("comfy_commit", "per_token_masks", "clip_guide",
                                                          "audio_guide", "tokenizer_special_tokens")}}
        plan["digest"] = digest({k: plan[k] for k in ("timebase", "resolved", "handle")})

        def tk(n):
            t = tb.ticks(n)
            return f"{t} ticks" + ("" if t.denominator == 1 else "  NOT INTEGER")
        rep = ["H3 Extension Plan",
               f"  generate        {length:4d} frames  {tk(length)}  ({tb.seconds(length):.3f} s)",
               f"  handle          {handle:4d} frames  {tk(handle)}  source [{source.start},{source.end}) -> destination [0,{handle})",
               f"  new material    {new:4d} frames  {tk(new)}",
               f"  surplus         {surplus:4d} frames  (trimmed)",
               f"  visual anchor   {v.upper()}",
               f"  audio anchor    {a.upper()}",
               f"  handle          protected, retime not allowed, full H3 VAE, never noised"]
        for k, why in reasons.items():
            rep.append(f"  resolved {k}: {why}")
        for n in notes:
            rep.append(f"  NOTE {n}")
        if caps.get("tokenizer_special_tokens") is False:
            rep.append("  WARNING this core predates #15808: dialogue tags tokenize wrong; update before judging audio")
        text = "\n".join(rep)
        log.info("\n" + text)
        return (json.dumps(plan), int(length), int(handle), int(new), text)


def _audio_slice(audio, start_frames, n_frames, tb: Timebase):
    """Sample-exact slice of an AUDIO dict for frames [start, start+n)."""
    sr = int(audio["sample_rate"])
    wf = audio["waveform"]
    tb2 = Timebase(tb.fps_num, tb.fps_den, tb.audio_hz, sr)
    s0 = tb2.samples(start_frames)
    n = tb2.samples(n_frames)
    warn = []
    if s0.denominator != 1 or n.denominator != 1:
        warn.append(f"audio slice not sample-exact at {sr} Hz / {tb.fps} fps (start {s0}, length {n}); rounded")
    a = int(round(s0)); b = a + int(round(n))
    out = wf[..., a:b]
    if out.shape[-1] < int(round(n)):
        warn.append(f"audio shorter than the frames it should cover: {out.shape[-1]} of {int(round(n))} samples")
    return {"waveform": out, "sample_rate": sr}, warn


class H3TailContext:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "plan": ("STRING", {"forceInput": True})},
                "optional": {"audio": ("AUDIO",)}}

    RETURN_TYPES = ("IMAGE", "AUDIO", "INT", "STRING", "STRING")
    RETURN_NAMES = ("tail_images", "tail_audio", "guide_frame_idx", "handle", "report")
    FUNCTION = "tail"
    CATEGORY = "MAINodes/alpha"
    DESCRIPTION = (
        "EXPERIMENTAL (alpha). The last HANDLE frames of a finished (recovered) segment and the "
        "sample-exact matching audio, to wire into MiniMaxH3AddGuide (image + audio, frame_idx 0) "
        "of the next segment. Full H3 VAE on the way in, no noise, no TAE: this is the seam.")

    def tail(self, images, plan, audio=None):
        p = json.loads(plan)
        tb = Timebase.from_dict(p["timebase"])
        hd = Handle.from_dict(p["handle"])
        h = hd.destination.frames
        n = int(images.shape[0])
        warn = []
        if h == 0:
            tail = images[-1:]
            warn.append("scene cut: handle is 0 frames; passing the last frame for reference only, do not wire it as a guide")
        else:
            if n < h:
                warn.append(f"segment has {n} frames, handle wants {h}; using all of them")
            tail = images[-min(h, n):]
        if hd.source.frames and hd.source.end != n:
            warn.append(f"plan placed the handle source at [{hd.source.start},{hd.source.end}) but the segment has {n} frames; slicing from the real end")
        if audio is not None and h:
            ta, w = _audio_slice(audio, n - tail.shape[0], tail.shape[0], tb)
            warn += w
        elif audio is not None:
            ta = audio
        else:
            ta = {"waveform": torch.zeros(1, 1, 0), "sample_rate": 48000}
            warn.append("no audio wired: the audio handle is empty, the next segment regenerates audio from zero")
        hd.source = Span.make(n - tail.shape[0], n, tb)
        rep = [f"H3 Tail Context: {tail.shape[0]} frames [{hd.source.start},{hd.source.end}) of {n}, "
               f"{tb.ticks(tail.shape[0])} ticks, audio {ta['waveform'].shape[-1]} samples @ {ta['sample_rate']} Hz -> guide at frame 0"]
        rep += ["  WARNING " + w for w in warn]
        text = "\n".join(rep)
        log.info(text)
        return (tail, ta, 0, json.dumps(hd.to_dict()), text)


class H3ProtectPrefix:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"hold_map": ("STRING", {"forceInput": True}),
                             "plan": ("STRING", {"forceInput": True})}}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("hold_map", "report")
    FUNCTION = "protect"
    CATEGORY = "MAINodes/alpha"
    DESCRIPTION = (
        "EXPERIMENTAL (alpha). Forces hold 1 over the carried prefix so the de-rope may not retime "
        "the continuity anchor; the oracle still saw it and so still knows the incoming velocity. "
        "Same idea as the handles of H3 Segment Crop and the window overlap. Background freeze on "
        "H3 V2V Init stays a separate, independent decision.")

    def protect(self, hold_map, plan):
        hm = json.loads(hold_map)
        h = Handle.from_dict(json.loads(plan)["handle"]).destination.frames
        holds = list(hm["holds"])
        before = sum(int(x) for x in holds[:h])
        for i in range(min(h, len(holds))):
            holds[i] = 1
        hm["holds"] = holds
        hm.setdefault("protected_prefix", h)
        text = (f"H3 Protect Prefix: holds[:{h}] = 1 ({before - min(h, len(holds))} dilated frames removed from the prefix; "
                f"{sum(int(x) for x in holds)} dilated frames remain of {len(holds)})")
        log.info(text)
        return (json.dumps(hm), text)


class H3Trim:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "plan": ("STRING", {"forceInput": True}),
                             "global_start": ("INT", {"default": 0, "min": 0, "max": 10**7,
                                                      "tooltip": "frames already on the assembled timeline before this segment's new material"})},
                "optional": {"audio": ("AUDIO",)}}

    RETURN_TYPES = ("IMAGE", "AUDIO", "INT", "STRING", "STRING")
    RETURN_NAMES = ("images", "audio", "global_end", "global_span", "report")
    FUNCTION = "trim"
    CATEGORY = "MAINodes/alpha"
    DESCRIPTION = (
        "EXPERIMENTAL (alpha). Drops the hidden handle prefix and any grid surplus from a recovered "
        "segment, audio sample-exact, and reports where the kept material sits on the global timeline. "
        "All trims map to ONE global clock; the final requested duration is one trim after assembly.")

    def trim(self, images, plan, global_start, audio=None):
        p = json.loads(plan)
        tb = Timebase.from_dict(p["timebase"])
        h = int(p["resolved"]["handle_frames"])
        new = int(p["resolved"]["new_frames"])
        n = int(images.shape[0])
        warn = []
        if n < h + new:
            warn.append(f"recovered segment has {n} frames, plan expected {h + new}; keeping what exists after the prefix")
        keep = images[h:h + new]
        if audio is not None:
            ta, w = _audio_slice(audio, h, keep.shape[0], tb)
            warn += w
        else:
            ta = None
        gs = Span.make(int(global_start), int(global_start) + keep.shape[0], tb)
        text = (f"H3 Trim: dropped {h} prefix + {max(0, n - h - keep.shape[0])} surplus, kept {keep.shape[0]} frames "
                f"({tb.ticks(keep.shape[0])} ticks) -> global [{gs.start},{gs.end})")
        if warn:
            text += "\n" + "\n".join("  WARNING " + w for w in warn)
        log.info(text)
        return (keep, ta, int(gs.end), json.dumps(gs.to_dict()), text)


NODE_CLASS_MAPPINGS = {
    "H3ExtensionPlan": H3ExtensionPlan,
    "H3TailContext": H3TailContext,
    "H3ProtectPrefix": H3ProtectPrefix,
    "H3Trim": H3Trim,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3ExtensionPlan": "H3 Extension Plan (alpha)",
    "H3TailContext": "H3 Tail Context (alpha)",
    "H3ProtectPrefix": "H3 Protect Prefix (alpha)",
    "H3Trim": "H3 Trim (alpha)",
}
