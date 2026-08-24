"""MAI Video Compare: a browser-native synchronized viewer for 2-6 renders.

The node does NO video processing. Each wired VIDEO is written once to the temp
directory as a small h264 preview (CPU encode, no VAE, no tensors kept), and the
widget in web/video_compare.js plays them with the flipbook's player: flip /
wipe / side-by-side for a pair, a grid for more, one set of live <video>
elements, audio on the hovered or locked source, synchronized scrub, frame
step, A/B flicker. "Winner" is a widget the UI sets; the node passes that
source through on the NEXT queue (Seed Hunter semantics: choose, then finalize
in a second execution, never pause a graph for a human).
"""
import json
import os
import time
from fractions import Fraction


def _spans_from_hold_map(hold_map):
    """Regenerated-window frame spans (inclusive, world frames) for a hold map.

    A hold above 1 marks a frame H3 Time Smear dilated, which is exactly the
    material a window regenerates, so the spans are the planner's bursts. That
    arithmetic is imported, never restated: one definition of a burst.
    """
    from .window_expand import _bursts
    holds = [int(h) for h in (json.loads(hold_map).get("holds") or [])]
    return [[int(a), int(b)] for a, b in _bursts(holds)]


def _preview_extras(hold_map="", curves=""):
    """The optional manifest keys the widget draws: `spans` (the regenerated
    band, and what the enter/exit blips fire on) and `curves` (per-frame lanes
    under the playhead). Absent inputs add no keys at all, so an old two-video
    call still emits the manifest it always did, byte for byte. Both are
    best-effort: a malformed string costs a line on stdout, never the render.
    """
    out = {}
    if str(hold_map).strip():
        try:
            spans = _spans_from_hold_map(hold_map)
            if spans:
                out["spans"] = spans
        except Exception as e:
            print(f"[MAIVideoCompare] hold_map ignored: {type(e).__name__}: {e}")
    if str(curves).strip():
        try:
            parsed = json.loads(curves)
            lanes = {str(k): [float(x) for x in v] for k, v in parsed.items()
                     if isinstance(v, (list, tuple)) and len(v) > 1}
            if lanes:
                out["curves"] = lanes
        except Exception as e:
            print(f"[MAIVideoCompare] curves ignored: {type(e).__name__}: {e}")
    return out


class MAIVideoCompare:
    DESCRIPTION = (
        "Compare 2-6 renders in the browser: flip / wipe / side-by-side for a "
        "pair, a synchronized grid for more. Hover a source to hear it, click "
        "to lock its audio, scrub or step all together, A/B flicker on a key. "
        "Costs no VRAM: previews are written once as small h264 files and the "
        "browser decodes them. Mark a winner in the widget; the next queue "
        "passes that source through `winner_video` (Seed Hunter: pick, then "
        "finalize in a second execution).")

    @classmethod
    def INPUT_TYPES(cls):
        req = {"winner": ("INT", {"default": 1, "min": 1, "max": 6,
                          "tooltip": "set by the viewer's star; which source winner_video passes through"}),
               "preview_crf": ("INT", {"default": 23, "min": 10, "max": 40})}
        opt = {}
        for i in range(1, 7):
            opt[f"video_{i}"] = ("VIDEO",)
        for i in range(1, 7):
            opt[f"label_{i}"] = ("STRING", {"default": ""})
        opt.update(cls._viewer_inputs())
        return {"required": req, "optional": opt, "hidden": {"unique_id": "UNIQUE_ID"}}

    @staticmethod
    def _viewer_inputs():
        """Viewer-only extras, and they go LAST wherever they are used: an
        input inserted anywhere else shifts every older workflow's widget
        order. MAISeedHunter re-appends them after its seeds for the same
        reason.
        """
        return {
            "hold_map": ("STRING", {"default": "", "forceInput": True,
                         "tooltip": "H3 Time Smear's hold_map_used: the viewer draws the regenerated window as a band and fires enter/exit blips on it"}),
            "curves": ("STRING", {"default": "", "forceInput": True,
                       "tooltip": "JSON {name: [per-frame floats]}; drawn as lanes under the playhead"}),
        }

    RETURN_TYPES = ("VIDEO", "INT", "STRING")
    RETURN_NAMES = ("winner_video", "winner_index", "manifest")
    FUNCTION = "compare"
    OUTPUT_NODE = True
    CATEGORY = "image/minimax/video"

    def compare(self, winner=1, preview_crf=23, unique_id=None, hold_map="",
                curves="", **kw):
        import folder_paths
        from comfy_api.latest import Types
        tmp = folder_paths.get_temp_directory()
        sub = "mai_compare"
        os.makedirs(os.path.join(tmp, sub), exist_ok=True)
        stamp = time.strftime("%H%M%S")
        items, videos = [], {}
        for i in range(1, 7):
            v = kw.get(f"video_{i}")
            if v is None:
                continue
            w, h = v.get_dimensions()
            fn = f"cmp_{unique_id or 'n'}_{stamp}_{i}.mp4"
            path = os.path.join(tmp, sub, fn)
            v.save_to(path, format=Types.VideoContainer("mp4"), codec=Types.VideoCodec("h264"), crf=preview_crf)
            comp = v.get_components()
            items.append({"index": i, "label": kw.get(f"label_{i}") or f"source {i}",
                          "filename": fn, "subfolder": sub, "type": "temp",
                          "frames": int(comp.images.shape[0]), "fps": float(comp.frame_rate),
                          "width": w, "height": h, "audio": comp.audio is not None})
            videos[i] = v
        if not videos:
            raise ValueError("wire at least two videos to compare")
        win = winner if winner in videos else min(videos)
        manifest = {"items": items, "winner": win}
        manifest.update(_preview_extras(hold_map, curves))
        return {"ui": {"mai_compare": [manifest]},
                "result": (videos[win], int(win), json.dumps(manifest))}


class MAISeedHunter(MAIVideoCompare):
    """Video Compare that knows about seeds: cheap candidates in (with the seed
    each ran on), the starred one's seed out, for the finalize pass's noise node.
    Pin / reject are view state in the widget; the decision that matters is the
    star, and it only takes effect on the next queue."""

    DESCRIPTION = (
        "Seed hunting: wire 2-6 cheap candidates and the seed each ran on, watch "
        "them synchronized (hover to hear, flicker, frame-lock), star the keeper. "
        "The next queue passes the starred candidate's seed out of `winner_seed` "
        "(wire it into the finalize pass's RandomNoise) and its video out of "
        "`winner_video`. Pick, then finalize: two executions, never a graph "
        "waiting on a human.")

    @classmethod
    def INPUT_TYPES(cls):
        t = MAIVideoCompare.INPUT_TYPES()
        tail = {k: t["optional"].pop(k) for k in MAIVideoCompare._viewer_inputs()}
        for i in range(1, 7):
            t["optional"][f"seed_{i}"] = ("INT", {"default": 0, "min": 0, "max": 2**53 - 1,
                                         "tooltip": f"the seed candidate {i} ran on"})
        t["optional"].update(tail)      # the viewer extras stay last here too
        return t

    RETURN_TYPES = ("VIDEO", "INT", "INT", "STRING")
    RETURN_NAMES = ("winner_video", "winner_seed", "winner_index", "manifest")

    def compare(self, winner=1, preview_crf=23, unique_id=None, **kw):
        seeds = {i: kw.pop(f"seed_{i}", 0) for i in range(1, 7)}
        for i in range(1, 7):
            if kw.get(f"video_{i}") is not None and not kw.get(f"label_{i}"):
                kw[f"label_{i}"] = f"seed {seeds[i]}"
        res = MAIVideoCompare.compare(self, winner=winner, preview_crf=preview_crf, unique_id=unique_id, **kw)
        video, win, manifest = res["result"]
        m = json.loads(manifest)
        for it in m["items"]:
            it["seed"] = seeds.get(it["index"], 0)
        res["ui"]["mai_compare"] = [m]
        res["result"] = (video, int(seeds.get(win, 0)), win, json.dumps(m))
        return res


NODE_CLASS_MAPPINGS = {"MAIVideoCompare": MAIVideoCompare, "MAISeedHunter": MAISeedHunter}
NODE_DISPLAY_NAME_MAPPINGS = {"MAIVideoCompare": "MAI Video Compare (2-6, synchronized)",
                              "MAISeedHunter": "MAI Seed Hunter (compare + winner seed)"}
