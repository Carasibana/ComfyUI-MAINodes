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
        return {"required": req, "optional": opt, "hidden": {"unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = ("VIDEO", "INT", "STRING")
    RETURN_NAMES = ("winner_video", "winner_index", "manifest")
    FUNCTION = "compare"
    OUTPUT_NODE = True
    CATEGORY = "image/minimax/video"

    def compare(self, winner=1, preview_crf=23, unique_id=None, **kw):
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
        for i in range(1, 7):
            t["optional"][f"seed_{i}"] = ("INT", {"default": 0, "min": 0, "max": 2**53 - 1,
                                         "tooltip": f"the seed candidate {i} ran on"})
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
