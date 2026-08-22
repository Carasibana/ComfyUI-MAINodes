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
        opt = {}
        for i in range(1, 7):
            opt[f"video_{i}"] = ("VIDEO",)
            opt[f"label_{i}"] = ("STRING", {"default": ""})
        opt["winner"] = ("INT", {"default": 1, "min": 1, "max": 6,
                          "tooltip": "set by the viewer's star; which source winner_video passes through"})
        opt["preview_crf"] = ("INT", {"default": 23, "min": 10, "max": 40})
        return {"required": {}, "optional": opt, "hidden": {"unique_id": "UNIQUE_ID"}}

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


NODE_CLASS_MAPPINGS = {"MAIVideoCompare": MAIVideoCompare}
NODE_DISPLAY_NAME_MAPPINGS = {"MAIVideoCompare": "MAI Video Compare (2-6, synchronized)"}
