"""De-rope for any model: retime one hold map onto a target model's clock.

  H3ClockRemap   hold_map (from H3 Jerk Oracle / H3 Manual Hold Map / H3 Drawn
                 Plan) + a model profile -> hold_map on that model's clock, with
                 the target grid carried INSIDE the map so H3 Time Smear pads to
                 the right legal length and H3 Exact Recover inverts it exactly.
  H3SaveHoldMap  write the hold map beside the render (<prefix>.holdmap.json) so
                 the staircase ruler and the next remap can read what was held.

The smear / recover / audio nodes are pixel-level and model-agnostic already;
this module is the grid law lifted out of them into data (model_profiles.py).
"""
import json
import os

try:
    from .model_profiles import (load_profiles, normalize, remap_holds, pad_to_legal,
                                 report as _report)
except ImportError:                              # tests import this top-level
    from model_profiles import (load_profiles, normalize, remap_holds, pad_to_legal,
                                report as _report)

CUSTOM = "custom (use the fields below)"


class H3ClockRemap:
    DESCRIPTION = (
        "EXPERIMENTAL (alpha), new 2026-08-21. Retimes a hold map onto another "
        "model's clock.\n\n"
        "Wire any hold map (H3 Jerk Oracle, H3 Manual Hold Map, H3 Drawn Plan) "
        "and pick the model that will regenerate the smeared frames. Holds of "
        "2+ are scaled by the profile's hold_scale and quantised to whole "
        "latent time blocks; holds of 1 stay 1 (plain video on any model). The "
        "output map carries the target's legal frame grid, so H3 Time Smear "
        "pads to THAT grid (LTX-2.5: 8k+1) instead of H3's 17k+5, and H3 "
        "Exact Recover / H3 Audio Recover invert it unchanged.\n\n"
        "'minimax-h3' is the identity. 'custom' exposes the fields for a model "
        "that has no preset yet; a working custom row can be saved as a preset "
        "by adding it to <user dir>/mainodes_models.json (see "
        "DEROPE_ANY_MODEL.md). The report is the price tag and says when the "
        "profile is unmeasured or the retime needs more than one pass.")

    @classmethod
    def INPUT_TYPES(cls):
        profiles = load_profiles()
        return {"required": {
            "hold_map": ("STRING", {"default": "", "forceInput": True,
                         "tooltip": "per-frame integer holds, oracle format"}),
            "model_profile": (list(profiles) + [CUSTOM], {"default": "ltx-2.5"}),
        }, "optional": {
            "custom_block": ("INT", {"default": 8, "min": 1, "max": 64,
                             "tooltip": "custom: frames per latent time block"}),
            "custom_hold_scale": ("INT", {"default": 4, "min": 1, "max": 32,
                                  "tooltip": "custom: world frames one oracle hold unit buys"}),
            "custom_legal_step": ("INT", {"default": 8, "min": 1, "max": 64,
                                  "tooltip": "custom: legal lengths are step*k + offset"}),
            "custom_legal_offset": ("INT", {"default": 1, "min": 0, "max": 64}),
            "custom_fps": ("FLOAT", {"default": 25.0, "min": 1.0, "max": 120.0, "step": 0.5}),
            "custom_cap_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 600.0, "step": 0.5,
                                   "tooltip": "custom: longest single pass in seconds; 0 = no cap"}),
        }}

    RETURN_TYPES = ("STRING", "INT", "STRING", "STRING")
    RETURN_NAMES = ("hold_map", "length", "report", "profile_json")
    FUNCTION = "remap"
    CATEGORY = "image/minimax/motion"

    def remap(self, hold_map, model_profile, custom_block=8, custom_hold_scale=4,
              custom_legal_step=8, custom_legal_offset=1, custom_fps=25.0,
              custom_cap_seconds=0.0):
        m = json.loads(hold_map)
        holds_in = [int(h) for h in m["holds"]]
        if model_profile == CUSTOM:
            prof = normalize("custom", dict(
                name="custom", block=custom_block, hold_scale=custom_hold_scale,
                legal=(custom_legal_step, custom_legal_offset), fps=custom_fps,
                cap_seconds=custom_cap_seconds or None, measured=False,
                note="typed into H3 Clock Remap"))
        else:
            profiles = load_profiles()
            if model_profile not in profiles:
                raise ValueError(f"unknown model profile {model_profile!r}; "
                                 f"known: {sorted(profiles)}")
            prof = profiles[model_profile]
        holds_out = remap_holds(holds_in, prof)
        holds_pad, total, pad = pad_to_legal(holds_out, prof["legal"])
        rep = _report(holds_in, holds_pad, total, pad, prof)
        out = {"holds": holds_pad, "world_len": m.get("world_len", len(holds_in)),
               "profile": prof["id"], "legal": list(prof["legal"]), "fps": prof["fps"],
               "source_holds": holds_in}
        print("[MAINodes] H3ClockRemap " + rep.replace("\n", " | "))
        return (json.dumps(out), int(total), rep, json.dumps(prof))


class H3SaveHoldMap:
    DESCRIPTION = (
        "Writes a hold map beside the render as <output>/<filename_prefix>"
        ".holdmap.json. Wire H3 Time Smear's hold_map_used (what was actually "
        "held, after padding) and the same filename_prefix as the SaveVideo of "
        "that arm. The file is what the staircase ruler and a later H3 Clock "
        "Remap read; without it a render's clock is lost the moment the graph "
        "closes.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "hold_map": ("STRING", {"default": "", "forceInput": True}),
            "filename_prefix": ("STRING", {"default": "video/derope"}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "image/minimax/motion"

    def save(self, hold_map, filename_prefix):
        import folder_paths
        root = folder_paths.get_output_directory()
        base = os.path.join(root, filename_prefix + ".holdmap")
        os.makedirs(os.path.dirname(base), exist_ok=True)
        path, k = base + ".json", 1
        while os.path.exists(path):
            k += 1
            path = f"{base}_{k:05d}.json"
        data = json.loads(hold_map)
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"[MAINodes] H3SaveHoldMap -> {path}")
        return {"ui": {"text": [os.path.relpath(path, root)]}, "result": (path,)}


NODE_CLASS_MAPPINGS = {"H3ClockRemap": H3ClockRemap, "H3SaveHoldMap": H3SaveHoldMap}
NODE_DISPLAY_NAME_MAPPINGS = {"H3ClockRemap": "H3 Clock Remap (any model, presets)",
                              "H3SaveHoldMap": "H3 Save Hold Map (sidecar)"}
