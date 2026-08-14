"""The two ComfyUI nodes of the timeline surface, plus the recorder pair.

Two-phase execution, no mid-graph pause (spec): ANALYZE is cheap and writes
plan.json; COMPILE+RENDER is phase 2. The auto checkbox is phase 2's entry:
auto = accept the proposal verbatim and compile immediately (the zero-click
mode), off = the plan is written and you open an editor.

These nodes are THIN CLIENTS. Every rule lives in timeline/ (semantics) and
timeline/h3/ (the model's reality); this file only moves strings around and
draws a preview.
"""
import json
import os

import torch

from . import price, recorder, schema
from .h3 import propose
from .h3.compile import H3Backend
# through gridlaw's loader, not `from ..motion import ANY`: the same import
# then works whether the pack is loaded as a ComfyUI package or as a plain
# directory on sys.path (which is how the suites load it).
from .h3.gridlaw import motion as _motion

ANY = _motion.ANY

DEFAULT_PLAN_DIR = "/mnt/work/ai/apps/ComfyUI/output/timeline"
DEFAULT_GRAPH_DIR = "/mnt/work/ai/apps/ComfyUI-ModelCatalog/workflows"


def _strip(plan, compiled=None, height=160, width=None):
    """Preview strip: the drawn density envelope over the clip, with the
    compiled hold map underneath so the quantization is visible (spec: the
    UI draws the quantization honestly)."""
    frames = int(plan["clip"]["frames"])
    w = int(width or max(256, min(1024, frames * 4)))
    lanes = schema.density_lanes(plan)
    env = lanes[0]["envelope"] if lanes else [[0, 1.0]]
    ceiling = float(lanes[0].get("ceiling", 4.0)) if lanes else 4.0
    img = torch.zeros(height, w, 3)
    img[:, :, 2] = 0.06
    for gy in range(1, int(ceiling) + 1):          # density gridlines
        y = height - 1 - int((gy - 1) / max(ceiling - 1, 1e-9) * (height - 1))
        img[max(0, min(height - 1, y)), :, :] = 0.18
    for x in range(w):
        f = int(x / float(w) * frames)
        r = schema.sample_envelope(env, f)
        y = height - 1 - int((r - 1.0) / max(ceiling - 1.0, 1e-9) * (height - 1))
        y = max(0, min(height - 1, y))
        img[y:, x, 0] = 0.85                       # filled area, red
        img[y:, x, 1] = 0.35
        img[y, x, :] = torch.tensor([1.0, 0.85, 0.2])
    if compiled:
        art = compiled.get("artifact", compiled)
        w0 = art["window"]["start"]
        wl = art["window"]["len"]
        holds = art["hold_map"]["holds"]
        hmax = float(max(holds))
        band = max(6, height // 8)
        for x in range(w):
            f = int(x / float(w) * frames)
            if w0 <= f < w0 + wl:
                h = holds[f - w0]
                v = 0.25 + 0.75 * (h / hmax)
                img[height - band:, x, 0] = v
                img[height - band:, x, 1] = v * 0.6
                img[height - band:, x, 2] = 0.1
        for edge in (w0, w0 + wl - 1):             # window boundaries
            x = min(w - 1, int(edge / float(frames) * w))
            img[:, x, :] = torch.tensor([0.2, 0.9, 1.0])
    return img[None]


class H3TimelineAnalyze:
    """Phase 1: oracle -> plan.json (+ auto-compile)."""

    DESCRIPTION = (
        "EXPERIMENTAL (alpha), new 2026-08-14. The timeline surface's front "
        "half. Wire an oracle's profile output (H3 Jerk Oracle and/or H3 "
        "Indecision Oracle) and this writes a PLAN DOCUMENT: a per-clip JSON "
        "that says, in dilation ratios over time, how the shot should be "
        "retimed. The plan is the source of truth; a graph is only ever "
        "compiled FROM it.\n\n"
        "Both oracles wired is the validated combination: jerk catches fast "
        "PROPS, indecision catches detail the model is still arguing with "
        "itself about, and they disagree in both directions, so the proposal "
        "takes the max of the two rank-normalized profiles.\n\n"
        "auto=True is the zero-click mode: the proposal is compiled straight "
        "into a minted graph and the report prices it. auto=False stops at "
        "the plan, for editing in the widget or the web editor.\n\n"
        "The density ceiling is yours to raise. 4x is the measured sweet spot, "
        "not a wall: if you want 8x across the whole clip and are cool "
        "spending the time, raise it and read the price meter.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "clip_path": ("STRING", {"default": "",
                          "tooltip": "the source clip this plan is about (its file name is what the compiled graph's LoadVideo will use)"}),
            "frames": ("INT", {"default": 124, "min": 39, "max": 3600,
                       "tooltip": "world length of the clip, on the 17k+5 grid; wire H3 Video Fit's length"}),
            "plan_path": ("STRING", {"default": "",
                          "tooltip": "where to write plan.json; empty = output/timeline/<clip>.plan.json"}),
            "auto": ("BOOLEAN", {"default": True,
                     "tooltip": "True: accept the proposal and compile it immediately (zero click). False: write the plan and stop"}),
        }, "optional": {
            "jerk_profile": ("STRING", {"forceInput": True,
                             "tooltip": "H3 Jerk Oracle's profile output"}),
            "indecision_profile": ("STRING", {"forceInput": True,
                                    "tooltip": "H3 Indecision Oracle's profile output"}),
            "ceiling": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 16.0, "step": 0.5,
                        "tooltip": "the lane's top end in generated frames per world frame; the output is NOT slower or longer, it is generated denser. The price meter is what makes a high ceiling safe"}),
            "threshold": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 0.99, "step": 0.01,
                          "tooltip": "rank-scale saliency below which the clip stays at real time (the oracle's q, on a scale both signals share)"}),
            "prompt": ("STRING", {"default": "", "multiline": True}),
            "width": ("INT", {"default": 1152, "min": 64, "max": 4096, "step": 32}),
            "height": ("INT", {"default": 640, "min": 64, "max": 4096, "step": 32}),
            "regen_strength": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01,
                               "tooltip": "how much of the denoise schedule the regeneration gets (H3's inject)"}),
            "steps": ("INT", {"default": 25, "min": 1, "max": 100}),
            "seed": ("INT", {"default": 20260817, "min": 0, "max": 0xffffffffffffffff}),
            "output_prefix": ("STRING", {"default": "video/timeline"}),
            "graph_path": ("STRING", {"default": "",
                           "tooltip": "where auto-compile mints the .api.json; empty = workflows/<plan name>.api.json"}),
        }}

    RETURN_TYPES = ("STRING", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("plan_path", "preview", "graph_path", "report")
    FUNCTION = "analyze"
    CATEGORY = "latent/minimax/timeline"

    def analyze(self, clip_path, frames, plan_path, auto,
                jerk_profile=None, indecision_profile=None, ceiling=4.0,
                threshold=0.75, prompt="", width=1152, height=640,
                regen_strength=0.45, steps=25, seed=20260817,
                output_prefix="video/timeline", graph_path=""):
        plan = schema.new_plan(
            clip_path, frames=frames, fps=24.0, width=width, height=height,
            proposed_by="H3TimelineAnalyze",
            settings={"prompt": prompt, "regen_strength": float(regen_strength),
                      "steps": int(steps), "seed": int(seed),
                      "output_prefix": output_prefix})
        lane = propose.propose_density_lane(
            frames, jerk=jerk_profile, indecision=indecision_profile,
            ceiling=float(ceiling), threshold=float(threshold))
        plan["lanes"].append(lane)
        schema.note_edit(plan, "H3TimelineAnalyze",
                         f"proposed temporal lane from {lane['proposer']}")
        schema.validate_or_raise(plan)

        base = os.path.splitext(os.path.basename(clip_path or "clip"))[0]
        plan_path = plan_path or os.path.join(DEFAULT_PLAN_DIR,
                                              f"{base}.plan.json")
        peak = max(r for _f, r in lane["envelope"])
        ch = propose.challenge_ceiling(peak)
        lines = [f"plan {plan['id'][:8]} rev {plan['revision']}: "
                 f"{len(lane['envelope'])} control points, peak {peak:.2f}x, "
                 f"ceiling {ceiling:g}, proposer {lane['proposer']}"]
        if ch:
            plan.setdefault("provenance", {}).setdefault(
                "flags", []).append(ch)
            lines.append(f"RECIPE CHALLENGE (flagged, not overridden): "
                         f"{ch['why']}")

        compiled, out_graph = None, ""
        if auto:
            out_graph = graph_path or os.path.join(
                DEFAULT_GRAPH_DIR, f"{base}_timeline.api.json")
            res = H3Backend().compile(plan, {"graph_path": out_graph})
            compiled = res.compiled
            lines.append(res.report)
        schema.save(plan, plan_path)
        lines.insert(1, f"plan written: {plan_path}")
        if out_graph:
            lines.insert(2, f"graph minted: {out_graph}")
        return (plan_path, _strip(plan, compiled), out_graph, "\n".join(lines))


class H3TimelineRender:
    """Phase 2: plan.json -> a minted graph, priced.

    SHIPPED IN v0: this node COMPILES and MINTS; it does not execute the
    compiled graph in-process. Execution stays with the launcher
    (queue_scene.py), which is the project's only sanctioned way to start a
    render and carries the collision and node_error guards. Wiring the
    compiled section into a live sampler inside one graph is T2 work.
    """

    DESCRIPTION = (
        "EXPERIMENTAL (alpha), new 2026-08-14. Phase 2 of the timeline "
        "surface: read a plan.json (yours, the oracle's, or one somebody "
        "posted), compile it to a legal H3 graph, and price it.\n\n"
        "v0 emits the minted graph PATH and the exact launch line rather "
        "than executing it in place: launching belongs to queue_scene.py, "
        "which guards output-prefix collisions and refuses partial graphs.\n\n"
        "If the plan was edited after it was last compiled, this recompiles "
        "it — the compiled block is a cache keyed to the plan's revision, "
        "never a second source of truth.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "plan_path": ("STRING", {"default": "", "forceInput": False}),
        }, "optional": {
            "graph_path": ("STRING", {"default": "",
                           "tooltip": "empty = workflows/<plan name>.api.json"}),
            "port": ("INT", {"default": 8189, "min": 8188, "max": 8199,
                     "tooltip": "instance the launch line targets, and whose /system_stats the VRAM fit line reads"}),
            "recorder_path": ("STRING", {"default": recorder.DEFAULT_PATH,
                              "tooltip": "flight-recorder jsonl; layer (b) of the price meter calibrates from YOUR runs only"}),
        }}

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("graph_path", "launch_command", "report")
    FUNCTION = "render"
    CATEGORY = "latent/minimax/timeline"

    def render(self, plan_path, graph_path="", port=8189,
               recorder_path=recorder.DEFAULT_PATH):
        plan = schema.load(plan_path)
        base = os.path.splitext(os.path.basename(plan_path))[0]
        base = base.replace(".plan", "")
        graph_path = graph_path or os.path.join(DEFAULT_GRAPH_DIR,
                                                f"{base}_timeline.api.json")
        stale = schema.compiled_is_stale(plan, H3Backend.backend_id)
        res = H3Backend().compile(plan, {"graph_path": graph_path,
                                         "port": int(port),
                                         "recorder_path": recorder_path})
        schema.save(plan, plan_path)
        cmd = ("/mnt/work/ai/venvs/comfyui-cu132/bin/python "
               "/mnt/work/ai/apps/ComfyUI-ModelCatalog/benchmarks/scripts/"
               f"queue_scene.py {graph_path} --tag timeline --port {int(port)}")
        report = "\n".join([
            f"plan {plan.get('id', '?')[:8]} rev {plan.get('revision', 0)} "
            f"({'recompiled after an edit' if stale else 'cache was current'})",
            res.report,
            f"minted: {graph_path}",
            "launch with:  " + cmd,
        ])
        return (graph_path, cmd, report)


class H3RecordStart:
    """Flight recorder, front half: reset the torch peak counters."""

    DESCRIPTION = ("Alpha. Drop this in front of the expensive node and H3 "
                   "Record Stop after it. Together they append ONE local "
                   "JSON line per render (work units, resolution, model tag, "
                   "s/step, torch peak bytes) that the timeline's price meter "
                   "calibrates from. Local file, nothing phones home.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"anything": (ANY, {}),
                             "key": ("STRING", {"default": "default"})}}

    RETURN_TYPES = (ANY,)
    RETURN_NAMES = ("anything",)
    FUNCTION = "go"
    CATEGORY = "latent/minimax/timeline"

    def go(self, anything, key="default"):
        recorder.start(key)
        return (anything,)


class H3RecordStop:
    """Flight recorder, back half: read the peaks and append the line."""

    DESCRIPTION = H3RecordStart.DESCRIPTION

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "anything": (ANY, {}),
            "key": ("STRING", {"default": "default"}),
            "work_units": ("INT", {"default": 0, "min": 0, "max": 100000,
                           "tooltip": "latent tokens this run generated (the compiled plan reports it)"}),
            "steps": ("INT", {"default": 25, "min": 1, "max": 1000}),
        }, "optional": {
            "pixels": ("INT", {"default": 0, "min": 0, "max": 1 << 30}),
            "frames": ("INT", {"default": 0, "min": 0, "max": 10000}),
            "model": ("STRING", {"default": "minimax-h3-int8-convrot"}),
            "path": ("STRING", {"default": recorder.DEFAULT_PATH}),
            "log_path": ("STRING", {"default": "",
                         "tooltip": "ComfyUI's log file, if you have one: the streamed flag is READ from its partial-load lines, never guessed"}),
        }}

    RETURN_TYPES = (ANY, "STRING")
    RETURN_NAMES = ("anything", "record")
    FUNCTION = "go"
    CATEGORY = "latent/minimax/timeline"

    def go(self, anything, key="default", work_units=0, steps=25, pixels=0,
           frames=0, model="minimax-h3-int8-convrot",
           path=recorder.DEFAULT_PATH, log_path=""):
        rec = recorder.stop(key, work_units=work_units, steps=steps,
                            pixels=pixels, frames=frames, model=model,
                            path=path, log_path=log_path or None)
        return (anything, json.dumps(rec))


NODE_CLASS_MAPPINGS = {
    "H3TimelineAnalyze": H3TimelineAnalyze,
    "H3TimelineRender": H3TimelineRender,
    "H3RecordStart": H3RecordStart,
    "H3RecordStop": H3RecordStop,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3TimelineAnalyze": "H3 Timeline Analyze (alpha)",
    "H3TimelineRender": "H3 Timeline Render (alpha)",
    "H3RecordStart": "H3 Flight Recorder Start (alpha)",
    "H3RecordStop": "H3 Flight Recorder Stop (alpha)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "price"]
