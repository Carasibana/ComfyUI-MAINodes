#!/usr/bin/env python3
"""Two doors, one answer: the node surface vs the compiler's minted graph.

  /mnt/work/ai/venvs/comfyui-cu132/bin/python tests/test_node_parity.py
      Synthetic, no GPU, no models, no renders.

A plan can reach a render two ways: the compiler mints a whole graph from
it (H3 Timeline Render), or a graph author wires the plan's compiled
answers into stock H3 nodes (H3 Drawn Plan / H3 Plan Settings / H3 Plan
Estimate). Those two doors must produce the same execution. This suite
compares the node outputs against the FIELDS OF THE MINTED GRAPH itself,
not against a second calculation, so a drift in either door fails here.

What it checks:

  1. GEOMETRY PARITY: hold map, window start/len, dilated length and guide
     frame come out of the nodes exactly as the compiler wrote them into
     the graph it mints.
  2. SETTINGS PARITY: inject, steps, seed, prompt, width, height and the
     delivery prefix match the graph's widgets.
  3. COST PARITY: the estimate node agrees with the compiled artifact, and
     abstains (-1) rather than inventing minutes.
  4. SPLICE PARITY: the splice_map the node emits describes the same three
     pieces the minted graph's ImageFromBatch/ImageBatch chain splices,
     and drives H3 Segment Splice at feather 0 to the same frames.
  5. THE RANGES ROUTE'S LIMIT, measured rather than assumed: a flat-rate
     map survives the H3 Manual Hold Map round trip, a DITHERED one does
     not, which is why hold_map exists.
  6. THE THIN-LOADER LAW extends to every new node: no grid law, no token
     arithmetic, no 17s in any node body.

Exit code 0 = pass.
"""
import inspect
import json
import os
import sys
import types

import torch                                             # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)
sys.path.insert(0, PACK)

if "comfy.nested_tensor" not in sys.modules:
    class _StubNested:
        def __init__(self, tensors):
            self.tensors = list(tensors)
            self.is_nested = True

        def unbind(self):
            return self.tensors

    _pkg = sys.modules.setdefault("comfy", types.ModuleType("comfy"))
    _mod = types.ModuleType("comfy.nested_tensor")
    _mod.NestedTensor = _StubNested
    _pkg.nested_tensor = _mod
    sys.modules["comfy.nested_tensor"] = _mod

from timeline import nodes, schema                        # noqa: E402
from timeline.h3 import compile as h3compile              # noqa: E402
from timeline.h3.gridlaw import motion as M               # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(("  PASS  " if ok else "  FAIL  ") + name + ("  " + detail if detail else ""))
    if not ok:
        FAILS.append(name)


def out(node_cls, plan, **kw):
    """Node outputs as a name -> value dict."""
    vals = node_cls().load("", json.dumps(plan), **kw)
    return dict(zip(node_cls.RETURN_NAMES, vals))


def a_plan(envelope, frames=124, **settings):
    s = {"regen_strength": 0.45, "steps": 25, "seed": 20260817,
         "prompt": "a shot", "output_prefix": "video/parity"}
    s.update(settings)
    plan = schema.new_plan("clip.mp4", frames=frames, fps=24, width=1152,
                           height=640, proposed_by="test", settings=s)
    plan["lanes"].append(schema.generation_density_lane(envelope, ceiling=8.0))
    return plan


CASES = {
    "windowed 2x tail": a_plan([[0, 1.0], [72, 1.0], [73, 2.0], [123, 2.0]]),
    "uniform 2.5x (dithered)": a_plan([[0, 2.5], [123, 2.5]]),
    "mid-clip 6x burst": a_plan([[0, 1.0], [67, 1.0], [68, 6.0], [90, 6.0],
                                 [91, 1.0], [123, 1.0]]),
    "tiny burst (region widens)": a_plan([[0, 1.0], [20, 1.0], [21, 2.0],
                                          [30, 2.0], [31, 1.0], [123, 1.0]]),
    "other settings": a_plan([[0, 1.0], [50, 1.0], [51, 3.0], [90, 3.0],
                              [91, 1.0], [123, 1.0]],
                             regen_strength=0.30, steps=8, seed=7,
                             prompt="another shot",
                             output_prefix="video/other"),
}


def main():
    B = h3compile.H3Backend()

    print("1-2. GEOMETRY AND SETTINGS PARITY vs the minted graph")
    for name, plan in CASES.items():
        res = B.compile(json.loads(json.dumps(plan)))
        g, art = res.graph, res.compiled["artifact"]
        d = out(nodes.H3DrawnPlan, plan)
        s = out(nodes.H3PlanSettings, plan)
        geometry = {
            "hold_map": (json.loads(d["hold_map"]),
                         json.loads(g["404"]["inputs"]["hold_map"])),
            "window_start": (d["window_start"], g["410"]["inputs"]["batch_index"]),
            "window_len": (d["window_len"], g["410"]["inputs"]["length"]),
            "dilated_frames": (d["dilated_frames"], g["524"]["inputs"]["length"]),
            "guide_frame": (d["guide_frame"], g["406"]["inputs"]["frame_idx"]),
        }
        settings = {
            "inject": (s["inject"], g["522"]["inputs"]["inject"]),
            "steps": (s["steps"], g["522"]["inputs"]["total_steps"]),
            "seed": (s["seed"], g["523"]["inputs"]["noise_seed"]),
            "prompt": (s["prompt"], g["524"]["inputs"]["prompt"]),
            "width": (s["width"], g["524"]["inputs"]["width"]),
            "height": (s["height"], g["524"]["inputs"]["height"]),
            "output_prefix": (s["output_prefix"],
                              g["541"]["inputs"]["filename_prefix"]),
        }
        bad = {k: v for k, v in dict(geometry, **settings).items()
               if v[0] != v[1]}
        check(f"  {name}: every wired value equals the minted graph's",
              not bad, str(bad)[:160] if bad else
              f"window {d['window_start']}+{d['window_len']}, "
              f"{d['dilated_frames']}f dilated, guide {d['guide_frame']}, "
              f"inject {s['inject']:g}/{s['steps']}st, seed {s['seed']}")
        check(f"  {name}: length and fps are the clip's, untouched",
              (d["length"], d["fps"]) == (int(plan["clip"]["frames"]),
                                          int(plan["clip"]["fps"])),
              f"{d['length']}f @{d['fps']}")
        # the artifact is the other authority: the node must not disagree
        check(f"  {name}: ...and equal the compiled artifact",
              json.loads(d["hold_map"]) == art["hold_map"]
              and d["window_start"] == art["window"]["start"]
              and d["dilated_frames"] == art["dilated_frames"]
              and d["guide_frame"] == art["guide_dilated_idx"])

    print("\n3. COST PARITY")
    for name, plan in list(CASES.items())[:3]:
        res = B.compile(json.loads(json.dumps(plan)))
        art = res.compiled["artifact"]
        e = out(nodes.H3PlanEstimate, plan)
        check(f"  {name}: equivalent clip time matches the artifact",
              round(e["equivalent_clip_time"], 6)
              == round(art["equivalent_clip_time_x"], 6),
              f"{e['equivalent_clip_time']:.3f}x vs "
              f"{art['equivalent_clip_time_x']:.3f}x")
        check(f"  {name}: work units are the dilated token count",
              e["work_units"] == art["tokens"]["dilated"],
              f"{e['work_units']} vs {art['tokens']['dilated']}")
    e = out(nodes.H3PlanEstimate, CASES["windowed 2x tail"],
            recorder_path="/nonexistent/recorder.jsonl")
    check("uncalibrated seconds ABSTAINS at -1 instead of guessing",
          e["seconds"] == -1.0 and "abstain" in e["report"],
          f"seconds={e['seconds']}")

    print("\n4. SPLICE PARITY")
    for name in ("windowed 2x tail", "mid-clip 6x burst"):
        plan = CASES[name]
        res = B.compile(json.loads(json.dumps(plan)))
        g = res.graph
        d = out(nodes.H3DrawnPlan, plan)
        sp = json.loads(d["splice_map"])
        # what the minted graph splices: head, window crop, tail
        pieces = {nid: (n["inputs"]["batch_index"], n["inputs"]["length"])
                  for nid, n in g.items()
                  if n["class_type"] == "ImageFromBatch" and nid != "405"}
        head = pieces.get("411", (0, 0))
        tail = pieces.get("414", (sp["end"] + 1, 0))
        check(f"  {name}: splice_map names the same window the graph crops",
              (sp["start"], sp["end"]) == (pieces["410"][0],
                                           pieces["410"][0] + pieces["410"][1] - 1)
              and sp["world_len"] == int(plan["clip"]["frames"]),
              str(sp))
        check(f"  {name}: head + window + tail tile the clip exactly",
              head[1] + pieces["410"][1] + tail[1] == sp["world_len"]
              and head[0] == 0 and (tail[1] == 0 or tail[0] == sp["end"] + 1),
              f"head {head}, window {pieces['410']}, tail {tail}")
        # and the splice node reproduces that tiling on real tensors
        base = torch.arange(sp["world_len"], dtype=torch.float32
                            ).reshape(-1, 1, 1, 1).repeat(1, 2, 2, 3)
        seg = torch.full((pieces["410"][1], 2, 2, 3), -1.0)
        spliced = M.H3SegmentSplice().splice(baseline=base, segment=seg,
                                             splice_map=d["splice_map"],
                                             feather_frames=0)[0]
        inside = [i for i in range(sp["world_len"])
                  if float(spliced[i, 0, 0, 0]) == -1.0]
        check(f"  {name}: H3 Segment Splice at feather 0 replaces exactly "
              f"the window",
              inside == list(range(sp["start"], sp["end"] + 1))
              and spliced.shape[0] == sp["world_len"],
              f"{len(inside)} frames replaced, {spliced.shape[0]} out")
        # THE ONLY RUNTIME DIFFERENCE between the two doors, settled on
        # tensors instead of on a render: the compiler route cats three
        # slices, the node route splices; the frames must be bit-equal.
        seg2 = torch.rand(pieces["410"][1], 4, 4, 3)
        base2 = torch.rand(sp["world_len"], 4, 4, 3)
        parts = [base2[:head[1]]] if head[1] else []
        parts.append(seg2)
        if tail[1]:
            parts.append(base2[tail[0]:tail[0] + tail[1]])
        compiler_route = torch.cat(parts, dim=0)
        node_route = M.H3SegmentSplice().splice(
            baseline=base2, segment=seg2, splice_map=d["splice_map"],
            feather_frames=0)[0]
        check(f"  {name}: node-route splice is BIT-EQUAL to the compiler's cat",
              node_route.shape == compiler_route.shape
              and torch.equal(node_route, compiler_route),
              f"{tuple(node_route.shape)} vs {tuple(compiler_route.shape)}, "
              f"max abs diff "
              f"{float((node_route - compiler_route).abs().max()):.3g}")

    print("\n5. THE RANGES ROUTE'S LIMIT (measured, not assumed)")
    for name, exact in (("windowed 2x tail", True),
                        ("uniform 2.5x (dithered)", False)):
        plan = CASES[name]
        d = out(nodes.H3DrawnPlan, plan)
        want = json.loads(d["hold_map"])["holds"]
        got = json.loads(M.H3ManualHoldMap().build(
            length=d["length"], fps=d["fps"], ranges=d["ranges"], hold=2,
            ramp=False, bridge=0)[0])["holds"]
        got_window = got[d["window_start"]:d["window_start"] + d["window_len"]]
        same = got_window == want
        check(f"  {name}: ranges route reproduces the map = {exact}",
              same is exact,
              f"{M._hold_runs_str(want)[:34]} vs {M._hold_runs_str(got_window)[:34]}")
    check("hold_map is the route that always reproduces it",
          all(json.loads(out(nodes.H3DrawnPlan, p)["hold_map"])["holds"]
              == B.compile(json.loads(json.dumps(p)))
              .compiled["artifact"]["hold_map"]["holds"]
              for p in CASES.values()))

    print("\n6. THE THIN-LOADER LAW, on every node in the surface")
    for cls in (nodes.H3DrawnPlan, nodes.H3PlanSettings, nodes.H3PlanEstimate):
        body = inspect.getsource(cls.load)
        check(f"  {cls.__name__}.load carries no grid law",
              "17" not in body and "token" not in body and "legal" not in body
              and "39" not in body,
              f"{len(body.splitlines())} lines")
    helper = inspect.getsource(nodes._compile_for_nodes)
    check("  the shared loader carries none either",
          "17" not in helper and "token" not in helper and "39" not in helper)
    for name in ("H3DrawnPlan", "H3PlanSettings", "H3PlanEstimate"):
        check(f"  {name} is registered with a display name",
              nodes.NODE_CLASS_MAPPINGS.get(name) is getattr(nodes, name)
              and name in nodes.NODE_DISPLAY_NAME_MAPPINGS)

    print()
    if FAILS:
        print(f"FAILED: {len(FAILS)} check(s): " + ", ".join(FAILS))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
