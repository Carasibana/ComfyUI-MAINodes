#!/usr/bin/env python3
"""Unit test for H3 Drawn Plan: a drawn plan document -> hold-map ranges.

  /mnt/work/ai/venvs/comfyui-cu132/bin/python tests/test_drawn_plan.py
      Synthetic, no GPU, no models, no renders. torch is needed because the
      node module draws a preview and the grid law lives in motion.py.

What it checks:

  1. THE ROUND TRIP: a plan document loaded by the node emits ranges that
     drive H3 Manual Hold Map to the SAME hold map as the ranges a human
     would have typed for that shot. Both the file input and the pasted
     string reach the same answer.
  2. SECONDS SURVIVE: every emitted range parses back to the exact world
     frames it came from, at the clip's own fps.
  3. THE NODE IS A LOADER: its window, hold map and length are the
     compiler's, byte-for-byte, not a second opinion computed here.
  4. BAD INPUT IS A SENTENCE: malformed JSON, a missing file, a non-object
     document and an uncompilable plan each raise one readable ValueError
     naming the problem, with no json/traceback vocabulary in the message.
  5. It is registered, so ComfyUI actually shows it.

Exit code 0 = pass.
"""
import inspect
import json
import os
import sys
import tempfile
import types

import torch                                             # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)
sys.path.insert(0, PACK)

# motion.py's node classes import comfy's nested tensor; stub it exactly as
# tests/test_expand_to_end.py does so the suite runs outside ComfyUI.
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


def drawn_plan():
    """What the editor exports for the v3.1 fight-scene shot: real time
    through frame 72, then 2x to the end of a 124-frame clip."""
    plan = schema.new_plan("/tmp/seedhunt_20260963_00001_.mp4", frames=124,
                           fps=24, width=1152, height=640,
                           proposed_by="human:editor",
                           settings={"regen_strength": 0.45, "steps": 25,
                                     "seed": 20260817,
                                     "output_prefix": "video/drawn"})
    plan["lanes"].append(schema.generation_density_lane(
        [[0, 1.0], [72, 1.0], [73, 2.0], [123, 2.0]], ceiling=4.0,
        proposer="human:editor"))
    return plan


def manual_map(ranges, length=124, fps=24, hold=2):
    """The ranges as H3 Manual Hold Map reads them. ramp/bridge off: the
    compiler already shaped the map, and the node's report says so."""
    hold_map, segments, _report = M.H3ManualHoldMap().build(
        length=length, fps=fps, ranges=ranges, hold=hold, ramp=False,
        bridge=0)
    return json.loads(hold_map), segments


def err_of(fn):
    try:
        fn()
    except Exception as e:                                # noqa: BLE001
        return type(e).__name__, str(e)
    return "", ""


def main():
    node = nodes.H3DrawnPlan()
    plan = drawn_plan()

    print("== 1. the round trip: exported plan -> node -> H3 Manual Hold Map")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "drawn.plan.json")
        schema.save(json.loads(json.dumps(plan)), p)
        ranges, length, fps, w0, wlen, report = node.load(p)
        pasted = node.load("", json.dumps(plan))

    print("  ranges: %s" % ranges)
    print("  window: %d..%d (%df), length %d, fps %d" % (w0, w0 + wlen - 1,
                                                         wlen, length, fps))
    check("the file and the pasted document give the same ranges",
          pasted[0] == ranges, ranges)

    typed = "73-123:2"                       # what a human types for this shot
    from_node, seg_node = manual_map(ranges)
    from_typed, seg_typed = manual_map(typed)
    check("node ranges and typed ranges compile to the same hold map",
          from_node == from_typed,
          "%s vs %s" % (h3compile.G.hold_runs_str(from_node["holds"]),
                        h3compile.G.hold_runs_str(from_typed["holds"])))
    check("...and to the same segments string", seg_node == seg_typed,
          seg_node)
    print("  hold map: %s (world_len %d), segments %s"
          % (h3compile.G.hold_runs_str(from_node["holds"]),
             from_node["world_len"], seg_node))

    print("\n== 2. the seconds are the world clock's")
    parsed = []
    for part in ranges.split(","):
        span, _, h = part.strip().rpartition(":")
        a_s, _, b_s = span.partition("-")
        parsed.append((int(round(float(a_s.strip().rstrip("s")) * fps)),
                       int(round(float(b_s.strip().rstrip("s")) * fps)),
                       int(h)))
    check("every emitted range parses back to whole world frames",
          parsed == [(73, 123, 2)], str(parsed))

    print("\n== 3. the node is a loader, not a second compiler")
    B = h3compile.H3Backend()
    art = B.compile(json.loads(json.dumps(plan))).compiled["artifact"]
    check("window comes from the compiler",
          (w0, wlen) == (art["window"]["start"], art["window"]["len"]),
          "%s vs %s" % ((w0, wlen), (art["window"]["start"],
                                     art["window"]["len"])))
    check("the ranges are the compiled hold map, formatted",
          ranges == h3compile.ranges_from_holds(art["hold_map"]["holds"],
                                                art["window"]["start"], fps))
    check("length is the clip's world length, untouched", length == 124,
          str(length))
    check("the report tells the user to turn ramp off",
          "ramp OFF" in report and "bridge 0" in report)
    body = inspect.getsource(nodes.H3DrawnPlan.load)
    check("the node's code carries no grid law: no 17s, no token arithmetic",
          "17" not in body and "token" not in body and "legal" not in body,
          "%d lines" % len(body.splitlines()))

    print("\n== 4. bad input is one readable sentence")
    kind, msg = err_of(lambda: node.load("", "{\"clip\": {\"frames\": 124,}"))
    print("  malformed json -> %s: %s" % (kind, msg))
    check("malformed JSON: a ValueError that names the line and column",
          kind == "ValueError" and "not valid JSON" in msg
          and "line 1" in msg and "\n" not in msg, msg)
    kind, msg = err_of(lambda: node.load("/nonexistent/nope.plan.json"))
    print("  missing file  -> %s: %s" % (kind, msg))
    check("a missing file says so", kind == "ValueError" and "no plan file" in msg,
          msg)
    kind, msg = err_of(lambda: node.load("", "[1, 2, 3]"))
    print("  json array    -> %s: %s" % (kind, msg))
    check("valid JSON that is not a plan says what it is",
          kind == "ValueError" and "not a plan document" in msg, msg)
    kind, msg = err_of(lambda: node.load("", ""))
    check("nothing at all asks for something", kind == "ValueError"
          and "plan_path" in msg, msg)

    flat = drawn_plan()
    flat["lanes"] = []                        # a plan with nothing to compile
    kind, msg = err_of(lambda: node.load("", json.dumps(flat)))
    print("  no lane       -> %s: %s" % (kind, msg))
    check("an uncompilable plan reports the backend's own refusal",
          kind == "ValueError" and "cannot be compiled" in msg
          and "generation_density" in msg, msg)

    print("\n== 5. lanes this backend cannot compile yet")
    with_pin = drawn_plan()
    with_pin["lanes"].append(schema.pin_lane(68, authority=0.6))
    kind, msg = err_of(lambda: node.load("", json.dumps(with_pin)))
    print("  pin, default  -> %s: %s" % (kind, msg))
    check("a pin refuses by default: nothing is dropped behind your back",
          kind == "ValueError" and "pin" in msg, msg)
    r2 = node.load("", json.dumps(with_pin), ignore_uncompiled_lanes=True)
    check("...and compiles to the same ranges when you opt in",
          r2[0] == ranges, "%r vs %r" % (r2[0], ranges))
    check("...saying out loud which lane it skipped",
          "SKIPPED" in r2[5] and "pin" in r2[5],
          "; ".join(l for l in r2[5].splitlines() if l.startswith("SKIPPED")))

    print("\n== 6. registration")
    check("H3DrawnPlan is registered with a display name",
          nodes.NODE_CLASS_MAPPINGS.get("H3DrawnPlan") is nodes.H3DrawnPlan
          and "H3DrawnPlan" in nodes.NODE_DISPLAY_NAME_MAPPINGS)

    print()
    if FAILS:
        print(f"FAILED: {len(FAILS)} check(s): " + ", ".join(FAILS))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
