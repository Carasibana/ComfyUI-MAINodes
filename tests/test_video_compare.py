#!/usr/bin/env python3
"""MAI Video Compare node side: the viewer extras, and the promise that adding
them changed nothing for a workflow that predates them.

  python tests/test_video_compare.py
      Synthetic, no GPU, no models, no ComfyUI running. folder_paths and
      comfy_api are stubbed; comfy.nested_tensor is stubbed for motion.py,
      which window_expand imports for the burst arithmetic.

What it checks:

  1. HOLD MAP -> SPANS on a real map: the pier graph's H3ManualHoldMap
     (length 243, ranges "132-150", hold 4, ramp on, bridge 8) run through
     the node's converter, which must agree frame for frame with
     window_expand._bursts (the arithmetic is imported, so a drift here
     means someone copied it).
  2. THE MANIFEST carries `spans` and `curves` when they are wired, and
     carries neither key when they are not.
  3. BYTE COMPATIBILITY: a two-video call with no hold_map and no curves
     emits exactly the manifest string the pre-change node emitted, and
     `winner_video` / `winner_index` are untouched.
  4. WIDGET ORDER: INPUT_TYPES before vs after for both classes is
     append-only, including MAISeedHunter, whose seeds must stay ahead of
     the new inputs.

Exit code 0 = pass.
"""
import importlib
import json
import os
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(PACK))
sys.path.insert(0, os.environ.get("COMFYUI_DIR", "/mnt/work/ai/apps/ComfyUI"))

# The pack directory name is not an importable identifier, so bind it to one:
# the modules use relative imports and must be loaded AS a package.
_pkg = types.ModuleType("mainodes_pack")
_pkg.__path__ = [PACK]
sys.modules["mainodes_pack"] = _pkg

if "comfy.nested_tensor" not in sys.modules:      # motion.py's only comfy import
    class _StubNested:
        def __init__(self, tensors, *a, **k):
            self.tensors = tensors

        def unbind(self):
            return self.tensors
    _c = sys.modules.setdefault("comfy", types.ModuleType("comfy"))
    _m = types.ModuleType("comfy.nested_tensor")
    _m.NestedTensor = _StubNested
    _c.nested_tensor = _m
    sys.modules["comfy.nested_tensor"] = _m

vc = importlib.import_module("mainodes_pack.video_compare")
we = importlib.import_module("mainodes_pack.window_expand")
motion = importlib.import_module("mainodes_pack.motion")

FAILS = []


def check(name, ok, detail=""):
    print(("  ok   " if ok else "  FAIL ") + name + (("  " + detail) if detail else ""))
    if not ok:
        FAILS.append(name)


# ---------------------------------------------------------------------------
# 1. hold map -> regenerated-window spans, on the pier graph's real map
# ---------------------------------------------------------------------------
print("1. hold_map -> spans (pier: length 243, ranges '132-150', hold 4, ramp, bridge 8)")
PIER = motion.H3ManualHoldMap().build(length=243, fps=24, ranges="132-150",
                                      hold=4, ramp=True, bridge=8)[0]
holds = json.loads(PIER)["holds"]
spans = vc._spans_from_hold_map(PIER)
runs = []
for i, h in enumerate(holds):
    if not runs or runs[-1][0] != h:
        runs.append([h, i, i])
    else:
        runs[-1][2] = i
print("     holds: " + ", ".join(f"{r[0]}x{r[2] - r[1] + 1}f (f{r[1]}-f{r[2]})" for r in runs))
print("     spans: " + json.dumps(spans))
check("world length preserved", len(holds) == 243, f"{len(holds)}")
check("one regenerated window", len(spans) == 1, json.dumps(spans))
check("span agrees with window_expand._bursts",
      spans == [[a, b] for a, b in we._bursts(holds)],
      json.dumps([list(x) for x in we._bursts(holds)]))
check("span covers every dilated frame and no real-time frame",
      all(holds[f] > 1 for f in range(spans[0][0], spans[0][1] + 1))
      and (spans[0][0] == 0 or holds[spans[0][0] - 1] == 1)
      and (spans[0][1] == 242 or holds[spans[0][1] + 1] == 1))
check("the typed range 132-150 lies inside the span",
      spans[0][0] <= 132 and spans[0][1] >= 150,
      f"typed 132-150, snapped {spans[0][0]}-{spans[0][1]}")

TWO = json.dumps({"holds": [1] * 4 + [3] * 6 + [1] * 5 + [2] * 3 + [1] * 2, "world_len": 20})
check("two bursts -> two spans", vc._spans_from_hold_map(TWO) == [[4, 9], [15, 17]],
      json.dumps(vc._spans_from_hold_map(TWO)))
check("a flat map has no span",
      vc._spans_from_hold_map(json.dumps({"holds": [1] * 39, "world_len": 39})) == [])

# ---------------------------------------------------------------------------
# 2 + 3. the manifest, and the old call byte for byte
# ---------------------------------------------------------------------------
print("2. manifest extras / 3. byte compatibility")


class _Img:
    def __init__(self, n):
        self.shape = (n, 480, 864, 3)


class _Comp:
    def __init__(self, n, fps, audio):
        self.images = _Img(n)
        self.frame_rate = fps
        self.audio = audio


class FakeVideo:
    """Enough VIDEO to drive the node: dimensions, a file write, components."""

    def __init__(self, n=243, fps=24.0, w=864, h=480, audio=None):
        self.n, self.fps, self.w, self.h, self.audio = n, fps, w, h, audio

    def get_dimensions(self):
        return self.w, self.h

    def save_to(self, path, **kw):
        with open(path, "wb") as f:
            f.write(b"\x00")

    def get_components(self):
        return _Comp(self.n, self.fps, self.audio)


TMP = tempfile.mkdtemp(prefix="mai_compare_test_")
_fp = types.ModuleType("folder_paths")
_fp.get_temp_directory = lambda: TMP
sys.modules["folder_paths"] = _fp
_api = types.ModuleType("comfy_api")
_latest = types.ModuleType("comfy_api.latest")
_latest.Types = types.SimpleNamespace(VideoContainer=lambda s: s, VideoCodec=lambda s: s)
_api.latest = _latest
sys.modules["comfy_api"] = _api
sys.modules["comfy_api.latest"] = _latest
vc.time.strftime = lambda f: "121212"          # freeze the preview filename stamp

A, B = FakeVideo(audio=object()), FakeVideo(audio=None)
old = vc.MAIVideoCompare().compare(winner=2, preview_crf=23, unique_id="7",
                                   video_1=A, video_2=B, label_1="ctrl", label_2="warm")
EXPECT = json.dumps({"items": [
    {"index": 1, "label": "ctrl", "filename": "cmp_7_121212_1.mp4", "subfolder": "mai_compare",
     "type": "temp", "frames": 243, "fps": 24.0, "width": 864, "height": 480, "audio": True},
    {"index": 2, "label": "warm", "filename": "cmp_7_121212_2.mp4", "subfolder": "mai_compare",
     "type": "temp", "frames": 243, "fps": 24.0, "width": 864, "height": 480, "audio": False}],
    "winner": 2})
check("old two-video manifest is byte-identical", old["result"][2] == EXPECT,
      "got " + old["result"][2][:70] + "...")
check("old manifest has no new keys", list(json.loads(old["result"][2])) == ["items", "winner"])
check("winner passthrough unchanged", old["result"][0] is B and old["result"][1] == 2)
empty = vc.MAIVideoCompare().compare(winner=2, preview_crf=23, unique_id="7", hold_map="",
                                     curves="", video_1=A, video_2=B, label_1="ctrl", label_2="warm")
check("wiring the new inputs empty changes nothing", empty["result"][2] == EXPECT)

CURVES = json.dumps({"A": [0.0, 0.5, 1.0, 0.25], "B": [1.0, 0.5, 0.0, 0.75]})
wired = vc.MAIVideoCompare().compare(winner=1, preview_crf=23, unique_id="7", hold_map=PIER,
                                     curves=CURVES, video_1=A, video_2=B)
m = json.loads(wired["result"][2])
check("manifest carries spans", m.get("spans") == spans, json.dumps(m.get("spans")))
check("manifest carries curves", m.get("curves") == {"A": [0.0, 0.5, 1.0, 0.25], "B": [1.0, 0.5, 0.0, 0.75]})
check("ui payload is the same dict", wired["ui"]["mai_compare"][0]["spans"] == spans)
junk = vc.MAIVideoCompare().compare(winner=1, preview_crf=23, unique_id="7",
                                    hold_map="{not json", curves="[1,2]", video_1=A, video_2=B)
check("garbage in the new inputs never fails the render",
      list(json.loads(junk["result"][2])) == ["items", "winner"])

sh = vc.MAISeedHunter().compare(winner=2, preview_crf=23, unique_id="9", video_1=A, video_2=B,
                                seed_1=11, seed_2=22, hold_map=PIER, curves=CURVES)
msh = json.loads(sh["result"][3])
check("seed hunter keeps the extras and its seeds",
      msh.get("spans") == spans and [it["seed"] for it in msh["items"]] == [11, 22]
      and sh["result"][1] == 22)

# ---------------------------------------------------------------------------
# 4. widget order audit: append-only, both classes
# ---------------------------------------------------------------------------
print("4. widget order")
OLD_REQ = ["winner", "preview_crf"]
OLD_OPT = [f"video_{i}" for i in range(1, 7)] + [f"label_{i}" for i in range(1, 7)]
OLD_OPT_SH = OLD_OPT + [f"seed_{i}" for i in range(1, 7)]
for cls, old_req, old_opt in ((vc.MAIVideoCompare, OLD_REQ, OLD_OPT),
                              (vc.MAISeedHunter, OLD_REQ, OLD_OPT_SH)):
    t = cls.INPUT_TYPES()
    req, opt = list(t["required"]), list(t["optional"])
    print(f"     {cls.__name__}")
    print(f"       required before {old_req}")
    print(f"       required after  {req}")
    print(f"       optional before {old_opt}")
    print(f"       optional after  {opt}")
    check(f"{cls.__name__}: required untouched", req == old_req)
    check(f"{cls.__name__}: optional is the old order plus a tail",
          opt[:len(old_opt)] == old_opt, f"prefix {opt[:len(old_opt)] == old_opt}")
    check(f"{cls.__name__}: the tail is exactly the two new inputs",
          opt[len(old_opt):] == ["hold_map", "curves"], json.dumps(opt[len(old_opt):]))
    for k in ("hold_map", "curves"):
        spec = t["optional"][k]
        check(f"{cls.__name__}: {k} is an optional forced input with an empty default",
              spec[0] == "STRING" and spec[1].get("forceInput") is True and spec[1].get("default") == "")

print()
print("FAILURES: " + (", ".join(FAILS) if FAILS else "none"))
sys.exit(1 if FAILS else 0)
