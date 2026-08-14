#!/usr/bin/env python3
"""Unit test for the timeline surface: plan schema, backend seam, compiler.

  /mnt/work/ai/venvs/comfyui-cu132/bin/python tests/test_timeline_compile.py
      Synthetic, no GPU, no models, no renders. torch is needed only
      because timeline/h3/gridlaw.py reaches into motion.py for the grid
      law (deliberately: reuse it, never fork it).

What it checks:

  1. THE ROUND TRIP the packet asks for: a plan describing the v3.1
     fight-scene window compiles to the same execution the hand-built
     t2c_w45e_v001.api.json encodes — window 68..123, hold map
     [1]*5+[2]*51, 107 dilated frames, 17 -> 32 tokens, tail guide at
     dilated 105 — and the emitted graph's links all resolve.
  2. DITHERING: a drawn 2.5x is met in aggregate by mixed integer holds,
     and the achieved average tracks the drawn one across ratios.
  3. CEILINGS: an envelope above the lane ceiling is refused by the
     schema and clamped by the compiler (with a warning), never silently
     obeyed.
  4. LEGALITY SWEEP: over many envelopes/windows, every compile lands on
     the 17k+5 grid, adds a multiple of 17, starts on the 17-phase, keeps
     the window inside the clip, and places the guide at the prefix sum.
  5. PRICE: monotone in work, layer (b) abstains without calibration and
     answers with it, the VRAM fit line says STUB.
  6. THE EPISTEMIC BOUNDARY (amendment 2): the semantic schema imports and
     validates a plan in a subprocess with NO h3 module and NO torch
     loaded; recipe values carry evidence; a deprecated value refuses to
     be read and keeps its reason; structural spec facts agree with the
     grid law they describe.
  7. IDENTITY: compiled sections record backend/spec/recipe/compiler
     versions and derive from (plan id, revision); a semantic edit bumps
     the revision and makes the compiled cache stale.

Exit code 0 = pass.
"""
import json
import os
import subprocess
import sys

# The pack's own __init__ imports comfy, so the suites load the pack's
# modules directly rather than through it (same trick as
# tests/test_expand_to_end.py). timeline/ is a plain package on disk, so
# adding the pack dir to sys.path is enough.
HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)
sys.path.insert(0, PACK)

from timeline import price, schema                        # noqa: E402
from timeline.h3 import compile as h3compile              # noqa: E402
from timeline.h3 import gridlaw as G                      # noqa: E402
from timeline.h3 import propose as h3propose              # noqa: E402
from timeline.h3 import recipe as h3recipe                # noqa: E402
from timeline.h3 import spec as h3spec                    # noqa: E402

W45E = ("/mnt/work/ai/apps/ComfyUI-ModelCatalog/workflows/"
        "t2c_w45e_v001.api.json")

FAILS = []


def check(name, ok, detail=""):
    print(("  PASS  " if ok else "  FAIL  ") + name + ("  " + detail if detail else ""))
    if not ok:
        FAILS.append(name)


def fight_scene_plan():
    """The v3.1 fight-scene window, said semantically: real time through
    frame 72, then 2x to the end of the 124-frame clip."""
    plan = schema.new_plan("/tmp/seedhunt_20260963_00001_.mp4", frames=124,
                           fps=24, width=1152, height=640,
                           proposed_by="test",
                           settings={"regen_strength": 0.45, "steps": 25,
                                     "seed": 20260817,
                                     "output_prefix": "video/t2c_timeline"})
    plan["lanes"].append(schema.temporal_lane(
        [[0, 1.0], [72, 1.0], [73, 2.0], [123, 2.0]], ceiling=4.0))
    return plan


def main():
    B = h3compile.H3Backend()

    # ---- 1. the round trip against the hand-built graph
    print("1. ROUND TRIP vs t2c_w45e_v001.api.json")
    hand = json.load(open(W45E))
    hand_holds = json.loads(hand["404"]["inputs"]["hold_map"])["holds"]
    hand_guide = hand["406"]["inputs"]["frame_idx"]
    hand_len = hand["524"]["inputs"]["length"]
    hand_w0 = hand["410"]["inputs"]["batch_index"]
    hand_wlen = hand["410"]["inputs"]["length"]

    plan = fight_scene_plan()
    check("the plan validates as pure semantics", schema.validate(plan) == [],
          str(schema.validate(plan)))
    res = B.compile(plan)
    c = res.compiled
    check("window matches the hand-built crop",
          (c["window"]["start"], c["window"]["len"]) == (hand_w0, hand_wlen),
          f"compiled {c['window']} vs hand {hand_w0}+{hand_wlen}")
    check("dilated length matches", c["dilated_frames"] == hand_len == 107,
          f"{c['dilated_frames']} vs {hand_len}")
    check("token count matches: 17 -> 32",
          (c["tokens"]["world"], c["tokens"]["dilated"]) == (17, 32),
          str(c["tokens"]))
    check("tail guide lands on the hand-placed dilated frame",
          c["guide_dilated_idx"] == hand_guide == 105,
          f"{c['guide_dilated_idx']} vs {hand_guide}")
    check("the compiled hold map IS the hand-built one",
          c["hold_map"]["holds"] == hand_holds,
          G.hold_runs_str(c["hold_map"]["holds"]))
    check("added frames are a multiple of 17",
          c["added_frames"] % 17 == 0 and c["added_frames"] == 51,
          str(c["added_frames"]))
    check("dilated length is on the 17k+5 grid", G.is_legal(c["dilated_frames"]))

    g = res.graph
    ids = set(g)
    missing = [k for k, v in g.items() for a in v["inputs"].values()
               if isinstance(a, list) and a[0] not in ids]
    check("every link in the minted graph resolves", not missing, str(missing[:4]))
    check("the graph carries the compiled map verbatim",
          json.loads(g["404"]["inputs"]["hold_map"])["holds"] == hand_holds)
    check("the node is told NOT to rewrite the map again",
          g["404"]["inputs"]["expand_to_end"] is False)
    check("class types match the hand-built recipe node for node",
          all(g[k]["class_type"] == hand[k]["class_type"]
              for k in ("403", "404", "406", "410", "522", "526", "530",
                        "531", "413", "541")))
    check("inject came from the plan, not from a constant",
          g["522"]["inputs"]["inject"] == 0.45
          and g["522"]["inputs"]["inject"]
          == schema.setting(plan, "regen_strength"))

    # ---- 2. dithering
    print("\n2. DITHERING (fractional ratios met in aggregate)")
    holds = h3compile.dither_holds([2.5] * 56, 141)
    check("a flat 2.5x over 56 frames spends exactly 141 dilated frames",
          sum(holds) == 141 and min(holds) >= 1, G.hold_runs_str(holds))
    check("...as MIXED integer rates, not one rate plus a lump",
          set(holds) == {2, 3} and abs(sum(holds) / 56 - 2.5) < 0.02,
          f"rates {sorted(set(holds))}, mean {sum(holds) / 56:.3f}")
    check("...alternating, so the retime is even across the span",
          max(abs(holds[i] - holds[i + 1]) for i in range(len(holds) - 1)) == 1)
    for r in (1.5, 2.0, 2.25, 3.0, 3.75):
        p = schema.new_plan("/tmp/c.mp4", frames=124, width=1152, height=640)
        p["lanes"].append(schema.temporal_lane([[0, r], [123, r]], ceiling=4.0))
        cc = B.compile(p).compiled
        got = cc["achieved_average_ratio"]
        check(f"drawn {r}x over the whole clip -> achieved {got:.3f}x "
              f"(grid-rounded)", abs(got - r) <= 0.15,
              f"{cc['dilated_frames']}f from {cc['window']['len']}f")

    # ---- 3. ceilings
    print("\n3. CEILINGS (per-lane, user-settable)")
    over = schema.new_plan("/tmp/c.mp4", frames=124, width=1152, height=640)
    over["lanes"].append(schema.temporal_lane([[0, 1.0], [60, 6.0], [123, 1.0]],
                                              ceiling=4.0))
    probs = schema.validate(over)
    check("the schema refuses a point above its lane ceiling",
          any("above the lane ceiling" in p for p in probs), str(probs[:1]))
    over["lanes"][0]["ceiling"] = 8.0
    check("raising the ceiling makes the same envelope legal",
          schema.validate(over) == [], str(schema.validate(over)))
    cc = B.compile(over).compiled
    check("an 8x ceiling really is spent (no hidden 4x wall)",
          max(cc["hold_map"]["holds"]) >= 5,
          f"peak hold x{max(cc['hold_map']['holds'])}, "
          f"{cc['dilated_frames']}f")

    # ---- 4. legality sweep
    print("\n4. LEGALITY SWEEP")
    bad = []
    n_sweep = 0
    for frames in (39, 56, 90, 124, 209):
        for lo in (0, 7, 20, 40):
            for span in (5, 17, 40):
                for r in (1.5, 2.0, 4.0):
                    hi = min(frames - 1, lo + span)
                    if lo >= frames - 1:
                        continue
                    n_sweep += 1
                    p = schema.new_plan("/tmp/c.mp4", frames=frames,
                                        width=1152, height=640)
                    p["lanes"].append(schema.temporal_lane(
                        [[0, 1.0], [lo, 1.0], [min(lo + 1, hi), r], [hi, r],
                         [min(hi + 1, frames - 1), 1.0],
                         [frames - 1, 1.0]], ceiling=4.0))
                    if schema.validate(p):
                        continue
                    try:
                        cc = B.compile(p).compiled
                    except Exception as e:
                        bad.append((frames, lo, span, r, f"{type(e).__name__}: {e}"))
                        continue
                    w0, wl = cc["window"]["start"], cc["window"]["len"]
                    holds = cc["hold_map"]["holds"]
                    why = []
                    if not G.is_legal(cc["dilated_frames"]):
                        why.append("dilated off grid")
                    if not G.is_legal(wl):
                        why.append("window length off grid")
                    if w0 % 17:
                        why.append("window start off the 17-phase")
                    if w0 + wl > frames:
                        why.append("window runs past the clip")
                    if len(holds) != wl:
                        why.append("hold map is not window-local")
                    if cc["added_frames"] % 17:
                        why.append("added frames not a multiple of 17")
                    if cc["guide_dilated_idx"] != sum(holds[:wl - 1]):
                        why.append("guide is not the prefix sum")
                    if cc["dilated_frames"] < 39:
                        why.append("under the 39-frame floor")
                    if why:
                        bad.append((frames, lo, span, r, ";".join(why)))
    check(f"all {n_sweep} swept plans compile legally", not bad, str(bad[:3]))

    # ---- 5. price
    print("\n5. PRICE METER (geometry x complexity x calibration)")
    model = price.ComplexityModel(
        h3recipe.PROFILE.get("cost_exponent"),
        h3recipe.PROFILE.get("speed_anchors"))
    geoms = [price.Geometry(u, 32, steps=11, frames_plan=107, frames_plain=124,
                            pixels=1152 * 640)
             for u in (32, 47, 62, 92)]
    check("cost is monotone in work units",
          price.monotonic_check(model, geoms))
    check("the exponent is the recipe's, not a literal in price.py",
          abs(model.exponent - 1.55) < 1e-9, str(model.exponent))
    e0 = price.estimate(geoms[2], model, price.Calibration([]))
    check("layer (a) always answers", e0.multiplier > 1.0,
          f"{e0.multiplier:.2f}x")
    check("layer (b) abstains without calibration and says why",
          e0.seconds is None and "not calibrated" in e0.lines[2],
          e0.lines[2][:90])
    calib = price.Calibration([
        {"work_units": 32, "s_per_step": 4.97, "streamed": False},
        {"work_units": 47, "s_per_step": 9.1, "streamed": False}])
    e1 = price.estimate(geoms[2], model, calib)
    check("layer (b) answers once the recorder has runs",
          e1.seconds and e1.seconds > 0, f"{e1.seconds / 60:.1f} min")
    check("layer (c) declares itself a stub",
          "STUB" in e1.lines[-1], e1.lines[-1][:80])
    e2 = price.estimate(geoms[3], model, price.Calibration([
        {"work_units": 32, "s_per_step": 4.97, "streamed": False},
        {"work_units": 80, "s_per_step": 90.0, "streamed": True}]))
    check("a streamed run brackets the cliff: bigger plans go red",
          e2.fit == "red", f"{e2.fit}: {e2.fit_why[:60]}")
    check("the compiled plan carries its own estimate",
          c["estimate"]["equivalent_clip_time_x"] > 0,
          f"{c['estimate']['equivalent_clip_time_x']:.2f}x a plain 124f render")

    # ---- 6. the epistemic boundary
    print("\n6. EPISTEMIC BOUNDARY (semantics vs model vs experiment)")
    code = (
        "import sys, json;"
        f"sys.path.insert(0, {PACK!r});"
        "from timeline import schema as s;"
        "p = s.new_plan('/tmp/c.mp4', frames=124, fps=24);"
        "p['lanes'].append(s.temporal_lane([[0,1.0],[60,2.5],[123,1.0]]));"
        "assert s.validate(p) == [], s.validate(p);"
        "assert abs(s.sample_envelope(p['lanes'][0]['envelope'], 30) - 1.75) < 1e-9;"
        "bad = [m for m in sys.modules if 'torch' in m or 'numpy' in m "
        "or '.h3' in m or m.endswith('h3')];"
        "print(json.dumps(bad))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True)
    leaked = json.loads(out.stdout or "null") if out.returncode == 0 else None
    check("the semantic schema validates a plan with NO h3 and NO torch "
          "loaded", out.returncode == 0 and leaked == [],
          (out.stderr or str(leaked))[-200:])

    stray = schema.validate(dict(plan, lanes=[dict(plan["lanes"][0],
                                                   compiled_hold_map=[1, 2])]))
    check("a backend artifact in a semantic lane is a validation error",
          any("backend artifact" in p for p in stray), str(stray[:1]))

    prof = h3recipe.PROFILE
    noev = [k for k, v in prof.values.items() if not v.evidence]
    check("every recipe value cites evidence", not noev, str(noev))
    check("confidence is one of the declared levels",
          all(v.confidence in h3recipe.CONFIDENCE for v in prof.values.values()))
    check("deprecation is first class: the value is kept with its reason",
          "cost_exponent_legacy" in prof.deprecated_keys()
          and prof.record("cost_exponent_legacy").superseded_by == "cost_exponent"
          and "motion.py" in prof.record("cost_exponent_legacy").deprecation_reason)
    try:
        prof.get("cost_exponent_legacy")
        ok = False
    except KeyError as e:
        ok = "deprecated" in str(e)
    check("...and reading a deprecated value refuses, naming the successor", ok)
    ch = prof.challenges("d_max_sweet_spot", 8, "test", "because")
    check("a challenge FLAGS and does not override",
          ch["profile_value"] == 4 and "flagged" in ch["resolution"])
    check("a proposal above the sweet spot raises that flag",
          h3propose.challenge_ceiling(6.0) is not None
          and h3propose.challenge_ceiling(3.0) is None)

    S = h3spec.SPEC
    check("structural spec agrees with the grid law it describes",
          all(S.is_legal_length(n) == G.is_legal(n)
              for n in range(0, 400))
          and all(S.token_count(n) == G.token_count_exact(n)
                  for n in (5, 22, 39, 56, 90, 107, 124, 209)))
    check("audio-exact lengths are the ones the chaining rule names",
          S.audio_exact_lengths == (39, 90, 141, 192))
    check("the recipe's tail guard is the one the shipped code enforces",
          prof.get("max_end_tail") == G.MAX_END_TAIL == 17)

    # ---- 7. identity and derivation
    print("\n7. IDENTITY / DERIVATION LINKS")
    v = c["versions"]
    check("the compiled section records what produced it",
          v["model_spec"] == S.version and v["recipe_profile"] == prof.version
          and v["compiler"] == h3compile.COMPILER_VERSION
          and c["backend"] == "h3", json.dumps(v))
    check("...and which plan revision it came from",
          c["derived_from"] == {"plan_id": plan["id"],
                                "revision": plan["revision"]},
          json.dumps(c["derived_from"]))
    check("a fresh compile of the same revision is not stale",
          not schema.compiled_is_stale(plan, "h3"))
    schema.note_edit(plan, "test", "artist raised the envelope")
    check("a semantic edit bumps the revision and staleness fires",
          plan["revision"] == 1 and schema.compiled_is_stale(plan, "h3"),
          f"rev {plan['revision']}, compiled from rev "
          f"{c['derived_from']['revision']}")
    g2 = B.compile(fight_scene_plan()).graph
    check("compilation is deterministic for the same plan content",
          json.dumps(g2, sort_keys=True) == json.dumps(res.graph, sort_keys=True))

    # ---- 8. the backend seam
    print("\n8. BACKEND SEAM")
    caps = B.capabilities()
    check("capabilities name the compiled lanes and the versions",
          caps["lanes_compiled"] == ("temporal",)
          and caps["compiler"] == h3compile.COMPILER_VERSION)
    unsup = fight_scene_plan()
    unsup["lanes"].append({"type": "pin", "enabled": True, "frame": 104,
                           "authority": 1.0})
    check("an uncompilable lane is refused BY NAME, not ignored",
          any("pin" in p and "not compiled" in p for p in B.validate(unsup)),
          str(B.validate(unsup)[:1]))

    print()
    if FAILS:
        print(f"FAILED: {len(FAILS)} check(s): " + ", ".join(FAILS))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
