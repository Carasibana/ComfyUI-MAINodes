#!/usr/bin/env python3
"""Unit test for expand_to_end, the toggle (default ON) on H3 Time Smear and
H3 Temporal Insert that stops an expansion span from dropping back to
real time before the clip ends.

  python tests/test_expand_to_end.py
      Synthetic, no GPU, no models. comfy.nested_tensor is stubbed.

What it checks:

  1. NO-FIRE cases are returned bit-identical: uniform dilation, a map that
     already ends inside an expansion, a rate-1-only map, and (the tail
     guard, operator ruling 2026-08-14) any rate-1 tail longer than 17
     world frames, which is rest rather than an end jump. The boundary is
     pinned at exactly 17 (fires) and 18 (does not), and the adaptive
     oracle shape kitsune_dash-style graphs produce, which ends in [1]*39,
     is checked to pass through untouched.
  2. FIRE cases against maps taken from real graphs:
     - the t2c_c / t2c_inj045 map [1]*17+[2]*34+[1]*5 (56 world, 90f)
       -> [1]*17+[2]*27+[3]*12 (107f)
     - the OR45-old shape [1]*34+[2]*17+[1]*5 -> the 0.45 round's minted
       [1]*34+[2]*10+[3]*12 (90f), the mixed-rate precedent
     - the two-burst t2c_w map (90 world, 141f) -> 158f
  3. INVARIANTS over a sweep of maps: world length preserved, output on the
     17k+5 grid, the last hold is > 1 (it now runs to the end), no hold is
     ever reduced, rates only rise toward the end inside the rewritten
     region, and _snap_holds is a no-op afterwards.
  4. DETERMINISM and idempotence: same map in, same map out, twice.
  5. THE UNDER-39 PAD interaction: _legal_ceil's floor of 39 is absorbed by
     the expansion instead of piling onto the final frame, so a 20-frame
     clip no longer ends on a 10x freeze.
  6. NODE WIRING: both nodes fire, log one line naming before and after,
     put the note in their report, and agree on the resulting grid; both
     reproduce today's behaviour exactly at expand_to_end=False; H3 Exact
     Recover still inverts the smear losslessly.
  7. THE PLAN COMPARISON the packet asks for: the rewritten t2c_c map vs
     the hand-built W45E map [1]*5+[2]*51.

Exit code 0 = pass.
"""
import importlib.util
import io
import json
import os
import sys
import types
from contextlib import redirect_stdout

import torch

HERE = os.path.dirname(os.path.abspath(__file__))

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

spec = importlib.util.spec_from_file_location(
    "mainodes_motion", os.path.join(os.path.dirname(HERE), "motion.py"))
motion = importlib.util.module_from_spec(spec)
spec.loader.exec_module(motion)

FAILS = []

# maps lifted from the minted graphs (ComfyUI-ModelCatalog workflows/)
T2C_C = [1] * 17 + [2] * 34 + [1] * 5          # t2c_c, t2c_inj045: 56w -> 90f
OR45_OLD = [1] * 34 + [2] * 17 + [1] * 5       # same window, expansion later
T2C_W = ([1] * 8 + [2] * 19 + [1] * 24 + [2] * 32 + [1] * 7)   # 90w -> 141f
W45E_HAND = [1] * 5 + [2] * 51                 # the hand-built 107f map

# what H3JerkOracle emits for a mid-clip burst on a 124f clip (balanced
# preset, ramp on, bridge 8; measured on a synthetic latent 2026-08-14).
# Its tail is 39 frames of REST, so the guard must leave it alone: firing
# would take it from 250 to 294 dilated frames.
ORACLE_REST = ([1] * 35 + [2] * 4 + [3] * 4 + [4] * 34 + [3] * 4 + [2] * 4
               + [1] * 39)


def check(name, ok, detail=""):
    print(("  PASS  " if ok else "  FAIL  ") + name + ("  " + detail if detail else ""))
    if not ok:
        FAILS.append(name)


def legal(n):
    return n >= 39 and (n - 5) % 17 == 0


def main():
    exp = motion.expand_hold_map_to_end

    # ---- 1. no-fire cases
    print("1. NO-FIRE (returned unchanged)")
    for name, hm in [
        ("uniform x4 on 124f (the kitsune_dash / default smear shape)", [4] * 124),
        ("uniform x2 on 39f", [2] * 39),
        ("rate-1 only (nothing to expand)", [1] * 56),
        ("already ends inside the expansion", [1] * 5 + [2] * 51),
        ("expansion already runs to the last frame, mixed rates",
         [1] * 17 + [2] * 27 + [3] * 12),
        ("TAIL GUARD: the adaptive oracle shape, [1]*39 of rest at the end",
         ORACLE_REST),
        ("TAIL GUARD boundary: an 18-frame rate-1 tail is rest",
         [1] * 20 + [3] * 18 + [1] * 18),
    ]:
        out, note = exp(hm)
        check(name, out == hm and note is None,
              f"note={note!r}" if note else "")
    check("the guard is one whole group of world frames",
          motion.MAX_END_TAIL == motion.LEGAL_STEP == 17,
          f"MAX_END_TAIL={motion.MAX_END_TAIL}")
    on17, note17 = exp([1] * 20 + [3] * 18 + [1] * 17)
    check("TAIL GUARD boundary: a 17-frame rate-1 tail still fires",
          note17 is not None and on17[-1] > 1 and legal(sum(on17)),
          f"{motion._hold_runs_str(on17)} = {sum(on17)}f")

    # ---- 2. fire cases, exact expected maps
    print("\n2. FIRE (the end-jump shape)")
    cases = [
        ("t2c_c / t2c_inj045 map, 56w 90f", T2C_C,
         [1] * 17 + [2] * 27 + [3] * 12, 107),
        ("OR45-old shape -> the 0.45 round's minted mixed-rate map", OR45_OLD,
         [1] * 34 + [2] * 10 + [3] * 12, 90),
        ("t2c_w two-burst map, 90w 141f", T2C_W,
         [1] * 8 + [2] * 19 + [1] * 24 + [2] * 29 + [3] * 10, 158),
        ("a ramped oracle tail keeps the boundary rate",
         [1] * 20 + [2, 3, 4, 4, 4, 3, 2] + [1] * 5,
         None, None),
    ]
    for name, hm, want, want_len in cases:
        out, note = exp(hm)
        if want is None:
            check(name, out[-1] > 1 and legal(sum(out)) and len(out) == len(hm),
                  f"{motion._hold_runs_str(out)} = {sum(out)}f")
            continue
        check(name, out == want and sum(out) == want_len,
              f"{motion._hold_runs_str(out)} = {sum(out)}f")
        check(f"  ...{want_len}f is on the 17k+5 grid", legal(want_len),
              f"(({want_len}-5) % 17 = {(want_len - 5) % 17})")
        check("  ...it says so on the console line",
              note and motion._hold_runs_str(hm) in note
              and motion._hold_runs_str(out) in note, str(note))

    # ---- 3. invariants over a sweep
    print("\n3. INVARIANTS over a sweep of end-jump maps")
    sweep = []
    for head in (0, 3, 17, 34):
        for rate in (2, 3, 4):
            for span in (5, 12, 34, 51):
                for tail in (1, 5, 12, 17):
                    sweep.append([1] * head + [rate] * span + [1] * tail)
    bad = []
    for hm in sweep:
        out, note = exp(hm)
        if note is None:
            bad.append(("did not fire", hm[:3], len(hm)))
            continue
        region = len(out) - 1
        while region > 0 and out[region - 1] >= 2 and out[region - 1] <= out[region]:
            region -= 1
        rising = all(out[i] <= out[i + 1] for i in range(region, len(out) - 1))
        if not (len(out) == len(hm) and legal(sum(out)) and out[-1] > 1
                and all(o >= h for o, h in zip(out, hm)) and rising
                and motion._snap_holds(list(out)) == out):
            bad.append((motion._hold_runs_str(hm), motion._hold_runs_str(out)))
    check(f"all {len(sweep)} swept maps: length kept, 17k+5, ends held, "
          "no hold reduced, rates rise at the tail, snap is a no-op",
          not bad, str(bad[:3]))
    guard = [[1] * head + [rate] * span + [1] * tail
             for head in (0, 3, 17, 34) for rate in (2, 3, 4)
             for span in (5, 12, 34, 51) for tail in (18, 25, 40)]
    check(f"all {len(guard)} maps with a rate-1 tail over 17 pass through "
          "bit-identical", all(exp(hm) == (hm, None) for hm in guard))

    # ---- 4. determinism / idempotence
    print("\n4. DETERMINISM")
    a, _ = exp(T2C_C)
    b, _ = exp(list(T2C_C))
    c, cnote = exp(a)
    check("same map in, same map out", a == b, motion._hold_runs_str(a))
    check("idempotent: a rewritten map is already to the end",
          c == a and cnote is None)
    check("the input list is not mutated", T2C_C[-1] == 1)

    # ---- 5. the under-39 pad interaction
    print("\n5. UNDER-39 PAD (_legal_ceil floors at 39)")
    short = [1] * 5 + [2] * 10 + [1] * 5           # 20 world frames, 30f
    out, _ = exp(short)
    padded = motion._snap_holds(list(short))
    check("expand_to_end absorbs the pad into the expansion",
          out == [1] * 5 + [2] * 11 + [3] * 4 and sum(out) == 39,
          f"{motion._hold_runs_str(out)} = {sum(out)}f")
    check("...instead of the 10x freeze the plain tail pad leaves",
          padded[-1] == 10 and max(out) == 3,
          f"plain pad last hold {padded[-1]}, expanded max rate {max(out)}")

    # ---- 6. the nodes
    print("\n6. NODE WIRING")
    smear = motion.H3TimeSmear()
    img = torch.arange(56 * 2 * 2 * 3, dtype=torch.float32).reshape(56, 2, 2, 3)
    hm_json = json.dumps({"holds": T2C_C, "world_len": 56})
    buf = io.StringIO()
    with redirect_stdout(buf):
        frames, used, length, rep = smear.smear(images=img, dilation=4,
                                                hold_map=hm_json)
    log = buf.getvalue().strip().splitlines()
    check("H3TimeSmear fires: 90f -> 107f", int(length) == 107 and
          frames.shape[0] == 107, f"length={length} batch={frames.shape[0]}")
    check("H3TimeSmear emits the rewritten map, already snapped",
          json.loads(used)["holds"] == a and json.loads(used)["world_len"] == 56)
    check("H3TimeSmear logs exactly one line naming before and after",
          len(log) == 1 and "expand_to_end" in log[0]
          and "[1]*17+[2]*34+[1]*5 (90f)" in log[0]
          and "[1]*17+[2]*27+[3]*12 (107f)" in log[0], log[0] if log else "no log")
    check("the note is in the report too", "expand_to_end" in rep)
    recovered = motion.H3ExactRecover().recover(images=frames, hold_map=used)[0]
    check("H3ExactRecover still inverts it losslessly",
          torch.equal(recovered, img), str(tuple(recovered.shape)))

    buf = io.StringIO()
    with redirect_stdout(buf):
        f0, used0, len0, rep0 = smear.smear(images=img, dilation=4,
                                            hold_map=hm_json,
                                            expand_to_end=False)
    check("expand_to_end=False reproduces today's behaviour exactly",
          int(len0) == 90 and json.loads(used0)["holds"] == T2C_C
          and "expand_to_end" not in rep0 and buf.getvalue() == "",
          f"length={len0}")

    node = motion.H3TemporalInsert()
    lat = {"samples": torch.randn(1, 24, motion._token_count(56), 4, 6)}
    buf = io.StringIO()
    with redirect_stdout(buf):
        out_l, _mask, used_i, rep_i = node.insert(samples=lat,
                                                  hold_map=hm_json)
    ilog = buf.getvalue().strip().splitlines()
    t_dil = out_l["samples"].unbind()[0].shape[2]
    check("H3TemporalInsert fires and expands onto the 107f token grid",
          t_dil == motion._token_count(107) == 32, f"t_dil={t_dil}")
    check("H3TemporalInsert emits the same map H3TimeSmear does",
          json.loads(used_i)["holds"] == json.loads(used)["holds"])
    check("H3TemporalInsert logs one line and reports the rewrite",
          len(ilog) == 1 and "expand_to_end" in ilog[0]
          and "expand_to_end" in rep_i, ilog[0] if ilog else "no log")
    buf = io.StringIO()
    with redirect_stdout(buf):
        out0, _m0, u0, r0 = node.insert(samples=lat, hold_map=hm_json,
                                        expand_to_end=False)
    check("...and expand_to_end=False is today's behaviour exactly",
          out0["samples"].unbind()[0].shape[2] == motion._token_count(90) == 27
          and json.loads(u0)["holds"] == T2C_C and buf.getvalue() == "",
          str(out0["samples"].unbind()[0].shape[2]))
    check("both toggles are exposed as BOOLEAN, default True",
          motion.H3TimeSmear.INPUT_TYPES()["optional"]["expand_to_end"]
          == ("BOOLEAN", motion.H3TimeSmear.INPUT_TYPES()["optional"]
              ["expand_to_end"][1]) and
          motion.H3TimeSmear.INPUT_TYPES()["optional"]["expand_to_end"][1]["default"] is True and
          motion.H3TemporalInsert.INPUT_TYPES()["optional"]["expand_to_end"][1]["default"] is True)

    # ---- 7. the plan comparison against the hand-built W45E map
    print("\n7. PLAN COMPARISON vs the hand-built W45E map")
    auto = a
    h_auto, d_auto, tb_a, td_a, plan_a = motion.temporal_insert_map(auto)
    h_hand, d_hand, tb_h, td_h, plan_h = motion.temporal_insert_map(W45E_HAND)
    check("same world length, same dilated length, same token grid",
          (len(h_auto), d_auto, td_a) == (len(h_hand), d_hand, td_h),
          f"auto {len(h_auto)}w {d_auto}f {td_a}tok | "
          f"hand {len(h_hand)}w {d_hand}f {td_h}tok")
    check("both run the expansion through the final world frame",
          h_auto[-1] > 1 and h_hand[-1] > 1,
          f"auto ends x{h_auto[-1]}, hand ends x{h_hand[-1]}")
    same = sum(1 for pa, ph in zip(plan_a, plan_h) if pa == ph)
    exact_a = sum(1 for p in plan_a if p[4] >= 0)
    exact_h = sum(1 for p in plan_h if p[4] >= 0)
    print(f"    auto {motion._hold_runs_str(auto)}  copied {exact_a}/{td_a}")
    print(f"    hand {motion._hold_runs_str(W45E_HAND)}  copied {exact_h}/{td_h}")
    print(f"    token-times with an identical plan row: {same}/{td_a}")
    check("the two maps are NOT the same plan (different rate distribution)",
          same < td_a, f"{same}/{td_a} rows identical")

    print()
    if FAILS:
        print(f"FAILED: {len(FAILS)} check(s): " + ", ".join(FAILS))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
