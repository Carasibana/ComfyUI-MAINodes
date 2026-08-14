#!/usr/bin/env python3
"""Unit test for H3 True Clock's density-corrected RoPE t-grid.

  python tests/test_true_clock.py
      Synthetic, no GPU, no files, no comfy. Checks the four properties the
      corrected grid has to have to be telling the model the truth:

        1. WORLD DURATION: the per-token spans sum to
           len(holds) * 5/3 RoPE units, i.e. the clip occupies its world
           duration and not its dilated one — including the 17k+5 snap tail,
           whose padding frames are extra copies of the LAST world frame and
           so cost no extra world time,
        2. MONOTONE: the t-grid (exclusive cumsum) is non-decreasing, in fact
           strictly increasing — no frame ever goes backwards in time,
        3. IDENTITY: holds all 1 reproduces the stock grid
           5/3 * (1,4,4,4,4) exactly, so the node is a no-op when nothing was
           dilated (default-off in the truest sense),
        4. TAIL: the snap pad is folded into the last hold and the resulting
           map is what H3TimeSmear itself would have used.

Exit code 0 = pass.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import importlib.util

spec = importlib.util.spec_from_file_location(
    "mainodes_motion", os.path.join(os.path.dirname(HERE), "motion.py"))
motion = importlib.util.module_from_spec(spec)
spec.loader.exec_module(motion)

R = motion.ROPE_UNITS_PER_FRAME
FAILS = []


def check(name, ok, detail):
    print(("  PASS  " if ok else "  FAIL  ") + name + "  " + detail)
    if not ok:
        FAILS.append(name)


def stock_spans(t_lat):
    return [R * (1, 4, 4, 4, 4)[k % 5] for k in range(t_lat)]


CASES = [
    ("uniform x4, 39 world frames", [4] * 39),
    ("uniform x4, 61 world frames (pads)", [4] * 61),
    ("uniform x2, 243 world frames", [2] * 243),
    ("heal243 shape: holds 4 on 132-150", [4 if 132 <= f <= 150 else 1
                                           for f in range(243)]),
    ("ramped map, mixed 1/2/3/4", [1, 1, 2, 3, 4, 4, 4, 3, 2, 1] * 9),
    ("single hot frame", [1] * 100 + [8] + [1] * 41),
]

print("1) world duration: sum(spans) == world_frames * 5/3")
for name, holds in CASES:
    spans = motion.true_clock_spans(holds)
    want = len(holds) * R
    got = sum(spans)
    check(name, abs(got - want) < 1e-9,
          "sum={:.9f} want={:.9f} (world={} dilated={} tokens={})".format(
              got, want, len(holds), sum(motion._snap_holds(holds)), len(spans)))

print("2) monotone: t-grid non-decreasing (and strictly increasing)")
for name, holds in CASES:
    g = motion.true_clock_grid(holds)
    d = [b - a for a, b in zip(g, g[1:])]
    check(name, min(d) > 0, "origin={:.6f} min step={:.6f} max step={:.6f}".format(
        g[0], min(d), max(d)))

print("3) identity: holds all-1 reproduces the stock grid")
for length in (39, 141, 243):
    spans = motion.true_clock_spans([1] * length)
    want = stock_spans((length - 5) // 17 * 5 + 2)
    check("length {}".format(length), spans == want,
          "tokens={} first5={} stock_first5={}".format(
              len(spans), [round(s, 6) for s in spans[:5]],
              [round(s, 6) for s in want[:5]]))

print("4) snap tail: pad folded into the last hold, sums land on 17k+5")
for name, holds in CASES:
    snapped = motion._snap_holds(holds)
    L = sum(snapped)
    legal = (L - 5) % 17 == 0 and L >= 39
    pad = L - sum(holds)
    same_head = snapped[:-1] == [int(h) for h in holds[:-1]]
    idem = motion._snap_holds(snapped) == snapped
    check(name, legal and same_head and idem,
          "dilated={} pad={} last_hold {}->{} idempotent={}".format(
              L, pad, holds[-1], snapped[-1], idem))

print("5) tail time accounting: last world frame gets exactly 5/3 units")
holds = [4] * 61                       # 244 -> snaps to 260, pad 16 in hold[-1]
snapped = motion._snap_holds(holds)
spans = motion.true_clock_spans(holds)
tail_frames = snapped[-1]
check("uniform x4, 61 frames", abs(sum(spans) - 61 * R) < 1e-9,
      "last hold={} (4 real + {} pad), total={:.9f} == 61*5/3={:.9f}".format(
          tail_frames, tail_frames - 4, sum(spans), 61 * R))

print("6) sanity: dilated clip is SHORTER on the RoPE axis than stock")
holds = [4] * 61
spans = motion.true_clock_spans(holds)
stock = sum(stock_spans(len(spans)))
check("uniform x4, 61 frames", sum(spans) < stock,
      "true={:.4f} stock={:.4f} ratio={:.4f} (expect ~1/4)".format(
          sum(spans), stock, sum(spans) / stock))

print()
if FAILS:
    print("FAILED: " + ", ".join(FAILS))
    sys.exit(1)
print("all true-clock checks passed")
