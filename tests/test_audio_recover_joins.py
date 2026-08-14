#!/usr/bin/env python3
"""Regression test for H3 Audio Recover's segment joins.

  python tests/test_audio_recover_joins.py
      Synthetic unit test, no GPU, no files. Retimes a tonal source over a
      hold map that mixes 2/3/4 holds with passthrough spans and checks the
      three things the click fix bought:

        1. total length is still sample-exact against the world clock (the
           crossfade overlaps SOURCE material, it never shortens output),
        2. the run joins are no longer step discontinuities -- max |dx| over
           each join's fade region is within a small multiple of the max |dx|
           inside the segments themselves,
        3. h==1 passthrough runs are still bit-identical copies of the source
           outside the fade regions, i.e. nothing about the untouched-dialog
           promise changed.

      Before the fix, joins carried single-sample steps hundreds of times the
      local |dx| (istft came back hop-quantized and the shortfall was filled
      with digital silence, then butt-spliced).

Exit code 0 = pass.
"""
import json
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import importlib.util

spec = importlib.util.spec_from_file_location(
    "mainodes_motion", os.path.join(os.path.dirname(HERE), "motion.py"))
motion = importlib.util.module_from_spec(spec)
spec.loader.exec_module(motion)

SR, FPS = 32000, 24
XFADE = int(round(0.005 * SR))   # must match the node's fade length
JOIN_WIN = 512                   # one stft hop back from each join: wide
GUARD = 8                        # enough to cover the old zero-pad tail too,
                                 # which otherwise poisons the interior stat
JOIN_TOL = 1.5                   # join |dx| allowed vs interior max |dx|
                                 # (measured: 0.45-0.73x fixed, 3.3-3.7x pre)


def source(n):
    """Tonal + broadband, deterministic: smooth enough that a butt-splice
    stands out, dense enough that the vocoder has something to chew on."""
    t = torch.arange(n, dtype=torch.float32) / SR
    g = torch.Generator().manual_seed(63)
    x = (0.5 * torch.sin(2 * math.pi * 220.0 * t)
         + 0.25 * torch.sin(2 * math.pi * 613.0 * t + 0.7)
         + 0.1 * torch.sin(2 * math.pi * 3.0 * t) * torch.sin(2 * math.pi * 1750.0 * t)
         + 0.02 * torch.randn(n, generator=g))
    return torch.stack([x, x * 0.8])


def main():
    holds = ([1] * 10 + [2] * 8 + [1] * 6 + [3] * 9 + [1] * 4
             + [4] * 7 + [2] * 5 + [1] * 12)
    spf = SR / FPS
    n_src = int(round(sum(holds) * spf))
    x = source(n_src)
    wav = x[None]

    runs = []
    for h in holds:
        if runs and runs[-1][0] == h:
            runs[-1][1] += 1
        else:
            runs.append([h, 1])
    print(f"runs: {[(h, c) for h, c in runs]}  source {n_src} samples")

    node = motion.H3AudioRecover()
    out, = node.recover({"waveform": wav, "sample_rate": SR},
                        json.dumps({"holds": holds}), fps=FPS,
                        reference=None, reference_mix=0.0)
    y = out["waveform"].reshape(2, -1)

    # 1. sample-exact world clock
    expect = sum(int(round(c * spf)) for _, c in runs)
    assert y.shape[1] == expect, f"retimed length {y.shape[1]} != {expect}"
    assert torch.isfinite(y).all(), "non-finite samples in the retimed track"
    print(f"PASS length: retimed {y.shape[1]} samples == world clock exactly")

    # walk the runs to recover output offsets and source spans
    offs, spans, cursor, off = [], [], 0.0, 0
    for h, count in runs:
        tgt = int(round(count * spf))
        s0 = int(round(cursor))
        cursor += h * count * spf
        offs.append(off)
        spans.append((h, s0, int(round(cursor)), tgt))
        off += tgt

    # 2. joins are not outliers
    d = (y[:, 1:] - y[:, :-1]).abs().amax(dim=0)
    mask = torch.ones(d.shape[0], dtype=torch.bool)
    for o in offs[1:]:
        mask[max(0, o - JOIN_WIN):o + GUARD] = False
    interior = float(d[mask].max())
    worst, worst_at = 0.0, -1
    for i, o in enumerate(offs[1:], start=1):
        w = float(d[max(0, o - JOIN_WIN):o + GUARD].max())
        if w > worst:
            worst, worst_at = w, o
        print(f"   join @{o:7d} (h {spans[i - 1][0]} -> {spans[i][0]}): "
              f"max|dx| {w:.6g} = {w / interior:.2f}x interior")
    assert worst <= JOIN_TOL * interior, (
        f"join at {worst_at} steps {worst:.6g}, {worst / interior:.1f}x the "
        f"interior max |dx| {interior:.6g}: the seam is still a click")
    print(f"PASS joins: worst join {worst:.6g} = {worst / interior:.2f}x "
          f"interior max |dx| {interior:.6g} (tolerance {JOIN_TOL}x)")

    # 3. passthrough runs untouched outside the fades
    checked = 0
    for i, (o, (h, s0, s1, tgt)) in enumerate(zip(offs, spans)):
        if h != 1:
            continue
        # both ends carry a fade by design: this run's own pre-roll on the
        # left, the NEXT run's pre-roll blended into its tail on the right
        a = XFADE if o else 0
        z = tgt - XFADE if i + 1 < len(spans) else tgt
        got = y[:, o + a:o + z]
        want = x[:, s0 + a:s0 + a + got.shape[1]]
        assert torch.equal(got, want), (
            f"passthrough run at output {o} (source {s0}) is not a "
            f"bit-identical copy: max delta {(got - want).abs().max():.3g}")
        checked += got.shape[1]
    print(f"PASS passthrough: {checked} h==1 samples bit-identical to source "
          f"outside the {XFADE}-sample fades")


if __name__ == "__main__":
    main()
    print("ALL PASS")
