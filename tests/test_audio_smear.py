#!/usr/bin/env python3
"""Regression test for H3 Audio Smear, the inverse of H3 Audio Recover.

  python tests/test_audio_smear.py

Synthetic, no GPU, no files. Smear stretches a world-clock track onto the
dilated clock the smeared video init lives on; Recover compresses it back.
The pair has to close:

  1. smeared length is sample-exact against sum(holds) frames
  2. smear -> recover returns the source length sample-exact
  3. the round trip preserves TIMING (envelope correlation), which is the
     only property the seed has to carry - a phase vocoder does not
     preserve phase, so sample-wise correlation is expected to be poor and
     is not asserted
  4. an all-ones hold map is a no-op

Why the timing matters: the seeded init exists so pass 2 renders a slowed
performance instead of inventing one at natural rate. If the stretch
misplaces events, the lips follow the wrong clock and Exact Recover /
Audio Recover compress something that was never slow.

Exit code 0 = pass.
"""
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from motion import H3AudioRecover, H3AudioSmear  # noqa: E402

SR, FPS = 32000, 24
FAIL = []


def check(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg)
    if not ok:
        FAIL.append(msg)


def tone(n_frames):
    """Amplitude-modulated tone: timing is legible in the envelope."""
    t = torch.arange(int(n_frames / FPS * SR)) / SR
    x = torch.sin(2 * math.pi * 220 * t) * (0.5 + 0.5 * torch.sin(2 * math.pi * 5 * t))
    return {"waveform": x.float()[None, None, :], "sample_rate": SR}


def envelope(v, w=256):
    return torch.nn.functional.avg_pool1d(v.abs()[None, None], w, w)[0, 0]


def corr(a, b):
    n = min(a.shape[0], b.shape[0])
    a, b = a[:n], b[:n]
    return float(torch.dot(a - a.mean(), b - b.mean())
                 / (a.std() * b.std() * n + 1e-9))


def main():
    # a burst in the middle, calm either side - the shape the oracle emits
    holds = [1] * 30 + [4] * 30 + [1] * 30
    hm = json.dumps({"holds": holds})
    src = tone(len(holds))
    n_src = src["waveform"].shape[-1]

    sm = H3AudioSmear().smear(src, hm, FPS)[0]
    want = int(round(sum(holds) * SR / FPS))
    check(abs(sm["waveform"].shape[-1] - want) <= 1,
          f"smeared length sample-exact: {sm['waveform'].shape[-1]} vs {want}")

    rt = H3AudioRecover().recover(sm, hm, FPS)[0]
    check(abs(rt["waveform"].shape[-1] - n_src) <= 1,
          f"round trip returns the world clock: {rt['waveform'].shape[-1]} vs {n_src}")

    c = corr(envelope(src["waveform"].reshape(-1)), envelope(rt["waveform"].reshape(-1)))
    check(c > 0.9, f"round trip preserves timing: envelope corr {c:+.3f} (> 0.9)")

    # an unheld map must not touch the track's length
    flat = json.dumps({"holds": [1] * 60})
    f = H3AudioSmear().smear(tone(60), flat, FPS)[0]
    check(abs(f["waveform"].shape[-1] - int(round(60 * SR / FPS))) <= 1,
          "all-ones hold map is a no-op on length")

    print("\n" + ("ALL PASS" if not FAIL else f"{len(FAIL)} FAILED"))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
