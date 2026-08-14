#!/usr/bin/env python3
"""Regression test for H3 Audio Recover's clock discipline.

Two modes, run occasionally (needs torch + torchaudio; ffmpeg for mode 2):

  python tests/test_audio_recover.py
      Synthetic unit test, no GPU, no files: retimed length is sample-exact
      against the world clock across mixed hold runs, reference_mix=1
      returns the reference bit-for-bit, and the unwired-reference warning
      fires.

  python tests/test_audio_recover.py BASELINE.mp4 RECOVERED.mp4
      File-pair check on a real render made with reference_mix=1: the
      recovered file's audio must BE the baseline's track. Verifies
      duration parity (within one AAC frame of codec padding), zero-lag
      alignment, and waveform correlation. Catches the two historical
      failures: cumulative retime truncation (recovered came back tens of
      ms short), and doubled/echoed audio (correlation collapses or the
      lag peak leaves zero).

Exit code 0 = pass.
"""
import json
import math
import os
import subprocess
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
AAC_FRAME = 1024  # samples; codec padding tolerance for file mode


def synthetic():
    holds = [1] * 20 + [4] * 30 + [1] * 20 + [2] * 17 + [1] * 20
    spf = SR / FPS
    n_src = int(round(sum(holds) * spf))
    wav = torch.zeros(1, 2, n_src)
    for f in range(sum(holds)):
        i = int(f * spf)
        wav[:, :, i:i + 40] = 1.0

    node = motion.H3AudioRecover()
    hm = json.dumps({"holds": holds})

    runs = []
    for h in holds:
        if runs and runs[-1][0] == h:
            runs[-1][1] += 1
        else:
            runs.append([h, 1])
    expect = sum(int(round(c * spf)) for _, c in runs)

    out, = node.recover({"waveform": wav, "sample_rate": SR}, hm, fps=FPS,
                        reference=None, reference_mix=0.0)
    got = out["waveform"].shape[-1]
    assert got == expect, f"retimed length {got} != world target {expect}"
    print(f"PASS length: retimed {got} samples == world clock exactly")

    ref = torch.randn(1, 2, int(round(len(holds) * spf)))
    out2, = node.recover({"waveform": wav, "sample_rate": SR}, hm, fps=FPS,
                         reference={"waveform": ref, "sample_rate": SR},
                         reference_mix=1.0)
    n_out = out2["waveform"].shape[-1]
    assert torch.equal(out2["waveform"], ref[:, :, :n_out]), \
        "mix=1.0 output is not the reference bit-for-bit"
    print(f"PASS identity: mix=1.0 == reference[:{n_out}] bit-for-bit")

    node.recover({"waveform": wav, "sample_rate": SR}, hm, fps=FPS,
                 reference=None, reference_mix=1.0)
    print("PASS warning path executed (check console line above)")


def pcm(path):
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-f", "s16le", "-ac", "1",
         "-ar", str(SR), "-"], capture_output=True).stdout
    import numpy as np
    return np.frombuffer(raw, dtype=np.int16).astype(np.float64)


def file_pair(baseline, recovered):
    import numpy as np
    a, b = pcm(baseline), pcm(recovered)
    dur_gap = abs(len(a) - len(b))
    assert dur_gap <= AAC_FRAME, (
        f"duration gap {dur_gap} samples ({dur_gap / SR * 1000:.1f} ms) "
        f"exceeds one AAC frame: the recovered track lost or gained time "
        f"(historical truncation bug looked like ~68 ms short per 5 s)")
    print(f"PASS duration: gap {dur_gap} samples "
          f"({dur_gap / SR * 1000:.1f} ms) within codec padding")

    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    # lag via cross-correlation on a centered 2 s window
    w = min(2 * SR, n // 2)
    mid = n // 2
    seg_a = a[mid - w // 2: mid + w // 2]
    span = SR // 10  # search +-100 ms
    best_lag, best_c = 0, -2.0
    for lag in range(-span, span + 1, 8):
        seg_b = b[mid - w // 2 + lag: mid + w // 2 + lag]
        if len(seg_b) != len(seg_a):
            continue
        denom = (np.std(seg_a) * np.std(seg_b)) or 1e-9
        c = float(np.mean((seg_a - seg_a.mean()) * (seg_b - seg_b.mean())) / denom)
        if c > best_c:
            best_lag, best_c = lag, c
    assert abs(best_lag) <= 16, (
        f"audio is time-shifted by {best_lag / SR * 1000:.1f} ms against the "
        f"baseline (doubling/echo territory)")
    assert best_c > 0.90, (
        f"waveform correlation {best_c:.3f} too low for a mix=1.0 render: "
        f"the recovered audio is not the baseline track")
    print(f"PASS identity: lag {best_lag} samples, correlation {best_c:.3f}")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        file_pair(sys.argv[1], sys.argv[2])
    else:
        synthetic()
    print("ALL PASS")
