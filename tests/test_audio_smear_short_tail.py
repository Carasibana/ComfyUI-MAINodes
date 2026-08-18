# SPDX-License-Identifier: GPL-3.0-or-later
"""Repro of the 2026-08-18 user report: H3AudioSmear istft "window overlap add
min" when the LAST run of the hold map is stretched and the pass-1 audio is
shorter than frames/fps (H3's 40 Hz audio clock, up to +-12.5 ms).

Run from the ComfyUI root with the ComfyUI venv:
    python custom_nodes/ComfyUI-MAINodes/tests/test_audio_smear_short_tail.py
"""
import json
import os
import sys

sys.argv = sys.argv[:1]
sys.path.insert(0, os.path.abspath("."))
import torch  # noqa: E402

import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location("mainodes_motion", "custom_nodes/ComfyUI-MAINodes/motion.py")
motion = importlib.util.module_from_spec(spec)
spec.loader.exec_module(motion)

sr, fps = 32000, 24
holds = [1] * 9 + [2] * 4 + [3] * 4 + [4] * 22 + [3] * 4 + [2] * 4 + [1] * 21 + [2] * 1 + [3] * 4 + [4] * 12 + [3] * 1 + [2] * 9 + [3] * 12
frames = len(holds)                       # 248
tgt_frames = sum(holds)                   # 277 - wait: dilated frame count in the report
exact = int(round(frames / fps * sr))     # 330667 samples for 248 f
for short_by in (0, 544, 800):            # 0 = audio exactly frames/fps; 544 = the user's case (17 ms)
    n = exact - short_by
    wav = torch.randn(1, 2, n) * 0.1
    audio = {"waveform": wav, "sample_rate": sr}
    hold_map = json.dumps({"holds": holds})
    try:
        out = motion.H3AudioSmear().smear(audio, hold_map, fps=fps)[0]
        want = int(round(sum(holds) / fps * sr))
        got = out["waveform"].shape[-1]
        print(f"short_by={short_by:4d}: OK, out {got} samples (target {want}, diff {got - want})")
    except RuntimeError as e:
        print(f"short_by={short_by:4d}: RuntimeError: {str(e)[:120]}")
