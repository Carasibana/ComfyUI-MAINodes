#!/usr/bin/env python3
"""Unit test for H3 V2V Init's time-varying freeze mask (torch only, no GPU,
no ComfyUI process).

  python tests/test_v2v_time_mask.py

Checks the pixel-frame -> latent-token map and the pooling that rides on it:
spans tile the clip exactly for legal lengths, every 5th token is a singleton
sitting on a 17-multiple frame, a mask that switches on AT a 17-multiple
lights that singleton and nothing earlier, a mixed token max-pools (any
regenerate frame wins), off-length masks resample nearest, and a T==1 mask
reproduces the static-union result.

Exit code 0 = pass.
"""
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

LEGAL = [39, 90, 141, 192]


def t_lat_of(length):
    return (length - 5) // 17 * 5 + 2


def test_spans():
    for length in LEGAL:
        t_lat = t_lat_of(length)
        spans = motion._token_frame_spans(t_lat)
        # inverts the node's own t_lat -> length formula
        assert (t_lat - 2) // 5 * 17 + 5 == length, length
        # tile [0, length) with no gap or overlap
        assert spans[0][0] == 0 and spans[-1][1] == length - 1, spans[-1]
        for (a, b), (c, _) in zip(spans, spans[1:]):
            assert b == c - 1, (a, b, c)
        sizes = [b - a + 1 for a, b in spans]
        assert sum(sizes) == length, (sum(sizes), length)
        # (1,4,4,4,4) rhythm; singletons on 17-multiples
        for t, (a, b) in enumerate(spans):
            if t % 5 == 0:
                assert (a, b) == (17 * (t // 5),) * 2, (t, a, b)
            else:
                assert b - a + 1 == 4, (t, a, b)
        print(f"PASS spans len={length}: t_lat={t_lat} sizes={sizes} "
              f"singleton frames={[a for t, (a, b) in enumerate(spans) if t % 5 == 0]}")


def test_switch_on_17():
    for length in (39, 141):
        t_lat = t_lat_of(length)
        for onset in range(17, length, 17):
            m = torch.zeros(length, 2, 2)
            m[onset:] = 1.0
            tok = motion._tokenize_mask_time(m, t_lat, length)
            hot = [t for t in range(t_lat) if tok[t].max() > 0]
            first = onset // 17 * 5
            assert hot == list(range(first, t_lat)), (length, onset, hot)
            assert first % 5 == 0
            print(f"PASS onset len={length} frame={onset}: first hot token "
                  f"{first} (singleton), tokens 0..{first - 1} cold")


def test_mixed_token_maxpool():
    length, t_lat = 39, t_lat_of(39)
    spans = motion._token_frame_spans(t_lat)
    a, b = spans[3]              # a 4-frame token
    assert (a, b) == (9, 12), spans[3]
    for f in range(a, b + 1):
        m = torch.zeros(length, 1, 1)
        m[f] = 1.0
        tok = motion._tokenize_mask_time(m, t_lat, length)
        hot = [t for t in range(t_lat) if tok[t].max() > 0]
        assert hot == [3], (f, hot)
    # partial spatial coverage: the union of the covered frames' pixels
    m = torch.zeros(length, 1, 4)
    m[9, 0, 0] = 1.0
    m[12, 0, 3] = 1.0
    tok = motion._tokenize_mask_time(m, t_lat, length)
    assert tok[3].flatten().tolist() == [1.0, 0.0, 0.0, 1.0], tok[3]
    print("PASS mixed token: frames 9..12 all light token 3 only; "
          "pixel-wise max within the token = [1,0,0,1]")


def test_resample():
    length, t_lat = 90, t_lat_of(90)
    # half-rate mask: frame f of the clip reads source frame floor(f/2)
    src = torch.zeros(45, 1, 1)
    src[9:] = 1.0                       # -> clip frames 18.. are hot
    tok = motion._tokenize_mask_time(src, t_lat, length)
    hot = [t for t in range(t_lat) if tok[t].max() > 0]
    # frame 18 sits in token 5*1+1 = 6 (span 18..21)
    assert hot == list(range(6, t_lat)), hot
    ref = torch.zeros(length, 1, 1)
    ref[18:] = 1.0
    assert torch.equal(tok, motion._tokenize_mask_time(ref, t_lat, length))
    print(f"PASS resample: T=45 -> {length} nearest, onset frame 18 -> first "
          f"hot token 6, identical to the already-length mask")


def test_t1_equals_static():
    for length in LEGAL:
        t_lat = t_lat_of(length)
        m = (torch.rand(1, 6, 7) > 0.5).float()
        tok = motion._tokenize_mask_time(m, t_lat, length)
        static = m.max(dim=0, keepdim=True).values.expand(t_lat, 6, 7)
        assert torch.equal(tok, static), length
    print("PASS T==1: pooled result equals the static-union expand for "
          f"lengths {LEGAL}")


if __name__ == "__main__":
    test_spans()
    test_switch_on_17()
    test_mixed_token_maxpool()
    test_resample()
    test_t1_equals_static()
    print("ALL PASS")
