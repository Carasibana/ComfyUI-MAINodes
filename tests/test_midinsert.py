#!/usr/bin/env python3
"""Unit test for H3 Mid Insert, the node that changes the temporal token
topology MID-denoise: the coarse grid stops at a handoff sigma_s, the new
token-times are lerped into the STILL-NOISY latent (with a measured variance
top-up), and the dilated grid finishes the schedule.

  python tests/test_midinsert.py
      Synthetic, no GPU, no models. comfy.nested_tensor is stubbed.

What it checks:

  1. GRID PARITY WITH H3TemporalInsert. Same hold map, same latent,
     noise_topup 0 -> the expanded video tensor is BIT-IDENTICAL to the
     sibling node's (maxabs 0.0), same snapped hold map, same copied /
     inserted split. This is what makes "shared arithmetic" a fact rather
     than an intention; it is checked with expand_to_end off AND on.
  2. THE VARIANCE MATHS, on synthetic noise with a KNOWN correlation. Base
     tokens are a unit-variance AR(1) chain along t, so every bracketing
     pair has lag-1 correlation exactly rho:
       - the recovered rho (report, mean over tokens and channels) is within
         5% of the truth at rho = 0.0 / 0.5 / 0.9,
       - noise_topup 0 leaves the lerp at Var = (1-w)^2 + w^2 + 2w(1-w)rho,
       - noise_topup 1 restores the neighbours' variance (v_tgt = 1),
       - noise_topup 0.5 lands between them,
       - the node's own identity self-check prints ~0.
  3. PASSTHROUGH: every copied token-time is a bit-copy of its base token
     even with a full top-up (the top-up touches inserted tokens ONLY), the
     input latent is not mutated, and the draw is reproducible from seed.
  4. AUDIO IS UNTOUCHED: the nested audio component comes out bit-exact and
     BY REFERENCE, and - unlike H3TemporalInsert - a plain video latent stays
     plain rather than gaining a fabricated zero audio track.
  5. NO NOISE MASK is emitted and an inbound one is dropped (repaint would
     re-noise mid-schedule rows from a clean latent).
  6. THE REPORT names the split, the measured rho, the deficit, the top-up,
     the DisableNoise requirement and the audio caveat; and with sigma_s
     given it prints the noise-only bound.

Exit code 0 = pass.
"""
import importlib.util
import json as _json
import os
import re
import sys
import types

import torch

HERE = os.path.dirname(os.path.abspath(__file__))

# The node builds real comfy NestedTensors; stub the module so the test needs
# nothing but torch. Same duck type comfy's has (unbind() / .is_nested).
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

# The T2a probe's retime, the same map tests/test_temporal_insert.py pins:
# 124 world frames, hold 2 on frames 85..118. 37 -> 47 tokens, 18 inserted.
T2A_HOLDS = [1] * 85 + [2] * 34 + [1] * 5
T2A_MAP = '{"holds": %s, "world_len": 124}' % T2A_HOLDS
# NOTE: this map has the end-jump shape (a rate-1 tail behind a rate-2 span),
# so the expand_to_end=True parity leg exercises the rewrite on both nodes.

H = W = 64          # 4096 spatial cells per channel: rho's sampling error
C, T = 24, 37       # is ~1/sqrt(4096) per channel, /sqrt(24) after the mean


def check(name, ok, detail=""):
    print(("  PASS  " if ok else "  FAIL  ") + name + ("  " + detail if detail else ""))
    if not ok:
        FAILS.append(name)


def plain_latent(seed=7, h=6, w=10):
    torch.manual_seed(seed)
    return {"samples": torch.randn(1, C, T, h, w)}


def ar1_latent(rho, seed=3):
    """Base tokens as a unit-variance AR(1) chain along t: every adjacent
    pair - and every bracketing pair the insert plan uses is adjacent - has
    correlation exactly rho, with no signal component to confound it."""
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(1, C, T, H, W, generator=g)
    x = torch.empty_like(z)
    x[:, :, 0] = z[:, :, 0]
    k = (1.0 - rho * rho) ** 0.5
    for t in range(1, T):
        x[:, :, t] = rho * x[:, :, t - 1] + k * z[:, :, t]
    return {"samples": x}


def rep_float(rep, pattern):
    m = re.search(pattern, rep)
    assert m, f"report has no {pattern!r}:\n{rep}"
    return float(m.group(1))


def main():
    mid = motion.H3MidInsert()
    sib = motion.H3TemporalInsert()

    # ---- 1. grid parity with H3TemporalInsert
    for label, e2e in (("expand_to_end off", False), ("expand_to_end on", True)):
        lat_a, lat_b = plain_latent(), plain_latent()
        sout, _nm, sused, srep = sib.insert(samples=lat_a, hold_map=T2A_MAP,
                                            init_mode="lerp", expand_to_end=e2e)
        mout, mused, mrep = mid.insert(samples=lat_b, hold_map=T2A_MAP,
                                       noise_topup=0.0, seed=1,
                                       expand_to_end=e2e)
        sv = sout["samples"].unbind()[0]
        mv = mout["samples"]                       # plain in, plain out
        worst = float((sv - mv).abs().max()) if sv.shape == mv.shape else -1.0
        check(f"grid parity ({label}): expanded video is BIT-IDENTICAL to "
              "H3TemporalInsert's", worst == 0.0,
              f"shapes {tuple(sv.shape)} vs {tuple(mv.shape)}, maxabs {worst!r}")
        check(f"grid parity ({label}): same snapped hold map",
              _json.loads(sused) == _json.loads(mused))
        runs = lambda r, key: re.search(key + r": \d+ token-times \[([^\]]*)\]", r).group(1)
        check(f"grid parity ({label}): same copied split",
              runs(srep, "copied verbatim \\(mask 0, frozen\\)")
              == runs(mrep, "copied verbatim, STILL NOISY and still denoising "
                            r"\(no freeze mask\)"),
              runs(mrep, "copied verbatim, STILL NOISY and still denoising "
                         r"\(no freeze mask\)"))
        check(f"grid parity ({label}): same inserted split",
              runs(srep, r"inserted \(mask 1, regenerate\)")
              == runs(mrep, r"inserted \(lerp of noisy neighbours\)"),
              runs(mrep, r"inserted \(lerp of noisy neighbours\)"))

    # ---- 2. the variance maths on known correlation
    # token 26 is the T2a worked example: w = 0.400 between base 25 and 26
    W_HI = 0.400
    for rho in (0.0, 0.5, 0.9):
        lat = ar1_latent(rho)
        x = lat["samples"]
        raw, _u, rrep = mid.insert(samples=lat, hold_map=T2A_MAP,
                                   noise_topup=0.0, seed=5, sigma_s=0.5335)
        full, _u2, frep = mid.insert(samples=ar1_latent(rho), hold_map=T2A_MAP,
                                     noise_topup=1.0, seed=5)
        half, _u3, _hrep = mid.insert(samples=ar1_latent(rho), hold_map=T2A_MAP,
                                      noise_topup=0.5, seed=5)
        got = rep_float(rrep, r"correlation rho .*?: mean (-?[\d.]+)")
        check(f"rho={rho}: recovered correlation from the neighbour residual",
              abs(got - rho) <= max(0.05 * rho, 0.02),
              f"got {got:.4f}, want {rho}")
        ident = rep_float(rrep, r"identity self-check .*?: ([\d.e+-]+)")
        check(f"rho={rho}: the derivation checks out on the tensor "
              "(|Var(lerp) - (v_tgt - deficit)|/v_tgt ~ 0)", ident < 1e-5,
              f"{ident:.3e}")

        v_raw = float(raw["samples"][:, :, 26].float().var(dim=(-2, -1)).mean())
        v_full = float(full["samples"][:, :, 26].float().var(dim=(-2, -1)).mean())
        v_half = float(half["samples"][:, :, 26].float().var(dim=(-2, -1)).mean())
        v_tgt = float(((1 - W_HI) * x[:, :, 25].var(dim=(-2, -1))
                       + W_HI * x[:, :, 26].var(dim=(-2, -1))).mean())
        want_raw = ((1 - W_HI) ** 2 + W_HI ** 2 + 2 * W_HI * (1 - W_HI) * rho)
        check(f"rho={rho}: noise_topup 0 is the variance-DEFICIENT raw lerp "
              f"(Var -> {want_raw:.4f} v)",
              abs(v_raw / v_tgt - want_raw) < 0.03,
              f"Var(lerp)/v_tgt {v_raw / v_tgt:.4f} want {want_raw:.4f}")
        check(f"rho={rho}: noise_topup 1 restores the neighbours' variance",
              abs(v_full / v_tgt - 1.0) < 0.03,
              f"Var(topup)/v_tgt {v_full / v_tgt:.4f}")
        check(f"rho={rho}: noise_topup 0.5 lands between the two",
              v_raw - 1e-9 <= v_half <= v_full + 1e-9 or rho >= 1.0,
              f"{v_raw:.4f} <= {v_half:.4f} <= {v_full:.4f}")
        # deficit vs the flow bound: with pure unit noise and no content
        # change the measured deficit must sit AT the noise-only bound
        meas = rep_float(rrep, r"deficit .*?: mean ([\d.]+)")
        bound = rep_float(rrep, r"explains a deficit of ([\d.]+)")
        ratio = rep_float(rrep, r"= ([\d.]+)x that")
        check(f"rho={rho}: the sigma_s bound is printed and the measured/bound "
              "ratio is self-consistent",
              abs(ratio - meas / bound) < 0.02, f"{meas:.5f} / {bound:.5f} "
              f"= {meas / bound:.3f}, report says {ratio:.2f}x")

    # ---- 3. passthrough, non-mutation, reproducibility
    lat = plain_latent()
    v0 = lat["samples"].clone()
    _holds, _dil, _tb, _td, plan = motion.temporal_insert_map(T2A_HOLDS)
    exact = [(n, p[4]) for n, p in enumerate(plan) if p[4] >= 0]
    out, _u, _r = mid.insert(samples=lat, hold_map=T2A_MAP, noise_topup=1.0,
                             seed=9)
    ov = out["samples"]
    worst = max(float((ov[:, :, n] - v0[:, :, b]).abs().max()) for n, b in exact)
    check("PASSTHROUGH: every copied token-time is a BIT-COPY even under a "
          "full top-up", worst == 0.0,
          f"maxabs over {len(exact)} tokens = {worst!r}")
    check("the input latent is not mutated", torch.equal(lat["samples"], v0))
    again, _u, _r = mid.insert(samples=plain_latent(), hold_map=T2A_MAP,
                               noise_topup=1.0, seed=9)
    check("the top-up draw is reproducible from seed",
          torch.equal(again["samples"], ov))
    diff, _u, _r = mid.insert(samples=plain_latent(), hold_map=T2A_MAP,
                              noise_topup=1.0, seed=10)
    check("...and a different seed gives a different init",
          not torch.equal(diff["samples"], ov))
    ins = [n for n, p in enumerate(plan) if p[4] < 0]
    check("the top-up moved the inserted tokens and nothing else",
          not torch.equal(diff["samples"][:, :, ins], ov[:, :, ins]) and
          torch.equal(diff["samples"][:, :, [n for n, _b in exact]],
                      ov[:, :, [n for n, _b in exact]]))

    # ---- 4. audio
    torch.manual_seed(5)
    audio_in = torch.randn(1, 32, 2, 207)          # the 124f clip's audio latent
    audio_ref = audio_in.clone()
    Nested = sys.modules["comfy.nested_tensor"].NestedTensor
    av = {"samples": Nested([plain_latent()["samples"], audio_in])}
    aout, _au, arep = mid.insert(samples=av, hold_map=T2A_MAP,
                                 noise_topup=1.0, seed=9)
    av_v, av_a = aout["samples"].unbind()
    check("AUDIO UNTOUCHED: bit-exact out", torch.equal(av_a, audio_ref) and
          tuple(av_a.shape) == (1, 32, 2, 207), str(tuple(av_a.shape)))
    check("AUDIO UNTOUCHED: passed by reference, not copied", av_a is audio_in)
    check("the video half is still expanded", av_v.shape[2] == 47)
    check("the report states the clock mismatch it leaves behind",
          "still the BASE clip's clock" in arep and
          f"207 ticks against {motion._audio_latent_t(158)}" in arep,
          [l for l in arep.splitlines() if l.startswith("AUDIO:")][0])
    _pv, _pu, prep = mid.insert(samples=plain_latent(), hold_map=T2A_MAP,
                                noise_topup=0.0, seed=9)
    check("a plain video latent stays PLAIN (no fabricated zero audio track, "
          "unlike H3TemporalInsert)",
          not getattr(_pv["samples"], "is_nested", False) and
          _pv["samples"].dim() == 5 and "stays VIDEO-ONLY" in prep,
          str(tuple(_pv["samples"].shape)))

    # ---- 5. no noise mask, and an inbound one is dropped
    check("no noise_mask is emitted", "noise_mask" not in out)
    masked = plain_latent()
    masked["noise_mask"] = torch.ones(1, 1, T, 6, 10)
    mout2, _u, mrep2 = mid.insert(samples=masked, hold_map=T2A_MAP,
                                  noise_topup=0.0, seed=1)
    check("an inbound noise_mask is DROPPED (it would re-noise mid-schedule "
          "rows from a clean latent)",
          "noise_mask" not in mout2 and "was DROPPED" in mrep2)

    # ---- 6. the report
    _rep = mid.insert(samples=ar1_latent(0.5), hold_map=T2A_MAP,
                      noise_topup=1.0, seed=2, sigma_s=0.5335)[2]
    for frag in ("mid-denoise insert: world 124f -> dilated 158f",
                 "t_lat 37 -> 47 tokens (+10)",
                 "copied verbatim, STILL NOISY and still denoising",
                 "inserted (lerp of noisy neighbours): 18 token-times",
                 "measured neighbour correlation rho",
                 "measured variance deficit w(1-w)Var(x_hi - x_lo)",
                 "top-up applied: noise_topup 1.00",
                 "identity self-check",
                 "noise-only bound at sigma_s 0.5335",
                 "NO NOISE MASK is emitted",
                 "PASS B MUST USE DisableNoise",
                 "T2a rule 1", "all on 17-multiples", "AUDIO:"):
        check(f"report says {frag!r}", frag in _rep)
    print("\n--- report (AR(1) rho 0.5, topup 1.0, sigma_s 0.5335)\n" + _rep)

    print()
    if FAILS:
        print(f"FAILED: {len(FAILS)} check(s): " + ", ".join(FAILS))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
