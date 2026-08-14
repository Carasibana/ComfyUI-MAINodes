#!/usr/bin/env python3
"""Unit test for H3 Temporal Insert, the node that does the time-smear retime
in LATENT space: expand the video latent onto the dilated token grid,
interpolate the inserted token-times, freeze the originals with a mask.

  python tests/test_temporal_insert.py
      Synthetic, no GPU, no models. comfy.nested_tensor is stubbed.

What it checks:

  1. THE T2a MAP, row for row against the probe's worked example
     (ComfyUI-ModelCatalog benchmarks/scripts/tinterp/token_map.csv, the
     124f clip with hold 2 on world frames 85..118): 124 -> 158 frames,
     37 -> 47 tokens, and the exact interpolation weights - smear token 26
     is w=0.400 between base tokens 25 and 26. The map is PORTED from that
     harness, so this test is what stops it drifting.
  2. THE EXACTNESS INVARIANT. Every token-time whose target lands on a base
     token centre (all the copied originals, and inserted singletons aligned
     to base 17-multiples) is a BIT-COPY of that base token: maxabs 0.0, not
     "close". T2a measured corr 1.0 there and the node must not do worse.
  3. LEGAL-SNAP PARITY with H3TimeSmear on three hold maps, including one
     that needs a tail pad: same dilated length, same snapped hold map, so
     the pixel and latent routes can never disagree about the grid.
  4. MASK CORRECTNESS: inserted token-times 1, originals 0, nothing partial,
     full spatial extent, and the mask rides on the samples output (that is
     how it reaches the sampler) as well as on its own output.
  5. NOISE MODE fills only the inserted tokens, leaves the copies verbatim,
     and is reproducible from noise_seed.
  6. NESTED AV: the audio component comes through bit-exact and by
     reference; a plain video latent gets a zero audio track sized for the
     DILATED length.
  7. THE REPORT names the lengths, the token counts, the split, the
     17-multiple alignment rule and the audio caveat.

Exit code 0 = pass.
"""
import importlib.util
import os
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

# The T2a probe's retime: 124 world frames, hold 2 on frames 85..118 (34
# frames, starting on a 17-multiple). 158 = 17*9+5 is legal, no tail pad.
# NOTE: this map has the end-jump shape (a rate-1 tail behind a rate-2
# span), so every call below passes expand_to_end=False: the T2a probe's
# worked example is what this suite pins, and the toggle would rewrite it.
# The toggle itself is pinned by tests/test_expand_to_end.py.
T2A_HOLDS = [1] * 85 + [2] * 34 + [1] * 5
T2A_MAP = json_map = '{"holds": %s, "world_len": 124}' % T2A_HOLDS

# (token, target_base_time, lo, hi, w_hi) straight out of token_map.csv.
T2A_ROWS = [
    (1, 2.5, 0, 1, 1.0),      # identity region: lands on base token 1's centre
    (25, 85.0, 24, 25, 1.0),  # the hold's first frame, still an exact anchor
    (26, 86.0, 25, 26, 0.400),   # THE worked example
    (27, 88.0, 26, 27, 0.125),
    (28, 90.0, 26, 27, 0.625),
]
# tokens whose target lands exactly on a base token centre -> bit-copies
T2A_EXACT = list(range(26)) + [35, 45, 46]


def check(name, ok, detail=""):
    print(("  PASS  " if ok else "  FAIL  ") + name + ("  " + detail if detail else ""))
    if not ok:
        FAILS.append(name)


def base_latent(t=37, h=6, w=10, seed=7):
    torch.manual_seed(seed)
    return {"samples": torch.randn(1, 24, t, h, w)}


def main():
    node = motion.H3TemporalInsert()

    # ---- 1. the T2a map
    holds, dil, t_base, t_dil, plan = motion.temporal_insert_map(T2A_HOLDS)
    check("T2a retime: 124f -> 158f dilated", dil == 158, str(dil))
    check("T2a token grid: 37 -> 47 tokens",
          (t_base, t_dil) == (37, 47), f"{t_base} -> {t_dil}")
    check("no tail pad needed (158 = 17*9+5), hold map unchanged",
          holds == T2A_HOLDS and sum(holds) == 158, f"sum={sum(holds)}")
    for tok, t, lo, hi, w in T2A_ROWS:
        pt, plo, phi, pw, pex = plan[tok]
        if pex >= 0:                      # exact hits are recorded as a copy
            plo, phi, pw = pex - 1, pex, 1.0
        ok = (pt == t and (plo, phi) == (lo, hi) and abs(pw - w) < 1e-12)
        check(f"token {tok}: base time {t}, lerp {lo}/{hi} w={w}", ok,
              f"got t={pt} {plo}/{phi} w={pw!r}")
    check("smear token 26's weight is EXACTLY 0.400 (the csv's row)",
          plan[26][3] == 0.4, repr(plan[26][3]))
    check("the exact-hit set matches the probe's",
          [n for n, p in enumerate(plan) if p[4] >= 0] == T2A_EXACT,
          str([n for n, p in enumerate(plan) if p[4] >= 0]))
    check("18 of the 47 token-times are inserted, 29 copied",
          len([p for p in plan if p[4] < 0]) == 18 and len(T2A_EXACT) == 29,
          f"{len([p for p in plan if p[4] < 0])} inserted")

    # ---- 2 + 4. run the node: bit-copies and mask
    lat = base_latent()
    v0 = lat["samples"]
    out, mask_out, used, rep = node.insert(samples=lat, hold_map=T2A_MAP,
                                           init_mode="lerp",
                                           expand_to_end=False)
    ov, oa = out["samples"].unbind()
    mv, ma = out["noise_mask"].unbind()
    check("the video latent is expanded along t only",
          tuple(ov.shape) == (1, 24, 47, 6, 10), str(tuple(ov.shape)))
    worst = max(float((ov[:, :, n] - v0[:, :, plan[n][4]]).abs().max())
                for n in T2A_EXACT)
    check("EXACTNESS: every exact-hit token is a BIT-COPY of its base token",
          worst == 0.0, f"maxabs over {len(T2A_EXACT)} tokens = {worst!r}")
    ins = [n for n, p in enumerate(plan) if p[4] < 0]
    n26 = (0.6 * v0[:, :, 25] + 0.4 * v0[:, :, 26])
    check("an inserted token is the plain lerp of its bracketing base tokens",
          float((ov[:, :, 26] - n26).abs().max()) < 1e-6,
          f"maxabs {float((ov[:, :, 26] - n26).abs().max()):.3e}")
    check("mask shape is (1,1,t_dil,h,w)", tuple(mv.shape) == (1, 1, 47, 6, 10),
          str(tuple(mv.shape)))
    check("inserted token-times are marked 1 (regenerate)",
          all(float(mv[:, :, n].min()) == 1.0 for n in ins), f"{len(ins)} tokens")
    check("original token-times are marked 0 (frozen)",
          all(float(mv[:, :, n].max()) == 0.0 for n in T2A_EXACT))
    check("the mask is hard everywhere (no partial cells)",
          bool(((mv == 0) | (mv == 1)).all()))
    check("the audio half of the mask is all ones",
          bool((ma == 1).all()) and ma.shape[1:] == oa.shape[1:],
          str(tuple(ma.shape)))
    check("the mask rides on the samples output (this is how it reaches "
          "the sampler)", out["noise_mask"] is mask_out["samples"])
    check("the input latent is not mutated",
          tuple(lat["samples"].shape) == (1, 24, 37, 6, 10) and
          torch.equal(lat["samples"], v0))

    # ---- 3. legal-snap parity with H3TimeSmear
    smear = motion.H3TimeSmear()
    import json as _json
    cases = [("T2a map (no pad)", T2A_HOLDS),
             ("uniform x2 on 39f (pad)", [2] * 39),
             ("adaptive burst on 90f (pad)",
              [1] * 20 + [3] * 12 + [2] * 8 + [1] * 50)]
    for name, hm in cases:
        n = len(hm)
        img = torch.zeros(n, 4, 4, 3)
        _, used_smear, length_smear, _ = smear.smear(
            images=img, dilation=1, expand_to_end=False,
            hold_map=_json.dumps({"holds": hm, "world_len": n}))
        h2, dil2, tb2, td2, _p = motion.temporal_insert_map(hm)
        s = _json.loads(used_smear)
        check(f"snap parity, {name}",
              s["holds"] == h2 and int(length_smear) == dil2 and
              s["world_len"] == n,
              f"smear {length_smear}f last hold {s['holds'][-1]} | "
              f"insert {dil2}f last hold {h2[-1]} -> {td2} tokens")

    # ---- 5. noise mode
    lat2 = base_latent()
    nout, _nm, _u, nrep = node.insert(samples=lat2, hold_map=T2A_MAP,
                                      init_mode="noise", noise_seed=11,
                                      expand_to_end=False)
    nv = nout["samples"].unbind()[0]
    worst_n = max(float((nv[:, :, n] - v0[:, :, plan[n][4]]).abs().max())
                  for n in T2A_EXACT)
    check("noise mode still bit-copies the originals", worst_n == 0.0,
          f"maxabs {worst_n!r}")
    check("noise mode fills the inserted tokens with unit noise",
          abs(float(nv[:, :, ins].std()) - 1.0) < 0.05 and
          not torch.equal(nv[:, :, 26], ov[:, :, 26]),
          f"std {float(nv[:, :, ins].std()):.4f}")
    again, _, _, _ = node.insert(samples=base_latent(), hold_map=T2A_MAP,
                                 init_mode="noise", noise_seed=11,
                                 expand_to_end=False)
    check("noise mode is reproducible from noise_seed",
          torch.equal(again["samples"].unbind()[0], nv))
    diff, _, _, _ = node.insert(samples=base_latent(), hold_map=T2A_MAP,
                                init_mode="noise", noise_seed=12,
                                expand_to_end=False)
    check("...and a different seed gives a different init",
          not torch.equal(diff["samples"].unbind()[0], nv))

    # ---- 6. audio: nested passthrough vs plain input
    torch.manual_seed(5)
    audio_in = torch.randn(1, 32, 2, 207)          # 124f clip's audio latent
    audio_ref = audio_in.clone()
    Nested = sys.modules["comfy.nested_tensor"].NestedTensor
    av = {"samples": Nested([base_latent()["samples"], audio_in])}
    aout, _am, _au, arep = node.insert(samples=av, hold_map=T2A_MAP,
                                       expand_to_end=False)
    av_v, av_a = aout["samples"].unbind()
    check("nested AV: the audio comes out bit-exact",
          torch.equal(av_a, audio_ref) and tuple(av_a.shape) == (1, 32, 2, 207),
          str(tuple(av_a.shape)))
    check("nested AV: the audio is passed by reference, not copied",
          av_a is audio_in)
    check("nested AV: the video is still expanded", av_v.shape[2] == 47)
    check("nested AV: the audio mask matches the audio it guards",
          tuple(aout["noise_mask"].unbind()[1].shape) == (1, 32, 2, 207))
    check("a plain video latent gets a ZERO audio track on the DILATED clock",
          tuple(oa.shape) == (1, 32, 2, motion._audio_latent_t(158)) and
          bool((oa == 0).all()),
          f"{tuple(oa.shape)} vs base clock {motion._audio_latent_t(124)}")

    # ---- 7. hold_map passthrough + report
    u = _json.loads(used)
    check("hold_map passes through in H3TimeSmear's format, already snapped",
          u["holds"] == holds and u["world_len"] == 124,
          f"world_len {u['world_len']}, {len(u['holds'])} holds")
    check("the passthrough map drives H3TrueClock to the dilated token count",
          len(motion.true_clock_spans(u["holds"])) == 47,
          str(len(motion.true_clock_spans(u["holds"]))))
    for frag in ("world 124f -> dilated 158f", "t_lat 37 -> 47 tokens",
                 "copied verbatim (mask 0, frozen): 29 token-times [0-25,35,45-46]",
                 "inserted (mask 1, regenerate): 18 token-times [26-34,36-44]",
                 "T2a rule 1", "all on 17-multiples", "AUDIO CAVEAT",
                 "sigma >= ~0.5"):
        check(f"report says {frag!r}", frag in rep)
    check("the report flags an OFF-anchor hold span",
          "OFF the 17-multiple phase" in node.insert(
              samples=base_latent(), expand_to_end=False,
              hold_map=_json.dumps({"holds": [1] * 80 + [2] * 39 + [1] * 5,
                                    "world_len": 124}))[3])
    check("the nested-AV report states the audio is on the BASE clock",
          "audio passed through untouched" in arep and
          "BASE clip's clock" in arep)
    print("\n--- report (T2a map, lerp)\n" + rep)

    print()
    if FAILS:
        print(f"FAILED: {len(FAILS)} check(s): " + ", ".join(FAILS))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
