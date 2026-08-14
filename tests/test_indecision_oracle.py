#!/usr/bin/env python3
"""Unit test for H3 Indecision Oracle, the experimental x0-jitter oracle.

  python tests/test_indecision_oracle.py
      Synthetic, no GPU, no comfy, no models. Writes fake X0 Tap dumps into
      a temp dir.

What it checks:

  1. PARITY with the desk study. The map math in motion.py is a port of
     benchmarks/scripts/indecision/x0_jitter.py in ComfyUI-ModelCatalog; if
     that file is reachable, every ported function is asserted bit-identical
     to the original on synthetic tensors (jitter_map, detrend_phase,
     normalize, degeneracy_check, token_count). This is the only thing
     stopping the two copies from drifting.
  2. DEGENERACY. A dump where 7 of 12 token rows are pinned (x0 identical
     between the two taps) is a masked run: the map is a picture of the
     noise mask. The report must SHOUT, not whisper.
  3. BLEND MODES on known inputs: max is elementwise max, weighted is the
     convex combination, both after per-source rank normalization.
  4. JERK PASSTHROUGH is exactly H3JerkOracle. Same latent, same knobs ->
     byte-identical hold_map, segments, window. If this ever fails, the A/B
     is comparing two compilers instead of two signals.
  5. CONTRACT SHAPE. The emitted hold_map is parsed by the real consumers:
     H3TimeSmear (adaptive mode), H3ExactRecover (round trip back to the
     world clock), and H3WindowPlan's parse.
  6. STEP FALLBACK. The shipped pr15375 graphs tap 0,1,12,24 only, so the
     default 6->12 pair is NOT on disk there. auto_fallback picks the
     closest available pair and says so; off, it raises with the list.

Exit code 0 = pass.
"""
import importlib.util
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "mainodes_motion", os.path.join(os.path.dirname(HERE), "motion.py"))
motion = importlib.util.module_from_spec(spec)
spec.loader.exec_module(motion)

# The validated original. Optional: it lives in a different repo, and cv2 may
# not be importable here. When it is reachable the parity block runs.
X0_JITTER = ("/mnt/work/ai/apps/ComfyUI-ModelCatalog/benchmarks/scripts/"
             "indecision/x0_jitter.py")
try:
    _s = importlib.util.spec_from_file_location("x0_jitter_ref", X0_JITTER)
    ref = importlib.util.module_from_spec(_s)
    _s.loader.exec_module(ref)
except Exception as e:                       # noqa: BLE001
    ref = None
    REF_WHY = f"{type(e).__name__}: {e}"

FAILS = []


def check(name, ok, detail=""):
    print(("  PASS  " if ok else "  FAIL  ") + name + ("  " + detail if detail else ""))
    if not ok:
        FAILS.append(name)


# ---------------------------------------------------------------- fixtures

FRAMES, RES = 39, 64                    # T = 12 tokens, 4x4 latent, 2x2 grid
T_LAT = (FRAMES - 5) // 17 * 5 + 2      # 12
HL = RES // 16                          # 4


def make_dump(root, name, steps, pinned_rows=()):
    """Write fake X0 Tap files: payload["video"] flat, video + 64 audio elems.
    `pinned_rows` are token rows held IDENTICAL across every step, which is
    what a mask-composited run looks like."""
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    g = torch.Generator().manual_seed(7)
    base = torch.randn(24, T_LAT, HL, HL, generator=g)
    for i, s in enumerate(steps):
        z = base + 0.3 * (i + 1) * torch.randn(24, T_LAT, HL, HL, generator=g)
        for t in pinned_rows:
            z[:, t] = base[:, t]
        flat = torch.cat([z.reshape(-1), torch.zeros(64)])
        torch.save({"step": s, "total_steps": 25, "sigma": 1.0 - s / 25.0,
                    "video": flat}, os.path.join(d, f"x0_step{s:03d}.pt"))
    return d


def latent():
    g = torch.Generator().manual_seed(11)
    return {"samples": torch.randn(1, 24, T_LAT, HL, HL, generator=g)}


def images(n=FRAMES, hw=8):
    return torch.rand(n, hw, hw, 3, generator=torch.Generator().manual_seed(3))


BASE = dict(length=FRAMES, width=RES, height=RES, step_a=6, step_b=12,
            q=0.75, d_max=4, ramp=True)


# ---------------------------------------------------------------- the tests

def main():
    node = motion.H3IndecisionOracle()
    root = tempfile.mkdtemp(prefix="h3indec_")
    try:
        # ---- 1. parity with x0_jitter.py -------------------------------
        print("\n1. PARITY with x0_jitter.py")
        if ref is None:
            check("x0_jitter.py importable", False, REF_WHY)
        else:
            g = torch.Generator().manual_seed(1)
            za = torch.randn(24, T_LAT, HL, HL, generator=g)
            zb = torch.randn(24, T_LAT, HL, HL, generator=g)
            mine, theirs = motion._jitter_map(za, zb), ref.jitter_map(za, zb)
            check("jitter_map is bit-identical",
                  mine.shape == theirs.shape and np.array_equal(mine, theirs),
                  f"{mine.shape} max|d|={np.abs(mine - theirs).max():.3e}")
            check("detrend_phase is bit-identical",
                  np.array_equal(motion._detrend_phase(mine),
                                 ref.detrend_phase(theirs)))
            for m in ("rank", "z"):
                check(f"normalize('{m}') is bit-identical",
                      np.array_equal(motion._map_normalize(mine, m),
                                     ref.normalize(theirs, m)))
            check("degeneracy_check is identical",
                  motion._degeneracy_check(mine) == ref.degeneracy_check(theirs),
                  str(motion._degeneracy_check(mine)))
            check("token_count is identical",
                  all(motion._x0_token_count(f) == ref.token_count(f)
                      for f in (5, 39, 73, 124, 192, 294)),
                  f"39 -> {motion._x0_token_count(39)}, "
                  f"124 -> {motion._x0_token_count(124)}")
            # pinned rows -> exact zeros, on both sides
            zc = zb.clone()
            zc[:, :7] = za[:, :7]
            dm, dt = motion._jitter_map(za, zc), ref.jitter_map(za, zc)
            check("pinned rows read EXACTLY zero in both copies",
                  motion._degeneracy_check(dm)["zero_token_rows"]
                  == ref.degeneracy_check(dt)["zero_token_rows"] == list(range(7)),
                  str(motion._degeneracy_check(dm)["zero_token_rows"]))

        # ---- 2. the clean run ------------------------------------------
        print("\n2. CLEAN RUN, indecision mode")
        clean = make_dump(root, "clean", [0, 1, 6, 12, 24])
        out = node.read(dump_dir=clean, mode="indecision", samples=latent(),
                        **BASE)
        hold_map, segs, w0, wlen, prof, report, comp, heat = out
        holds = json.loads(hold_map)["holds"]
        check("hold_map covers every world frame", len(holds) == FRAMES,
              f"{len(holds)} holds, world_len="
              f"{json.loads(hold_map)['world_len']}")
        check("profile has one entry per latent token",
              len(prof.split()) == T_LAT, f"{len(prof.split())} tokens")
        check("no degeneracy shout on a clean dump",
              "DEGENERATE" not in report, report.splitlines()[0])
        check("report flags the node as experimental",
              "EXPERIMENTAL" in report,
              next(l for l in report.splitlines() if "EXPERIMENTAL" in l)[:70])
        check("window is on the 17-frame grid and inside the clip",
              w0 % 17 == 0 and 0 <= w0 < FRAMES and w0 + wlen <= FRAMES + 17,
              f"window_start={w0} window_len={wlen}")
        check("comparison carries the A/B numbers",
              "Spearman rho" in comp and "top-decile IoU" in comp,
              comp.splitlines()[1].strip())
        check("comparison names the not-a-superset caveat",
              "superset" in comp, comp.splitlines()[-1].strip()[:70])
        check("heat previews both maps side by side (no frames wired)",
              heat.ndim == 4 and heat.shape[0] == 1 and
              heat.shape[2] == 2 * heat.shape[1] * 4,   # 2 panels, 8 cols/2 rows
              f"heat {tuple(heat.shape)}")
        hoverlay = node.read(dump_dir=clean, mode="indecision",
                            samples=latent(), images=images(), **BASE)[7]
        check("heat overlays the frames when images are wired",
              tuple(hoverlay.shape) == (FRAMES, 8, 16, 3),
              f"heat {tuple(hoverlay.shape)}")
        check("comparison is refused (politely) with no latent wired",
              "wire the LATENT" in node.read(dump_dir=clean,
                                             mode="indecision", **BASE)[6])

        # ---- 3. degeneracy ---------------------------------------------
        print("\n3. DEGENERACY (masked / pinned run)")
        pinned = make_dump(root, "pinned", [6, 12], pinned_rows=range(7))
        rep = node.read(dump_dir=pinned, mode="indecision", **BASE)[5]
        line = [l for l in rep.splitlines() if "DEGENERATE" in l]
        check("7 of 12 zero rows (58%) fires the loud warning", bool(line),
              line[0][:120] if line else rep)
        check("...and says it is a picture of the mask",
              bool(line) and "NOISE MASK" in line[0])
        half = make_dump(root, "onethird", [6, 12], pinned_rows=range(3))
        rep3 = node.read(dump_dir=half, mode="indecision", **BASE)[5]
        check("3 of 12 zero rows (25%) does NOT fire the loud one",
              "DEGENERATE" not in rep3 and "zero jitter" in rep3,
              [l for l in rep3.splitlines() if "zero jitter" in l][0][:80])

        # ---- 3b. the 0->1 warning --------------------------------------
        rep01 = node.read(dump_dir=clean, mode="indecision",
                          **dict(BASE, step_a=0, step_b=1))[5]
        check("the 0->1 pair carries its own degeneracy warning",
              "chunk-phase ramp" in rep01,
              [l for l in rep01.splitlines() if "chunk-phase" in l][0][:90])

        # ---- 4. blend modes on known inputs ----------------------------
        print("\n4. BLEND MODES")
        A = np.array([[[0.0, 1.0], [2.0, 3.0]]], dtype=np.float32)
        B = np.array([[[3.0, 2.0], [1.0, 0.0]]], dtype=np.float32)
        An, Bn = motion._map_normalize(A), motion._map_normalize(B)
        check("rank normalization maps 4 cells onto 0,1/3,2/3,1",
              np.allclose(np.sort(An.ravel()), [0, 1 / 3, 2 / 3, 1.0]),
              str(An.ravel().tolist()))
        mx = motion._blend_maps("blend max", An, Bn, 0.5)
        check("blend max is elementwise max of the ranks",
              np.allclose(mx.ravel(), [1.0, 2 / 3, 2 / 3, 1.0]),
              str(mx.ravel().tolist()))
        for w, want in ((0.0, Bn), (1.0, An), (0.5, 0.5 * An + 0.5 * Bn)):
            got = motion._blend_maps("blend weighted w", An, Bn, w)
            check(f"blend weighted w={w:g}", np.allclose(got, want),
                  str(np.round(got.ravel(), 3).tolist()))
        lat = latent()
        for m in ("blend max", "blend weighted w"):
            hm = node.read(dump_dir=clean, mode=m, samples=lat, **BASE)[0]
            check(f"'{m}' produces a legal hold map",
                  len(json.loads(hm)["holds"]) == FRAMES and
                  min(json.loads(hm)["holds"]) >= 1,
                  f"peak x{max(json.loads(hm)['holds'])}")
        try:
            node.read(dump_dir=clean, mode="blend max", **BASE)
            check("a blend without `samples` refuses", False, "no assert raised")
        except AssertionError as e:
            check("a blend without `samples` refuses", "needs the LATENT" in str(e),
                  str(e)[:80])

        # ---- 5. jerk passthrough == H3JerkOracle -----------------------
        print("\n5. JERK PASSTHROUGH is byte-identical to H3 Jerk Oracle")
        jerk = motion.H3JerkOracle()
        for q, d_max, bridge in ((0.75, 4, 8), (0.70, 4, 8), (0.85, 3, 0)):
            j = jerk.read(samples=lat, length=FRAMES, q=q, d_max=d_max,
                          ramp=True, preset="custom", bridge=bridge)
            i = node.read(dump_dir=clean, mode="jerk passthrough", samples=lat,
                          bridge=bridge,
                          **dict(BASE, q=q, d_max=d_max))
            check(f"q={q} d_max={d_max} bridge={bridge}: hold_map identical",
                  j[0] == i[0],
                  f"peak x{max(json.loads(i[0])['holds'])}, "
                  f"{sum(1 for h in json.loads(i[0])['holds'] if h > 1)} held")
            check(f"q={q} d_max={d_max} bridge={bridge}: segments+window identical",
                  (j[1], j[2], j[3]) == (i[1], i[2], i[3]),
                  f"segments={i[1]!r} window={i[2]}+{i[3]}")
        check("passthrough reads no dump and says so",
              "jerk passthrough" in node.read(dump_dir="/nonexistent",
                                              mode="jerk passthrough",
                                              samples=lat, **BASE)[6])

        # ---- 6. contract shape: the real consumers parse it ------------
        print("\n6. CONTRACT SHAPE (the consumers parse the emitted map)")
        hold_map = node.read(dump_dir=clean, mode="indecision", **BASE)[0]
        imgs = images()
        smeared, used, length, srep = motion.H3TimeSmear().smear(
            images=imgs, dilation=4, hold_map=hold_map)
        check("H3TimeSmear consumes it in ADAPTIVE mode",
              smeared.shape[0] == length and "adaptive" in srep,
              f"{FRAMES} -> {length} frames; {srep.split(';')[-1].strip()}")
        check("the smeared length is on the 17k+5 grid",
              (length - 5) % 17 == 0, f"length={length}")
        recovered = motion.H3ExactRecover().recover(images=smeared,
                                                    hold_map=used)[0]
        check("H3ExactRecover inverts it losslessly",
              recovered.shape == imgs.shape and torch.equal(recovered, imgs),
              f"{tuple(recovered.shape)}")
        wholds = [int(h) for h in json.loads(hold_map)["holds"]]   # H3WindowPlan
        check("H3WindowPlan's parse accepts it",
              len(wholds) == FRAMES and all(isinstance(h, int) for h in wholds))
        gated = motion.H3ManualHoldMap().build(
            length=FRAMES, fps=24, ranges="0-38", hold=4, ramp=True, bridge=8,
            oracle_hold_map=hold_map)
        check("H3ManualHoldMap gates on it",
              len(json.loads(gated[0])["holds"]) == FRAMES,
              gated[2].splitlines()[0][:80])

        # ---- 7. step fallback -------------------------------------------
        print("\n7. STEP FALLBACK (shipped graphs tap 0,1,12,24)")
        coarse = make_dump(root, "coarse", [0, 1, 12, 24])
        check("available steps are read off disk",
              motion._x0_available_steps(coarse) == [0, 1, 12, 24],
              str(motion._x0_available_steps(coarse)))
        a, b, note = motion.H3IndecisionOracle._resolve_pair(coarse, 6, 12, True)
        check("6->12 falls back to the closest available pair",
              (a, b) == (1, 12) and "step fallback" in note, f"{a}->{b}")
        check("...and the note lists what is on disk and what is validated",
              "[0, 1, 12, 24]" in note and "6->12" in note and "12->24" in note,
              note.strip()[:110])
        rep = node.read(dump_dir=coarse, mode="indecision", **BASE)[5]
        check("the fallback is shouted in the report", "step fallback" in rep,
              [l for l in rep.splitlines() if "fallback" in l][0][:90])
        try:
            motion.H3IndecisionOracle._resolve_pair(coarse, 6, 12, False)
            check("auto_fallback off raises", False, "no error raised")
        except ValueError as e:
            check("auto_fallback off raises with the available list",
                  "[0, 1, 12, 24]" in str(e), str(e)[:110])
        check("12->24 is taken as-is when it exists",
              motion.H3IndecisionOracle._resolve_pair(coarse, 12, 24, True)
              == (12, 24, ""))
        try:
            node.read(dump_dir=os.path.join(root, "nope"), mode="indecision",
                      **BASE)
            check("an empty dump_dir refuses", False, "no assert raised")
        except AssertionError as e:
            check("an empty dump_dir refuses", "X0 Tap" in str(e), str(e)[:90])

        # ---- 8. the honest headline -------------------------------------
        print("\n8. DEFAULTS UNCHANGED ELSEWHERE")
        check("H3JerkOracle still defaults to q=0.75 d_max=4 ramp=True",
              [motion.H3JerkOracle.INPUT_TYPES()["required"][k]["default"]
               if isinstance(motion.H3JerkOracle.INPUT_TYPES()["required"][k], dict)
               else motion.H3JerkOracle.INPUT_TYPES()["required"][k][1]["default"]
               for k in ("q", "d_max", "ramp")] == [0.75, 4, True])
        check("H3TimeSmear still defaults to uniform dilation 4",
              motion.H3TimeSmear.INPUT_TYPES()["required"]["dilation"][1]["default"] == 4)
        check("the new node is registered and marked experimental",
              motion.TIMESMEAR_CLASS_MAPPINGS["H3IndecisionOracle"]
              is motion.H3IndecisionOracle and
              "experimental" in
              motion.TIMESMEAR_DISPLAY_MAPPINGS["H3IndecisionOracle"],
              motion.TIMESMEAR_DISPLAY_MAPPINGS["H3IndecisionOracle"])
        check("its outputs mirror the jerk oracle's, then add two",
              motion.H3IndecisionOracle.RETURN_NAMES[:6]
              == motion.H3JerkOracle.RETURN_NAMES and
              motion.H3IndecisionOracle.RETURN_TYPES[:6]
              == motion.H3JerkOracle.RETURN_TYPES,
              str(motion.H3IndecisionOracle.RETURN_NAMES))

        print("\n  sample comparison report:")
        for line in comp.splitlines():
            print("    " + line)
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print()
    if FAILS:
        print(f"FAILED: {len(FAILS)} check(s): " + ", ".join(FAILS))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
