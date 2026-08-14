#!/usr/bin/env python3
"""Unit test for H3 Latent Upscale, the node that resizes the VIDEO half of
an H3 nested AV latent and leaves the audio alone.

  python tests/test_latent_upscale.py
      Synthetic, no GPU, no comfy, no models.

What it checks:

  1. SHAPES. The video's h/w become the requested size; batch, channels and
     the TIME axis are untouched (the token clock is not this node's
     business).
  2. AUDIO PASSTHROUGH is bit-exact and the nested container survives.
  3. BOTH MODES run and differ: nearest-exact replicates cell values
     verbatim (a 2x nearest upscale is a 2x2 block copy), bilinear does not.
  4. GRID SNAPPING. Targets land on the legal /32-pixel = /2-cell grid, from
     scale factors AND from explicit width/height, and an odd request is
     snapped rather than accepted.
  5. PLAIN LATENTS. A bare VAEEncode video latent (no audio) works, because
     that is where the node sits in the latent-resident graph.
  6. NO-OP. scale 1.0 changes nothing and returns the video by identity.
  7. NOISE MASK, if present, follows the video (nearest) so a freeze mask
     does not silently apply at the wrong resolution.

Exit code 0 = pass.
"""
import importlib.util
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "mainodes_motion", os.path.join(os.path.dirname(HERE), "motion.py"))
motion = importlib.util.module_from_spec(spec)
spec.loader.exec_module(motion)

FAILS = []


def check(name, ok, detail=""):
    print(("  PASS  " if ok else "  FAIL  ") + name + ("  " + detail if detail else ""))
    if not ok:
        FAILS.append(name)


class FakeNested:
    """Stands in for comfy.nested_tensor.NestedTensor: same duck type."""
    def __init__(self, tensors):
        self.tensors = list(tensors)
        self.is_nested = True

    def unbind(self):
        return self.tensors


def av_latent(h=30, w=54, t=32):
    torch.manual_seed(3)
    return {"samples": FakeNested([torch.randn(1, 24, t, h, w),
                                   torch.randn(1, 32, 2, 178)])}


def main():
    node = motion.H3LatentUpscale()

    # 1 + 2. nested AV latent, 2x
    lat = av_latent()
    audio_in = lat["samples"].unbind()[1]
    audio_ref = audio_in.clone()
    out, rep = node.upscale(samples=lat, scale=2.0, mode="bilinear")
    v, a = out["samples"].unbind()
    check("2x resizes only h/w of the video",
          tuple(v.shape) == (1, 24, 32, 60, 108), str(tuple(v.shape)))
    check("the time axis is untouched", v.shape[2] == 32, str(v.shape[2]))
    check("the container is still nested", getattr(out["samples"], "is_nested", False))
    check("audio comes out bit-exact", a.shape == audio_ref.shape and
          torch.equal(a, audio_ref), str(tuple(a.shape)))
    check("audio is passed by reference (not copied)", a is audio_in)
    check("the input latent's video is not mutated",
          tuple(lat["samples"].unbind()[0].shape) == (1, 24, 32, 30, 54))
    check("the report names both halves",
          "30x54 cells -> 60x108 cells" in rep.replace(" -> ", " -> ") or
          "video 30x54 -> 60x108 cells" in rep, rep.splitlines()[0])
    check("the report says the audio was untouched",
          "audio passed through untouched" in rep, rep.splitlines()[-1])

    # 3. both modes; nearest-exact = 2x2 block copy of the source cells
    src = av_latent()
    v0 = src["samples"].unbind()[0]
    near, _ = node.upscale(samples=av_latent(), scale=2.0, mode="nearest-exact")
    vn = near["samples"].unbind()[0]
    check("nearest-exact 2x is a verbatim 2x2 block copy",
          torch.equal(vn[:, :, :, ::2, ::2], v0) and
          torch.equal(vn[:, :, :, 1::2, 1::2], v0),
          f"max|diff| {float((vn[:, :, :, ::2, ::2] - v0).abs().max()):.3e}")
    bil, _ = node.upscale(samples=av_latent(), scale=2.0, mode="bilinear")
    vb = bil["samples"].unbind()[0]
    check("bilinear differs from nearest", not torch.equal(vb, vn),
          f"mean|diff| {float((vb - vn).abs().mean()):.4f}")
    check("bilinear stays in range of the source",
          float(vb.min()) >= float(v0.min()) - 1e-5 and
          float(vb.max()) <= float(v0.max()) + 1e-5,
          f"[{float(vb.min()):.3f}, {float(vb.max()):.3f}] vs "
          f"[{float(v0.min()):.3f}, {float(v0.max()):.3f}]")

    # 4. grid snapping
    check("cells snap to the /2 grid (=/32 px)",
          [motion._snap_cells(px) for px in (864, 1728, 1000, 900, 33, 8)]
          == [54, 108, 62, 56, 2, 2],
          str([motion._snap_cells(px) for px in (864, 1728, 1000, 900, 33, 8)]))
    odd, rep_odd = node.upscale(samples=av_latent(), scale=1.5, mode="bilinear")
    vo = odd["samples"].unbind()[0]
    check("a scale landing off-grid is snapped, not accepted",
          tuple(vo.shape[3:]) == (46, 82),  # 45->46, 81->82 cells
          f"{tuple(vo.shape[3:])} cells = {vo.shape[3] * 16}x{vo.shape[4] * 16} px")
    wh, _ = node.upscale(samples=av_latent(), scale=2.0, width=1728, height=960,
                         mode="bilinear")
    vw = wh["samples"].unbind()[0]
    check("explicit width/height override scale",
          tuple(vw.shape[3:]) == (60, 108), str(tuple(vw.shape[3:])))
    wh2, _ = node.upscale(samples=av_latent(), scale=2.0, width=1000,
                          mode="bilinear")
    vw2 = wh2["samples"].unbind()[0]
    check("a nonzero axis snaps and the zero axis still uses scale",
          tuple(vw2.shape[3:]) == (60, 62), str(tuple(vw2.shape[3:])))

    # 5. plain video latent (VAEEncode output, no audio)
    plain = {"samples": torch.randn(1, 24, 32, 30, 54)}
    pout, prep = node.upscale(samples=plain, scale=2.0, mode="bilinear")
    check("a plain video latent upscales and stays plain",
          isinstance(pout["samples"], torch.Tensor) and
          tuple(pout["samples"].shape) == (1, 24, 32, 60, 108),
          str(tuple(pout["samples"].shape)))
    check("...and the report says there is no audio",
          "no audio component" in prep, prep.splitlines()[-1])

    # 6. no-op
    same = av_latent()
    sout, srep = node.upscale(samples=same, scale=1.0, mode="bilinear")
    check("scale 1.0 is a no-op (video by identity)",
          sout["samples"] is same["samples"], srep.splitlines()[0])

    # 7. noise mask follows the video
    masked = av_latent()
    masked["noise_mask"] = FakeNested([torch.ones(1, 1, 32, 30, 54),
                                       torch.ones(1, 32, 2, 178)])
    mout, _ = node.upscale(samples=masked, scale=2.0, mode="bilinear")
    mv, ma = mout["noise_mask"].unbind()
    check("a nested noise_mask is resized with the video",
          tuple(mv.shape) == (1, 1, 32, 60, 108), str(tuple(mv.shape)))
    check("the audio mask is left alone", tuple(ma.shape) == (1, 32, 2, 178),
          str(tuple(ma.shape)))

    print()
    if FAILS:
        print(f"FAILED: {len(FAILS)} check(s): " + ", ".join(FAILS))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
