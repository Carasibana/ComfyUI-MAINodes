#!/usr/bin/env python3
"""Unit test for H3 Latent Bank, the node that keeps the pass-1 sampler out
of window items that only need its result.

  python tests/test_latent_bank.py
      Synthetic, no GPU, no comfy, no models. Writes into a temp dir.

What it checks:

  1. LAZY GATE. check_lazy_status asks for `samples` (which is what drags in
     the sampler, the guider, the noise and the pass-1 encode) only when the
     bank cannot serve. Miss -> ["samples"], hit -> [], refresh -> ["samples"]
     even on a hit.
  2. ROUNDTRIP on a plain LATENT, including the extra keys a latent dict can
     carry (noise_mask, batch_index).
  3. NESTED PACKING. An H3 AV latent's `samples` is a comfy NestedTensor, not
     a tensor. It is stored as its member tensors (video + audio) so a bank
     file does not depend on comfy's class layout; the rebuild itself needs
     comfy and is exercised in tests/test_latent_bank_lazy_graph.py.
  4. KEYING. seed and fingerprint are the ONLY things folded into the
     filename, which is the staleness contract stated in the node
     description: a new seed or a new fingerprint misses, everything else is
     the user's problem.
  5. SIZE. float16 storage roughly halves the file (the reason the node
     banks latents and not decoded frames is size, so size is a test).
  6. PASSTHROUGH and a never-fatal write, same as the conditioning bank.

Exit code 0 = pass.
"""
import importlib.util
import os
import shutil
import sys
import tempfile

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


def plain_latent():
    torch.manual_seed(0)
    return {"samples": torch.randn(1, 24, 32, 52, 30),
            "noise_mask": torch.ones(1, 1, 32, 52, 30),
            "batch_index": [0]}


def av_latent():
    torch.manual_seed(1)
    return {"samples": FakeNested([torch.randn(1, 24, 32, 52, 30),
                                   torch.randn(1, 32, 2, 178)])}


def same(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.shape == b.shape and \
            a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and set(a) == set(b) and \
            all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def main():
    node = motion.H3LatentBank()
    keep, refresh = motion.H3LatentBank.MODES
    root = tempfile.mkdtemp(prefix="h3latbank_")
    try:
        lat = plain_latent()
        kw = dict(bank_key="pass1", store_dir=root, seed=20260951,
                  fingerprint="12step_beta_lora1.0_0.4MP")

        want = node.check_lazy_status(mode=keep, samples=None, **kw)
        check("lazy gate asks for the sampler on a MISS", want == ["samples"],
              f"got {want}")

        out, rep = node.bank(samples=lat, mode=keep, **kw)
        path = motion.H3LatentBank._path(root, "pass1", kw["seed"],
                                         kw["fingerprint"])
        mb32 = os.path.getsize(path) / 1e6
        check("bank writes the file", os.path.exists(path),
              f"{path} ({mb32:.1f} MB)")
        check("write path returns the SAME object", out is lat)
        check("write report says MISS and names the shapes",
              rep.splitlines()[0].startswith("bank MISS") and
              "samples: (1, 24, 32, 52, 30)" in rep, rep.splitlines()[1])

        want = node.check_lazy_status(mode=keep, samples=None, **kw)
        check("lazy gate skips the sampler on a HIT", want == [], f"got {want}")
        want = node.check_lazy_status(mode=refresh, samples=None, **kw)
        check("refresh re-asks for the sampler", want == ["samples"],
              f"got {want}")

        got, rep = node.bank(samples=None, mode=keep, **kw)
        check("roundtrip is value-identical (all latent keys)", same(lat, got))
        check("read report says HIT and no sampler",
              rep.splitlines()[0].startswith("bank HIT") and
              "sampler upstream was not executed" in rep, rep.splitlines()[0])

        # 3. nested (AV) latent: packed structurally, video + audio kept
        av = av_latent()
        packed = motion._latent_pack(av)
        check("an AV latent packs as its member tensors",
              isinstance(packed["samples"], dict) and
              len(packed["samples"]["__nested__"]) == 2 and
              tuple(packed["samples"]["__nested__"][1].shape) == (1, 32, 2, 178),
              str([tuple(t.shape) for t in packed["samples"]["__nested__"]]))
        check("the packed AV latent is on the CPU",
              all(t.device.type == "cpu"
                  for t in packed["samples"]["__nested__"]))
        check("the AV report names both streams",
              "nested[2]" in "\n".join(motion._latent_shape_report(av)),
              motion._latent_shape_report(av)[0])

        # 4. keying
        for label, alt in (("a new seed", dict(kw, seed=kw["seed"] + 1)),
                           ("a new fingerprint",
                            dict(kw, fingerprint="6step_beta"))):
            want = node.check_lazy_status(mode=keep, samples=None, **alt)
            check(f"{label} MISSES the bank", want == ["samples"], f"got {want}")
        check("no seed and no fingerprint keys a bare bank_key file",
              motion.H3LatentBank._path(root, "pass1", None, None)
              == os.path.join(root, "pass1.latent.pt"))

        # 5. float16 storage halves the file
        node.bank(samples=lat, mode=refresh, store_dtype="float16 (half the file)",
                  **dict(kw, bank_key="pass1_half"))
        half = motion.H3LatentBank._path(root, "pass1_half", kw["seed"],
                                         kw["fingerprint"])
        mb16 = os.path.getsize(half) / 1e6
        check("float16 storage roughly halves the file",
              0.45 < mb16 / mb32 < 0.6, f"{mb32:.1f} MB -> {mb16:.1f} MB")

        # 6. a failed write is not a failed render
        blocked = os.path.join(root, "ro")
        os.makedirs(blocked)
        os.chmod(blocked, 0o500)
        out, rep = node.bank(samples=lat, mode=keep, bank_key="pass1",
                             store_dir=os.path.join(blocked, "sub"),
                             seed=kw["seed"], fingerprint=kw["fingerprint"])
        check("an unwritable store passes the latent through", out is lat)
        check("...and says so in the report", "NOT BANKED" in rep,
              rep.splitlines()[-1])
        os.chmod(blocked, 0o700)

        print(f"\n  size note: this 107-frame 480x832 AV latent banks at "
              f"{mb32:.1f} MB float32 / {mb16:.1f} MB float16; the same clip "
              f"decoded to float32 frames (107x832x480x3) would be "
              f"{107 * 832 * 480 * 3 * 4 / 1e6:.0f} MB")
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
