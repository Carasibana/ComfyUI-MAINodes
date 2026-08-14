#!/usr/bin/env python3
"""Unit test for H3 Conditioning Bank, the node that keeps the text encoder
out of per-window queue items.

  python tests/test_conditioning_bank.py
      Synthetic, no GPU, no comfy, no models. Writes into a temp dir.

What it checks, i.e. what the node has to get right for the TE spike to
actually go away:

  1. LAZY GATE. check_lazy_status asks for the `conditioning` input (which
     is what drags in the encode node and its 15 GB text encoder) ONLY when
     the bank cannot serve. Miss -> ["conditioning"], hit -> [], refresh ->
     ["conditioning"] even on a hit.
  2. ROUNDTRIP. A banked conditioning comes back equal value-for-value,
     including the nested H3 payload (minimax_keyframes, each with its own
     condition latent) and pooled_output.
  3. CPU. Everything banked is on the CPU, so a bank written under
     --gpu-only loads on a card that cannot hold it.
  4. PROMPT FINGERPRINT. Two different prompt strings under the same
     bank_key are two different files, so an edited prompt misses instead
     of being served a stale take.
  5. PASSTHROUGH. On the write path the node returns the SAME object it was
     given (the encode item must be bit-identical to today).
  6. A FAILED WRITE IS NOT A FAILED RENDER. An unwritable store_dir reports
     "NOT BANKED" and still returns the conditioning.

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


def make_cond():
    """The shape MiniMaxH3ImageToVideo actually emits: [tensor, dict] with a
    pooled output, the token tags and a keyframe carrying a cond latent."""
    torch.manual_seed(0)
    return [[torch.randn(1, 137, 5120),
             {"pooled_output": torch.randn(1, 5120),
              "minimax_token_tags": torch.zeros(1, 137, dtype=torch.long),
              "minimax_keyframes": [{"resolved_frame_index": 0,
                                     "latent": torch.randn(1, 24, 2, 48, 27)}]}]]


def same(a, b, path="cond"):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.shape == b.shape and \
            a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and set(a) == set(b) and \
            all(same(a[k], b[k], f"{path}.{k}") for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y, f"{path}[{i}]")
                                        for i, (x, y) in enumerate(zip(a, b)))
    return a == b


def all_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.device.type == "cpu"
    if isinstance(obj, dict):
        return all(all_cpu(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return all(all_cpu(v) for v in obj)
    return True


def main():
    node = motion.H3ConditioningBank()
    keep, refresh = motion.H3ConditioningBank.MODES
    root = tempfile.mkdtemp(prefix="h3condbank_")
    try:
        cond = make_cond()
        kw = dict(bank_key="w_run", store_dir=root, prompt="a knight, running")

        # 1. lazy gate, miss
        want = node.check_lazy_status(mode=keep, conditioning=None, **kw)
        check("lazy gate asks for the encode on a MISS", want == ["conditioning"],
              f"got {want}")

        # write path
        out, rep = node.bank(conditioning=cond, mode=keep, **kw)
        path = motion.H3ConditioningBank._path(root, "w_run", kw["prompt"])
        check("bank writes the file", os.path.exists(path),
              f"{path} ({os.path.getsize(path) / 1e6:.1f} MB)"
              if os.path.exists(path) else "missing")
        check("write path returns the SAME object", out is cond)
        check("write report says MISS", rep.splitlines()[0].startswith("bank MISS"),
              rep.splitlines()[0])

        # 1b. lazy gate, hit
        want = node.check_lazy_status(mode=keep, conditioning=None, **kw)
        check("lazy gate skips the encode on a HIT", want == [], f"got {want}")
        want = node.check_lazy_status(mode=refresh, conditioning=None, **kw)
        check("refresh re-asks for the encode", want == ["conditioning"],
              f"got {want}")

        # 2 + 3. read path
        got, rep = node.bank(conditioning=None, mode=keep, **kw)
        check("roundtrip is value-identical", same(cond, got))
        check("banked tensors are on the CPU", all_cpu(got))
        check("read report says HIT and no TE",
              rep.splitlines()[0].startswith("bank HIT") and
              "text encoder not loaded" in rep, rep.splitlines()[0])
        check("report names the payload",
              "minimax_keyframes[1]" in rep and "pooled_output(1, 5120)" in rep,
              rep.splitlines()[1] if len(rep.splitlines()) > 1 else rep)

        # 4. prompt fingerprint
        other = dict(kw, prompt="a knight, walking")
        want = node.check_lazy_status(mode=keep, conditioning=None, **other)
        check("an edited prompt MISSES the bank", want == ["conditioning"],
              f"got {want}")
        check("the two prompts key different files",
              motion.H3ConditioningBank._path(root, "w_run", other["prompt"])
              != path)
        # ... and an unwired prompt is its own (unfingerprinted) key
        check("unwired prompt keys a bare bank_key file",
              motion.H3ConditioningBank._path(root, "w_run", None)
              == os.path.join(root, "w_run.cond.pt"))

        # 6. a failed write is not a failed render
        blocked = os.path.join(root, "ro")
        os.makedirs(blocked)
        os.chmod(blocked, 0o500)
        out, rep = node.bank(conditioning=cond, mode=keep, bank_key="w_run",
                             store_dir=os.path.join(blocked, "sub"),
                             prompt=kw["prompt"])
        check("an unwritable store passes the conditioning through", out is cond)
        check("...and says so in the report", "NOT BANKED" in rep,
              rep.splitlines()[-1])
        os.chmod(blocked, 0o700)
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
