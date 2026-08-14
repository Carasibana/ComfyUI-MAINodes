#!/usr/bin/env python3
"""Executor-level proof for H3 Latent Bank: does the pass-1 sampler run?

  python tests/test_latent_bank_lazy_graph.py
      Needs a ComfyUI checkout (COMFYUI_DIR, default /mnt/work/ai/apps/ComfyUI)
      but NO models, NO GPU and NO server: it drives ComfyUI's real
      PromptExecutor over a three-node graph in-process, on the CPU. A stub
      node stands in for SamplerCustomAdvanced and counts its executions.
      Its output is a real comfy NestedTensor AV latent, so the bank's
      nested pack/rebuild is exercised for real here (the synthetic test in
      tests/test_latent_bank.py cannot import comfy).

Same four items as tests/test_conditioning_bank_lazy_graph.py, because the
failure is the same one: the node cache serves an unchanged requeue, and
loses the entry to any interleaved workflow (CLASSIC drops what the current
prompt does not use, comfy_execution/caching.py:175; RAM_PRESSURE evicts
older generations through ram_release, execution.py:800) or to a restart.
Without the bank the baseline pass is re-sampled 3 times over 4 window
items; with it, once.

Exit code 0 = pass.
"""
import asyncio
import copy
import importlib.util
import os
import shutil
import sys
import tempfile

COMFY = os.environ.get("COMFYUI_DIR", "/mnt/work/ai/apps/ComfyUI")
HERE = os.path.dirname(os.path.abspath(__file__))

if not os.path.isdir(os.path.join(COMFY, "comfy_execution")):
    print(f"SKIP: no ComfyUI checkout at {COMFY} (set COMFYUI_DIR)")
    sys.exit(0)

sys.path.insert(0, COMFY)
os.chdir(COMFY)
sys.argv = [sys.argv[0], "--cpu", "--disable-auto-launch"]

import comfy.options                                            # noqa: E402
comfy.options.enable_args_parsing()

import torch                                                    # noqa: E402
import comfy.nested_tensor                                      # noqa: E402
import execution                                                # noqa: E402
import nodes                                                    # noqa: E402
import server                                                   # noqa: E402

spec = importlib.util.spec_from_file_location(
    "mainodes_motion", os.path.join(os.path.dirname(HERE), "motion.py"))
motion = importlib.util.module_from_spec(spec)
spec.loader.exec_module(motion)

FAILS = []
RAN = {"sample": 0}


def check(name, ok, detail=""):
    print(("  PASS  " if ok else "  FAIL  ") + name + ("  " + detail if detail else ""))
    if not ok:
        FAILS.append(name)


class StubSampler:
    """Stands in for the pass-1 SamplerCustomAdvanced: executing it is the
    expensive thing, so we only have to count executions. It emits a real H3
    shaped AV latent (video + audio NestedTensor), small enough for a test."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"seed": ("INT", {"default": 0})}}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "go"
    CATEGORY = "test"

    def go(self, seed):
        RAN["sample"] += 1
        torch.manual_seed(seed)
        video = torch.randn(1, 24, 4, 6, 4)
        audio = torch.randn(1, 32, 2, 9)
        return ({"samples": comfy.nested_tensor.NestedTensor((video, audio)),
                 "noise_mask": torch.ones(1, 1, 4, 6, 4)},)


class StubLatentSink:
    """The rest of the window graph, carrying the per-window widget."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"samples": ("LATENT",),
                             "window": ("INT", {"default": 0})}}
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "go"
    CATEGORY = "test"

    def go(self, samples, window):
        StubLatentSink.last = samples
        return {}


class StubOther:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"n": ("INT", {"default": 0})}}
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "go"
    CATEGORY = "test"

    def go(self, n):
        return {}


SEED = 20260951


def graph(store, use_bank, window):
    g = {"1": {"class_type": "StubSampler", "inputs": {"seed": SEED}}}
    src = ["1", 0]
    if use_bank:
        g["2"] = {"class_type": "H3LatentBank",
                  "inputs": {"samples": ["1", 0], "bank_key": "proof",
                             "store_dir": store,
                             "mode": motion.H3LatentBank.MODES[0],
                             "seed": SEED,
                             "fingerprint": "12step_beta_0.4MP"}}
        src = ["2", 0]
    g["3"] = {"class_type": "StubLatentSink",
              "inputs": {"samples": src, "window": window}}
    return g


async def run(ex, g, tag):
    before = RAN["sample"]
    await ex.execute_async(copy.deepcopy(g), tag, {}, ["3"])
    assert ex.success, f"{tag} failed"
    return RAN["sample"] - before


async def main():
    server.PromptServer(asyncio.get_event_loop())
    nodes.NODE_CLASS_MAPPINGS.update({
        "StubSampler": StubSampler, "StubLatentSink": StubLatentSink,
        "StubOther": StubOther, "H3LatentBank": motion.H3LatentBank})

    args = {"lru": 0, "ram": 10.0, "ram_inactive": 128.0}
    root = tempfile.mkdtemp(prefix="h3latproof_")
    other = {"9": {"class_type": "StubOther", "inputs": {"n": 1}}}
    truth = None
    try:
        for ctype in (execution.CacheType.RAM_PRESSURE,
                      execution.CacheType.CLASSIC):
            cname = ctype.name + (" (ComfyUI default)"
                                  if ctype == execution.CacheType.RAM_PRESSURE
                                  else " (--cache-classic)")
            for use_bank in (False, True):
                label = "AFTER (bank)" if use_bank else "BEFORE (no bank)"
                tag = f"{cname} {label}"
                RAN["sample"] = 0
                store = os.path.join(root, f"{ctype.name}_{int(use_bank)}")
                def new_ex():
                    return execution.PromptExecutor(
                        server.PromptServer.instance, cache_type=ctype,
                        cache_args=args)
                ex = new_ex()
                print(f"\n{tag}")

                n = await run(ex, graph(store, use_bank, 0), "i1")
                check(f"{tag}: item 1 (cold) samples once", n == 1,
                      f"{n} sample(s)")
                if truth is None:
                    truth = StubLatentSink.last

                n = await run(ex, graph(store, use_bank, 1), "i2")
                check(f"{tag}: item 2 (window 0->1) does not re-sample", n == 0,
                      f"{n} sample(s)")

                await ex.execute_async(copy.deepcopy(other), "interleaved",
                                       {}, ["9"])
                n = await run(ex, graph(store, use_bank, 2), "i3")
                want3 = 0 if use_bank else 1
                check(f"{tag}: item 3 after an interleaved workflow -> "
                      f"{n} sample(s)", n == want3, f"expected {want3}")

                ex = new_ex()                      # ComfyUI restarted
                n = await run(ex, graph(store, use_bank, 3), "i4")
                want4 = 0 if use_bank else 1
                check(f"{tag}: item 4 after a RESTART -> {n} sample(s)",
                      n == want4, f"expected {want4}")

                print(f"        total pass-1 renders over 4 window items: "
                      f"{RAN['sample']}")
                if use_bank:
                    got = StubLatentSink.last
                    s = got["samples"]
                    ok = (isinstance(s, comfy.nested_tensor.NestedTensor) and
                          len(s.unbind()) == 2 and
                          torch.equal(s.unbind()[0], truth["samples"].unbind()[0]) and
                          torch.equal(s.unbind()[1], truth["samples"].unbind()[1]) and
                          torch.equal(got["noise_mask"], truth["noise_mask"]))
                    check(f"{tag}: the banked AV latent rebuilds bit-identical",
                          ok, f"video {tuple(s.unbind()[0].shape)} + audio "
                              f"{tuple(s.unbind()[1].shape)}")
                    f = [x for x in os.listdir(store) if x.endswith(".latent.pt")]
                    check(f"{tag}: a bank file exists", bool(f),
                          f"{f} "
                          f"({os.path.getsize(os.path.join(store, f[0])) / 1e6:.2f} MB)"
                          if f else "none")
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print()
    if FAILS:
        print(f"FAILED: {len(FAILS)} check(s): " + ", ".join(FAILS))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
