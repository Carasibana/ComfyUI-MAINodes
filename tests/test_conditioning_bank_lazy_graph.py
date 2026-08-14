#!/usr/bin/env python3
"""Executor-level proof for H3 Conditioning Bank: does the text encoder run?

  python tests/test_conditioning_bank_lazy_graph.py
      Needs a ComfyUI checkout (COMFYUI_DIR, default /mnt/work/ai/apps/ComfyUI)
      but NO models, NO GPU and NO server: it drives ComfyUI's real
      PromptExecutor over a three-node graph in-process, on the CPU. A stub
      node stands in for the encode node and records whether it executed;
      that stub IS the text encoder for the purposes of this test, since
      executing the encode node is exactly what loads the ~15 GB TE.

Four queue items, the shape the rolling-window flow actually has, run under
BOTH node caches: RAM_PRESSURE (ComfyUI's default) and CLASSIC
(--cache-classic, and what older ComfyUI did).

  item 1  cold cache               -> encode RUNS (unavoidable, once)
  item 2  only 'window' changed    -> encode SKIPPED. ComfyUI's own cache
                                      already serves the conditioning, so
                                      "every window re-encodes the prompt"
                                      is FALSE for an uninterrupted requeue.
  item 3  another workflow queued in between
                                   -> BOTH caches re-encode. CLASSIC because
                                      _clean_cache keeps only what the
                                      CURRENT prompt uses
                                      (comfy_execution/caching.py:175);
                                      RAM_PRESSURE because execute_async
                                      calls ram_release(ram_inactive_headroom)
                                      after every node (execution.py:800) and
                                      that headroom is min(128 GB, total_ram),
                                      which is above `available` on any real
                                      box, so every entry from an older
                                      generation goes. THIS is the spike:
                                      one interleaved graph and the next
                                      window item reloads the text encoder.
  item 4  fresh PromptExecutor, i.e. ComfyUI restarted
                                   -> ALWAYS re-encodes: an in-memory cache
                                      cannot survive a process.

With the bank wired in, items 2, 3 and 4 never execute the encode node at
all, under either cache.

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
import execution                                                # noqa: E402
import nodes                                                    # noqa: E402
import server                                                   # noqa: E402

spec = importlib.util.spec_from_file_location(
    "mainodes_motion", os.path.join(os.path.dirname(HERE), "motion.py"))
motion = importlib.util.module_from_spec(spec)
spec.loader.exec_module(motion)

FAILS = []
RAN = {"encode": 0}


def check(name, ok, detail=""):
    print(("  PASS  " if ok else "  FAIL  ") + name + ("  " + detail if detail else ""))
    if not ok:
        FAILS.append(name)


class StubEncode:
    """Stands in for MiniMax H3 Image to Video: executing it is what loads
    the text encoder, so we only have to count executions."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"text": ("STRING", {"default": ""})}}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "go"
    CATEGORY = "test"

    def go(self, text):
        RAN["encode"] += 1
        torch.manual_seed(len(text))
        return ([[torch.randn(1, 8, 16),
                  {"pooled_output": torch.randn(1, 16),
                   "minimax_keyframes": [{"resolved_frame_index": 0,
                                          "latent": torch.randn(1, 4, 2, 2)}]}]],)


class StubSink:
    """The rest of the window graph: carries the per-window widget, so the
    cache signature moves exactly the way H3 Window Plan's 'window' does."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"conditioning": ("CONDITIONING",),
                             "window": ("INT", {"default": 0})}}
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "go"
    CATEGORY = "test"

    def go(self, conditioning, window):
        StubSink.last = conditioning
        return {}


class StubOther:
    """Some other workflow the operator queues in between."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"n": ("INT", {"default": 0})}}
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "go"
    CATEGORY = "test"

    def go(self, n):
        return {}


def graph(store, use_bank, window, prompt_text="a knight, running"):
    g = {"1": {"class_type": "StubEncode", "inputs": {"text": prompt_text}}}
    src = ["1", 0]
    if use_bank:
        g["2"] = {"class_type": "H3ConditioningBank",
                  "inputs": {"conditioning": ["1", 0], "bank_key": "proof",
                             "store_dir": store,
                             "mode": motion.H3ConditioningBank.MODES[0],
                             "prompt": prompt_text}}
        src = ["2", 0]
    g["3"] = {"class_type": "StubSink",
              "inputs": {"conditioning": src, "window": window}}
    return g


async def run(ex, g, tag):
    before = RAN["encode"]
    await ex.execute_async(copy.deepcopy(g), tag, {}, ["3"])
    assert ex.success, f"{tag} failed"
    return RAN["encode"] - before


async def main():
    server.PromptServer(asyncio.get_event_loop())
    nodes.NODE_CLASS_MAPPINGS.update({
        "StubEncode": StubEncode, "StubSink": StubSink, "StubOther": StubOther,
        "H3ConditioningBank": motion.H3ConditioningBank})

    args = {"lru": 0, "ram": 10.0, "ram_inactive": 128.0}
    root = tempfile.mkdtemp(prefix="h3condproof_")
    other = {"9": {"class_type": "StubOther", "inputs": {"n": 1}}}
    try:
        for ctype in (execution.CacheType.RAM_PRESSURE,
                      execution.CacheType.CLASSIC):
            cname = ctype.name + (" (ComfyUI default)"
                                  if ctype == execution.CacheType.RAM_PRESSURE
                                  else " (--cache-classic)")
            for use_bank in (False, True):
                label = ("AFTER (bank)" if use_bank else "BEFORE (no bank)")
                tag = f"{cname} {label}"
                RAN["encode"] = 0
                store = os.path.join(root, f"{ctype.name}_{int(use_bank)}")
                new_ex = lambda: execution.PromptExecutor(
                    server.PromptServer.instance, cache_type=ctype,
                    cache_args=args)
                ex = new_ex()
                print(f"\n{tag}")

                n = await run(ex, graph(store, use_bank, 0), "i1")
                check(f"{tag}: item 1 (cold) encodes once", n == 1,
                      f"{n} encode(s)")

                n = await run(ex, graph(store, use_bank, 1), "i2")
                check(f"{tag}: item 2 (window 0->1) does not encode", n == 0,
                      f"{n} encode(s)")

                await ex.execute_async(copy.deepcopy(other), "interleaved",
                                       {}, ["9"])
                n = await run(ex, graph(store, use_bank, 2), "i3")
                # Both caches drop it: CLASSIC because _clean_cache keeps
                # only the current prompt's keys, RAM_PRESSURE because
                # execute_async calls ram_release(ram_inactive_headroom)
                # after every node and that headroom is min(128 GB,
                # total_ram) -- i.e. above `available` on any real box, so
                # every entry from an older generation is evicted.
                want3 = 0 if use_bank else 1
                check(f"{tag}: item 3 after an interleaved workflow -> "
                      f"{n} encode(s)", n == want3, f"expected {want3}")

                ex = new_ex()                      # ComfyUI restarted
                n = await run(ex, graph(store, use_bank, 3), "i4")
                want4 = 0 if use_bank else 1
                check(f"{tag}: item 4 after a RESTART -> {n} encode(s)",
                      n == want4, f"expected {want4}")

                print(f"        total encodes over 4 window items: "
                      f"{RAN['encode']}")
                if use_bank:
                    c = StubSink.last
                    check(f"{tag}: the sink got the banked conditioning",
                          isinstance(c, list) and
                          tuple(c[0][0].shape) == (1, 8, 16) and
                          len(c[0][1]["minimax_keyframes"]) == 1,
                          f"{tuple(c[0][0].shape)}")
                    check(f"{tag}: a bank file exists",
                          any(f.endswith(".cond.pt") for f in os.listdir(store)),
                          str(os.listdir(store)))
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
