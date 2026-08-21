#!/usr/bin/env python3
"""Build an NVFP4 MiniMax H3 checkpoint from the Comfy-Org bf16 source.

Every selected block linear (blocks.N.attn.qkv_proj / attn.out_proj / mlp.fc1 / mlp.fc2)
is quantised with comfy's own TensorCoreNVFP4Layout (comfy_kitchen quantize_nvfp4:
e2m1 packed 2/byte, fp8 e4m3 scale per 16, fp32 per-tensor scale) and written in the
per-layer `comfy_quant` style that the NVFP4 text encoder uses, so ComfyUI's
MixedPrecisionOps loads it and runs the quantise-input-then-fp4-matmul path on sm120.
Everything else (embeds, refiner, adaln, norms, heads) is copied verbatim from the bf16
source, as the shipped W4A8 "mixed" file does.

    python3 tools/build_nvfp4_checkpoint.py --source /path/to/minimax_h3_fl2va_pruned_bf16.safetensors \
        --out-dir /path/to/ComfyUI/models/diffusion_models/minimax_h3 --regime all
    regimes: all (qkv,out,fc1,fc2 x blocks 0-49) | nofc2 (qkv,out,fc1) | mid (all x blocks 2-47)
             | mid_nofc2 | fc1 (fc1 only)
    --input-scale amax.json   optional static activation scales (A6 idea 8; not used by default)

Output: minimax_h3_fl2va_pruned_nvfp4_<regime>.safetensors in --out-dir, plus a sidecar
        .census.json with per-layer weight rel-rms vs bf16. Runs in ~30 s on a GPU once the
        source file is in page cache. Needs ComfyUI + comfy-kitchen importable (run from the
        ComfyUI directory or set PYTHONPATH).
"""
import argparse
import json
import os
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.environ.get("COMFYUI_DIR", "."))
import comfy.quant_ops as qo  # noqa: E402

PROJ_KEYS = {"qkv": "attn.qkv_proj", "out": "attn.out_proj", "fc1": "mlp.fc1", "fc2": "mlp.fc2"}
REGIMES = {
    "all": ("qkv,out,fc1,fc2", 0, 49),
    "nofc2": ("qkv,out,fc1", 0, 49),
    "mid": ("qkv,out,fc1,fc2", 2, 47),
    "mid_nofc2": ("qkv,out,fc1", 2, 47),
    "fc1": ("fc1", 0, 49),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="the Comfy-Org bf16 file (minimax_h3_fl2va_pruned_bf16.safetensors)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--regime", default="all", choices=sorted(REGIMES))
    ap.add_argument("--input-scale", default=None, help="json {layer_key: amax} -> static input_scale = amax/(448*6)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    projs, lo, hi = REGIMES[a.regime]
    projs = projs.split(",")
    amax = json.load(open(a.input_scale)) if a.input_scale else {}
    global SRC
    SRC = a.source
    name = f"minimax_h3_fl2va_pruned_nvfp4_{a.regime}.safetensors"
    out = a.out or os.path.join(a.out_dir, name)
    dev = torch.device("cuda")
    tensors, census = {}, {}
    t0 = time.time()
    with safe_open(SRC, "pt") as s:
        keys = list(s.keys())
        for i, k in enumerate(keys):
            quant = None
            if k.startswith("blocks.") and k.endswith(".weight"):
                parts = k.split(".")
                b = int(parts[1])
                sub = ".".join(parts[2:-1])
                for p in projs:
                    if sub == PROJ_KEYS[p] and lo <= b <= hi:
                        quant = p
            t = s.get_tensor(k)
            if quant is None:
                tensors[k] = t.contiguous()
                continue
            w = t.to(dev, torch.bfloat16)
            qdata, params = qo.TensorCoreNVFP4Layout.quantize(w)
            base = k[: -len(".weight")]
            tensors[f"{base}.weight"] = qdata.contiguous().cpu()                       # uint8 [out, in/2]
            tensors[f"{base}.weight_scale"] = params.block_scale.contiguous().cpu()     # e4m3 [out, in/16]
            tensors[f"{base}.weight_scale_2"] = params.scale.reshape(()).float().cpu()   # fp32 scalar
            if base in amax:
                tensors[f"{base}.input_scale"] = torch.tensor(float(amax[base]) / (448.0 * 6.0), dtype=torch.float32)
            tensors[f"{base}.comfy_quant"] = torch.frombuffer(bytearray(json.dumps({"format": "nvfp4"}).encode()), dtype=torch.uint8).clone()
            # census: dequantise and measure rel-rms against bf16, on GPU
            from comfy_kitchen.tensor import QuantizedTensor
            qt = QuantizedTensor(qdata, "TensorCoreNVFP4Layout", params)
            wd = qt.dequantize().to(torch.float32)
            rel = ((wd - w.float()).norm() / w.float().norm()).item()
            census[base] = {"proj": quant, "block": b, "shape": list(w.shape), "rel_rms_nvfp4": rel}
            del w, wd, qt
            if i % 50 == 0:
                print(f"{i}/{len(keys)} {k} rel {rel:.4f}  {time.time()-t0:.0f}s", flush=True)
    meta = {"a6_regime": a.regime, "a6_projections": ",".join(projs), "a6_blocks": f"{lo}-{hi}",
            "a6_source": os.path.basename(SRC), "a6_built": time.strftime("%Y-%m-%d %H:%M"),
            "a6_input_scale": "static" if amax else "dynamic (per call / per chunk amax)"}
    save_file(tensors, out, metadata=meta)
    json.dump({"meta": meta, "layers": census}, open(out.replace(".safetensors", ".census.json"), "w"), indent=1)
    rels = [v["rel_rms_nvfp4"] for v in census.values()]
    print(f"wrote {out} ({os.path.getsize(out)/1e9:.2f} GB, {len(census)} nvfp4 layers, median rel-rms {sorted(rels)[len(rels)//2]:.4f}); {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
