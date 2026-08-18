# SPDX-License-Identifier: GPL-3.0-or-later
"""Gate (a) for vram_lab: streamed block == stock DiTBlock on a random tiny block.

Run from the ComfyUI root with the ComfyUI venv:
    CUDA_VISIBLE_DEVICES=1 python custom_nodes/ComfyUI-MAINodes/tests/test_vram_lab_block.py

Uses comfy.ops.disable_weight_init (bf16 weights, no quantization) so this
proves the schedule; the int8 row-exactness is a separate kernel property
measured 2026-08-18. Prints max abs diff per configuration; PASS threshold is
a handful of bf16 ulps at the residual's magnitude (matmul order differs
between chunked and whole GEMMs on cuBLAS, so bit-equality is not expected on
this path).
"""
import os
import sys

sys.path.insert(0, os.getcwd())
import torch

import comfy.options
comfy.options.enable_args_parsing()
import comfy.model_management  # noqa: E402
import comfy.ops  # noqa: E402
from comfy.ldm.minimax.model import DiTBlock, rope_rotation_table  # noqa: E402

sys.path.insert(0, os.path.join(os.getcwd(), "custom_nodes"))
from importlib import import_module  # noqa: E402
vram_lab = import_module("ComfyUI-MAINodes.vram_lab")

torch.manual_seed(0)
dev = "cuda"
dt = torch.bfloat16
hidden, heads, hd, ffn, t_dim = 512, 8, 64, 1024, 256
S = 6000
ops = comfy.ops.disable_weight_init

block = DiTBlock(hidden, heads, hd, ffn, t_dim, 1e-6, 1e-6, dtype=dt, device=dev, operations=ops)
with torch.no_grad():
    for p in block.parameters():
        p.copy_(torch.randn_like(p) * 0.02)
        p.requires_grad_(False)
torch.set_grad_enabled(False)

x0 = torch.randn(S, hidden, device=dev, dtype=dt)
t_emb = torch.randn(3, t_dim, device=dev, dtype=dt)
# three segments with different modulation rows, like text | audio | video
segments = [(0, 300, 1), (300, 900, 2), (900, S, 0)]
rot_dim = 32
angles = torch.randn(S, rot_dim, device=dev) * 0.5
angles[:, rot_dim // 2:] = angles[:, :rot_dim // 2]
rope = rope_rotation_table(angles, dt)  # [1, S, 1, rot/2, 2, 2]

with torch.no_grad():
    ref = block(x0.clone(), t_emb, segments, rope, transformer_options={})
    scale = ref.float().abs().max().item()
    ulp = 2.0 ** (torch.floor(torch.log2(torch.tensor(scale))).item() - 7)  # bf16 ulp at max magnitude
    print(f"ref max |x| = {scale:.4f}, bf16 ulp there ~ {ulp:.2e}")
    ok = True
    for cfg in [dict(q_chunk=S, kv_chunk=S, mlp_chunk=S),
                dict(q_chunk=1024, kv_chunk=1024, mlp_chunk=1024),
                dict(q_chunk=700, kv_chunk=1500, mlp_chunk=0),
                dict(q_chunk=1024, kv_chunk=1024, mlp_chunk=1024, kv_block=1024),
                dict(q_chunk=512, kv_chunk=4096, mlp_chunk=2048, kv_block=700)]:
        out = vram_lab.streamed_block_forward(block, x0.clone(), t_emb, segments, rope, {}, **cfg)
        d = (out.float() - ref.float()).abs().max().item()
        verdict = "PASS" if d <= 8 * ulp else "FAIL"
        ok &= verdict == "PASS"
        print(f"{verdict} {cfg}: max abs diff {d:.3e} ({d / ulp:.1f} ulp)")
    # segment-modulation sanity: a wrong segment clip would show as a large diff at boundaries
    out = vram_lab.streamed_block_forward(block, x0.clone(), t_emb, segments, rope, {}, q_chunk=250, kv_chunk=250, mlp_chunk=250)
    d = (out.float() - ref.float()).abs()
    print(f"chunk 250 (cuts every segment): max {d.max().item():.3e}, at rows 299/300 {d[299].max().item():.2e}/{d[300].max().item():.2e}")
    ok &= d.max().item() <= 8 * ulp
print("ALL PASS" if ok else "SOME FAIL")
sys.exit(0 if ok else 1)
