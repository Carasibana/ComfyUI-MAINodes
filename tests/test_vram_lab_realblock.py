# SPDX-License-Identifier: GPL-3.0-or-later
"""Gate (a') for vram_lab on the REAL H3 weights: stock DiTBlock vs streamed on
block 0 with a real PackedLayout, phase by phase, so a mismatch is localised.

    CUDA_VISIBLE_DEVICES=0 H3_CKPT=<ckpt> python custom_nodes/ComfyUI-MAINodes/tests/test_vram_lab_realblock.py
        models/diffusion_models/minimax_h3/minimax_h3_fl2va_pruned_int8_convrot.safetensors
"""
import os
import sys

sys.path.insert(0, os.getcwd())
import torch

import comfy.options
comfy.options.enable_args_parsing()
import comfy.model_management as mm  # noqa: E402
import comfy.sd  # noqa: E402
from comfy.ldm.minimax.model import PackedLayout, rope_rotation_table  # noqa: E402

sys.path.insert(0, os.path.join(os.getcwd(), "custom_nodes"))
from importlib import import_module  # noqa: E402
vram_lab = import_module("ComfyUI-MAINodes.vram_lab")

torch.set_grad_enabled(False)
path = os.environ["H3_CKPT"]
patcher = comfy.sd.load_diffusion_model(path)
mm.load_models_gpu([patcher], force_full_load=True)
dm = patcher.model.diffusion_model
dev = mm.get_torch_device()
dt = torch.bfloat16
block = dm.blocks[0]
print("block0 qkv weight:", type(block.attn.qkv_proj.weight).__name__, getattr(getattr(block.attn.qkv_proj.weight, "_layout_cls", None), "__name__", None) if hasattr(block.attn.qkv_proj.weight, "_layout_cls") else "")

# real gate layout: 64 text tokens, 27 latent frames at 48x86, 150 audio tokens
layout = PackedLayout(64, 27, 48, 86, 150)
S = layout.seq_len
print("seq_len", S, "segments", layout.segments)
rope = rope_rotation_table(dm.rope_freqs(layout.position_ids, dev), dt)
t_vals = torch.tensor([0.4, 0.55], dtype=torch.float32, device=dev)
if dm.use_adaln_curves:
    table = mm.cast_to(dm.adaln_t_table, device=dev)
    pos = t_vals.clamp(0.0, 1.0) * (table.shape[0] - 1)
    i0 = pos.floor().long().clamp(max=table.shape[0] - 2)
    t_emb = torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))
else:
    t_emb = dm.time_embedder(t_vals).to(dt)
seg_tag = {"text": 1, "video": 0, "audio": 2, "cond": 0, "ref_img": 0, "cond_audio": 2, "ref_audio": 2}
t_row = {"text": 0, "video": 0, "audio": 1, "cond": 0, "ref_img": 0, "cond_audio": 1, "ref_audio": 1}
segments = [(a, b, t_row[k] * 3 + seg_tag[k]) for a, b, k in layout.segments]

torch.manual_seed(1)
x0 = (torch.randn(S, dm.hidden_size, device=dev, dtype=dt) * 1.5)

def ulps(d, ref):
    scale = ref.float().abs().max().item()
    ulp = 2.0 ** (torch.floor(torch.log2(torch.tensor(max(scale, 1e-30)))).item() - 7)
    return d.max().item() / ulp

ref = block(x0.clone(), t_emb, segments, rope, transformer_options={})
ref2 = block(x0.clone(), t_emb, segments, rope, transformer_options={})
print(f"stock twice: bit-equal {torch.equal(ref, ref2)}")
for cfg in [dict(q_chunk=S, kv_chunk=S, mlp_chunk=S),
            dict(q_chunk=S, kv_chunk=S, mlp_chunk=4096),
            dict(q_chunk=S, kv_chunk=4096, mlp_chunk=S),
            dict(q_chunk=4096, kv_chunk=S, mlp_chunk=S),
            dict(q_chunk=4096, kv_chunk=4096, mlp_chunk=4096),
            dict(q_chunk=4096, kv_chunk=4096, mlp_chunk=4096, kv_block=8192)]:
    out = vram_lab.streamed_block_forward(block, x0.clone(), t_emb, segments, rope, {}, **cfg)
    d = (out.float() - ref.float()).abs()
    print(f"{'EXACT' if d.max().item() == 0 else 'diff '} {cfg}: max {d.max().item():.3e} ({ulps(d, ref):.2f} ulp) mean {d.mean().item():.3e} frac!=0 {(d > 0).float().mean().item():.4f}")
