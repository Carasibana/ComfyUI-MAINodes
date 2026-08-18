# Low VRAM, low RAM: the de-rope on a 16 GB card

Status 2026-08-18: alpha. Everything below was measured on one machine, fenced
down to look like small ones. Nothing has been run on a real 16 GB card yet;
that is the next thing, and it is a dogfood ask, not a research question.

## The short version

The de-rope pass at `d_max 4` on an 8 to 12 second clip is ~200k packed tokens.
The stock H3 block builds its fused QKV and SwiGLU tensors for the whole
sequence at once, 8.6 and 15.4 GiB at that length, and that is what OOMs a
24 GB card, and at 1376x768 a 96 GB one. `H3 Streamed Blocks` runs every DiT
block in token chunks with the same math. Together with two things ComfyUI
already ships (dynamic VRAM, `--fast-disk`) it changes the small-card story
from "OOM" to "same speed":

| card | machine | 294 f -> 702 f de-rope at 1376x768 (~217k tokens) | s/step (pass 2) |
|---|---|---|---|
| 96 GB, everything resident (`--gpu-only`) | 91 GB | renders | 311 |
| 32 GB (fenced), normal mode, `--fast-disk` | 64 GB (cgroup) | renders, 25 GiB VRAM peak | 318 |
| 24 GB (fenced) | 32 GB (cgroup) | renders, 23.5 GiB VRAM peak, 25.8 GB RSS | 318 |
| **16 GB (fenced)** | **32 GB (cgroup)** | **renders, 15.5 GiB VRAM peak, 27.4 GB RSS** | **316** |
| 16 GB (fenced) | 32 or 64 GB | 906 f (~275k tokens): out of memory | |
| 24 GB (fenced) | 32 GB | 906 f: renders, RSS 29.8 GB (RAM at the edge) | 515 |

Same seed, the streamed result is bit-equal to the stock result on the int8
and W4A8 checkpoints (video and audio latents compared tensor by tensor). It
is not a quality dial. It is also not slower: at these lengths the step is
attention-bound and the weight traffic hides under it (see "Why no speed
penalty").

## What to run

1. Start ComfyUI **without** `--gpu-only` and **with** `--fast-disk`.
   Dynamic VRAM (the default) streams the DiT weights and keeps as many
   resident as the card allows; `--fast-disk` reads them from the NVMe page
   cache instead of holding a copy of every model in RAM. Without it a
   normal-mode run of this pipeline held 59 GB of RSS; with it, 36 GB.
2. Load `examples/motion_pipeline_lowvram.json` (API twin alongside). It is
   the `motion_pipeline_ref2va_audioinit` graph with:
   - the W4A8 checkpoint (`minimax_h3_ref2va_pruned_w4a8_mixed`, 11 GB) in the
     UNET loader,
   - `H3 Streamed Blocks` in the model chain, defaults (16k-token chunks,
     `final_layer_gemm` = exact),
   - `H3 Free Cache` between the pass-2 sampler and its decodes,
   - `H3 Evict Text Encoder` after each prompt encode.
   The KJNodes sage-attention patches and FFN chunker from the parent graph
   are not in it: sage attention is an approximation and the exactness gates
   here were run on PyTorch's flash attention.
3. Set your prompt, reference image and canvas as usual.

## What to expect

**VRAM.** The activation peak inside a forward is card-independent:
`0.3 GiB + ~53 KB per packed token` (K/V buffers 28.7 KB/token, the residual
10.8, chunk transients and two tensors ComfyUI's forward keeps alive).
At 702 frames / 217k tokens that is +11.9 GiB. Dynamic VRAM fills whatever is
left with resident weights, so the process peak reads close to the card size
on every card; what decides whether a length fits is `weights it must keep +
activations + ~1 GB context`. Measured: 702 frames fits a 16 GB card, 906 does
not (275k tokens: +14.9 GiB of activations plus the minimum resident weights).
Frames here are 1376x768; other canvases scale by tokens
(`latent_frames x (W/32) x (H/32)`).

**RAM. This is the tight budget, not VRAM.** With `--fast-disk`, process RSS
was 27.4 GB on the 16 GB card and 25.8 GB on the 24 GB card for a single
702-frame pass, 36 to 40 GB for the full de-rope pipeline (it holds the pass-1
decodes, the smear IMAGE, the heat map: a 702-frame fp32 IMAGE is 8.9 GB), and
59 GB without `--fast-disk`. Add your own headroom: an idle Ubuntu desktop is
~2 GB, an idle Windows 11 desktop 4 to 8 GB, ComfyUI itself 3 to 5 GB before
any model. So: 16 GB card + 32 GB RAM works on a lean Linux box and is
already paging on Windows; 48 to 64 GB is comfortable. Under our 64 GB fence
the kernel reclaimed page cache before killing anything; a Windows box would
pagefile instead, which is slower but alive.

**Time.** 316 to 318 s per pass-2 step at 217k tokens on the fenced 16, 24 and
32 GB cards versus 311 resident on 96 GB. On the 90-frame clip the 16 GB card
did 8.2 s/step. The text encoder costs 3 to 4.5 s per new prompt from a warm
page cache (a cold read of the 15 GB file is longer), and 1 to 2.5 GiB of
VRAM peak on a 16 GB card, which the evict node removes.

**The text encoder on tiny cards**, measured on the fenced 16 GB card, 90 frames:

| arm | wall | encode | VRAM peak | RSS |
|---|---|---|---|---|
| stock CLIPLoader | 221 s | 3.0 s | 15.7 | 14.3 |
| second prompt in the same process | 218 s | 1.7 s | 16.6 | 15.5 |
| `H3 Evict Text Encoder` after the encode | 221 s | 3.1 s | 13.9 | 14.7 |
| `CLIPLoader device=cpu` | 272 s | 55 s | 14.0 | 26.9 |
| conditioning bank hit (encoder never runs) | 218 s | none | 14.7 | 12.4 |

Nothing thrashed. Evict is the default in the example graph because a normal
run types a new prompt; `H3 Conditioning Bank` is for repeated prompts (seed
hunts, windows, extension chains) and needs the sampler's latent to come from
somewhere other than the encode node (`EmptyMiniMaxH3LatentAV` for T2VA, the
latent bank for a banked pass 1), or the encode node runs anyway for its
latent output.

## How far you can go (extrapolated from the lines above)

The activation line predicted the 906-frame point on the 24 GB card before it
was measured (14.9 GiB both ways), so the table below is the line, not a
guess; treat it as +-1 frame-block. "Max tokens" assumes dynamic VRAM keeps
the minimum of resident weights (2 to 3 GB measured) and ~1 GB of context.

| card | max packed tokens | frames at 1376x768, single de-rope pass | RAM at that length, single pass / full pipeline |
|---|---|---|---|
| 16 GB | ~230k | ~740 (measured: 702 renders, 906 does not) | ~28 / ~40 GB |
| 24 GB | ~380k | ~1230 (measured: 906 renders at 515 s/step) | ~33 / ~45 GB |
| 32 GB | ~530k | ~1720 | ~39 / ~51 GB |

RAM per frame is ~12 MB (one fp32 IMAGE at 1376x768) on top of a ~19 GB base
for a single pass with `--fast-disk`; the full pipeline holds two to three
more IMAGEs of the clip. Past ~1100 frames the step time (attention grows with
the square of the sequence: 316 s at 702 frames, 515 at 906, ~650 at 1110)
is the practical limit before memory is. Other canvases: convert to tokens
with `latent_frames x (W/32) x (H/32)`, where `latent_frames = (frames-5)/17*5+2`.

The example graph has not yet been rendered end to end on the fenced card;
its pieces have (same nodes, same passes). That gate is next.

## Why no speed penalty

At 217k tokens about 250 of the ~315 seconds per forward are attention, which
grows with the square of the sequence. Streaming the whole 11 to 12.5 GB DiT
once per forward is ~0.5 s over PCIe from RAM and ~2 s from an NVMe page
cache, and dynamic VRAM prefetches the next layers while the current ones
compute. Weight traffic is a fixed cost per forward; attention is quadratic;
the long clip hides the traffic completely, and on the short one prefetch
did the same (8.2 s/step on 16 GB versus ~9 resident on 96 GB). This box has
PCIe 5 x16 and a fast NVMe; a PCIe 4 desktop halves the bandwidth, which is
still nothing against a 300 s step. A machine whose RAM cannot hold the
weight file's page cache and whose SSD is slow could show it on short clips.

## What is exact, and what is not

- `H3 Streamed Blocks` on int8 and W4A8 checkpoints: bit-equal to stock. The
  mechanism is that comfy_kitchen quantises activations per row and the int8
  GEMM accumulates in int32, which is associative, so chunking rows cannot
  change the sum. NVFP4/FP8 would need one shared activation scale and are
  untested; bf16 checkpoints are numerically equivalent (fp accumulation
  order), not identical.
- Query chunking is exact only while every chunk keeps PyTorch's flash
  kernel off its split-KV path (measured boundary `heads x ceil(L/64) >=
  0.8 x 2 x SMs`; a 259-token tail once made a different clip). The node sizes
  chunks from the SM count with a 2.6x margin and has a `self_check` that
  compares stock and streamed on the first block's real input.
- The output head has an exact mode (default) and a chunked-GEMM mode that
  differs by ~1e-6 per step in fp32 and produced a different clip after 25
  steps. It is labelled and off. `kv_block` is experimental and as built does
  not lower memory; leave it at 0.
- `H3 Free Cache` and `H3 Evict Text Encoder` change no math.

## The environment this was measured in

One workstation: two RTX PRO 6000 Blackwell (96 GB, 188 SMs), EPYC 9554,
91 GB RAM, NVMe, Ubuntu, ComfyUI 0.33.0, PyTorch 2.14 nightly cu132,
comfy_kitchen native ops, PyTorch flash attention (comfy default), MiniMax-H3
W4A8 and int8 checkpoints. Small cards were simulated by holding VRAM with a
balloon process so ComfyUI, dynamic VRAM and torch all saw a real 16 / 24 /
32 GB card; small machines by running ComfyUI in a cgroup with a hard memory
limit and swap off. Only ComfyUI ran during the measurements. Not measured:
Windows, other GPUs, ROCm, non-flash attention backends, real PCIe 4 boxes.

The graph of the lines (VRAM activation line with per-card ceilings, RSS by
graph type) is `LOWVRAM_lines.html` next to this file; it grows as the sweep
does.
