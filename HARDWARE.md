# Hardware vs settings

Which graph and which dials for the machine you have. Two numbers matter
and they are not the same number: **VRAM** decides whether the dilated
pass fits, **system RAM** decides whether the whole pipeline fits, because
the decoded frames of pass 1, the smeared frames and the recovered frames
all live on the host as float images. People with a 24 GB card and 32 GB
of RAM hit the RAM wall first.

Everything here was measured on one box (RTX PRO 6000 Blackwell 96 GB,
91 GB RAM) with the smaller cards simulated by fencing VRAM and RAM to a
hard budget, so a "16 GB" row means a real 16 GiB ceiling, desktop
included. Cells marked *extrapolated* come from the fitted line in
[`LOWVRAM.md`](LOWVRAM.md#how-far-you-can-go-extrapolated-from-the-lines-above)
and have not been rendered. If you run one of those, your seconds per step
is the most useful thing you can send back.

## Pick a row

| VRAM | RAM | start from | checkpoint | what to expect |
|---|---|---|---|---|
| 96 GB | 64 GB+ | [`examples/motion_pipeline_ref2va_audioinit.json`](examples/motion_pipeline_ref2va_audioinit.json) | int8 | 192 frames at 1 MP in ~12 min; a 702-frame de-rope at 1376x768 renders with everything resident |
| 32 GB | 64 GB | the same, or [`motion_pipeline_lowvram.json`](examples/motion_pipeline_lowvram.json) past ~10 s | int8 or W4A8 | the 702-frame de-rope at 25 GiB peak, 318 s/step, same speed as the 96 GB card (the step is attention-bound at that length) |
| 24 GB | 32 GB | [`motion_pipeline_lowvram.json`](examples/motion_pipeline_lowvram.json) | W4A8 | 702 frames at 23.5 GiB peak, 25.8 GB RSS; 906 frames renders at 515 s/step with RAM at the edge (29.8 GB) |
| 16 GB | 32 GB | [`motion_pipeline_lowvram.json`](examples/motion_pipeline_lowvram.json) | W4A8 + NVFP4 text encoder | 702 frames at 15.5 GiB peak, 27.4 GB RSS, 316 s/step; 906 frames does not fit |
| 15 GiB (a 16 GB card with its desktop up) | 48 GB | the same with `kv_store` = `kvi8s` | W4A8 | the exact K/V store runs out of memory at the second forward; `kvi8r` renders (+9.5 GiB, 374 s/step), `kvi8s` renders (219 s/step), `kvi8s` + `trim_forward` renders with zero allocator trims |
| 12 GB | any | not measured | | the line says ~150k packed tokens, roughly a 5 s clip de-roped at 1 MP. [`motion_pipeline_rolling_window_lowvram.json`](examples/motion_pipeline_rolling_window_lowvram.json) makes peak memory follow the window, not the clip, and is the graph to try; reports wanted |

The ComfyUI flags the small-card rows were measured with: dynamic VRAM on
(the default since 0.33), `--fast-disk`, and nothing else. `--reserve-vram`
is a global hammer and was not needed on any row.

## How long a clip fits

Packed tokens is the unit the card cares about, not seconds. A frame at
1376x768 is about 310 tokens after dilation, and a de-rope at `d_max` 4
on a bursty clip roughly doubles to triples the frame count, so the
dilated pass is what you are sizing for, not the clip you typed.

| VRAM | max packed tokens | dilated frames at 1376x768, one pass | RAM at that length, single pass / full pipeline |
|---|---|---|---|
| 16 GB | ~230k | ~740 (measured: 702 renders, 906 does not) | ~28 / ~40 GB |
| 24 GB | ~380k | ~1230 (measured: 906 renders) | ~33 / ~45 GB |
| 32 GB | ~530k | ~1720 *extrapolated* | ~39 / ~51 GB *extrapolated* |

Beyond the row you are on, the answers in order of cheapness: lower
`d_max` (3 instead of 4 is a third fewer dilated frames), raise `q` (0.85
dilates only the hottest spans), drop the resolution of pass 1 and let
[`motion_pipeline_upscale_derope.json`](examples/motion_pipeline_upscale_derope.json)
do the de-rope at the size you want, then the rolling window.

## Seconds per step

The only per-step number we can give you is ours, and it scales
superlinearly with tokens (exponent ~1.7), so double the frames costs
more than double the time:

| clip | card | s/step |
|---|---|---|
| 5 s at 1024x1024, plain pipeline | RTX PRO 6000 | ~11.5 |
| 702 dilated frames at 1376x768 (~217k tokens), pass 2 | RTX PRO 6000, or a fenced 16 to 32 GB budget of it | 311 to 318 |

A 5090 user reported the VAE encode of 120 frames at ~1 MP at 20.5 s
(fp16) and 0.41 s with the TAE encoder; our own encoder line is 0.35 s per
dilated frame at 1.2 MP, flat 5 GiB, and the int8 VAE is no faster at
encoding (it only halves the weights). The encode is the stage that hurts
most on a small card, which is why the TAE matters: it needs a ComfyUI
newer than 2026-08-17 (core PR #15695).

## Quality tiers on the same hardware

Every row above also has a time dial that is not hardware at all:

| want | pass 1 | pass 2 | cost vs one baseline |
|---|---|---|---|
| scout a prompt | 0.2 MP, 12 steps | turbo, 0.4 MP | ~1x, 95 s on our card |
| the normal final | 12 steps, base model | turbo, 6-step `beta`, inject 0.70 | ~1.3x |
| the slow final | 25 steps | base model, inject 0.70 | ~3.5x |

The middle row is the August 19 recipe and it wants the whole recipe
(`linear_quadratic` + `gradient_estimation` on pass 1, `beta` on pass 2,
seeded audio): the turbo LoRA dropped into the 25-step graph on its own
renders jerky. [`examples/README.md`](examples/README.md) has the two
recipes side by side.

## System RAM, specifically

Measured on the 702-frame de-rope: 25.8 to 27.4 GB RSS on the small-card
rows, with the text encoder's CPU mirror the avoidable part. What helps,
in order: `H3 Evict Text Encoder` after each prompt encode (1 to 2.5 GiB
of VRAM peak back on a 16 GB card, and the mirror goes with it), keeping
pass 1 at the resolution you need and no higher, and not running two
ComfyUI instances that each hold a copy of the weights on the host. A
`--gpu-only` instance removes the host mirror entirely but makes unload a
no-op, so it only suits a card that holds everything.

## Send numbers back

An issue with your card, RAM, ComfyUI version, the graph, the packed token
count from the `H3 Jerk Oracle` cost readout, and your seconds per step
fills in a row of this page for everyone after you. Failures on content
unlike ours are worth more than successes on content like ours.
