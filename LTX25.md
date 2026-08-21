# LTX-2.5 as a de-rope / upscale pass (branch `ltx25-derope`, not shipped)

This branch carries the graphs and tools behind the public de-rope deck
(https://matlowai.github.io/flipbook/derope.html), so you can run the LTX-2.5
arms yourself. It is a branch on purpose: what ships on `main` is the H3
Motion Lab, and this will not merge until the oracle-shaped clock below exists.

## What it does today: a FLAT clock

An H3 de-rope (`H3JerkOracle` -> `H3TimeSmear` -> regenerate -> `H3ExactRecover`)
slows each frame by its own hold (1 to 4, with ramps and bridges), which is what
makes the regenerated motion land on the right frames. The LTX-2.5 graphs here
use the H3 oracle only to choose WHERE the window sits (its hot span), and inside
the window every frame is held exactly d times (`ImageBatchRepeatInterleaving`,
d = 8 or 16), padded to LTX's legal length (8k+1), regenerated from a given
starting sigma, and recovered by taking every d-th frame.

Why flat: LTX compresses time by 8, so a held frame only owns its own latent
block when its hold is a multiple of 8 frames. Holds of 1 to 4 share blocks, and
measured on two scenes that is exactly where d=8 barely responds to the denoise
dial (0.94 to 0.90 across 0.1 to 0.9) while d=16 moves it (0.90 to 0.28).

Per-pass cap: 57 source frames at d=16 (`positional_embedding_max_pos[0] = 20`
with time in absolute seconds). Full coverage of a longer clip is several
overlapping windows joined by `tools/ltx25/splice_windows.py`, which measures
PSNR across each overlap before it blends.

## In progress: the oracle-shaped clock, and a per-model adaptation layer

The right version keeps the H3 hold map as metadata and scales it onto the
target model's clock: LTX's 8-frame block as the unit, hold-1 frames left at
native rate (that is just video for LTX), hold 4 -> 16x, ramps -> 8x. Sized on
the saved oracle profiles that fits one LTX pass (panrun 712 / swordspin 816
dilated frames against a 913 cap), same cost as the uniform d=16 window with
the time budget spent where the oracle put it.

The generic form is a small per-model config (temporal compression, legal
frame grid, fps, positional cap, recover phase) behind one "clock remap" node,
so the same hold map drives H3, LTX-2.5, or Wan 2.2 once the config row exists.
The oracle itself reads H3 latents; a pixel-domain oracle would make the
front end model-agnostic too.

## Files

- `examples/ltx25/` - four graphs as run for the deck, API JSON (`*_api.json`,
  what actually executed) and UI JSON (paste on the canvas):
  `panrun_or_d16_d040` (uniform d=16 de-rope of the oracle window),
  `swordspin_or_d8_d050` (d=8), `swordspin_or_x2_d8_d050_d030` (one-pass x2
  latent upsample of a d=8 pass), `panrun_ltx_d040_full` (first of the three
  windows behind the spliced full-length clip).
  Source clips are referenced by basename: put yours in ComfyUI `input/`.
- `tools/ltx25/mint_ltx25_derope.py` - mints a uniform de-rope graph from the
  stock LTX-2.5 T2V template (`--frames --fps --dilation --seed ...`).
- `tools/ltx25/mint_ltx25.py` - the plain LTX-2.5 graph minter the ladder used.
- `tools/ltx25/ltx25_oracle_window.py` - reads a saved H3 oracle profile and
  returns the legal LTX window over its hot span.
- `tools/ltx25/splice_windows.py` - joins window renders, checking alignment.
- `tools/ltx25/ui2api.py` - the UI -> API converter the minters depend on.

Model files are the stock Comfy-Org LTX-2.5 set (22B transformer, its VAE and
audio VAE, the x2 latent upsampler) plus the MiniMax-H3 set for the H3 arms.
