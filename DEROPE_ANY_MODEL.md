# De-rope for any model (branch `generic-derope`, not shipped)

This branch carries two things: the LTX-2.5 graphs and tools behind the public
de-rope deck (https://matlowai.github.io/flipbook/derope.html), and the
any-model layer that lets one hold map drive any regenerating model through
per-model presets. It is a branch on purpose: what ships on `main` is the H3
Motion Lab; this merges once the oracle-shaped LTX arm has been measured.

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

## The any-model layer (new nodes)

The smear, recover and audio nodes were already pixel-level and model-agnostic;
what was H3-specific was the grid law (17k+5 lengths, holds in H3 frames). That
law is now data, `model_profiles.py`, and one node applies it:

- **H3 Clock Remap (any model, presets)** - wire any hold map (H3 Jerk Oracle,
  H3 Manual Hold Map, H3 Drawn Plan), pick the model that will regenerate the
  frames. Holds of 2+ are scaled by the profile's `hold_scale` and quantised
  to whole latent time blocks; holds of 1 stay 1 (plain video anywhere). The
  output map carries the target's legal grid, so **H3 Time Smear pads to that
  grid** (LTX-2.5: 8k+1, not 17k+5) and H3 Exact Recover / H3 Audio Recover
  invert it unchanged. `minimax-h3` is the identity: existing graphs do not
  change by construction (tests assert it). The report is the price tag: frames,
  seconds at the model's fps, windows needed under its cap, and a loud
  UNMEASURED line for any profile whose numbers did not come from a ladder.
- **H3 Save Hold Map (sidecar)** - writes `<prefix>.holdmap.json` beside the
  render. Without it a render's clock is lost when the graph closes; with it the
  staircase ruler can read H3 arms and a later remap can retime the same plan.
- **H3 Jerk Oracle `profile_mode`: "value |d3| camera-compensated"** - aligns
  each latent frame to its predecessor by the best small integer shift before
  differencing, so a pan or scroll stops scoring as jerk (the documented cause
  of panrun dilating 124 -> 345 frames). Same compiler, different signal.

### Profiles

| id | block | hold_scale | legal | fps | cap | measured |
|---|---|---|---|---|---|---|
| `minimax-h3` | 1 | 1 | 17k+5 | 24 | VRAM | yes |
| `ltx-2.5` | 8 | 4 | 8k+1 | 48 (the conditioning frame_rate) | 20 s / pass = 960 frames | yes (2026-08-21 ladder) |
| `wan-2.2 (unmeasured)` | 4 | 4 | 4k+1 | 16 | none | NO - placeholder |

`hold_scale` is measured, never derived: LTX-2.5 needed two 8-frame blocks per
hot hold (d=16) before the dial responded at all, which its compression ratio
alone would not have told you.

### Adding a model without touching code

Pick `custom (use the fields below)` on the node and type the five numbers; the
report stamps the result unmeasured. When a row works, save it as a preset:
create `<ComfyUI user dir>/mainodes_models.json` (or point
`$MAINODES_MODELS_JSON` at a file) with

```json
{"my-model": {"name": "My Model", "block": 4, "hold_scale": 4,
              "legal": [4, 1], "fps": 16, "cap_seconds": null, "measured": false}}
```

and restart: it appears in the dropdown, and a row with a preset's id
overrides that preset. Send us the row and the ladder behind it.

### Sizing the oracle-shaped LTX arm

On the saved oracle profiles the x4 rule gives panrun 712 / swordspin 816
dilated frames over the hot span, under the 960-frame cap (913 was the largest legal length used), so it fits one
pass at the same cost as the uniform d=16 window with the time budget spent
where the oracle put it. That arm is the one measurement left before this
merges. The x8 rule (hold 4 -> 32x) needs two windows per scene.

## Files

- `examples/ltx25/derope_any_model_ltx25_api.json` - the any-model path on
  LTX-2.5: H3 Manual Hold Map (or an oracle) -> H3 Clock Remap (`ltx-2.5`) ->
  H3 Time Smear -> the LTX-2.5 pass -> H3 Exact Recover, with H3 Save Hold Map
  beside the render. API JSON (drag onto the canvas); the UI twin lands once
  an instance with these nodes can convert it.
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
