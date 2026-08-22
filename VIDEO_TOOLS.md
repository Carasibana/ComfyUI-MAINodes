# MAI video tools (branch `generic-derope`, alpha)

Three nodes and one widget that remove drudgery around renders without taking
any decision away from you. None of them runs a model.

## MAI Video Out (file + sidecars + draft preview)

Save Video with the record beside it. Frames (+ audio) go through ComfyUI's
own encoder (h264 / av1, CRF), exactly like Create Video + Save Video; the
extras are each off unless wired or toggled:

- `hold_map` -> `<prefix>_00001_.holdmap.json`: the clock that produced these
  frames. H3 Load Hold Map and the deck's hold-map staircase read it.
- `write_meta` -> `<prefix>_00001_.meta.json`: seeds, models, LoRAs, steps,
  samplers, inject schedule, encode time, your `notes`, all read out of the
  graph that ran. The compare viewer's metadata strip and the scorer's labels
  come from here instead of from file names.
- `draft_preview` + `latent`: decode the sampled latent through the tiny VAE
  in `models/vae_approx` (`taeh3` for H3; any stem) into `<prefix>_draft.mp4`.
  The tiny VAE is loaded for that call and released after; with the toggle
  off nothing is loaded and nothing runs. Cost measured 2026-08-21: 8 ms per
  latent frame at 864x480, 21 ms at 1344x768, 42 ms at 1080p.

## MAI Load Video (path) / MAI Select Every Nth

The two things the graphs still took from VideoHelperSuite: a video by
absolute or input-relative path (as ComfyUI's VIDEO type, frames and fps
alongside), and every n-th frame from an offset. For a smeared clip use H3
Exact Recover instead; it is exact.

## MAI Video Compare (2-6, synchronized)

A media UI, not a processing stage. The node writes each wired VIDEO once as a
small h264 preview into the temp directory (CPU, no VAE, no tensors kept) and
the widget plays them in the browser with the flipbook's player: one set of
live `<video>` elements, buffered start-together, flip / wipe / side-by-side
for a pair, a grid for more.

- hover a source to hear it, click to lock its audio
- space play/pause all, arrows step all one frame (shift: 12), F flicker A/B
- 1-6 pick the pair (last two pressed), Enter stars the B side
- the star sets the node's `winner` widget; the NEXT queue passes that source
  through `winner_video` (and `winner_index`). Pick, then finalize in a second
  execution: a graph never waits on a human mid-run.

The flipbook deck (https://matlowai.github.io/flipbook/derope.html) is the
offline twin, same player code.

## MAI Seed Hunter (compare + winner seed)

Video Compare that knows about seeds: wire 2-6 cheap candidates and the seed
each ran on (labels default to the seed), star the keeper, and the NEXT queue
passes the starred candidate's seed out of `winner_seed` for the finalize
pass's RandomNoise (and its video out of `winner_video`). The seed-hunt pass
and the finalize pass are two executions by design.

## MAI Motion Editor additions

- **Play** (space): the filmstrip plays at fps; in the dilated clock each
  frame lingers for its hold, so the slowdown is seen, not inferred.
- **clock: world / dilated**: the timeline's x-axis reads as "when" (frame)
  or as "how long it costs" (cumulative holds). The compiled hold curve from
  the last run is drawn over the jerk profile in both.

## What these replace

Our graphs used VHS_LoadVideoPath and VHS_SelectEveryNthImage; both are
covered. Create Video + Save Video stay valid; MAI Video Out is those plus the
sidecars. The compare node replaces opening three files and scrubbing them to
roughly the same place.
