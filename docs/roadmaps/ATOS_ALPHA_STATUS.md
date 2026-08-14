# ATOS alpha surface: status ledger

Living document, updated 2026-08-13. What the alpha-fenced features have
shipped, what has been validated, and what remains, in priority order.
Companions: `ROADMAP.md` (methods under investigation), `TESTING_ALPHA.md`
(how to exercise the alpha surface), `TUNING.md` (the operating doctrine).

## Shipped and validated (alpha label still on)

- **Rolling-window regeneration** (`H3 Window Plan` / `H3 Window Collect`,
  the requeue driver). Coverage defaults to the full clip so calm spans get
  the same second-pass repaint as the action; one-click batch queueing via
  the window widget's increment control; windows bank durably between queue
  items. Measured on a fenced 32 GB budget: the one-pass de-rope peaked at
  30.7 GB and streamed the DiT layer by layer, two windows peaked at
  22.5 / 24.6 GB with weights resident, at wall-time parity (440 s vs
  460 s). Example validated end to end. To drop the alpha label: field
  reports from real 24 to 32 GB cards.
- **Arbitrary source video** (`H3 Video Fit`): fits any clip to the 17k+5
  grid with audio cut to match. First GPU executions 2026-08-13. The
  shipped example carries a placeholder filename by design.
- **Audio clock discipline**: `H3 Audio Recover` retimes sample-exact
  against the world clock; splice and window collect place audio at
  absolute frame-derived offsets, so seams cannot accumulate error.
  Regression guard in `tests/test_audio_recover.py` (synthetic mode and a
  real-render file-pair mode). The 24 fps video vs 40 Hz audio-latent
  incommensurability is documented in TUNING with the aligned lengths
  (39 / 90 / 141 / 192 frames).
- **Cost readouts** on the oracle, smear, and manual hold map: shipped; the
  minutes estimate is a ballpark and says so.

## Rules minted this cycle (measured, envelopes stated in TUNING)

- A turbo pass 1 tolerates only a SHALLOW turbo pass 2: 1 step at inject
  0.20 held up in playback; 3 or 6 effective steps at 0.50 broke into
  mosaic on the same clip, while the identical recipe is clean over a bare
  12-step pass 1. The init is the variable, not the recipe.
- Pixel-space polish taps do not stack: every extra tap pays a VAE
  encode/decode roundtrip that costs more detail than a shallow solve
  returns. One tap is the recipe.
- `reference_mix` defaults to 1.0, the pass-1 track intact.

## Open, in priority order

1. **Conditioning cache across windows.** The text-encoder spike is what
   breaks small cards (21.2 GB resident vs 15.4 GB sustained, measured),
   and the requeue driver re-pays it per window for an unchanged prompt.
2. **Per-token abstention.** The shipped `abstain_below` gates the whole
   clip; window seam quality wants per-region abstain so cold cuts exist
   wherever the clip is genuinely calm.
3. **Window driver bake-off.** `H3WindowLoop` and `H3WindowExpand` have
   never rendered. Compare all three drivers on renders, cull to one.
4. **Clock trim for exported clips.** Concatenation outside this pack
   accumulates up to 12.5 ms of audio-length error per clip; a trim step
   on the way out closes it.
5. **Cost model calibration.** The superlinear per-step exponent was fitted
   at a single dilation; the exponent is untested elsewhere.
6. **SAM-based subject targeting: NEXT UP (design reviewed 2026-08-13).**
   Human-seeded subject selection driving the existing Manual Hold Map /
   Motion Composite / V2V freeze machinery, so "fix my subject, leave the
   background alone" stops requiring hand-drawn masks. Design review
   findings that now shape the build: (a) the first enabler is a
   TIME-VARYING freeze mask in `H3 V2V Init`, which today unions any mask
   over time into one static plane; the latent `noise_mask` is already
   shaped `(1, 1, t_lat, h, w)`, so this is tractable; (b) any tracking
   state must name which clock its masks live on (world frames, dilated
   latent frames, or segment-local frames; three clocks are in play);
   (c) video segmenters' documented weak spot is thin fast-moving objects
   and long occlusions, exactly this tool's regime, so mid-clip re-seeding
   is a design requirement, not a fallback.
7. **Adaptive compute / spatial foveation: the standing research thread.**
   Per-step cost is superlinear in token count (measured exponent ~1.7),
   so genuine spatial token reduction pays more than proportionally. The
   spatial sibling of the rolling window. Probes will follow the same
   measure-then-mint loop as the temporal work.
8. **Continuation and temporal-inpainting UX.** The clock discipline above
   is the prerequisite and is done.

## Recently retired

- Same-resolution audio-only repaint: at delivery resolution it costs the
  same as a quality video pass, and the video pass buys better pixels AND
  base-model audio. The low-resolution-conditioning variant (cheap full-step
  audio against a small pinned video) remains open.
- LTX transfer experiment: deprioritized after weak first-pass community
  reports for extreme motion; the useful weights are parked for later
  low-motion upscaling work.
