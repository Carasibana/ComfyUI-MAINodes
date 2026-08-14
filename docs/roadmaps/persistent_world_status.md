# Persistent generative world: status

Updated 2026-08-13. The full design lives in
[`persistent_world_design.md`](persistent_world_design.md): H3 Ref2VA as
the center of a persistent Gaussian world memory with agentic composition,
transactional world updates, history preservation, and a 3D -> H3 -> 3D
loop. Status: **design only, nothing built.**

## What exists today

- The design thesis itself (1,900 lines, argued end to end).
- Building blocks already shipped in this pack that the design assumes:
  the v2v source path (`H3 Video Fit`), seam and audio clock discipline,
  rolling-window regeneration for long or large work, and the Ref2VA
  full-reference prompting contract.

## Not built

Everything stateful: the persistent Gaussian store, world-update
transactions, change detection, local coordinate systems, the
QuerySplat-class lift, and any engine bridge.

## First steps, cheapest first

1. **Single-scene loop probe**: render a scene, lift it to splats, re-render
   from a new viewpoint via Ref2VA, and measure what survives (identity,
   layout, lighting). One scene, no persistence, answers whether the loop's
   core assumption holds.
2. **Confidence plumbing**: cross-run divergence maps and oracle profiles as
   per-region confidence, feeding the design's confidence-is-first-class
   requirement.
3. **Dependencies from the alpha ledger**: continuation UX and SAM-based
   targeting (see `ATOS_ALPHA_STATUS.md`) are the same machinery this
   design will lean on for world updates.

## Mechanism unifications (2026-08-14 synthesis)

Three findings from the motion program supply machinery this design was
missing:

- **Per-token noise labels as the world's read/write interface.** Store
  per-region confidence in the world memory; express it back to the
  model on re-render as partial denoise labels: well-observed regions
  arrive nearly finished and are preserved, uncertain regions arrive
  noisy and are invented. Confidence-weighted regeneration IS the
  transactional update semantics; a label ramp at the boundary is the
  handover between known and unknown world. This also resolves the
  "lift must not directly replace the world" tension: lift confidence
  maps to noise level, the model adjudicates.
- **One world clock.** The positional time axis is physical (frame-
  linear), and generation supports arbitrary time origins. Rendered
  clips can carry their true world-time origin and become temporally
  addressable in the world's coordinate frame instead of each starting
  at t=0.
- **The budgeted-run shape generalizes.** Budgeted pieces + a banked,
  resumable store + an explicit seam policy + priced pre-run reports:
  the rolling window proves the shape over time; spatial tiles and
  chained scene atoms are the same shape over space and sequence.
  Build them on that skeleton.
