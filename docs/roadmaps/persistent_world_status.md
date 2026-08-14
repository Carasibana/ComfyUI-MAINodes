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
