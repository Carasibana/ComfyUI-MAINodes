# Research notes: Adaptive Temporal Over-Sampling (ATOS)

Working notes for turning Motion Lab into a paper. Everything here is
hypothesis and plan, not established result; the demos in the README are
same-seed comparisons on our test clips, and the whole point of this
document is to close the gap between "it looks great here" and evidence.

## The claim we think we can defend

When motion exceeds the temporal representational capacity of a
temporally compressed video generator, do not interpolate the bad output
afterward. Detect the overloaded spans from the model's own latent,
temporally dilate exactly those spans with integer frame holds, let the
generative prior re-solve the motion video-to-video at partial denoise
with more temporal room, then return to the original clock by exact
frame selection. Training-free, no auxiliary model.

Candidate title: "Adaptive Temporal Over-Sampling for Training-Free
Fast-Motion Repair in Latent Video Diffusion".

Three contributions as implemented today:

1. Latent motion-overload detection. The jerk oracle reads a
   phase-normalized third temporal difference of the video latent. No
   optical flow, no segmentation, no extra network.
2. Generative temporal over-sampling with exact recovery. Integer holds,
   partial-denoise regeneration, frame-selection recovery. The slowdown
   lives in the content as a speed ramp; the model clock stays uniform.
3. Adaptive and early allocation. Quantile threshold, bridge and ramp
   turn the signal into a compute policy (about 2.9x frame budget vs
   uniform 4x on our clips); the probe variant reads the oracle from an
   early x0 estimate instead of a finished baseline.

## Positioning vs prior art (as of 2026-08)

- DLFR-VAE and VGDFR recognize that temporal information density is not
  uniform, and spend FEWER resources where motion is low, to accelerate.
  ATOS inverts the direction: it spends MORE effective temporal capacity
  where motion overloads the representation, to repair. Same
  observation, opposite allocation.
- DiffuseSlide is training-free high-FPS generation via keyframes and
  sliding-window denoising; it raises output frame rate rather than
  repairing a native-rate clip and returning to the original clock.
- Large-motion VAE work documents that temporal compression degrades
  under large motion, which is the underlying problem, addressed there
  by training better VAEs rather than at inference time.
- A fresh literature sweep is required before drafting. Search terms:
  training-free temporal super-resolution inside latent video diffusion,
  motion-adaptive resampling at inference, work citing VGDFR.

## What honesty requires

- The README currently implies causality (one token cannot hold four
  distinct poses, therefore the smear). Until the probe below runs, the
  paper says: we hypothesize these failures arise in part from
  fixed-rate temporal compression becoming insufficient for
  high-information motion.
- The latent third difference is a heuristic overload proxy, not
  physical jerk. Whether it beats simpler signals is an experiment, not
  an assumption.
- Everything is currently demonstrated on one model family. The method's
  components (grid constants, phase normalization, token mapping) are
  model-specific in code even if the idea is not.

## Evidence plan

1. Fast-motion benchmark: 30 to 50 prompts x 3 seeds. Categories: rigid
   fast motion, articulated humans, rotations and backflips,
   projectiles, camera pans, camera plus subject motion, plus 10 to 20
   low-motion negative controls. Vary every axis except the one being
   measured.
2. Baselines: native generation; plain partial-denoise V2V with no
   dilation at the same inject; uniform 2x/3x/4x dilation; adaptive
   (ours); adaptive gated to user ranges; post-hoc frame interpolation
   on the baseline; a dynamic-frame-rate method as a reference point if
   feasible.
3. Ablations: difference order (first, second, third), phase
   normalization on/off, quantile q, bridge, ramp, d_max, inject 0.5 to
   0.8, final-latent oracle vs early-x0 oracle across probe step counts.
   Include an external detector baseline: optical-flow magnitude and
   flow acceleration on decoded frames.
4. Causal probes:
   - VAE-only roundtrip: encode/decode real fast-motion footage at
     native speed vs slowed variants and measure reconstruction error
     against local motion magnitude. Tests the compression-bottleneck
     hypothesis with no sampler in the loop.
   - Failure severity vs oracle score on human-labeled clips.
   - Trajectory-bank instrument: checkpoint a run every step, run the
     oracle offline on each step's estimate, and plot agreement with the
     final map as a function of denoising step. This is the figure that
     justifies the early-probe variant.
5. Metrics: separate scores for repair (did the smear go) and
   preservation (choreography, identity, background), plus a blinded
   human A/B in playback. Still frames mislead about temporal quality in
   both directions; playback judgment is primary.
6. Quality/compute Pareto: artifact reduction vs dilated tokens and wall
   time, with uniform, adaptive, gated and segment-cropped variants as
   points on one curve.
7. Cross-model transfer: port oracle, smear and recovery to at least one
   other temporally compressed video model and report the outcome either
   way. This is the highest-value single experiment.

## Ideas ledger (directions, not commitments)

- Spatiotemporal allocation D(x, y, t). The pipeline already carries
  soft spatial masks end to end, but spatial masks only control
  blending; the DiT still pays for every token. A real spatial compute
  claim needs either token dropping/merging inside the model or a
  spatial analog of the segment crop (crop a subject window, regenerate,
  splice with feathered seams).
- Human-seeded motion attachment. The user marks the subject once; a
  segmenter propagates it; an attachment envelope then grows from
  co-motion rather than object identity, so swords, boards, hair and
  splashes join the repair region while a panning background does not.
  Keep the detector (when) and the targeting (where) separable so
  ablations stay clean.
- Latent co-motion test. Correlate per-cell temporal-difference series
  against the subject-averaged series to find regions participating in
  the same motion event, directly in the latent, no optical flow. The
  existing drift overlay (velocity high, jerk low) is a crude ancestor.
- Mask hysteresis. Once a region becomes attached, keep it attached for
  a short window unless evidence says it separated; prevents flicker on
  hands, props and cloth. Attachment-then-separation events (a thrown
  ball spawning its own track) are future work.
- Latent freeze without feathering. The V2V Init freeze mask currently
  feathers in pixel space and pools to fractional latent cells, but the
  feather may be unnecessary or even harmful in latent space: the VAE
  decode and overlapping receptive fields already smooth a hard cell
  edge, while fractional cells denoise as frozen/live blends that may
  sit off-manifold and mush the seam. Hard 0/1 cells keep each cell in
  one regime and let attention reconcile the boundary. Same-seed A/B at
  mask_feather 0 vs 32 vs 64 decides it. The pixel-space composite
  feather is a separate mechanism and still needed there.
- Camera-compensated jerk. Subtract the dominant global motion component
  per token before thresholding, to stop camera pans from inflating the
  dilated spans. Progression: global jerk, camera-compensated jerk,
  spatially localized overload.
- Beat-anchored recovery. Dilated time sometimes gets filled with extra
  action beats instead of slower ones (two backflips become four). That
  failure is itself evidence about the model's action-density prior and
  deserves measurement, plus a recovery mode that pins beat count.

## Status

The targeting/editor/segment nodes that make the adaptive and targeted
variants usable are alpha (2026-08-09) and under manual test; see
TESTING_ALPHA.md. The classic pipeline nodes are unchanged and
regression-tested against the previous release.
