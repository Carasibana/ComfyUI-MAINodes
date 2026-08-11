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
- Duplicate-motion (mirror) detection: PROBED AND NOT SOLVED by the
  co-motion statistic, 2026-08-09. Two synthetic results worth keeping.
  (1) The oracle is scale-invariant to duplication: because `q` is a
  quantile of the profile, a reflection showing the same motion at the
  same time yields a bit-identical hold map (profile correlation
  0.9999). The old "mirrors make the oracle over-dilate" line in TUNING
  was wrong and is now corrected; the real amplifier is `bridge`
  welding a temporally separated reflection burst to the body's burst
  (+26% dilated time in the probe, +18% of it from bridge alone).
  (2) Lagged correlation between per-cell jerk profiles CANNOT separate
  a mirror from two independent dancers: one body's own parts already
  correlate at ~1.0, so the confuser floor sits above the signal
  (margin -0.145 with heterogeneous limbs). Sharpening: the same
  statistic that is right for motion ATTACHMENT (you want the whole
  body, sword included, to score as one event) is structurally wrong
  for DE-DUPLICATION. A duplicate detector needs the geometric fact a
  mirror actually provides, approximate reflection symmetry of the
  spatial jerk map about some axis, not temporal correlation. Untested;
  needs a real mirror clip, since a synthetic with uniform blocks
  passes any symmetry test trivially.
- **The oracle has no "nothing is wrong here" state, and that is the
  mechanical explanation for the overzealousness complaint.** Measured
  2026-08-09 on synthetic latents. Two clips, same object, same average
  speed: one at constant velocity (true trajectory jerk exactly 0), one
  in stop/go lurches (trajectory jerk 0.347). The oracle is not blind to
  the difference, its profile peak contrast is about 2x higher on the
  jerky clip. But it still asked for **3.06x dilation on the clip whose
  jerk is exactly zero**, versus 3.47x on the jerky one. The cause is
  structural: `q` is a quantile of the profile, so the oracle always
  selects the top (1-q) of tokens no matter what the clip contains. It
  can rank, but it cannot abstain. A clip that needs no repair still
  gets a 3x budget spent on its fastest quarter.
  Second measurement, real clips: in the value domain the third
  difference is nearly the same signal as the first, corr(|d1|,|d3|) =
  0.960 and 0.977 on two of our renders. That is expected analytically,
  since a textured object passing a location makes the value there
  pulse, and a pulse has large differences of every order even at
  constant velocity. So the score is heavily contaminated by motion
  energy: it is closer to "how much is moving" than to "how abruptly
  the motion changes", and the name oversells it.
  Two consequences. (1) An **absolute** gate belongs next to the
  relative one, so a smooth clip can be told it needs nothing, which
  would also stop the pipeline inventing beats in dilated time it never
  needed. (2) The trajectory-domain measurement is the more honest
  instrument: tracking a body's centroid and differentiating THAT gave
  jerk a visibly narrower profile than velocity (top-decile share 0.29
  vs 0.22, and 0.47 vs 0.37 on a second clip) with only moderate
  correlation between them (r = 0.46 and 0.72), which is the separation
  the value-domain score fails to make. Method, so the numbers are reproducible without our scripts: build two
  synthetic latent sequences with matched mean speed, one constant-velocity
  and one lurching, run the shipped oracle on both; then for the real clips,
  track a bright-region centroid per frame and differentiate the TRAJECTORY
  rather than the latent values.
- **PARKED FOR A SECOND OPINION, 2026-08-09. Reframe: we
  built an allocator, not a meter.** The shipped oracle answers "given
  that I am going to spend a budget, where should it go", a relative
  allocation between roughly 5x slow and not at all, spreading the extra
  time as well as it can. The question it does not answer is "how much
  extra time does THIS clip actually need". Those are different
  problems and only the second one can decline to act. Degenerate case
  that shows the gap plainly: a clip that is high jerk from start to
  finish. The quantile still picks a top quarter, which is arbitrary,
  and the other three quarters go unhelped even though they need help
  just as much. Uniform dilation is the only current answer and it is a
  blunt one.
  Seed for the meter, not a decision: the project's founding sentence
  already contains an absolute unit. A token spans four frames, and the
  failure is that those four frames need four distinct poses the token
  cannot hold. So the natural absolute quantity is **demand over
  capacity**, how much pose change a span requires against how much one
  token can represent, with the hold count falling out as the ratio
  rounded up. That gives dilate-everything on a uniformly violent clip,
  dilate-nothing on a smooth one, and it is calibratable once against
  the speed at which smear actually appears rather than being fixed by
  a quantile forever.
  **Second parked observation (an impression from playback, explicitly not
  yet tested): global frame jerk, for example the background decelerating,
  seems to produce less dangerous artifacts than articulated subject
  jerk.** If that survives testing, a plausible reason is that a global
  motion is low dimensional, a whole-frame shift is one simple thing a
  token can represent cheaply, whereas an articulated body needs many
  independent degrees of freedom in the same budget. That would mean
  the quantity that matters is not jerk alone but jerk weighted by the
  complexity of whatever is jerking, and it strengthens the case for
  measuring residual jerk after global motion is subtracted rather than
  raw jerk. Do not act on this before it is measured; both halves are open.
- Mask hysteresis. Once a region becomes attached, keep it attached for
  a short window unless evidence says it separated; prevents flicker on
  hands, props and cloth. Attachment-then-separation events (a thrown
  ball spawning its own track) are future work.
- Latent freeze without feathering (now the default). Feathering may be
  unnecessary or even harmful in latent space: the VAE decode and
  overlapping receptive fields already smooth a hard cell edge, while
  fractional cells denoise as frozen/live blends that may sit
  off-manifold and mush the seam. Hard 0/1 cells keep each cell in one
  regime and let attention reconcile the boundary. mask_feather 0 now
  snaps boundary cells to 0/1; the ratifying same-seed A/B (0 vs 32 vs
  64) is on the alpha checklist. The pixel-space composite feather is a
  separate mechanism and still needed there.
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
