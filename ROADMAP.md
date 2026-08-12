# Roadmap: methods we are investigating

Companion to `RESEARCH_NOTES_ATOS.md` (the claim and the evidence plan),
`TUNING.md` (symptom to dial, including the rejections) and
`TESTING_ALPHA.md` (what the alpha nodes still need from human hands).

Everything below is a direction, not a promise. Items are grouped by the
problem they attack, and each says what would count as success and what the
cheapest first measurement is. Where we have already measured something that
contradicts an earlier claim of ours, it is written down as such.

House rules that constrain every item here: motion is judged in playback
because stills lie in both directions; a measurement earns its compute only
if it beats the cheap baseline it replaces; and anything shipped as a default
has to survive a same-seed A/B.

---

## 1. The oracle: from allocator to meter

**The problem, measured and published in the research notes:** the jerk
oracle ranks but cannot abstain. Because the threshold is a quantile of the
profile, a clip with exactly zero trajectory jerk still received 3.06x
dilation. And in the value domain the third difference is nearly the same
signal as the first (corr 0.960 and 0.977 on real clips), so the score is
contaminated by motion energy. This is the single most important open
problem in the method.

Directions, roughly in order of expected value per hour:

- **Paired-speed reachability.** Generate the same choreography at two
  content speeds and diff the latents. Tokens whose state actually changes
  between the two are the tokens that motion speed reaches; tokens that look
  hot in both are hot for some other reason, which is exactly the
  textured-object-passing-a-location confound. Borrowed from the practice of
  probing a system with paired semantic perturbations rather than noise.
  Success: a per-token relevance score that separates the two synthetic
  clips (zero jerk versus lurching) where the quantile does not. Cost: two
  renders plus offline analysis of saved latents.
- **Demand over capacity as an absolute unit.** A token spans four frames;
  the failure is that those four frames need more distinct poses than one
  token can hold. So the natural absolute quantity is required pose change
  divided by representable pose change, with the hold count falling out as
  that ratio rounded up. Calibrated once against the speed at which smear
  actually appears, instead of fixed by a quantile forever. Success: the
  meter says "dilate nothing" on a smooth clip and "dilate everything" on a
  uniformly violent one, both of which the quantile gets wrong today.
- **Trajectory-domain measurement.** Showing promise on real clips
  (top-decile share 0.29 versus 0.22 for velocity, with only moderate
  correlation between them). Track a region centroid and differentiate the
  trajectory rather than the latent values. Now selectable on the oracle as
  `profile_mode`. **Caveat measured while shipping it:** on a smooth
  synthetic blob the centroid path is nearly noise-free, so its peak-to-mean
  contrast is not comparable to the value domain's there (1.21x versus 1.88x
  separation between a constant-velocity and a lurching toy). Treat it as an
  ablation option, not as an established improvement, until it is measured on
  real textured content.
- **Knee-finding instead of a fixed quantile.** Run the cheap early-x0 probe
  at several dilation factors and take the knee of the curve per clip.
  Unlike a quantile, a knee can land at 1.0x, which is the abstention we
  currently cannot express.
- **Absolute calibration by temporal capacity profiling.** Feed the VAE
  synthetic controlled-speed content (moving gratings, rotating props, speed
  sweeps) and measure reconstruction fidelity against speed to get the
  model's temporal transfer curve. Useful beyond this repo: it would be a
  reusable way to compare base models on temporal capacity.

**Ablation owed regardless:** delta order 1 versus 2 versus 3, phase
normalization on and off, against external baselines (optical flow magnitude
and flow acceleration on decoded frames). If the third difference wins or
ties while being free, that is the result. If it loses, the honest move is to
say so and rename the thing.

## 2. Detection quality

- **Camera-compensated jerk.** Subtract the dominant global motion component
  per token before thresholding, to stop pans inflating the dilated spans.
  Progression: global jerk, then camera-compensated, then spatially
  localized overload.
- **Stabilize, repair, unstabilize.** The alternative to teaching the oracle
  to ignore pans: remove the pan first with a coarse global warp, run the
  pipeline where subject motion is the only motion, then invert the warp.
  A classic VFX move that turns the hardest confound into preprocessing.
  Worth comparing directly against camera compensation.
- **Cross-modal detection.** The model co-generates audio, so audio onset
  density is a second, independent account of where the action beats are.
  Where video-hot and audio-quiet disagree is a good candidate signature for
  camera-pan inflation. This also gives the oracle a **free baseline it must
  beat**: a latent detector that cannot outperform onset detection on
  co-generated audio has not earned its complexity.
- **Mirror and reflection double counting.** Published correction in TUNING:
  a synchronized reflection leaves the hold map bit-identical, and the real
  amplifier is `bridge` welding a temporally separated reflection to the
  body's burst. A real detector needs approximate reflection symmetry of the
  spatial jerk map, not temporal correlation, which we probed and found
  cannot work (one body's own limbs correlate at ~1.0, so the confuser floor
  sits above the signal). Untested on real mirror footage.

## 3. Allocation and cost

- **Fixed-budget bidirectional retiming.** Today the method BUYS temporal
  capacity for bursts. The general object is a monotone retiming function
  whose two signs are dilation and decimation: compress the quiet spans and
  spend exactly what you saved on the bursts, for a constant token budget.
  "Repair at zero marginal cost" is a much stronger claim than "3.5x for
  quality." Open risk: decimated frames have no generated source, so
  recovery there is nearest-held-frame and may judder in quiet spans.
  Playback decides.
- **Honest cost accounting, already shipped.** Per-step time is superlinear
  in token count (measured exponent 1.75 here, ~1.64 in a field report,
  `COST_EXP 1.7` in the code). Reports now state time multipliers rather than
  frame ratios. The practical consequence is counter-intuitive and worth
  repeating: many small windows can be cheaper than one large one, so
  pass-count is the wrong unit for comparing any of these approaches.
- **Spatial crop regeneration.** Time targeting and segment crop reduce
  FLOPs; spatial masks do not, they only blend. A spatial crop analog of
  segment crop would make "foveated" literal. Risks are context loss and
  resolution mismatch at the seam.

## 4. Beat duplication and anchoring

The model fills dilated time with extra beats instead of slowing the existing
ones (two backflips become four). This is the most visible remaining artifact
and it has a clear line of attack.

- **Sparse pinned time.** We already have the choreography: the baseline.
  Sampling sparse pose anchors from it and pinning them at their retimed
  positions gives the regeneration a ladder to climb instead of room to
  improvise. Success: beat count stops scaling with dilation factor.
- **Dependency, and related work worth reading.** Stock ComfyUI restricts H3
  keyframe anchors to first and last (`comfy/ldm/minimax/model.py` raises on
  anything else). Two community packs already work around this by handing
  stock a legal index and rewriting only the temporal column after the packed
  layout is built:
  [ComfyUI-H3-Motion-Context-MultiRef](https://github.com/seitanism/ComfyUI-H3-Motion-Context-MultiRef),
  whose `patch_layout.py` also compensates for reference blocks advancing the
  packing cursor, and
  [ComfyUI-H3-Multishot](https://github.com/jlucasmcrell/ComfyUI-H3-Multishot),
  which additionally exposes a condition-strength node. Both are worth a look
  if you are working this area. Upstream
  [PR #15439](https://github.com/Comfy-Org/ComfyUI/pull/15439) would make
  arbitrary guides native, which is the outcome we would prefer.
- **Two upstream bugs we found while verifying that, worth reporting.**
  (1) Keyframe condition rows are positioned relative to `text_len`, but
  reference blocks advance the packing cursor and the target is placed at the
  final cursor, so with any reference present a first/last keyframe sits at
  the wrong temporal coordinate by exactly the reference advance. (2) The
  refs branch in `model_base.py` unconditionally overwrites the keyframe
  `cond_video_latents` list, so the two features cannot currently coexist and
  most likely error on a shape mismatch before the mis-timing is visible.
  Anyone combining Ref2VA references with keyframes should be aware.
- **Audio onsets as beat anchors.** Align the retimed audio's impacts to the
  baseline's impact times and let video recovery inherit those anchor points.
  Attacks the same problem from the other modality.

## 5. Byproducts worth shipping

- **The slow-motion insert reel.** Exact recovery discards most of what the
  expensive pass generated, but those are genuine intermediate poses at up to
  4x temporal density, with pitch-preserved slowed audio available from the
  same machinery. Exporting each repaired burst as a slow-motion insert is
  nearly free and is production-useful on its own.
- **Quantization agreement as an artifact pre-filter.** Running the same seed
  through two quantizations of the same weights is a controlled perturbation:
  content that survives it is something the model committed to, content that
  moves is underdetermined. If that holds, it is an automatic way to
  pre-sort renders before a human looks, which is the real bottleneck in
  any comparison-heavy workflow. Speculative and cheap to test.

## 6. Validation we owe

- **VAE-only roundtrip probe.** Take real fast-motion footage, encode and
  decode at native speed versus slowed variants, and measure reconstruction
  error against local motion magnitude. This tests "temporal compression is
  the bottleneck" with no sampler and no DiT in the loop. Until it runs, the
  causality language in the README stays soft.
- **Cross-model transfer.** The largest external-validity risk is that
  everything here is fitted to one model's packing geometry. Porting the
  oracle, the smear and the recovery to a second temporally compressed video
  model is the experiment that upgrades this from a trick to a method, even
  as a partial result.
- **Blinded playback A/B.** Repair score and preservation score reported
  separately, because a method that removes the smear by removing the motion
  is not a fix.

## 7. An adjacent approach we are evaluating, not pursuing yet

Rather than dilating time so the compressed representation can hold the
motion, one could refuse the compression: generate every frame as its own
standalone image latent by sliding the model's native five-slot stencil
across world time and anchoring each pass on the previous pass's frames.
Four passes shifted by one frame cover 17 contiguous frames at full rate.

We think this is a genuinely interesting sibling to ATOS rather than a rival,
and the two would compose (use the cheap detector to decide where the
expensive representation is worth it). Two things to know before anyone
sinks time into it: priced in passes it sounds absurd, but priced in the
measured superlinear cost model it lands much closer to adaptive dilation
than the pass count suggests; and its likely failure mode is per-frame decode
incoherence, since decoding each frame independently discards the temporal
smoothing that overlapping receptive fields normally provide.

## 8. Explicitly not doing, with reasons

Kept here so nobody re-derives them:

- **RoPE time warping.** Warping per-token time spans was implemented and
  rejected: it produced boundary stutter, and the root cause was traced to
  chunked VAE decode on rate-warped content. Content-level slowdown on a
  uniform model clock is the approach that survived.
- **Automatic spatial masks.** Rejected in playback. Automatic
  oracle-heat-driven compositing degraded other artifacts even where it
  fixed background timing. Hand-authored masks are a different question and
  are still open.
- **Post-hoc frame interpolation.** It is a baseline to compare against, not
  a component. Interpolating a smeared frame cannot recover a pose the
  generator never created.
- **Feathering latent freeze masks.** Now defaulting to hard cells, on the
  argument that fractional edge cells denoise as half-frozen blends while the
  decode smooths a hard cell edge for free. Ratification A/B is on the alpha
  checklist and the default reverts if the ramp wins.

---

## How much work each of these is

Rough sizing against this codebase, for anyone deciding where to jump in.
The architecture helps more than it looks: `_jerk_profile` is a dozen lines
of numpy and `_compile_hold_map` is a shared seam, so **any new detector is a
drop-in** and most detection work is hours rather than days. The expensive
items are the ones that change the time REPRESENTATION or reach into ComfyUI
internals.

**Already shipped on this branch (alpha, defaults unchanged):** the
abstention gate (`abstain_below`) and alternative detectors (`profile_mode`:
value |d3|, value |d1| baseline, trajectory centroid) on H3 Jerk Oracle. The
default path is bit-identical to what shipped before.

| item | size | why |
|---|---|---|
| delta-order ablation | done | now a dropdown on the oracle |
| absolute abstention gate | done | contrast floor, off by default |
| camera-compensated profile | hours | another `_jerk_profile` mode, same seam |
| audio-onset profile as the cheap baseline | ~half day | solved DSP, emits the same profile shape |
| slow-motion insert export | ~half day | the recover node already selects frames; emit the discarded ones |
| paired-speed reachability | ~1 day | two renders plus offline latent diff, no new node |
| knee-finding over dilation factors | ~1 day | orchestration around the existing probe schedule |
| quantization agreement pre-filter | ~1 day | two renders per clip plus a divergence map |
| mirror symmetry detector | ~1 day | needs real mirror footage; synthetics pass trivially |
| spatial crop regeneration | ~2 days | mirrors the existing time crop/splice pair; seam risk |
| fixed-budget bidirectional retiming | ~2 days | holds below 1 change the hold-map representation AND recovery semantics |
| sparse pinned time / anchor ladder | ~2-3 days | needs the packed-layout temporal patch or upstream #15439, plus maintenance risk against upstream churn |
| stabilize, repair, unstabilize | ~2-3 days | a whole preprocessing stage with warp and inverse warp |
| VAE roundtrip probe | ~1 day | a script, not a node |
| temporal capacity profiling | ~2-3 days | synthetic content generation plus a sweep |
| cross-model transfer | large | a port, not a feature |

## What would actually help

If you are running this pack, the most useful things you can send back are
same-seed before and after pairs on content that breaks it, a note on which
dial you reached for, and your seconds-per-step so cost estimates can be
calibrated across hardware. Failure cases on content unlike ours are worth
more to us than successes on content like ours.
