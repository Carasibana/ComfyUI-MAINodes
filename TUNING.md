# Tuning the Motion Lab pipeline

A working guide for dialing the pipeline to a specific user's content and
taste. Written for humans and for AI assistants doing the tuning on a
user's behalf. Everything here comes from same-seed comparisons on our
test clips; expect your content to move the numbers a little.

## First: pick the graph

| priority | graph | cost (vs one baseline render) |
|---|---|---|
| best quality, audio dial available (recommended for finals) | `examples/motion_pipeline.json` | ~3.5x |
| turbo inside the pipeline (not recommended: see below) | `examples/motion_pipeline_turbo.json` | ~1.6x |
| fastest, no full baseline (scouting only: the probe init is not good enough to feed a base-model finals pass, we tried) | `examples/motion_pipeline_probe_expert.json` | ~1x or less |

The working rhythm we recommend: iterate prompts and seeds with plain
turbo generations to learn what a prompt gives you globally, then run
the keeper through the base pipeline. Do not pair the turbo LoRA with
the pipeline's regeneration pass for finals: if a clip earned the
pipeline, it earned the base model, and turbo-in-pipeline trades away
quality exactly where you decided quality matters. The faster graphs
exist for different budgets, not as the default.

The probe graph never finishes the baseline, so there is no full-speed
audio track to blend and the preview output is intentionally blurry. If
the user cares about the audio dial, use the first two.

## Second: ask the user what they actually care about

The dials trade four things against each other: sharpness in the bursts,
pose fidelity to the source, render time, and audio feel. Get a ranking
before touching anything. A user who says "the flip looks melted" wants a
different dial than one who says "it changed my character's pose."

## Symptom to dial

| user says | do this |
|---|---|
| motion still smears inside a burst | raise `d_max` toward 4 if lowered; lower `q` toward 0.70; check `bridge` is not 0 |
| brief stutter or soft frames mid-burst | `bridge: 8` (fills the dip between burst peaks) |
| poses drift from the source (head angle, hand position) | `bridge: 0` first; if not enough, lower `inject` toward 0.5 |
| output ignores the source choreography, invents moves | `inject` is too high; come down toward 0.6 |
| source artifacts leak into the output | `inject` is too low; go up toward 0.7 |
| too slow / too expensive | raise `q` toward 0.85 (tighter spans); switch to the turbo graph; then the probe graph |
| turbo output changes appearance (adds ornament, shifts style) | lower LoRA strength 0.8 toward 0.65; or raise `base_head` in the expert schedule so more structure forms on the base model |
| audio feels thin | raise `reference_mix` (needs the non-probe graphs); it is happiest near 0 or 1, mid values can double misaligned impacts |
| audio impacts feel soft | known vocoder trait; try `reference_mix: 1` for baseline foley, or accept lean |
| unprompted speech or vocal sounds appear | any voice your prompt describes without verbatim words gets gap-filled: sometimes a coherent invented line, sometimes non-language vocalizing. Script the words in the dialogue format, declare the voice indistinct by design, or cut the voice mention from the prompt |
| dialog sounds processed | check whether the speech overlaps a burst; unheld spans pass through untouched, so only speech during bursts is affected. `reference_mix: 1` restores the original line |
| probe init loses choreography | raise `probe_steps` from 6 toward 10 |
| background details change in regenerated spans (a flag recolors, props swap) | known limitation: detailed backgrounds re-roll during dilation. Try `inject` toward 0.5 (closer init tracking); simple backgrounds barely show it. A subject-only "foveated" mode that never regenerates the background is on the roadmap |
| camera pans or scrolls make the dilated spans wider than the action | known: camera motion raises the oracle's jerk floor globally. Raise `q` toward 0.85 as a stopgap; camera-motion-compensated jerk is on the roadmap |
| an action repeats or doubles in regenerated spans (two backflips become four) | the model fills dilated time with extra beats instead of slowing the existing ones; its prior for action density wins over the init. Try `inject` toward 0.5 (closer init tracking), and state exact counts positively in the prompt ("performs exactly two backflips in total"). Mirrors and reflections make it worse: the oracle reads the same motion twice and over-dilates. On the roadmap as beat-anchored recovery |
| background elements (birds, crowds, traffic) speed up during bursts | open problem, honestly. Two remedies tried and rejected in playback: post-hoc compositing (objects pop at the mask boundary) and the `freeze_threshold` latent freeze (degrades other artifacts). The freeze knob exists if your content favors that trade. The `show_drift` heatmap overlay at least shows you where the effect will occur |

## VRAM expectations (measured)

Weights: the int8 DiT is 20 GB, the text encoder 15 GB (offloads after
encoding), the video VAE 4.9 GB. Activations at 1.0 MP run about 0.1 GB
per latent token: a 5 s clip (37 tokens) needs ~5 GB, and its dilated
regeneration pass (87 tokens) ~9 GB. Roughly half that at 0.5 MP.

- 96 GB: everything stays resident, ~11.5 s/step.
- 32 GB (5090 class): the DiT fits, but the dilated pass of a 5 s clip
  lands around 30 GB total, right at the cliff, so ComfyUI starts
  offloading and steps balloon (~20 s/it reports). What helps, in order:
  start with 2 to 3 second clips (the dilated span is what hurts), drop
  the regeneration to 0.5 MP, or use the probe graph so there are far
  fewer of those expensive steps. Or switch stacks entirely: see the
  featherweight numbers below.
- 24 GB: featherweight plus 0.5 MP plus short clips territory. Untested
  by us so far.

Cost scales with the burst spans, not the clip length, so a long clip
with one short burst is far cheaper than these worst-case numbers.

### Featherweight stack, measured (ComfyUI 0.31+)

The smallest community-published variant of each piece: w4a8 DiT
(12.5 GB, Kijai/MiniMax-H3-experimental), int8_convrot video VAE
(3.2 GB), nvfp4 AWQ text encoder (15.7 GB, offloads after encoding).
Needs ComfyUI 0.31 with comfy-kitchen 0.2.28. Full de-roping pipeline
measured on one card (peaks are torch-allocated for the whole process):

| run | wall time | peak, TE resident | peak with TE offloaded |
|---|---|---|---|
| 3 s at 0.4 MP | 4.0 min | 34.4 GB | ~19 GB |
| 3 s at 0.7 MP | 6.3 min | 35.2 GB | ~20 GB |
| 5 s at 1.0 MP | 29 min | 43.8 GB | ~28 GB peak, ~20 GB sustained |

The last column is what consumer cards see: without `--gpu-only` the
text encoder leaves VRAM after encoding, which is default behavior. So
a 5 s, 1.0 MP de-rope fits a 32 GB card with a few GB of headroom (the
28 GB figure is a brief spike, under a minute total), and 3 s clips run
comfortably. Resolution barely moves the peak (0.4 to 0.7 MP added less
than 1 GB); frame count is what costs.

Two honest notes. Quality: the featherweight output is coherent and
sharp in our runs, but a given seed renders a DIFFERENT take of the
scene than the int8 stack; quantization changes the trajectory, so
compare quality, not pixels. Speed: on our card the w4a8 pipeline ran
somewhat slower per clip than int8 (29 vs low-20s minutes for the 5 s
case); on a 32 GB card that is the wrong comparison, because int8 does
not fit and offload-thrash costs far more.

### Known-good environment (this works for me)

Every number in this document was measured on this exact stack, on
Blackwell silicon, so it is a safe reference point for 5090-class cards:

```
GPU:      RTX PRO 6000 Blackwell / driver 610.43.02
Python:   3.12.13
PyTorch:  2.14.0.dev20260801+cu132 (nightly; Blackwell wants the cu13x builds)
CUDA:     13.2, cuDNN 9.24, Triton 3.8.0
Attention: sageattention (no flash-attn, no xformers)
ComfyUI:  0.31.1 with comfy-kitchen 0.2.28
```

If a 5090 runs far slower than the table above, check the torch build
first: a stable cu126/cu128 wheel on Blackwell can cost you more than
any dial in this document.

### Conditioning modes (I2VA, FL2VA, Ref2VA)

The pipeline works with all of them, unmodified, at the shipped
defaults. Tested end to end on the featherweight stack: wire your
`first_frame` / `last_frame` / reference inputs into BOTH conditioning
nodes (the baseline's and the regeneration's) and run.

- I2VA: the first-frame pin lands at t=0 on any clock, so the
  regeneration honors it exactly. Recovered frame 0 matched the
  reference in our test.
- FL2VA: the last-frame pin names a timestamp that lands mid-clip on
  the regeneration's dilated clock, and it does not matter: at inject
  0.70 the injected trajectory owns the timing and the pins act as
  pose guidance. The recovered clip hit the anchor pose at the true
  end. At much lower inject values this could in principle fight; if
  your FLF de-rope stalls mid-clip, raise inject back toward 0.70.
- Ref2VA (the ref2va checkpoint, w4a8 available): reference tokens
  carry no timeline, so they pass through dilation untouched. Subject
  identity held through the full de-rope in our test.

## Method

- Change one dial at a time, same seed, and compare in playback. Still
  frames lie about temporal quality in both directions; we have been
  burned by this repeatedly.
- The oracle heatmap (`H3 Jerk Heatmap`) shows where the pipeline will
  spend its budget. If the hot region misses the artifact the user is
  pointing at, fix `q` before touching anything else.
- Sharpness metrics rank frames usefully within one clip but do not
  decide between settings. The user's eyes decide.
- Cost scales with the burst span, not the clip length. A long clip with
  one short burst is cheap to fix.

## Defaults, and why

`q 0.75, d_max 4, ramp on, bridge 8, inject 0.70` is the playback-ratified
starting point. `inject 0.50` measured sharper with closer motion tracking
on our clips and is one preset away; some of us prefer it. The expert
schedule defaults (`total 8, inject 0.70, base_head 2`) give the turbo
tail its native 4 steps.

## Refining this guide

When a tuning session finds a symptom/dial pair this table lacks, add it.
That is the point of the file.
