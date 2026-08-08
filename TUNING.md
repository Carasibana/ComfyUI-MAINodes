# Tuning the Motion Lab pipeline

A working guide for dialing the pipeline to a specific user's content and
taste. Written for humans and for AI assistants doing the tuning on a
user's behalf. Everything here comes from same-seed comparisons on our
test clips; expect your content to move the numbers a little.

## First: pick the graph

| priority | graph | cost (vs one baseline render) |
|---|---|---|
| best quality, audio dial available | `examples/motion_pipeline.json` | ~3.5x |
| good quality, fast | `examples/motion_pipeline_turbo.json` | ~1.6x |
| fastest, no full baseline | `examples/motion_pipeline_probe_expert.json` | ~1x or less |

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
| dialog sounds processed | check whether the speech overlaps a burst; unheld spans pass through untouched, so only speech during bursts is affected. `reference_mix: 1` restores the original line |
| probe init loses choreography | raise `probe_steps` from 6 toward 10 |
| background elements (birds, crowds, traffic) speed up during bursts | add `H3 Motion Composite` after recovery: subject from the regeneration, background from the baseline. Raise `grow` if the subject's edges flicker between sources |

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
