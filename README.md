# ComfyUI-MAINodes

Custom nodes for MiniMax-H3. Two groups: **Motion Lab** (a test-time fix
for fast-motion smearing) and **Contact-Sheet diffusion** (five views of
one subject from one reference image).

## Motion Lab

H3 smears bursty motion: backflips, fast sword arcs, whip-fast reversals.
The cause is structural. One latent token spans four pixel frames, and at
high motion speed those four frames need four distinct poses that a single
token can't hold. Re-denoising the affected region doesn't help, because
the missing poses were never generated in the first place.

This pipeline works around that at inference time. It re-generates the
clip as a slowed-down version of itself, seeded from the original. Frames
where motion is too fast get held (repeated) so the model has more
temporal room, the result is generated video-to-video from that retimed
init at partial denoise, and the original frame rate is recovered
afterward by dropping the held frames. The oracle that decides where to
slow down reads the clip's own latent. No extra model, no training.

![baseline vs inject 0.70 vs inject 0.50](assets/kitsune_threeway.gif)

Baseline on the left smears the aerial spin into a blob; the two
regenerated settings render it clean and keep the choreography. New scene,
default knobs, no per-clip tuning
([view](https://matlowai.github.io/ComfyUI-MAINodes/#kitsune) /
[download](https://github.com/matlowai/ComfyUI-MAINodes/raw/main/assets/kitsune_threeway_sbs.mp4)).

All demos play on one page: **https://matlowai.github.io/ComfyUI-MAINodes/**

Demo clips:
- baseline vs regenerated, same seed, real time: left smears through the
  backflip, right doesn't
  ([view](https://matlowai.github.io/ComfyUI-MAINodes/#derope) /
  [download](https://github.com/matlowai/ComfyUI-MAINodes/raw/main/assets/baseline_vs_regenerated_sbs.mp4))
- uniform vs adaptive hold maps, the bridge trade-off described below
  ([view](https://matlowai.github.io/ComfyUI-MAINodes/#adaptive) /
  [download](https://github.com/matlowai/ComfyUI-MAINodes/raw/main/assets/uniform_vs_adaptive_sbs.mp4))
- the oracle, watching: heat pools where motion runs too hot, and the
  strip lights up as the burst arrives
  ([view](https://matlowai.github.io/ComfyUI-MAINodes/#oracle) /
  [download](https://github.com/matlowai/ComfyUI-MAINodes/raw/main/assets/oracle_map.mp4))
- fast motion under a panning camera: parasol burst mid-pan
  ([view](https://matlowai.github.io/ComfyUI-MAINodes/#panrun) /
  [download](https://github.com/matlowai/ComfyUI-MAINodes/raw/main/assets/panrun_sbs.mp4))
- nine tiers, one seed: every quality/speed rung with render times in the
  header
  ([view](https://matlowai.github.io/ComfyUI-MAINodes/#ladder) /
  [download](https://github.com/matlowai/ComfyUI-MAINodes/raw/main/assets/preview_ladder_grid.mp4))

| baseline vs regenerated | the oracle, watching |
|---|---|
| ![baseline vs regenerated](assets/derope_sbs.gif) | ![the oracle, watching](assets/oracle_map.gif) |

Already good, and slightly better: uniform dilation, then the adaptive
map without and with `bridge`. Same seed all three.

![uniform vs adaptive, without and with bridge](assets/uniform_vs_adaptive.gif)

```
(baseline video) -> VAEDecode frames        (baseline latent)
        |                                        |
        v                                        v
   H3TimeSmear  <-- hold_map ------------- H3JerkOracle
        |  (integer holds)                       |
        v                                        |
    VAEEncode -> H3V2VInit -> SamplerCustomAdvanced
                                  ^
              H3InjectSchedule ---/
                                  |
                              VAEDecode -> H3ExactRecover -> original fps
```

Three ready-made graphs live in [`examples/`](examples/):
[`motion_pipeline.json`](examples/motion_pipeline.json) drags straight
onto the ComfyUI canvas;
[`motion_pipeline_api.json`](examples/motion_pipeline_api.json) is the
same graph in API format for scripted use; and
[`motion_pipeline_turbo.json`](examples/motion_pipeline_turbo.json) is the
same pipeline with the regeneration pass running on the LightX2V 4-step
turbo LoRA (Kijai's ComfyUI conversion, strength 0.8, er_sde with a beta
schedule, 3 of 4 steps after injection). Point the LoRA loader at
wherever you saved the conversion; community strength range is 0.65 to
0.8, and v0.1 of that LoRA is a preview, so judge results accordingly.

How we actually use this after trying every combination: **turbo is for
getting your prompt right, the pipeline is for the keeper, and mixing
them is a waste of time.** Iterate prompts and seeds on plain turbo
generations to learn what you will get globally, then run the winner
through the base pipeline. Putting the turbo LoRA inside the
regeneration pass saves a few minutes on a clip you have already decided
deserves the full treatment, and it costs quality on exactly that clip;
we do not recommend it. The turbo-inside graphs remain for people with
different budgets. One wrinkle that did earn its keep: starting the
turbo PREVIEW on the base model for the first couple of steps before
handing off to turbo (H3 Expert Schedule with inject 1.0) may buy
preview fidelity cheaply; we are still testing it. A unified workflow
with a preview/final toggle (H3 Mode Switch, lazy: only the chosen path
executes) is the intended end state.

A fourth graph,
[`motion_pipeline_probe_expert.json`](examples/motion_pipeline_probe_expert.json),
is the fast path: instead of finishing the baseline it runs only the
first 6 steps (H3 Probe Schedule, configurable) and reads the oracle and
the init from the early x0 estimate, then regenerates with a base-model
head and a turbo tail (H3 Expert Schedule). Cheapest of the set; no
full-speed audio track to blend, and the saved preview is intentionally
rough.

All of them generate or probe a baseline, read its oracle, regenerate,
and recover, in one queue item. The oracle's length and the regeneration
length are wired dynamically, so changing the clip duration needs no
other edits. Each node's info button documents its inputs.

### Nodes

| node | knob | default | notes |
|---|---|---|---|
| H3 Jerk Oracle | `q` | 0.75 | jerk quantile treated as "hot"; higher = tighter span, lower cost |
| | `d_max` | 4 | peak hold count; below 4, smearing starts returning in our tests |
| | `ramp` | on | smooth shoulders on the hold curve; hard steps caused visible stutter |
| | `bridge` | 8 | fill dips between peaks of the same burst (see below); 0 = off |
| | `preset` | balanced | balanced / max quality / economy; `custom` uses the knobs |
| H3 Time Smear | `dilation` | 4 | uniform hold count, used when no hold_map is wired |
| H3 Inject Schedule | `inject` | 0.70 | fraction of the denoise schedule that runs. Lower keeps more of the init (including its artifacts); higher lets the model drift from the source choreography. 0.5 to 0.8 is the useful range |
| | `preset` | 0.70 | 0.70 / 0.50 / 0.80; `custom` uses the knob |
| H3 V2V Init | `length` | 0 (auto) | wraps the encoded init as H3's joint AV latent; audio regenerates with the video |
| H3 Exact Recover | | | drops held frames per the hold map; recovery is frame selection, not resampling |
| H3 Audio Recover | `fps` | 24 | retimes the regenerated audio to the original clock with the same hold map, pitch preserved, so the recovered video keeps its own foley |
| | `reference_mix` | 0 | thickness dial: the regenerated foley is scored for the slowed take and comes back leaner (arguably more realistic); blend the baseline's denser full-speed track back in, 0 to 1. The two performances drift slightly in timing, so mid values can double misaligned impacts; the dial is happiest near its ends |
| H3 Jerk Heatmap | `alpha`, `strip_height` | 0.55, 96 | the oracle-watching overlay from the demo clip, as a node |
| H3 Probe Schedule | `probe_steps` | 6 | run only the head of the baseline; the early x0 feeds the oracle and the init. Raise it if the init loses choreography |
| H3 Expert Schedule | `base_head` | 2 | split the injected schedule: base-model head for structure, turbo tail for refinement (tail defaults to turbo's native 4 steps) |
| H3 Trajectory Bank | `every_n` | 1 | wraps a sampler and checkpoints the trajectory latent each step (~7 MB per step for a 5 s clip) |
| H3 Trajectory Load | `step` | 5 | resume a banked run from any step with its remaining schedule; swap the model, LoRA, or guider and continue without recomputing the head |
| H3 V2V Init | `freeze_threshold` | 0 (off) | background freeze, experimental and not recommended: it fixes background timing but degraded other artifacts in our playback tests. Kept as a knob for content where the trade goes the other way |
| H3 Motion Composite | | | deprecated: post-hoc compositing made moving background objects pop at the mask boundary in playback |

A tuning guide for all of this, written for humans and for AI assistants
working on a user's behalf, is in [`TUNING.md`](TUNING.md).

### bridge and inject

Both settings change the output in ways that are a preference, not a
ranking. From same-seed comparisons on our test clips:

- `bridge: 8` (default): the hold plateau covers each burst fully.
  Sharpest output, motion tracking equal to uniform dilation, about 2.9x
  frame budget. Poses can drift slightly from the baseline (a head angle
  on a landing, that kind of thing).
- `bridge: 0`: holds follow the raw oracle curve. Closest to the
  baseline's poses; a few soft frames can remain where the curve dips
  inside a burst.
- no hold_map (uniform `dilation: 4`): most conservative, highest cost.
- `inject 0.70` vs `0.50`: 0.50 measured sharper with closer motion
  tracking on our clips; 0.70 has been the safer default in playback.
  Try both on your content.

### Notes on the approach

- A reference conditions every step at full strength and will copy the
  source's artifacts. An init decays with noise: at `inject 0.70` the
  baseline's smear detail is destroyed while its coarse motion survives.
- The model's clock stays uniform. The slowdown exists only in the
  content, as a speed ramp, so there is no boundary where the DiT and the
  VAE disagree about time. (Warping the RoPE time axis directly was
  tried; it produced boundary stutter.)
- Holds are integer, so recovering the original frame rate is exact frame
  selection.

## Contact-Sheet diffusion

Five standalone image latents packed on the model's time axis, jointly
denoised, decoded independently. Use with a Turnaround LoRA from
[matlod/minimax-h3-turnaround](https://huggingface.co/matlod/minimax-h3-turnaround).
Nodes: **H3 Contact Sheet**, **H3 Contact Sheet Decode**; a scripted
example is in [`example_api_workflow.py`](example_api_workflow.py).
Previously published as ComfyUI-H3-ContactSheet; that repo remains up for
existing installs.

## Install

```
cd ComfyUI/custom_nodes
git clone https://github.com/matlowai/ComfyUI-MAINodes
```

Restart ComfyUI. Nodes appear under `latent/minimax/motion`,
`image/minimax/motion`, and `sampling/custom_sampling/schedulers`.

## License

MIT
