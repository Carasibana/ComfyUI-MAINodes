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
([full clip](assets/kitsune_threeway_sbs.mp4)).

Demo clips (in [`assets/`](assets/)):
- [baseline vs regenerated, same seed, real time](assets/baseline_vs_regenerated_sbs.mp4):
  left smears through the backflip, right doesn't
- [uniform vs adaptive hold maps](assets/uniform_vs_adaptive_sbs.mp4):
  the bridge trade-off described below
- [the oracle, watching](assets/oracle_map.mp4): heat pools where motion
  runs too hot, and the strip along the bottom lights up as the burst
  arrives

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

All of them generate a baseline, read its oracle, regenerate, and
recover, in one queue item. The oracle's length and the regeneration
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
| | `reference_mix` | 0 | thickness dial: the regenerated foley is scored for the slowed take and comes back leaner (arguably more realistic); blend the baseline's denser full-speed track back in, 0 to 1 |
| H3 Jerk Heatmap | `alpha`, `strip_height` | 0.55, 96 | the oracle-watching overlay from the demo clip, as a node |

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
