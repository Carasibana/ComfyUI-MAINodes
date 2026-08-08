# ComfyUI-MAINodes

Custom nodes for MiniMax-H3. Two groups: **Motion Lab** (a test-time fix
for fast-motion smearing) and **Contact-Sheet diffusion** (five views of
one subject from one reference image).

## Motion Lab

H3 smears bursty motion — backflips, fast sword arcs, whip-fast reversals.
The cause is structural: one latent token spans four pixel frames, and at
high motion speed those four frames need four distinct poses that a single
token can't hold. Re-denoising the affected region doesn't help, because
the missing poses were never generated in the first place.

This pipeline works around that at inference time. It re-generates the clip
as a slowed-down version of itself, seeded from the original: frames where
motion is too fast get held (repeated) so the model has more temporal room,
the result is generated video-to-video from that retimed init at partial
denoise, and the original frame rate is recovered afterward by dropping the
held frames. The oracle that decides *where* to slow down reads the clip's
own latent — no extra model, no training.

Demo clips (in [`assets/`](assets/)):
- [baseline vs regenerated, same seed, real time](assets/baseline_vs_regenerated_sbs.mp4) —
  left smears through the backflip, right doesn't
- [uniform vs adaptive hold maps](assets/uniform_vs_adaptive_sbs.mp4) —
  the bridge trade-off described below
- [oracle overlay](assets/oracle_map.mp4) — where and when the oracle sees
  excessive motion

![baseline vs regenerated](assets/derope_sbs.gif)

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

A runnable API-format graph is in
[`examples/motion_pipeline_api.json`](examples/motion_pipeline_api.json):
it generates a baseline, reads its oracle, regenerates, and recovers, in
one queue item. Each node's info button documents its inputs.

### Nodes

| node | knob | default | notes |
|---|---|---|---|
| H3 Jerk Oracle | `q` | 0.75 | jerk quantile treated as "hot"; higher = tighter span, lower cost |
| | `d_max` | 4 | peak hold count; below 4 smearing starts returning in our tests |
| | `ramp` | on | smooth shoulders on the hold curve; hard steps caused visible stutter |
| | `bridge` | 8 | fill dips between peaks of the same burst (see below); 0 = off |
| | `preset` | balanced | balanced / max quality / economy; `custom` uses the knobs |
| H3 Time Smear | `dilation` | 4 | uniform hold count, used when no hold_map is wired |
| H3 Inject Schedule | `inject` | 0.70 | fraction of the denoise schedule that runs. Lower keeps more of the init (including its artifacts); higher lets the model drift from the source choreography. 0.5–0.8 is the useful range |
| | `preset` | 0.70 | 0.70 / 0.50 / 0.80; `custom` uses the knob |
| H3 V2V Init | `length` | 0 (auto) | wraps the encoded init as H3's joint AV latent; audio regenerates with the video |
| H3 Exact Recover | | | drops held frames per the hold map; recovery is frame selection, not resampling |
| H3 Jerk Heatmap | `alpha`, `strip_height` | 0.55, 96 | diagnostic overlay of the oracle plus a per-token profile strip |

### bridge and inject

Both settings change the output in ways that are a preference, not a
ranking. From same-seed comparisons on our test clips:

- `bridge: 8` (default): the hold plateau covers each burst fully.
  Sharpest output, motion tracking equal to uniform dilation, ~2.9× frame
  budget. Poses can drift slightly from the baseline (e.g. a head angle on
  a landing).
- `bridge: 0`: holds follow the raw oracle curve. Closest to the
  baseline's poses; a few soft frames can remain where the curve dips
  inside a burst.
- no hold_map (uniform `dilation: 4`): most conservative, highest cost.
- `inject 0.70` vs `0.50`: 0.50 measured sharper with closer motion
  tracking on our clips; 0.70 has been the safer default in playback.
  Try both on your content.

### Notes on the approach

- A reference conditions every step at full strength and will copy the
  source's artifacts; an init decays with noise. At `inject 0.70` the
  baseline's smear detail is destroyed while its coarse motion survives.
- The model's clock stays uniform. The slowdown exists only in the content
  (a speed ramp), so there is no boundary where the DiT and the VAE
  disagree about time — warping the RoPE time axis directly was tried and
  produced boundary stutter.
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
