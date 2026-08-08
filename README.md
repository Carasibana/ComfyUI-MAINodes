# ComfyUI-MAINodes

MatlowAI's MiniMax-H3 node collection — research nodes that graduated to
production, in one pack. Public research: findings, measurements, and the
occasional retraction, all in the open.

Two families:

## 1. Motion Lab — de-roping fast motion at test time

H3 ropes/smears bursty motion (backflips, whip-fast sword arcs): one latent
token carries four pixel frames and cannot hold four distinct sharp poses.
This pipeline regenerates the clip as a *time-dilated performance* seeded
from your own baseline — the model renders the same choreography with more
frames of temporal capacity where the jerk oracle says it needs them, then
exact frame selection recovers real time. No retraining, no reference
plumbing, one joint generation on one timeline.

| baseline vs regenerated (same seed, world clock) | the oracle, watching |
|---|---|
| ![de-rope side by side](assets/derope_sbs.gif) | ![oracle map](assets/oracle_map.gif) |

```
(baseline video) -> VAEDecode frames        (baseline latent)
        |                                        |
        v                                        v
   H3TimeSmear  <-- hold_map ------------- H3JerkOracle
        |  (integer holds, C1 ramps,             |
        |   valley bridging)                     | (also: segments,
        v                                        |  window, profile)
    VAEEncode -> H3V2VInit -> SamplerCustomAdvanced
                                  ^
              H3InjectSchedule ---/   (inject 0.70)
                                  |
                              VAEDecode -> H3ExactRecover -> 24fps real time
                                              (hold_map)
```

A ready-to-run API-format graph is in
[`examples/motion_pipeline_api.json`](examples/motion_pipeline_api.json)
(generates a baseline, reads its oracle, regenerates, recovers — POST it to
`/prompt` or rebuild the wiring in the UI; every node's ⓘ info button
documents its knobs and ranges).

### Nodes & knobs (defaults = measured-best on our benchmarks)

| node | knob | default | what it does |
|---|---|---|---|
| **H3 Jerk Oracle** | `preset` | balanced | balanced / max-quality (wide plateau) / economy — `custom` frees the knobs |
| | `q` | 0.75 | jerk quantile that counts as "hot"; higher = tighter span, lower cost |
| | `d_max` | 4 | peak hold count on the hottest tokens |
| | `ramp` | on | C1 ramp shoulders — hard steps jitter |
| | `bridge` | 8 | fill inter-peak valleys at d_max (see *two flavors* below); 0 = off |
| **H3 Time Smear** | `dilation` | 4 | uniform hold count when no hold_map is wired |
| **H3 Inject Schedule** | `preset` | balanced 0.70 | 0.70 balanced / 0.50 faithful-detail / 0.80 loose — `custom` frees the knob |
| | `inject` | **0.70** | the big one: how deep the v2v injection starts. Lower inherits baseline artifacts; higher invents choreography |
| **H3 Jerk Heatmap** | `alpha` | 0.55 | oracle overlay opacity |
| | `strip_height` | 96 | jerk-profile bar strip with playhead (0 = off) |

**H3 Exact Recover** inverts the smear by frame selection (never
resampling) — integer holds mean 24fps recovery is lossless by construction.

### Two flavors, both kept on purpose

Measured on the same seed, judged in playback:

- **`bridge` on (default)** — full plateau across each burst. Sharpest
  results, choreography tracking statistically equal to uniform 4×
  dilation, at ~2.9× budget. Trade-off: poses can drift subtly from the
  baseline (a head turn on a landing, that kind of thing).
- **`bridge: 0`** — the hold curve follows the raw oracle. Poses stay
  closest to the baseline; a few soft frames can survive where the curve
  dips inside a burst.
- **No hold_map at all** (uniform `dilation: 4`) — the zero-artifact
  reference point; highest cost, flattest pacing.

It's a dial, not a doctrine. Start with the default, and if a specific pose
matters more than sharpness, turn `bridge` off.

### Why this shape (the 60-second version)

- Roping is an information deficit, not a rendering bug — re-denoising the
  same tokens can't recover poses that were never generated. Capacity has
  to come from somewhere: here, from more frames per world-second.
- A *reference* rides every step at full fidelity and copies artifacts in;
  an *init* decays with noise. Inject at 0.70 and the baseline's smear
  detail is destroyed while its coarse choreography survives.
- The model's own clock stays uniform — the nonuniform timeline lives in
  the content (a speed-ramp, which video models render natively), so there
  are no rate boundaries where DiT and VAE can disagree.

## 2. Contact-Sheet diffusion (five views from one reference)

Five standalone image latents packed on the model's time axis, jointly
denoised, independently decoded. Pair with a Turnaround LoRA from
[matlod/minimax-h3-turnaround](https://huggingface.co/matlod/minimax-h3-turnaround).
Nodes: **H3 Contact Sheet**, **H3 Contact Sheet Decode**; a scripted example
lives in [`example_api_workflow.py`](example_api_workflow.py).
(Previously published as ComfyUI-H3-ContactSheet — that repo stays up for
existing installs; this is the consolidated home going forward.)

## Install

```
cd ComfyUI/custom_nodes
git clone https://github.com/matlowai/ComfyUI-MAINodes
```

Restart ComfyUI. Nodes appear under `latent/minimax/motion`,
`image/minimax/motion`, `sampling/custom_sampling/schedulers`, and the
contact-sheet pair under their existing categories.

## License

MIT. Research notes behind the defaults are being
written up; numbers in the tooltips come from measured A/Bs, not vibes.
