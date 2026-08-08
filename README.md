# ComfyUI-MatlowNodes

MatlowAI's MiniMax-H3 node collection — research nodes that graduated to
production, in one pack. Public research: findings, measurements, and the
occasional retraction, all in the open.

Two families:

## 1. Contact-Sheet diffusion (five views from one reference)

Five standalone image latents packed on the model's time axis, jointly
denoised, independently decoded. Pair with a Turnaround LoRA from
[matlod/minimax-h3-turnaround](https://huggingface.co/matlod/minimax-h3-turnaround).
Nodes: **H3 Contact Sheet**, **H3 Contact Sheet Decode**.
(Previously published as ComfyUI-H3-ContactSheet — that repo stays up for
existing installs; this is the consolidated home going forward.)

## 2. Motion Lab — de-roping fast motion at test time

H3 ropes/smears bursty motion (backflips, whip-fast sword arcs): one latent
token carries four pixel frames and cannot hold four distinct sharp poses.
This pipeline regenerates the clip as a *time-dilated performance* seeded
from your own baseline — the model renders the same choreography with more
frames of temporal capacity where the jerk oracle says it needs them, then
exact frame selection recovers real time. No retraining, no reference
plumbing, one joint generation on one timeline.

```
(baseline video) -> VAEDecode frames        (baseline latent)
        |                                        |
        v                                        v
   H3TimeSmear  <-- hold_map ------------- H3JerkOracle
        |  (integer holds, C1 ramps)             |
        v                                        | (also: segments,
    VAEEncode -> H3V2VInit -> SamplerCustomAdvanced   window, profile)
                                  ^
              H3InjectSchedule ---/   (inject 0.70)
                                  |
                              VAEDecode -> H3ExactRecover -> 24fps real time
                                              (hold_map)
```

### Nodes & knobs (defaults = measured-best on our benchmarks)

| node | knob | default | what it does |
|---|---|---|---|
| **H3 Jerk Oracle** | `q` | 0.75 | jerk quantile that counts as "hot"; higher = tighter span |
| | `d_max` | 4 | peak hold count on the hottest tokens |
| | `ramp` | on | C1 ramp shoulders (1,2,…,d_max,…,2,1) — hard steps jitter |
| **H3 Time Smear** | `dilation` | 4 | uniform hold count when no hold_map is wired |
| **H3 Inject Schedule** | `inject` | **0.70** | the big one: how deep the v2v injection starts. Lower inherits artifacts from the baseline; higher drifts toward free generation (invented choreography) |
| | `total_steps` | 25 | schedule the fraction is taken from |
| **H3 Jerk Heatmap** | `alpha` | 0.55 | oracle overlay opacity |
| | `strip_height` | 96 | jerk-profile bar strip with playhead (0 = off) |

**H3 Exact Recover** inverts the smear by frame selection (never
resampling) — integer holds mean 24fps recovery is lossless by
construction.

**H3 Jerk Heatmap** is the show-your-work tile: heat pools where the
oracle sees trouble, the strip shows when. Wire the baseline's latent +
decoded frames into it and you can *watch* the oracle think.

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

## Install

```
cd ComfyUI/custom_nodes
git clone https://github.com/matlowai/ComfyUI-MatlowNodes
```

Restart ComfyUI. Nodes appear under `latent/minimax/motion`,
`image/minimax/motion`, `sampling/custom_sampling/schedulers`, and the
contact-sheet pair under their existing categories.
