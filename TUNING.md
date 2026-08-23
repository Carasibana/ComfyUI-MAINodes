# Tuning the Motion Lab pipeline

A working guide for dialing the pipeline to a specific user's content and
taste. Written for humans and for AI assistants doing the tuning on a
user's behalf. Everything here comes from same-seed comparisons on our
test clips; expect your content to move the numbers a little.

## First: pick the graph

The start-here index with diagrams is `examples/README.md`; the hardware
rows are `HARDWARE.md`. The short table:

| priority | graph | cost (vs one baseline render) |
|---|---|---|
| the normal starting point (since 2026-08-19): 12-step base pass 1, turbo 6-step pass 2, audio seeded | `examples/motion_pipeline_ref2va_audioinit.json` | ~1.3x; 192 frames at 1 MP in ~12 min on our card |
| best quality, base model throughout, audio dial available | `examples/motion_pipeline.json` | ~3.5x; the same clip at 25 steps both passes is ~3x the row above |
| a 16 to 24 GB card | `examples/motion_pipeline_lowvram.json` | as the first row; see `LOWVRAM.md` for the fenced measurements |
| prompt and seed scouting, not a final | `examples/motion_pipeline_fast_iterate.json` | ~1x or less (0.2 -> 0.4 MP, ~95 s) |
| user knows where the problem is (two-pass: review the oraclemap, type ranges, requeue) | `examples/motion_pipeline_targeted.json` | baseline + regen cost of YOUR spans only |
| GUI editing on the node (blocks, painting, automation) | `examples/motion_pipeline_editor.json` | as targeted |
| maximum speed for one burst in a longer clip (segment crop + splice) | `examples/motion_pipeline_editor_segment.json` | baseline + regen of window+handles only; the crop report states the ratio |
| cheap pass 1, de-rope at the delivery size | `examples/motion_pipeline_upscale_derope.json` | 89% of native detail in 83% of the time (0.4 -> 1.5 MP) |
| de-roping footage that already exists (no baseline render) | `examples/archive/motion_pipeline_v2v_source.json` [alpha] | regen only, ~2.5x one baseline-equivalent render; there is no baseline pass |
| the dilated pass does not fit your card (OOM, or steps balloon while weights stream) | `examples/motion_pipeline_rolling_window.json` [alpha] | peak memory scales with the biggest WINDOW, not the clip. At the budget that forces a split: ~1.4x the generated frames, and on a card at the offload cliff wall time comes back to parity because the DiT stays resident instead of streaming. Measured on a fenced 32 GB budget: 30.7 GB peak + layer streaming one-pass vs 24.6 GB resident windowed, 460 s vs 440 s |
| earlier recipes (turbo-in-the-old-graph, probe + expert, featherweight, i50, split LoRA) | `examples/archive/` | kept so old write-ups resolve; do not start from these |

On turbo, the rule moved on 2026-08-19 and it is worth stating exactly.
Pass 1 decides the choreography and wants the base model at 12 steps or
more. Pass 2 repairs bursts from a known init and is where the turbo LoRA
belongs, **provided the whole pass-2 recipe comes with it**: `gradient_estimation`
and `linear_quadratic` on pass 1, a 6-step `beta` schedule on pass 2, the
audio seeded through `H3 Audio Smear`. The turbo LoRA dropped into the
25-step `res_multistep` graph with nothing else changed renders jerky and
pixelated (measured 2026-08-22, seven arms, all scrapped). That is what the
old "no turbo in the pipeline" rule was actually measuring.

The working rhythm: iterate prompts and seeds on the fast-iterate graph to
learn what a prompt gives you globally, then run the keeper through the
starting-point graph. The base model on pass 2 (the 25-step graph) is
better quality still, at three times the wall; reach for it when a final
has earned the time.

## Second: ask the user what they actually care about

The dials trade four things against each other: sharpness in the bursts,
pose fidelity to the source, render time, and audio feel. Get a ranking
before touching anything. A user who says "the flip looks melted" wants a
different dial than one who says "it changed my character's pose."

## Symptom to dial

| user says | do this |
|---|---|
| the reference image stops influencing pass 2 (or an upscale pass) after a graph change | before touching any reference dial, put `BasicScheduler` / stock sigmas back. One field report (2026-08-23) lost all reference influence to a third-party scheduler node and got it back with `BasicScheduler`; not a same-sigma comparison, so it is a diagnostic first step, not a theorem about that scheduler. Only then: reference strength, prompt roles, `ref_image_size` |
| dialogue or audio sounds wrong: mangled words, tags read aloud, audio that copies the audio reference instead of borrowing its voice | check the core first: `H3 Capability Probe` prints `tokenizer_special_tokens`; before ComfyUI PR #15808 (2026-08-22) `<d>` and six other H3 tags tokenized as two tokens each, and every dialogue reading on such a core is suspect. Update, re-render, then tune steps, turbo and audio refs |
| motion still smears inside a burst | raise `d_max` toward 4 if lowered; lower `q` toward 0.70; check `bridge` is not 0 |
| brief stutter or soft frames mid-burst | `bridge: 8` (fills the dip between burst peaks) |
| poses drift from the source (head angle, hand position) | `bridge: 0` first; if not enough, lower `inject` toward 0.5 |
| output ignores the source choreography, invents moves | `inject` is too high; come down toward 0.6 |
| source artifacts leak into the output | `inject` is too low; go up toward 0.7 |
| too slow / too expensive | raise `q` toward 0.85 (tighter spans); switch to the turbo graph; then the probe graph |
| coarse mosaic / tiling in a turbo pass 2 when pass 1 was ALSO turbo (the seed-hunt iterate-then-finish flow) | a turbo init only tolerates a SHALLOW turbo pass 2. Judged in playback, one fight clip, fl2va, 0.7 to 1.5 MP, rank-21 at 1.0: 1 step at inject 0.20 clean and good; 3 steps at 0.50 and 6 steps at 0.50 both out of quality bounds. A later playback check on a second content class found 1 step at 0.30 already visibly griddy, so 0.20 is the measured ceiling, not a conservative pick, and 0.30 to 0.50 is progressive tiling onset. The same 3-at-0.50 recipe is clean over a bare 12-step pass 1, so the init is the variable, not the recipe. For keepers from turbo seed picks: stay at 1 step / 0.20, or hand pass 2 to the base model. Do NOT stack a second shallow tap through pixel space: each extra tap pays a VAE encode/decode roundtrip that costs more than a 0.20 solve returns (judged: 2 taps a step below 1) | split the pass with the rolling-window graph: `H3 Window Plan` + `H3 Window Collect`. Set `max_dilated_frames` below the dilated count your card last survived, read the plan report before rendering, queue one window per item. Leave `coverage` on `full clip` on upscale graphs or the calm spans stay at baseline resolution |
| turbo output changes appearance (adds ornament, shifts style) | lower LoRA strength 0.8 toward 0.65; or raise `base_head` in the expert schedule so more structure forms on the base model |
| audio feels thin | raise `reference_mix` (needs the non-probe graphs); it is happiest near 0 or 1, mid values can double misaligned impacts |
| audio impacts feel soft | known vocoder trait; try `reference_mix: 1` for baseline foley, or accept lean |
| unprompted speech or vocal sounds appear | any voice your prompt describes without verbatim words gets gap-filled: sometimes a coherent invented line, sometimes non-language vocalizing. Script the words in the dialogue format, declare the voice indistinct by design, or cut the voice mention from the prompt |
| dialog sounds processed | check whether the speech overlaps a burst; unheld spans pass through untouched, so only speech during bursts is affected. `reference_mix: 1` restores the original line |
| probe init loses choreography | raise `probe_steps` from 6 toward 10 |
| background details change in regenerated spans (a flag recolors, props swap) | known limitation: detailed backgrounds re-roll during dilation. Try `inject` toward 0.5 (closer init tracking); simple backgrounds barely show it. A subject-only "foveated" mode that never regenerates the background is on the roadmap |
| camera pans or scrolls make the dilated spans wider than the action | known: camera motion raises the oracle's jerk floor globally. Raise `q` toward 0.85 as a stopgap, or gate the oracle through `H3 Manual Hold Map` with the ranges where the real action is; camera-motion-compensated jerk is on the roadmap |
| the oracle dilates spans the user does not care about / the pass is too expensive for one burst | `H3 Manual Hold Map` in gate mode: wire the oracle's hold_map, type the ranges that matter, everything else recovers to hold 1. The report output shows the effective length before the expensive pass runs |
| an action repeats or doubles in regenerated spans (two backflips become four) | the model fills dilated time with extra beats instead of slowing the existing ones; its prior for action density wins over the init. Try `inject` toward 0.5 (closer init tracking), and state exact counts positively in the prompt ("performs exactly two backflips in total"). On the roadmap as beat-anchored recovery |
| the clip contains a mirror or a big reflection and the beats double | do NOT blame the oracle first. Synthetic probe, 2026-08-09: a reflection showing the SAME motion at the SAME time leaves the oracle's output bit-identical (its threshold is a quantile of the profile, so duplicating the motion scales every token equally and selects exactly the same ones; profile shape correlation 0.9999). What does bite is a reflection prominent at a DIFFERENT time than the body, which a camera tilt produces (mirror early, subject late): that leaves two separated hot clusters and `bridge` welds them into one long plateau, +26% dilated time in the probe, of which `bridge` alone is +18%. So on mirror content try `bridge: 0` first, then `inject` toward 0.5, then explicit counts in the prompt. Everything left over after that is the model's action-density prior, not the oracle |
| background elements (birds, crowds, traffic) speed up during bursts | the AUTOMATIC remedies stay rejected (oracle-mask compositing popped at the boundary, blanket freeze degraded other artifacts). The manual path is new and promising: draw the keep-baseline region yourself (`show_drift` on the heatmap shows what to lasso in blue), wire it into `H3 Motion Composite` `mask` with `invert_mask` on, feather 48-64 smoothstep, and put the seam on a real edge (horizon, rooftop) where nothing moves. Use the `H3 V2V Init` `mask` freeze instead when background and subject share lighting or contact. First internal A/B (demo clip, invented flock + pagoda both removed, static seam) awaits playback ratification |

## VRAM expectations (measured)

An honesty note on the peak numbers in this section: they were read
from `nvidia-smi` on a large card, where the reading includes whatever
the caching allocator felt like keeping, so treat them as upper bounds
rather than requirements (we have since measured a smaller model
peaking HIGHER than a bigger one this way). The portable units are
latent token count for activations, and the `loaded partially` /
`lowvram patches` lines in the ComfyUI log for whether your card is
actually streaming weights. When the dilated pass is what does not
fit, the rolling-window graph caps the token count per pass: see the
table above.

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
text encoder leaves VRAM after encoding, which is default behavior.
The first column is the one **H3 Conditioning Bank** attacks: it banks
the encoded conditioning to disk and takes its `conditioning` input
lazily, so a queue item that hits the bank never executes the encode
node and the text encoder is not loaded at all. Wire it on any flow
that re-runs one prompt: rolling windows, seed hunts, extension chains.
Requeueing an unchanged graph does not need it (ComfyUI's node cache
already serves the conditioning); queueing a second workflow in
between, restarting, or editing anything upstream of the encode does,
and that is the normal shape of a session. **H3 Latent Bank** does the
same for a sampled pass: on the rolling-window graphs the pass-1
baseline is lost to the same evictions and re-renders in full before
window item 2 can start, so bank its LATENT (4.8 MB for a 107-frame
480x832 clip; the decoded frames would be 513 MB) and wire the noise
seed into the node's `seed` input so a new seed misses instead of
serving the old take. So
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

## Chained clips and the audio clock

H3's video runs at 24 fps but its audio latent runs a 40 Hz clock, and
the two only agree on durations that are multiples of 1/8 s. Of the
legal `17k+5` frame counts, only k = 2, 5, 8, 11 ... (39, 90, 141, 192
frames) land exactly; every other length ships an audio stream up to
12.5 ms longer or shorter than its video (124 frames: audio 8.3 ms
long; 107: 8.3 ms short). One clip: harmless. Concatenate segments for
a long continuation and the error compounds into audible A/V drift --
we have watched a 10-minute extension from another tool visibly
separate by the end.

What to do about it:
- for anything destined for chaining, prefer the aligned lengths
  (90 / 141 / 192 frames);
- when assembling segments, trim or pad each segment's audio to exactly
  `frames / 24` seconds before concatenation, so no segment exports its
  rounding error to the next;
- inside this pack you are already covered: H3 Audio Recover, H3
  Segment Splice, and H3 Window Collect place and size audio on the
  absolute world clock (sample-exact per segment), so seams do not
  accumulate error.

`tests/test_audio_recover.py` is the regression guard for all of this:
run it bare for the synthetic sample-exact checks (no GPU), or hand it
a `baseline.mp4 recovered.mp4` pair from any `reference_mix: 1` render
and it verifies the recovered audio really is the baseline track --
duration within one AAC frame, zero lag, full correlation. It caught a
pre-fix render 64 ms short on its first outing.

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

## Manual targeting (time ranges and spatial masks)

When the user can point at the problem, let them. Two tools, one per
axis:

- Time: `H3 Manual Hold Map` converts typed ranges (`36-60,
  1.5s-2.4s:3`; frames or seconds, ends inclusive) into an
  oracle-format hold map. Gate mode (oracle hold_map wired in) is the
  default recommendation: the oracle decides hold strength, the user
  decides where it is allowed to spend. Ranges snap outward to the
  token grid, so trust the segments output over the typed numbers, and
  show the user the report output: it states world length vs effective
  regeneration length, and estimates minutes when given a measured
  s/step. Cost scales with held spans, so this is also the speed
  answer for long clips with one burst.
- Space: `H3 Motion Composite` `mask` (pixel space, full feather
  control: size, linear/smoothstep/gaussian profile, in/out/centered
  direction) pastes baseline timing back outside the mask after
  recovery. `H3 V2V Init` `mask` freezes the region during generation
  instead; its default is HARD latent cells (mask_feather 0): every
  ~16 px cell fully frozen or fully live, no fractional blend cells.
  Rationale: half-frozen cells denoise as frozen/live mixtures that can
  themselves mush the seam, while the decode's receptive-field overlap
  smooths a hard cell edge for free. mask_feather above 0 restores the
  pooled fractional ramp if a hard seam ever shows in playback (report
  it; that reading is one A/B from ratified). Either way this path
  cannot do fine pixel feathering; the composite is the fine-feather
  tool.

The GUI for both axes at once is the `H3 Motion Editor` node
(`examples/motion_pipeline_editor.json`): timeline blocks, per-frame
painting, per-block dials, and automation envelopes (hold, feather,
strength) compiled into the same hold_map/mask wires. Queue once to
load the filmstrip, edit on the node, queue again; the baseline stays
cached. Its mask output is already feathered, so the composite runs
with `mask_is_soft` on and its own feather at 0. When tuning for a
user who cannot or will not paint, author `editor_state` JSON for
them; the contract is in the node docstring.

Compute honesty when a user asks where the speed comes from: TIME
targeting is the lever. Sampler cost scales with the dilated frame
count, so gating holds to the user's window helps, and `H3 Segment
Crop` multiplies that by dropping the un-held world from the regen
pass entirely (its report states the ratio). SPATIAL masks do not
reduce FLOPs; the DiT still processes every token and the mask only
controls blending. Sell the mask as quality control and the window as
the speedup.

A mask authored without frame-by-frame attention WILL cut off props
that swing with the subject (a sword mid-cartwheel leaves a static
blob and gets truncated at the boundary). Fixes, in order: paint
per-frame in the editor following the prop, enlarge the static region
to the prop's full arc, or feather outward. This is exactly the
static-vs-per-frame trade; a human on the timeline beats any static
region for rotating props.

Method rules for masks: static union masks cannot pop (the boundary
never moves); route the seam along a real image edge, not through sky
mid-gradient; lasso generously and let feather work; per-frame mask
batches reintroduce the moving-boundary pop and need harder feather
and a playback check. The two-pass rhythm for the targeted graph is
queue once, read `mainodes_targeted_oraclemap`, type ranges, requeue.

## When the expansion stops before the clip does (expand_to_end)

A hold map that expands a burst and then drops back to rate 1 for the
last few world frames reads as a small jump at the end of the shot: the
picture is in slow motion and then, a beat before the cut, it is not.
`H3 Time Smear` and `H3 Temporal Insert` both carry an `expand_to_end`
toggle, default ON, for exactly that shape.

What it does when it fires: the trailing rate-1 run is lifted to the
rate of the span in front of it, and the resulting length is put back
on the 17k+5 grid inside that same span, spending the deficit on the
LAST frames so rates only rise toward the end. The t2c fight window
`[1]*17+[2]*34+[1]*5` (56 world frames, 90 dilated) becomes
`[1]*17+[2]*27+[3]*12` (107 dilated); the same shape with the burst
later, `[1]*34+[2]*17+[1]*5`, becomes `[1]*34+[2]*10+[3]*12` (90
dilated). Mixed rates are the normal outcome, not a fallback: the added
frame count has to be a multiple of 17 and a single rate rarely lands
there.

What it never touches: uniform dilation, a map that already ends inside
an expansion, a map that is rate 1 all the way, and any rate-1 tail
longer than 17 world frames. That last one is the tail guard: one whole
group of frames or more of real time at the end of a shot is intended
rest, not an end jump, and adaptive oracle maps produce it constantly
(a 124-frame map ending in 39 quiet frames would otherwise go from 250
to 294 dilated frames to slow down a section nobody asked to slow
down). Those come back bit-identical, so existing graphs keep their
results. It also never
lowers a hold, so the user never loses expansion they asked for, and it
does not move where the expansion begins: only the tail is rewritten.
A rewrite prints one line to the console with the before and after map
and repeats it in the node's report, so nobody has to guess whether it
fired. `expand_to_end` off is the old behaviour exactly.

Cost warning worth saying out loud before turning it on: running the
span to the end usually buys frames. 90 to 107 dilated frames is about
19% more frame data and more than that in time, because per-step cost
is superlinear in tokens. The report output prices it.

Two traps that cost a round each while this was being built:

1. `H3 Inject Schedule`'s `preset` overrides the `inject` knob unless
   it is set to `custom`. Typing 0.45 into `inject` and leaving the
   preset at "balanced 0.70" renders 0.70, silently.
2. `H3 Time Smear` pads any segment under 39 frames. `_legal_ceil`
   floors at 39, so a 20-frame window smeared at rate 2 comes out at 39
   frames with the whole pad folded into the final hold, which is a
   long freeze on the last frame. With `expand_to_end` on, that pad is
   absorbed by the expansion instead (`[1]*5+[2]*10+[1]*5` becomes
   `[1]*5+[2]*11+[3]*4`, still 39 frames, no freeze). Either way, do
   not hand a sub-39-frame window to the smear and expect the length
   you asked for.

## Source footage instead of a baseline render (alpha)

`examples/archive/motion_pipeline_v2v_source.json` swaps the first pass for a file
on disk: `LoadVideo -> GetVideoComponents -> H3 Video Fit -> ImageScale ->
VAEEncode`, and from there the graph is the standard one. The oracle reads
the source's own encoded latent; nothing else in the chain changes.

Two things to get right, in this order.

1. Frame count. H3's legal lengths are 5, 22, 39 ... 17k+5. An arbitrary
   file lands between them, and the VAE does NOT complain: `encode_temporal`
   pads the last chunk by repeating the final frame, so a 312-frame clip is
   encoded as 323 with 11 frozen frames on the end. The oracle then reads
   those tokens as calm and under-dilates the real ending. `H3 Video Fit`
   trims to 311 instead and hands the count straight to the oracle's
   `length`. Read its report; it states which end lost frames.
2. Canvas. The source is resized to the `ResolutionSelector` dimensions with
   a centre crop, and the same width/height feed the conditioning. Pick the
   aspect ratio that matches your footage or you will crop the subject out.

`max_frames` on the fit node is the cost lever. Per-step time goes as
tokens**1.7, so capping a long source before fitting saves more than it
looks: 192 frames capped to 150 fits to 141, which is 42 tokens instead of
57, about 1.7x faster per step.

Audio: the graph keeps YOUR track. Exact recovery restores the world clock
frame for frame, so the source audio still lines up with the regenerated
video; it is wired into `H3 Audio Recover`'s `reference` at `reference_mix
1.0`. Lower the mix to hear the model's own foley for the slowed
performance instead, or blend. If your source has no audio stream, the fit
node's audio output carries nothing and that link must be removed.

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

## Trying the indecision oracle (experimental)

`H3 Indecision Oracle` is a second signal for the same hold map: the
model's own x0 jitter instead of the clip's jerk. It is not on by
default and nothing else moved for it.

- Cost: free if pass 1 already ran through `X0 Tap` with two steps
  dumped. It reads files, it does not sample.
- Compare, do not switch. Put it where the jerk oracle sits, flip `mode`
  between `indecision`, `jerk passthrough` and `blend max` with
  everything else fixed, and judge in playback. `jerk passthrough` is
  the same numbers the jerk oracle produces, so the only variable is
  the signal.
- Keep `q`, `d_max`, `ramp` and `bridge` matched across the arms. They
  feed the same compiler, and an unmatched knob turns the A/B into a
  comparison of two hold-map compilers.
- The `comparison` output is the measurement, not the vibe: whole-map
  Spearman, top-decile IoU, and the token-times where the two sources
  disagree most. Look at those token-times in playback first, since
  that is where the choice actually shows.
- If the map looks like a rectangle rather than like the picture, check
  the report for the degeneracy line. Masked and repaint runs pin token
  rows to exactly zero jitter and the oracle then draws the mask.
- Expected failure mode: fast small props. Jitter under-ranks them
  relative to frame-diff, which is the argument for `blend max` over
  `indecision` alone.

## Defaults, and why

`q 0.75, d_max 4, ramp on, bridge 8, inject 0.70` is the playback-ratified
starting point. The shipped default is now `inject 0.48`: at 0.50 and above some
scenes measurably loosen reference adherence, and 0.48 keeps the sharpness
without that risk. `inject 0.50` measured sharper with closer motion tracking
on our clips and is one preset away; some of us prefer it. The expert
schedule defaults (`total 8, inject 0.70, base_head 2`) give the turbo
tail its native 4 steps.

## Identity vs strength: the reference rule

Whether a higher inject or denoise is safe depends on what is anchoring
the subject, not on the route.

- No substantial references in the graph: keep the repaint strength
  under 0.5 when the subject's identity must match the source. Identity
  drift is measurable from about 0.30 and structural past 0.5; below
  that line the init still owns the subject, above it the model's prior
  does, and no prompt wording buys it back.
- With substantial references (identity refs, FLF pins, a strong anchor
  frame): higher strength is fine, and on the high-step routes (the
  full de-rope at 25+ steps) it often helps more, because the
  references carry identity so the extra strength buys motion quality
  instead of drift. The pinned FLF de-rope holding identity at inject
  0.70 above is the worked example.

## Refining this guide

When a tuning session finds a symptom/dial pair this table lacks, add it.
That is the point of the file.
