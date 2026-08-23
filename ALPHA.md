# What is alpha here, and how alpha is it

This pack ships finished work and unfinished work in the same install. The
finished work is the Motion Lab de-rope pipeline and the contact sheet; it
has been measured, it has example graphs, and its dials have numbers behind
them in TUNING.md. Everything on this page is the other kind.

Alpha here means one or more of: it has run on one machine and no others,
the interesting half of it is not built, its interface will change, or it
is a real capability that simply has not been tested on enough material to
know where it stops working. Each entry below says which.

Nothing on this page changes existing behaviour or defaults. Alpha nodes
carry `(alpha)` in the name you see in the ComfyUI menu, and the subsystems
load behind a guarded loader, so a broken alpha module cannot take the rest
of the pack down with it.

If you find something wrong, an issue with the symptom and the graph is
worth more than a fix. Symptom-and-dial pairs belong in TUNING.md.

---

## Concept Lab

`concept_lab/`, and the nodes `MAI Concept Capture Arm / Flush` and
`MAI Concept Inject Delta`.

A research subsystem on the bet that a reusable concept does not have to
live in trained weights and can instead live in a measured functional
delta: measure what a piece of conditioning actually does to the model,
factor that into reusable components, compile them back through the model's
own conditioning channels.

**State: the interesting half is not built.** What exists is the data
layer (contracts, workspace, verbs, three surfaces over them) and an H3
capture tap that rides along on a render. What does not exist is anything
that turns a capture into a factor, or a factor into conditioning. Unbuilt
verbs raise and name the task that blocks them rather than returning an
empty result, because in a subsystem whose output is evidence a quiet
nothing is worse than a refusal.

Full status table, the layering rules, and a Contributing section naming
the four things most worth receiving: **`concept_lab/README.md`**. The
design decisions and their reasoning are in `concept_lab/DECISIONS.md`.

## Extension (long clips from short renders)

`h3_extend.py`: `H3 Extension Plan`, `H3 Tail Context`, `H3 Protect Prefix`,
`H3 Prefix Freeze Mask`, `H3 Trim`, `H3 Seam Normalize`; the shared types
in `capsule_types.py`; the graph `examples/motion_pipeline_extend_api.json`;
edge-protection dials on `H3 Jerk Oracle` (`protect_tail`) and
`H3 Window Plan` (`edge_protect`).

Short segments generated and de-roped one at a time, the last 39 frames
carried into the next segment, the overlap trimmed at assembly. Integer
time throughout: the 141/39 atom adds exactly 102 frames and 170 audio
ticks per segment. On a core with per-token masks (#15375) the handle is
WRITTEN INTO the next segment's own pass-1 latent under a time-varying
mask, with the audio handle on an audio-only `MiniMaxH3AddGuide`; on older
cores the whole handle rides the guide. The de-rope holds the prefix at 1
AND freezes it in pass 2; a non-final segment's last 17 frames are also
held at 1, because a gesture that runs into the cut has no "after" to slow
into and comes back fast otherwise. `H3 Seam Normalize` fits per-channel
linear-light gains on the hidden prefix (each VAE round trip darkens it
~2.4%) and applies them to the new material; its audio rms gain measured
NEGATIVE and defaults off.

**State: measured on two content sets, one machine, two segments.** The
masked path beats the image guide everywhere we measured: join jerk 0.86x
to 1.1x the clip's ordinary frame-to-frame motion vs 5x to 6.5x for the
guide, camera velocity continuous through the join instead of reversing,
handle within 2.2 to 3.3/255 of the carried tail, ~20% less wall (no
guide rows in every block). The closing-gesture fix measured 1.55x -> 1.06x.
Unmeasured: more than two segments (the drift curve), the 192/90 atom,
lower inject on continuation segments, dialogue across a join, and the
audio ambience bed, which still steps at the cut (a per-tick audio mask
is the planned fix). The API graph is the only form shipped so far.

## The timeline surface

Nodes: `H3 Drawn Plan`, `H3 Plan Settings`, `H3 Plan Estimate`,
`H3 Timeline Analyze`, `H3 Timeline Render`, `H3 Flight Recorder Start` and
`Stop`.

A plan document as the single source of truth: something proposes a plan,
a human edits it, a compiler turns it into a legal graph. The node surface
reaches parity with the compiler route, and the splice is proven on tensors.

**State: real but young.** The interface is expected to move, and the
editing experience around it is not built. If you are looking for the
production route, use the ordinary de-rope graphs.

## The audio init for dialogue

`H3 Audio Smear`, plus `audio_latent` / `audio_mode` on `H3 V2V Init` and
`audio_source` on `H3 Audio Recover`.

Fixes a real defect: a de-rope breaks speech, because the picture gets an
init and obeys it while the audio starts from zeros, so pass 2 writes a
fresh performance at natural rate and moves the mouth to that. Seeding the
audio rows with the baseline performance stretched onto the same dilated
clock makes pass 2 render a genuinely slowed take. The README's
"Dialogue through a de-rope" section has the adoption steps.

**State: it works, on material we have not varied enough.** The geometry
is solid: the smear/recover round trip is sample-exact in both directions
with envelope correlation 0.975, and the init costs nothing measurable in
picture quality. But it has been heard on a handful of clips, all
two-speaker sword-fight material at hold factors around 4 and dilations
between 2.4x and 2.7x. A single speaker, speech over music, non-anime
footage, and other hold factors are all unmeasured. If you try one of
those, that result is worth reporting whichever way it goes.

## The motion adapter (pilot)

A rank-16 LoRA trained on the de-rope task itself, applied to the de-rope
pass only. Documented with its measured settings and its known costs in the
README's "The motion adapter (pilot)" section, and released as an
intermediate option rather than a finished one. A more ambitious all-in-one
adapter is in progress and may not work.

## Manual mask paths

The `(alpha)`-tagged inputs on `H3 V2V Init` and `H3 Motion Composite`:
manual region masks, final-alpha masks, and the time-varying mask path.
These are exercised by the checklist in **`TESTING_ALPHA.md`**, which
covers the 2026-08-09 node batch and still wants human passes on the
interactive widget and the drag-in workflows.

---

## VRAM Lab

`H3 Streamed Blocks`, `H3 Memory Probe`, `H3 Free Cache`, `H3 Evict Text
Encoder` (`vram_lab.py`, 2026-08-18). Alpha because it has run on one
machine (RTX PRO 6000 Blackwell, 188 SMs) and its exactness is a kernel
property, not a code property: query chunking is bit-equal only while every
chunk keeps PyTorch's flash kernel off its split-KV path (measured boundary
`heads x ceil(L/64) >= 0.8 x 2 x SMs`; the node sizes chunks from the SM
count with a 2.6x margin and carries a `self_check` that compares stock and
streamed on the first block's real input). int8 and W4A8 checkpoints are
bit-equal by mechanism (row-wise activation scales, int32 accumulate);
NVFP4/FP8 would need one shared activation scale and are untested; bf16 is
numerically equivalent, not identical. `kv_block` is experimental and as
built does not lower memory (leave it at 0). The output head has an exact
mode (default) and a chunked-GEMM mode that changes the clip after 25 steps
by ~1e-6 per step of fp32 difference; that mode is labelled and off.

What was measured, one graph (294 f -> 702 f de-rope at 1376x768, ~217k
tokens): torch activations peak +11.9 GiB over the weight floor at any
card; a 16 GB card in a 32 GB machine renders it with ComfyUI's dynamic VRAM
and `--fast-disk` at 316 s/step; a 96 GB card resident is 311 s/step. RAM is
the small-machine ceiling in normal mode (CPU copies of the models without
`--fast-disk`, and CPU-side IMAGE intermediates); the text encoder costs
1 to 2.5 GiB of peak on a 16 GB card and about 3 s per new prompt, and
either eviction or the conditioning bank removes it. Untested: other GPUs,
Windows, ROCm, non-flash attention backends (cuDNN and mem-efficient are
chunk-invariant at every length here but 1.1 to 2x slower; no two backends
are bit-equal to each other).

The `kv_store` options (`kvi8r`, `kvi8s`) are the approximation tier of the
same node: half the K/V bytes (kvi8r measured +9.5 GiB forward peak against
the exact +11.9 at 217k tokens; kvi8s attends on int8/fp8 tensor cores
straight from the store, ~1.6x faster attention standalone), same-seed
renders are sibling takes, and only kvi8r's first cut has been in front of
eyes ("almost perfect" on the de-rope side by side, one clip, one viewer).
The node defaults to the exact store; the low-VRAM example graph turns kvi8r
on (operator's call for that graph); the table with the numbers and what is
still owed is in LOWVRAM.md. The int8_convrot video VAE now in the example graph is also an
approximation (60 dB PSNR against the fp16 decoder on the same latent).

## Where to look next

| | |
|---|---|
| `concept_lab/README.md` | Concept Lab status, layering, contributing |
| `TESTING_ALPHA.md` | hands-on checklist for the manual mask and editor batch |
| `TUNING.md` | dials with measured numbers, for the finished pipeline |
| `ROADMAP.md` | where the unfinished parts are meant to go |
