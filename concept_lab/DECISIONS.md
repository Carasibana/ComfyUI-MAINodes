# Concept Lab decisions

Append only. Never edit an old entry: add a new one with `supersedes:`
instead. An edited history means the next person re-litigates a settled
argument without knowing they are doing it, and the path looking straighter
than it was is precisely the damage.

Format: `DEC-CL-nnnn · STATUS · date · title`, then Decision / Alternatives
/ Consequences.

---

## DEC-CL-0001 · ACCEPTED · 2026-08-15 · Fresh subsystem inside MAINodes, named `concept_lab/`

**Decision.** Functional-conditioning work lands as a new package inside
ComfyUI-MAINodes rather than as an extension of any existing
reference-format or adapter-format code. The name is `concept_lab/`.

**Alternatives.** `concept_surgery/` (the source bundle's first suggestion,
and it carries the ancestor program's vocabulary) — not chosen; it reads
sharper than the work is, on a public node pack.
`functional_conditioning/` — most accurate, longest imports, unwieldy node
names. Operator chose `concept_lab/` to sit alongside Motion Lab and Voice
Lab: a place you work rather than an operation you perform.

**Consequences.** Backend-specific formats are compiler *targets*, not core
types. Node display names will read "MAI Concept ...".

---

## DEC-CL-0002 · ACCEPTED · 2026-08-15 · The functional delta is the primary object

**Decision.** The primary scientific object is the matched change in model
function caused by a conditioning intervention, not the content of the
source asset and not a similarity between latents.

**Consequences.** Source-latent similarity alone can never certify a
semantic factor. Every factor claim has to reach an effect on the target
stream.

---

## DEC-CL-0003 · ACCEPTED · 2026-08-15 · Model-neutral core, H3 as first backend

**Decision.** Core data and factor APIs know nothing model-specific.
Segment names, block counts, sampler shapes and layout rules live behind
`backends/`. H3 is first because MAINodes already has H3 tooling and H3 has
a native image/video/audio reference path.

**Consequences.** `types.py` treats segment names as opaque strings. The
second backend should not require rewriting the first one's evidence.

---

## DEC-CL-0004 · ACCEPTED · 2026-08-15 · Three arms, because a reference changes more than its content

**Decision.** First reference experiments capture three arms: actual
reference `R`, shape-matched control `N`, and no reference `0`.
`Delta_content = R - N` is the primary object; `N - 0` and `R - 0` are kept
as diagnostics.

**Why.** Verified against the installed ComfyUI: `PackedLayout`
(`comfy/ldm/minimax/model.py:300`) advances its cursor past every reference
block before placing the target streams, and references also enter the
Qwen3-VL presentation path. So a two-arm on/off difference mixes semantic
content, the presence of a reference, and a moved time origin for the
target stream.

**Consequences.** `N` is a structural control, not a metaphysical zero.
Several control strata must be tried and the sensitivity reported before
`R - N` is called canonical. `backends/h3.py:plan_arms` returns the strata
rather than picking one.

---

## DEC-CL-0005 · ACCEPTED · 2026-08-15 · Capture the target stream, not only the reference rows

**Decision.** v0 captures include target-video DiT state after selected
blocks. Reference-row state is secondary diagnostic evidence.

**Why.** The question is not how the reference is represented. It is how
the reference changes what gets generated.

---

## DEC-CL-0006 · ACCEPTED · 2026-08-15 · Ids are computed over an explicit identity payload

**Decision.** Every id is a hash of a named identity payload, never of the
whole record. Adding a field to a manifest must not move ids that already
exist on disk. If a new field genuinely belongs to a capture's identity, it
joins `identity()` **and** `PROTOCOL_VERSION` is bumped in the same commit,
deliberately, with an entry here.

**Also decided.** Floats are rounded to 12 significant digits before
hashing, so a sigma arriving as float32 and the same sigma arriving as
float64 name one condition rather than two.

**Alternatives.** Hashing the whole record — rejected: every bank on disk
goes stale the next time anyone adds a note field, for no scientific
reason. Ignoring floats' representation — rejected: an id that disagrees
about whether two identical conditions are identical is an id that lies.

**Consequences.** `tests/test_concept_lab_contracts.py` check 2 asserts
both directions: identity fields move the id, notes and timestamps and
status do not.

---

## DEC-CL-0007 · ACCEPTED · 2026-08-15 · Same id with different content is refused

**Decision.** Writing a record under an existing id with different bytes
raises. Writing identical content is idempotent.

**Why.** A capture id names a condition. If the same condition produced
different bytes, one of the two runs is not what its manifest says, and
silently keeping the newer one destroys the only evidence that something is
wrong.

---

## DEC-CL-0008 · ACCEPTED · 2026-08-15 · Two-root workspace, neither root in the repo

**Decision.** Manifests default under the ComfyUI output directory
(`output/h3_concept`), tensors default to `/mnt/weights/ai/concept-space/tensors`
on a box that has it, falling back to `<index root>/tensors`. Both are
overridable per call and by `H3_CONCEPT_INDEX_DIR` / `H3_CONCEPT_TENSOR_DIR`.

**Why.** Manifests are for reading and diffing and belong where a human
browses. Tensors are scientific evidence, can be large, and belong on the
array. Neither belongs in a public repository, because a concept pack can
encode a likeness and the failure mode is somebody running `git add -A`.

---

## DEC-CL-0009 · ACCEPTED · 2026-08-15 · Imported evidence keeps its origin and never merges

**Decision.** An imported space lands under `imported/<origin space id>/`
with the origin stamped on each record. Imported records are never
re-exported under this space's id. Importing your own export back into
itself is refused.

**Why.** The whole subsystem turns measurements into claims. Somebody
else's measurement informing your work is good; becoming indistinguishable
from your own is how a claim loses its scope.

---

## DEC-CL-0010 · ACCEPTED · 2026-08-15 · Packs are private by default; export names what it held back

**Decision.** `ConceptPackManifest.visibility` defaults to `private`.
Export excludes private packs unless explicitly asked, and reports every
exclusion with its reason. Source assets are referenced by content hash and
never copied into a pack.

**Consequences.** The MCP surface can export the shareable set and can see
what was withheld; releasing a private pack is a human keystroke through
the CLI. A model cannot prove who is asking.

---

## DEC-CL-0011 · ACCEPTED · 2026-08-15 · Three surfaces, one verb layer

**Decision.** `api.py` holds one implementation of every operation and
returns plain JSON-able dicts. `cli.py`, `mcp.py` and a future `nodes.py`
are thin clients with no rules in them. No tensors, ComfyUI types or torch
objects cross the api boundary.

**Why.** It is what lets an agent drive the subsystem over a pipe, lets the
tests run with no GPU, and stops the three surfaces drifting into three
implementations.

**Consequences.** When a verb eventually touches tensors it returns
references, not handles. If that becomes painful, it gets decided here
rather than worked around in one surface.

---

## DEC-CL-0012 · ACCEPTED · 2026-08-15 · Unbuilt verbs refuse and name their task

**Decision.** A verb whose machinery does not exist raises
`NotYetImplemented` carrying the task id that blocks it. It never returns
an empty or placeholder result.

**Why.** The output of this subsystem is evidence. A quiet nothing gets
recorded and later read as a measurement.

---

## DEC-CL-0013 · ACCEPTED · 2026-08-15 · No ComfyUI nodes until there is something to run

**Decision.** `concept_lab` registers no nodes and the pack's `__init__.py`
is untouched. Nodes arrive at T12, wrapping proven verbs.

**Why.** Existing MAINodes behaviour and defaults must not move for
additive alpha work. And a half-built research surface in a UI is how ten
unstable knobs become somebody's saved workflow.

**Consequences.** The guarded-import line that adds `concept_lab.nodes` to
the pack's loader is a one-liner when the time comes; the loader already
does exactly this for `window_loop`, `window_expand` and `timeline.nodes`.

---

## DEC-CL-0014 · ACCEPTED · 2026-08-15 · Held-out, adversarial and null splits are not optional

**Decision.** A corpus carries four splits and `corpus_status` reports the
missing ones as warnings. No factor ships on extraction-set scores.

**Status of the enforcement.** The warnings exist; a *containment* check
(no source hash appearing in two splits) does not yet, and is proposed in
the review brief. Recorded here so the gap is visible rather than assumed
closed.

---

## DEC-CL-0015 · ACCEPTED · 2026-08-15 · Process launch is an identity dimension of a capture; arms of one comparison share one launch

**Decision.** `FunctionalCaptureManifest` carries a
`process` fingerprint and `process["launch_id"]` joins `identity()`. The
launch id is minted once per interpreter (`types.process_fingerprint`).
Host, pid, start time and Python version stay OUT of identity: a manifest
must be re-readable on another machine without its id moving.

Corollary, and it is a protocol constraint rather than a nicety: R, N and 0
are three separate forwards, so all arms of one comparison — and the
replicates that make a delta — must be captured inside one process launch.
A capture interrupted and resumed in a new process yields an arm 0 that is
not comparable row-for-row to the R and N from the old one.
`capture_plan` prints this in `protocol_notes` and stamps the current
`process_launch_id`.

**Why.** On this rig the H3 pipeline is bit-identical WITHIN a process and
merely close ACROSS processes: per-process kernel autotuning puts a
~41-48 dB PSNR floor between two cold-process renders at the same seed
(`.claude/skills/h3-latent-mechanics/SKILL.md` §7 in ComfyUI-ModelCatalog).
Before this entry, `identity()` excluded tensor hashes while `Space.put`
compared canonical JSON that includes `TensorRef.sha256`, so two honest
reruns at one seed in two processes produced the SAME capture id and
DIFFERENT bytes, and `put` accused the second of not being what its
manifest said. The first legitimate replicate anyone captured would have
hit it.

**Alternatives.** A mandatory explicit `replicate` set by the caller —
rejected: it defaults silently to 0, so the failure returns the first time
someone forgets, which is exactly the population that needs the guard. A
tolerance-based collision check ("close enough bytes are the same capture")
— rejected: within a launch the pipeline is exact, so a tolerance hides a
real defect in the case where the check is most informative.

**Consequences.** `PROTOCOL_VERSION` bumps to `concept_lab/0.2.0`
(DEC-CL-0006's rule: a field entering identity bumps the protocol,
deliberately). DEC-CL-0007's rationale is amended below. T3 will refuse
`alignment="exact_rows"` when `FunctionalDeltaManifest.same_process()` is
False; the two `*_process_launch_id` fields are there for it now, as
non-identity provenance. The cold-process floor gets MEASURED in E0 rather
than assumed: the ~41-48 dB figure was taken on decoded renders, not on
captured block states, and nobody has read the block-state number yet.

---

## DEC-CL-0007 · AMENDMENT (2026-08-15) · rationale narrowed, decision unchanged

The refusal stands: same id with different bytes still raises. The reason
it is correct is narrower than the original entry said. WITHIN one process
launch the pipeline is bit-identical, so different bytes under one id mean
one of the runs is not what its manifest says. ACROSS launches two honest
reruns differ by kernel autotuning alone, and calling that a defect would
have refused the first real replicate — which is why the launch id is now
part of a capture's identity (DEC-CL-0015) and those two runs are two
captures.

`Space.put`'s message now says both halves: "Same id, different bytes.
Within one process launch this means the id does not describe the
condition; if these came from different launches the capture is missing its
process fingerprint (DEC-CL-0015)."

---

## DEC-CL-0016 · ACCEPTED · 2026-08-15 · Corpora, tap specs and plans are definitions and want version control (amends DEC-CL-0008's rationale, not its default)

**Decision.** The default index root does not move (DEC-CL-0008 stands, and
it is the operator's call). But `Space.status()` reports
`index_root_tracked` and warns when the index root is not inside a git work
tree, and `corpus_new`, `corpus_add`, `corpus_status` and `capture_plan`
carry the same warning. The recommendation is to point
`H3_CONCEPT_INDEX_DIR` at a directory inside a PRIVATE repository, or to
copy `corpora/` and `plans/` into one.

**Why.** Manifests are not one kind of thing. Captures and deltas are
RESULTS and can be re-measured from a corpus plus a stack. Corpora, tap
specs and plans are DEFINITIONS: they encode which contrasts somebody chose
to shoot, and no tensor reconstructs them. The ancestor program's F30 is
the paid-for version of this: the definitions behind four findings lived
only under a gitignored output directory
(`docs/concept_lab/FABLE_SYNTHESIS_2026-08-15.md` §3.8).

**Alternatives.** Moving the default index root into a repo — not taken
here; DEC-CL-0008 settled the default deliberately and a public node pack
is the wrong place for it to land by accident. Silence — rejected: the
loss is invisible until the day someone wants to re-run the experiment.

**Consequences.** Status and the corpus verbs warn. Whether the default
moves is the operator's decision, and this entry does not pre-empt it.

---

## DEC-CL-0017 · ACCEPTED · 2026-08-15 · Split containment is enforced, and lineage counts as containment (closes DEC-CL-0014's open gap)

**Decision.** `ConceptCorpus.containment()` lists every `content_hash` and
every `source_id` that appears in more than one split, with the splits and
row ids. `corpus_status` reports it and warns per collision;
`capture_plan` REFUSES, naming the value and the splits, unless
`allow_containment=True` is passed — and then the plan carries
`containment_overridden` so the exception is in the record rather than in
somebody's memory. `ConceptCorpusRow.source_id` is new and enters the row
identity.

**Why.** DEC-CL-0014 recorded this as a visible gap. A content hash alone
does not close it: two crops of one take are different bytes and the same
observation, so held-out means nothing if a lineage id is not checked too.
Refusal rather than a warning at this one point, because plan time is the
last moment when the fix is free.

**Consequences.** A corpus written before this entry has no `source_id`
anywhere, which reads as "not recorded" and never as "no collision".
Absence is skipped, not treated as a shared value.

---

## DEC-CL-0018 · ACCEPTED · 2026-08-15 · The capture tap ships as two thin alpha nodes now (amends DEC-CL-0013's timing, not its rule)

**Decision.** Capture must run INSIDE the ComfyUI process — the states live
for microseconds inside the DiT block loop and the launch is part of the
condition (DEC-CL-0015) — so the seam needs a node to exist. Two ship now,
`MAIConceptCaptureArm` and `MAIConceptCaptureFlush`, under the category
`MAI/concept (alpha)`. They hold no rules: the Arm node parses its strings
into a `TapSpec`, reads a stack fingerprint off the live `ModelPatcher`,
opens a `TapSession` and installs it on a CLONE; the Flush node calls
`finalize()`. Every refusal comes from `concept_lab/`.

DEC-CL-0013's rule stands and is what makes this safe: existing MAINodes
behaviour and defaults do not move. The registration is the one-liner that
entry predicted, in the same guarded loader tuple as `window_loop`,
`window_expand` and `timeline.nodes`, so a broken alpha module cannot take
the pack down with it.

**Alternatives.** A monkeypatched sampler — rejected: `ModelPatcher`'s
`add_wrapper_with_key` / `set_model_patch_replace` are the sanctioned seam,
and the true-clock precedent is a monkeypatch we would rather not repeat.
Waiting for T12 — rejected: nothing else in the subsystem can produce a
capture, so waiting means the data layer keeps being tested against itself.

**Consequences.** E0's gates are run THROUGH these nodes: a baseline render,
a tap-installed-disabled render and a tap-enabled render at one seed in one
process must decode to pixel-identical frames. If the pass-through gate
fails, the nodes are pulled, not patched around. Until those gates run live,
the nodes are alpha and say so in their DESCRIPTION.

**Also decided here (mechanical, same change).** `TapSpec.store` becomes a
comma-joined subset of `("stats", "frame_mean", "sketch", "full")` and is
canonicalized to that order, because one recorded forward is expensive and
the four reductions answer different questions off the same rows. Single
words stay legal and keep their meaning, so no id already on disk moves.

---

## DEC-CL-0019 · ACCEPTED · 2026-08-15 · Capture identity is measured at the model boundary, not declared; the tap refuses before its first byte and streams into a partial dir

**Decision.** Three parts, one rule.

1. A capture's `ConditionVariant` is built by the tap on its FIRST
   diffusion-model call, from what the model was handed: the text-encoder
   output (`sha256` of `context` through float32) as the first
   `ConditioningSource`, then one source per `minimax_payload["refs"]` block
   and one per `keyframes` block, hashed over their latents in packed order.
   `variant.name` stays a LABEL and stays out of identity; `variant.prompt`
   stays `None` (the human text is not reachable at this seam, and the
   encoder output the model consumed is the stronger identity). Noise-aug
   levels, `audio_scale`, the token-tag digest and the payload seed ride in
   the text source's `preprocess`, which `ConditioningSource.identity()`
   already covers, so no schema changed and `PROTOCOL_VERSION` does not move.
2. The payload seed must equal the seed the Arm node was given. The node
   cannot read the sampler's seed, so it is copied by hand; a mismatch now
   raises naming both numbers instead of mislabelling the capture.
3. `_ensure_capture_id()` refuses BEFORE the first write when the id is
   already on disk (manifest present, or a non-partial tensor directory),
   naming the existing variant, replicate, launch and paths. Tensors stream
   into `<tensor_dir>/<id>.partial/` while `TensorRef.path` records the final
   `<id>/...`; `finalize()` files the manifest through `api.capture_record`
   and only then renames the directory. `api.capture_exists()` is the stdlib
   door the tap knocks on. `Space.doctor()` reports leftover `*.partial`
   directories as `unfinished_capture` problems.

**Why.** Both halves are the reported-vs-consumed family the ancestor
program kept paying for, and both were caught live rather than reasoned
about. `MAIConceptCaptureArm` built `ConditionVariant(name, arm)` with
`sources=()` and `prompt=None`, and `ConditionVariant.identity()` is
`{arm, sources, prompt}`, so every R cell hashed to one id and every N cell
to another regardless of which plate the model actually saw: 12 of 15 E0.5
cells collided. Worse, the same-id refusal lived in `capture_record`, which
runs at the END, while `h3_tap` had already streamed tensors into
`<tensor_dir>/<id>/`. A collision therefore overwrote the earlier capture's
tensors before anything detected it, and E0 replicate 0's tensors were
destroyed that way. A refusal that fires after the damage is not a refusal.

**Alternatives.** Putting `variant_name` into identity — rejected: it makes
the label load-bearing, so a typo becomes a new condition and a copied node
silently merges two. Declaring sources from the node's inputs — rejected:
that is the reported number again, and the plate a node points at is not
necessarily the latent the model got. Buffering tensors until finalize —
rejected: a capture's worth of block states is exactly what T4 refuses to
hold in VRAM. Deleting the partial dir on failure — rejected: it is the
evidence that something failed.

**How to apply.** Arms of one comparison differ by what they feed the model,
not by what they are called; two arms with different plates now get
different ids automatically, and two arms with the same plate and the same
label get the same id and the second one is refused. An honest rerun of a
condition inside one launch increments `replicate`. A `*.partial` directory
in the tensor root is wreckage: `space doctor` lists it, the operator
deletes it, nothing else touches it.

**Consequences.** `tests/test_concept_lab_tap.py` grew sections 7-9
(identity from context and refs, refuse-before-writing and partial/promote,
render fingerprint). The render fingerprint is evidence and not identity
(DEC-CL-0006 stands): `MAIConceptCaptureFlush` hashes frame 0 and the whole
decoded batch as uint8 into the manifest notes, so E0's pass-through gate is
a hash comparison rather than two humans watching two videos.

---

## DEC-CL-0020 · ACCEPTED · 2026-08-15 · The first injection is additive at the tap, in frame_mean form, with its controls built in

**Decision.** The first injection is additive at the tap, in `frame_mean`
(and optionally separable frame + patch) form, on the target-video rows,
with `zero` / `time_shuffle` / `sign_flip` controls built in; it is an
instrument for E0.5's premise test, not the compiler.

Mechanically: `MAIConceptInjectDelta` loads a delta pack (a `.safetensors`
of `b<block>_s<step>/frame_mean` `[latent_t, hidden]` maps plus an optional
`patch_mean` `[frame_rows, hidden]`, with a sidecar JSON carrying
`latent_t`, `frame_rows`, `hidden`, the captured steps and the source
capture ids) and, at the post-block point where `TapSession._record` reads,
adds `alpha * d_frame[t]` to every row of latent frame `t` of the video
segment, in place in the block's OWN output dict. `separable` adds
`alpha * (d_frame[t] + d_patch[p] - grand)`, the rank-2 map with the shared
grand mean counted once. `captured_only` fires on the pack's steps;
`nearest_all` fires every step off the nearest captured entry, ties to the
earlier one. Step index, layout stash and `cond_or_uncond` handling are the
tap's, shared rather than copied (`h3_tap.schedule_and_sigma` /
`step_from_sigma`, `TapSession._layout_signature`).

**Why.** The capture side can be right and mean nothing: a delta that
separates R from N in a bank is a statistic until something is generated
with it. Additive-at-the-tap is the smallest thing that spends the delta
where it was measured, so a null result indicts the delta rather than a
translation layer. The controls ship WITH it rather than after it because
the two questions the first read has to answer are both control questions:
`zero` runs every line of the patch with zeroes (if that render moves, the
plumbing is the effect, not the delta), and `time_shuffle` keeps the
delta's energy while destroying its timing (if that renders like `none`,
the timing content is not what is doing the work). A control invented after
a promising render is a control chosen to survive it.

**Alternatives.** Replacing rather than adding (set the rows' frame means to
the pack's) — rejected for the first read: it destroys the run's own
content and makes a null result unattributable. Injecting through the
conditioning (a new ref block) — rejected: that is a different experiment
about the packer, and it cannot address a chosen block or step. Waiting for
a compiler that maps a factor to a schedule — rejected: the compiler's
first input is exactly the measurement this node produces. A distinct patch
NAME so a tap and an inject could share a block — rejected on the source:
the block loop reads only `patches_replace["dit"]`
(comfy/ldm/minimax/model.py:643-655), so a patch under any other name never
fires, and comfy keeps ONE callable per (name, block, index)
(model_patcher.py:93-112), so a second install on one block silently
replaces the first. The injector therefore refuses on block overlap and
says to put the tap at a later block, where it records the injected state.

**How to apply.** One pack, one arm per render, all arms in one launch
(DEC-CL-0015 binds injections too). Read `none` against `zero` before
reading `none` against anything else. The geometry guard is
(latent_t, frame_rows) recomputed off the live layout the way
`TapSession._write` computes them, checked before any block fires, because
the manifests carry segments and a position hash rather than the packed
signature tuple; text_len and audio_t are free to differ, so a delta may be
spent under a different prompt. `alpha` is the dose and it is not
calibrated: the report logs the applied norms per (block, step) on the
ComfyUI console at the last step, and that is the number to quote.

**Consequences.** `concept_lab/backends/h3_inject.py` is the second and last
file in the subsystem that imports torch; `types.py`, `space.py` and `api.py`
stay stdlib-only. `h3_tap`'s step arithmetic moved to two module functions so
both sides read one implementation; `TapSession` behaviour is unchanged and
its suite is untouched and green. `tests/test_concept_lab_inject.py` is the
new offline suite (pack reading, zero/frame/separable arithmetic, controls,
step mapping, geometry refusals, cond_only, tap+inject on one clone). The
node is ALPHA and has NOT been run live: E0.5's premise test is the first
run, and if the `zero` arm is not bit-identical to an unpatched render the
node gets pulled rather than patched around.

---

## DEC-CL-0021 · ACCEPTED · 2026-08-15 · Injection gains a full-state mode and a spatial-shuffle control

**Decision.** `MAIConceptInjectDelta` gains `mode='full'`, which adds the
whole stored block state (`b<block>_s<step>/full`, `[latent_t * frame_rows,
hidden]` in the video segment's packed row order) row for row instead of a
reduction, and `control='space_shuffle'`, which permutes the frame_rows axis
with one seeded permutation shared by every frame. `frame_mean` injection
was measured to act as a timing-blind global bias (E0.5, 2026-08-15), so the
object injected must be the object measured to agree.

**Why.** E0.5 spent a frame map and got a global push:
`/mnt/work/ai/apps/ComfyUI-ModelCatalog/docs/concept_lab/E05_RESULTS_2026-08-15.md`.
A frame map has already averaged the spatial axis away, so the arm has no
"where" to be wrong about, and `time_shuffle` is the only control the object
can even fail. The full state is the object the tap files and the object the
cosines were computed on, and `space_shuffle` is the control that asks
whether its placement is doing the work. A null on the reduction indicts the
reduction, not the delta, and that ambiguity is what this closes.

**Alternatives.** Reading the reduction harder (more blocks, more steps,
bigger alpha) — rejected: it varies the dose of a quantity already known to
be timing-blind. Holding the pack's full states in RAM the way the
reductions are held — rejected: ~160 MB per (block, step) at production
geometry. They are read one key at a time on first use and cached only in
the model's own device and dtype, only for the (block, step) pairs a session
selected, which is at most `blocks x steps` of them.

**How to apply.** `space_shuffle` in `mode='frame'` REFUSES: frame_mean has
no space axis and the arm would render identically to `none`, which is a
control that cannot fail and therefore is not one. `mode='full'` refuses at
session build if any selected (block, step) has no `full` tensor. Sidecar
keys are unchanged; a pack may carry reductions, full states, or both, and
the shape guard is `(latent_t, frame_rows, hidden)` as before.

**Consequences.** `tests/test_concept_lab_inject.py` grows section 10 (full
mode: exact add on video rows only, laziness, both shuffles as permutations
of the axis they name, fp16 packs applied at the model's dtype); the pack
reader now enumerates keys through `safe_open` rather than `load_file`, so a
full pack's bytes are never all resident. The node is still ALPHA and the
full mode has NOT been run live.
