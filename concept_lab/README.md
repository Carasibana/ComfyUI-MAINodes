# Concept Lab (alpha, nothing measured yet)

A reusable concept does not have to live in trained weights. It can live in
a measured functional delta.

That is the whole bet. Measure what a piece of conditioning actually does to
the model, factor that measured effect into reusable components, and compile
the components back through the model's own conditioning channels.

```
examples -> functional deltas -> factorization -> portable operator
         -> native conditioning
```

The primitive is a difference between two matched forward passes. For model
state `h` at layer `l`, sampling coordinate `s`, output segment `q`, base
input `x`, and a conditioning source `C`:

```
Delta(C; x, l, s, q) = h(x | C) - h(x | control(C))
```

The control is doing the real work in that line. On H3 the interesting first
delta is not "reference versus no reference", because adding a reference
changes the packed layout, the presentation tokens, and the time origin of
the target stream all at once. It is "actual reference versus a
shape-matched control reference", which holds all three fixed and leaves
only the content.

## Status

**Alpha, and deliberately narrow.** What exists is the data layer (the
contracts, the workspace, the verbs, and three surfaces over them) plus the
H3 capture tap that rides along on a render. What does not exist is anything
that turns a capture into a factor, or a factor into conditioning.

| built | not built |
|---|---|
| contracts with deterministic ids (`types.py`) | the delta engine (T3) |
| the workspace, export and import (`space.py`) | factorization strategies (T6+) |
| the verb layer (`api.py`) | the compiler (T9) |
| shell and agent surfaces (`cli.py`, `mcp.py`) | the rest of the node surface (T12) |
| the H3 backend's verified anchors (`backends/h3.py`) | |
| capture (T4/T5): tap + nodes built, E0 gates NOT yet run live | |

Unbuilt verbs raise and name the task that blocks them. They do not return
an empty result. In a subsystem whose output is evidence, a quiet nothing is
worse than a refusal, because the nothing gets written down.

Three alpha nodes are registered (`MAI Concept Capture Arm` / `Flush`,
DEC-CL-0018, and `MAI Concept Inject Delta`, DEC-CL-0020), through the pack's
guarded loader, so a broken alpha module cannot take the pack down. No
existing MAINodes behaviour or default moves.

## Layering, and why it is strict

```
types.py      CONTRACTS ONLY. Stdlib. No torch, no comfy, no H3.
space.py      the workspace: manifests you can read, tensors you can't.
api.py        THE VERBS. One implementation of every operation.
backends/     everything model-specific. base.py is the seam, h3.py is first.
strategies/   factorization algorithms (they arrive with their experiments).

cli.py        shell client over api.py
mcp.py        typed agent client over api.py
nodes.py      ComfyUI client over api.py (the capture pair, alpha)
```

Three surfaces, one authority. A verb that is not in `api.py` does not
exist, and no surface holds a rule the others cannot see. That is what makes
a graph, a shell and an agent interchangeable rather than three
reimplementations that drift.

`types.py` and `space.py` stay importable in a process with no torch and no
comfy in it, and the test suite asserts exactly that in a subprocess. The
point of a contract is that it is readable and writable by things that are
not ComfyUI.

## The workspace

Two roots, because the two kinds of thing want different homes.

```
<index root>            small JSON you want to browse and diff
    SPACE.json
    corpora/  captures/  deltas/  packs/  plans/
    imported/<origin space id>/...
<tensor root>           the heavy evidence
    <space id>/...
```

Defaults are `output/h3_concept` next to the renders, and
`/mnt/weights/ai/concept-space/tensors` on a box that has it (falling back
to `<index root>/tensors` so a fresh checkout elsewhere still works).
Override with `H3_CONCEPT_INDEX_DIR` and `H3_CONCEPT_TENSOR_DIR`.

Neither default is inside the repository. That is not tidiness: a concept
pack can encode a likeness, and the way that goes wrong is a research
artifact sitting in a working tree when somebody runs `git add -A`.

But the index root holds two different kinds of thing, and only one of them
is a result. Captures and deltas can be re-measured from a corpus and a
model stack. Corpora, tap specs and plans are **definitions**: they are the
record of which contrasts somebody chose, and no tensor reconstructs them.
So `space status` reports `index_root_tracked` and the corpus verbs warn
when the index root is not inside a git work tree. The fix is one
environment variable pointing at a **private** repository (or a copy of
`corpora/` and `plans/` into one):

```
export H3_CONCEPT_INDEX_DIR=~/research-private/concept_space
```

The default does not move (DEC-CL-0008 settled it); the warning exists
because losing the definitions is invisible until the day you want to run
the experiment again (DEC-CL-0016).

### Export and import

An export bundles manifests and their tensors into one zip regardless of
which root they sat in. An import lands somebody else's space **beside**
yours rather than in it:

```
<index root>/imported/<their space id>/...
<tensor root>/<their space id>/...
```

with the origin stamped on every record. Their numbers can inform your work
and can never be mistaken for your own measurements. Packs marked private
are held back from an export unless you ask for them in as many words, and
every exclusion is reported with its reason — nothing is dropped silently.

## Using it

From a shell:

```
python -m concept_lab.cli space init --label mine
python -m concept_lab.cli corpus new wave
python -m concept_lab.cli corpus add wave --kind video --ref alice_wave.mp4 \
    --label subject=alice --label motion=wave --split extract --group alice
python -m concept_lab.cli corpus confounds wave --target motion
python -m concept_lab.cli capture plan wave
python -m concept_lab.cli space export mine
```

Every command prints JSON, so a pipeline and an agent read the same bytes.

From an agent, over MCP:

```
<venv>/bin/python -m concept_lab.mcp
```

Same verbs, typed. Two things are deliberately absent from that surface:
anything that runs a model (capture costs GPU time on a shared box, so the
agent plans and a human launches), and exporting a private pack (a model
cannot prove who is asking, and export is a publishing act). An agent that
can measure everything and publish nothing is the right shape here.

## The two checks that are cheap and worth running first

**`corpus confounds`** asks which other labels move exactly with the one you
care about. If every waving clip also smiles, then a "wave" factor is a
wave-and-smile factor, and no amount of solver will tell you that. The fix
is another clip, not another algorithm. This costs nothing and answers a
question that is expensive to answer after a capture.

**`capture plan`** costs a three-arm capture without running it: forward
count, tensors written, and every protocol gap the corpus carries. It says
plainly that it measured nothing.

## Rules this subsystem tries to make structural

Each of these is here because a previous research program paid for it.

- **A delta needs a matched control.** Not "the intervention off" — a
  control that holds every structural nuisance fixed.
- **Same id, different bytes is an error.** If one condition produced two
  different results, one of the runs is not what its manifest says, and
  keeping the newer one destroys the evidence that something is wrong.
- **Extraction success is not a result.** Splits are `extract`, `validate`,
  `adversarial`, `null`, and a factor that has only seen the first one is a
  hypothesis.
- **Depth and time are coordinates, not metadata.** A factor collapsed to
  one global vector before anyone measured the layer and timestep profile
  has thrown away the answer.
- **Record the stack, never infer it from a filename.** A capture without
  its condition is a number without a condition.
- **A diagnostic is not a behavioural result.** A decomposition is a
  hypothesis until it predicts output the model was not fitted on.
- **Nothing gets promoted quietly.** Every claim carries one of PROPOSED,
  MEASURED, SCOPED, FAILED, RETRACTED, SUPERSEDED, OPEN.

## Tests

```
python tests/test_concept_lab_contracts.py
python tests/test_concept_lab_tap.py        # torch, no comfy, no GPU
```

Synthetic. No GPU, no models, no renders, no torch. Thirteen check groups
covering the import boundary, id determinism (including which fields are
*not* allowed to move an id, and the process launch that now belongs to
one), roundtrips, refusals, the collision check, confound detection,
export/import provenance, the H3 anchors read from the installed ComfyUI
source, unbuilt verbs refusing, the CLI surface, the version-control
warning, tensor hashing in `doctor`, and the two guards that have to fire:
split containment and the shape-matched control arm.

The tap suite is the second one and needs torch (it never imports ComfyUI):
six check groups over pass-through exactness, step selection, what each
store writes, the manifest and its id, the refusals, and the disabled arm
writing nothing. What it cannot check offline is the only thing that
matters in the end, which is why E0 exists: that a real render is
pixel-identical with the tap installed.

## Reading order for the design

`__init__.py` for the layering, `types.py` for the id rules and why they are
shaped that way, `backends/h3.py` for the H3 anchors with their source
citations, `DECISIONS.md` for what has been settled and should not be
quietly relitigated.
