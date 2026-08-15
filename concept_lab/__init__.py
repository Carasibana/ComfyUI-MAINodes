"""Concept Lab: measure what conditioning does to the model, factor that
effect into reusable pieces, and compile the pieces back through the
model's own conditioning channels.

The thesis in one line: a reusable concept does not have to live in trained
weights. It can live in a measured functional delta.

    examples -> functional deltas -> factorization -> portable operator
             -> native conditioning

Layering, and it is load-bearing (the same discipline timeline/ uses):

    concept_lab/types.py      CONTRACTS ONLY. Stdlib. No torch, no comfy,
                              no H3. Dataclasses, canonical JSON, and the
                              deterministic ids everything else keys on.
    concept_lab/space.py      the workspace: a manifest root you can see, a
                              tensor root that holds the heavy evidence, and
                              the export/import that moves a space between
                              machines with provenance intact.
    concept_lab/api.py        THE VERBS. One implementation of every
                              operation. Returns plain JSON-able dicts.
    concept_lab/backends/     everything model-specific. base.py is the
                              seam; h3.py is the first backend.
    concept_lab/strategies/   factorization algorithms (arrive with their
                              experiments, not before).

    concept_lab/cli.py        shell client over api.py
    concept_lab/mcp.py        typed agent client over api.py
    concept_lab/nodes.py      ComfyUI client over api.py (not yet written;
                              see IMPLEMENTATION_NOTES.md for why)

Three surfaces, one authority. A verb that is not in api.py does not
exist, and no surface is allowed to hold a rule the others cannot see.
That is what makes a graph, a shell and an agent interchangeable here.

This file imports NOTHING on purpose. `concept_lab.types` and
`concept_lab.space` must stay importable in a process with no torch and no
comfy in it (tests/test_concept_lab_contracts.py asserts exactly that in a
subprocess), because the whole point of the contracts is that they are
readable and writable by things that are not ComfyUI.

Status vocabulary, used on every claim in this subsystem:

    PROPOSED    idea, not measured
    MEASURED    measured under a named condition
    SCOPED      measured, valid only on a named distribution
    FAILED      tested, did not pass its gate
    RETRACTED   an earlier reading was wrong
    SUPERSEDED  replaced by a better formulation
    OPEN        important unknown, no adequate test yet

Nothing gets promoted quietly.
"""
