"""The timeline surface: a semantic plan document, a compiler seam, and a
price meter.

    oracle/proposers -> plan.json <-> [editor UIs] -> backend -> graph -> queue_scene

Layering, and it is load-bearing:

    timeline/schema.py    SEMANTIC INTENT ONLY. Stdlib. No H3, no torch.
    timeline/backend.py   the four-method seam (capabilities/validate/
                          compile/estimate). One implementation, no factory.
    timeline/price.py     geometry x complexity model x machine calibration,
                          three separable factors.
    timeline/recorder.py  the flight recorder that calibrates layer (b).
    timeline/h3/          EVERYTHING H3-specific: model spec (structural,
                          relied on), recipe profile (experiment-derived,
                          challengeable, cited), grid law, compiler, oracle
                          proposer.
    timeline/nodes.py     thin ComfyUI clients. No rules live here.

A UI is a viewer/editor of the plan and never contains legality math.
Adding a capability means adding a lane type, not a node pack.

NOTE: this file imports NOTHING on purpose. `timeline.schema` must be
importable with no torch, no comfy and no h3 in the process (the test
suite asserts exactly that in a subprocess), so the ComfyUI node
registrations live in timeline/nodes.py and the pack's __init__ imports
that submodule directly.
"""
