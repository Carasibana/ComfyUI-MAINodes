# MAINodes Flow Control, Safe Functions, and LLM Decisions

## Technical / UX specification v0.2

Status: accepted for build, 2026-08-28. Phase 1 is in progress on the
`flow-control` branch. This document is the authority for the `flow/`
package; the earlier v0.1 draft it replaces was a design conversation, not a
build spec, and its self-assigned quality scores are dropped here.

Every claim below about what ComfyUI core does was checked against core
0.33.0 on the build machine (`execution.py`, `comfy_extras/nodes_logic.py`,
`comfy_extras/nodes_math.py`, `comfy_api/latest/_io.py`). Where a mechanism
is unproven the text says so and names the probe that settles it.

---

## 0. What core already provides (do not rebuild)

| Need | Core node / mechanism | Where |
|---|---|---|
| Two-way lazy branch on a generic type | `If/Else Switch` (`ComfySwitchNode`), `MatchType` inputs, `lazy=True`, `check_lazy_status` | `comfy_extras/nodes_logic.py` |
| Branch on "is this input connected" | `Soft Switch` (`ComfySoftSwitchNode`) | same |
| Scalar expression to BOOL / INT / FLOAT | `Math Expression` (`ComfyMathExpression`), simpleeval, Autogrow inputs `a`..`z` typed Float / Int / Boolean | `comfy_extras/nodes_math.py` |
| Boolean glue | Not / And / Or / Invert Boolean, String Compare | `nodes_logic.py`, `nodes_string.py` |
| Real laziness | `check_lazy_status` returns the input names it still needs; the executor makes those strong links and re-schedules the node; unrequested inputs' exclusive ancestors never run | `execution.py` (`make_input_strong_link`) |
| Graph-level loops | node expansion (`GraphBuilder`, `ExecutionBlocker`, `NodeOutput(expand=...)`) | `comfy_execution/graph_utils.py` |
| Per-item mapping over lists | any node fed a Comfy list runs once per item (`INPUT_IS_LIST` false) | `execution.py` |
| Regions | subgraphs are flattened by the frontend at queue time; the backend never sees a boundary | frontend |
| Cache without running skipped branches | cache keys come from prompt structure plus `IS_CHANGED`, never from outputs | `comfy_execution/caching.py` |

Consequences:

* The "resize only when scale is not 1" case is solvable in pure core today:
  `Math Expression("a != 1.0")` into `If/Else Switch` with `on_true` from
  Resize and `on_false` from the original image. Three nodes.
* The core logic nodes are `is_experimental=True`; they are hidden until the
  frontend setting for experimental nodes is on. Every example that uses them
  says so.
* This package therefore adds only what core lacks: predicates on data (image
  size, mask coverage, sequence length) rather than scalars, the fused Gate
  that makes the common case one node with passthrough by default, N-way
  lazy dispatch, list Filter / Partition, a debug probe, a restricted
  function node, and typed LLM decisions. Nothing here duplicates a core
  node.

---

## 1. Goals

The system MUST provide:

* True conditional execution: the skipped branch's exclusive producers never
  run. Selecting between two computed values does not qualify.
* One node for "process this when the condition holds, otherwise pass the
  original through".
* Predicates over Comfy values, not only widget scalars.
* Generic operation over IMAGE, MASK, LATENT, CONDITIONING, MODEL, AUDIO,
  VIDEO and any other type `MatchType` accepts.
* Headless / API execution identical to editor execution, with no MAINodes
  JavaScript loaded.
* Portable workflow JSON. Frontend sugar compiles to real nodes.
* Deterministic execution and sane caching.
* Visible reasons for why a branch ran or was skipped.
* A foundation for list control flow (Filter, Partition, For Each).
* A restricted, Python-like function node without `eval` or `exec`.
* No network, filesystem, subprocess, environment, reflection, or import
  capability reachable from workflow-authored text.

Non-goals for v1: multiple dynamic outputs on Safe Function, recursion,
per-item laziness inside one node, a general agent runtime.

---

## 2. Core principle

The editor may present imperative-looking behaviour ("run this node when
`scale != 1.0`"). The executable workflow stays declarative Comfy dataflow.
No correctness-critical behaviour depends on frontend bypass state,
queue-time patching, browser-only state, or scheduler mutation. A workflow
that branches one way in the editor branches the same way through the API.

The security boundary, stated once:

> Workflow authors compose approved computation. They never acquire new
> capabilities. Workflow JSON stays data, not code.

This package does not lower ComfyUI's existing bar (custom node packs are
arbitrary code, and that is unchanged); it refuses to lower it further.

---

## 3. Architecture

Four layers, each usable without the ones above it.

* **A. Expressions.** One allowlist evaluator over the Python `ast`,
  shared by Condition, Gate, Filter, Partition and later Safe Function.
  Syntax is a strict superset of core `Math Expression`, so an expression
  moves between the two nodes unchanged.
* **B. Lazy primitives.** Gate, Lazy Select, plus core's Switch nodes.
* **C. Frontend sugar.** "Run when / Skip when" on a node or a subgraph,
  compiling to exactly one Gate node plus links. Phase 2.
* **D. Programmable nodes.** Safe Function (restricted interpreter, Phase 3)
  and LLM Judge / LLM Choose (typed model decisions, Phase 4).

---

## 4. Nodes (Phase 1)

All nodes are V3 (`comfy_api.latest`, the version the rest of the pack
already imports; revisit the pin at release). Category `MAINodes/Flow`.
Node ids are prefixed `MAIFlow`.

### 4.1 Gate ("process if")

```
expression   STRING   default "a != 1.0"
source       T        MatchType, lazy
processed    T        MatchType (same template), lazy
values       Autogrow a..z of AnyType (values named in the expression)
->  result   T
```

Semantics: `processed if expression else source`.

`check_lazy_status` evaluates the expression over the resolved `values`
and requests exactly one of `source` / `processed`. Both are lazy so the
untaken side is never requested. This is the acceptance requirement: with
the expression false, the producer of `processed` runs zero times.

`ui` output on every run:

```
{"flow": [{"node": "Gate", "took": "processed" | "source",
           "expression": "...", "values": {"a": 1.0, "image": "IMAGE[1,512,512,3]"}}]}
```

Scalars are reported verbatim; tensors and opaque values as `KIND[shape]`.

### 4.2 Condition

```
expression   STRING
values       Autogrow a..z of AnyType
->  BOOL, FLOAT, INT, STRING
```

For feeding core `If/Else Switch`, Lazy Select, or anything else that takes
a scalar. Same evaluator and same `ui` report as Gate.

### 4.3 Lazy Select

```
selector     INT (MultiType INT | STRING when labels are given)
labels       STRING  optional, comma-separated case names ("draft,normal,max")
case_0..     Autogrow (prefix "case_") of MatchType T, lazy
default      T  MatchType, lazy, optional
->  result   T
```

Only the selected case is requested. Out-of-range selector: `default` if
connected, otherwise a validation error naming the selector value.

Unproven mechanism: Autogrow over a `MatchType` template. `_io.py` only
forbids `DynamicInput` templates and `MatchType.Input` is a plain `Input`,
so it is not rejected, but no core node does it and the frontend must
resolve one template across grown slots. Probe first (section 15). If it
fails, the fallback is eight fixed `MatchType` lazy optional inputs using
the `MISSING` idiom from `Soft Switch`.

### 4.4 Filter and Partition (lists)

```
Filter:     items (list of AnyType), expression, values a..z
            -> kept (list), kept_count INT
Partition:  items, expression, values
            -> kept (list), rejected (list), kept_count INT, rejected_count INT
```

`INPUT_IS_LIST` on the node; `is_output_list` on the list outputs. The
expression is evaluated once per item with `item`, `index`, `count` bound,
plus the scalar `values`.

Why Partition is the per-item primitive: `check_lazy_status` is mapped over
list inputs and its requests are unioned, so laziness is all-or-nothing per
input, never per item. Partition into `kept` / `rejected` and feed the
expensive branch only `kept`; a branch fed an empty list maps zero times.
Probe confirms the empty-list case (section 15).

Two distinct batch notions must stay visibly distinct in the UI:

* a Comfy **list** (Python list; nodes map over it), handled by these nodes;
* a **tensor batch** (`IMAGE[B,H,W,C]`), handled by capabilities such as
  `image.select(image, mask)` inside expressions.

### 4.5 Flow Probe (debug)

```
value        AnyType   passthrough
name         STRING
salt         INT       default 0 (vary it to defeat the result cache in tests)
delay_s      FLOAT     default 0
->  value
```

On execute: appends one line to `<output_dir>/flow_probe/<name>.count`,
sleeps `delay_s`, returns `value` unchanged, and reports the running count
in `ui`. This is how the acceptance gates count executions without a
websocket client. Shipped, because users will want it for the same reason.

### 4.6 Identity fast paths (guidance, not a node)

Nodes with an obvious identity case (scale 1, rotation 0, blur radius 0,
strength 0) should return immediately. Gate handles the case where the node
cannot know that a whole branch is unnecessary.

---

## 5. Expression language and capability registry

### 5.1 Grammar (allowlist over `ast`)

Allowed: int / float / string / bool / None literals; names; unary `-`
and `not`; `+ - * / // % **` (exponent capped at 4000 as core does);
chained comparisons; `and` / `or`; conditional expression `x if c else y`;
calls to registered capabilities by bare or dotted name; subscripts with an
int literal or int name on a registered sequence result; tuple and list
literals up to 64 elements.

Rejected, with the policy in the error message: attribute access on values,
lambda, comprehensions, generator expressions, walrus, starred, f-strings,
`__` anywhere in a name, and any name that is neither a bound input nor a
registered capability.

Limits: source 4 KB, 2000 AST nodes, nesting depth 32, 256 calls per
evaluation. Exceeding a limit is a validation error before execution.

Dotted names such as `image.width(x)` are resolved directly against the
registry. No Python module object is ever exposed. There is no `getattr`.

### 5.2 Value handling

Scalars cross as Python scalars. Everything else (tensors, dicts such as
LATENT, model patchers, conditioning lists) is wrapped in an opaque `Ref`
carrying its Comfy type name. The evaluator cannot inspect a `Ref`; only a
capability unwraps it. Capabilities are functional: no in-place mutation of
an input that another node may also consume.

### 5.3 Registry

```python
Capability(id, version, fn, arg_types, return_type, predicate_safe, cost)
```

v1 packs:

| Pack | Capabilities |
|---|---|
| `core.math` | the core Math Expression set, bare names: `sum min max abs round pow sqrt ceil floor log log2 log10 sin cos tan int float` |
| `core.logic` | `clamp(x, lo, hi) between(x, lo, hi) near(x, target, eps) coalesce(a, b) is_none(x)` |
| `image` | `width height megapixels batch aspect` |
| `mask` | `coverage is_empty` |
| `latent` | `frames shape` (shape as a tuple) |
| `seq` | `length` |

All v1 capabilities are `predicate_safe`. Transform capabilities
(`image.resize`, `latent.blend`, `image.select`) arrive with Safe Function
and are rejected inside a Gate or Condition expression, with the error
saying why.

Only installed extension code registers capabilities. Third-party packs
register under their own prefix and are shown separately in the UI. A
workflow cannot add to the registry.

Workflow metadata records the runtime and pack versions used; a load with a
missing or older pack fails validation before queueing.

---

## 6. Frontend sugar: Run when / Skip when (Phase 2)

Right-click on a compatible node:

```
Conditional Execution
    Run always
    Run when...
    Skip when...
    Edit condition...
    Show compiled flow
    Remove condition
```

"Run when" compiles to exactly one Gate node: `processed` from the guarded
node's output, `source` from the guarded node's matching input, expression
from the editor, and the referenced widget values converted to links.

That last step is the hidden cost and it is mandatory: the guarded node's
`scale` widget must become a link shared by the node and the Gate. Two
inline copies of the value in the API prompt would drift, and the headless
run would branch differently from the editor. Do not read another node's
widget through the hidden prompt; it is invisible to the cache signature.

Passthrough mapping is inferred only when unambiguous (one input and one
output of the same type). Otherwise the author picks, and the choice is
workflow data.

The generated Gate is a real node. The frontend may collapse it; "Show
compiled flow" reveals it. Removing the condition removes the node and
restores links.

Regions: select nodes, convert to a subgraph, apply "Run when" to the
subgraph node. Subgraph I/O types are fixed at creation and the frontend
flattens at queue time, so nothing here depends on dynamic type propagation
across a boundary.

---

## 7. Observability

After execution, a Gate shows `took: processed` / `took: source` with the
expression and the values it saw, from the `ui` payload. Skipped producers
are dimmed by the frontend for the most recent run.

The documented trap: attaching a preview or any second consumer to a lazy
branch makes that branch required, and it will run. The FLOW.md user doc
states this in its first screen and Flow Probe exists so users can count
instead of guess.

---

## 8. Safe Function (Phase 3)

A node whose body is a restricted, Python-like function:

```python
def main(original: IMAGE, restored: IMAGE, enabled: BOOL = True, strength: FLOAT = 1.0) -> IMAGE:
    if not enabled:
        return original
    if strength <= 0:
        return original
    return restored
```

"Apply Signature" refreshes the visible inputs. Inputs are lazy by default.

### 8.1 Language

Statements: assignment, `if / elif / else`, `for x in <bounded iterable>`,
`break`, `continue`, `return`.

Expressions: the section 5 grammar.

Not available: `while`, `import`, `class`, `lambda`, `try`, `raise`,
`with`, `global`, `nonlocal`, `del`, decorators, comprehensions, `yield`,
`async`, dynamic code of any kind, recursion.

`for` iterates only `range(n)`, evaluator lists and tuples, and registered
sequence results. `for _ in range(limit): ... break` expresses every bounded
retry or search with the bound visible in the source. `while` is left out
for that reason, not for safety; the same design choice Starlark made, and
it can be revisited with evidence.

### 8.2 Budgets

Serialized on the node, so editor and API behave identically:

```
max loop iterations   default 1000      (shared across nested loops)
max interpreted ops   default 50000
max capability calls  default 5000
max collection size   default 10000
```

No setting means unlimited. An installation ceiling may lower any of them;
the effective limit is the minimum. Exceeding a budget stops the function
with the limit, the line, and the setting to change.

Capabilities may declare `cost`; a total-cost budget can be enforced later
without changing the registry shape.

### 8.3 Planning mode (laziness)

`check_lazy_status` runs the interpreter in planning mode: unresolved inputs
are `Unknown`; the first `Unknown` that control flow depends on stops
planning and names it (`needs: ["scale"]`). Inputs on untaken branches are
never named.

Planning is re-entered each time an input resolves, so planning mode never
executes a transform capability: every transform call yields `Unknown`. If
control flow then depends on it, that is a validation error ("transform
used in a branch decision"). Predicate-safe capabilities on resolved values
run in planning mode; they are cheap by declaration.

### 8.4 Inputs and outputs

Signature-driven inputs do not map onto Autogrow, whose template is one
type. Inputs are `AnyType` sockets named by the signature, validated in
`validate_inputs` against the declared types. Frontend type checking is
lost on this node; backend validation restores it. Rendering inputs from
the signature text needs custom JavaScript, so this node is not
frontend-free to build; it is frontend-free to run, which is what matters.

One declared output in v1.

### 8.5 Security acceptance (Gate E)

All of the following fail validation with a policy message:

```python
import os
open("/etc/passwd")
__import__("requests")
getattr(x, "__class__")
x.__class__
eval("1+1")
exec("...")
subprocess.run(...)
socket.socket(...)
(lambda: 0)()
[i for i in range(10)]
```

Hosting policy: administrators can disable Safe Function entirely, allow a
subset of packs, and lower the budgets. Disabling it never disables
Condition, Gate, Select, Filter or Partition.

---

## 9. LLM decisions (Phase 4)

The model is a predicate / selector producer feeding Gate and Lazy Select.
It is never inside the safe runtime, and it can only choose among branches
the author enumerated and fill arguments a schema validates. Prompt
injection riding in on an image or caption can pick the wrong branch; it
cannot acquire a capability. The section 2 boundary holds unchanged.

### 9.1 LLM Judge

```
prompt       STRING
images       IMAGE  optional (a contact sheet or sampled frames, not every frame)
schema       one of: BOOL | INT | FLOAT | STRING | JSON (schema text)
seed         INT    controls re-run only; part of the cache key
-> value (typed), raw STRING
```

Structured output is what makes this plug into Gate without a parse step:
JSON-schema constrained decoding on local OpenAI-compatible servers
(llama.cpp, vLLM, Ollama), `output_config.format` on the Anthropic API.

### 9.2 LLM Choose

Each Lazy Select case gets a name and a description on the node. They become
one strict tool per case, the model is forced to call one, the returned
tool name is the selector, and the parsed `input` becomes typed args via the
Safe Function signature mechanism. The branch subgraphs are the tools.

### 9.3 Loop

Tool result back to the model, repeat: a node-expansion loop with a
max-turns bound. After Judge and Choose exist.

### 9.4 Providers and keys

OpenAI-compatible endpoint first (covers local and most hosted), Anthropic
native second (strict tools, structured outputs). Keys and endpoints are
server-side configuration; workflow JSON never carries a key. `ui` reports
model id, request id, and the parsed decision.

Core's `Anthropic Claude`, OpenAI and Gemini nodes are text-out only; this
is not a duplicate.

---

## 10. Serialization and caching

Every feature survives save / load, copy / paste, templates, API export,
execution without the MAINodes frontend, and migration. Frontend decoration
never holds the only copy of execution semantics.

Cache identity includes expression or function source, signature, runtime
and pack versions, budgets, and scalar inputs, all of which are ordinary
inputs and therefore already in core's signature. Unselected lazy branches
never run for fingerprinting; core guarantees this.

Migration rule inherited from the pack: new inputs are appended last, never
inserted, or every saved workflow shifts a slot.

---

## 11. Acceptance gates

| Gate | Test |
|---|---|
| A. Real laziness | Flow Probe behind `processed`; expression false; probe count stays 0 |
| B. Passthrough | for IMAGE, MASK, LATENT, STRING, INT: expression false returns `source` unchanged |
| C. Exclusivity | Lazy Select with a probe per case: exactly one count increments |
| D. Headless parity | the example API graph run through the harness with no frontend branches identically |
| E. Expression security | the section 8.5 list, minus statements, rejected by the evaluator |
| F. Limits | oversized source, deep nesting, huge literals, exponent bombs rejected before execution |
| G. Lists | Filter / Partition counts; a Partition `kept` list of zero items sends the downstream probe count to 0 |
| H. No frontend | Gate D is the proof; no JavaScript exists in Phase 1 |

Test harness: a subprocess ComfyUI on a free localhost port with `--cpu`,
`--disable-all-custom-nodes`, an extra-model-paths file pointing
`custom_nodes` at a directory that symlinks this package under a private
name, and that name whitelisted. Results read from `/history`; run counts
from Flow Probe files. Never a production port, never the live checkout.

---

## 12. Phases

1. Expressions, registry, Gate, Condition, Lazy Select, Filter, Partition,
   Flow Probe, tests A-H, two example graphs (core-only logic tour; resize
   gate). **In progress.**
2. Run when / Skip when, passthrough mapping, compiled-flow reveal, dimming.
3. Safe Function: parser, validator, interpreter, planner, signature UI,
   transform capabilities, security tests.
4. LLM Judge, LLM Choose, then the bounded loop.
5. For Each over a subgraph body (node expansion), `image.select`, after
   real batch cases are collected.

---

## 13. Product principle

Beginner: "Run Resize when scale != 1". Intermediate: Condition into Gate
or Switch. Advanced: `def main(...)`. All three compile to the same idea:

> Only evaluate the part of the graph required to produce the selected
> result.

---

## 14. Style and repo rules for this package

Public repo. No community or operator names in code, docs, examples or
commit messages; describe the question, not the person. No em-dashes. No
self-assessment prose. Examples ship as API-form JSON plus editor-form JSON
once a browser has produced it; until then the README says which is which.

---

## 15. Open probes (answer before the design freezes)

1. Autogrow over a `MatchType` template (Lazy Select). Pass: grown slots
   share one resolved type in the frontend and lazy requests work per slot.
2. Autogrow over an `AnyType` template (Gate / Condition values).
3. Empty Comfy list into a node maps zero times (Partition per-item
   avoidance).
4. `NodeOutput(ui=...)` on a V3 node with a `MatchType` output round-trips
   to `/history`.
