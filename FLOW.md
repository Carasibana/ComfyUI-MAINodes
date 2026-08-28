# Flow control

Conditional execution for ComfyUI graphs: run the expensive part of a graph
only when it is needed, and pass the original through when it is not.

> Only evaluate the part of the graph required to produce the selected
> result.

Phase 1 is backend only. There is no JavaScript in this feature, so a graph
behaves the same in the editor and through the API.

## Read this first: what makes a branch skip

A branch is skipped only when the guarded node is its EXCLUSIVE consumer.
Attach a preview, a save, or a second gate to the same producer and the
producer runs, because something else asked for it. This is the one trap
everybody hits. `Flow Probe` exists so you can count executions instead of
guessing: it appends a line to `<output>/flow_probe/<name>.count` every
time it runs, and it reports the running count on the node.

## What core already does, and you should use

| Need | Core node |
|---|---|
| Two way lazy branch on any type | `If/Else Switch` |
| Branch on "is this input connected" | `Soft Switch` |
| Scalar expression to BOOL / INT / FLOAT | `Math Expression` |
| Boolean glue | `Not`, `And`, `Or`, `Invert Boolean`, `String Compare` |

"Resize only when scale is not 1" is already three core nodes:
`Math Expression("a != 1.0")` into `If/Else Switch`. Those logic nodes are
flagged experimental, so turn on the frontend setting for experimental
nodes to find them. `examples/flow/core_logic_tour_api.json` is that graph.

This package adds what core lacks: predicates over data rather than widget
scalars, a fused "process if" that is one node, N way lazy dispatch, list
Filter and Partition, and the probe.

## The nodes

**Gate (process if)**. `processed if expression else source`. Both inputs
are lazy and exactly one of them is requested, so the untaken side never
runs. Connect the original to `source`, the expensive chain to `processed`,
and the values the expression names to `a`, `b`, ... The node reports which
side it took, the expression, and the values it saw.

**Condition**. The same expression, out as BOOL, FLOAT, INT and STRING, for
feeding `If/Else Switch`, `Lazy Select`, or anything else scalar.

**Lazy Select**. Up to eight cases plus a `default`, chosen by index or by
a name from the `labels` list. Only the selected case is produced.

**Filter (list)** and **Partition (list)**. Per item predicates over a
Comfy list, with `item`, `index` and `count` bound in the expression.
Laziness over a list is all or nothing per input, never per item, so the
way to avoid per item work is to Partition and feed the expensive branch
only `kept`.

**Flow Probe**. Passthrough plus a counter. Vary `salt` when you want a
second run to be counted rather than served from the result cache.

## Expressions

A strict superset of `Math Expression`, so an expression moves between the
two nodes unchanged. Available: literals, names, `not`, `- + * / // % **`,
chained comparisons, `and` / `or`, `x if c else y`, list and tuple
literals, integer subscripts, and calls to registered capabilities.

Bare names are core's set: `sum min max abs round pow sqrt ceil floor log
log2 log10 sin cos tan int float`, plus `clamp between near coalesce
is_none`.

Dotted names read data:

| Pack | Capabilities |
|---|---|
| `image` | `width height megapixels batch aspect` |
| `mask` | `coverage is_empty` |
| `latent` | `frames shape` |
| `seq` | `length` |

```
image.megapixels(a) > 2.0
mask.coverage(m) > 0.01 and latent.frames(l) >= 39
between(a, 0.9, 1.1)
```

Not available, with the policy in the error message: attribute access on a
value, lambda, comprehensions, f strings, walrus, starred arguments, `__`
anywhere in a name, and any name that is neither a connected value nor a
registered capability. Values that are not scalars are opaque inside an
expression; only a capability can open one. Limits are 4 KB of source,
2000 syntax nodes, depth 32, 256 calls per evaluation, 64 element literals
and an exponent cap of 4000, all refused before the graph runs.

## Two behaviours of the runtime worth knowing

Measured on core 0.33.0, both of them things a graph can hit.

* A grown input (the `a`, `b`, ... slots) cannot be lazy. The executor
  reads laziness from the unexpanded input schema, where the grown names do
  not exist, so it makes them strong links. That is why `Lazy Select` has
  fixed case slots rather than growing ones.
* An empty Comfy list does not make a downstream node run zero times. A
  node with any other input fails with an IndexError, and a node whose only
  input is the empty list is called once with default arguments. Guard an
  empty `kept` with a Gate on `kept_count` rather than relying on the list
  being empty.

## Safe Function

A node whose body is a restricted, Python-like function. There is no
`eval`, no `exec` and no imports: the body is parsed, checked against an
allowlist, and run by an interpreter that can only call registered
capabilities.

```python
def main(original: IMAGE, restored: IMAGE, enabled: BOOL = True, strength: FLOAT = 1.0) -> IMAGE:
    if not enabled:
        return original
    if strength <= 0:
        return original
    return restored
```

**Parameters bind to the sockets a..l positionally.** The first parameter
reads socket `a`, the second `b`, and so on, whatever they are called in the
source: the names are for the author, and API form always uses the letters,
so the node runs with no frontend at all. A parameter with a default may
leave its socket unconnected. Annotations (`IMAGE`, `MASK`, `LATENT`, `BOOL`,
`INT`, `FLOAT`, `STRING`) are checked against the value that arrives.

The sockets are lazy and the function plans them itself. It runs first with
the unconnected inputs unknown, stops at the first branch that depends on
one, asks for exactly that socket, and resumes. A socket on a branch the
body never reaches is never produced, so the example above never runs the
restore chain when `enabled` is false.

| Available | Not available |
|---|---|
| assignment to one plain name, plain or augmented (`x = 1`, `x += 1`) | `while` (use `for _ in range(limit)` with `break`) |
| `if` / `elif` / `else` | `import`, `class`, nested `def`, decorators |
| `for x in range(n)`, a list, a tuple, a capability result | `try`, `raise`, `with`, `global`, `nonlocal`, `del`, `assert` |
| `break`, `continue`, `return` | comprehensions, `lambda`, `yield`, f-strings, `%` on a string, `async` |
| the expression language above, plus transforms | recursion, attribute access, dynamic code of any kind |

Transforms (`image.resize`, `image.crop`, `image.flip`, `image.select`,
`mask.invert`, `mask.threshold`, `latent.blend`, `seq.concat`) are available
here and refused inside a Gate, Condition, Filter or Partition expression,
because those nodes decide a branch and a decision has to be cheap to plan.

Five budgets stop a runaway body, all five on the node so the editor and the
API behave the same: `max_iterations` (shared across nested loops),
`max_ops`, `max_calls`, `max_collection`, a running total of the sequence
elements and characters allocated, and `max_tensor_elements`, a running
total of the elements the transforms return. Two resources, two units, two
budgets: charged against one number, a single `image.flip` on a 64x64x3
image costs 12288 against a collection ceiling of 10000. Exceeding one names
the limit, the line and the setting to change. An installation can lower any
of them with a `flow_policy.json` at the pack root; the effective limit is
the lower of the two, there is no unlimited setting, and the shipped ceiling
sits well above the node default, so raising a budget on the node works and
only a ceiling somebody lowered makes the message point at the file.

Values are bounded as well as programs. A sequence refuses to grow past
`MAX_RESULT_LENGTH`, and an integer past `MAX_INT_BITS`, because `x = x * x`
in a loop reaches gigabytes of int in forty steps with every budget nearly
untouched and a bigint multiply is one uninterruptible call.


## Examples

`examples/flow/README.md` covers the shipped graphs. They are API form;
editor form is owed.
