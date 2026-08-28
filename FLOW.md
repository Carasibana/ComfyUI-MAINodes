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

## Examples

`examples/flow/README.md` covers both shipped graphs. They are API form;
editor form is owed.
