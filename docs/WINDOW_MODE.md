# Window mode: retiming an excerpt of a longer clip

Most of this pack retimes a whole clip. Window mode is the other shape: a
long clip passes through untouched except for one excerpt, which is
expanded on the model's time axis (`H3 Temporal Insert`, or a pixel-space
`H3 Time Smear`) and regenerated. It is the cheapest way to fix one burst
in a long shot, and it is the easiest graph in the pack to edit wrongly.

The window is not one number. It lives in FOUR coupled places, and all
four move together or none of them do.

## The four coupled parameters

1. **Prefix crop** (an `ImageFromBatch` before the processed excerpt).
   `length` is how many frames pass through untouched ahead of the
   window.
2. **Excerpt crop** (a second `ImageFromBatch`). `batch_index` is the
   prefix length, `length` is the excerpt in world frames. The excerpt
   must be a legal H3 length, `17k+5`, because it goes through the VAE.
3. **Hold map** (`H3 Temporal Insert`, or whatever writes the map).
   `holds` carries one entry PER WORLD FRAME of the excerpt, not per
   token, and the value is the local dilation. `world_len` is the excerpt
   length, and `sum(holds)` is the dilated frame count.
4. **Conditioning length** (the `MiniMaxH3ImageToVideo` family node
   feeding the regeneration). `length` here is `sum(holds)`, the DILATED
   total, and it must also be `17k+5`.

Number 4 is the one that gets forgotten, and it fails in a way that does
not point at itself: a latent longer than its conditioning renders
garbage, and every downstream mask is nearest-neighbour resampled to the
old proportions, so the visible artifact sits at the OLD window boundary.
If a window edit produces a broken render whose damage is parked where
the window used to be, this is the parameter to check first.

## Why "start at second X" quantizes

Write the excerpt as a head `h` at hold 1, a window `w` at hold `d`, and
a tail `t` at hold 1:

```
excerpt = h + w + t      must be 17k+5
dilated = h + d*w + t    must be 17k+5
```

Subtract them and `(d-1)*w = 0 mod 17`. At `d = 2` the window length `w`
has to be a multiple of 17 world frames, and then `h` is a multiple of 17
too. So window starts land on a 17-frame lattice, which at 24 fps is
0.708 s per rung. A request to start at second 1 lands on 0.71 s or on
1.42 s; there is no rung between them. Pick one and know which way you
rounded.

## Procedure

1. Write the target geometry down as `(prefix, h, w, t, d)` and check
   BOTH sums against `17k+5` before you touch the graph.
2. Edit all four literals in one pass. Assert the old value as you
   replace each one; that is what catches a graph that has drifted from
   the geometry you think it has.
3. If an endpoint is fixed ("it has to end on the same frame"), solve for
   the start rather than sliding a fixed-length window: the lattice moves
   both ends.
4. Prefer window starts on a 17-multiple (frames 0, 17, 34, ...).
   `H3 Temporal Insert` recovers inserted singleton tokens that land on
   those anchors exactly; off-anchor ones are its worst case.

`H3 Window Plan` and `H3 Window Collect` solve a different problem
(splitting one clip's dilated pass into budget-sized windows for a small
card) and do their own arithmetic. This page is about hand-built excerpt
graphs, where the four numbers are yours to keep consistent.
