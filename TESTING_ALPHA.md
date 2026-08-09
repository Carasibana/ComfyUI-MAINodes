# Alpha test checklist (2026-08-09 nodes)

Covers H3 Manual Hold Map, H3 Motion Editor (and its widget), H3 Segment
Crop, H3 Segment Splice, and the manual mask paths on H3 Motion
Composite / H3 V2V Init. The Python compile paths have been exercised via
the API; the interactive widget and the drag-in workflows need human
passes. Findings welcome as issues; symptom/dial pairs belong in
TUNING.md.

Setup: any ComfyUI with the H3 models working (0.30 and 0.31 front ends
both need coverage). Keep the browser dev console open; any red line
during widget use is a finding. All three graphs wire clip length from
one source node; drop to a short legal duration (56 or 73 frames at
24fps) while iterating. Reminder when reviewing saved clips elsewhere:
H3 audio is 32 kHz and some players decode that as silence; resample to
48 kHz AAC if you share files.

## 1. Targeted graph (motion_pipeline_targeted.json)

- [ ] First queue with the shipped range saves baseline / oraclemap /
      dilated / final, no errors, final plays clean
- [ ] The Manual Hold Map report (the price tag) is actually visible
      somewhere; if you had to wire your own text display, note it
- [ ] Two-pass rhythm: watch oraclemap, retype ranges to the real burst,
      requeue; baseline sampler must NOT re-run; dilated length and
      report shrink
- [ ] Syntax: multi-range, seconds ("0.5s-1.1s"), per-range ":hold",
      mixed; segments output echoes token-snapped (wider) spans
- [ ] Error UX: reversed range and out-of-bounds range produce readable
      messages, not a stack wall
- [ ] Gate mode semantics: with the oracle wired, your ranges choose
      WHERE and the oracle chooses hold strength (typed :hold values are
      ignored); unwire the oracle and your holds apply directly
- [ ] s_per_step set to a measured value gives a plausible minutes
      estimate
- [ ] Spatial branch: unmute, paint a background lasso mask, queue;
      with invert_mask on, the painted region returns to baseline
      timing; the seam (feather 64 smoothstep on a real edge) is
      invisible in playback

## 2. Editor graph (motion_pipeline_editor.json)

Cold load
- [ ] Node renders with toolbar, timeline, paint hint, price line; the
      raw editor_state textarea is hidden; console clean
- [ ] Save and reload the workflow before any queue; node intact

First queue (no blocks)
- [ ] Behaves like the classic adaptive pipeline (oracle passes through,
      mask is all ones)
- [ ] Widget populates: filmstrip, green jerk profile, "last run" report

Timeline
- [ ] Scrub on filmstrip/ruler; playhead and frame label track
- [ ] Drag on the block lane creates a block snapped to the token grid;
      "+ block" works at the playhead; snap toggle off allows free edges
- [ ] Move and resize via bracket handles (are the 4 px hit zones big
      enough?); multiple blocks get distinct colors; labels show
      "oracle"/"hN" and a star once painted
- [ ] Delete button and Delete key; drags under 2 frames create nothing
- [ ] Undo button and ctrl+z; ctrl+z inside the editor must NOT undo the
      ComfyUI graph, and editor keys must not trigger canvas hotkeys;
      clicking outside restores normal hotkeys

Painting
- [ ] Unpainted selected block shows the red tint and the "paint to
      narrow it" notice
- [ ] Brush paints with the circle cursor; erase removes; size slider
      works at both extremes; "clear frame" wipes only the current frame
- [ ] Playhead outside the selected block: painting no-ops with notice
- [ ] "paint: whole block" strokes persist across the block's frames
- [ ] Onion skin ghosts previous (blue) and next (green) frame strokes;
      toggle kills them
- [ ] Mouse wheel over the paint area steps frames without zooming the
      graph canvas; arrow keys step too

Dials and automation
- [ ] All per-block dials respond; hold changes move the price line
- [ ] "A" buttons open lanes for hold/feather/strength; click adds
      points, drag shapes, double-click deletes; values clamp to range
      and block span; the button lights when an envelope exists
- [ ] Known cosmetic: pre-queue price estimates oracle blocks at hold 4,
      so it can overshoot the post-run report

Edit and requeue (the core promise)
- [ ] After an edit, only the regeneration side re-runs; if the baseline
      sampler re-runs, that is the P1 finding
- [ ] Painted region regenerates, unpainted stays baseline; block edges
      ease via the fade dial with no hard pop
- [ ] Trap to confirm and document: adding ANY block gates the oracle,
      so bursts outside your blocks are not de-roped
- [ ] invert_mask and outside_blocks=regenerated behave as labeled

Persistence
- [ ] Save + browser refresh mid-edit: blocks/strokes/envelopes survive;
      the filmstrip may be blank until the next queue (temp thumbnails);
      confirm it degrades to the hint, not broken images
- [ ] After a ComfyUI restart, the first queue repopulates everything

## 3. Segment graph (motion_pipeline_editor_segment.json)

- [ ] First queue with an empty editor crops to the oracle's window plus
      handles; the crop report states the speedup ratio
- [ ] A tight block around the burst improves the ratio and the wall
      time vs the plain editor graph
- [ ] Video seam: step frames at both window boundaries; look for tone
      or brightness drift across the crossfade; compare feather_frames
      0 vs 6
- [ ] Audio seam: headphones at both splice points; no clicks; total
      duration matches baseline; sync holds after the splice
- [ ] Edge cases: block touching frame 0 or the last frame; two separate
      bursts produce ONE window spanning both (current behavior); the
      report should make that obvious

## Triage

- P1: baseline re-runs on edit; state loss on save/reload; crashes;
  unreadable errors on common mistakes
- P2: interaction jank, price/report confusion, seams below the playback
  bar
- P3: cosmetics (canvas text is not hidpi-scaled yet, known)
