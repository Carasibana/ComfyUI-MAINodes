# compare_deck

A standalone review page for a batch of clips. You hand it a JSON manifest
and it writes one self-contained HTML file: a card grid with hover-play and
audio on click, a sortable metric table with inline bars, a head to head A/B
that plays both sides in sync (side by side or as a wipe) with a waveform
timeline and loop brackets, starred picks you can copy back as markdown, and
an export that records the current A/B view to a video file.

The videos stay where they are. The page references them by path or URL and
nothing but the HTML is produced, so no media is ever committed here.

Live example: <https://matlowai.github.io/flipbook/trueclock.html> (33
clips, three metric columns, region spans marked on the timeline).

## Use

Two steps: draft a manifest from a folder of clips, fill in the argument the
deck is making, build the page.

```
python3 tools/compare_deck/make_manifest.py /path/to/clips deck.json \
    --title "block sweep" --posters /path/to/posters --metrics scores.csv \
    --src-prefix media/sweep --poster-prefix media/sweep/posters
# edit deck.json: fill group and desc on each row, label the metric columns
python3 tools/compare_deck/build_compare_page.py deck.json deck.html
```

`make_manifest.py` probes fps, duration and aspect with ffprobe, matches
posters by stem, and merges metrics by stem from either a CSV (first column
is the stem unless you pass `--metrics-key`) or a directory of one flat JSON
per clip. It leaves `group` and `desc` empty on purpose. Grouping is the
claim the deck makes and a guessed one reads as evidence, so it stays a
human step. `--force` is required to overwrite an existing manifest, and
clips that disagree on fps are an error until you pick one with `--fps`.

`build_compare_page.py` inlines the two vendored muxers (see
`vendor/NOTICE`) so the frame-exact export works with no network. Both
tools resolve their own paths from `__file__` and run from any cwd.

## Manifest

Top level:

| key | meaning |
|---|---|
| `title` | document title, also the localStorage key for stars |
| `h1`, `lede_html` | heading and one intro paragraph, HTML verbatim |
| `aspect` | CSS aspect ratio for the players, e.g. `"864/480"` |
| `fps` | the clock for frame numbers, spans, marks and export (default 24) |
| `media` | optional base path; rows may then omit `src`/`poster` and get `MEDIA/<arm>.mp4` and `.jpg` |
| `cols` | `[[key, label, decimals, sortdir], ...]`, the metric columns in table order |
| `groups` | the group filter's options; derived from the rows when empty |
| `rows` | one object per clip, below |

Per row: `arm` (the id, and the stem for derived paths), `group`, `desc`,
`src`, `poster`, any of the metric keys named in `cols`, and optionally
`curve` (a list of per-frame values drawn as a sparkline and on the
timeline), `ref_curve` (a second faint line to compare against), `color`
(dot colour, defaults by group), and `spans` (`[[start_frame, end_frame]]`,
shaded on the timeline and blipped on entry and exit).

The first row is the reference: it sorts to the top and both players load it
at startup. Rows in the `control` or `reference` group get the green dot.

Minimal example:

```json
{"title": "block sweep", "aspect": "864/480", "fps": 24,
 "cols": [["flash", "seam flash x", 2, 1]],
 "rows": [
   {"arm": "control", "group": "control", "desc": "stock, no change",
    "src": "media/control.mp4", "flash": 2.09, "spans": [[124, 157]]},
   {"arm": "pb4049", "group": "late-blocks", "desc": "physical in blocks 40-49",
    "src": "media/pb4049.mp4", "flash": 1.31, "spans": [[124, 157]]}]}
```

## Keys in the page

`space` play/pause, `f` swap A and B, `1`/`2` which side you hear, `i`/`o`
loop in and out at the playhead. The gear opens region frames, a frame
counter and mark blips. Export offers VP9 or AV1 webm and H.264 mp4; the
frame-exact path seeks and encodes every frame, so it is not a screen
recording and does not drop frames under load.
