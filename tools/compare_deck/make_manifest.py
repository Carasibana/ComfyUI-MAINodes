#!/usr/bin/env python3
"""Starter manifest for build_compare_page.py, from a directory of mp4s.

Point it at a folder of clips and it writes the boring half of a manifest:
one row per clip, arm from the filename stem, src (and poster) paths, plus
fps, duration and aspect probed with ffprobe. Metrics merge in by stem from
either a directory of one-JSON-per-clip or a single CSV.

It does NOT invent groups or descriptions. Both come out as empty strings
for a human to fill, because the grouping is the argument the deck is
making and a guessed one is worse than a blank.

  python3 make_manifest.py CLIPS_DIR manifest.json \\
      --title "de-rope block sweep" \\
      --posters posters/ --metrics metrics.csv \\
      --src-prefix media/derope --poster-prefix posters

Paths in the manifest are `<prefix>/<filename>`. The built page resolves
them relative to the HTML file, so pass the prefix as the page will see it
(default: the CLIPS_DIR / --posters values exactly as typed).
"""
import argparse, csv, json, os, subprocess, sys


def probe(path):
    """(fps, duration_s, width, height) for one file, via ffprobe."""
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=r_frame_rate,width,height",
           "-show_entries", "format=duration", "-of", "json", path]
    try:
        raw = subprocess.run(cmd, capture_output=True, text=True,
                             check=True).stdout
    except FileNotFoundError:
        sys.exit("ffprobe not found on PATH; it is required")
    except subprocess.CalledProcessError as e:
        sys.exit("ffprobe failed on %s: %s" % (path, e.stderr.strip()))
    d = json.loads(raw)
    if not d.get("streams"):
        sys.exit("no video stream in %s" % path)
    st = d["streams"][0]
    num, _, den = st.get("r_frame_rate", "0/1").partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 0.0
    dur = float(d.get("format", {}).get("duration") or 0.0)
    return fps, dur, int(st.get("width") or 0), int(st.get("height") or 0)


def load_metrics(path, key_col):
    """{stem: {metric: value}} from a CSV file or a directory of JSONs."""
    out = {}
    if os.path.isdir(path):
        for f in sorted(os.listdir(path)):
            if not f.endswith(".json"):
                continue
            with open(os.path.join(path, f), encoding="utf-8") as fh:
                d = json.load(fh)
            if not isinstance(d, dict):
                sys.exit("%s is not a flat JSON object" % os.path.join(path, f))
            out[os.path.splitext(f)[0]] = d
        return out
    with open(path, newline="", encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        if rd.fieldnames is None:
            sys.exit("%s has no header row" % path)
        kc = key_col or rd.fieldnames[0]
        if kc not in rd.fieldnames:
            sys.exit("column %r not in %s (has %s)"
                     % (kc, path, ", ".join(rd.fieldnames)))
        for r in rd:
            stem = os.path.splitext(str(r[kc]).strip())[0]
            out[stem] = {k: v for k, v in r.items() if k != kc}
    return out


def coerce(v):
    """CSV and JSON both arrive stringly; numbers sort and bar, strings do not."""
    if isinstance(v, (int, float)) or v is None:
        return v
    s = str(v).strip()
    if s == "":
        return None
    try:
        return int(s) if s.lstrip("+-").isdigit() else float(s)
    except ValueError:
        return s


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("clips_dir", help="directory of clips (probed, not copied)")
    p.add_argument("out_json", help="manifest to write")
    p.add_argument("--glob-ext", default=".mp4",
                   help="clip extension to pick up (default .mp4)")
    p.add_argument("--posters", metavar="DIR",
                   help="directory of posters; matched by stem, .jpg then .png")
    p.add_argument("--metrics", metavar="PATH",
                   help="a CSV file, or a directory of one flat JSON per clip; "
                        "merged onto rows by filename stem")
    p.add_argument("--metrics-key", metavar="COL",
                   help="CSV column holding the clip stem (default: first column)")
    p.add_argument("--src-prefix", metavar="P",
                   help="path prefix for src in the page (default: CLIPS_DIR)")
    p.add_argument("--poster-prefix", metavar="P",
                   help="path prefix for poster in the page (default: --posters)")
    p.add_argument("--title", default="compare deck", help="page title")
    p.add_argument("--h1", help="page heading (default: --title)")
    p.add_argument("--lede-html", default="",
                   help="one paragraph of intro HTML, verbatim")
    p.add_argument("--fps", type=float,
                   help="force the page clock instead of using the probed fps")
    p.add_argument("--aspect",
                   help='force aspect, e.g. "16/9" (default: probed from the '
                        "first clip)")
    p.add_argument("--force", action="store_true",
                   help="overwrite out_json if it exists")
    a = p.parse_args()

    if os.path.exists(a.out_json) and not a.force:
        sys.exit("%s exists; pass --force to overwrite" % a.out_json)
    if not os.path.isdir(a.clips_dir):
        sys.exit("%s is not a directory" % a.clips_dir)
    names = sorted(f for f in os.listdir(a.clips_dir) if f.endswith(a.glob_ext))
    if not names:
        sys.exit("no %s files in %s" % (a.glob_ext, a.clips_dir))

    metrics = load_metrics(a.metrics, a.metrics_key) if a.metrics else {}
    src_prefix = a.clips_dir if a.src_prefix is None else a.src_prefix
    poster_prefix = a.posters if a.poster_prefix is None else a.poster_prefix

    rows, fps_seen, aspect, mkeys = [], [], None, []
    for name in names:
        stem = os.path.splitext(name)[0]
        fps, dur, w, h = probe(os.path.join(a.clips_dir, name))
        fps_seen.append(fps)
        if aspect is None and w and h:
            aspect = "%d/%d" % (w, h)
        row = {"arm": stem, "group": "", "desc": "",
               "src": os.path.join(src_prefix, name).replace(os.sep, "/"),
               "dur_s": round(dur, 3)}
        if a.posters:
            for ext in (".jpg", ".png"):
                if os.path.exists(os.path.join(a.posters, stem + ext)):
                    row["poster"] = os.path.join(
                        poster_prefix, stem + ext).replace(os.sep, "/")
                    break
            else:
                print("no poster for %s" % stem, file=sys.stderr)
        for k, v in (metrics.get(stem) or {}).items():
            row[k] = coerce(v)
            if k not in mkeys:
                mkeys.append(k)
        if metrics and stem not in metrics:
            print("no metrics row for %s" % stem, file=sys.stderr)
        rows.append(row)

    uniq = sorted(set(round(f, 4) for f in fps_seen if f))
    if a.fps is None and len(uniq) > 1:
        sys.exit("clips disagree on fps (%s); pass --fps to pick one"
                 % ", ".join(str(u) for u in uniq))
    fps = a.fps if a.fps is not None else (uniq[0] if uniq else 24)

    cols = [["dur_s", "duration (s)", 2, 1]]
    cols += [[k, k, 2, 1] for k in mkeys]
    man = {"title": a.title, "h1": a.h1 or a.title, "lede_html": a.lede_html,
           "aspect": a.aspect or aspect or "16/9",
           "fps": int(fps) if float(fps).is_integer() else fps,
           "media": "", "cols": cols, "groups": [], "rows": rows}
    with open(a.out_json, "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=1)
        fh.write("\n")
    print("wrote %s (%d rows, fps %s, aspect %s, %d metric cols); "
          "fill in group and desc before building"
          % (a.out_json, len(rows), man["fps"], man["aspect"], len(mkeys)))


if __name__ == "__main__":
    main()
