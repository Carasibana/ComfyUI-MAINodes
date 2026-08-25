#!/usr/bin/env python3
"""Widget order is a compatibility surface: every node's INPUT_TYPES vs the
last published release.

  python tests/test_widget_order_vs_release.py
      Synthetic, no GPU, no models, no ComfyUI running (only a ComfyUI
      checkout on the path: COMFYUI_DIR, default /mnt/work/ai/apps/ComfyUI).

  python tests/test_widget_order_vs_release.py --regen
      REGENERATE THE BASELINE FIXTURE from the published tag. Extracts the
      tag's tree with `git show published-1.0.6:<file>` into a temp dir
      (the tag is NEVER checked out into a live tree), imports that pack,
      and rewrites tests/fixtures/widget_order_1.0.6.json. Do this only
      when a NEW release tag becomes the baseline: pass --tag published-X.Y.Z
      and commit the new fixture with the release.

WHY: a saved workflow stores widget VALUES positionally, in required-then-
optional declaration order. Insert an input anywhere but the end and every
older workflow silently shifts one slot - the values land on the wrong
widgets and the render is wrong without an error. So the pack's rule is
append-only.

What it checks, for every node id present in BOTH the baseline and the
current tree:

  1. REQUIRED is unchanged and in order. A new required key is a violation
     too, not just a reorder: required widgets precede optional ones, so
     appending one shifts every optional widget of that node.
  2. OPTIONAL is the baseline list as a prefix, plus appended keys only.
  3. A node that is in the baseline and GONE from the current tree FAILS
     loudly. Removing a published node is a release decision, never a
     silent default (it also fires if a module failed to import - the
     loader keeps the pack alive when an alpha module raises).
  4. A brand-new node id passes and is listed as a notice.
  Type changes on a shared key and hidden-input changes are NOTICES, not
  failures: they do not move widget slots.

Exit code 0 = pass.
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)
FIXTURE_DIR = os.path.join(HERE, "fixtures")
DEFAULT_TAG = "published-1.0.6"
COMFY = os.environ.get("COMFYUI_DIR", "/mnt/work/ai/apps/ComfyUI")

# What a pack needs to import. Media, docs and the web assets are not code
# and would only slow the extraction down.
EXTRACT_SUFFIXES = (".py", ".json", ".toml")
EXTRACT_SKIP_DIRS = ("assets/", "examples/", "docs/", "web/", "tests/")

FAILS = []
NOTICES = []


def check(name, ok, detail=""):
    print(("  PASS  " if ok else "  FAIL  ") + name + ("  " + detail if detail else ""))
    if not ok:
        FAILS.append(name)


def _type_name(spec):
    """A stable, printable name for an input's declared type."""
    if not isinstance(spec, (tuple, list)) or not spec:
        return repr(spec)[:32]
    t = spec[0]
    if isinstance(t, str):
        return t
    if isinstance(t, (list, tuple)):
        return "COMBO"          # choices are runtime file lists; not stable
    return type(t).__name__


def load_pack(root, modname):
    """Import a pack directory AS a package and return its NODE_CLASS_MAPPINGS.

    The directory name is not an importable identifier and the modules use
    relative imports, so bind the root to a name of our choosing.
    """
    if COMFY not in sys.path:
        sys.path.insert(0, COMFY)
    spec = importlib.util.spec_from_file_location(
        modname, os.path.join(root, "__init__.py"),
        submodule_search_locations=[root])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod.NODE_CLASS_MAPPINGS


def snapshot(mappings):
    """node id -> its INPUT_TYPES key order, per section."""
    out = {}
    for node_id, cls in mappings.items():
        t = cls.INPUT_TYPES()
        rec = {"class": f"{cls.__module__.split('.', 1)[-1]}.{cls.__name__}"}
        for section in ("required", "optional", "hidden"):
            d = t.get(section) or {}
            rec[section] = list(d)
            if section != "hidden":
                rec[section + "_types"] = {k: _type_name(v) for k, v in d.items()}
        out[node_id] = rec
    return out


def extract_tag(tag, dest):
    """Materialize the tag's code with `git show`, without touching a tree."""
    names = subprocess.run(["git", "-C", PACK, "ls-tree", "-r", "--name-only", tag],
                           capture_output=True, text=True, check=True).stdout.split()
    n = 0
    for name in names:
        if not name.endswith(EXTRACT_SUFFIXES):
            continue
        if any(name.startswith(d) for d in EXTRACT_SKIP_DIRS):
            continue
        blob = subprocess.run(["git", "-C", PACK, "show", f"{tag}:{name}"],
                              capture_output=True, check=True).stdout
        path = os.path.join(dest, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(blob)
        n += 1
    return n


def regen(tag, fixture):
    sha = subprocess.run(["git", "-C", PACK, "rev-list", "-n1", tag],
                         capture_output=True, text=True, check=True).stdout.strip()
    with tempfile.TemporaryDirectory(prefix="widget-order-") as tmp:
        n = extract_tag(tag, tmp)
        print(f"extracted {n} files of {tag} ({sha[:12]}) to {tmp}")
        nodes = snapshot(load_pack(tmp, "mainodes_baseline"))
    doc = {
        "_comment": ("Baseline INPUT_TYPES key order per node at the published "
                     "tag. Regenerate ONLY when a new release becomes the "
                     "baseline: python tests/test_widget_order_vs_release.py "
                     "--regen --tag <published-X.Y.Z>"),
        "tag": tag,
        "commit": sha,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "node_count": len(nodes),
        "nodes": dict(sorted(nodes.items())),
    }
    os.makedirs(FIXTURE_DIR, exist_ok=True)
    with open(fixture, "w") as fh:
        json.dump(doc, fh, indent=1, sort_keys=False)
        fh.write("\n")
    print(f"wrote {fixture}: {len(nodes)} nodes at {tag}")


def audit(fixture):
    with open(fixture) as fh:
        base = json.load(fh)
    baseline = base["nodes"]
    print(f"baseline: {base['tag']} ({base['commit'][:12]}), "
          f"{len(baseline)} nodes, generated {base['generated_utc']}")
    current = snapshot(load_pack(PACK, "mainodes_current"))
    print(f"current : {PACK}, {len(current)} nodes")
    print()

    shared = sorted(set(baseline) & set(current))
    gone = sorted(set(baseline) - set(current))
    new = sorted(set(current) - set(baseline))

    rows = []
    for node_id in shared:
        b, c = baseline[node_id], current[node_id]
        req_ok = b["required"] == c["required"]
        req_prefix = c["required"][:len(b["required"])] == b["required"]
        added_req = c["required"][len(b["required"]):] if req_prefix else []
        opt_prefix = c["optional"][:len(b["optional"])] == b["optional"]
        added_opt = c["optional"][len(b["optional"]):] if opt_prefix else []
        ok = req_ok and opt_prefix
        status = ("OK" if not (added_opt or added_req) else "OK +append") if ok else "VIOLATION"
        rows.append((node_id, status, len(b["required"]), len(c["required"]),
                     len(b["optional"]), len(c["optional"]),
                     added_req, added_opt, req_ok, req_prefix, opt_prefix, b, c))
        if b["hidden"] != c["hidden"]:
            NOTICES.append(f"{node_id}: hidden {b['hidden']} -> {c['hidden']}")
        for section in ("required", "optional"):
            bt, ct = b.get(section + "_types", {}), c.get(section + "_types", {})
            for k in b[section]:
                if k in ct and bt.get(k) != ct[k]:
                    NOTICES.append(f"{node_id}: {section} '{k}' type "
                                   f"{bt.get(k)} -> {ct[k]}")

    print(f"PER-NODE TABLE ({len(shared)} shared, {len(new)} new, {len(gone)} gone)")
    head = (f"{'node id':<26} {'status':<11} {'req':>7} {'opt':>7}  appended")
    print(head)
    print("-" * len(head))
    for (node_id, status, br, cr, bo, co, added_req, added_opt, *_rest) in rows:
        tail = []
        if added_req:
            tail.append("required+" + ",".join(added_req))
        if added_opt:
            tail.append("optional+" + ",".join(added_opt))
        print(f"{node_id:<26} {status:<11} {br:>3}->{cr:<3} {bo:>3}->{co:<3}  "
              + ("; ".join(tail) if tail else "-"))
    for node_id in new:
        c = current[node_id]
        print(f"{node_id:<26} {'NEW':<11} {'-':>3}->{len(c['required']):<3} "
              f"{'-':>3}->{len(c['optional']):<3}  brand new since {base['tag']}")
    for node_id in gone:
        b = baseline[node_id]
        print(f"{node_id:<26} {'GONE':<11} {len(b['required']):>3}->-   "
              f"{len(b['optional']):>3}->-    was in {base['tag']}")
    print()

    print("1. REQUIRED unchanged and in order")
    for row in rows:
        node_id, req_ok, req_prefix, b, c = row[0], row[8], row[9], row[11], row[12]
        if req_ok:
            continue
        kind = ("APPENDED required keys (they shift every optional widget)"
                if req_prefix else "REORDERED or REMOVED required keys")
        check(f"{node_id}: required keys unchanged and in order", False,
              f"{kind}: {b['required']} -> {c['required']}")
    if all(r[8] for r in rows):
        print(f"  PASS  all {len(rows)} shared nodes: required identical")

    print("\n2. OPTIONAL is baseline prefix + appends only")
    for row in rows:
        node_id, b, c = row[0], row[11], row[12]
        if row[10]:
            continue
        check(f"{node_id}: optional is the baseline order plus a tail",
              False, f"{b['optional']} -> {c['optional']}")
    if all(r[10] for r in rows):
        print(f"  PASS  all {len(rows)} shared nodes: optional prefix intact")
    appended = [(r[0], r[7]) for r in rows if r[7]]
    for node_id, keys in appended:
        print(f"        {node_id} appended {keys}")

    print("\n3. NO PUBLISHED NODE DISAPPEARED")
    check(f"every node id in {base['tag']} is still registered",
          not gone, ", ".join(gone) if gone else f"{len(baseline)} ids")

    print("\n4. BRAND-NEW NODES (notice only)")
    if new:
        for node_id in new:
            print(f"        {node_id}  ({current[node_id]['class']}, "
                  f"{len(current[node_id]['required'])} required / "
                  f"{len(current[node_id]['optional'])} optional)")
    else:
        print("        none")

    print("\nNOTICES (type / hidden changes; no widget slot moves)")
    for n in NOTICES or ["        none"]:
        print(f"        {n}" if n.strip() != "none" else n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regen", action="store_true",
                    help="rewrite the baseline fixture from the tag")
    ap.add_argument("--tag", default=DEFAULT_TAG)
    args = ap.parse_args()
    fixture = os.path.join(FIXTURE_DIR,
                           "widget_order_%s.json" % args.tag.replace("published-", ""))
    if args.regen:
        regen(args.tag, fixture)
        return 0
    if not os.path.exists(fixture):
        print(f"MISSING FIXTURE {fixture} - regenerate with --regen --tag {args.tag}")
        return 1
    audit(fixture)
    print()
    if FAILS:
        print(f"FAILED: {len(FAILS)} check(s): " + ", ".join(FAILS))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
