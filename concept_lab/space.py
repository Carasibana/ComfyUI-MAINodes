"""The space: where a concept lab's evidence lives, and how it travels.

A space has two roots on purpose.

    index root    small JSON you want to see, browse, diff and read in a
                  file browser. Corpora, capture manifests, delta manifests,
                  pack manifests, and SPACE.json. Defaults next to the
                  renders, under the ComfyUI output directory.
    tensor root   the heavy evidence. Block states, sketches, factors.
                  Defaults to <index root>/tensors; point H3_CONCEPT_TENSOR_DIR
                  at a big disk on a box that has one.

Both are overridable per call and by environment:

    H3_CONCEPT_INDEX_DIR    the manifests
    H3_CONCEPT_TENSOR_DIR   the tensors

Neither default is inside the repository, and that is deliberate rather
than tidy: MAINodes is public, a concept pack can encode a likeness, and
the way that goes wrong is a research artifact sitting in a working tree
when somebody runs `git add -A`.

Export and import are the point of the two-root design. An export bundles
manifests and their tensors into one zip regardless of where they were
sitting; an import lands somebody else's space beside yours WITHOUT
pretending you measured it:

    <index root>/imported/<origin space id>/...
    <tensor root>/<origin space id>/...

Every imported record keeps its origin stamped. Their numbers can inform
your work and can never be mistaken for your own measurements, which
matters more here than usual, because the whole subsystem is a machine for
turning measurements into claims.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import zipfile
from typing import Optional

from concept_lab import types as T

INDEX_ENV = "H3_CONCEPT_INDEX_DIR"
TENSOR_ENV = "H3_CONCEPT_TENSOR_DIR"

DEFAULT_INDEX_REL = os.path.join("output", "h3_concept")
DEFAULT_TENSOR_ABS = ""          # no site default; env or <index root>/tensors

SPACE_FILE = "SPACE.json"
BUNDLE_SUFFIX = ".h3space.zip"

# Subdirectories of the index root. One kind of record per directory, so a
# human with a file browser can find things without a tool.
DIRS = ("corpora", "captures", "deltas", "packs", "plans", "imported")


class SpaceError(RuntimeError):
    pass


def _comfy_output_dir() -> Optional[str]:
    """ComfyUI's output directory if we happen to be running inside it.

    Guarded: concept_lab.space must import in a process with no comfy, and
    the test suite checks that. Outside ComfyUI this returns None and the
    caller falls back to the working directory, which is what a CLI run from
    the pack directory wants anyway.
    """
    try:
        import folder_paths                                  # noqa: PLC0415
        return folder_paths.get_output_directory()
    except Exception:
        return None


def default_index_root() -> str:
    env = os.environ.get(INDEX_ENV)
    if env:
        return os.path.abspath(os.path.expanduser(env))
    out = _comfy_output_dir()
    if out:
        return os.path.join(out, "h3_concept")
    return os.path.abspath(DEFAULT_INDEX_REL)


def default_tensor_root(index_root: str) -> str:
    env = os.environ.get(TENSOR_ENV)
    if env:
        return os.path.abspath(os.path.expanduser(env))
    if os.path.isdir(os.path.dirname(DEFAULT_TENSOR_ABS)):
        return DEFAULT_TENSOR_ABS
    return os.path.join(index_root, "tensors")


def under_version_control(path: str) -> bool:
    """Is `path` inside a git work tree? Walk up looking for a `.git` entry.

    Stdlib only and deliberately dumb: no subprocess, no git library, and a
    `.git` file (a worktree or submodule pointer) counts the same as a
    directory. It answers one question, "would a commit here keep these
    files", and it is allowed to be approximate in the safe direction.
    """
    p = os.path.abspath(os.path.expanduser(path))
    while True:
        if os.path.exists(os.path.join(p, ".git")):
            return True
        parent = os.path.dirname(p)
        if parent == p:
            return False
        p = parent


VC_WARNING = (
    "index root is not inside a git work tree. Corpora, tap specs and plans "
    "are DEFINITIONS, not results, and are not reconstructible from tensors; "
    "point H3_CONCEPT_INDEX_DIR at a directory inside a PRIVATE repository "
    "or copy corpora/ and plans/ into one (DEC-CL-0016).")


def _atomic_write(path: str, data: str) -> None:
    """Write via a temp file in the same directory, then rename.

    A half-written manifest is worse than a missing one: the missing one is
    obviously missing, and the half-written one gets parsed.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


class Space:
    """One workspace. Cheap to construct, does nothing until asked."""

    def __init__(self, index_root: Optional[str] = None,
                 tensor_root: Optional[str] = None):
        self.index_root = os.path.abspath(os.path.expanduser(
            index_root or default_index_root()))
        self.tensor_root = os.path.abspath(os.path.expanduser(
            tensor_root or default_tensor_root(self.index_root)))
        self._header: Optional[dict] = None

    # ------------------------------------------------------------ paths
    @property
    def space_file(self) -> str:
        return os.path.join(self.index_root, SPACE_FILE)

    def dir(self, kind: str, origin: Optional[str] = None) -> str:
        """Where records of `kind` live. `origin` routes imported records
        into their own namespace so ids from two spaces cannot collide."""
        if kind not in DIRS:
            raise SpaceError(f"unknown record kind {kind!r}, want {DIRS}")
        if origin and origin != self.space_id:
            return os.path.join(self.index_root, "imported", origin, kind)
        return os.path.join(self.index_root, kind)

    def tensor_dir(self, origin: Optional[str] = None) -> str:
        return os.path.join(self.tensor_root, origin or self.space_id)

    # ----------------------------------------------------------- header
    @property
    def exists(self) -> bool:
        return os.path.isfile(self.space_file)

    @property
    def header(self) -> dict:
        if self._header is None:
            if not self.exists:
                raise SpaceError(
                    f"no space at {self.index_root}. Run `space init` first: "
                    "a space is created deliberately, because creating one by "
                    "accident is how evidence ends up in two places.")
            self._header = _read_json(self.space_file)
        return self._header

    @property
    def space_id(self) -> str:
        return self.header["space_id"]

    def init(self, label: str = "", force: bool = False) -> dict:
        if self.exists and not force:
            return self.header
        header = {
            "space_id": T.new_space_id(),
            "label": label or os.path.basename(self.index_root),
            "created": T.now_iso(),
            "protocol_version": T.PROTOCOL_VERSION,
            "schema": T.SCHEMA_VERSION,
            "index_root": self.index_root,
            "tensor_root": self.tensor_root,
        }
        for d in DIRS:
            os.makedirs(os.path.join(self.index_root, d), exist_ok=True)
        os.makedirs(os.path.join(self.tensor_root, header["space_id"]),
                    exist_ok=True)
        _atomic_write(self.space_file, json.dumps(header, indent=2))
        self._header = header
        return header

    # ------------------------------------------------------ record i/o
    _EXT = {"corpora": ".corpus.json", "captures": ".capture.json",
            "deltas": ".delta.json", "plans": ".plan.json"}

    def _record_path(self, kind: str, rid: str,
                     origin: Optional[str] = None) -> str:
        if kind == "packs":
            return os.path.join(self.dir("packs", origin), T.safe_name(rid),
                                "manifest.json")
        return os.path.join(self.dir(kind, origin),
                            T.safe_name(rid) + self._EXT[kind])

    def put(self, kind: str, rid: str, record, origin: Optional[str] = None,
            allow_replace: bool = False) -> str:
        """Write a record. Same id with different content is an error.

        This is the collision check the capture bank depends on. A capture id
        names a condition; if the same condition produced different bytes,
        one of the two runs is not what its manifest says it is, and quietly
        keeping the newer one destroys the evidence that something is wrong.

        That reasoning holds WITHIN one process launch, where this pipeline
        is bit-identical. Across launches two honest reruns really do differ,
        which is why the launch id is part of a capture's identity
        (DEC-CL-0015): a replicate in a new interpreter gets its own id and
        never reaches this refusal.
        """
        path = self._record_path(kind, rid, origin)
        payload = record.to_dict() if hasattr(record, "to_dict") else record
        text = json.dumps(payload, indent=2, sort_keys=True,
                          ensure_ascii=False)
        if os.path.exists(path) and not allow_replace:
            old = _read_json(path)
            if T.canonical_json(old) != T.canonical_json(payload):
                raise SpaceError(
                    f"{kind}/{rid} already exists with DIFFERENT content. "
                    f"Same id, different bytes. Within one process launch "
                    f"this means the id does not describe the condition; if "
                    f"these came from different launches the capture is "
                    f"missing its process fingerprint (DEC-CL-0015). "
                    f"Refusing to overwrite {path}")
            return path
        _atomic_write(path, text)
        return path

    def get(self, kind: str, rid: str, origin: Optional[str] = None) -> dict:
        path = self._record_path(kind, rid, origin)
        if not os.path.exists(path):
            raise SpaceError(f"no {kind} record {rid!r} at {path}")
        return _read_json(path)

    def list(self, kind: str, include_imported: bool = True) -> list:
        """Every record of a kind, local first, each tagged with its origin."""
        out = []
        roots = [(self.space_id, self.dir(kind))]
        if include_imported:
            imp = os.path.join(self.index_root, "imported")
            if os.path.isdir(imp):
                for origin in sorted(os.listdir(imp)):
                    p = os.path.join(imp, origin, kind)
                    if os.path.isdir(p):
                        roots.append((origin, p))
        for origin, root in roots:
            if not os.path.isdir(root):
                continue
            if kind == "packs":
                for name in sorted(os.listdir(root)):
                    man = os.path.join(root, name, "manifest.json")
                    if os.path.isfile(man):
                        out.append({"id": name, "origin": origin,
                                    "path": man,
                                    "local": origin == self.space_id})
                continue
            ext = self._EXT[kind]
            for name in sorted(os.listdir(root)):
                if name.endswith(ext):
                    out.append({"id": name[:-len(ext)], "origin": origin,
                                "path": os.path.join(root, name),
                                "local": origin == self.space_id})
        return out

    # ---------------------------------------------------------- status
    def status(self) -> dict:
        h = dict(self.header)
        counts = {k: len(self.list(k)) for k in
                  ("corpora", "captures", "deltas", "packs", "plans")}
        imported = []
        imp = os.path.join(self.index_root, "imported")
        if os.path.isdir(imp):
            imported = sorted(os.listdir(imp))
        h["counts"] = counts
        h["imported_spaces"] = imported
        h["tensor_bytes"] = _tree_bytes(self.tensor_root)
        tracked = under_version_control(self.index_root)
        h["index_root_tracked"] = tracked
        h["warnings"] = [] if tracked else [VC_WARNING]
        return h

    def doctor(self) -> dict:
        """Everything that is inconsistent, said plainly.

        Checks tensor refs resolve, their bytes still hash to what the
        manifest claims, and capture ids agree with the manifests that carry
        them. A capture whose id does not recompute is the loudest possible
        signal that a schema changed under the space.

        Hashing is the point of the exercise: presence only proves a file is
        there, and a truncated or half-copied tensor is present. A ref that
        never carried a sha256 cannot be checked at all, which is a weaker
        problem than a wrong one, so it lands in `warnings` and does not
        clear `ok`.
        """
        problems, warnings = [], []
        for kind in ("captures", "deltas"):
            for rec in self.list(kind):
                d = _read_json(rec["path"])
                if kind == "captures":
                    try:
                        man = T.FunctionalCaptureManifest.from_dict(d)
                        if man.capture_id != rec["id"]:
                            problems.append({
                                "kind": "id_mismatch", "record": rec["path"],
                                "detail": f"filename says {rec['id']}, "
                                          f"recomputes to {man.capture_id}"})
                    except Exception as e:
                        problems.append({"kind": "unreadable",
                                         "record": rec["path"],
                                         "detail": f"{type(e).__name__}: {e}"})
                for t in d.get("tensors", ()):
                    p = os.path.join(self.tensor_dir(rec["origin"]),
                                     t.get("path", ""))
                    if not os.path.isfile(p):
                        problems.append({"kind": "missing_tensor",
                                         "record": rec["path"], "detail": p})
                        continue
                    want = t.get("sha256")
                    if not want:
                        warnings.append({
                            "kind": "unhashed_tensor",
                            "record": rec["path"], "detail": p,
                            "why": "no sha256 in the ref, so nothing about "
                                   "these bytes can be verified"})
                        continue
                    got = T.file_sha256(p)
                    if got != want:
                        problems.append({
                            "kind": "hash_mismatch", "record": rec["path"],
                            "detail": p,
                            "why": f"manifest says {want}, file hashes to "
                                   f"{got}"})
        problems.extend(self._unfinished_captures())
        return {"space_id": self.space_id, "problems": problems,
                "warnings": warnings, "ok": not problems}

    def _unfinished_captures(self) -> list:
        """`<tensor id>.partial` directories: captures that died mid-run.

        The tap streams tensors into `<id>.partial` and promotes the
        directory to `<id>` only after the manifest is filed (DEC-CL-0019),
        so a leftover partial is a capture whose bytes nothing references. It
        is a PROBLEM and not an error: the space is not corrupt, it is
        carrying wreckage, and the operator is the one who deletes it.
        """
        out = []
        if not os.path.isdir(self.tensor_root):
            return out
        for origin in sorted(os.listdir(self.tensor_root)):
            d = os.path.join(self.tensor_root, origin)
            if not os.path.isdir(d):
                continue
            for name in sorted(os.listdir(d)):
                p = os.path.join(d, name)
                if not name.endswith(".partial") or not os.path.isdir(p):
                    continue
                out.append({
                    "kind": "unfinished_capture", "record": p,
                    "detail": f"capture {name[:-len('.partial')]} in space "
                              f"{origin} wrote tensors and never filed a "
                              f"manifest",
                    "why": "the tap promotes <id>.partial to <id> only after "
                           "the manifest lands, so these bytes belong to a "
                           "run that died or was refused. Nothing references "
                           "them; delete the directory."})
        return out

    # ---------------------------------------------------------- export
    def export(self, out_path: str, corpora=None, captures=None, deltas=None,
               packs=None, include_private: bool = False,
               include_tensors: bool = True, note: str = "") -> dict:
        """Bundle a selection into one zip, tensors included.

        `None` for a selection means "everything of that kind that is local".
        Imported records are never re-exported: passing on somebody else's
        measurements under your space id is how provenance dies.

        Packs marked private are excluded unless include_private is set, and
        the exclusion is reported by name. Nothing here truncates silently:
        if a record is left out, EXPORT.json says which and why.
        """
        if not out_path.endswith(BUNDLE_SUFFIX):
            out_path += BUNDLE_SUFFIX
        sel = {"corpora": corpora, "captures": captures, "deltas": deltas,
               "packs": packs}
        included, excluded, tensors = [], [], []

        for kind, want in sel.items():
            have = {r["id"]: r for r in self.list(kind, include_imported=False)}
            ids = sorted(have) if want is None else list(want)
            for rid in ids:
                rec = have.get(rid)
                if rec is None:
                    excluded.append({"kind": kind, "id": rid,
                                     "reason": "not found in this space"})
                    continue
                d = _read_json(rec["path"])
                if kind == "packs" and d.get("visibility", "private") \
                        == "private" and not include_private:
                    excluded.append({
                        "kind": kind, "id": rid,
                        "reason": "visibility=private; pass include_private "
                                  "to export it anyway"})
                    continue
                included.append({"kind": kind, "id": rid, "path": rec["path"]})
                if include_tensors:
                    for t in d.get("tensors", ()):
                        rel = t.get("path")
                        if not rel:
                            continue
                        src = os.path.join(self.tensor_dir(), rel)
                        if os.path.isfile(src):
                            tensors.append((rel, src))
                        else:
                            excluded.append({
                                "kind": "tensor", "id": rel,
                                "reason": f"referenced by {kind}/{rid} but "
                                          f"not present at {src}"})

        manifest = {
            "bundle_schema": 1,
            "origin_space_id": self.space_id,
            "origin_label": self.header.get("label", ""),
            "exported": T.now_iso(),
            "protocol_version": T.PROTOCOL_VERSION,
            "note": note,
            "included": [{"kind": i["kind"], "id": i["id"]} for i in included],
            "excluded": excluded,
            "tensor_count": len(tensors),
        }

        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".",
                    exist_ok=True)
        with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("EXPORT.json", json.dumps(manifest, indent=2))
            z.writestr(SPACE_FILE, json.dumps(
                {k: v for k, v in self.header.items()
                 if k not in ("index_root", "tensor_root")}, indent=2))
            for i in included:
                if i["kind"] == "packs":
                    base = os.path.dirname(i["path"])
                    for root, _dirs, files in os.walk(base):
                        for f in files:
                            full = os.path.join(root, f)
                            arc = os.path.join(
                                "packs", i["id"],
                                os.path.relpath(full, base))
                            z.write(full, arc)
                else:
                    z.write(i["path"], os.path.join(
                        i["kind"], os.path.basename(i["path"])))
            for rel, src in tensors:
                z.write(src, os.path.join("tensors", rel))

        manifest["bundle"] = os.path.abspath(out_path)
        manifest["bytes"] = os.path.getsize(out_path)
        return manifest

    # ---------------------------------------------------------- import
    def import_bundle(self, bundle_path: str,
                      allow_replace: bool = False) -> dict:
        """Land another space beside yours, origin intact.

        Records go to imported/<their space id>/, tensors to
        <tensor root>/<their space id>/. Nothing is merged into your
        namespace and nothing is renamed. If they exported the same record
        twice with different bytes, the second one is refused, loudly, in the
        report rather than in a traceback halfway through.
        """
        if not os.path.isfile(bundle_path):
            raise SpaceError(f"no bundle at {bundle_path}")
        landed, refused = [], []
        with zipfile.ZipFile(bundle_path) as z:
            try:
                meta = json.loads(z.read("EXPORT.json"))
            except KeyError:
                raise SpaceError(
                    f"{bundle_path} has no EXPORT.json; not a concept space "
                    f"bundle")
            origin = meta.get("origin_space_id")
            if not origin:
                raise SpaceError("bundle does not name its origin space")
            if origin == self.space_id:
                raise SpaceError(
                    "this bundle came from THIS space. Importing it would "
                    "make local measurements look imported. Export to a "
                    "different space, or read the bundle directly.")
            proto = meta.get("protocol_version")
            if proto != T.PROTOCOL_VERSION:
                refused.append({
                    "kind": "protocol", "id": str(proto),
                    "reason": f"bundle protocol {proto} != local "
                              f"{T.PROTOCOL_VERSION}; ids are not comparable "
                              f"across protocol versions. Records still land, "
                              f"marked with their own protocol."})
            for name in z.namelist():
                if name in ("EXPORT.json", SPACE_FILE) or name.endswith("/"):
                    continue
                parts = name.split("/")
                head = parts[0]
                if head == "tensors":
                    dest = os.path.join(self.tensor_dir(origin),
                                        *parts[1:])
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    with z.open(name) as src, open(dest, "wb") as out:
                        shutil.copyfileobj(src, out)
                    landed.append({"kind": "tensor", "id": "/".join(parts[1:])})
                    continue
                if head not in DIRS:
                    refused.append({"kind": "unknown", "id": name,
                                    "reason": "not a record directory"})
                    continue
                dest = os.path.join(self.dir(head, origin), *parts[1:])
                payload = json.loads(z.read(name))
                if isinstance(payload, dict):
                    payload.setdefault("origin", origin)
                    payload["_imported_from"] = origin
                text = json.dumps(payload, indent=2, sort_keys=True,
                                  ensure_ascii=False)
                if os.path.exists(dest) and not allow_replace:
                    if T.canonical_json(_read_json(dest)) != \
                            T.canonical_json(payload):
                        refused.append({
                            "kind": head, "id": "/".join(parts[1:]),
                            "reason": "already imported with different "
                                      "content; refusing to overwrite"})
                        continue
                _atomic_write(dest, text)
                landed.append({"kind": head, "id": "/".join(parts[1:])})
        return {"origin_space_id": origin,
                "origin_label": meta.get("origin_label", ""),
                "bundle": os.path.abspath(bundle_path),
                "landed": landed, "refused": refused,
                "index_dir": os.path.join(self.index_root, "imported", origin),
                "tensor_dir": self.tensor_dir(origin)}


def _tree_bytes(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def open_space(index_root: Optional[str] = None,
               tensor_root: Optional[str] = None,
               create: bool = False, label: str = "") -> Space:
    s = Space(index_root, tensor_root)
    if create and not s.exists:
        s.init(label=label)
    return s
