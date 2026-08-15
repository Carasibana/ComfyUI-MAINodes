"""The verbs. One implementation of every operation in this subsystem.

Three surfaces call this module and nothing else: cli.py for a shell, mcp.py
for an agent, nodes.py for a graph. None of them holds a rule the others
cannot see, which is what makes them interchangeable. If you find yourself
writing logic in a surface, it belongs here.

Every verb takes plain arguments and returns a plain JSON-able dict. No
tensors cross this boundary, no ComfyUI types, no torch. That is not
fastidiousness: it is what lets an agent drive the whole subsystem over a
pipe, and what lets the tests run in a process with no GPU in it.

Verbs that are not implemented yet RAISE. They do not return an empty
result that looks like a measurement. A research tool that quietly returns
nothing is worse than one that refuses, because the nothing gets recorded.
"""
from __future__ import annotations

import os
from typing import Optional

from concept_lab import types as T
from concept_lab.space import (VC_WARNING, Space, SpaceError, open_space,
                               under_version_control)


class NotYetImplemented(RuntimeError):
    """A verb whose task in the ladder has not been built.

    Carries the task id so the message says what has to happen first,
    rather than leaving the caller to guess whether they held it wrong.
    """

    def __init__(self, verb: str, task: str, why: str = ""):
        super().__init__(
            f"{verb} is not built yet (blocked on {task})."
            + (f" {why}" if why else ""))
        self.verb = verb
        self.task = task


def _space(index_root=None, tensor_root=None, create=False) -> Space:
    return open_space(index_root, tensor_root, create=create)


def _vc_warnings(s: Space) -> list:
    """The definitions-are-not-results warning, or nothing.

    Corpora, tap specs and plans are the part of this subsystem no tensor
    can reconstruct. Every verb that writes one says so when they are
    landing outside version control (DEC-CL-0016).
    """
    return [] if under_version_control(s.index_root) else [VC_WARNING]


# ------------------------------------------------------------ the space
def space_init(index_root: Optional[str] = None,
               tensor_root: Optional[str] = None,
               label: str = "") -> dict:
    """Create a workspace. Idempotent: an existing space is returned as-is."""
    s = Space(index_root, tensor_root)
    existed = s.exists
    header = s.init(label=label)
    return {"created": not existed, **header}


def space_status(index_root: Optional[str] = None,
                 tensor_root: Optional[str] = None) -> dict:
    return _space(index_root, tensor_root).status()


def space_doctor(index_root: Optional[str] = None,
                 tensor_root: Optional[str] = None) -> dict:
    return _space(index_root, tensor_root).doctor()


def space_export(out_path: str, corpora=None, captures=None, deltas=None,
                 packs=None, include_private: bool = False,
                 note: str = "", index_root: Optional[str] = None,
                 tensor_root: Optional[str] = None) -> dict:
    """Bundle a selection of this space into one portable zip.

    Selections default to everything local. Private packs are held back
    unless include_private is set, and every exclusion is named in the
    result with its reason.
    """
    return _space(index_root, tensor_root).export(
        out_path, corpora=corpora, captures=captures, deltas=deltas,
        packs=packs, include_private=include_private, note=note)


def space_import(bundle_path: str, index_root: Optional[str] = None,
                 tensor_root: Optional[str] = None) -> dict:
    """Land somebody else's exported space beside yours, origin intact."""
    return _space(index_root, tensor_root).import_bundle(bundle_path)


def space_list(kind: str, index_root: Optional[str] = None,
               tensor_root: Optional[str] = None) -> dict:
    s = _space(index_root, tensor_root)
    return {"space_id": s.space_id, "kind": kind, "records": s.list(kind)}


# ----------------------------------------------------------- the corpus
def corpus_new(name: str, notes: str = "", index_root: Optional[str] = None,
               tensor_root: Optional[str] = None) -> dict:
    """Start an empty experiment design table."""
    s = _space(index_root, tensor_root, create=True)
    corpus = T.ConceptCorpus(name=name, rows=(), origin=s.space_id,
                             notes=notes)
    path = s.put("corpora", name, corpus)
    return {"name": name, "corpus_id": corpus.corpus_id, "path": path,
            "n_rows": 0, "warnings": _vc_warnings(s)}


def corpus_add(name: str, source_kind: str, source_ref: str,
               labels: Optional[dict] = None, split: str = "extract",
               matched_group: Optional[str] = None,
               preprocess: Optional[dict] = None, notes: str = "",
               source_id: Optional[str] = None,
               index_root: Optional[str] = None,
               tensor_root: Optional[str] = None) -> dict:
    """Add one labelled observation.

    The source is hashed if it is a readable file. A row whose bytes cannot
    be hashed is still allowed (a URL, a not-yet-fetched asset) but says so,
    because content hash is what makes a corpus reproducible and a missing
    one should be visible rather than assumed.

    `source_id` is the shoot/take/source-file lineage: two clips cut from
    one take are different bytes and the same observation, and only a
    lineage id catches that when they land in different splits.
    """
    s = _space(index_root, tensor_root, create=True)
    d = s.get("corpora", name)
    corpus = T.ConceptCorpus.from_dict(d)

    content_hash = None
    hashed = False
    if source_ref and os.path.isfile(source_ref):
        content_hash = T.file_sha256(source_ref)
        hashed = True

    src = T.ConditioningSource(kind=source_kind, ref=source_ref,
                               content_hash=content_hash,
                               preprocess=preprocess or {})
    row = T.ConceptCorpusRow(source=src, labels=labels or {}, split=split,
                             matched_group=matched_group,
                             source_id=source_id, notes=notes)
    rows = list(corpus.rows) + [row.to_dict()]
    updated = T.ConceptCorpus(name=corpus.name, rows=tuple(rows),
                              created=corpus.created, origin=corpus.origin,
                              notes=corpus.notes)
    path = s.put("corpora", name, updated, allow_replace=True)
    return {"name": name, "row_id": row.row_id, "hashed": hashed,
            "content_hash": content_hash, "source_id": source_id,
            "n_rows": len(rows),
            "corpus_id": updated.corpus_id, "path": path,
            "warnings": _vc_warnings(s)}


def corpus_status(name: str, index_root: Optional[str] = None,
                  tensor_root: Optional[str] = None) -> dict:
    s = _space(index_root, tensor_root)
    corpus = T.ConceptCorpus.from_dict(s.get("corpora", name))
    by_split = {k: len(v) for k, v in corpus.by_split().items()}
    groups = {k: len(v) for k, v in corpus.matched_groups().items()}
    unpaired = sorted(g for g, n in groups.items() if n < 2)
    contain = corpus.containment()
    return {
        "name": corpus.name, "corpus_id": corpus.corpus_id,
        "n_rows": len(corpus.rows), "by_split": by_split,
        "label_keys": corpus.label_keys(),
        "matched_groups": groups,
        "singleton_groups": unpaired,
        "containment": contain,
        "warnings": (_corpus_warnings(by_split, unpaired)
                     + _containment_warnings(contain) + _vc_warnings(s)),
    }


def _containment_warnings(contain: dict) -> list:
    """One line per source that sits in two splits.

    Named per value rather than counted: "3 collisions" sends someone
    hunting, a hash and two split names sends them to the row.
    """
    out = []
    for key in ("content_hash", "source_id"):
        for c in contain.get(key, ()):
            out.append(
                f"CONTAINMENT: {key} {c['value']} appears in splits "
                f"{c['splits']} (rows {c['rows']}). A held-out score over a "
                f"row the extraction set already saw is an extraction score")
    return out


def _corpus_warnings(by_split: dict, singleton_groups: list) -> list:
    """The things a corpus is usually missing, said before anyone spends GPU.

    These are protocol warnings, not errors. A corpus with no adversarial
    rows can still be captured; it just cannot support a factor claim, and
    finding that out after the capture is the expensive way.
    """
    out = []
    if not by_split.get("validate"):
        out.append("no VALIDATE rows: a factor fitted here cannot be shown "
                   "to survive unseen examples of the same concept")
    if not by_split.get("adversarial"):
        out.append("no ADVERSARIAL rows: nothing will catch a factor that "
                   "is really tracking background, camera or a near-neighbour "
                   "gesture")
    if not by_split.get("null"):
        out.append("no NULL rows: false activation and collateral drift "
                   "cannot be measured without examples that contain no "
                   "target concept")
    if singleton_groups:
        out.append(f"matched groups with one member: {singleton_groups}. "
                   "A matched contrast needs both arms or the subtraction "
                   "has nothing to cancel against")
    return out


def corpus_confounds(name: str, target: str,
                     index_root: Optional[str] = None,
                     tensor_root: Optional[str] = None) -> dict:
    """Which labels move exactly with the target label.

    The cheapest honest thing this subsystem can tell you, and the one most
    worth running before a capture: if every waving clip also smiles, no
    solver recovers a wave factor from that corpus, and the fix is another
    clip rather than another algorithm.
    """
    s = _space(index_root, tensor_root)
    corpus = T.ConceptCorpus.from_dict(s.get("corpora", name))
    return corpus.confounds(target)


# ---------------------------------------------------------- the capture
def capture_plan(corpus: str, backend: str = "h3", blocks=None, steps=None,
                 step_quantiles=None, segments=None, store: str = "stats",
                 arms=("R", "N", "0"), seed: int = 0, replicates: int = 1,
                 allow_containment: bool = False,
                 index_root: Optional[str] = None,
                 tensor_root: Optional[str] = None) -> dict:
    """Work out exactly which forwards a three-arm capture would need.

    Runs no model. This is the costing and sanity step: it reports the
    forward count, the arms, the tap, and every protocol warning the corpus
    carries, so an expensive capture is entered deliberately.

    It REFUSES on split containment. A warning is the right weight for a
    missing adversarial split, which costs a claim later; the same source in
    two splits invalidates the held-out number itself, and this is the last
    moment before GPU time where that is free to fix. `allow_containment`
    exists for the deliberate exception and is recorded in the plan.
    """
    from concept_lab.backends import get_backend

    s = _space(index_root, tensor_root)
    c = T.ConceptCorpus.from_dict(s.get("corpora", corpus))
    be = get_backend(backend)

    contain = c.containment()
    collisions = [{"key": k, **col} for k in ("content_hash", "source_id")
                  for col in contain[k]]
    if collisions and not allow_containment:
        first = collisions[0]
        raise T.ContractError(
            f"corpus {corpus!r} leaks across splits: {first['key']} "
            f"{first['value']} appears in splits {first['splits']} "
            f"(rows {first['rows']}); {len(collisions)} collision(s) total. "
            f"A validate row the extraction set already contains cannot "
            f"show a factor survives anything. Fix the corpus, or pass "
            f"allow_containment=True to plan it anyway and have the plan "
            f"say so")

    tap = T.TapSpec(
        blocks=tuple(blocks) if blocks else tuple(be.default_blocks()),
        steps=tuple(steps) if steps else (),
        step_quantiles=tuple(step_quantiles) if step_quantiles else
        ((0.15, 0.5, 0.85) if not steps else ()),
        segments=tuple(segments) if segments else tuple(be.default_segments()),
        store=store)
    be.validate_tap(tap)

    n_rows = len(c.rows)
    n_arms = len(arms)
    n_steps = len(tap.steps) or len(tap.step_quantiles)
    forwards = n_rows * n_arms * replicates
    taps = forwards * n_steps * len(tap.blocks) * len(tap.segments)

    warn = _corpus_warnings({k: len(v) for k, v in c.by_split().items()},
                            sorted(g for g, v in c.matched_groups().items()
                                   if len(v) < 2))
    if "0" in arms and "N" not in arms:
        warn.append("arm '0' without arm 'N': R-0 mixes semantic content "
                    "with the presence and layout change a reference causes. "
                    "Keep the shape-matched control or the delta is confounded")
    warn += _containment_warnings(contain) + _vc_warnings(s)
    out = {
        "corpus": corpus, "corpus_id": c.corpus_id, "backend": backend,
        "arms": list(arms), "seed": seed, "replicates": replicates,
        "tap": tap.to_dict(),
        "forwards": forwards, "tap_points": taps,
        # Forwards are the unit everyone reaches for and the wrong one here:
        # a reference block roughly doubles the packed sequence and per-step
        # cost is superlinear in it, so two forwards are not two of anything.
        "cost": {"forwards": forwards, "cost_model": be.cost_model(),
                 "note": "forwards are not the cost unit on this backend; "
                         "price by tokens x steps"},
        "process_launch_id": T.process_fingerprint()["launch_id"],
        "protocol_notes": [
            "All arms of one comparison (R, N, 0 and their replicates for a "
            "delta) must be captured inside ONE process launch; a capture "
            "resumed in a new process is a new replicate and is not "
            "comparable row-for-row to arms from the old one.",
        ],
        "warnings": warn,
        "note": "capture_plan runs no model. Nothing here is a measurement.",
    }
    if collisions:
        out["containment_overridden"] = collisions
    return out


def arms_check(corpus: str, r_row_id: str, n_row_id: str,
               backend: str = "h3", index_root: Optional[str] = None,
               tensor_root: Optional[str] = None) -> dict:
    """Does the control row actually match the reference row's shape?

    `plan_arms` says what must match. This asserts it against two real rows
    before anyone spends the forwards, and refuses on the first disagreement
    naming the key and both values.

    A key nobody recorded is NOT a pass. The media was never probed, so it
    lands in `unverifiable` and says so: a control that silently differs in
    frame count turns R-N from semantic content into content plus layout,
    which is the exact confound the three-arm protocol exists to remove.
    """
    from concept_lab.backends import get_backend

    s = _space(index_root, tensor_root)
    c = T.ConceptCorpus.from_dict(s.get("corpora", corpus))
    rows = {r.row_id: r for r in c._rows()}
    for rid in (r_row_id, n_row_id):
        if rid not in rows:
            raise T.ContractError(
                f"corpus {corpus!r} has no row {rid!r}; it has "
                f"{sorted(rows)}")
    r_src = rows[r_row_id]._source()
    n_src = rows[n_row_id]._source()
    be = get_backend(backend)
    must_match = list(be.plan_arms(r_src)["arm_N"]["must_match"])

    r_pre = dict(r_src.preprocess or {})
    n_pre = dict(n_src.preprocess or {})
    checked, unverifiable = [], []
    for key in must_match:
        if key == "modality":
            a, b = r_src.kind, n_src.kind
        elif key in r_pre or key in n_pre:
            a, b = r_pre.get(key), n_pre.get(key)
        else:
            unverifiable.append(key)
            continue
        if a != b:
            raise T.ContractError(
                f"arms do not match on {key!r}: R row {r_row_id} has {a!r}, "
                f"N row {n_row_id} has {b!r}. R and N must produce identical "
                f"packed layouts, or R-N is not semantic content")
        checked.append({"key": key, "value": a})
    return {
        "corpus": corpus, "backend": backend,
        "r_row_id": r_row_id, "n_row_id": n_row_id,
        "must_match": must_match,
        "checked": checked,
        "unverifiable": unverifiable,
        "ok": True,
        "note": ("unverifiable keys were not recorded in either row's "
                 "preprocess; the media was not probed, so this is silence "
                 "rather than agreement"),
    }


def capture_record(manifest: dict, index_root: Optional[str] = None,
                   tensor_root: Optional[str] = None) -> dict:
    """File a manifest that a capture has ALREADY produced.

    The tap runs inside the ComfyUI process and writes its tensors as it
    goes (T4); this verb is the moment the record becomes evidence. It does
    not run a model, which is why it is allowed on the agent surface: it
    takes a dict somebody else measured and puts it in the bank under the id
    the dict's own contents compute to.

    The recompute is the point. A manifest may CLAIM a capture_id; if the
    claim disagrees with what the fields hash to, one of the two is wrong
    and filing it would put a lie in the index, so this refuses and prints
    both. Collision semantics are `Space.put`'s and unchanged: identical
    bytes are idempotent, different bytes under one id raise (DEC-CL-0007,
    narrowed by DEC-CL-0015).
    """
    claimed = (manifest or {}).get("capture_id")
    man = T.FunctionalCaptureManifest.from_dict(manifest)
    cid = man.capture_id
    if claimed and claimed != cid:
        raise T.ContractError(
            f"capture_record: the manifest claims capture_id {claimed!r} but "
            f"its own fields hash to {cid!r}. One of them is wrong; filing it "
            f"would key the bank on a claim rather than on the condition")
    s = _space(index_root, tensor_root, create=True)
    payload = man.to_dict()
    payload["capture_id"] = cid
    path = s.put("captures", cid, payload)
    return {"capture_id": cid, "path": path, "n_tensors": len(man.tensors),
            "stack_id": man.stack_id,
            "process_launch_id": (man.process or {}).get("launch_id"),
            "warnings": _vc_warnings(s)}


def capture_run(*a, **k):
    raise NotYetImplemented(
        "capture_run", "T4/T5",
        "The H3 block tap and the three-arm protocol are not built. "
        "capture_plan works today and costs nothing.")


def delta_build(*a, **k):
    raise NotYetImplemented(
        "delta_build", "T3",
        "Segment alignment has to refuse mismatched layouts before it is "
        "allowed to subtract anything.")


def factorize(*a, **k):
    raise NotYetImplemented(
        "factorize", "T6",
        "No strategy ships before its experiment. Consensus and matched "
        "delta arrive together with E1/E3.")


def pack_compile(*a, **k):
    raise NotYetImplemented(
        "pack_compile", "T9",
        "Compilation is blocked until a factor exists that survives its "
        "held-out and adversarial gates.")


# ------------------------------------------------------------- the pack
def pack_list(index_root: Optional[str] = None,
              tensor_root: Optional[str] = None) -> dict:
    s = _space(index_root, tensor_root)
    out = []
    for rec in s.list("packs"):
        try:
            d = s.get("packs", rec["id"], origin=rec["origin"])
        except SpaceError:
            continue
        out.append({"pack_id": rec["id"], "origin": rec["origin"],
                    "local": rec["local"], "name": d.get("name"),
                    "visibility": d.get("visibility"),
                    "status": d.get("status"),
                    "n_factors": len(d.get("factors", ()))})
    return {"space_id": s.space_id, "packs": out}


def pack_show(pack_id: str, origin: Optional[str] = None,
              index_root: Optional[str] = None,
              tensor_root: Optional[str] = None) -> dict:
    s = _space(index_root, tensor_root)
    return s.get("packs", pack_id, origin=origin)


VERBS = {
    "space_init": space_init,
    "space_status": space_status,
    "space_doctor": space_doctor,
    "space_export": space_export,
    "space_import": space_import,
    "space_list": space_list,
    "corpus_new": corpus_new,
    "corpus_add": corpus_add,
    "corpus_status": corpus_status,
    "corpus_confounds": corpus_confounds,
    "capture_plan": capture_plan,
    "arms_check": arms_check,
    "capture_record": capture_record,
    "capture_run": capture_run,
    "delta_build": delta_build,
    "factorize": factorize,
    "pack_compile": pack_compile,
    "pack_list": pack_list,
    "pack_show": pack_show,
}
