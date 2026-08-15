"""The contracts. Stdlib only: no torch, no comfy, no H3, no numpy.

Everything in this subsystem keys on the ids minted here, so this file has
one job and has to be boring about it: given the same inputs, produce the
same id, on any machine, in any process, forever.

Two rules make that survivable as the schema grows.

1. An id is computed over an explicit IDENTITY PAYLOAD, never over the whole
   record. Adding a field to a manifest (a note, a diagnostic, a timestamp)
   must not move ids that already exist on disk, or every bank in the space
   goes stale for no scientific reason. If a new field genuinely belongs to
   the identity of a capture, it goes in identity() AND PROTOCOL_VERSION
   gets bumped in the same commit, deliberately, with a line in DECISIONS.md.

2. Floats are rounded to 12 significant digits before hashing. A sigma that
   arrives as float32 and a sigma that arrives as float64 name the same
   experimental condition, and an id that disagrees about that is an id that
   lies. 12 digits is far below the noise floor of anything we condition on
   and far above the precision anyone reports.

The model-neutral part of the project lives here. Segment names, block
counts, sampler details and layout rules belong to a backend: this file
treats them as opaque strings and numbers on purpose, so that the second
backend does not require a rewrite of the first one's evidence.
"""
from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import os
import platform
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

# Bumping this invalidates every id in the space. It is a scientific act,
# not a refactor: it says "records made before this line describe a
# different protocol than records made after it".
#
# 0.2.0 (2026-08-15): the process launch id joined the capture identity.
# See DEC-CL-0015: on this rig the pipeline is bit-identical WITHIN a
# process and merely close ACROSS processes, so a launch is a replicate
# dimension and two honest reruns are two conditions, not one.
PROTOCOL_VERSION = "concept_lab/0.2.0"

# Schema version of the manifests themselves. Free to move for additive,
# non-identity fields.
SCHEMA_VERSION = 1


# --------------------------------------------------------------- status
class Status:
    """Every claim carries one of these. Never promote quietly."""

    PROPOSED = "PROPOSED"
    MEASURED = "MEASURED"
    SCOPED = "SCOPED"
    FAILED = "FAILED"
    RETRACTED = "RETRACTED"
    SUPERSEDED = "SUPERSEDED"
    OPEN = "OPEN"

    ALL = (PROPOSED, MEASURED, SCOPED, FAILED, RETRACTED, SUPERSEDED, OPEN)


# The four splits. Extraction-set success is never sufficient for a claim,
# so the corpus cannot be built without somewhere to put the other three.
SPLITS = ("extract", "validate", "adversarial", "null")

# Control arms for a conditioning comparison.
#   R  the actual reference
#   N  a shape-matched control: same count, same dimensions, same modality,
#      same presentation length, different content
#   0  no reference at all
# R-N is the semantic content delta. N-0 is the price of presence. Both get
# recorded, because on H3 the second one is not small and not boring.
ARMS = ("R", "N", "0", "free")

# A concept pack can encode identity even with no source images in it, so
# visibility is a field on the artifact and not a matter of where the file
# happens to sit. Default private; export refuses private packs unless the
# caller says otherwise in as many words.
VISIBILITY = ("private", "shareable")


class ContractError(ValueError):
    """A record refused itself. Always says which field and why."""


# ------------------------------------------------------ canonical form
def _round_float(x: float) -> float:
    if x != x or x in (float("inf"), float("-inf")):    # NaN/inf survive as-is
        return x
    if x == 0:
        return 0.0
    return float(f"{x:.12g}")


def _normalize(o: Any) -> Any:
    """dataclasses -> dicts, tuples -> lists, floats -> 12 sig digits.

    Keys are left alone; sorting happens in the dump. Anything that is not
    JSON-able raises here rather than at write time, where the traceback
    would point at the wrong file.
    """
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return _normalize(dataclasses.asdict(o))
    if isinstance(o, dict):
        return {str(k): _normalize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_normalize(v) for v in o]
    if isinstance(o, bool) or o is None or isinstance(o, (int, str)):
        return o
    if isinstance(o, float):
        return _round_float(o)
    if isinstance(o, bytes):
        return base64.b64encode(o).decode("ascii")
    raise ContractError(f"not canonicalizable: {type(o).__name__}")


def canonical_json(o: Any) -> str:
    """The exact bytes we hash. Sorted keys, no whitespace, unicode kept."""
    return json.dumps(_normalize(o), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def digest(o: Any) -> str:
    """Full sha256 over the canonical form. Prefixed, so a hash in a
    manifest always says what kind of hash it is."""
    return "sha256:" + hashlib.sha256(
        canonical_json(o).encode("utf-8")).hexdigest()


def short_id(o: Any, n: int = 16) -> str:
    """A 16-hex handle for filenames and log lines. Never the only record
    of identity: the full digest lives in the manifest beside it."""
    return digest(o).split(":", 1)[1][:n]


def file_sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


_SAFE = re.compile(r"[^A-Za-z0-9._-]")


def safe_name(s: str, fallback: str = "unnamed") -> str:
    out = _SAFE.sub("_", str(s)).strip("_")
    return out or fallback


def new_space_id() -> str:
    return uuid.uuid4().hex[:16]


# ------------------------------------------------------ the process
# Minted once per interpreter and never again, which is the whole point:
# it is the only cheap handle that is unique across launches and constant
# within one. Kernel autotuning makes this rig bit-identical within a
# process and only ~41-48 dB apart across cold processes, so "which launch"
# is an experimental condition, not provenance trivia (DEC-CL-0015).
_LAUNCH_ID = uuid.uuid4().hex[:16]
_LAUNCH_STARTED = now_iso()


def process_fingerprint() -> dict:
    """Who and when, for one interpreter. Stdlib only, no torch.

    `launch_id` is the load-bearing key and the only field that enters a
    capture's identity. host/pid/started are provenance: a manifest has to
    be re-readable on another machine without its id moving.

    `torch` and `cuda` are left empty here on purpose. Reading them means
    importing torch, and this file must stay importable in a process that
    has none; the capture backend fills them at capture time (T4).
    """
    return {
        "host": platform.node(),
        "pid": os.getpid(),
        "started": _LAUNCH_STARTED,
        "python": platform.python_version(),
        "launch_id": _LAUNCH_ID,
        "torch": None,
        "cuda": None,
    }


# ---------------------------------------------------------- base record
@dataclass
class Record:
    """Shared serialization. Subclasses declare fields and identity()."""

    def to_dict(self) -> dict:
        d = _normalize(dataclasses.asdict(self))
        d["_type"] = type(self).__name__
        d["_schema"] = SCHEMA_VERSION
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False,
                          sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict):
        d = dict(d)
        d.pop("_type", None)
        d.pop("_schema", None)
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - names
        if unknown:
            # Forward compatibility is a choice, not an accident: a record
            # written by a newer schema is readable, and the extra fields
            # are dropped loudly rather than silently reinterpreted.
            for k in sorted(unknown):
                d.pop(k)
        return cls(**{k: v for k, v in d.items() if k in names})

    @classmethod
    def from_json(cls, s: str):
        return cls.from_dict(json.loads(s))

    def identity(self) -> dict:
        raise NotImplementedError


# ------------------------------------------------- model stack identity
@dataclass
class Adapter(Record):
    """A LoRA or distillation component in the loaded stack."""

    name: str
    strength: float = 1.0
    hash: Optional[str] = None

    def identity(self) -> dict:
        return {"name": self.name, "strength": self.strength,
                "hash": self.hash}


@dataclass
class ModelStackFingerprint(Record):
    """What was actually loaded when the numbers were taken.

    Not what the filename implies. The lesson this field set exists for:
    a research program once fit an edit under one model stack and deployed
    it under another, and the cross-arm comparisons stayed valid only
    because the wrong condition happened to be constant. Record the stack or
    the capture is a number without a condition.
    """

    checkpoint: str
    checkpoint_hash: Optional[str] = None
    vae: Optional[str] = None
    text_encoder: Optional[str] = None
    quantization: Optional[str] = None
    adapters: tuple = ()
    model_options: dict = field(default_factory=dict)
    comfy_version: Optional[str] = None
    comfy_commit: Optional[str] = None
    mainodes_commit: Optional[str] = None
    backend: str = "unknown"
    protocol_version: str = PROTOCOL_VERSION

    def identity(self) -> dict:
        return {
            "backend": self.backend,
            "checkpoint": self.checkpoint,
            "checkpoint_hash": self.checkpoint_hash,
            "vae": self.vae,
            "text_encoder": self.text_encoder,
            "quantization": self.quantization,
            "adapters": [a.identity() if isinstance(a, Adapter)
                         else Adapter.from_dict(a).identity()
                         for a in self.adapters],
            "model_options": self.model_options,
            "comfy_commit": self.comfy_commit,
            "mainodes_commit": self.mainodes_commit,
            "protocol_version": self.protocol_version,
        }

    @property
    def stack_id(self) -> str:
        return short_id(self.identity())

    def differs_from(self, other: "ModelStackFingerprint") -> list:
        """Which identity fields moved. This is what a staleness warning
        should print: 'stale' with no field name is not actionable."""
        a, b = self.identity(), other.identity()
        return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


# --------------------------------------------------- conditioning input
@dataclass
class ConditioningSource(Record):
    """Anything that changes the model's conditioning.

    Descriptive, not a tensor. It carries enough to recreate the condition
    and a content hash to prove which bytes were used. Source assets stay
    where the user put them: a concept pack references them by hash and does
    not copy them, which keeps packs small and keeps private material from
    riding along by accident.
    """

    kind: str                       # image | video | audio | text | keyframe | pack
    ref: Optional[str] = None       # path or uri, as given
    content_hash: Optional[str] = None
    preprocess: dict = field(default_factory=dict)   # resize/crop/fps/frames
    notes: str = ""

    KINDS = ("image", "video", "audio", "text", "keyframe", "pack")

    def __post_init__(self):
        if self.kind not in self.KINDS:
            raise ContractError(
                f"ConditioningSource.kind={self.kind!r} not in {self.KINDS}")

    def identity(self) -> dict:
        return {"kind": self.kind, "content_hash": self.content_hash,
                "preprocess": self.preprocess,
                # ref is included ONLY when there is no content hash: a path
                # is a weak identity and we would rather key on bytes.
                "ref": None if self.content_hash else self.ref}


@dataclass
class ConditionVariant(Record):
    """One arm of a controlled comparison.

    The name is yours; the arm is the protocol's. Keeping them separate lets
    a corpus carry five differently-built neutral controls that are all arm
    'N', which is exactly the sensitivity sweep OQ11 asks for.
    """

    name: str
    arm: str = "free"
    sources: tuple = ()
    prompt: Optional[str] = None
    notes: str = ""

    def __post_init__(self):
        if self.arm not in ARMS:
            raise ContractError(
                f"ConditionVariant.arm={self.arm!r} not in {ARMS}")

    def identity(self) -> dict:
        srcs = []
        for s in self.sources:
            s = s if isinstance(s, ConditioningSource) \
                else ConditioningSource.from_dict(s)
            srcs.append(s.identity())
        return {"arm": self.arm, "sources": srcs, "prompt": self.prompt}


# ------------------------------------------------------------- the tap
@dataclass
class TapSpec(Record):
    """Where to listen, and how much to keep.

    Blocks and steps are an experiment config and are NOT baked into any
    format: today's [0, 5, 10, 20, 30, 40, 49] is a guess that a measured
    depth profile is expected to replace.

    `segments` are backend segment names, opaque here. `store` is the memory
    ladder: stats first, full only where separability evidence has already
    earned the disk.
    """

    blocks: tuple = ()
    steps: tuple = ()                    # explicit sampler step indices
    step_quantiles: tuple = ()           # or schedule-relative positions
    segments: tuple = ()
    store: str = "stats"                 # stats | full | sketch
    dtype: str = "bf16"
    sketch_dim: Optional[int] = None
    sketch_seed: Optional[int] = None

    STORES = ("stats", "full", "sketch")

    def __post_init__(self):
        if self.store not in self.STORES:
            raise ContractError(
                f"TapSpec.store={self.store!r} not in {self.STORES}")
        if self.steps and self.step_quantiles:
            raise ContractError(
                "TapSpec: give steps OR step_quantiles, not both; a capture "
                "that can be selected two ways cannot be reproduced one way")
        if self.store == "sketch" and not self.sketch_dim:
            raise ContractError(
                "TapSpec.store='sketch' needs sketch_dim and sketch_seed: an "
                "unrecorded random projection is an unreproducible capture")

    def identity(self) -> dict:
        return {"blocks": list(self.blocks), "steps": list(self.steps),
                "step_quantiles": list(self.step_quantiles),
                "segments": list(self.segments), "store": self.store,
                "dtype": self.dtype, "sketch_dim": self.sketch_dim,
                "sketch_seed": self.sketch_seed}


@dataclass
class LayoutSignature(Record):
    """The packed layout a capture was taken under.

    This is the field that makes alignment refusable. Two captures may be
    subtracted row by row only when their signatures agree; otherwise the
    delta engine has to align by named segment or refuse outright. Absolute
    row index is never an alignment argument, because a reference block
    changes where the target stream starts.
    """

    segments: tuple = ()          # ((kind, start, stop), ...)
    total_rows: int = 0
    position_hash: Optional[str] = None

    def identity(self) -> dict:
        return {"segments": [list(s) for s in self.segments],
                "total_rows": self.total_rows,
                "position_hash": self.position_hash}

    @property
    def signature_id(self) -> str:
        return short_id(self.identity())

    def segment(self, kind: str):
        for s in self.segments:
            if s[0] == kind:
                return (int(s[1]), int(s[2]))
        return None

    def same_shape_as(self, other: "LayoutSignature") -> bool:
        return self.identity() == other.identity()


@dataclass
class TensorRef(Record):
    """One saved tensor, and the coordinates that make it mean something.

    Depth and time are first-class coordinates here, not metadata. A factor
    that gets collapsed to one global vector before anyone has measured the
    layer/timestep profile has thrown away the answer to the question.
    """

    key: str
    path: str                     # relative to the space's tensor root
    shape: tuple = ()
    dtype: str = "bf16"
    sha256: Optional[str] = None
    segment: Optional[str] = None
    block: Optional[int] = None
    step: Optional[int] = None
    sigma: Optional[float] = None
    t_video: Optional[float] = None
    kind: str = "state"           # state | stats | sketch | delta

    def identity(self) -> dict:
        return {"key": self.key, "shape": list(self.shape),
                "dtype": self.dtype, "segment": self.segment,
                "block": self.block, "step": self.step,
                "sigma": _round_float(self.sigma) if self.sigma is not None
                else None,
                "kind": self.kind}


# ---------------------------------------------------------- the capture
@dataclass
class FunctionalCaptureManifest(Record):
    """A reproducible record of selected internal states for one condition.

    capture_id = sha256(stack + variant + seed + sampler + tap + protocol),
    which is the key the bank refuses to overwrite with different content.
    """

    stack: Any                      # ModelStackFingerprint or its dict
    variant: Any                    # ConditionVariant or its dict
    tap: Any                        # TapSpec or its dict
    seed: int = 0
    sampler: dict = field(default_factory=dict)   # sampler/scheduler/steps/shift
    sigmas: tuple = ()              # what the model actually saw
    layout: Optional[Any] = None    # LayoutSignature, filled in at run time
    tensors: tuple = ()
    corpus_row_id: Optional[str] = None
    replicate: int = 0              # capture-reservoir replicate index
    # process_fingerprint() of the launch that measured this. Only
    # process["launch_id"] enters identity(); host, pid and started are
    # provenance, so re-reading a manifest somewhere else cannot move its id.
    process: dict = field(default_factory=dict)
    created: str = field(default_factory=now_iso)
    origin: Optional[str] = None    # space id that measured this
    status: str = Status.MEASURED
    notes: str = ""

    def _stack(self) -> ModelStackFingerprint:
        return self.stack if isinstance(self.stack, ModelStackFingerprint) \
            else ModelStackFingerprint.from_dict(self.stack)

    def _variant(self) -> ConditionVariant:
        return self.variant if isinstance(self.variant, ConditionVariant) \
            else ConditionVariant.from_dict(self.variant)

    def _tap(self) -> TapSpec:
        return self.tap if isinstance(self.tap, TapSpec) \
            else TapSpec.from_dict(self.tap)

    def identity(self) -> dict:
        return {
            "stack": self._stack().identity(),
            "variant": self._variant().identity(),
            "tap": self._tap().identity(),
            "seed": self.seed,
            "sampler": _normalize(self.sampler),
            "replicate": self.replicate,
            # The launch, and nothing else about the process. Two honest
            # reruns of the same condition in two interpreters differ in
            # bytes, so they are two captures and not a collision.
            "process_launch_id": (self.process or {}).get("launch_id"),
            "protocol_version": PROTOCOL_VERSION,
        }

    @property
    def capture_id(self) -> str:
        return short_id(self.identity())

    @property
    def stack_id(self) -> str:
        return self._stack().stack_id


@dataclass
class FunctionalDeltaManifest(Record):
    """A difference between two aligned captures.

    `alignment` is recorded because it is a claim: 'exact_rows' asserts the
    two layouts were identical, 'segment' asserts only that a named segment
    was matched. A delta whose alignment is unstated is not evidence.
    """

    a_capture_id: str
    b_capture_id: str
    alignment: str = "segment"      # exact_rows | segment
    segments: tuple = ()
    tensors: tuple = ()
    label: str = ""                 # e.g. "content" for R-N, "presence" N-0
    # Which launch each side came from. Non-identity provenance: a delta is
    # named by the two captures it subtracts, and those already carry their
    # launches. T3 will refuse alignment="exact_rows" when same_process() is
    # False, because across launches the two arms differ by autotuning as
    # well as by the intervention and a row-for-row subtraction would call
    # that difference a result (DEC-CL-0015).
    a_process_launch_id: Optional[str] = None
    b_process_launch_id: Optional[str] = None
    created: str = field(default_factory=now_iso)
    origin: Optional[str] = None
    status: str = Status.MEASURED
    notes: str = ""

    ALIGNMENTS = ("exact_rows", "segment")

    def __post_init__(self):
        if self.alignment not in self.ALIGNMENTS:
            raise ContractError(
                f"alignment={self.alignment!r} not in {self.ALIGNMENTS}")

    def identity(self) -> dict:
        return {"a": self.a_capture_id, "b": self.b_capture_id,
                "alignment": self.alignment, "segments": list(self.segments),
                "label": self.label, "protocol_version": PROTOCOL_VERSION}

    def same_process(self) -> Optional[bool]:
        """Were both arms measured in one interpreter? None if unrecorded.

        None is not False on purpose: a delta built before launch ids were
        recorded cannot answer the question, and answering it anyway is how
        an unknown becomes a claim.
        """
        if not self.a_process_launch_id or not self.b_process_launch_id:
            return None
        return self.a_process_launch_id == self.b_process_launch_id

    @property
    def delta_id(self) -> str:
        return short_id(self.identity())


# ----------------------------------------------------------- the corpus
@dataclass
class ConceptCorpusRow(Record):
    """One labelled observation.

    Labels are free-form on purpose. The ontology (identity, motion, camera,
    scene, expression, clothing, lighting) is a hypothesis this project is
    supposed to test, not a schema it should assume.
    """

    source: Any                     # ConditioningSource or dict
    labels: dict = field(default_factory=dict)
    split: str = "extract"
    matched_group: Optional[str] = None
    # Shoot / take / source-file lineage, free-form. Two rows cut from one
    # clip are not independent observations even when their bytes differ,
    # so this is what containment() checks alongside the content hash.
    source_id: Optional[str] = None
    row_id: Optional[str] = None
    notes: str = ""

    def __post_init__(self):
        if self.split not in SPLITS:
            raise ContractError(f"split={self.split!r} not in {SPLITS}")
        if self.row_id is None:
            self.row_id = short_id(self.identity(), 12)

    def _source(self) -> ConditioningSource:
        return self.source if isinstance(self.source, ConditioningSource) \
            else ConditioningSource.from_dict(self.source)

    def identity(self) -> dict:
        return {"source": self._source().identity(),
                "labels": _normalize(self.labels), "split": self.split,
                "matched_group": self.matched_group,
                "source_id": self.source_id}


@dataclass
class ConceptCorpus(Record):
    """An experiment design table.

    The corpus is where the disentangling actually happens. Most of what a
    factorization can recover was decided when someone chose which contrasts
    to shoot, which is why confounds() is in the contracts and not in some
    later analysis module.
    """

    name: str
    rows: tuple = ()
    created: str = field(default_factory=now_iso)
    origin: Optional[str] = None
    notes: str = ""

    def _rows(self) -> list:
        return [r if isinstance(r, ConceptCorpusRow)
                else ConceptCorpusRow.from_dict(r) for r in self.rows]

    def identity(self) -> dict:
        return {"name": self.name,
                "rows": [r.identity() for r in self._rows()]}

    @property
    def corpus_id(self) -> str:
        return short_id(self.identity())

    def by_split(self) -> dict:
        out = {s: [] for s in SPLITS}
        for r in self._rows():
            out[r.split].append(r)
        return out

    def matched_groups(self) -> dict:
        out = {}
        for r in self._rows():
            if r.matched_group:
                out.setdefault(r.matched_group, []).append(r)
        return out

    def label_keys(self) -> list:
        keys = set()
        for r in self._rows():
            keys.update(r.labels)
        return sorted(keys)

    def containment(self) -> dict:
        """Which sources appear in more than one split.

        Structural leakage, not statistical: the same bytes in `extract` and
        `validate` means the held-out score is an extraction score wearing a
        different hat, and `source_id` catches the version of that which a
        content hash cannot (two crops of one take are different bytes and
        the same observation).

        Returns, per key kind, a list of
        {"value", "splits": [...], "rows": [...]} for every value seen in
        two or more splits. Rows carrying no value for a key are skipped:
        absence is not a collision.
        """
        out = {"content_hash": [], "source_id": []}
        getters = {
            "content_hash": lambda r: r._source().content_hash,
            "source_id": lambda r: r.source_id,
        }
        for key, get in getters.items():
            seen = {}
            for r in self._rows():
                v = get(r)
                if not v:
                    continue
                seen.setdefault(v, {"splits": set(), "rows": []})
                seen[v]["splits"].add(r.split)
                seen[v]["rows"].append(r.row_id)
            for v, info in sorted(seen.items()):
                if len(info["splits"]) > 1:
                    out[key].append({"value": v,
                                     "splits": sorted(info["splits"]),
                                     "rows": sorted(info["rows"])})
        out["ok"] = not (out["content_hash"] or out["source_id"])
        return out

    def confounds(self, target: str) -> dict:
        """Which other labels move exactly with `target`.

        Cheap, deterministic, and the honest version of a result everyone
        wants to skip past: if every waving clip also smiles, a 'wave' factor
        is a wave-and-smile factor and no amount of solver will tell you.

        Returns, per other label key:
            perfect     the two labels are in bijection over the rows that
                        have both, so they cannot be separated at all
            partial     a value of target that only ever co-occurs with one
                        value of the other label
            free        varies independently enough to be separable
        Rows missing either label are counted and reported, never dropped
        silently.

        This is a LABEL test. It cannot see a nuisance nobody labelled
        (background, camera, lighting). `free` means the labels are
        separable, not that the corpus is unconfounded.
        """
        rows = self._rows()
        report = {"target": target, "n_rows": len(rows), "labels": {}}
        tvals = [r.labels.get(target) for r in rows]
        if all(v is None for v in tvals):
            report["error"] = f"no row carries label {target!r}"
            return report
        for key in self.label_keys():
            if key == target:
                continue
            pairs = [(t, r.labels.get(key)) for t, r in zip(tvals, rows)
                     if t is not None and r.labels.get(key) is not None]
            missing = len(rows) - len(pairs)
            fwd, back = {}, {}
            for t, o in pairs:
                fwd.setdefault(t, set()).add(o)
                back.setdefault(o, set()).add(t)
            pinned = sorted(str(t) for t, os_ in fwd.items() if len(os_) == 1)
            bijection = (pairs and all(len(v) == 1 for v in fwd.values())
                         and all(len(v) == 1 for v in back.values()))
            verdict = "perfect" if bijection else (
                "partial" if pinned else "free")
            report["labels"][key] = {
                "verdict": verdict,
                "pinned_target_values": pinned,
                "n_pairs": len(pairs),
                "n_missing": missing,
            }
        return report


# ------------------------------------------------------------ the pack
@dataclass
class FactorRecord(Record):
    """A candidate semantic operator, plus the evidence that earns the name.

    A factor is not named after the direction someone hoped for. It is named
    after behaviour it predicted on data it was not fitted to, and until
    then it keeps status PROPOSED and a working label.
    """

    factor_id: str
    label: str = ""
    representation: str = "vector"   # vector|subspace|prototype|trajectory|probe|sparse
    coordinate_system: str = ""      # which space it lives in, backend words
    blocks: tuple = ()
    steps: tuple = ()
    segments: tuple = ()
    strength_convention: str = ""
    tensor_keys: tuple = ()
    strategy: str = ""
    strategy_params: dict = field(default_factory=dict)
    extraction_rows: tuple = ()
    diagnostics: dict = field(default_factory=dict)
    held_out: dict = field(default_factory=dict)
    adversarial: dict = field(default_factory=dict)
    null: dict = field(default_factory=dict)
    status: str = Status.PROPOSED
    failure_notes: str = ""

    def __post_init__(self):
        if self.status not in Status.ALL:
            raise ContractError(f"status={self.status!r} not in {Status.ALL}")

    def identity(self) -> dict:
        return {"factor_id": self.factor_id,
                "representation": self.representation,
                "coordinate_system": self.coordinate_system,
                "blocks": list(self.blocks), "steps": list(self.steps),
                "segments": list(self.segments), "strategy": self.strategy,
                "strategy_params": _normalize(self.strategy_params),
                "extraction_rows": list(self.extraction_rows)}


@dataclass
class ConceptPackManifest(Record):
    """What was learned about a concept. Portable, versioned, and not a
    backend's compiled tensor.

    Compiled artifacts live under compiled/<backend>/ and are caches tied to
    a model fingerprint. If the fingerprint moves, the cache is stale and the
    factor evidence keeps exactly the scope it was measured under, which is
    the scope of the old stack.
    """

    name: str
    pack_id: Optional[str] = None
    concept_types: tuple = ()
    factors: tuple = ()
    stacks: tuple = ()              # fingerprints the evidence came from
    corpus_ids: tuple = ()
    visibility: str = "private"
    status: str = Status.PROPOSED
    created: str = field(default_factory=now_iso)
    origin: Optional[str] = None
    license_note: str = ""
    notes: str = ""

    def __post_init__(self):
        if self.visibility not in VISIBILITY:
            raise ContractError(
                f"visibility={self.visibility!r} not in {VISIBILITY}")
        if self.status not in Status.ALL:
            raise ContractError(f"status={self.status!r} not in {Status.ALL}")
        if self.pack_id is None:
            self.pack_id = short_id(self.identity(), 12)

    def _factors(self) -> list:
        return [f if isinstance(f, FactorRecord) else FactorRecord.from_dict(f)
                for f in self.factors]

    def identity(self) -> dict:
        return {"name": self.name,
                "concept_types": list(self.concept_types),
                "factors": [f.identity() for f in self._factors()],
                "corpus_ids": list(self.corpus_ids),
                "protocol_version": PROTOCOL_VERSION}


@dataclass
class ConceptPlan(Record):
    """A composition request: which factors, how strongly, what to reject.

    Written by a human in a text box or by an agent in a file. Same schema,
    same validator, same compiler. There is no plan format that only one of
    them can produce.
    """

    name: str = "plan"
    positive: dict = field(default_factory=dict)     # factor_id -> weight
    reject: dict = field(default_factory=dict)       # factor_id -> weight
    preserve: tuple = ()                             # factor_ids held fixed
    scope: dict = field(default_factory=dict)        # per-layer/time overrides
    backend: str = ""
    pack_ids: tuple = ()
    created: str = field(default_factory=now_iso)
    notes: str = ""

    def identity(self) -> dict:
        return {"positive": _normalize(self.positive),
                "reject": _normalize(self.reject),
                "preserve": list(self.preserve),
                "scope": _normalize(self.scope),
                "backend": self.backend, "pack_ids": list(self.pack_ids)}

    @property
    def plan_id(self) -> str:
        return short_id(self.identity(), 12)


# What the loaders below can reconstruct, keyed by the _type stamp.
RECORD_TYPES = {c.__name__: c for c in (
    Adapter, ModelStackFingerprint, ConditioningSource, ConditionVariant,
    TapSpec, LayoutSignature, TensorRef, FunctionalCaptureManifest,
    FunctionalDeltaManifest, ConceptCorpusRow, ConceptCorpus, FactorRecord,
    ConceptPackManifest, ConceptPlan)}


def load_record(d: dict):
    """Rebuild a record from its dict using the _type stamp it carries."""
    t = d.get("_type")
    if t not in RECORD_TYPES:
        raise ContractError(f"unknown record type {t!r}")
    return RECORD_TYPES[t].from_dict(d)
