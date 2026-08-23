"""Shared data vocabulary for stage handoff (alpha, 2026-08-23).

No persistence here. These are the structures the extension planner emits
today and a Capsule will serialize tomorrow, frozen now so the first
feature built to consume a Capsule does not need an adapter around it.

Time is integer or rational everywhere. Seconds are a display field.

  Timebase   fps as a rational, the 40 Hz H3 audio-latent clock, sample rate
  Span       a half-open frame range with its tick and sample counts
  Handle     a typed span shared by extension prefix, segment-crop handles
             and rolling-window overlap: role, where it comes from, where it
             lands, whether it is protected from retiming
  ConditioningFingerprint   what the prompt MEANT on this runtime (#15808
             made prompt text + model name insufficient)
  ExecutionFingerprint      what actually ran: commits, hashes, the
             capability snapshot, requested / resolved / executed values

Every structure round-trips through ``to_dict`` / ``from_dict`` and hashes
through ``canonical_json`` (sorted keys, no whitespace) so two producers of
the same content get the same digest.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from typing import Optional

SCHEMA = "mai.h3.capsule-types/1"

H3_FRAME_GRID = (17, 5)        # legal clip length = 17k + 5
H3_AUDIO_HZ = 40               # audio latent ticks per second


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def digest(obj) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def align_up(n: int, k: int = H3_FRAME_GRID[0], r: int = H3_FRAME_GRID[1]) -> int:
    """Generation length: core rounds UP to 17k+5 (align_frame_count)."""
    n = max(int(n), r)
    while n % k != r:
        n += 1
    return n


def align_down(n: int, k: int = H3_FRAME_GRID[0], r: int = H3_FRAME_GRID[1]) -> int:
    """Clip guide / ref video length: core rounds DOWN to 17k+5; below 5 a
    guide becomes a single frame (1), zero stays zero."""
    n = int(n)
    if n <= 0:
        return 0
    if n < r:
        return 1
    while n % k != r:
        n -= 1
    return n


@dataclass
class Timebase:
    fps_num: int = 24
    fps_den: int = 1
    audio_hz: int = H3_AUDIO_HZ
    sample_rate: int = 0           # 0 = unknown until an AUDIO arrives

    @property
    def fps(self) -> Fraction:
        return Fraction(self.fps_num, self.fps_den)

    def ticks(self, frames: int) -> Fraction:
        return Fraction(frames) * self.audio_hz / self.fps

    def samples(self, frames: int) -> Fraction:
        return Fraction(frames) * self.sample_rate / self.fps

    def seconds(self, frames: int) -> float:
        return float(Fraction(frames) / self.fps)

    def clock_aligned(self, frames: int) -> bool:
        return self.ticks(frames).denominator == 1

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: d[k] for k in ("fps_num", "fps_den", "audio_hz", "sample_rate") if k in d})


@dataclass
class Span:
    """Half-open [start, end) in frames, with the derived counts spelled out
    so a reader never re-derives them from a different clock."""
    start: int
    end: int
    ticks: Optional[str] = None      # Fraction as "num/den" string, exact
    samples: Optional[str] = None

    @property
    def frames(self) -> int:
        return self.end - self.start

    @staticmethod
    def make(start: int, end: int, tb: Timebase) -> "Span":
        n = end - start
        s = Span(int(start), int(end), str(tb.ticks(n)),
                 str(tb.samples(n)) if tb.sample_rate else None)
        return s

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(d["start"], d["end"], d.get("ticks"), d.get("samples"))


@dataclass
class Handle:
    """One concept for extension prefix, segment-crop handle, window overlap."""
    role: str                         # extension_prefix | segment_crop | window_overlap | edit_region
    source: Span                      # where the pixels come from (previous clip coords)
    destination: Span                 # where they land (new clip coords)
    global_span: Optional[Span] = None  # on the assembled timeline
    protected: bool = True            # the de-rope may not retime it
    retime_allowed: bool = False
    ownership: str = "source"         # who owns the pixels at assembly: source | new | blend
    visual_anchor: str = "guide"      # per_token_mask | guide | none
    audio_anchor: str = "guide"       # guide | regenerated | none
    notes: list = field(default_factory=list)

    def to_dict(self):
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d):
        return cls(role=d["role"], source=Span.from_dict(d["source"]),
                   destination=Span.from_dict(d["destination"]),
                   global_span=Span.from_dict(d["global_span"]) if d.get("global_span") else None,
                   protected=d.get("protected", True), retime_allowed=d.get("retime_allowed", False),
                   ownership=d.get("ownership", "source"), visual_anchor=d.get("visual_anchor", "guide"),
                   audio_anchor=d.get("audio_anchor", "guide"), notes=list(d.get("notes", [])))


@dataclass
class ConditioningFingerprint:
    prompt_sha: str = ""
    tokenizer_special_tokens: Optional[bool] = None   # #15808 present on the runtime that encoded it
    comfy_commit: Optional[str] = None
    ref_roles: dict = field(default_factory=dict)     # "<Picture 1>": "identity", "<Video 1>": "motion"
    visual_cond_noise_aug: Optional[float] = None
    audio_cond_noise_aug: Optional[float] = None

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass
class ExecutionFingerprint:
    comfy_commit: Optional[str] = None
    mainodes_commit: Optional[str] = None
    capabilities: dict = field(default_factory=dict)   # the probe snapshot that ran
    requested: dict = field(default_factory=dict)
    resolved: dict = field(default_factory=dict)
    executed: dict = field(default_factory=dict)       # filled after the run; empty = not yet
    reasons: dict = field(default_factory=dict)        # why resolved != requested

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})
