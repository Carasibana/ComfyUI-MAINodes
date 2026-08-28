"""Capability registry for flow expressions (spec 5.3).

A capability is the only thing that may unwrap a ``Ref``. Names are
resolved against this registry as strings before any value is touched, so
there is no module object, no ``getattr`` and no attribute surface reachable
from workflow text.

Bare names (``sqrt``, ``clamp``) resolve through the ``core.math`` and
``core.logic`` packs, which is what keeps an expression portable between
this evaluator and core's Math Expression node.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

from .expr import MAX_EXPONENT, ExprError, Ref


@dataclass(frozen=True)
class Capability:
    id: str
    version: int
    fn: Callable
    arg_types: tuple = ()
    return_type: str = "ANY"
    predicate_safe: bool = True
    cost: float = 0.0


REGISTRY: dict[str, Capability] = {}
BARE_NAMES: dict[str, str] = {}          # bare name -> full id
_BARE_PACKS = ("core.math", "core.logic")


def register(cap: Capability) -> Capability:
    """Register by full id; ids in a bare pack also answer to a bare name."""
    if cap.id in REGISTRY:
        raise ValueError(f"capability '{cap.id}' is already registered")
    REGISTRY[cap.id] = cap
    pack, _, bare = cap.id.rpartition(".")
    if pack in _BARE_PACKS:
        BARE_NAMES[bare] = cap.id
    return cap


def resolve(name: str | None) -> Capability | None:
    """Full id first, then the bare-name map. Never raises."""
    if not name:
        return None
    cap = REGISTRY.get(name)
    if cap is not None:
        return cap
    full = BARE_NAMES.get(name)
    return REGISTRY.get(full) if full else None


def _add(cid, fn, *, return_type="ANY", predicate_safe=True, cost=0.0, version=1):
    return register(Capability(cid, version, fn, (), return_type, predicate_safe, cost))


def unref(value):
    """The one place a Ref is opened."""
    return value.value if isinstance(value, Ref) else value


def _tensor(value, what):
    t = unref(value)
    if getattr(t, "shape", None) is None:
        raise ExprError(f"{what} expects a tensor value, got {type(t).__name__}")
    return t


def _image(value):
    t = _tensor(value, "image.*")
    if len(t.shape) != 4:
        raise ExprError(f"image.* expects IMAGE[B,H,W,C], got a {len(t.shape)}-D tensor")
    return t


def _latent(value):
    d = unref(value)
    if not isinstance(d, dict) or "samples" not in d:
        raise ExprError("latent.* expects a LATENT dict with 'samples'")
    return d["samples"]


# core.math: the exact set core's Math Expression exposes, same semantics.
def _variadic_sum(*args):
    if len(args) == 1 and hasattr(args[0], "__iter__"):
        return sum(args[0])
    return sum(args)


def _safe_pow(base, exp):
    if abs(exp) > MAX_EXPONENT:
        raise ExprError(f"exponent {exp} exceeds the maximum allowed ({MAX_EXPONENT})")
    return pow(base, exp)


_MATH = {
    "sum": _variadic_sum, "min": min, "max": max, "abs": abs, "round": round,
    "pow": _safe_pow, "sqrt": math.sqrt, "ceil": math.ceil, "floor": math.floor,
    "log": math.log, "log2": math.log2, "log10": math.log10, "sin": math.sin,
    "cos": math.cos, "tan": math.tan, "int": int, "float": float,
}
for _name, _fn in _MATH.items():
    _add(f"core.math.{_name}", _fn, return_type="FLOAT")


# core.logic
def _clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


def _between(x, lo, hi):
    return lo <= x <= hi


def _near(x, target, eps):
    return abs(x - target) <= eps


def _coalesce(a, b):
    return b if a is None else a


_add("core.logic.clamp", _clamp, return_type="FLOAT")
_add("core.logic.between", _between, return_type="BOOLEAN")
_add("core.logic.near", _near, return_type="BOOLEAN")
_add("core.logic.coalesce", _coalesce)
_add("core.logic.is_none", lambda x: x is None, return_type="BOOLEAN")


# image: IMAGE is [B,H,W,C]
_add("image.width", lambda x: int(_image(x).shape[2]), return_type="INT")
_add("image.height", lambda x: int(_image(x).shape[1]), return_type="INT")
_add("image.batch", lambda x: int(_image(x).shape[0]), return_type="INT")
_add("image.megapixels",
     lambda x: float(_image(x).shape[1] * _image(x).shape[2]) / 1e6, return_type="FLOAT")
_add("image.aspect",
     lambda x: float(_image(x).shape[2]) / float(_image(x).shape[1]), return_type="FLOAT")

# mask: coverage is the mean of the mask tensor
_add("mask.coverage", lambda x: float(_tensor(x, "mask.coverage").mean()), return_type="FLOAT")
_add("mask.is_empty",
     lambda x: float(_tensor(x, "mask.is_empty").mean()) == 0.0, return_type="BOOLEAN")

# latent: 5-D samples are [B,C,T,H,W]; anything else is a single frame
_add("latent.frames",
     lambda x: int(_latent(x).shape[2]) if len(_latent(x).shape) == 5 else 1, return_type="INT")
_add("latent.shape", lambda x: tuple(int(s) for s in _latent(x).shape), return_type="LIST")


def _seq_length(x):
    v = unref(x)
    if isinstance(v, (list, tuple, str)):
        return len(v)
    shape = getattr(v, "shape", None)
    if shape is not None:
        return int(shape[0])
    raise ExprError(f"seq.length expects a sequence or a tensor, got {type(v).__name__}")


_add("seq.length", _seq_length, return_type="INT")


def pack_of(cap_id: str) -> str:
    return cap_id.rpartition(".")[0]


def ids() -> list[str]:
    return sorted(REGISTRY)
