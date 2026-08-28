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

from .expr import MAX_EXPONENT, ExprError, Ref, kind_of


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


# --------------------------------------------------------------- transforms
#
# Transforms arrive with Safe Function (spec 5.3). They are NOT
# predicate_safe: a Gate, Condition, Filter or Partition refuses them,
# because those nodes decide a branch and planning must stay pure and cheap.
# All of them are functional (a new tensor, never an in-place write on a
# value another node also consumes) and every allocation is capped by the
# installation's max_pixels.
import torch

from . import policy


def _pixels(count: int, what: str) -> None:
    ceiling = policy.ceilings()["max_pixels"]
    if count > ceiling:
        raise ExprError(f"{what} would allocate {count} pixels, over the "
                        f"max_pixels limit of {ceiling}")


def _whole(value, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExprError(f"{what} expects whole numbers, got {type(value).__name__}")
    return int(value)


def _resize(image, width, height):
    t = _image(image)
    w, h = _whole(width, "image.resize"), _whole(height, "image.resize")
    if w < 1 or h < 1:
        raise ExprError(f"image.resize needs a positive size, got {w}x{h}")
    _pixels(int(t.shape[0]) * w * h * int(t.shape[3]), "image.resize")
    out = torch.nn.functional.interpolate(t.movedim(-1, 1).float(), size=(h, w),
                                          mode="bilinear", align_corners=False)
    return Ref(out.movedim(1, -1), "IMAGE")


def _crop(image, x, y, width, height):
    t = _image(image)
    x0, y0 = _whole(x, "image.crop"), _whole(y, "image.crop")
    w, h = _whole(width, "image.crop"), _whole(height, "image.crop")
    if x0 < 0 or y0 < 0 or w < 1 or h < 1:
        raise ExprError(f"image.crop needs a positive box, got {w}x{h} at {x0},{y0}")
    if y0 + h > int(t.shape[1]) or x0 + w > int(t.shape[2]):
        raise ExprError(f"image.crop box {w}x{h} at {x0},{y0} leaves an image of "
                        f"{int(t.shape[2])}x{int(t.shape[1])}")
    return Ref(t[:, y0:y0 + h, x0:x0 + w, :].clone(), "IMAGE")


def _flip(image, horizontal=True):
    t = _image(image)
    return Ref(torch.flip(t, dims=[2] if horizontal else [1]), "IMAGE")


def _select(image, indices):
    t = _image(image)
    items = unref(indices)
    if isinstance(items, (int, float)) and not isinstance(items, bool):
        items = [items]
    if not isinstance(items, (list, tuple)):
        raise ExprError(f"image.select expects a list of indices, got {type(items).__name__}")
    batch = int(t.shape[0])
    picked = []
    for index in items:
        i = _whole(index, "image.select")
        if not -batch <= i < batch:
            raise ExprError(f"image.select index {i} is out of range for a batch of {batch}")
        picked.append(i % batch)
    if not picked:
        raise ExprError("image.select needs at least one index")
    _pixels(len(picked) * int(t.shape[1]) * int(t.shape[2]) * int(t.shape[3]), "image.select")
    return Ref(t[picked].clone(), "IMAGE")


def _mask_invert(mask):
    return Ref(1.0 - _tensor(mask, "mask.invert").float(), "MASK")


def _mask_threshold(mask, t):
    if isinstance(t, bool) or not isinstance(t, (int, float)):
        raise ExprError(f"mask.threshold expects a number, got {type(t).__name__}")
    return Ref((_tensor(mask, "mask.threshold") > float(t)).float(), "MASK")


def _latent_blend(a, b, t):
    if isinstance(t, bool) or not isinstance(t, (int, float)):
        raise ExprError(f"latent.blend expects a number, got {type(t).__name__}")
    first, second = _latent(a), _latent(b)
    if tuple(first.shape) != tuple(second.shape):
        raise ExprError(f"latent.blend needs matching shapes, got {tuple(first.shape)} "
                        f"and {tuple(second.shape)}")
    mixed = dict(unref(a))
    mixed["samples"] = first * (1.0 - float(t)) + second * float(t)
    return Ref(mixed, "LATENT")


def _seq_concat(a, b):
    first, second = unref(a), unref(b)
    if isinstance(first, (list, tuple)) and isinstance(second, (list, tuple)):
        return list(first) + list(second)
    if getattr(first, "shape", None) is not None and getattr(second, "shape", None) is not None:
        if tuple(first.shape[1:]) != tuple(second.shape[1:]):
            raise ExprError(f"seq.concat needs matching shapes after the batch dimension, "
                            f"got {tuple(first.shape)} and {tuple(second.shape)}")
        _pixels(int(first[0].numel()) * (int(first.shape[0]) + int(second.shape[0])),
                "seq.concat")
        return Ref(torch.cat([first, second], dim=0), kind_of(first))
    raise ExprError(f"seq.concat joins two lists or two tensors, got "
                    f"{type(first).__name__} and {type(second).__name__}")


TRANSFORMS = {
    "image.resize": _resize, "image.crop": _crop, "image.flip": _flip,
    "image.select": _select, "mask.invert": _mask_invert,
    "mask.threshold": _mask_threshold, "latent.blend": _latent_blend,
    "seq.concat": _seq_concat,
}
for _cid, _transform in TRANSFORMS.items():
    _add(_cid, _transform, return_type="ANY", predicate_safe=False, cost=1.0)
