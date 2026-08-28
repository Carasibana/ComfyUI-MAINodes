"""Installation policy for Safe Function budgets (spec 8.2).

Budgets are serialized on the node so the editor and the API behave
identically. An installation may lower any of them, never raise them: the
effective limit is ``min(node, policy)``. There is no unlimited value; 0 or
a negative number is a validation error naming the field, because "no
setting means unlimited" is how a shared workflow takes a host down.

The ceiling sits ABOVE the node default, not on it. One dict serving as
both made every default its own ceiling, so "raise max_iterations on the
node" was advice that could not work and every budget message pointed at
this file. Ceilings ship at roughly a hundred times the defaults: the node
setting is what a user tunes, and lowering a ceiling is a deliberate act by
an administrator, which is the only case the flow_policy wording should
describe.

The optional override is a JSON object at the pack root, so an
administrator edits one file and never the code:

    {"max_iterations": 100, "max_pixels": 8000000}
"""
from __future__ import annotations

import json
import os

DEFAULTS = {
    "max_iterations": 1000,          # total loop steps, shared across nested loops
    "max_ops": 50000,                # interpreted operations
    "max_calls": 5000,               # capability calls
    "max_collection": 10000,         # sequence elements and characters
    "max_tensor_elements": 100_000_000,   # elements of transform results
    "max_pixels": 64_000_000,        # per transform result, see capabilities
}

CEILINGS = {
    "max_iterations": 100_000,
    "max_ops": 5_000_000,
    "max_calls": 500_000,
    "max_collection": 1_000_000,
    "max_tensor_elements": 1_000_000_000,
    # max_pixels is a per-result cap rather than a budget, and a node cannot
    # ask for more of it, so its ceiling is its default
    "max_pixels": 64_000_000,
}

POLICY_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "flow_policy.json")

_CACHE: dict = {}


def ceilings(path: str | None = None) -> dict:
    """CEILINGS with the JSON override applied, cached on the file mtime."""
    path = path or POLICY_PATH
    try:
        stamp = os.stat(path).st_mtime_ns
    except OSError:
        return dict(CEILINGS)
    cached = _CACHE.get(path)
    if cached is not None and cached[0] == stamp:
        return dict(cached[1])
    values = dict(CEILINGS)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            loaded = json.load(fh)
        for field, value in (loaded or {}).items():
            if field in CEILINGS and isinstance(value, (int, float)) \
                    and not isinstance(value, bool) and value > 0:
                values[field] = type(CEILINGS[field])(value)
    except (OSError, ValueError):
        # a broken policy file must not take the ceilings with it
        return dict(CEILINGS)
    _CACHE[path] = (stamp, dict(values))
    return values


def effective(field: str, requested=None, path: str | None = None):
    """The node's setting lowered by the installation ceiling.

    No setting at all means the node default, never the ceiling: a caller
    that omits a budget is asking for the shipped behaviour, not for the
    most an installation would tolerate.
    """
    ceiling = ceilings(path)[field]
    if requested is None:
        requested = DEFAULTS[field]
    return min(ceiling, requested)


def check_positive(field: str, value) -> str | None:
    """The message for a budget that is not a positive number, or None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return f"{field} must be a positive whole number, got {value!r}"
    if value <= 0:
        return (f"{field} must be greater than 0, got {value}; "
                f"there is no unlimited setting")
    return None
