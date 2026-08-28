"""Installation policy: Safe Function budgets (spec 8.2), LLM providers (9.4).

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

    {"max_iterations": 100, "llm_providers": {"local": {...}}}

It is the pack root's flow_policy.json unless MAINODES_FLOW_POLICY names
another one, which is how a sandbox gets its own without writing into a pack.
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
POLICY_ENV = "MAINODES_FLOW_POLICY"      # a sandbox points this at its own file
# Named LLM endpoints (spec 9.4). A workflow carries the NAME, never a base
# url and never a key: that is what stops a shared workflow deciding where a
# user's images go. The key is read from the environment variable the
# provider names.
LLM_PROVIDERS = {
    "local": {"kind": "openai_compatible",
              "base_url": "http://127.0.0.1:8080/v1",
              "api_key_env": "MAINODES_LLM_KEY_LOCAL",
              "default_model": "local-model"},
}
# max_tokens is not in DEFAULTS/CEILINGS because it has no node default here
# and no effective() semantics: the node ships its own default and this is the
# only ceiling over it
LLM_MAX_TOKENS_CEILING = 32768           # above the node default of 512

_CACHE: dict = {}


def policy_path(path: str | None = None) -> str:
    """An explicit path, then the env override, then the pack root's file."""
    return path or os.environ.get(POLICY_ENV) or POLICY_PATH


def _loaded(path: str) -> dict:
    """The parsed policy file, cached on its mtime; {} if absent or broken."""
    try:
        stamp = os.stat(path).st_mtime_ns
    except OSError:
        return {}                # no file, and a broken one, change nothing
    cached = _CACHE.get(path)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    try:
        with open(path, "r", encoding="utf-8") as fh:
            loaded = json.load(fh)
    except (OSError, ValueError):
        return {}
    _CACHE[path] = (stamp, loaded if isinstance(loaded, dict) else {})
    return _CACHE[path][1]


def ceilings(path: str | None = None) -> dict:
    """CEILINGS with the JSON override applied, cached on the file mtime."""
    values = dict(CEILINGS)
    for field, value in _loaded(policy_path(path)).items():
        if field in CEILINGS and isinstance(value, (int, float)) \
                and not isinstance(value, bool) and value > 0:
            values[field] = type(CEILINGS[field])(value)
    return values


def llm_providers(path: str | None = None) -> dict:
    """The shipped providers, replaced name by name from the policy file."""
    providers = {name: dict(spec) for name, spec in LLM_PROVIDERS.items()}
    loaded = _loaded(policy_path(path)).get("llm_providers")
    for name, spec in (loaded or {}).items() if isinstance(loaded, dict) else ():
        if isinstance(spec, dict):
            providers[str(name)] = dict(spec)
    return providers


def llm_max_tokens(requested, path: str | None = None) -> int:
    """The node's max_tokens lowered by the installation ceiling."""
    # int|float like ceilings(): 2048.0 in a policy file meant the same thing
    # to an administrator and was silently ignored here
    ceiling = _loaded(policy_path(path)).get("llm_max_tokens")
    ok = isinstance(ceiling, (int, float)) and not isinstance(ceiling, bool) and ceiling > 0
    return min(int(ceiling) if ok else LLM_MAX_TOKENS_CEILING, int(requested))


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
