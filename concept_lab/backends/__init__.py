"""Backends: everything a specific model knows about itself.

The core is model-neutral by construction, so all the words that only mean
something to one model (segment names, block counts, sampler shapes, where
the tap goes) live behind this seam. H3 is the first backend because
MAINodes already has substantial H3 tooling and H3 has a native reference
path; nothing in the core assumes it is the only one.

Registration is a dict, not a plugin system. When there are three backends
that is still the right amount of machinery.
"""
from concept_lab.backends.base import Backend, BackendError    # noqa: F401


def get_backend(name: str) -> Backend:
    from concept_lab.backends.h3 import H3Backend

    table = {"h3": H3Backend}
    if name not in table:
        raise BackendError(f"unknown backend {name!r}; have {sorted(table)}")
    return table[name]()


def list_backends() -> list:
    return ["h3"]
