"""The backend seam. Four questions and two operations, same shape as the
timeline package's backend: small enough that a second implementation is a
day's work rather than a negotiation.

    capabilities()      what this model can be asked for
    validate_tap()      is this experiment config expressible here
    fingerprint()       what is actually loaded right now
    plan_arms()         how to build a shape-matched control for this model

    capture()           run the forwards, write the evidence
    compile()           realize a plan as native conditioning

The last two are where a backend earns its keep and where the first one is
still empty. A seam with one implementation and no second in sight is
usually a fiction, so this one is deliberately minimal: it declares only
what the H3 work actually needs, and the second backend gets to widen it.
"""
from __future__ import annotations

from concept_lab import types as T


class BackendError(RuntimeError):
    pass


class Backend:
    name = "base"

    # ------------------------------------------------------------ facts
    def capabilities(self) -> dict:
        """What this backend supports, as data an agent can branch on."""
        raise NotImplementedError

    def default_blocks(self) -> tuple:
        """A first tap ladder. An experiment config, never a format fact."""
        raise NotImplementedError

    def default_segments(self) -> tuple:
        """Which named segments a v0 capture should keep."""
        raise NotImplementedError

    def segment_vocabulary(self) -> tuple:
        """Every segment name this backend's packed layout can produce."""
        raise NotImplementedError

    def cost_model(self) -> dict:
        """How this backend's cost actually scales, as data a planner reads.

        Forward count is the unit everyone reaches for and it is wrong
        wherever conditioning changes the packed sequence length. A backend
        that has measured its own scaling says so here; the default says
        nothing rather than guessing.
        """
        return {"note": f"{self.name} has not declared a cost model; "
                        f"forward count is the only unit available and it "
                        f"is probably not the right one"}

    def validate_tap(self, tap: T.TapSpec) -> None:
        """Refuse a tap this backend cannot honour, and say which field."""
        vocab = self.segment_vocabulary()
        bad = [s for s in tap.segments if s not in vocab]
        if bad:
            raise BackendError(
                f"{self.name}: unknown segment(s) {bad}; this backend's "
                f"packed layout has {list(vocab)}")

    def fingerprint(self, **kw) -> T.ModelStackFingerprint:
        """What is loaded, read from the live stack. Never from a filename."""
        raise NotImplementedError

    def plan_arms(self, source: T.ConditioningSource) -> dict:
        """How to build the shape-matched control for this source.

        A structural control is backend-specific because what counts as
        'same shape' is: same reference count, same post-preprocess
        dimensions, same modality, same presentation token length. Getting
        this wrong is the difference between measuring reference content
        and measuring that a reference exists.
        """
        raise NotImplementedError

    # ------------------------------------------------------- operations
    def capture(self, *a, **k):
        raise NotImplementedError

    def compile(self, *a, **k):
        raise NotImplementedError
