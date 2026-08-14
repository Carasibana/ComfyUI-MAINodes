"""The seam between semantic plans and one video model's reality.

Four methods, one implementation (H3). This is deliberately NOT a plugin
system: there is no factory, no registry of backends, no second backend.
The seam exists so H3's grid law, window planning, guide placement and
expand-to-end semantics live in timeline/h3/ and nothing above it has to
know they exist (architecture amendment, 2026-08-14 night).

Stdlib only.
"""
import abc


class CompileResult(object):
    """What a backend hands back.

    graph:    the executable artifact (for H3: a ComfyUI .api.json dict).
    compiled: the cache block to store in plan['compiled'][backend_id]
              — hold maps, window geometry, dilated lengths, graph path.
    report:   human-readable receipt, printed by nodes and CLIs.
    warnings: things the operator should see but that do not block.
    """

    def __init__(self, graph, compiled, report, warnings=None):
        self.graph = graph
        self.compiled = compiled
        self.report = report
        self.warnings = list(warnings or [])

    def __repr__(self):
        return (f"CompileResult(nodes={len(self.graph or {})}, "
                f"warnings={len(self.warnings)})")


class VideoModelBackend(abc.ABC):
    """A model family's opinion about how to execute a plan."""

    backend_id = "abstract"

    @abc.abstractmethod
    def capabilities(self):
        """-> dict. What lane types compile, what the model can be asked
        for, which structural limits are non-negotiable. UIs use this to
        grey out lanes instead of guessing."""

    @abc.abstractmethod
    def validate(self, plan, ctx=None):
        """-> list of problems with running THIS plan on THIS model.
        Semantic validation already happened in schema.validate; this is
        the model's own refusals (unsupported lane, illegal length that
        cannot be reached, budget below the smallest possible pass)."""

    @abc.abstractmethod
    def compile(self, plan, ctx=None):
        """-> CompileResult. Must be deterministic: same plan revision in,
        same graph out."""

    @abc.abstractmethod
    def estimate(self, plan, ctx=None):
        """-> price.Estimate. The plan's cost on this model, without
        running anything."""
