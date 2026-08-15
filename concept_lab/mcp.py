"""The agent surface: the same verbs as the CLI, typed.

    <mcp-venv>/bin/python -m concept_lab.mcp

Built on the same MCPServer/pydantic shape the h3-audio-lab server uses, so
one venv serves both and the two read alike. The mcp and pydantic imports
are inside this module rather than the package, so a ComfyUI install with
neither still loads concept_lab fine; only running the server needs them.

What is deliberately NOT on this surface:

  space_export with include_private. Export is a publishing act, packs can
  encode a likeness, and a model cannot prove who is asking. The agent can
  export the shareable set and can see exactly which packs were held back
  and why; releasing a private one is a human's keystroke through the CLI.

  Any verb that would run a model. Capture costs GPU time on a shared box
  with production instances on it, so the agent plans and a human launches.
  capture_plan gives the agent the whole cost picture for free.

An agent that can measure everything and publish nothing is the right shape
for this subsystem.
"""
from __future__ import annotations

import sys
from typing import Optional

from pydantic import BaseModel, Field

from mcp.server import MCPServer

from concept_lab import api

mcp = MCPServer(
    name="concept-lab",
    title="Concept Lab",
    instructions=(
        "Functional conditioning: measure what a reference does to the "
        "model, factor it, compile it back. Records here are EVIDENCE and "
        "carry their scope: a capture belongs to one model stack and a "
        "factor belongs to the split it was fitted on. Nothing on this "
        "surface runs a model or publishes a concept pack; plan the work "
        "and hand the launch to a human. Imported records keep their origin "
        "space id and are never your own measurements."),
)


# ------------------------------------------------------------- outputs
class SpaceOut(BaseModel):
    space_id: str
    label: str = ""
    index_root: str = ""
    tensor_root: str = ""
    counts: dict = Field(default_factory=dict)
    imported_spaces: list = Field(default_factory=list)
    tensor_bytes: int = 0


class CorpusOut(BaseModel):
    name: str
    corpus_id: str
    n_rows: int
    by_split: dict = Field(default_factory=dict)
    label_keys: list = Field(default_factory=list)
    matched_groups: dict = Field(default_factory=dict)
    singleton_groups: list = Field(default_factory=list)
    containment: dict = Field(
        default_factory=dict,
        description="Sources appearing in more than one split, by "
                    "content_hash and by source_id. Any entry here means a "
                    "held-out score would be an extraction score.")
    warnings: list = Field(
        default_factory=list,
        description="Protocol gaps. Not errors: a corpus missing its "
                    "adversarial split can still be captured, it just "
                    "cannot support a factor claim afterwards.")


class ConfoundOut(BaseModel):
    target: str
    n_rows: int
    labels: dict = Field(
        description="Per label: verdict perfect|partial|free. 'perfect' "
                    "means the two labels are in bijection over these rows "
                    "and no method separates them; the fix is another "
                    "example, not another solver.")
    error: Optional[str] = None


class PlanOut(BaseModel):
    corpus: str
    backend: str
    arms: list
    forwards: int = Field(description="model forward passes this would cost")
    tap_points: int = Field(description="tensors this would write")
    tap: dict
    cost: dict = Field(
        default_factory=dict,
        description="Forward count is not the cost unit on H3: a reference "
                    "block roughly doubles the packed sequence and per-step "
                    "cost is superlinear in it. Price by tokens x steps.")
    process_launch_id: str = ""
    protocol_notes: list = Field(
        default_factory=list,
        description="Constraints on how the capture may be run, not advice. "
                    "All arms of one comparison belong to one process "
                    "launch.")
    containment_overridden: list = Field(default_factory=list)
    warnings: list = Field(default_factory=list)
    note: str = ""


class ExportOut(BaseModel):
    bundle: str
    origin_space_id: str
    included: list
    excluded: list = Field(
        description="What was left out and why. Private packs land here "
                    "unless a human exports them from the shell.")
    tensor_count: int = 0
    bytes: int = 0


class ImportOut(BaseModel):
    origin_space_id: str
    origin_label: str = ""
    landed: list
    refused: list
    index_dir: str
    tensor_dir: str


# --------------------------------------------------------------- tools
@mcp.tool(description="Status of the local concept space: what evidence "
                      "exists, which other spaces have been imported, and "
                      "how much tensor data is on disk.")
def space_status() -> SpaceOut:
    return SpaceOut(**api.space_status())


@mcp.tool(description="Check the space for inconsistencies: capture ids "
                      "that no longer recompute, tensor references that do "
                      "not resolve. Run this before trusting any number.")
def space_doctor() -> dict:
    return api.space_doctor()


@mcp.tool(description="List records of one kind, local and imported, each "
                      "tagged with the space that measured it.")
def space_list(kind: str) -> dict:
    return api.space_list(kind)


@mcp.tool(description="Start a new labelled corpus (an experiment design "
                      "table).")
def corpus_new(name: str, notes: str = "") -> dict:
    return api.corpus_new(name, notes=notes)


@mcp.tool(description="Add one labelled observation to a corpus. Labels are "
                      "free-form; split is extract|validate|adversarial|"
                      "null; matched_group ties rows that differ in exactly "
                      "one factor; source_id is the shoot/take lineage, and "
                      "two rows sharing it are one observation however "
                      "different their bytes are.")
def corpus_add(name: str, source_kind: str, source_ref: str,
               labels: Optional[dict] = None, split: str = "extract",
               matched_group: Optional[str] = None,
               source_id: Optional[str] = None,
               notes: str = "") -> dict:
    return api.corpus_add(name, source_kind, source_ref, labels=labels,
                          split=split, matched_group=matched_group,
                          source_id=source_id, notes=notes)


@mcp.tool(description="Corpus composition plus the protocol gaps it "
                      "carries. Read the warnings before planning a "
                      "capture, not after.")
def corpus_status(name: str) -> CorpusOut:
    return CorpusOut(**api.corpus_status(name))


@mcp.tool(description="Which other labels move exactly with a target "
                      "label. The cheapest honest check available: if every "
                      "example of the target also carries some other "
                      "attribute, no factorization can separate them.")
def corpus_confounds(name: str, target: str) -> ConfoundOut:
    return ConfoundOut(**api.corpus_confounds(name, target))


@mcp.tool(description="Cost and sanity-check a three-arm capture without "
                      "running anything. Returns forward count, tap points, "
                      "and every protocol warning. Launching the capture "
                      "itself is a human's call.")
def capture_plan(corpus: str, backend: str = "h3", store: str = "stats",
                 replicates: int = 1) -> PlanOut:
    return PlanOut(**api.capture_plan(corpus, backend=backend, store=store,
                                      replicates=replicates))


@mcp.tool(description="Assert that a control row really is shape-matched to "
                      "a reference row before spending the forwards. "
                      "Refuses on a mismatch naming the key; keys neither "
                      "row recorded come back as `unverifiable`, which is "
                      "silence and not agreement.")
def arms_check(corpus: str, r_row_id: str, n_row_id: str,
               backend: str = "h3") -> dict:
    return api.arms_check(corpus, r_row_id, n_row_id, backend=backend)


@mcp.tool(description="Export the shareable part of this space to a "
                      "portable bundle. Packs marked private are held back "
                      "and reported in `excluded`.")
def space_export(out_path: str, note: str = "") -> ExportOut:
    res = api.space_export(out_path, include_private=False, note=note)
    return ExportOut(**{k: res[k] for k in
                        ("bundle", "origin_space_id", "included", "excluded",
                         "tensor_count", "bytes")})


@mcp.tool(description="Import another space's bundle. Its records land in "
                      "their own namespace with their origin stamped and "
                      "never merge into your own measurements.")
def space_import(bundle_path: str) -> ImportOut:
    return ImportOut(**api.space_import(bundle_path))


@mcp.tool(description="List concept packs, local and imported, with their "
                      "visibility and evidence status.")
def pack_list() -> dict:
    return api.pack_list()


if __name__ == "__main__":
    sys.exit(mcp.run())
