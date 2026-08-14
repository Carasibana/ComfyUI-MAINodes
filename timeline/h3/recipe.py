"""H3RecipeProfile — the EXPERIMENT-DERIVED half of what we know.

The epistemic boundary (amendment 2, item 1): structural facts live in
spec.py and are relied on. Everything here came out of a run, so the
compiler PREFERS these values and a proposer may CHALLENGE one — where a
challenge means the plan is flagged for eyes, never a silent override.

Every value carries lightweight machine-readable provenance (amendment 2,
item 2): a RecipeValue with value, confidence and evidence as plain string
refs into our packet/report namespace. No provenance database, no new
infrastructure; strings now, richer later. DEPRECATION IS FIRST CLASS: a
superseded value keeps its record and the reason it existed, so the system
remembers why a workaround was there after the workaround is gone.

Profiles version; code does not. Add a new profile, do not edit history.

Stdlib only.
"""

RECIPE_VERSION = "h3-recipe-2026-08-14"

CONFIDENCE = ("ratified",      # operator ruled on it
              "measured",      # a run says so, reproduced
              "provisional")   # one run, or inherited from a single report


class RecipeValue(object):
    __slots__ = ("value", "confidence", "evidence", "note",
                 "deprecated", "superseded_by", "deprecation_reason")

    def __init__(self, value, confidence, evidence, note="",
                 deprecated=False, superseded_by=None,
                 deprecation_reason=""):
        assert confidence in CONFIDENCE, confidence
        self.value = value
        self.confidence = confidence
        self.evidence = list(evidence)
        self.note = note
        self.deprecated = bool(deprecated)
        self.superseded_by = superseded_by
        self.deprecation_reason = deprecation_reason

    def get(self):
        return self.value

    def as_dict(self):
        d = {"value": self.value, "confidence": self.confidence,
             "evidence": list(self.evidence), "note": self.note}
        if self.deprecated:
            d.update({"deprecated": True,
                      "superseded_by": self.superseded_by,
                      "deprecation_reason": self.deprecation_reason})
        return d

    def __repr__(self):
        return f"RecipeValue({self.value!r}, {self.confidence})"


class H3RecipeProfile(object):
    """Values the compiler prefers. Challengeable, versioned, cited."""

    version = RECIPE_VERSION

    values = {
        # --- de-roping doctrine ------------------------------------------
        "inject_default": RecipeValue(
            0.45, "ratified",
            ["PACKET_045_round_2026-08-14", "STATE_2026-08-14_NIGHT.md"],
            "regen strength for the windowed v3.1 recipe; the 0.45-0.50 band "
            "is where the window recipe was tuned"),
        "inject_band": RecipeValue(
            (0.45, 0.50), "measured",
            ["docs/MOTION_PASS1_RECIPES.md", "PACKET_045_round_2026-08-14"],
            "outside this band the windowed recipe was not characterised"),
        "d_max_sweet_spot": RecipeValue(
            4, "measured", ["motion.py H3JerkOracle PRESETS"],
            "peak hold count; 2-3 saves time but starts to rope again"),
        "max_end_tail": RecipeValue(
            17, "ratified", ["expand_to_end toggle round 2026-08-14",
                             "tests/test_expand_to_end.py"],
            "a rate-1 tail longer than one whole group is intended REST, not "
            "an end jump: a 124f oracle map ending in [1]*39 would otherwise "
            "be rewritten from 250 to 294 dilated frames"),
        "bridge_tokens": RecipeValue(
            8, "measured", ["motion.py H3JerkOracle bridge"],
            "fill plateau dips between peaks of the same burst; a dip is "
            "where mid-burst artifacts come back (4 of 5 in the v1 map)"),
        "handle_frames": RecipeValue(
            12, "measured", ["H3WindowPlan defaults", "window round 2026-08-12"],
            "context frames each side of a regenerated window"),
        "window_budget_dilated_frames": RecipeValue(
            209, "provisional", ["H3WindowPlan defaults"],
            "209 = 17x12+5, 62 tokens; a per-card ceiling, not a law"),

        # --- pinning / conditioning --------------------------------------
        "pin_policy": RecipeValue(
            "token_leading", "measured", ["pinning study 2026-08-13"],
            "token-leading pins are clean; off-position pins echo at +17"),
        "split_sigmas_at_shift12": RecipeValue(
            False, "ratified", ["h3-spatial-refine skill (FaceRefine study)"],
            "never SplitSigmas at shift 12 (their ruling, adopted)"),

        # --- cost model ---------------------------------------------------
        # Layer (a)'s exponent. NOTE the deprecated predecessor below: this
        # is exactly the case the deprecation record exists for.
        "cost_exponent": RecipeValue(
            1.55, "measured",
            ["PACKET_timeline_T0T1_2026-08-14",
             "maxq token sweep 1664x928 2026-08-14"],
            "per-step time goes as tokens**1.55; fitted from the two 1664x928 "
            "maxq anchors below (27->13.24, 47->31.27 s/it: ratio 2.362 over "
            "1.741 = 1.551)"),
        "cost_exponent_legacy": RecipeValue(
            1.7, "measured", ["motion.py COST_EXP", "2026-08-09 37->92 token run"],
            "the pack's shipped exponent, fitted on one clip one card (1.75) "
            "and split with a field report (1.64)",
            deprecated=True, superseded_by="cost_exponent",
            deprecation_reason="superseded 2026-08-14 by the maxq token sweep, "
            "which fitted 1.55 at 1664x928. Kept because motion.py's node "
            "reports still quote 1.7, so a user comparing the two numbers "
            "is not looking at a bug"),
        "speed_anchors": RecipeValue(
            (
                # tokens, s/step, pixels, device, note
                {"work_units": 27, "seconds_per_step": 13.24,
                 "pixels": 1664 * 928, "device": "maxq",
                 "note": "1664x928 maxq token sweep 2026-08-14"},
                {"work_units": 47, "seconds_per_step": 31.27,
                 "pixels": 1664 * 928, "device": "maxq",
                 "note": "1664x928 maxq token sweep 2026-08-14"},
                {"work_units": 32, "seconds_per_step": 4.97,
                 "pixels": 1152 * 640, "device": "maxq",
                 "note": "t2c_c gate-1 render 2026-08-14 21:55 (15 sampled "
                         "steps, 4.97 s/it mean, 107f dilated window)"},
            ),
            "measured",
            ["PACKET_timeline_T0T1_2026-08-14",
             "gate 1 t2c_c_00004_ 2026-08-14"],
            "reference points only; layer (b) predicts from the USER's own "
            "flight recorder, never from these"),
        "overhead_seconds": RecipeValue(
            6.7, "measured", ["motion.py OVERHEAD_S", "2026-08-12 step-count fit"],
            "non-sampling seconds per render (setup, VAE encode/decode), "
            "already divided by that run's dilation"),

        # --- continuation --------------------------------------------------
        # NOTE (review round 3, item 6): the audio-exact segment lengths used
        # to live here as a stored preference. They are not a preference and
        # not an experimental finding — they are arithmetic on H3's 40 Hz
        # audio clock against its 24 fps video clock, so they moved to
        # H3ModelSpec.audio_exact_lengths() where they are COMPUTED. Nothing
        # in this profile may restate them.
        "overlap_presets_frames": RecipeValue(
            (17, 39, 56), "provisional",
            ["TIMELINE_SURFACE_SPEC_2026-08-14.md parked research 1"],
            "the coherence-vs-overlap curve is NOT measured yet; these are "
            "the depths the parked experiment will test"),
    }

    # ---- access

    def get(self, key):
        rv = self.values[key]
        if rv.deprecated:
            raise KeyError(f"{key} is deprecated ({rv.deprecation_reason}); "
                           f"use {rv.superseded_by}")
        return rv.value

    def record(self, key):
        return self.values[key]

    def evidence(self, key):
        return list(self.values[key].evidence)

    def challenges(self, key, value, by, why):
        """A proposer disagreeing with a recipe value. This NEVER changes
        the value: it returns a flag the compiler attaches to the plan so a
        human looks. Structural facts (spec.py) cannot be challenged at all.
        """
        rv = self.values[key]
        return {"kind": "recipe_challenge", "key": key,
                "profile_value": rv.value, "profile_confidence": rv.confidence,
                "proposed": value, "by": by, "why": why,
                "evidence": list(rv.evidence),
                "resolution": "flagged for review; profile value was used"}

    def deprecated_keys(self):
        return [k for k, v in self.values.items() if v.deprecated]

    def as_dict(self):
        return {"version": self.version,
                "values": {k: v.as_dict() for k, v in self.values.items()}}


PROFILE = H3RecipeProfile()
