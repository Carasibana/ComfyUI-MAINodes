"""The H3 backend: everything the rest of the timeline must not know.

  spec.py     H3ModelSpec     — STRUCTURAL facts. Relied on, never overridden.
  recipe.py   H3RecipeProfile — EXPERIMENT-DERIVED values with provenance.
                                Preferred, challengeable, versioned.
  gridlaw.py  the single import point for motion.py's map machinery.
  compile.py  H3Backend: plan -> legal .api.json.
  propose.py  oracle profiles -> a ratio envelope.
"""
