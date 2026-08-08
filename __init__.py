"""ComfyUI-MatlowNodes — MatlowAI's MiniMax-H3 node collection.

Two families in one pack:
- Contact-Sheet diffusion (five coordinated views from one reference) —
  V3 extension nodes, unchanged from ComfyUI-H3-ContactSheet.
- Motion Lab (v2v time-smear de-roping pipeline: jerk oracle, time smear,
  inject schedule, exact recovery, oracle heatmap) — V1 nodes, defaults
  set to measured-best values.

The original ComfyUI-H3-ContactSheet repo remains for existing installs;
this is the consolidated home going forward.
"""
from .contact_sheet import H3ContactSheetExtension
from .motion import TIMESMEAR_CLASS_MAPPINGS, TIMESMEAR_DISPLAY_MAPPINGS


def comfy_entrypoint():
    return H3ContactSheetExtension()


NODE_CLASS_MAPPINGS = dict(TIMESMEAR_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS = dict(TIMESMEAR_DISPLAY_MAPPINGS)
