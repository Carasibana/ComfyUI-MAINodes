"""The ONE access point to motion.py's grid law. Nothing here re-derives it.

The compiler needs expand_hold_map_to_end, the 17k+5 helpers, the token
clock and the window planner. Those are already implemented, tested and
shipped in ComfyUI-MAINodes/motion.py, and a second implementation would
drift from the nodes users actually run — the packet's rule: reuse the map
machinery, never fork the math. So this module imports them and re-exports
them under stable names.

Import strategy: normal relative import inside ComfyUI; file-path load when
the pack is used outside a package (the test suites do exactly that, see
tests/test_expand_to_end.py).

CAVEAT (deviation, deliberate): motion.py imports numpy and torch at module
level, so importing this module pulls them in. The rest of timeline/ is
stdlib-only and does not import this one — only timeline/h3/ does. Making
the grid law itself importable without torch means moving those ~20 pure
functions out of motion.py, which is a change to a file that is currently
dirty under another writer; that migration is left to T2 and is the reason
this shim exists rather than a copy.
"""
import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_MOTION_PY = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "motion.py")


def _load_motion():
    try:                                   # inside ComfyUI: the package path
        from .. import motion as m         # noqa: F401
        return m
    except Exception:
        pass
    for name in ("mainodes_motion", "motion"):
        if name in sys.modules and hasattr(sys.modules[name],
                                           "expand_hold_map_to_end"):
            return sys.modules[name]
    spec = importlib.util.spec_from_file_location("mainodes_motion", _MOTION_PY)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mainodes_motion"] = mod
    spec.loader.exec_module(mod)
    return mod


motion = _load_motion()

# --- grid law -------------------------------------------------------------
LEGAL_STEP = motion.LEGAL_STEP                     # 17
MAX_END_TAIL = motion.MAX_END_TAIL                 # the tail guard
legal_ceil = motion._legal_ceil                    # snap up (floors at 39)
grid_floor = motion._grid_floor
grid_ceil = motion._grid_ceil
token_count = motion._token_count
snap_holds = motion._snap_holds
expand_hold_map_to_end = motion.expand_hold_map_to_end
hold_runs = motion._hold_runs
hold_runs_str = motion._hold_runs_str

# --- token clock ----------------------------------------------------------
tok_start_frame = motion._tok_start_frame
frame_token = motion._frame_token
token_frame_spans = motion._token_frame_spans
token_centers = motion._token_centers
is_tok_start = motion._is_tok_start
TOK_OFFSETS = motion.TOK_OFFSETS

# --- planning -------------------------------------------------------------
temporal_insert_map = motion.temporal_insert_map
plan_windows = motion._plan_windows
cut_is_cold = motion._cut_is_cold
grid_grow = motion._grid_grow
seg_holds = motion._seg_holds
audio_latent_t = motion._audio_latent_t


def is_legal(n):
    n = int(n)
    return n >= 5 and (n - 5) % LEGAL_STEP == 0


def token_count_exact(frames):
    """token_count() routes through legal_ceil's k>=2 clamp, which prices a
    legal 5- or 22-frame clip as 39 frames. For a length already known to be
    legal, index the grid directly (the H3VideoFit precedent)."""
    frames = int(frames)
    assert is_legal(frames), f"{frames} is not on the 17k+5 grid"
    return (frames - 5) // LEGAL_STEP * 5 + 2
