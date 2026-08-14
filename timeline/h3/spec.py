"""H3ModelSpec — STRUCTURAL facts about the MiniMax-H3 model/VAE family.

The epistemic boundary (amendment 2, item 1): everything in this file is a
property of the artifact. The compiler RELIES on these; nothing may
override them, no proposer may challenge them, and a "measurement" that
contradicts one of them is a bug in the measurement. Values that came out
of an EXPERIMENT and could be overturned by a better experiment live in
recipe.py instead, where they are challengeable.

Every field cites where the fact comes from: [SRC] = read out of the
implementation, [AUTH] = upstream statement.

Stdlib only.
"""

SPEC_VERSION = "h3-1.0"


class H3ModelSpec(object):
    """One frozen description of the model family the compiler targets."""

    version = SPEC_VERSION
    family = "minimax-h3"

    # Legal pixel lengths are 17k+5: the VAE encodes video in 17-frame
    # groups plus a 5-frame lead-in. [SRC comfy_extras/nodes_minimax_h3.py]
    temporal_group_size = 17
    length_offset = 5
    min_legal_length = 5

    # A 17-frame group is 5 latent time tokens covering (1,4,4,4,4) frames,
    # so every 5th token is a singleton on a 17-multiple frame (the native
    # keyframe anchors). [SRC comfy/ldm/minimax/model.py:30-91]
    tokens_per_group = 5
    token_frame_pattern = (1, 4, 4, 4, 4)
    token_leading_offsets = (0, 1, 5, 9, 13)

    # MM-RoPE's temporal coordinate is PHYSICAL: 5/3 RoPE units per world
    # frame. [SRC comfy/ldm/minimax/model.py]
    rope_units_per_frame = 5.0 / 3.0

    # One frame rate. A 30 fps source goes in as 24 fps and plays slower.
    fps = 24

    # The audio latent runs a 40 Hz clock against video's 24 fps, so most
    # legal lengths carry up to +-12.5 ms of audio-length error, and chained
    # segments accumulate it into visible A/V drift. [MEAS, minted ops rule]
    #
    # WHICH lengths are exact is DERIVED from these two clocks, not stored
    # (review round 3, item 6): a length is audio-exact iff frames*40/24 is
    # an integer, i.e. frames % 3 == 0, intersected with the 17k+5 grid.
    # It is arithmetic on the artifact, so it belongs here and can never
    # drift from the clocks it comes from.
    audio_latent_fps = 40

    # Spatial: 16 image pixels per latent cell, legal latent sizes are
    # multiples of 2 cells (= 32 px). [SRC motion.py LATENT_CELL/LATENT_GRID]
    latent_cell_px = 16
    latent_grid_cells = 2

    supported_conditioning = ("prompt", "image_guide", "reference_images",
                              "audio_reference", "repaint_mask")

    # ---- derived structure (arithmetic on the facts above, not opinions)

    def is_legal_length(self, n):
        n = int(n)
        return n >= self.min_legal_length and (
            (n - self.length_offset) % self.temporal_group_size == 0)

    def token_count(self, frames):
        """Latent time tokens for a legal pixel length."""
        frames = int(frames)
        assert self.is_legal_length(frames), (
            f"{frames} is not on the {self.temporal_group_size}k+"
            f"{self.length_offset} grid")
        return ((frames - self.length_offset) // self.temporal_group_size
                * self.tokens_per_group + 2)

    def latent_cells(self, width, height):
        return (int(width) // self.latent_cell_px,
                int(height) // self.latent_cell_px)

    def audio_latent_frames(self, frames):
        """Audio-latent length for a pixel length, as a FRACTION, so the
        exactness question is answerable rather than rounded away."""
        return int(frames) * self.audio_latent_fps, self.fps

    def audio_is_exact(self, frames):
        num, den = self.audio_latent_frames(frames)
        return self.is_legal_length(frames) and num % den == 0

    def audio_exact_lengths(self, max_frames=200):
        """Derived, not stored: [39, 90, 141, 192] for the default bound."""
        return tuple(n for n in range(self.min_legal_length, int(max_frames) + 1)
                     if self.audio_is_exact(n))

    def audio_error_ms(self, frames):
        """How far off an inexact length is, in milliseconds of audio."""
        num, den = self.audio_latent_frames(frames)
        rem = num % den
        off = min(rem, den - rem) if rem else 0
        return off / float(self.audio_latent_fps) * 1000.0


SPEC = H3ModelSpec()
