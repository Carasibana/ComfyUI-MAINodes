"""Oracle outputs -> a GENERATION-DENSITY envelope (the plan's first draft).

The two oracles already shipped in the pack disagree in both directions and
neither is a superset of the other (RESULTS.md, 7 scenes), so the validated
combination is a DIVISION OF LABOUR, not an average:

  - JERK reads the clip's own motion out of its latent. It is what catches
    a fast PROP crossing the frame (kitsune_dash token 21: motion rank
    0.974 while jitter read 0.040).
  - INDECISION reads the model's own uncertainty out of two x0 taps. It
    catches SELF-SALIENCY — detail the model is still arguing with itself
    about — where frame-diff has nothing to say (+0.51 with static detail
    on the quietest third of token-times).

So: propose max(jerk, indecision) on rank-normalized profiles. Taking the
max is deliberate — either oracle raising its hand is a reason to spend
frames there, and averaging would let a confident zero from one signal veto
the other.

The envelope is emitted in the SEMANTIC unit (generated frames per world
frame), never as a hold map: turning density into legal integer holds is
the compiler's job. Density is not playback speed - the recovery step puts
the clip back on the world clock.
"""
from .. import schema
from . import gridlaw as G
from .recipe import PROFILE


def _rank01(xs):
    """Rank-normalize to 0..1 so two differently-scaled profiles compare."""
    n = len(xs)
    if n == 0:
        return []
    if n == 1:
        return [0.0]
    order = sorted(range(n), key=lambda i: xs[i])
    out = [0.0] * n
    for r, i in enumerate(order):
        out[i] = r / float(n - 1)
    return out


def parse_profile(profile_str):
    """H3JerkOracle / H3IndecisionOracle emit their per-token profile as a
    space-separated string. Same parse for both."""
    if not profile_str:
        return []
    return [float(v) for v in str(profile_str).split()]


def combine(jerk=None, indecision=None):
    """-> per-token 0..1 saliency, and which sources contributed."""
    js = _rank01(parse_profile(jerk)) if jerk else []
    ix = _rank01(parse_profile(indecision)) if indecision else []
    if js and ix and len(js) != len(ix):
        raise ValueError(f"profiles disagree on token count: {len(js)} vs "
                         f"{len(ix)} — they must come from the same clip")
    if js and ix:
        return [max(a, b) for a, b in zip(js, ix)], "jerk+indecision"
    if js:
        return js, "jerk"
    if ix:
        return ix, "indecision"
    raise ValueError("no oracle profile given")


def envelope_from_saliency(sal, frames, ceiling=4.0, floor_ratio=1.0,
                           threshold=0.75, gamma=1.0):
    """Per-token saliency -> a ratio envelope on the WORLD frame axis.

    threshold: saliency below this stays at real time (the same idea as the
    oracle's q quantile, expressed on the rank scale so it means the same
    thing for both signals). Above it, the ratio ramps linearly from
    floor_ratio to the lane ceiling, which is what makes the drawn curve
    editable: a human sees a shape, not a step function.

    One control point per LATENT TOKEN (spec: envelopes quantize to latent
    frames, and the UI draws that quantization honestly).
    """
    pts = []
    for t, s in enumerate(sal):
        f = G.tok_start_frame(t)
        if f > frames - 1:
            break
        if s <= threshold:
            r = floor_ratio
        else:
            w = (s - threshold) / max(1e-9, 1.0 - threshold)
            r = floor_ratio + (ceiling - floor_ratio) * (w ** gamma)
        pts.append([f, round(float(r), 3)])
    if not pts:
        pts = [[0, floor_ratio]]
    if pts[0][0] != 0:
        pts.insert(0, [0, floor_ratio])
    if pts[-1][0] != frames - 1:
        pts.append([frames - 1, pts[-1][1]])
    return pts


def propose_density_lane(frames, jerk=None, indecision=None, ceiling=4.0,
                         threshold=0.75, gamma=1.0):
    sal, src = combine(jerk, indecision)
    env = envelope_from_saliency(sal, frames, ceiling=ceiling,
                                 threshold=threshold, gamma=gamma)
    return schema.generation_density_lane(env, ceiling=ceiling,
                                          proposer=f"oracle:{src}")


# Legacy alias for one release.
propose_temporal_lane = propose_density_lane


def challenge_ceiling(peak_ratio):
    """If a proposal wants more dilation than the recipe's measured sweet
    spot, that is a CHALLENGE to the recipe profile, not a licence to
    override it: the compiler runs the profile value and the plan carries a
    flag for eyes (amendment 2, item 1)."""
    d_max = PROFILE.get("d_max_sweet_spot")
    if peak_ratio <= d_max:
        return None
    return PROFILE.challenges(
        "d_max_sweet_spot", peak_ratio, by="oracle proposer",
        why=f"the envelope asks for {peak_ratio:.2f}x, above the measured "
            f"sweet spot {d_max}x; above it the pass gets expensive and the "
            f"de-rope stops improving in the runs we have")
