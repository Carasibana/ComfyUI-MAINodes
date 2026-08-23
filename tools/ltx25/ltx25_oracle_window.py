#!/usr/bin/env python3
"""Read the SAVED H3 oracle reference for a clip and hand back the span that
actually needs de-roping, expressed on LTX-2.5's frame grid.

The reference is the H3 final latent dumped by the x0-tap bench run (not an
mp4 round trip - the latent is the thing the oracle was designed to read).
The profile and hot-span logic are imported from the shipped MAINodes oracle
so this agrees with H3JerkOracle by construction rather than by reimplementation.

    python3 ltx25_oracle_window.py --latent .../swordspin_fast/final.pt \
        --frames 73 --fps 24 [--q 0.75]
"""
import argparse, json, os, sys
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # the MAINodes root
import motion as M  # noqa: E402


def legal_floor(n):
    return 1 + 8 * ((max(1, int(n)) - 1) // 8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent", required=True)
    ap.add_argument("--frames", type=int, required=True, help="world frame count of the clip")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--channels", type=int, default=24)
    ap.add_argument("--spatial", type=int, default=64)
    ap.add_argument("--q", type=float, default=0.75)
    ap.add_argument("--d-max", type=int, default=4)
    ap.add_argument("--bridge", type=int, default=8)
    ap.add_argument("--json-out", default="")
    a = ap.parse_args()

    blob = torch.load(a.latent, map_location="cpu")
    v = blob["video"] if isinstance(blob, dict) else blob
    v = v.reshape(-1).float()
    per_frame = a.channels * a.spatial * a.spatial
    T = v.numel() // per_frame
    z = v[: T * per_frame].reshape(1, a.channels, T, a.spatial, a.spatial)

    prof = M._jerk_profile(z)
    holds, segs, w0_lat, wlen_lat, tok_d = M._profile_to_plan(
        prof, a.frames, a.q, a.d_max, True, a.bridge)

    hot = [i for i, h in enumerate(holds) if h > 1]
    f0, f1 = (hot[0], hot[-1]) if hot else (0, a.frames - 1)
    # LTX needs 8k+1; grow the span outward, then clamp to the clip
    span = f1 - f0 + 1
    n = min(legal_floor(span + 8), legal_floor(a.frames))
    start = max(0, min(f0 - (n - span) // 2, a.frames - n))

    out = dict(latent_shape=list(z.shape), profile=[round(float(x), 3) for x in prof],
               segments=segs, hot_frames=[f0, f1], holds=holds,
               ltx_start_frame=int(start), ltx_frames=int(n),
               peak_latent_frame=int(np.argmax(prof)),
               contrast=round(float(prof.max() / max(prof.mean(), 1e-8)), 3))
    print(json.dumps({k: v for k, v in out.items() if k not in ("profile", "holds")}, indent=1))
    print("profile:", " ".join(f"{x:.2f}" for x in prof))
    print("holds  :", "".join(str(min(h, 9)) for h in holds))
    if a.json_out:
        json.dump(out, open(a.json_out, "w"), indent=1)
        print("wrote", a.json_out)


if __name__ == "__main__":
    main()
