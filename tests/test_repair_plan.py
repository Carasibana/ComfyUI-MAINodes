#!/usr/bin/env python3
"""H3 Repair Plan / H3 Repair Splice arithmetic. CPU only, no GPU, no models,
no ComfyUI process.

  pytest tests/test_repair_plan.py        or        python tests/test_repair_plan.py

The validation cell is the measured one from EXTENSION_V0_2026-08-23 (the
"Repair trial" section): a 90-frame clip, bad frames 45-47 (0-based; the
operator's file-numbered 46-48), hold 3, a hard cut between 52 and 53. The
token snap gives 43..50, the cut at 53 pulls the mask through to 55, and the
splice keeps repaired 43..52 and hands back to the original at the cut.

Exit code 0 = pass.
"""
import importlib.util
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.realpath(__file__))   # realpath: runnable via a symlink
ROOT = os.path.dirname(HERE)

# loaded by PATH on purpose: the pack root must stay off sys.path (its
# __init__.py would otherwise be importable as a module called `__init__`)
_spec = importlib.util.spec_from_file_location(
    "mainodes_h3_repair", os.path.join(ROOT, "h3_repair.py"))
rp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rp)

PLAN = rp.H3RepairPlan()
SPLICE = rp.H3RepairSplice()


def _plan(length=90, bad_start=45, bad_end=47, hold=3, fps=24, cuts="", reach=8,
          images=None, width=0, height=0):
    hm, ranges, mask, plan, report = PLAN.plan(
        length, bad_start, bad_end, hold, fps, cuts, reach, images, width, height)
    return json.loads(hm), ranges, mask, json.loads(plan), report


def test_validation_cell():
    """length 90, bad 45-47, hold 3, cut 53, reach 8."""
    hm, ranges, mask, p, report = _plan(cuts="53")
    assert (p["regen_lo"], p["regen_hi"]) == (43, 55), (p["regen_lo"], p["regen_hi"])
    assert (p["splice_lo"], p["splice_hi"]) == (43, 52), (p["splice_lo"], p["splice_hi"])
    assert p["cut_used"] == 53, p["cut_used"]
    assert ranges == "43-55:3", ranges
    assert hm["holds"] == [1] * 43 + [3] * 13 + [1] * 34, rp._mo._hold_runs_str(hm["holds"])
    assert hm["world_len"] == 90 and len(hm["holds"]) == 90
    assert p["hold"] == 3 and p["length"] == 90
    assert p["dilated_lo"] == 43, p["dilated_lo"]
    assert p["dilated_hi"] == 43 + 13 * 3 - 1 == 81, p["dilated_hi"]
    assert p["dilated_len"] == sum(hm["holds"]) == 116, p["dilated_len"]
    # H3 Time Smear puts this on the 17k+5 grid by extending the LAST hold
    assert p["resolved"]["dilated_padded"] == 124 and p["resolved"]["dilated_pad"] == 8
    assert "exit on cut 53" in report, report
    print(report)


def test_no_cut():
    hm, ranges, mask, p, report = _plan(cuts="")
    assert (p["regen_lo"], p["regen_hi"]) == (43, 50), (p["regen_lo"], p["regen_hi"])
    assert (p["splice_lo"], p["splice_hi"]) == (43, 50), (p["splice_lo"], p["splice_hi"])
    assert p["cut_used"] == -1
    assert ranges == "43-50:3"
    assert hm["holds"] == [1] * 43 + [3] * 8 + [1] * 39
    assert "no cut within 8 frames" in report, report
    # a cut too far away is not used either (53 is 3 past 50, so reach 2 misses)
    _, _, _, p2, _ = _plan(cuts="53", reach=2)
    assert (p2["regen_hi"], p2["splice_hi"], p2["cut_used"]) == (50, 50, -1), p2


def test_straddle():
    """bad 50-53 with a cut at 53: the span already contains the cut."""
    _, ranges, _, p, report = _plan(bad_start=50, bad_end=53, cuts="53")
    # token(50) = 47..50, token(53) = 52..55
    assert (p["regen_lo"], p["regen_hi"]) == (47, 55), (p["regen_lo"], p["regen_hi"])
    assert p["splice_hi"] == 52 and p["splice_lo"] == 47 and p["cut_used"] == 53
    assert ranges == "47-55:3"


def test_group_boundary():
    """bad 16-17 spans the (1,4,4,4,4) group boundary: token 13..16 and the
    singleton token at 17."""
    _, ranges, _, p, _ = _plan(bad_start=16, bad_end=17)
    assert (p["regen_lo"], p["regen_hi"]) == (13, 17), (p["regen_lo"], p["regen_hi"])
    assert ranges == "13-17:3"
    assert p["resolved"]["token_span_lo"] == [13, 16] and p["resolved"]["token_span_hi"] == [17, 17]
    # every 5th token is a singleton on a 17-multiple, so 17 alone is one token
    _, _, _, p2, _ = _plan(bad_start=17, bad_end=17)
    assert (p2["regen_lo"], p2["regen_hi"]) == (17, 17), (p2["regen_lo"], p2["regen_hi"])


def test_regen_mask():
    """H3 V2V Init's time_varying branch takes a (T, H, W) MASK and folds T
    onto the token clock (motion._tokenize_mask_time), resampling when T is
    not the clip length. So the mask must ride the PADDED dilated length, the
    one H3 Time Smear actually emits."""
    _, _, mask, p, _ = _plan(cuts="53", width=64, height=32)
    T = p["resolved"]["dilated_padded"]
    assert tuple(mask.shape) == (T, 32, 64) == (124, 32, 64), mask.shape
    lo, hi = p["dilated_lo"], p["dilated_hi"]
    assert float(mask[:lo].max()) == 0.0 and float(mask[hi + 1:].max()) == 0.0
    assert float(mask[lo:hi + 1].min()) == 1.0
    assert int(mask.sum()) == (hi - lo + 1) * 32 * 64 == 39 * 32 * 64
    # and it survives the fold onto the token clock with the right tokens hot
    t_lat = (T - 5) // 17 * 5 + 2
    tok = rp._mo._tokenize_mask_time(mask, t_lat, T)
    hot = [t for t in range(t_lat) if float(tok[t].max()) > 0]
    spans = rp._mo._token_frame_spans(t_lat)
    assert hot == [t for t, (a, b) in enumerate(spans) if not (b < lo or a > hi)], hot
    # width/height come from images when not given
    imgs = torch.rand(90, 24, 48, 3)
    _, _, m2, _, _ = _plan(cuts="53", images=imgs)
    assert tuple(m2.shape) == (124, 24, 48), m2.shape


def test_mask_at_clip_end():
    """When the regen span runs to the last world frame, H3 Time Smear's grid
    pad is extra copies of THAT frame, so the pad belongs to the span."""
    _, _, mask, p, _ = _plan(bad_start=88, bad_end=89)
    assert (p["regen_lo"], p["regen_hi"]) == (86, 89)
    T = p["resolved"]["dilated_padded"]
    assert float(mask[T - 1].min()) == 1.0 and float(mask[p["dilated_lo"] - 1].max()) == 0.0
    assert p["resolved"]["mask_hi"] == T - 1


def test_entry_quiet_report():
    """images are report-only: a busy entry frame is named, never moved."""
    imgs = torch.zeros(90, 8, 8, 3)
    for f in range(90):                      # a still clip with one violent frame at 43
        imgs[f] = 0.5
    imgs[43] = 0.9
    _, _, _, p, report = _plan(cuts="53", images=imgs)
    assert p["splice_lo"] == 43, "the entry did NOT move"
    assert p["resolved"]["entry_quiet"] is False
    assert "BUSY" in report and "quieter frame within 4" in report, report
    ramp = torch.stack([torch.full((8, 8, 3), 0.001 * f) for f in range(90)])
    _, _, _, p2, report2 = _plan(cuts="53", images=ramp)
    assert p2["resolved"]["entry_quiet"] is True and "QUIET" in report2


def test_splice_node():
    """Frame-exact: original outside, repaired inside, and the seam numbers."""
    L = 90
    orig = torch.stack([torch.full((6, 8, 3), 0.30 + 0.001 * f) for f in range(L)])
    orig[53:] += 0.40                        # a hard cut between 52 and 53
    rep_clip = torch.stack([torch.full((6, 8, 3), 0.31 + 0.001 * f) for f in range(L)])
    rep_clip[53:] += 0.10                    # the model reinvented the next shot
    _, _, _, p, _ = _plan(cuts="53")
    plan_s = json.dumps(p)
    audio = {"waveform": torch.zeros(1, 2, 90 * 32000 // 24), "sample_rate": 32000}
    out, aud, report = SPLICE.splice(orig, rep_clip, plan_s, audio)
    assert out.shape == orig.shape
    assert torch.equal(out[:43], orig[:43]) and torch.equal(out[53:], orig[53:])
    assert torch.equal(out[43:53], rep_clip[43:53])
    assert aud is audio and "audio passthrough" in report
    assert "outside the splice: max abs pixel difference from the original 0 " in report, report
    # the exit seam IS the cut: output and original are close to each other there
    d_out = rp._luma_deltas(out)
    d_org = rp._luma_deltas(orig)
    assert abs(float(d_out[53]) - float(d_org[53])) / float(d_org[53]) < 0.05
    assert float(d_out[43]) < 5.0            # entry is a tenth of a level step
    assert "entry 42->43" in report and "exit 52->53" in report, report
    print(report)
    # non-CUDA CPU tensors out (H3ExactRecover trap: cuda frames crash ImageBatch)
    assert out.device.type == "cpu"


def test_splice_is_exact_outside():
    """Random content, every splice bound: outside the splice the output is
    bit-identical, inside it is the repaired clip, bit-identical."""
    torch.manual_seed(0)
    L = 90
    orig = torch.rand(L, 5, 7, 3)
    rep_clip = torch.rand(L, 5, 7, 3)
    for bad in ((45, 47), (16, 17), (0, 1), (86, 89)):
        _, _, _, p, _ = _plan(bad_start=bad[0], bad_end=bad[1], cuts="53")
        out, _, report = SPLICE.splice(orig, rep_clip, json.dumps(p))
        lo, hi = p["splice_lo"], p["splice_hi"]
        assert torch.equal(out[:lo], orig[:lo]) and torch.equal(out[hi + 1:], orig[hi + 1:]), bad
        assert torch.equal(out[lo:hi + 1], rep_clip[lo:hi + 1]), bad
        assert int((out - orig).abs().sum() > 0), bad


def test_expand_to_end_warning():
    """A regen span near the end leaves a short rate-1 tail, which H3 Time
    Smear's expand_to_end would rewrite - moving every dilated coordinate.
    The plan has to say so."""
    _, _, _, p_far, rep_far = _plan(bad_start=45, bad_end=47)
    assert "expand_to_end" not in rep_far, rep_far      # 39-frame tail = rest, no rewrite
    _, _, _, p_near, rep_near = _plan(bad_start=70, bad_end=71)
    assert "expand_to_end WOULD REWRITE" in rep_near, rep_near
    assert "Set expand_to_end OFF" in rep_near


def test_off_grid_length_is_named():
    _, _, _, p, report = _plan(length=100, bad_start=45, bad_end=47)
    assert "not on the 17k+5 grid" in report and "H3 generates 107" in report, report
    assert p["resolved"]["legal_length"] == 107


def test_degenerate_cut_rejected():
    """A cut AT the span start would leave nothing repaired; ignore it."""
    _, _, _, p, report = _plan(bad_start=43, bad_end=44, cuts="43")
    assert p["cut_used"] == -1 and p["splice_hi"] == 46, p
    assert "would leave nothing repaired" in report, report


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print("ALL PASS")
