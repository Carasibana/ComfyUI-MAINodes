#!/usr/bin/env python3
"""Unit test for the H3 delta injection (T6), synthetic and offline.

  <venv>/bin/python tests/test_concept_lab_inject.py
      torch is allowed, ComfyUI is not imported, no GPU and no model. The
      injector's whole claim is arithmetic on a block state, and arithmetic
      that can only be checked by spending a render is not checked.

The stand-ins are the tap suite's, plus one difference that matters: this
FakePatcher stores block patches the way comfy really does
(model_patcher.py:93-112, ONE callable per (name, block_name, index)), because
the tap-plus-inject collision this suite tests only exists in that structure.

  1. THE PACK. Keys, shapes, sidecar and provenance are read strictly, and
     every refusal names the numbers.
  2. ZERO IS THE PLUMBING CONTROL. The patch runs and the state is
     bit-identical to the block's own output.
  3. FRAME MODE. Exactly alpha*d per latent frame, on the video rows and
     nowhere else, in the block's OWN dict.
  4. SEPARABLE MODE. frame + patch - grand mean, once.
  5. CONTROLS. time_shuffle is a permutation of the same rows, sign_flip is
     the negation.
  6. STEP SELECTION. captured_only fires on the pack's steps; nearest_all
     fires every step and maps ties to the earlier entry.
  7. GEOMETRY REFUSES FIRST. A run whose (latent_t, frame_rows) or hidden
     width differs from the pack's refuses in the wrapper, before a block.
  8. COND ONLY. The unconditional forward is left alone.
  9. TAP + INJECT ON ONE CLONE. Overlapping blocks refuse; a tap at a LATER
     block records the injected state.
"""
import json
import os
import shutil
import sys
import tempfile

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concept_lab import types as T                                # noqa: E402
from concept_lab.backends import h3_inject                        # noqa: E402
from concept_lab.backends.h3_inject import (DeltaPack,            # noqa: E402
                                            InjectError,
                                            InjectSession)
from concept_lab.backends.h3_tap import TapSession                # noqa: E402

_fails = []

HIDDEN = 16
# text 10 | ref_img 24 | audio 8 | video 18   (latent_t 3 x frame_rows 6)
SEGS = ((0, 10, "text"), (10, 34, "ref_img"), (34, 42, "audio"),
        (42, 60, "video"))
SIGNATURE = (10, 3, 6, 4, 4)   # text_len, latent_t, latent_h, latent_w, audio_t
LATENT_T, FRAME_ROWS = 3, 6
VIDEO = (42, 60)
SCHED = [float(v) for v in torch.linspace(1.0, 0.0, 26).tolist()]
CTX = (torch.arange(1 * 10 * HIDDEN, dtype=torch.float32)
       .reshape(1, 10, HIDDEN) / 31.0)
CAPTURED = (4, 12, 20)


def ck(cond, label, detail=""):
    if not cond:
        _fails.append(label)
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}"
          + (f"  {detail}" if detail else ""))


def refuses(fn, label, must_contain=()):
    try:
        fn()
    except InjectError as e:
        missing = [m for m in must_contain if str(m) not in str(e)]
        ck(not missing, label, str(e)[:110] if not missing
           else f"message missing {missing}: {str(e)[:130]}")
        return
    except Exception as e:
        ck(False, label, f"wrong exception {type(e).__name__}: {e}")
        return
    ck(False, label, "did not refuse")


# ------------------------------------------------------------- stand-ins
class FakePatcher:
    """comfy's patch table, in miniature: one callable per (name, block)."""

    def __init__(self, model_options=None):
        self.model_options = model_options or {"transformer_options": {}}
        self.wrappers = []
        self.clones = 0

    def clone(self):
        self.clones += 1
        to = dict(self.model_options.get("transformer_options") or {})
        pr = to.get("patches_replace")
        if pr is not None:
            to["patches_replace"] = {k: dict(v) for k, v in pr.items()}
        c = FakePatcher({"transformer_options": to})
        c.wrappers = list(self.wrappers)
        return c

    def add_wrapper_with_key(self, wrapper_type, key, wrapper):
        self.wrappers.append((wrapper_type, key, wrapper))

    def set_model_patch_replace(self, patch, name, block_name, number):
        to = self.model_options["transformer_options"]
        pr = to.setdefault("patches_replace", {})
        pr.setdefault(name, {})[(block_name, number)] = patch

    @property
    def replaces(self):
        pr = ((self.model_options.get("transformer_options") or {})
              .get("patches_replace") or {})
        return {(name, bn, i): p for name, d in pr.items()
                for (bn, i), p in d.items()}


class FakeLayout:
    def __init__(self, segments=SEGS, signature=SIGNATURE):
        self.segments = segments
        self.seq_len = segments[-1][1]
        self.signature = signature
        self.position_ids = torch.arange(self.seq_len * 3,
                                         dtype=torch.float64).reshape(-1, 3)


def topts(step=12, cond_or_uncond=(0,), sched=SCHED):
    return {"sigmas": torch.tensor([sched[step]]),
            "sample_sigmas": torch.tensor(sched),
            "cond_or_uncond": list(cond_or_uncond)}


def base_img(layout=None, hidden=HIDDEN):
    n = (layout or FakeLayout()).seq_len
    return torch.arange(n * hidden, dtype=torch.float32).reshape(-1,
                                                                 hidden) / 97.0


def make_pack(root, name="pack", blocks=(0,), steps=CAPTURED,
              latent_t=LATENT_T, frame_rows=FRAME_ROWS, hidden=HIDDEN,
              with_patch=True, seed=0, sidecar="stem", keys=None, meta_over=None):
    """A delta pack on disk, the way the main session builds one with numpy.

    The two maps are drawn from ONE synthetic state, not independently:
    `frame_mean` is its patch average and `patch_mean` its time average, so
    they share a grand mean exactly as a real pack's do. A pack whose halves
    disagree about their grand mean would make the separable identity in [4]
    untestable.
    """
    rng = torch.Generator().manual_seed(seed)
    tensors = {}
    for b in blocks:
        for s in steps:
            state = torch.randn(latent_t, frame_rows, hidden, generator=rng)
            tensors[f"b{int(b):02d}_s{int(s):02d}/frame_mean"] = \
                state.mean(dim=1)
            if with_patch:
                tensors[f"b{int(b):02d}_s{int(s):02d}/patch_mean"] = \
                    state.mean(dim=0)
    if keys is not None:
        tensors = keys
    path = os.path.join(root, name + ".safetensors")
    save_file(tensors, path)
    meta = {"_type": "concept_delta_pack", "layout_signature": None,
            "frame_rows": frame_rows, "hidden": hidden, "latent_t": latent_t,
            "steps": {"n_steps": 25, "captured": [int(s) for s in steps]},
            "sources": ["cap_aaaa", "cap_bbbb"], "n_subjects": 2,
            "label": name, "notes": "synthetic pack, tests only"}
    meta.update(meta_over or {})
    side = (os.path.splitext(path)[0] + ".json") if sidecar == "stem" \
        else (path + ".json")
    with open(side, "w") as fh:
        json.dump(meta, fh)
    return path


def session(pack, **kw):
    kw.setdefault("alpha", 1.0)
    return InjectSession(pack=pack, **kw)


def drive(sess, patched, step=12, cond_or_uncond=(0,), block=0, img=None,
          layout=None, sched=SCHED, wrapper=True):
    """One wrapper call plus one block-patch call, as the model would."""
    layout = layout or FakeLayout()
    to = topts(step, cond_or_uncond, sched)
    if wrapper:
        sess._dm_wrapper(lambda *a, **k: "PASSED THROUGH", None,
                         torch.tensor([1.0]), CTX, transformer_options=to,
                         minimax_payload={"layout": layout, "seed": 42})
    img = base_img(layout) if img is None else img
    returned = {}

    def original_block(args):
        d = {"img": args["img"] * 2 + 1}
        returned["d"] = d
        return d

    patch = patched.replaces[("dit", "double_block", block)]
    out = patch({"img": img, "t_emb": None, "mod_segments": None,
                 "rope_freqs": None, "transformer_options": to},
                {"original_block": original_block})
    return out, returned.get("d"), img


def expect(img, deltas):
    """The block's own output plus the deltas, computed independently."""
    h = (img * 2 + 1).clone()
    g = h[VIDEO[0]:VIDEO[1]].view(LATENT_T, FRAME_ROWS, HIDDEN)
    for d in deltas:
        g += d
    return h


# ---------------------------------------------------------------- 1
def check_pack(root):
    print("\n[1] the pack is read strictly and refuses by the numbers")
    p = make_pack(root, "p1")
    pack = DeltaPack.load(p)
    ck(pack.blocks == [0] and pack.steps == list(CAPTURED)
       and pack.latent_t == 3 and pack.frame_rows == 6
       and pack.hidden == HIDDEN,
       "a pack loads its blocks, steps and geometry",
       json.dumps({k: pack.summary()[k] for k in
                   ("blocks", "steps", "latent_t", "frame_rows", "hidden")}))
    raw = load_file(p)
    ck(torch.equal(pack.entries[(0, 12)]["frame_mean"],
                   raw["b00_s12/frame_mean"])
       and torch.equal(pack.entries[(0, 12)]["patch_mean"],
                       raw["b00_s12/patch_mean"]),
       "and the tensors are the file's own bytes with control='none'")
    ck(pack.summary()["sources"] == ["cap_aaaa", "cap_bbbb"]
       and pack.summary()["n_subjects"] == 2
       and pack.summary()["pack_n_steps"] == 25,
       "the sidecar's provenance rides in the report", pack.label)

    p2 = make_pack(root, "p2", sidecar="full")
    ck(DeltaPack.load(p2).hidden == HIDDEN,
       "<pack>.safetensors.json is accepted as well as <pack>.json")

    lonely = os.path.join(root, "lonely.safetensors")
    save_file({"b00_s04/frame_mean": torch.zeros(3, HIDDEN)}, lonely)
    refuses(lambda: DeltaPack.load(lonely),
            "a pack with no sidecar refuses, listing both spellings it looked "
            "for", ("lonely.json", "lonely.safetensors.json"))
    refuses(lambda: DeltaPack.load(os.path.join(root, "nope.safetensors")),
            "a missing pack refuses with the path", ("nope.safetensors",))

    bad_type = make_pack(root, "p3", meta_over={"_type": "something else"})
    refuses(lambda: DeltaPack.load(bad_type),
            "a sidecar with the wrong _type refuses",
            ("something else", "concept_delta_pack"))

    no_geo = make_pack(root, "p4")
    with open(os.path.splitext(no_geo)[0] + ".json") as fh:
        m = json.load(fh)
    m.pop("frame_rows")
    with open(os.path.splitext(no_geo)[0] + ".json", "w") as fh:
        json.dump(m, fh)
    refuses(lambda: DeltaPack.load(no_geo),
            "a sidecar with no frame_rows refuses: the guard cannot be "
            "inferred", ("frame_rows",))

    wrong_shape = make_pack(root, "p5", keys={
        "b00_s04/frame_mean": torch.zeros(5, HIDDEN)})
    refuses(lambda: DeltaPack.load(wrong_shape),
            "a tensor whose shape disagrees with the sidecar refuses, naming "
            "both", ("(5, 16)", "(3, 16)"))

    bad_key = make_pack(root, "p6", keys={"block0_step4": torch.zeros(3, 4)})
    refuses(lambda: DeltaPack.load(bad_key),
            "an unparseable key refuses", ("block0_step4",))

    orphan = make_pack(root, "p7", keys={
        "b00_s04/patch_mean": torch.zeros(FRAME_ROWS, HIDDEN)})
    refuses(lambda: DeltaPack.load(orphan),
            "a patch_mean with no frame_mean refuses", ("(0, 4)",))

    refuses(lambda: DeltaPack.load(p, control="sideways"),
            "an unknown control refuses", ("sideways",))

    flat = DeltaPack.load(make_pack(root, "p8", with_patch=False))
    refuses(lambda: session(flat, mode="separable"),
            "mode 'separable' on a pack with no patch_mean refuses",
            ("separable", "patch_mean"))
    refuses(lambda: session(pack, blocks=(0, 7)),
            "a block the pack does not carry refuses", ("[7]", "[0]"))
    refuses(lambda: session(pack, step_mode="whenever"),
            "an unknown step_mode refuses", ("whenever",))
    refuses(lambda: session(pack, mode="wobble"),
            "an unknown mode refuses", ("wobble",))


# ---------------------------------------------------------------- 2
def check_zero(root):
    print("\n[2] control='zero' runs the patch and moves nothing")
    pack = DeltaPack.load(make_pack(root, "z"), control="zero")
    s = session(pack)
    p = FakePatcher()
    patched = s.install(p)
    ck(p.clones == 1 and not p.wrappers and not p.replaces,
       "the caller's patcher is untouched, the CLONE is patched",
       f"clones={p.clones}")
    ck([k for _t, k, _w in patched.wrappers] == ["concept_lab_inject"],
       "the diffusion-model wrapper is keyed 'concept_lab_inject'")
    out, block_dict, img = drive(s, patched, step=12)
    ck(out is block_dict, "the patch returns the block's OWN dict",
       f"id(out)={id(out)} id(block)={id(block_dict)}")
    ck(torch.equal(out["img"], img * 2 + 1),
       "and the state is bit-for-bit the block's output")
    ck(s.calls_injected == 1 and s.report()["injected"] == {"b00_s12": 1},
       "while the injection code path really ran (this is the control that "
       "separates the plumbing from the delta)",
       json.dumps(s.report()["injected"]))
    ck(s.report()["delta_norms"]["b00_s12"]["frame"] == 0.0,
       "with a zero norm", json.dumps(s.report()["delta_norms"]))

    d = session(DeltaPack.load(make_pack(root, "z2")), enabled=False)
    dp = d.install(FakePatcher())
    out2, bd2, img2 = drive(d, dp, step=12)
    ck(out2 is bd2 and torch.equal(out2["img"], img2 * 2 + 1)
       and d.calls_injected == 0 and d.calls_seen == 0,
       "enabled=False installs the patches and adds nothing at all",
       f"calls_seen={d.calls_seen}")


# ---------------------------------------------------------------- 3
def check_frame(root):
    print("\n[3] frame mode adds alpha*d per latent frame, on video rows only")
    path = make_pack(root, "f")
    pack = DeltaPack.load(path)
    f = pack.entries[(0, 12)]["frame_mean"]
    for alpha in (1.0, -0.5, 2.5):
        s = session(pack, alpha=alpha)
        out, _bd, img = drive(s, s.install(FakePatcher()), step=12)
        want = expect(img, [alpha * f[:, None, :]])
        ck(torch.allclose(out["img"], want, atol=1e-6),
           f"alpha={alpha}: out == block(img) + alpha*d_frame on the video "
           f"rows", f"max|diff|={float((out['img'] - want).abs().max()):.3e}")
        untouched = torch.equal(out["img"][:VIDEO[0]],
                                (img * 2 + 1)[:VIDEO[0]])
        ck(untouched, f"alpha={alpha}: text, ref_img and audio rows are "
           f"bit-identical to the block's output")
    s = session(pack)
    drive(s, s.install(FakePatcher()), step=12)
    ck(abs(s.report()["delta_norms"]["b00_s12"]["frame"]
           - float(torch.linalg.vector_norm(f))) < 1e-5,
       "the report carries the norm of the delta it applied",
       json.dumps(s.report()["delta_norms"]))
    ck(s.report()["layout"] == {"latent_t": 3, "frame_rows": 6,
                               "video_span": [42, 60]},
       "and the geometry it measured off the run",
       json.dumps(s.report()["layout"]))


# ---------------------------------------------------------------- 4
def check_separable(root):
    print("\n[4] separable mode is frame + patch - grand mean")
    pack = DeltaPack.load(make_pack(root, "sep"))
    e = pack.entries[(0, 12)]
    f, q = e["frame_mean"], e["patch_mean"]
    grand = f.mean(dim=0, keepdim=True)
    alpha = 0.75
    s = session(pack, mode="separable", alpha=alpha)
    out, _bd, img = drive(s, s.install(FakePatcher()), step=12)
    want = expect(img, [alpha * f[:, None, :],
                        alpha * (q - grand)[None, :, :]])
    ck(torch.allclose(out["img"], want, atol=1e-6),
       "out == block(img) + alpha*(d_frame[t] + d_patch[p] - grand)",
       f"max|diff|={float((out['img'] - want).abs().max()):.3e}")
    # the grand mean is counted once: the t-mean of the added delta is the
    # patch map, and its p-mean is the frame map
    added = (out["img"] - (img * 2 + 1))[VIDEO[0]:VIDEO[1]].view(
        LATENT_T, FRAME_ROWS, HIDDEN) / alpha
    # atol is 1e-5 because this one divides the applied delta back out by
    # alpha, which scales fp32 round-off up with it
    ck(torch.allclose(added.mean(dim=1), f, atol=1e-5)
       and torch.allclose(added.mean(dim=0), q, atol=1e-5),
       "so averaging the added delta over patches gives frame_mean back, and "
       "over frames gives patch_mean back",
       f"max|diff|={float((added.mean(dim=1) - f).abs().max()):.3e}")
    ck(set(s.report()["delta_norms"]["b00_s12"]) == {"frame", "patch"},
       "and both halves are reported",
       json.dumps(s.report()["delta_norms"]["b00_s12"]))


# ---------------------------------------------------------------- 5
def check_controls(root):
    print("\n[5] time_shuffle is a permutation, sign_flip is the negation")
    path = make_pack(root, "c")
    plain = DeltaPack.load(path)
    f = plain.entries[(0, 12)]["frame_mean"]
    shuffled = [(seed, DeltaPack.load(path, control="time_shuffle",
                                      control_seed=seed)
                 .entries[(0, 12)]["frame_mean"]) for seed in range(5)]
    norms = sorted(float(v) for v in f.norm(dim=1))
    ck(all(sorted(float(v) for v in t.norm(dim=1)) == norms
           for _s, t in shuffled),
       "every time_shuffle seed keeps the per-frame norms, sorted",
       f"norms={[round(n, 4) for n in norms]}")
    ck(all(any(torch.equal(t[i], f[j]) for j in range(LATENT_T))
           for _s, t in shuffled for i in range(LATENT_T)),
       "and every row is one of the original rows")
    moved = [s for s, t in shuffled if not torch.equal(t, f)]
    ck(moved, "at least one seed really reorders (same energy, wrong timing)",
       f"seeds that moved it: {moved}")
    ck(torch.equal(DeltaPack.load(path, control="time_shuffle",
                                  control_seed=moved[0])
                   .entries[(0, 12)]["frame_mean"], shuffled[moved[0]][1]),
       "and the permutation is seeded, so an arm re-runs from its label",
       f"control_seed={moved[0]}")
    ck(torch.equal(DeltaPack.load(path, control="time_shuffle",
                                  control_seed=0)
                   .entries[(0, 12)]["patch_mean"],
                   plain.entries[(0, 12)]["patch_mean"]),
       "time_shuffle touches the time axis only: patch_mean is unchanged")

    flip = DeltaPack.load(path, control="sign_flip")
    ck(torch.equal(flip.entries[(0, 12)]["frame_mean"], -f)
       and torch.equal(flip.entries[(0, 12)]["patch_mean"],
                       -plain.entries[(0, 12)]["patch_mean"]),
       "sign_flip negates both halves, so the separable delta negates cleanly")
    s = session(flip)
    out, _bd, img = drive(s, s.install(FakePatcher()), step=12)
    ck(torch.allclose(out["img"], expect(img, [-f[:, None, :]]), atol=1e-6),
       "and a sign_flip arm subtracts what a 'none' arm adds")


# ---------------------------------------------------------------- 6
def check_steps(root):
    print("\n[6] step selection: captured_only vs nearest_all")
    pack = DeltaPack.load(make_pack(root, "st", blocks=(0, 5)))
    s = session(pack, step_mode="captured_only")
    patched = s.install(FakePatcher())
    for step in range(25):
        for block in (0, 5):
            drive(s, patched, step=step, block=block)
    ck(s.calls_seen == 50 and s.calls_injected == 6,
       "captured_only fires 3 times per block over a 25-step schedule",
       f"seen={s.calls_seen} injected={s.calls_injected}")
    ck(s.report()["injected"] == {"b00_s04": 1, "b00_s12": 1, "b00_s20": 1,
                                 "b05_s04": 1, "b05_s12": 1, "b05_s20": 1},
       "one per (block, captured step)", json.dumps(s.report()["injected"]))
    ck(s.report()["n_steps_run"] == 25 and s.report()["blocks"] == [0, 5],
       "the report names the schedule it ran on and the blocks it patched")

    n = session(pack, step_mode="nearest_all", blocks=(0,))
    np_ = n.install(FakePatcher())
    for step in range(25):
        drive(n, np_, step=step, block=0)
    ck(n.calls_injected == 25,
       "nearest_all fires on EVERY step", f"injected={n.calls_injected}")
    # 0..8 -> 4 (8 is a tie with 12 and goes to the earlier entry),
    # 9..16 -> 12 (16 ties with 20), 17..24 -> 20
    ck(n.report()["injected"] == {"b00_s04": 9, "b00_s12": 8, "b00_s20": 8},
       "using the nearest captured entry, ties to the earlier one",
       json.dumps(n.report()["injected"]))
    for step, want in ((0, 4), (8, 4), (9, 12), (16, 12), (17, 20), (24, 20)):
        ck(n._pick_step(0, step) == want,
           f"step {step} maps to captured step {want}",
           f"got {n._pick_step(0, step)}")


# ---------------------------------------------------------------- 7
def check_geometry(root):
    print("\n[7] the geometry guard refuses in the wrapper, before a block")
    pack = DeltaPack.load(make_pack(root, "g"))
    # a consistent layout with 4 latent frames: the pack describes 3
    four = FakeLayout(segments=((0, 10, "text"), (10, 34, "ref_img"),
                                (34, 42, "audio"), (42, 66, "video")),
                      signature=(10, 4, 6, 4, 4))
    s = session(pack)
    patched = s.install(FakePatcher())
    refuses(lambda: drive(s, patched, step=12, layout=four),
            "(latent_t, frame_rows) that differs from the pack's refuses, "
            "printing both pairs", ("(4, 6)", "(3, 6)"))
    ck(s.calls_seen == 0 and s.calls_injected == 0,
       "and NOTHING was injected: the refusal is in the wrapper, before any "
       "block ran", f"calls_seen={s.calls_seen}")

    s2 = session(pack)
    p2 = s2.install(FakePatcher())
    inconsistent = FakeLayout(signature=(10, 4, 6, 4, 4))   # 24 != 18 rows
    refuses(lambda: drive(s2, p2, step=12, layout=inconsistent),
            "a layout whose signature does not describe its own video rows "
            "refuses", ("latent_t=4", "frame_rows=6", "18 rows"))

    s3 = session(pack)
    p3 = s3.install(FakePatcher())
    refuses(lambda: s3._dm_wrapper(lambda *a, **k: None, None,
                                   torch.tensor([1.0]), CTX,
                                   transformer_options=topts(),
                                   minimax_payload={"refs": []}),
            "a payload with no layout refuses, listing the keys it carried",
            ("refs",))

    s4 = session(pack)
    p4 = s4.install(FakePatcher())
    no_video = FakeLayout(segments=((0, 10, "text"), (10, 28, "ref_img")),
                          signature=(10, 3, 6, 4, 0))
    refuses(lambda: drive(s4, p4, step=12, layout=no_video),
            "a layout with no video segment refuses", ("video",))

    s5 = session(pack)
    p5 = s5.install(FakePatcher())
    refuses(lambda: drive(s5, p5, step=12,
                          img=base_img(hidden=HIDDEN)[:, :8].contiguous()),
            "a model whose hidden width is not the pack's refuses",
            ("8", "16"))

    s6 = session(pack)
    p6 = s6.install(FakePatcher())
    drive(s6, p6, step=12)
    other = FakeLayout(segments=((0, 12, "text"), (12, 30, "video")),
                       signature=(12, 3, 6, 4, 0))
    refuses(lambda: s6._dm_wrapper(lambda *a, **k: None, None,
                                   torch.tensor([1.0]), CTX,
                                   transformer_options=topts(),
                                   minimax_payload={"layout": other,
                                                    "seed": 42}),
            "a second layout inside one injection refuses", ("60", "30"))

    s7 = session(pack)
    p7 = s7.install(FakePatcher())
    refuses(lambda: drive(s7, p7, step=0, sched=[1.0]),
            "a one-value schedule refuses rather than guessing the step",
            ("1 value(s)",))


# ---------------------------------------------------------------- 8
def check_cond_only(root):
    print("\n[8] cond_only leaves the unconditional forward alone")
    pack = DeltaPack.load(make_pack(root, "co"))
    f = pack.entries[(0, 12)]["frame_mean"]
    s = session(pack)
    patched = s.install(FakePatcher())
    out, _bd, img = drive(s, patched, step=12, cond_or_uncond=(1,))
    ck(torch.equal(out["img"], img * 2 + 1) and s.calls_injected == 0,
       "an uncond call is bit-identical to the block's output",
       f"injected={s.calls_injected}")
    out2, _bd2, img2 = drive(s, patched, step=12, cond_or_uncond=(0,))
    ck(torch.allclose(out2["img"], expect(img2, [f[:, None, :]]), atol=1e-6)
       and s.calls_injected == 1,
       "and the cond call at the same step is injected")
    ck(s.cond_or_uncond_seen == ["1", "0"],
       "both are noted even though only one is used",
       f"seen={s.cond_or_uncond_seen}")

    b = session(pack, cond_only=False)
    bp = b.install(FakePatcher())
    outb, _bdb, imgb = drive(b, bp, step=12, cond_or_uncond=(1,))
    ck(torch.allclose(outb["img"], expect(imgb, [f[:, None, :]]), atol=1e-6),
       "cond_only=False injects on the unconditional forward too")


# ---------------------------------------------------------------- 9
def check_tap_and_inject(root):
    print("\n[9] a tap and an inject on one model clone")
    pack = DeltaPack.load(make_pack(root, "ti", blocks=(0,)))
    f = pack.entries[(0, 12)]["frame_mean"]
    tap = T.TapSpec(blocks=(0, 5), steps=(12,), segments=("video",),
                    store="frame_mean")
    stack = T.ModelStackFingerprint(checkpoint="synthetic", backend="h3")

    def make_tap(name, blocks):
        return TapSession(
            tap=T.TapSpec(blocks=blocks, steps=(12,), segments=("video",),
                          store="frame_mean"),
            arm="R", variant_name="injected", seed=42, replicate=0,
            stack=stack, index_root=os.path.join(root, name),
            tensor_root=os.path.join(root, name + "_t"),
            sampler_label="euler/25")

    # (a) the collision
    t_all = TapSession(tap=tap, arm="R", variant_name="v", seed=42,
                       replicate=0, stack=stack,
                       index_root=os.path.join(root, "TI"),
                       tensor_root=os.path.join(root, "TIt"))
    clash_model = t_all.install(FakePatcher())
    refuses(lambda: session(pack, blocks=(0,)).install(clash_model),
            "an inject on a block the tap already patched refuses instead of "
            "silently replacing it",
            ("[0]", "DIFFERENT blocks"))

    # (b) tap at a LATER block sees the injected state
    t = make_tap("TJ", (5,))
    m = t.install(FakePatcher())
    s = session(pack, blocks=(0,))
    m2 = s.install(m)
    to = topts(step=12)
    lay = FakeLayout()
    payload = {"layout": lay, "seed": 42}
    s._dm_wrapper(lambda *a, **k: "PASSED THROUGH", None, torch.tensor([1.0]),
                  CTX, transformer_options=to, minimax_payload=payload)
    t._dm_wrapper(lambda *a, **k: "PASSED THROUGH", None, torch.tensor([1.0]),
                  CTX, transformer_options=to, minimax_payload=payload)
    img = base_img(lay)

    def run_block(h, block, patches):
        def original_block(args):
            return {"img": args["img"] * 2 + 1}
        p = patches.get(("dit", "double_block", block))
        args = {"img": h, "t_emb": None, "mod_segments": None,
                "rope_freqs": None, "transformer_options": to}
        return (p(args, {"original_block": original_block}) if p
                else original_block(args))["img"]

    patches = m2.replaces
    ck(sorted(patches) == [("dit", "double_block", 0), ("dit", "double_block", 5)],
       "one clone carries the inject's block 0 and the tap's block 5",
       f"{sorted(patches)}")
    h = run_block(img, 0, patches)
    h = run_block(h, 5, patches)
    want0 = expect(img, [f[:, None, :]])
    ck(torch.allclose(h, want0 * 2 + 1, atol=1e-6),
       "the injected state flows on into the later block",
       f"max|diff|={float((h - (want0 * 2 + 1)).abs().max()):.3e}")
    rep = t.finalize()
    ref = [x for x in t.tensors if x.key.endswith("frame_mean")][0]
    got = load_file(os.path.join(t.space.tensor_dir(), ref.path))["frame_mean"]
    rows = (want0 * 2 + 1)[VIDEO[0]:VIDEO[1]].reshape(LATENT_T, FRAME_ROWS,
                                                      HIDDEN)
    ck(torch.allclose(got, rows.mean(dim=1), atol=1e-6),
       "and the tap records the INJECTED state, which is how an injected run "
       "gets captured", f"capture {rep['capture_id']}")
    ck(s.report()["calls_injected"] == 1 and t.calls_recorded == 1,
       "one injection, one capture", json.dumps(s.report()["injected"]))


def main():
    root = tempfile.mkdtemp(prefix="concept_lab_inject_test_")
    try:
        print("=== concept_lab H3 inject (synthetic, torch, no comfy, no GPU)"
              " ===")
        ck("comfy" not in sys.modules and "folder_paths" not in sys.modules,
           "the injector imports without ComfyUI",
           f"wrapper key={h3_inject.WRAPPER_KEY!r}")
        check_pack(root)
        check_zero(root)
        check_frame(root)
        check_separable(root)
        check_controls(root)
        check_steps(root)
        check_geometry(root)
        check_cond_only(root)
        check_tap_and_inject(root)
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print(f"\n=== {'ALL PASS' if not _fails else str(len(_fails)) + ' FAILED'} ===")
    for f in _fails:
        print(f"  FAILED: {f}")
    return 0 if not _fails else 1


if __name__ == "__main__":
    sys.exit(main())
