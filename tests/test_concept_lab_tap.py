#!/usr/bin/env python3
"""Unit test for the H3 block tap (T4), synthetic and offline.

  <venv>/bin/python tests/test_concept_lab_tap.py
      torch is allowed here and ComfyUI is NOT imported: the tap has to be
      testable without a GPU, without a model, and without the runtime it
      instruments, or the pass-through gate can only ever be checked by
      spending a render.

The stand-ins are deliberately dumb. A fake ModelPatcher that only records
what was hung on it, a fake block that returns `img * 2 + 1`, a fake packed
layout with the four segment kinds, and a fake transformer_options carrying
the three keys comfy/samplers.py:510-513 really sets. What is being tested
is this file's own arithmetic and refusals, not ComfyUI's.

  1. PASS-THROUGH IS EXACT. The patch returns the block's own dict object
     (identity, not equality) and the tensor is untouched, enabled or not.
     This is the gate the whole idea rests on: instrumentation that changes
     the render measures a different model than the one anyone ships.
  2. SELECTION. Steps are read back off the schedule, quantiles map onto
     the step index range, explicit steps win, and an unconditional forward
     records nothing.
  3. SHAPES. Every store writes what it says it writes, frame_mean is video
     only, the sketch is reproducible from its seed, and every ref carries
     a sha256.
  4. THE MANIFEST. The capture id recomputes from what landed on disk, the
     space's doctor verifies the bytes, a second identical session is
     idempotent and a replicate is a different capture.
  5. REFUSALS. A frame-row mismatch, a layout that changes mid-capture, a
     bogus store and a payload with no layout all raise, naming numbers.
  6. DISABLED WRITES NOTHING. Not "writes an empty manifest": nothing.
"""
import json
import os
import shutil
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concept_lab import types as T                                # noqa: E402
from concept_lab.backends import h3_tap                           # noqa: E402
from concept_lab.backends.h3_tap import TapError, TapSession      # noqa: E402
from concept_lab.space import Space                               # noqa: E402

_fails = []

HIDDEN = 16
# text 10 | ref_img 24 | audio 8 | video 18   (latent_t 3 x frame_rows 6)
SEGS = ((0, 10, "text"), (10, 34, "ref_img"), (34, 42, "audio"),
        (42, 60, "video"))
SIGNATURE = (10, 3, 6, 4, 4)      # text_len, latent_t, latent_h, latent_w, audio_t
SCHED = [float(v) for v in torch.linspace(1.0, 0.0, 26).tolist()]


def ck(cond, label, detail=""):
    if not cond:
        _fails.append(label)
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}"
          + (f"  {detail}" if detail else ""))


def refuses(fn, label, must_contain=()):
    try:
        fn()
    except TapError as e:
        missing = [m for m in must_contain if str(m) not in str(e)]
        ck(not missing, label, str(e)[:110] if not missing
           else f"message missing {missing}: {str(e)[:110]}")
        return
    except Exception as e:
        ck(False, label, f"wrong exception {type(e).__name__}: {e}")
        return
    ck(False, label, "did not refuse")


# ------------------------------------------------------------- stand-ins
class FakePatcher:
    """Records what a tap hangs on it. No model, no options, no behaviour."""

    def __init__(self):
        self.wrappers = []
        self.replaces = {}
        self.clones = 0

    def clone(self):
        self.clones += 1
        return FakePatcher()

    def add_wrapper_with_key(self, wrapper_type, key, wrapper):
        self.wrappers.append((wrapper_type, key, wrapper))

    def set_model_patch_replace(self, patch, name, block_name, number):
        self.replaces[(name, block_name, number)] = patch


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


def make_session(root, name="s", **kw):
    kw.setdefault("tap", T.TapSpec(blocks=(0, 5), step_quantiles=(0.5,),
                                   segments=("video",), store="stats"))
    kw.setdefault("arm", "R")
    kw.setdefault("variant_name", "actual_ref")
    kw.setdefault("seed", 42)
    kw.setdefault("replicate", 0)
    kw.setdefault("stack", T.ModelStackFingerprint(
        checkpoint="minimax-h3.safetensors", backend="h3",
        comfy_commit="7fe8a613"))
    kw.setdefault("index_root", os.path.join(root, name))
    kw.setdefault("tensor_root", os.path.join(root, name + "_t"))
    kw.setdefault("sampler_label", "euler/shift12/25")
    return TapSession(**kw)


def drive(session, patcher, layout=None, to=None, block=0, img=None):
    """One wrapper call plus one block-patch call, as the model would."""
    layout = layout or FakeLayout()
    to = to if to is not None else topts()
    img = img if img is not None else torch.arange(
        layout.seq_len * HIDDEN, dtype=torch.float32).reshape(-1, HIDDEN) / 97.0
    session._dm_wrapper(lambda *a, **k: "PASSED THROUGH", None,
                        torch.tensor([1.0]), None,
                        transformer_options=to,
                        minimax_payload={"layout": layout})
    returned = {}

    def original_block(args):
        d = {"img": args["img"] * 2 + 1}
        returned["d"] = d
        return d

    patch = patcher.replaces[("dit", "double_block", block)]
    out = patch({"img": img, "t_emb": None, "mod_segments": None,
                 "rope_freqs": None, "transformer_options": to},
                {"original_block": original_block})
    return out, returned.get("d"), img


def tensor_files(root):
    out = []
    for base, _dirs, files in os.walk(root):
        for f in files:
            if f.endswith(".safetensors"):
                out.append(os.path.join(base, f))
    return sorted(out)


# ---------------------------------------------------------------- 1
def check_passthrough(root):
    print("\n[1] pass-through is exact, enabled or not")
    for enabled in (True, False):
        s = make_session(root, name=f"p{int(enabled)}", enabled=enabled)
        p = FakePatcher()
        patched = s.install(p)
        ck(p.clones == 1 and not p.wrappers and not p.replaces,
           f"enabled={enabled}: the caller's patcher is untouched, the CLONE "
           f"is patched", f"clones={p.clones}")
        ck([k for _t, k, _w in patched.wrappers] == ["concept_lab_tap"],
           f"enabled={enabled}: the diffusion-model wrapper is keyed",
           f"type={patched.wrappers[0][0]!r}")
        ck(sorted(patched.replaces) == [("dit", "double_block", 0),
                                        ("dit", "double_block", 5)],
           f"enabled={enabled}: one block patch per tapped block",
           f"{sorted(patched.replaces)}")

        out, block_dict, img = drive(s, patched)
        ck(out is block_dict,
           f"enabled={enabled}: the patch returns the block's OWN dict",
           f"id(out)={id(out)} id(block)={id(block_dict)}")
        ck(torch.equal(out["img"], img * 2 + 1),
           f"enabled={enabled}: the state is bit-for-bit the block's output")
        ck(list(out.keys()) == ["img"],
           f"enabled={enabled}: nothing was added to the returned dict",
           f"keys={list(out.keys())}")

    s = make_session(root, name="pw")
    p = s.install(FakePatcher())
    ck(s._dm_wrapper(lambda *a, **k: "PASSED THROUGH", None,
                     torch.tensor([1.0]), None, transformer_options=topts(),
                     minimax_payload={"layout": FakeLayout()})
       == "PASSED THROUGH",
       "the wrapper returns the executor's result unchanged")


# ---------------------------------------------------------------- 2
def check_selection(root):
    print("\n[2] step selection off the schedule")
    tap = T.TapSpec(blocks=(0,), step_quantiles=(0.15, 0.5, 0.85),
                    segments=("video",), store="stats")
    s = make_session(root, name="sel", tap=tap)
    patched = s.install(FakePatcher())
    drive(s, patched, to=topts(step=12))
    # 26 sigmas -> 25 steps -> indices 0..24; q maps onto [0, n_steps-1]:
    # round(0.15*24)=4, round(0.5*24)=12, round(0.85*24)=20
    ck(s._selected == [4, 12, 20],
       "quantiles map onto the step index range: round(q * (n_steps - 1)) "
       "with n_steps = len(sample_sigmas) - 1 = 25",
       f"selected={s._selected}")
    ck(s._sampler == {"n_steps": 25, "sigma_first": SCHED[0],
                      "sigma_last": SCHED[-1], "label": "euler/shift12/25"},
       "the sampler block is read off the schedule", json.dumps(s._sampler))
    ck(s.calls_recorded == 1, "a selected step records",
       f"calls_recorded={s.calls_recorded}")

    drive(s, patched, to=topts(step=13))
    ck(s.calls_recorded == 1, "a call at a non-selected sigma records nothing",
       f"calls_recorded={s.calls_recorded}")

    drive(s, patched, to=topts(step=12, cond_or_uncond=(1,)))
    ck(s.calls_recorded == 1, "an uncond call records nothing",
       f"calls_recorded={s.calls_recorded}")
    ck(s.cond_or_uncond_seen == ["0", "1"],
       "both cond and uncond are noted even though only one is kept",
       f"seen={s.cond_or_uncond_seen}")

    e = make_session(root, name="sel2", tap=T.TapSpec(
        blocks=(0,), steps=(3, 7), segments=("video",), store="stats"))
    ep = e.install(FakePatcher())
    drive(e, ep, to=topts(step=7))
    ck(e._selected == [3, 7], "explicit steps override quantiles",
       f"selected={e._selected}")
    ck(e.calls_recorded == 1 and e.tensors[0].step == 7,
       "the recorded step index is the one the sigma names",
       f"step={e.tensors[0].step} sigma={e.tensors[0].sigma}")


# ---------------------------------------------------------------- 3
def check_shapes(root):
    print("\n[3] what each store actually writes")
    tap = T.TapSpec(blocks=(0,), steps=(12,), segments=("video", "ref_img"),
                    store="stats,frame_mean,sketch,full", sketch_dim=8,
                    sketch_seed=7)
    s = make_session(root, name="sh", tap=tap)
    patched = s.install(FakePatcher())
    _out, _d, img = drive(s, patched, to=topts(step=12))

    files = [os.path.basename(f) for f in tensor_files(s.tensor_root)]
    want = ["b00_s12_ref_img_full.safetensors",
            "b00_s12_ref_img_sketch.safetensors",
            "b00_s12_ref_img_stats.safetensors",
            "b00_s12_video_frame_mean.safetensors",
            "b00_s12_video_full.safetensors",
            "b00_s12_video_sketch.safetensors",
            "b00_s12_video_stats.safetensors"]
    ck(files == want, "one file per (block, step, segment, store), and NO "
       "frame_mean for a segment with no time axis", f"{files}")

    by_key = {t.key: t for t in s.tensors}
    ck(by_key["video/b0/s12/stats"].shape == (3, HIDDEN)
       and by_key["video/b0/s12/stats"].dtype == "fp32",
       "stats is [3, hidden] fp32 (mean, std, mean of squares)",
       f"{by_key['video/b0/s12/stats'].shape}")
    ck(by_key["video/b0/s12/frame_mean"].shape == (3, HIDDEN),
       "frame_mean is [latent_t, hidden]",
       f"{by_key['video/b0/s12/frame_mean'].shape}")
    ck(by_key["video/b0/s12/patch_mean"].shape == (6, HIDDEN)
       and by_key["video/b0/s12/patch_mean"].dtype == "fp16",
       "patch_mean is [frame_rows, hidden] fp16, in the same file",
       f"{by_key['video/b0/s12/patch_mean'].shape}")
    ck(by_key["video/b0/s12/patch_mean"].path
       == by_key["video/b0/s12/frame_mean"].path,
       "and it really is the same file", by_key["video/b0/s12/patch_mean"].path)
    ck(by_key["video/b0/s12/sketch"].shape == (18, 8)
       and by_key["video/b0/s12/sketch"].dtype == "fp16",
       "sketch is [n_rows, sketch_dim] fp16")
    ck(by_key["video/b0/s12/full"].shape == (18, HIDDEN)
       and by_key["video/b0/s12/full"].dtype == "bf16",
       "full is the rows themselves, bf16")
    ck(by_key["ref_img/b0/s12/stats"].shape == (3, HIDDEN),
       "the reference segment gets its own stats")
    ck(all(t.sha256 and t.sha256.startswith("sha256:") for t in s.tensors),
       "every TensorRef carries a sha256", f"n={len(s.tensors)}")
    ck({t.sigma for t in s.tensors} == {SCHED[12]},
       "and the sigma it was taken at", f"{sorted({t.sigma for t in s.tensors})}")

    # the arithmetic, checked against the state that went in
    from safetensors.torch import load_file
    st = load_file(os.path.join(s.space.tensor_dir(),
                                by_key["video/b0/s12/stats"].path))["stats"]
    rows = (img * 2 + 1)[42:60].to(torch.float32)
    ck(torch.allclose(st[0], rows.mean(dim=0), atol=1e-6)
       and torch.allclose(st[2], (rows * rows).mean(dim=0), atol=1e-4),
       "stats row 0 is the per-dim mean and row 2 the mean of squares")
    fm = load_file(os.path.join(s.space.tensor_dir(),
                                by_key["video/b0/s12/frame_mean"].path))
    ck(torch.allclose(fm["frame_mean"],
                      rows.reshape(3, 6, HIDDEN).mean(dim=1), atol=1e-6),
       "frame_mean averages the 6 patch rows of each of the 3 latent frames")

    # the projection has to be re-derivable from its seed alone
    s2 = make_session(root, name="sh2", tap=tap)
    p2 = s2.install(FakePatcher())
    drive(s2, p2, to=topts(step=12))
    ck(s._sketch_sha == s2._sketch_sha and s._sketch_sha is not None,
       "the same sketch_seed draws the same projection", s._sketch_sha[:23])
    a = load_file(os.path.join(s.space.tensor_dir(),
                               by_key["video/b0/s12/sketch"].path))["sketch"]
    b = load_file(os.path.join(s2.space.tensor_dir(),
                              [t for t in s2.tensors
                               if t.key == "video/b0/s12/sketch"][0].path))["sketch"]
    ck(torch.equal(a, b), "so the sketch itself reproduces")


# ---------------------------------------------------------------- 4
def check_manifest(root):
    print("\n[4] the manifest, the id, and the doctor")
    idx, ten = os.path.join(root, "M"), os.path.join(root, "Mt")
    tap = T.TapSpec(blocks=(0,), steps=(12,), segments=("video",),
                    store="stats,frame_mean")
    s = make_session(root, tap=tap, index_root=idx, tensor_root=ten)
    patched = s.install(FakePatcher())
    drive(s, patched, to=topts(step=12))
    cid_before = s._capture_id
    rep = s.finalize()

    ck(rep["capture_id"] == cid_before,
       "the capture id was fixed BEFORE any tensor was written",
       f"{cid_before}")
    ck(rep["n_tensors"] == 3 and rep["bytes_written"] > 0,
       "the report counts what landed",
       f"n={rep['n_tensors']} bytes={rep['bytes_written']}")

    space = Space(idx, ten)
    d = json.load(open(rep["path"]))
    man = T.FunctionalCaptureManifest.from_dict(d)
    ck(man.capture_id == rep["capture_id"],
       "the written manifest recomputes to its own filename id",
       man.capture_id)
    ck(man.process.get("launch_id") == T.process_fingerprint()["launch_id"]
       and man.process.get("torch") == torch.__version__,
       "the process fingerprint carries the launch and the torch it ran on",
       f"{man.process.get('torch')} / cuda {man.process.get('cuda')}")
    lay = T.LayoutSignature.from_dict(d["layout"])
    ck(lay.total_rows == 60 and lay.segment("video") == (42, 60)
       and lay.position_hash.startswith("sha256:"),
       "the layout signature is stored, segments named, positions hashed",
       f"{lay.segments}")
    notes = json.loads(man.notes)
    ck(notes["calls_recorded"] == 1 and notes["cond_or_uncond_seen"] == ["0"]
       and notes["enabled"] is True,
       "the notes say what was seen and what was kept", json.dumps(notes)[:90])
    # the manifest's floats came back through canonical_json's 12-significant
    # -digit rounding (DEC-CL-0006), which is why this is a tolerance and not
    # an equality: 0.9599999785423279 is stored as 0.959999978542
    ck(len(man.sigmas) == 26
       and all(abs(a - b) < 1e-9 for a, b in zip(man.sigmas, SCHED)),
       "the whole schedule is recorded, not just the taps",
       f"n_sigmas={len(man.sigmas)} first two={list(man.sigmas)[:2]}")

    doc = space.doctor()
    ck(doc["ok"] and not doc["problems"],
       "doctor verifies every tensor byte-for-byte",
       json.dumps(doc["problems"])[:90])

    # same condition, same launch, same bytes -> one capture, idempotent put
    s2 = make_session(root, tap=tap, index_root=idx, tensor_root=ten)
    drive(s2, s2.install(FakePatcher()), to=topts(step=12))
    rep2 = s2.finalize()
    ck(rep2["capture_id"] == rep["capture_id"],
       "two identical sessions in one launch are one capture id")
    ck(rep2["path"] == rep["path"], "and the put is idempotent, not a refusal")

    s3 = make_session(root, tap=tap, index_root=idx, tensor_root=ten,
                      replicate=1)
    drive(s3, s3.install(FakePatcher()), to=topts(step=12))
    rep3 = s3.finalize()
    ck(rep3["capture_id"] != rep["capture_id"],
       "a replicate is a different capture", rep3["capture_id"])
    ck(len(space.list("captures")) == 2, "and lands beside the first",
       f"n_captures={len(space.list('captures'))}")

    # a manifest that claims an id its own fields do not produce
    from concept_lab import api
    lying = dict(d)
    lying["capture_id"] = "0" * 16
    try:
        api.capture_record(lying, index_root=idx, tensor_root=ten)
        ck(False, "a manifest claiming the wrong id is refused")
    except T.ContractError as e:
        ck("0000000000000000" in str(e) and rep["capture_id"] in str(e),
           "a manifest claiming the wrong id is refused, printing both",
           str(e)[:100])


# ---------------------------------------------------------------- 5
def check_refusals(root):
    print("\n[5] refusals name the numbers that disagreed")
    tap = T.TapSpec(blocks=(0,), steps=(12,), segments=("video",),
                    store="frame_mean")
    s = make_session(root, name="r1", tap=tap)
    patched = s.install(FakePatcher())
    # signature says 4 latent frames, the segment has rows for 3
    bad_layout = FakeLayout(signature=(10, 4, 6, 4, 4))
    refuses(lambda: drive(s, patched, layout=bad_layout, to=topts(step=12)),
            "a frame-row mismatch raises naming both numbers",
            ("latent_t=4", "frame_rows=6", "24", "18 rows"))

    s2 = make_session(root, name="r2")
    p2 = s2.install(FakePatcher())
    drive(s2, p2, to=topts(step=12))
    other = FakeLayout(segments=((0, 12, "text"), (12, 30, "video")),
                       signature=(12, 3, 6, 4, 0))
    refuses(lambda: s2._dm_wrapper(lambda *a, **k: None, None,
                                   torch.tensor([1.0]), None,
                                   transformer_options=topts(),
                                   minimax_payload={"layout": other}),
            "a second layout signature inside one capture raises",
            ("60", "30"))

    s3 = make_session(root, name="r3")
    p3 = s3.install(FakePatcher())
    refuses(lambda: s3._dm_wrapper(lambda *a, **k: None, None,
                                   torch.tensor([1.0]), None,
                                   transformer_options=topts(),
                                   minimax_payload={"refs": []}),
            "a payload with no layout raises, listing the keys it did carry",
            ("refs",))

    try:
        T.TapSpec(blocks=(0,), segments=("video",), store="bogus")
        ck(False, "TapSpec refuses an unknown store")
    except T.ContractError as e:
        ck("bogus" in str(e), "TapSpec refuses an unknown store", str(e)[:80])
    ck(T.TapSpec(blocks=(0,), segments=("video",),
                 store="sketch,stats", sketch_dim=8).store == "stats,sketch",
       "a comma-joined store is canonicalized to STORE_KINDS order, so two "
       "spellings of one condition are one id")

    s4 = make_session(root, name="r4")
    p4 = s4.install(FakePatcher())
    refuses(lambda: drive(s4, p4, to={"cond_or_uncond": [0]}),
            "a transformer_options with no schedule raises rather than "
            "guessing the step", ("sample_sigmas",))

    s5 = make_session(root, name="r5")
    s5.install(FakePatcher())
    refuses(s5.finalize,
            "finalize refuses when nothing was recorded, instead of filing an "
            "empty capture", ("nothing was recorded",))


# ---------------------------------------------------------------- 6
def check_disabled(root):
    print("\n[6] disabled writes nothing at all")
    idx, ten = os.path.join(root, "N"), os.path.join(root, "Nt")
    s = make_session(root, tap=T.TapSpec(
        blocks=(0,), steps=(12,), segments=("video",),
        store="stats,frame_mean,full"), index_root=idx, tensor_root=ten,
        enabled=False)
    patched = s.install(FakePatcher())
    out, block_dict, img = drive(s, patched, to=topts(step=12))
    ck(out is block_dict and torch.equal(out["img"], img * 2 + 1),
       "the render still passes through exactly")
    rep = s.finalize()
    ck(rep == {"enabled": False, "calls_seen": 1,
               "note": "pass-through only; nothing recorded"},
       "finalize returns the pass-through-only dict", json.dumps(rep))
    ck(tensor_files(ten) == [], "no tensor file exists", f"{tensor_files(ten)}")
    ck(Space(idx, ten).list("captures") == [],
       "and no capture was filed")


def main():
    root = tempfile.mkdtemp(prefix="concept_lab_tap_test_")
    try:
        print("=== concept_lab H3 tap (synthetic, torch, no comfy, no GPU) ===")
        ck("comfy" not in sys.modules and "folder_paths" not in sys.modules,
           "the tap imports without ComfyUI",
           f"DIFFUSION_MODEL={h3_tap.DIFFUSION_MODEL!r}")
        check_passthrough(root)
        check_selection(root)
        check_shapes(root)
        check_manifest(root)
        check_refusals(root)
        check_disabled(root)
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print(f"\n=== {'ALL PASS' if not _fails else str(len(_fails)) + ' FAILED'} ===")
    for f in _fails:
        print(f"  FAILED: {f}")
    return 0 if not _fails else 1


if __name__ == "__main__":
    sys.exit(main())
