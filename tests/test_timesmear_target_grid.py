"""H3TimeSmear pads to the grid carried in a remapped hold map, and H3ExactRecover inverts it."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import torch
import motion as M
import derope_any


def test_smear_pads_to_ltx_grid_and_recovers_exactly():
    n = 20
    frames = torch.arange(n, dtype=torch.float32).view(n, 1, 1, 1).expand(n, 4, 4, 3).contiguous()
    hm = json.dumps({"holds": [1] * 4 + [4] * 10 + [1] * 6, "world_len": n})
    remapped, length, _, _ = derope_any.H3ClockRemap().remap(hm, "ltx-2.5")
    imgs, used, total, report = M.H3TimeSmear().smear(frames, 4, hold_map=remapped, expand_to_end=False, fps=25)
    assert total == length and (total - 1) % 8 == 0 and imgs.shape[0] == total
    u = json.loads(used)
    assert u["legal"] == [8, 1] and sum(u["holds"]) == total
    rec = M.H3ExactRecover().recover(imgs, used)[0]
    assert rec.shape[0] == n and torch.equal(rec[:, 0, 0, 0], torch.arange(n, dtype=torch.float32))


def test_plain_map_still_pads_to_h3_grid():
    n = 12
    frames = torch.zeros(n, 2, 2, 3)
    hm = json.dumps({"holds": [1] * 4 + [4] * 8, "world_len": n})
    _, used, total, _ = M.H3TimeSmear().smear(frames, 4, hold_map=hm, expand_to_end=False)
    assert (total - 5) % 17 == 0 and "legal" not in json.loads(used)


def test_camera_compensated_mode_ignores_a_pure_pan():
    # a texture that only translates frame to frame: the plain |d3| profile sees
    # motion everywhere, the compensated one sees (almost) nothing
    torch.manual_seed(0)
    base = torch.rand(1, 24, 1, 16, 16)
    frames = [torch.roll(base, shifts=(0, t), dims=(3, 4)) for t in range(12)]
    z = torch.cat(frames, dim=2)
    plain = M._jerk_profile(z, "value |d3| (default)", phase_norm=False)
    comp = M._jerk_profile(z, "value |d3| camera-compensated", phase_norm=False)
    assert comp.mean() < 0.05 * plain.mean()


if __name__ == "__main__":          # house style: python tests/<file>.py
    import inspect
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and inspect.isfunction(v)]
    for fn in fns:
        if "tmp_path" in inspect.signature(fn).parameters:
            import tempfile, pathlib
            fn(pathlib.Path(tempfile.mkdtemp()))
        else:
            fn()
        print("ok ", fn.__name__)
    print(f"{len(fns)} passed")
