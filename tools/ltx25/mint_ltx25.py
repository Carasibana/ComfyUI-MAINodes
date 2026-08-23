#!/usr/bin/env python3
"""Mint an LTX-2.5 API-format graph from ComfyUI's official t2v/i2v template.

The stock template is a subgraph; ui2api inlines it. This script then stamps
the axes we actually vary (prompt, negative, fps, seconds, megapixels, seed,
which video VAE) and rewrites the loader names to our ltx25/ package subdir.

The prompt-enhancer branch (TextGenerateLTX2Prompt + the gemma4_e2b encoder
that drives it) is REMOVED, not switched off: a stock ComfySwitchNode still
executes both sides, and an enhanced prompt is not the prompt we wrote.

    python3 mint_ltx25.py --scene tempest_ltx --arm stock \
        --prompt-file benchmarks/prompts/ltx25/tempest.txt \
        --fps 24 --seconds 5 --seed 20260821

Frame count is fps*seconds+1 and MUST satisfy frames % 8 == 1 (LTX-2.5
constraint); the script refuses otherwise rather than letting the latent
node silently floor it.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ui2api import convert  # noqa: E402

REPO = os.environ.get("LTX25_OUT_ROOT", os.getcwd())   # minted graphs land in $REPO/workflows
def _templates_dir():
    """The stock ComfyUI workflow templates, wherever the running venv keeps them."""
    import importlib.util
    for pkg in ("comfyui_workflow_templates_json", "comfyui_workflow_templates"):
        spec = importlib.util.find_spec(pkg)
        if spec and spec.submodule_search_locations:
            d = os.path.join(list(spec.submodule_search_locations)[0], "templates")
            if os.path.isdir(d):
                return d
    raise SystemExit("comfyui-workflow-templates is not installed in this venv")


TEMPLATES = _templates_dir()

# node ids inside the inlined subgraph (stable for this template revision)
N_PROMPT = "405:376"      # PrimitiveStringMultiline, positive
N_NEG = "405:373"         # CLIPTextEncode, negative
N_TEXTENC = "405:364"     # CLIPTextEncode, positive
N_SWITCH = "405:382"      # ComfySwitchNode (enhancer on/off)
N_ENHANCER = "405:380"    # TextGenerateLTX2Prompt
N_ENH_BOOL = "405:383"
N_ENH_CLIP = "405:393"    # gemma4_e2b loader, enhancer only
N_ENH_PREVIEW = "405:381"
N_FPS = "405:361"
N_SECONDS = "405:362"
N_SEED1 = "405:339"       # base pass RandomNoise
N_SEED2 = "405:338"       # refine pass RandomNoise
N_RES = "409"             # ResolutionSelector
N_UNET = "405:384"
N_CLIP = "405:387"
N_VAE_VIDEO = "405:385"
N_VAE_AUDIO = "405:386"
N_UPSCALER = "405:371"
N_SAVE = "75"

PKG = "ltx25/"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--negative", default="pc game, console game, video game, "
                                          "cartoon, childish, ugly")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--seconds", type=int, default=5)
    ap.add_argument("--megapixels", type=float, default=0.9)
    ap.add_argument("--aspect", default="16:9 (Widescreen)")
    ap.add_argument("--seed", type=int, default=20260821)
    ap.add_argument("--conv-vae", action="store_true",
                    help="decode with the faster conv VAE instead of the DiffVAE")
    ap.add_argument("--template", default="video_ltx2_5_t2v.json")
    ap.add_argument("--version", type=int, default=1)
    a = ap.parse_args()

    frames = a.fps * a.seconds + 1
    if frames % 8 != 1:
        sys.exit(f"REFUSING: fps*seconds+1 = {frames}, and LTX-2.5 requires "
                 f"frames % 8 == 1. Pick an fps/seconds pair whose product is "
                 f"a multiple of 8 (24x5=120 -> 121 ok; 25x8=200 -> 201 ok).")

    api = convert(json.load(open(os.path.join(TEMPLATES, a.template))))

    prompt = open(a.prompt_file).read().strip()
    api[N_PROMPT]["inputs"]["value"] = prompt
    api[N_NEG]["inputs"]["text"] = a.negative

    # cut the enhancer branch out entirely
    api[N_TEXTENC]["inputs"]["text"] = [N_PROMPT, 0]
    for nid in (N_SWITCH, N_ENHANCER, N_ENH_BOOL, N_ENH_CLIP, N_ENH_PREVIEW):
        api.pop(nid, None)

    api[N_FPS]["inputs"]["value"] = a.fps
    api[N_SECONDS]["inputs"]["value"] = a.seconds
    api[N_SEED1]["inputs"]["noise_seed"] = a.seed
    api[N_SEED2]["inputs"]["noise_seed"] = a.seed
    api[N_RES]["inputs"]["megapixels"] = a.megapixels
    api[N_RES]["inputs"]["aspect_ratio"] = a.aspect

    vid_vae = ("ltx-2.5-video-vae-conv-bf16.safetensors" if a.conv_vae
               else "ltx-2.5-video-vae-bf16.safetensors")
    api[N_UNET]["inputs"]["unet_name"] = PKG + api[N_UNET]["inputs"]["unet_name"]
    api[N_CLIP]["inputs"]["clip_name"] = PKG + api[N_CLIP]["inputs"]["clip_name"]
    api[N_VAE_VIDEO]["inputs"]["vae_name"] = PKG + vid_vae
    api[N_VAE_AUDIO]["inputs"]["vae_name"] = PKG + api[N_VAE_AUDIO]["inputs"]["vae_name"]
    api[N_UPSCALER]["inputs"]["model_name"] = PKG + api[N_UPSCALER]["inputs"]["model_name"]

    prefix = f"{a.scene}_{a.arm}"
    api[N_SAVE]["inputs"]["filename_prefix"] = f"video/{prefix}"

    out = os.path.join(REPO, "workflows",
                       f"{prefix}_v{a.version:03d}.api.json")
    with open(out, "w") as f:
        json.dump(api, f, indent=1)
    print(f"minted {out}")
    print(f"  frames {frames} @ {a.fps} fps ({a.seconds}s), {a.megapixels} MP "
          f"{a.aspect}, seed {a.seed}, vae {vid_vae}")
    audio_lat = frames / a.fps * 25.0
    print(f"  audio latents (25 Hz clock): {audio_lat:.4f} -> "
          f"{round(audio_lat)} stored; length error "
          f"{abs(round(audio_lat) - audio_lat) / 25 * 1000:.1f} ms")
    print(f"  launch: python3 benchmarks/scripts/queue_scene.py {out} "
          f"--tag {a.arm}")


if __name__ == "__main__":
    main()
