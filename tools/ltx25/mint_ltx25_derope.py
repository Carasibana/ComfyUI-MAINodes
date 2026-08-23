#!/usr/bin/env python3
"""Mint an LTX-2.5 uniform de-rope graph: base pass, dilated pass, exact recover.

One graph, three videos. The base pass is the stock ComfyUI LTX-2.5 template.
Its decoded frames are smeared on an integer hold (ImageBatchRepeatInterleaving),
padded up to LTX's legal 8k+1 grid, re-encoded, and rendered again on the
dilated clock at the SAME declared frame rate - which is what tells the model
the action is now d times slower. The dilated result is saved as its own clip
(this is the bullet-time product), then every d-th frame is taken back to
return the original frame count, muxed with the BASE audio.

Two hard constraints the script enforces rather than discovers at render time:

  frames % 8 == 1                LTX-2.5 model card, both base and dilated
  dilated_frames / fps <= 20 s   the DiT's positional_embedding_max_pos[0] is
                                 20, and the video RoPE coordinate is absolute
                                 seconds (frame_index / frame_rate), so a
                                 longer dilated clip extrapolates off the
                                 trained time axis

The second one is what caps dilation: at 25 fps the dilated pass can hold at
most 497 frames, so d=4 allows a 121-frame base and d=8 allows 61.

Audio in the dilated pass is a fresh empty latent sized to the DILATED frame
count, not the base one. That is deliberate: audio token time is absolute
seconds too, so handing the dilated pass a base-length audio latent tells the
model the soundtrack ends 1/d of the way into the shot.

    python3 mint_ltx25_derope.py --scene ltx25_tempest --prompt-file X \
        --frames 61 --fps 25 --dilation 8 --seed 20260821
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

# stock template node ids after inlining
N_PROMPT, N_NEG, N_TEXTENC = "405:376", "405:373", "405:364"
N_SWITCH, N_ENHANCER, N_ENH_BOOL, N_ENH_CLIP, N_ENH_PREV = (
    "405:382", "405:380", "405:383", "405:393", "405:381")
N_FPS, N_SECONDS, N_FRAMES_EXPR = "405:361", "405:362", "405:378"
N_LATENT_V, N_LATENT_A = "405:356", "405:366"
N_SEED1, N_SEED2 = "405:339", "405:338"
N_RES, N_UNET, N_CLIP = "409", "405:384", "405:387"
N_VAE_V, N_VAE_A, N_UPSCALER = "405:385", "405:386", "405:371"
N_SAVE = "75"
N_BASE_IMAGES = "405:374"     # VAEDecodeTiled, stage-2 pixels
N_BASE_AUDIO = "405:358"      # LTXVAudioVAEDecode, stage-2 audio
N_GUIDER = "405:388"
N_SAMPLER_SEL = "405:352"
N_FPS_EXPR = "405:359"        # slot 0 float, slot 1 int
N_STAGE1_SEP = "405:367"      # LTXVSeparateAVLatent of the half-res stage-1 sample

PKG = "ltx25/"
ROPE_SECONDS = 20.0           # positional_embedding_max_pos[0]


def legal_ceil(n):
    return 1 + 8 * ((max(1, int(n)) - 1 + 7) // 8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--negative", default="pc game, console game, video game, "
                                          "childish, ugly, low quality, blurry")
    ap.add_argument("--frames", type=int, required=True, help="base frame count, must be 8k+1")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--dilation", type=int, required=True)
    ap.add_argument("--megapixels", type=float, default=0.9)
    ap.add_argument("--aspect", default="16:9 (Widescreen)")
    ap.add_argument("--seed", type=int, default=20260821)
    ap.add_argument("--sigmas", default="0.85, 0.7250, 0.4219, 0.0",
                    help="dilated-pass schedule; the distilled stage-2 list by default")
    ap.add_argument("--dilate-from", choices=("stage2", "stage1"), default="stage2",
                    help="which base pixels the smear is built from. stage2 (default) "
                         "smears the finished full-res frames and refines at the same "
                         "resolution - no upscaling happens in the dilated pass. stage1 "
                         "smears the HALF-res stage-1 frames and runs the x2 latent "
                         "upsampler on the dilated latent, so the dilated pass is a "
                         "genuine upscale and de-rope in one, and the base's own "
                         "full-res refine is skipped.")
    ap.add_argument("--pad-position", choices=("prepend", "append"), default="prepend",
                    help="where the legal-grid pad frames go. PREPEND is the default "
                         "because LTX's grid is 1+8k: with the pad in front, hold group "
                         "j occupies pixel frames pad+d*j .. pad+d*j+d-1, which at d=8 "
                         "is exactly one latent temporal block. Appending instead makes "
                         "every group straddle two blocks, so one block has to encode "
                         "the tail of one source frame and the head of the next.")
    ap.add_argument("--version", type=int, default=1)
    a = ap.parse_args()

    n, d, fps = a.frames, a.dilation, a.fps
    if n % 8 != 1:
        sys.exit(f"REFUSING: base frames {n} is not on the 8k+1 grid")
    smeared = n * d
    L = legal_ceil(smeared)
    pad = L - smeared
    if L / fps > ROPE_SECONDS:
        sys.exit(f"REFUSING: dilated pass is {L} frames = {L/fps:.2f}s at {fps} fps, "
                 f"past the DiT's {ROPE_SECONDS:g}s time axis "
                 f"(positional_embedding_max_pos[0]). Largest LEGAL base at "
                 f"d={d}, fps={fps} is "
                 f"{1 + 8 * ((int((ROPE_SECONDS * fps) // d) - 1) // 8)} frames.")

    api = convert(json.load(open(os.path.join(TEMPLATES, "video_ltx2_5_t2v.json"))))

    # ---- base pass ------------------------------------------------------
    api[N_PROMPT]["inputs"]["value"] = open(a.prompt_file).read().strip()
    api[N_NEG]["inputs"]["text"] = a.negative
    api[N_TEXTENC]["inputs"]["text"] = [N_PROMPT, 0]     # cut the enhancer branch
    for nid in (N_SWITCH, N_ENHANCER, N_ENH_BOOL, N_ENH_CLIP, N_ENH_PREV):
        api.pop(nid, None)

    # frame count becomes an explicit primitive instead of fps*seconds+1
    api["dr_n"] = {"class_type": "PrimitiveInt", "inputs": {"value": n}}
    api[N_LATENT_V]["inputs"]["length"] = ["dr_n", 0]
    api[N_LATENT_A]["inputs"]["frames_number"] = ["dr_n", 0]
    for nid in (N_FRAMES_EXPR, N_SECONDS):
        api.pop(nid, None)

    api[N_FPS]["inputs"]["value"] = fps
    api[N_SEED1]["inputs"]["noise_seed"] = a.seed
    api[N_SEED2]["inputs"]["noise_seed"] = a.seed
    api[N_RES]["inputs"]["megapixels"] = a.megapixels
    api[N_RES]["inputs"]["aspect_ratio"] = a.aspect
    for nid, key in ((N_UNET, "unet_name"), (N_CLIP, "clip_name"),
                     (N_VAE_V, "vae_name"), (N_VAE_A, "vae_name"),
                     (N_UPSCALER, "model_name")):
        api[nid]["inputs"][key] = PKG + api[nid]["inputs"][key]

    stem = (f"{a.scene}_d{d}" + ("pre" if a.pad_position == "prepend" else "")
            + ("up" if a.dilate_from == "stage1" else ""))
    api[N_SAVE]["inputs"]["filename_prefix"] = f"video/{stem}_base"

    # ---- smear: repeat each base frame d times, pad up to the legal grid --
    if a.dilate_from == "stage1":
        # decode the half-res stage-1 latent; the x2 upsampler runs later, on the
        # DILATED latent, so this pass upscales and de-ropes at the same time
        api["dr_s1dec"] = {"class_type": "VAEDecodeTiled",
                           "inputs": {"samples": [N_STAGE1_SEP, 0], "vae": [N_VAE_V, 0],
                                      "tile_size": 512, "overlap": 64,
                                      "temporal_size": 64, "temporal_overlap": 16}}
        smear_src = ["dr_s1dec", 0]
    else:
        smear_src = [N_BASE_IMAGES, 0]
    api["dr_smear"] = {"class_type": "ImageBatchRepeatInterleaving",
                       "inputs": {"images": smear_src, "repeats": d}}
    init = ["dr_smear", 0]
    offset = 0
    if pad:
        front = a.pad_position == "prepend"
        src_idx = 0 if front else smeared - 1
        api["dr_padsrc"] = {"class_type": "ImageFromBatch",
                            "inputs": {"image": ["dr_smear", 0],
                                       "batch_index": src_idx, "length": 1}}
        api["dr_pad"] = {"class_type": "RepeatImageBatch",
                         "inputs": {"image": ["dr_padsrc", 0], "amount": pad}}
        pair = (["dr_pad", 0], ["dr_smear", 0]) if front else (["dr_smear", 0], ["dr_pad", 0])
        api["dr_init"] = {"class_type": "ImageBatch",
                          "inputs": {"image1": pair[0], "image2": pair[1]}}
        init = ["dr_init", 0]
        offset = pad if front else 0

    # ---- dilated pass ----------------------------------------------------
    api["dr_enc"] = {"class_type": "VAEEncode",
                     "inputs": {"pixels": init, "vae": [N_VAE_V, 0]}}
    api["dr_aud"] = {"class_type": "LTXVEmptyLatentAudio",
                     "inputs": {"frames_number": L, "frame_rate": [N_FPS_EXPR, 1],
                                "batch_size": 1, "audio_vae": [N_VAE_A, 0]}}
    vid_latent = ["dr_enc", 0]
    if a.dilate_from == "stage1":
        api["dr_up"] = {"class_type": "LTXVLatentUpsampler",
                        "inputs": {"samples": ["dr_enc", 0],
                                   "upscale_model": [N_UPSCALER, 0], "vae": [N_VAE_V, 0]}}
        vid_latent = ["dr_up", 0]
    api["dr_cat"] = {"class_type": "LTXVConcatAVLatent",
                     "inputs": {"video_latent": vid_latent, "audio_latent": ["dr_aud", 0]}}
    api["dr_noise"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": a.seed}}
    api["dr_sigmas"] = {"class_type": "ManualSigmas", "inputs": {"sigmas": a.sigmas}}
    api["dr_sample"] = {"class_type": "SamplerCustomAdvanced",
                        "inputs": {"noise": ["dr_noise", 0], "guider": [N_GUIDER, 0],
                                   "sampler": [N_SAMPLER_SEL, 0], "sigmas": ["dr_sigmas", 0],
                                   "latent_image": ["dr_cat", 0]}}
    api["dr_sep"] = {"class_type": "LTXVSeparateAVLatent",
                     "inputs": {"av_latent": ["dr_sample", 0]}}
    api["dr_dec"] = {"class_type": "VAEDecodeTiled",
                     "inputs": {"samples": ["dr_sep", 0], "vae": [N_VAE_V, 0],
                                "tile_size": 512, "overlap": 64,
                                "temporal_size": 64, "temporal_overlap": 16}}

    # the dilated clip is a deliverable in its own right (slow motion)
    api["dr_vid_dil"] = {"class_type": "CreateVideo",
                         "inputs": {"images": ["dr_dec", 0], "fps": [N_FPS_EXPR, 0],
                                    "bit_depth": 8}}
    api["dr_save_dil"] = {"class_type": "SaveVideo",
                          "inputs": {"video": ["dr_vid_dil", 0],
                                     "filename_prefix": f"video/{stem}_dilated",
                                     "format": "auto", "codec": "auto"}}

    # ---- exact recover: drop the pad, keep frame 0 of every hold group ----
    api["dr_trim"] = {"class_type": "ImageFromBatch",
                      "inputs": {"image": ["dr_dec", 0], "batch_index": offset,
                                 "length": smeared}}
    api["dr_rec"] = {"class_type": "VHS_SelectEveryNthImage",
                     "inputs": {"images": ["dr_trim", 0], "select_every_nth": d,
                                "skip_first_images": 0}}
    api["dr_vid_rec"] = {"class_type": "CreateVideo",
                         "inputs": {"images": ["dr_rec", 0], "fps": [N_FPS_EXPR, 0],
                                    "audio": [N_BASE_AUDIO, 0], "bit_depth": 8}}
    api["dr_save_rec"] = {"class_type": "SaveVideo",
                          "inputs": {"video": ["dr_vid_rec", 0],
                                     "filename_prefix": f"video/{stem}_recovered",
                                     "format": "auto", "codec": "auto"}}

    out = os.path.join(REPO, "workflows", f"{stem}_v{a.version:03d}.api.json")
    json.dump(api, open(out, "w"), indent=1)

    h, w = 0, 0
    print(f"minted {out}")
    print(f"  base      {n} frames @ {fps} fps = {n/fps:.3f}s, {a.megapixels} MP {a.aspect}")
    print(f"  dilated   x{d} -> {smeared} + {pad} pad = {L} frames = {L/fps:.3f}s "
          f"({L/fps/ROPE_SECONDS*100:.0f}% of the {ROPE_SECONDS:g}s time axis)")
    print(f"  latent T  {(n-1)//8+1} -> {(L-1)//8+1}  ({((L-1)//8+1)/((n-1)//8+1):.2f}x video tokens)")
    print(f"  audio     base {round(n/fps*25)} latents (err "
          f"{abs(round(n/fps*25)-n/fps*25)/25*1000:.1f} ms), dilated {round(L/fps*25)} latents "
          f"(err {abs(round(L/fps*25)-L/fps*25)/25*1000:.1f} ms)")
    blocks = ((L - 1) // 8 + 1)
    print(f"  pad       {pad} frames {a.pad_position}ed; hold group j starts at pixel "
          f"index {offset}+{d}j")
    if d == 8:
        aligned = (offset % 8) == 1
        print(f"  latent    {blocks} temporal blocks for {n} source frames -> "
              f"{'ONE BLOCK PER SOURCE FRAME' if aligned else 'groups STRADDLE block boundaries'}")
    print(f"  recover   {smeared} -> every {d}th from index {offset} -> {n} frames "
          f"@ {fps} fps, base audio muxed")
    print(f"  source    {'HALF-res stage-1 frames, x2 upsampler INSIDE the dilated pass' if a.dilate_from=='stage1' else 'full-res stage-2 frames, no upscaling in the dilated pass'}")
    print(f"  sigmas    {a.sigmas}")
    print(f"  saves     video/{stem}_base, video/{stem}_dilated, video/{stem}_recovered")


if __name__ == "__main__":
    main()
