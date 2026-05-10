"""
对比简单 caption（短、单一对象）与复杂 caption（长、多对象/多属性）下的生成效果。
两组各生成 NUM_EACH 张图像，分别计算 CLIP Score 并打印对比结论。
支持 refiner 细化和 xformers 显存优化
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_BASE_URL"] = "https://hf-mirror.com"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

import open_clip
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline

from logger_utils import setup_logger

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
CAPTION_JSON = PROJECT_ROOT / "data" / "annotations_trainval2017" / "annotations" / "captions_val2017.json"
SIMPLE_DIR = PROJECT_ROOT / "outputs" / "images" / "comparison_simple_sdxl"
COMPLEX_DIR = PROJECT_ROOT / "outputs" / "images" / "comparison_complex_sdxl"
RESULTS_FILE = PROJECT_ROOT / "outputs" / "results" / "complex_comparison_results_sdxl.json"

NUM_EACH = 50
SIMPLE_MAX_LEN = 40
COMPLEX_MIN_LEN = 80
SEED = 100
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
REFINER_MODEL_ID = "stabilityai/stable-diffusion-xl-refiner-1.0"
IMAGE_SIZE = 1024
NEGATIVE_PROMPT = (
    "blurry, ugly, distorted, low quality, worst quality, "
    "low resolution, jpeg artifacts, watermark, signature"
)


def load_captions(logger, num_each):
    with open(CAPTION_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_captions = [ann["caption"].strip() for ann in data["annotations"]]

    simple = [c for c in all_captions if len(c) <= SIMPLE_MAX_LEN]
    complex_ = [c for c in all_captions if len(c) >= COMPLEX_MIN_LEN]

    if len(simple) < num_each or len(complex_) < num_each:
        logger.error(f"Not enough captions: simple={len(simple)}, complex={len(complex_)}")
        sys.exit(1)

    import random
    random.seed(SEED)
    simple_sample = random.sample(simple, num_each)
    complex_sample = random.sample(complex_, num_each)

    logger.info(f"Simple captions (len <= {SIMPLE_MAX_LEN}): {len(simple)} available, using {num_each}")
    logger.info(f"Complex captions (len >= {COMPLEX_MIN_LEN}): {len(complex_)} available, using {num_each}")
    logger.info("Sample simple captions:")
    for c in simple_sample[:3]:
        logger.info(f"  {c}")
    logger.info("Sample complex captions:")
    for c in complex_sample[:3]:
        logger.info(f"  {c}")

    return simple_sample, complex_sample


def build_pipeline(logger, use_refiner=True, use_xformers=False, compile_unet=False):
    """加载 SDXL 模型（FP16），支持 xformers 和 refiner。"""
    logger.info(f"Loading base model {MODEL_ID} (FP16)...")
    base_pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    )
    
    base_pipe.enable_attention_slicing()
    base_pipe.vae.enable_slicing()
    base_pipe.vae.enable_tiling()
    
    if use_xformers:
        base_pipe.enable_xformers_memory_efficient_attention()
        logger.info("Enabled xformers memory efficient attention")
    
    base_pipe = base_pipe.to("cuda")
    
    if compile_unet:
        logger.info("Compiling UNet with torch.compile...")
        base_pipe.unet = torch.compile(base_pipe.unet, mode="max-autotune")
        logger.info("UNet compiled")
    
    refiner_pipe = None
    if use_refiner:
        logger.info(f"Loading refiner model {REFINER_MODEL_ID} (FP16)...")
        refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            REFINER_MODEL_ID,
            text_encoder_2=base_pipe.text_encoder_2,
            vae=base_pipe.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        )
        refiner_pipe.enable_attention_slicing()
        refiner_pipe.vae.enable_slicing()
        refiner_pipe.vae.enable_tiling()
        refiner_pipe = refiner_pipe.to("cuda")
        if use_xformers:
            refiner_pipe.enable_xformers_memory_efficient_attention()
        if compile_unet:
            logger.info("Compiling refiner UNet with torch.compile...")
            refiner_pipe.unet = torch.compile(refiner_pipe.unet, mode="max-autotune")
        logger.info("Refiner loaded")
    
    return base_pipe, refiner_pipe


@torch.no_grad()
def generate_group(base_pipe, refiner_pipe, captions, out_dir, group_name, batch_size=4, num_inference_steps=30, high_noise_frac=0.8):
    os.makedirs(out_dir, exist_ok=True)
    gen_files = []
    total = len(captions)
    pbar = tqdm(total=total, desc=f"Generating [{group_name}]", unit="img")
    
    use_refiner = refiner_pipe is not None
    
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_captions = captions[batch_start:batch_end]
        current_batch = batch_end - batch_start
        
        prompts = batch_captions
        negative_prompts = [NEGATIVE_PROMPT] * current_batch
        generators = [torch.Generator(device="cuda").manual_seed(SEED + (batch_start + i)) for i in range(current_batch)]
        
        if use_refiner:
            latent_result = base_pipe(
                prompt=prompts,
                negative_prompt=negative_prompts,
                height=IMAGE_SIZE,
                width=IMAGE_SIZE,
                num_inference_steps=num_inference_steps,
                denoising_end=high_noise_frac,
                output_type="latent",
                guidance_scale=7.5,
                generator=generators,
            )
            latents = latent_result[0]
            result = refiner_pipe(
                prompt=prompts,
                negative_prompt=negative_prompts,
                image=latents,
                num_inference_steps=num_inference_steps,
                denoising_start=high_noise_frac,
                guidance_scale=7.5,
                generator=generators,
            )
            images = result.images
        else:
            result = base_pipe(
                prompt=prompts,
                negative_prompt=negative_prompts,
                height=IMAGE_SIZE,
                width=IMAGE_SIZE,
                num_inference_steps=num_inference_steps,
                guidance_scale=7.5,
                generator=generators,
            )
            images = result.images
        
        for i, img in enumerate(images):
            idx = batch_start + i
            fname = f"{group_name}_{idx:03d}.png"
            img.save(os.path.join(out_dir, fname), format="PNG")
            gen_files.append(fname)
            pbar.set_postfix_str(f"last: {batch_captions[i][:45]}")
        
        pbar.update(current_batch)
        
        torch.cuda.empty_cache()
        
    return gen_files


def compute_clip_scores(captions, gen_files, out_dir, model, preprocess, tokenizer, device, batch_size=16):
    scores = []
    pairs = []
    total = len(captions)
    
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_captions = captions[batch_start:batch_end]
        batch_files = gen_files[batch_start:batch_end]
        current_batch = batch_end - batch_start
        
        img_tensors = []
        for caption, fname in zip(batch_captions, batch_files):
            img_path = os.path.join(out_dir, fname)
            img = Image.open(img_path).convert("RGB")
            img_tensors.append(preprocess(img))
        img_batch = torch.stack(img_tensors).to(device)
        text_batch = tokenizer(batch_captions).to(device)

        with torch.no_grad(), torch.amp.autocast("cuda" if device == "cuda" else "cpu"):
            img_feats = model.encode_image(img_batch)
            txt_feats = model.encode_text(text_batch)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)
            batch_scores = torch.diag(img_feats @ txt_feats.T).cpu().numpy()

        for score, caption, fname in zip(batch_scores, batch_captions, batch_files):
            scores.append(float(score))
            pairs.append({"caption": caption, "gen_file": fname, "clip_score": round(float(score), 4)})

    return float(np.mean(scores)), float(np.std(scores)), pairs


def main():
    parser = argparse.ArgumentParser(description="Compare simple vs complex captions with SDXL")
    parser.add_argument("--no-refiner", action="store_true", help="Disable SDXL refiner")
    parser.add_argument("--xformers", action="store_true", help="Enable xformers memory efficient attention")
    parser.add_argument("--num-each", type=int, default=NUM_EACH, help="Number of images per group")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for generation (RTX 4090 recommended: 4)")
    parser.add_argument("--steps", type=int, default=30, help="Number of inference steps")
    parser.add_argument("--high-noise-frac", type=float, default=0.8, help="High noise fraction for refiner")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile for UNet (significant speedup after compilation)")
    args = parser.parse_args()

    num_each = args.num_each
    
    logger = setup_logger("complex_compare")

    logger.info("=" * 60)
    logger.info("Complex Caption Comparison Experiment (SDXL)")
    logger.info("=" * 60)
    logger.info(f"Use refiner: {not args.no_refiner}")
    logger.info(f"Use xformers: {args.xformers}")
    logger.info(f"Use torch.compile: {args.compile}")
    logger.info(f"Images per group: {num_each}")
    logger.info(f"Inference steps: {args.steps}")

    if not CAPTION_JSON.exists():
        logger.error(f"Caption JSON not found: {CAPTION_JSON}")
        sys.exit(1)

    logger.info("[1/4] Loading and selecting captions...")
    simple_captions, complex_captions = load_captions(logger, num_each)
    logger.info(f"[1/4] Selected {len(simple_captions)} simple, {len(complex_captions)} complex")

    logger.info("[2/4] Loading model...")
    base_pipe, refiner_pipe = build_pipeline(logger, use_refiner=not args.no_refiner, use_xformers=args.xformers, compile_unet=args.compile)

    logger.info("[3/4] Generating images...")
    simple_files = generate_group(base_pipe, refiner_pipe, simple_captions, SIMPLE_DIR, "simple", 
                                 batch_size=args.batch_size, num_inference_steps=args.steps, 
                                 high_noise_frac=args.high_noise_frac)
    complex_files = generate_group(base_pipe, refiner_pipe, complex_captions, COMPLEX_DIR, "complex", 
                                   batch_size=args.batch_size, num_inference_steps=args.steps, 
                                   high_noise_frac=args.high_noise_frac)

    logger.info("[4/4] Computing CLIP Scores...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    clip_model = clip_model.to(device).eval()
    if args.compile:
        logger.info("Compiling CLIP model with torch.compile...")
        clip_model = torch.compile(clip_model, mode="max-autotune")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    s_mean, s_std, s_pairs = compute_clip_scores(
        simple_captions, simple_files, SIMPLE_DIR, clip_model, preprocess, tokenizer, device
    )
    c_mean, c_std, c_pairs = compute_clip_scores(
        complex_captions, complex_files, COMPLEX_DIR, clip_model, preprocess, tokenizer, device
    )

    delta = c_mean - s_mean
    delta_pct = delta / s_mean * 100

    results = {
        "simple": {
            "num": NUM_EACH,
            "caption_max_len": SIMPLE_MAX_LEN,
            "clip_score_mean": round(s_mean, 4),
            "clip_score_std": round(s_std, 4),
            "pairs": s_pairs,
        },
        "complex": {
            "num": NUM_EACH,
            "caption_min_len": COMPLEX_MIN_LEN,
            "clip_score_mean": round(c_mean, 4),
            "clip_score_std": round(c_std, 4),
            "pairs": c_pairs,
        },
        "delta": round(delta, 4),
        "delta_pct": round(delta_pct, 2),
    }

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info("=" * 60)
    logger.info("Comparison Summary")
    logger.info("=" * 60)
    logger.info(f"Simple  captions CLIP Score: {s_mean:.4f} ± {s_std:.4f}")
    logger.info(f"Complex captions CLIP Score: {c_mean:.4f} ± {c_std:.4f}")
    logger.info(f"Delta: {delta:+.4f}  ({delta_pct:+.2f}%)")
    if delta < 0:
        logger.info(f"=> 复杂描述 CLIP Score 下降 {abs(delta_pct):.2f}%（模型对多对象/长描述的对齐能力更弱）")
    else:
        logger.info(f"=> 复杂描述 CLIP Score 上升 {delta_pct:.2f}%（更丰富的描述提升了图文对齐）")
    logger.info(f"Results saved: {RESULTS_FILE}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
