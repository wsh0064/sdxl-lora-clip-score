import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline, StableDiffusionXLImg2ImgPipeline
from PIL import Image
from tqdm import tqdm

from logger_utils import setup_logger

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
CAPTION_JSON = PROJECT_ROOT / "data" / "annotations_trainval2017" / "annotations" / "captions_val2017.json"
VAL_IMAGES_DIR = PROJECT_ROOT / "data" / "val2017"

MODEL_IDS = {
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    "sdxl-refiner": "stabilityai/stable-diffusion-xl-refiner-1.0",
    "sd15": "runwayml/stable-diffusion-v1-5",
}
DEFAULT_SIZES = {
    "sdxl": 1024,
    "sd15": 512,
}

NEGATIVE_PROMPT = (
    "blurry, ugly, distorted, low quality, worst quality, "
    "low resolution, jpeg artifacts, watermark, signature, "
    "text, cropped, out of frame, duplicate"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Text-to-Image Generation using Stable Diffusion (XL or v1.5)")
    parser.add_argument("--model", choices=["sdxl", "sd15"], default="sdxl",
                        help="Model to use: sdxl (Stable Diffusion XL) or sd15 (SD v1.5)")
    parser.add_argument("--lora_weights", type=str, default="",
                        help="Path to LoRA weights directory (e.g., sdxl_lora_coco)")
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", dest="inference_steps", type=int, default=30)
    parser.add_argument("--height", type=int, default=0,
                        help="Image height (0 = auto: 1024 for sdxl, 512 for sd15)")
    parser.add_argument("--width", type=int, default=0,
                        help="Image width (0 = auto: 1024 for sdxl, 512 for sd15)")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--caption_json", type=str, default=str(CAPTION_JSON))
    parser.add_argument("--val_images_dir", type=str, default=str(VAL_IMAGES_DIR))
    parser.add_argument("--gen_dir", type=str, default="",
                        help="Output dir for generated images (default: generated_images_<model>)")
    parser.add_argument("--real_dir", type=str, default=str(PROJECT_ROOT / "data" / "real_images"))
    parser.add_argument("--mapping_file", type=str, default="",
                        help="Mapping JSON path (default: caption_image_mapping_<model>.json)")
    parser.add_argument("--negative_prompt", type=str, default=NEGATIVE_PROMPT)
    parser.add_argument("--skip_real_images", action="store_true")
    parser.add_argument(
        "--caption_selection",
        choices=["first", "random", "longest", "shortest"],
        default="first",
    )
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for generation (recommended: 4 for RTX 4090)")
    parser.add_argument("--use_refiner", action="store_true",
                        help="Use SDXL refiner model")
    parser.add_argument("--refiner_steps", type=int, default=20,
                        help="Number of refiner steps")
    parser.add_argument("--high_noise_frac", type=float, default=0.8,
                        help="Fraction of steps to run base model (0.8-0.9 recommended)")
    parser.add_argument("--use_xformers", action="store_true",
                        help="Use xformers memory efficient attention")
    parser.add_argument("--compile", action="store_true",
                        help="Compile model with torch.compile for faster inference")
    args = parser.parse_args()

    default_size = DEFAULT_SIZES[args.model]
    if args.height == 0:
        args.height = default_size
    if args.width == 0:
        args.width = default_size
    if not args.gen_dir:
        args.gen_dir = str(PROJECT_ROOT / "outputs" / "images" / f"generated_images_{args.model}")
    if not args.mapping_file:
        args.mapping_file = str(PROJECT_ROOT / "outputs" / "results" / f"caption_image_mapping_{args.model}.json")

    return args


def load_and_sample_captions(json_path: str, num_samples: int, selection_strategy: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_id_to_file = {img["id"]: img["file_name"] for img in data["images"]}
    image_id_to_anns: dict = {}
    for ann in data["annotations"]:
        iid = ann["image_id"]
        if iid not in image_id_to_anns:
            image_id_to_anns[iid] = []
        image_id_to_anns[iid].append(ann)

    all_image_ids = list(image_id_to_anns.keys())
    sampled_ids = random.sample(all_image_ids, min(num_samples, len(all_image_ids)))

    samples = []
    for iid in sampled_ids:
        anns = image_id_to_anns[iid]
        file_name = image_id_to_file[iid]

        if selection_strategy == "first":
            ann = anns[0]
        elif selection_strategy == "random":
            ann = random.choice(anns)
        elif selection_strategy == "longest":
            ann = max(anns, key=lambda a: len(a["caption"]))
        else:
            ann = min(anns, key=lambda a: len(a["caption"]))

        samples.append({"image_id": iid, "file_name": file_name, "caption": ann["caption"]})
    return samples


def build_pipeline(model_key: str, args, logger):
    model_id = MODEL_IDS[model_key]
    logger.info(f"Loading model: {model_id}")
    
    pipe_kwargs = {
        "torch_dtype": torch.float16,
        "use_safetensors": True,
        "variant": "fp16",
    }
    
    if model_key == "sdxl":
        pipe = StableDiffusionXLPipeline.from_pretrained(model_id, **pipe_kwargs)
    else:
        pipe_kwargs["safety_checker"] = None
        pipe = StableDiffusionPipeline.from_pretrained(model_id, **pipe_kwargs)
    
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    
    if args.use_xformers:
        pipe.enable_xformers_memory_efficient_attention()
    
    if args.compile:
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
    
    pipe = pipe.to("cuda")
    
    if args.lora_weights:
        pipe.load_lora_weights(args.lora_weights)
        logger.info(f"Loaded LoRA weights from: {args.lora_weights}")
    
    return pipe


def build_refiner_pipe(pipe, args, logger):
    logger.info(f"Loading refiner: {MODEL_IDS['sdxl-refiner']}")
    refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        MODEL_IDS["sdxl-refiner"],
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
        text_encoder_2=pipe.text_encoder_2,
        vae=pipe.vae,
    )
    refiner.enable_attention_slicing()
    refiner.enable_vae_slicing()
    refiner.enable_vae_tiling()
    
    if args.use_xformers:
        refiner.enable_xformers_memory_efficient_attention()
    
    if args.compile:
        refiner.unet = torch.compile(refiner.unet, mode="reduce-overhead", fullgraph=True)
    
    refiner = refiner.to("cuda")
    return refiner


@torch.no_grad()
def generate_images(pipe, samples, args, logger, refiner=None):
    os.makedirs(args.gen_dir, exist_ok=True)
    mapping = []

    batch_size = args.batch_size
    total_samples = len(samples)
    
    pbar = tqdm(total=total_samples, desc="Generating", unit="img")
    
    for batch_start in range(0, total_samples, batch_size):
        batch_end = min(batch_start + batch_size, total_samples)
        batch_samples = samples[batch_start:batch_end]
        current_batch_size = batch_end - batch_start
        
        prompts = [item["caption"] for item in batch_samples]
        negative_prompts = [args.negative_prompt] * current_batch_size
        generators = [torch.Generator(device="cuda").manual_seed(args.seed + batch_start + i) for i in range(current_batch_size)]
        
        base_kwargs = {
            "prompt": prompts,
            "negative_prompt": negative_prompts,
            "height": args.height,
            "width": args.width,
            "num_inference_steps": args.inference_steps,
            "guidance_scale": args.guidance_scale,
            "generator": generators,
        }
        
        if args.model == "sdxl" and args.use_refiner and refiner is not None:
            base_kwargs["output_type"] = "latent"
            base_kwargs["denoising_end"] = args.high_noise_frac
            
            result = pipe(**base_kwargs)
            
            refiner_result = refiner(
                prompt=prompts,
                negative_prompt=negative_prompts,
                num_inference_steps=args.inference_steps,
                denoising_start=args.high_noise_frac,
                image=result.latents,
            )
            images = refiner_result.images
        else:
            result = pipe(**base_kwargs)
            images = result.images
        
        for i, img in enumerate(images):
            idx = batch_start + i
            item = batch_samples[i]
            gen_name = f"gen_{idx:04d}.png"
            img.save(os.path.join(args.gen_dir, gen_name), format="PNG")
            logger.info(f"[{idx+1}/{total_samples}] Saved {gen_name} | caption: {item['caption'][:60]}")
            
            mapping.append({
                "index": idx,
                "image_id": item["image_id"],
                "gen_file": gen_name,
                "real_file": item["file_name"],
                "caption": item["caption"],
                "seed": args.seed + idx,
            })
        
        pbar.update(current_batch_size)

    pbar.close()
    return mapping


def copy_real_images(samples, args, logger):
    os.makedirs(args.real_dir, exist_ok=True)

    if not os.path.exists(args.val_images_dir):
        logger.warning(f"val2017 images not found at: {args.val_images_dir}")
        return 0

    copied = 0
    for idx, item in enumerate(tqdm(samples, desc="Copying real images", unit="img")):
        src = os.path.join(args.val_images_dir, item["file_name"])
        dst = os.path.join(args.real_dir, f"real_{idx:04d}.png")
        if os.path.exists(src):
            img = Image.open(src).convert("RGB").resize((1024, 1024), Image.LANCZOS)
            img.save(dst, format="PNG")
            copied += 1
        else:
            logger.warning(f"Real image not found: {src}")

    logger.info(f"Copied {copied}/{len(samples)} real images to {args.real_dir}")
    return copied


def main():
    args = parse_args()
    random.seed(args.seed)

    logger = setup_logger(f"generate_{args.model}")

    model_label = {"sdxl": "Stable Diffusion XL", "sd15": "Stable Diffusion v1.5"}[args.model]
    logger.info("=" * 60)
    logger.info(f"Text-to-Image Generation  ({model_label})")
    logger.info("=" * 60)
    logger.info(f"Model:           {MODEL_IDS[args.model]}")
    logger.info(f"Num samples:     {args.num_samples}")
    logger.info(f"Batch size:      {args.batch_size}")
    logger.info(f"Inference steps: {args.inference_steps}")
    logger.info(f"Image size:      {args.width}x{args.height}")
    logger.info(f"Guidance scale:  {args.guidance_scale}")
    logger.info(f"Gen dir:         {args.gen_dir}")
    logger.info(f"Mapping file:    {args.mapping_file}")
    if args.model == "sdxl":
        logger.info(f"Use refiner:     {args.use_refiner}")
        if args.use_refiner:
            logger.info(f"High noise frac: {args.high_noise_frac}")
            logger.info(f"Refiner steps:   {args.refiner_steps}")
    logger.info(f"Use xformers:    {args.use_xformers}")
    logger.info(f"Compile model:   {args.compile}")
    logger.info("=" * 60)

    if not os.path.exists(args.caption_json):
        logger.error(f"Caption JSON not found: {args.caption_json}")
        sys.exit(1)

    logger.info(f"[1/4] Sampling {args.num_samples} captions...")
    samples = load_and_sample_captions(args.caption_json, args.num_samples, args.caption_selection)
    logger.info(f"[1/4] Got {len(samples)} captions.")

    logger.info(f"[2/4] Loading {model_label} (FP16)...")
    pipe = build_pipeline(args.model, args, logger)

    refiner = None
    if args.model == "sdxl" and args.use_refiner:
        logger.info(f"[2/4] Loading SDXL Refiner...")
        refiner = build_refiner_pipe(pipe, args, logger)

    logger.info(f"[3/4] Generating {len(samples)} images...")
    mapping = generate_images(pipe, samples, args, logger, refiner=refiner)

    logger.info("[4/4] Copying real images...")
    if args.skip_real_images:
        logger.info("[4/4] Skipped.")
    else:
        copy_real_images(samples, args, logger)

    with open(args.mapping_file, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    logger.info(f"Mapping saved: {args.mapping_file}")

    logger.info("=" * 60)
    logger.info("Done!")
    logger.info(f"Generated: {args.gen_dir}")
    logger.info(f"Real:      {args.real_dir}")
    logger.info(f"Mapping:   {args.mapping_file}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
