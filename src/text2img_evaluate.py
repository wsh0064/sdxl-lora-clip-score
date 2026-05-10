import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/data/.cache/huggingface"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/data/.cache/huggingface/hub"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TORCH_HOME"] = "/data/.cache/torch"

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import open_clip
from pytorch_fid.fid_score import calculate_fid_given_paths

from logger_utils import setup_logger

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_GEN_DIR = PROJECT_ROOT / "outputs" / "images" / "results_100_samples" / "generated_images_sdxl"
DEFAULT_REAL_DIR = PROJECT_ROOT / "data" / "real_images"
DEFAULT_MAPPING_FILE = PROJECT_ROOT / "outputs" / "results" / "caption_image_mapping_sdxl.json"
DEFAULT_RESULTS_FILE = PROJECT_ROOT / "outputs" / "results" / "evaluation_results_sdxl.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Text-to-Image: FID + CLIP Score")
    parser.add_argument("--gen_dir", type=str, default=str(DEFAULT_GEN_DIR))
    parser.add_argument("--real_dir", type=str, default=str(DEFAULT_REAL_DIR))
    parser.add_argument("--mapping_file", type=str, default=str(DEFAULT_MAPPING_FILE))
    parser.add_argument("--results_file", type=str, default=str(DEFAULT_RESULTS_FILE))
    parser.add_argument("--clip_model", type=str, default="ViT-B-32")
    parser.add_argument("--clip_pretrained", type=str, default="openai")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--dims", type=int, default=2048)
    parser.add_argument("--skip_fid", action="store_true")
    parser.add_argument("--skip_clip", action="store_true")
    return parser.parse_args()


def resolve_device(requested: str, logger=None) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        msg = "CUDA not available, falling back to CPU"
        if logger:
            logger.warning(msg)
        return "cpu"
    return requested


def compute_fid(args, logger) -> float | None:
    logger.info("--- FID Score ---")
    for d in [args.gen_dir, args.real_dir]:
        if not os.path.exists(d):
            logger.error(f"Directory not found: {d}")
            return None

    device = resolve_device(args.device, logger)
    logger.info(f"Real:      {args.real_dir}")
    logger.info(f"Generated: {args.gen_dir}")
    logger.info(f"Device: {device}  Batch: {args.batch_size}  Dims: {args.dims}")

    fid = calculate_fid_given_paths(
        paths=[args.real_dir, args.gen_dir],
        batch_size=args.batch_size,
        device=device,
        dims=args.dims,
        num_workers=args.workers,
    )
    logger.info(f"FID Score: {fid:.4f}")
    return fid


def load_clip_model(args, logger):
    device = resolve_device(args.device, logger)
    logger.info(f"Loading CLIP: {args.clip_model} ({args.clip_pretrained})")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.clip_model, pretrained=args.clip_pretrained
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(args.clip_model)
    return model, preprocess, tokenizer, device


def compute_clip_scores(args, model, preprocess, tokenizer, device, logger):
    logger.info("--- CLIP Score ---")
    if not os.path.exists(args.mapping_file):
        logger.error(f"Mapping file not found: {args.mapping_file}")
        return None, None, []

    with open(args.mapping_file, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    logger.info(f"Computing scores for {len(mapping)} pairs...")
    all_scores = []
    all_pairs = []
    batch_size = args.batch_size

    for batch_start in tqdm(range(0, len(mapping), batch_size), desc="CLIP scoring", unit="batch"):
        batch_end = min(batch_start + batch_size, len(mapping))
        batch_items = mapping[batch_start:batch_end]
        current_batch = batch_end - batch_start

        img_batch = []
        text_batch = []
        valid_items = []

        for item in batch_items:
            img_path = os.path.join(args.gen_dir, item["gen_file"])
            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = preprocess(img)
                img_batch.append(img_tensor)
                text_batch.append(item["caption"])
                valid_items.append(item)
            except FileNotFoundError:
                logger.warning(f"Missing image: {img_path}")
                continue

        if not img_batch:
            continue

        img_input = torch.stack(img_batch).to(device)
        text_input = tokenizer(text_batch).to(device)

        with torch.no_grad(), torch.amp.autocast("cuda" if device == "cuda" else "cpu"):
            img_feat = model.encode_image(img_input)
            txt_feat = model.encode_text(text_input)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            scores = torch.diag(img_feat @ txt_feat.T).cpu().numpy()

        for score, item in zip(scores, valid_items):
            all_scores.append(float(score))
            all_pairs.append({
                "caption": item["caption"],
                "gen_file": item["gen_file"],
                "clip_score": round(float(score), 4),
            })

    if not all_scores:
        logger.error("No valid pairs found.")
        return None, None, []

    mean_s = float(np.mean(all_scores))
    std_s = float(np.std(all_scores))
    logger.info(f"CLIP Score mean: {mean_s:.4f}   std: {std_s:.4f}")

    sorted_pairs = sorted(all_pairs, key=lambda x: x["clip_score"], reverse=True)
    logger.info("Top 5:")
    for p in sorted_pairs[:5]:
        logger.info(f"  [{p['clip_score']:.4f}] {p['caption'][:65]}")
    logger.info("Bottom 5:")
    for p in sorted_pairs[-5:]:
        logger.info(f"  [{p['clip_score']:.4f}] {p['caption'][:65]}")

    return mean_s, std_s, all_pairs


def count_objects(caption: str) -> int:
    """用逗号和 and 的数量估算 caption 中描述的对象数量。"""
    text = caption.lower()
    separators = len(re.findall(r",\s*", text)) + len(re.findall(r"\band\b", text))
    if separators == 0:
        return 1
    elif separators == 1:
        return 2
    else:
        return 3


def bucket_by_object_count(all_pairs: list, logger) -> dict:
    """按对象数量分桶统计 CLIP Score。"""
    buckets: dict[int, list] = {1: [], 2: [], 3: []}
    for p in all_pairs:
        k = count_objects(p["caption"])
        buckets[min(k, 3)].append(p["clip_score"])

    labels = {1: "1 object ", 2: "2 objects", 3: "3+ objects"}
    logger.info("--- CLIP Score by Object Count (heuristic) ---")
    stats = {}
    for k in [1, 2, 3]:
        scores = buckets[k]
        if not scores:
            continue
        mean_v = float(np.mean(scores))
        std_v = float(np.std(scores))
        logger.info(f"  {labels[k]}:  mean={mean_v:.4f}  std={std_v:.4f}  n={len(scores)}")
        stats[labels[k].strip()] = {
            "mean": round(mean_v, 4),
            "std": round(std_v, 4),
            "n": len(scores),
        }
    return stats


def main():
    args = parse_args()
    logger = setup_logger("evaluate")

    logger.info("=" * 60)
    logger.info("Evaluation: FID + CLIP Score")
    logger.info("=" * 60)
    logger.info(f"Gen dir:    {args.gen_dir}")
    logger.info(f"Real dir:   {args.real_dir}")
    logger.info(f"Mapping:    {args.mapping_file}")
    logger.info(f"CLIP model: {args.clip_model} ({args.clip_pretrained})")
    logger.info("=" * 60)

    fid = None
    if not args.skip_fid:
        fid = compute_fid(args, logger)

    mean_clip = std_clip = None
    all_pairs = []
    bucket_stats = {}
    if not args.skip_clip:
        model, preprocess, tokenizer, device = load_clip_model(args, logger)
        mean_clip, std_clip, all_pairs = compute_clip_scores(args, model, preprocess, tokenizer, device, logger)
        if all_pairs:
            bucket_stats = bucket_by_object_count(all_pairs, logger)

    results = {
        "fid": round(fid, 4) if fid is not None else None,
        "clip_score_mean": round(mean_clip, 4) if mean_clip is not None else None,
        "clip_score_std": round(std_clip, 4) if std_clip is not None else None,
        "num_samples": len(all_pairs),
        "bucket_stats": bucket_stats,
        "pairs": all_pairs,
        "config": {
            "gen_dir": args.gen_dir,
            "real_dir": args.real_dir,
            "clip_model": args.clip_model,
            "clip_pretrained": args.clip_pretrained,
            "fid_dims": args.dims,
            "device": args.device,
        },
    }

    with open(args.results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info("=" * 60)
    logger.info("Evaluation Summary")
    logger.info("=" * 60)
    logger.info(f"FID Score:         {fid:.4f}" if fid is not None else "FID Score:         skipped")
    if mean_clip is not None:
        logger.info(f"CLIP Score mean:   {mean_clip:.4f}")
        logger.info(f"CLIP Score std:    {std_clip:.4f}")
    else:
        logger.info("CLIP Score:        skipped")
    logger.info(f"Num samples:       {len(all_pairs)}")
    logger.info(f"Results saved:     {args.results_file}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
