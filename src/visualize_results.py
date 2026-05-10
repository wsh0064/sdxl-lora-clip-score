"""
生成「真实 vs 生成」并排对比网格图，并单独输出 CLIP Score 最高/最低各 3 对的对比图。
"""
import json
import os
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from logger_utils import setup_logger

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
MAPPING_FILE = PROJECT_ROOT / "outputs" / "results" / "caption_image_mapping_sdxl.json"
RESULTS_FILE = PROJECT_ROOT / "outputs" / "results" / "evaluation_results_sdxl.json"
RESULTS_FILE_FALLBACK = PROJECT_ROOT / "outputs" / "results" / "evaluation_results.json"
GEN_DIR = PROJECT_ROOT / "outputs" / "images" / "results_100_samples" / "generated_images_sdxl"
REAL_DIR = PROJECT_ROOT / "data" / "real_images"
VIZ_DIR = PROJECT_ROOT / "outputs" / "visualization"

THUMB_SIZE = 256
CAPTION_H = 48
PADDING = 8
FONT_SIZE = 12
COLS_PER_PAIR = 2
PAIRS_PER_PAGE = 10


def get_font(size=FONT_SIZE):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def load_image(path: str, size=THUMB_SIZE) -> Image.Image:
    try:
        img = Image.open(path).convert("RGB")
        img = img.resize((size, size), Image.LANCZOS)
        return img
    except Exception:
        placeholder = Image.new("RGB", (size, size), color=(200, 200, 200))
        draw = ImageDraw.Draw(placeholder)
        draw.text((size // 4, size // 2), "NOT FOUND", fill=(100, 100, 100))
        return placeholder


def wrap_text(text: str, max_chars=38) -> str:
    lines = textwrap.wrap(text, width=max_chars)
    return "\n".join(lines[:2])


def build_pair_image(real_img, gen_img, caption: str, clip_score: float | None = None) -> Image.Image:
    font = get_font(FONT_SIZE)
    pair_w = THUMB_SIZE * 2 + PADDING * 3
    pair_h = THUMB_SIZE + CAPTION_H + PADDING * 2

    canvas = Image.new("RGB", (pair_w, pair_h), color=(245, 245, 245))
    draw = ImageDraw.Draw(canvas)

    canvas.paste(real_img, (PADDING, PADDING))
    canvas.paste(gen_img, (PADDING * 2 + THUMB_SIZE, PADDING))

    draw.text((PADDING, PADDING + THUMB_SIZE + 4), "Real", fill=(60, 100, 200), font=font)
    draw.text(
        (PADDING * 2 + THUMB_SIZE, PADDING + THUMB_SIZE + 4), "Generated", fill=(200, 80, 60), font=font
    )

    label = wrap_text(caption)
    if clip_score is not None:
        label = f"[{clip_score:.3f}] " + label
    draw.text((PADDING, PADDING + THUMB_SIZE + 18), label, fill=(50, 50, 50), font=font)

    return canvas


def save_grid(pairs_data, score_map, page_idx: int, logger):
    n = len(pairs_data)
    pair_w = THUMB_SIZE * 2 + PADDING * 3
    pair_h = THUMB_SIZE + CAPTION_H + PADDING * 2
    grid_w = pair_w
    grid_h = pair_h * n

    grid = Image.new("RGB", (grid_w, grid_h), color=(230, 230, 230))

    for row, item in enumerate(pairs_data):
        real_path = os.path.join(REAL_DIR, f"real_{item['index']:04d}.png")
        gen_path = os.path.join(GEN_DIR, item["gen_file"])
        real_img = load_image(real_path)
        gen_img = load_image(gen_path)
        clip_score = score_map.get(item["gen_file"])
        pair_img = build_pair_image(real_img, gen_img, item["caption"], clip_score)
        grid.paste(pair_img, (0, row * pair_h))

    out_path = VIZ_DIR / f"comparison_{page_idx:02d}.png"
    grid.save(out_path)
    logger.info(f"Saved: {out_path}")


def save_top_bottom(pairs_data, score_map, logger):
    scored = [
        (item, score_map.get(item["gen_file"], 0.0))
        for item in pairs_data
        if item["gen_file"] in score_map
    ]
    if not scored:
        logger.warning("No CLIP scores found, skipping top/bottom chart.")
        return

    scored.sort(key=lambda x: x[1], reverse=True)
    top3 = scored[:3]
    bottom3 = scored[-3:]
    selected = top3 + bottom3

    pair_w = THUMB_SIZE * 2 + PADDING * 3
    pair_h = THUMB_SIZE + CAPTION_H + PADDING * 2
    n = len(selected)
    grid = Image.new("RGB", (pair_w, pair_h * n), color=(230, 230, 230))
    font = get_font(14)
    draw = ImageDraw.Draw(grid)

    for row, (item, score) in enumerate(selected):
        real_path = os.path.join(REAL_DIR, f"real_{item['index']:04d}.png")
        gen_path = os.path.join(GEN_DIR, item["gen_file"])
        real_img = load_image(real_path)
        gen_img = load_image(gen_path)
        pair_img = build_pair_image(real_img, gen_img, item["caption"], score)
        grid.paste(pair_img, (0, row * pair_h))

    label_x = PADDING
    for row, (_, _) in enumerate(top3):
        draw.text((label_x, row * pair_h + 2), "TOP", fill=(0, 180, 0), font=font)
    for row, (_, _) in enumerate(bottom3):
        draw.text((label_x, (len(top3) + row) * pair_h + 2), "BTM", fill=(220, 0, 0), font=font)

    out_path = VIZ_DIR / "top_bottom.png"
    grid.save(out_path)
    logger.info(f"Saved: {out_path}")


def main():
    logger = setup_logger("visualize")

    logger.info("=" * 60)
    logger.info("Visualization: Real vs Generated")
    logger.info("=" * 60)

    if not MAPPING_FILE.exists():
        logger.error(f"Mapping file not found: {MAPPING_FILE}")
        logger.error("Please run text2img_generate_sdxl.py first.")
        return

    os.makedirs(VIZ_DIR, exist_ok=True)

    with open(MAPPING_FILE, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    score_map: dict[str, float] = {}
    results_file = RESULTS_FILE if RESULTS_FILE.exists() else RESULTS_FILE_FALLBACK
    if results_file.exists():
        with open(results_file, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
        for p in eval_data.get("pairs", []):
            score_map[p["gen_file"]] = p["clip_score"]
        logger.info(f"Loaded {len(score_map)} CLIP scores from {results_file.name}")
    else:
        logger.warning("evaluation_results.json not found; proceeding without CLIP scores.")

    pages = [
        mapping[i: i + PAIRS_PER_PAGE]
        for i in range(0, len(mapping), PAIRS_PER_PAGE)
    ]
    logger.info(f"Generating {len(pages)} comparison page(s) ({len(mapping)} pairs total)...")
    for page_idx, page_pairs in enumerate(pages):
        save_grid(page_pairs, score_map, page_idx, logger)

    logger.info("Generating top/bottom chart...")
    save_top_bottom(mapping, score_map, logger)

    logger.info("=" * 60)
    logger.info("Done!")
    logger.info(f"Output directory: {VIZ_DIR}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
