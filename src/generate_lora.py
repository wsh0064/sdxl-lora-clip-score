"""
使用 LoRA 权重生成图片
用法: python generate_lora.py
"""
import os
import subprocess
import sys
import traceback
from pathlib import Path

from logger_utils import setup_logger

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

logger = setup_logger("generate_lora")

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

cmd = [
    sys.executable, str(SCRIPT_DIR / "text2img_generate_sdxl.py"),
    "--model", "sdxl",
    "--num_samples", "100",
    "--lora_weights", str(PROJECT_ROOT / "models" / "sdxl_lora_coco"),
    "--gen_dir", str(PROJECT_ROOT / "outputs" / "images" / "generated_images_lora"),
    "--mapping_file", str(PROJECT_ROOT / "outputs" / "results" / "caption_image_mapping_lora.json"),
    "--seed", "42",
]

logger.info("=" * 60)
logger.info("使用 LoRA 权重生成图片")
logger.info("=" * 60)
logger.info("生成参数:")
logger.info("  - 模型: sdxl (stable-diffusion-xl-base-1.0)")
logger.info("  - 样本数: 100")
logger.info(f"  - LoRA 权重: {PROJECT_ROOT / 'models' / 'sdxl_lora_coco'}")
logger.info(f"  - 输出目录: {PROJECT_ROOT / 'outputs' / 'images' / 'generated_images_lora'}")
logger.info(f"  - 映射文件: {PROJECT_ROOT / 'outputs' / 'results' / 'caption_image_mapping_lora.json'}")
logger.info("  - 随机种子: 42")
logger.info("=" * 60)

try:
    subprocess.run(cmd, check=True)
    logger.info("=" * 60)
    logger.info("生成完成！")
    logger.info(f"图片已保存到: {PROJECT_ROOT / 'outputs' / 'images' / 'generated_images_lora'}")
    logger.info(f"映射文件: {PROJECT_ROOT / 'outputs' / 'results' / 'caption_image_mapping_lora.json'}")
    logger.info("接下来可以运行: python evaluate_lora.py")
    logger.info("=" * 60)
except subprocess.CalledProcessError as e:
    logger.error("=" * 60)
    logger.error(f"生成失败！返回码: {e.returncode}")
    logger.error("错误详情:")
    logger.error(traceback.format_exc())
    logger.error("=" * 60)
    sys.exit(1)
except Exception as e:
    logger.error("=" * 60)
    logger.error(f"发生未知错误: {str(e)}")
    logger.error(traceback.format_exc())
    logger.error("=" * 60)
    sys.exit(1)
