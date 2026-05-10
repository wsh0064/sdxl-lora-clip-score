#!/usr/bin/env python
"""
SDXL LoRA 训练入口
"""
import os
import subprocess
import sys
import traceback
from pathlib import Path

from logger_utils import setup_logger

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/data/.cache/huggingface"

logger = setup_logger("train_lora_sdxl")

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

cmd = [
    "accelerate", "launch",
    str(SCRIPT_DIR / "train_text_to_image_lora_sdxl.py"),
    "--pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0",
    "--train_data_dir=" + str(PROJECT_ROOT / "data" / "train_lora_dataset"),
    "--caption_column=text",
    "--resolution=1024",
    "--center_crop",
    "--random_flip",
    "--train_batch_size=1",
    "--gradient_accumulation_steps=4",
    "--max_train_steps=2000",
    "--learning_rate=1e-4",
    "--lr_scheduler=cosine",
    "--lr_warmup_steps=200",
    "--mixed_precision=bf16",
    "--rank=16",
    "--output_dir=" + str(PROJECT_ROOT / "models" / "sdxl_lora_coco"),
    "--checkpointing_steps=1000",
    "--seed=42",
]

logger.info("=" * 60)
logger.info("开始训练 SDXL LoRA")
logger.info("=" * 60)
logger.info("训练参数:")
logger.info("  - 预训练模型: stabilityai/stable-diffusion-xl-base-1.0")
logger.info(f"  - 训练数据: {PROJECT_ROOT / 'data' / 'train_lora_dataset'}")
logger.info("  - 图像分辨率: 1024")
logger.info("  - 批量大小: 1 (梯度累积 x4)")
logger.info("  - 训练步数: 2000 steps")
logger.info("  - 学习率: 1e-4")
logger.info("  - 学习率调度: cosine")
logger.info("  - 热身步数: 200")
logger.info("  - 混合精度: bf16")
logger.info("  - LoRA rank: 16")
logger.info(f"  - 输出目录: {PROJECT_ROOT / 'models' / 'sdxl_lora_coco'}")
logger.info("  - 检查点间隔: 1000 steps")
logger.info("  - 随机种子: 42")
logger.info("=" * 60)

try:
    subprocess.run(cmd, check=True)
    logger.info("=" * 60)
    logger.info("训练完成！")
    logger.info(f"LoRA 权重已保存到: {PROJECT_ROOT / 'models' / 'sdxl_lora_coco'}")
    logger.info("接下来可以运行: python generate_lora.py")
    logger.info("=" * 60)
except subprocess.CalledProcessError as e:
    logger.error("=" * 60)
    logger.error(f"训练失败！返回码: {e.returncode}")
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
