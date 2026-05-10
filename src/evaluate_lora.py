"""
评估 LoRA 模型，并与基线对比
用法: python evaluate_lora.py
"""
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

from logger_utils import setup_logger

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/data/.cache/huggingface"

logger = setup_logger("evaluate_lora")

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

cmd = [
    sys.executable, str(SCRIPT_DIR / "text2img_evaluate.py"),
    "--gen_dir", str(PROJECT_ROOT / "outputs" / "images" / "results_100_samples" / "generated_images_lora"),
    "--real_dir", str(PROJECT_ROOT / "data" / "real_images"),
    "--mapping_file", str(PROJECT_ROOT / "outputs" / "results" / "caption_image_mapping_lora.json"),
    "--results_file", str(PROJECT_ROOT / "outputs" / "results" / "evaluation_results_lora.json"),
]

logger.info("=" * 60)
logger.info("开始评估 LoRA 模型")
logger.info("=" * 60)
logger.info("评估参数:")
logger.info(f"  - 生成图片目录: {PROJECT_ROOT / 'outputs' / 'images' / 'results_100_samples' / 'generated_images_lora'}")
logger.info(f"  - 真实图片目录: {PROJECT_ROOT / 'data' / 'real_images'}")
logger.info(f"  - 映射文件: {PROJECT_ROOT / 'outputs' / 'results' / 'caption_image_mapping_lora.json'}")
logger.info(f"  - 结果文件: {PROJECT_ROOT / 'outputs' / 'results' / 'evaluation_results_lora.json'}")
logger.info("=" * 60)

try:
    subprocess.run(cmd, check=True)

    # 加载 LoRA 结果
    lora = json.load(open(str(PROJECT_ROOT / "outputs" / "results" / "evaluation_results_lora.json")))
    logger.info("\n" + "=" * 60)
    logger.info("LoRA 评估结果")
    logger.info("=" * 60)
    logger.info(f"FID Score:    {lora['fid']:.2f}")
    logger.info(f"CLIP Score:   {lora['clip_score_mean']:.4f} (±{lora['clip_score_std']:.4f})")
    logger.info(f"样本数量:     {lora['num_samples']}")
    logger.info("=" * 60)

    # 尝试加载基线进行对比
    base_path = PROJECT_ROOT / "outputs" / "results" / "evaluation_results_sdxl.json"
    if base_path.exists():
        base = json.load(open(base_path))
        logger.info("\n" + "=" * 60)
        logger.info("对比结果 (基线 SDXL vs LoRA)")
        logger.info("=" * 60)
        logger.info(f"基线 FID:    {base['fid']:.2f}")
        logger.info(f"LoRA FID:    {lora['fid']:.2f}")
        fid_change = lora['fid'] - base['fid']
        fid_change_pct = (1 - lora['fid']/base['fid']) * 100
        logger.info(f"  变化:      {fid_change:+.2f} ({fid_change_pct:+.1f}%)")
        logger.info("")
        logger.info(f"基线 CLIP:  {base['clip_score_mean']:.4f}")
        logger.info(f"LoRA CLIP:  {lora['clip_score_mean']:.4f}")
        clip_change = lora['clip_score_mean'] - base['clip_score_mean']
        clip_change_pct = (lora['clip_score_mean']/base['clip_score_mean'] - 1) * 100
        logger.info(f"  变化:      {clip_change:+.4f} ({clip_change_pct:+.1f}%)")
        logger.info("=" * 60)
    else:
        logger.info("\n提示: 未找到基线评估结果")
        logger.info("如需对比，请先运行基线评估:")
        logger.info("  python text2img_generate_sdxl.py --model sdxl --num_samples 100")
        logger.info("  然后运行基线评估后再运行本脚本")
        logger.info("=" * 60)

except subprocess.CalledProcessError as e:
    logger.error("=" * 60)
    logger.error(f"评估失败！返回码: {e.returncode}")
    logger.error("错误详情:")
    logger.error(traceback.format_exc())
    logger.error("=" * 60)
    sys.exit(1)
except FileNotFoundError as e:
    logger.error("=" * 60)
    logger.error(f"文件未找到: {str(e)}")
    logger.error("请确保已运行基线评估和 LoRA 生成")
    logger.error(traceback.format_exc())
    logger.error("=" * 60)
    sys.exit(1)
except Exception as e:
    logger.error("=" * 60)
    logger.error(f"发生未知错误: {str(e)}")
    logger.error(traceback.format_exc())
    logger.error("=" * 60)
    sys.exit(1)
