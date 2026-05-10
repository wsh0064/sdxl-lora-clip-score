"""
对比 100张 vs 200张 的评估结果
"""
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

# 加载结果
sdxl_100 = json.load(open(str(PROJECT_ROOT / "outputs" / "results" / "evaluation_results_sdxl.json")))
lora_100 = json.load(open(str(PROJECT_ROOT / "outputs" / "results" / "evaluation_results_lora.json")))
sdxl_200 = json.load(open(str(PROJECT_ROOT / "outputs" / "results" / "evaluation_results_sdxl_200.json")))
lora_200 = json.load(open(str(PROJECT_ROOT / "outputs" / "results" / "evaluation_results_lora_200.json")))

# 打印对比表格
print("=" * 100)
print("100张 vs 200张 对比结果")
print("=" * 100)
print()
print(f"{'指标':<15} {'基线(100)':<15} {'LoRA(100)':<15} {'基线(200)':<15} {'LoRA(200)':<15}")
print("-" * 100)
print(f"{'FID':<15} {sdxl_100['fid']:<15.2f} {lora_100['fid']:<15.2f} {sdxl_200['fid']:<15.2f} {lora_200['fid']:<15.2f}")
print(f"{'CLIP':<15} {sdxl_100['clip_score_mean']:<15.4f} {lora_100['clip_score_mean']:<15.4f} {sdxl_200['clip_score_mean']:<15.4f} {lora_200['clip_score_mean']:<15.4f}")
print(f"{'样本数':<15} {sdxl_100['num_samples']:<15} {lora_100['num_samples']:<15} {sdxl_200['num_samples']:<15} {lora_200['num_samples']:<15}")
print()

print("=" * 100)
print("详细分析")
print("=" * 100)
print()

# 100张结果分析
fid_100_improve = lora_100["fid"] - sdxl_100["fid"]
fid_100_pct = (1 - lora_100["fid"] / sdxl_100["fid"]) * 100
clip_100_improve = lora_100["clip_score_mean"] - sdxl_100["clip_score_mean"]
clip_100_pct = (lora_100["clip_score_mean"] / sdxl_100["clip_score_mean"] - 1) * 100

print("【100张样本】")
print(f"  FID:  基线={sdxl_100['fid']:.2f} -> LoRA={lora_100['fid']:.2f}")
print(f"         变化: {fid_100_improve:+.2f} ({fid_100_pct:+.1f}%)")
print(f"  CLIP: 基线={sdxl_100['clip_score_mean']:.4f} -> LoRA={lora_100['clip_score_mean']:.4f}")
print(f"         变化: {clip_100_improve:+.4f} ({clip_100_pct:+.1f}%)")
print()

# 200张结果分析
fid_200_improve = lora_200["fid"] - sdxl_200["fid"]
fid_200_pct = (1 - lora_200["fid"] / sdxl_200["fid"]) * 100
clip_200_improve = lora_200["clip_score_mean"] - sdxl_200["clip_score_mean"]
clip_200_pct = (lora_200["clip_score_mean"] / sdxl_200["clip_score_mean"] - 1) * 100

print("【200张样本】")
print(f"  FID:  基线={sdxl_200['fid']:.2f} -> LoRA={lora_200['fid']:.2f}")
print(f"         变化: {fid_200_improve:+.2f} ({fid_200_pct:+.1f}%)")
print(f"  CLIP: 基线={sdxl_200['clip_score_mean']:.4f} -> LoRA={lora_200['clip_score_mean']:.4f}")
print(f"         变化: {clip_200_improve:+.4f} ({clip_200_pct:+.1f}%)")
print()

# 样本数影响分析
print("【样本数影响】")
print(f"  FID基线: 100张={sdxl_100['fid']:.2f} -> 200张={sdxl_200['fid']:.2f} "
      f"(降低{sdxl_100['fid']-sdxl_200['fid']:.2f}, 样本量越大FID越准确)")
print(f"  FID LoRA: 100张={lora_100['fid']:.2f} -> 200张={lora_200['fid']:.2f} "
      f"(降低{lora_100['fid']-lora_200['fid']:.2f})")
print()

print("【关键结论】")
if fid_200_improve < 0:
    print("  ✅ LoRA微调有效！200张样本下FID降低，生成质量更接近真实图片")
else:
    print("  ⚠️ FID变化不明显，需要更多训练迭代")
    
if clip_200_improve > 0:
    print("  ✅ CLIP提升，图文对齐能力增强")
print()
print("=" * 100)
