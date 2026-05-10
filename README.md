# SDXL LoRA 微调与CLIP质量评估项目

本项目用于实现 Stable Diffusion XL (SDXL) 的 LoRA 微调，以及配套的图像生成、质量评估（FID/CLIP Score）、结果可视化全流程。

## 📁 项目结构

```
CLIP/
├── src/                          # 源代码目录
│   ├── train_lora.py                # LoRA训练入口脚本
│   ├── train_text_to_image_lora_sdxl.py  # 主训练脚本
│   ├── generate_lora.py             # 生成图片脚本
│   ├── text2img_generate_sdxl.py    # SDXL生成脚本
│   ├── evaluate_lora.py             # LoRA评估入口
│   ├── text2img_evaluate.py         # FID/CLIP评估脚本
│   ├── prepare_lora_dataset.py      # 数据集准备脚本
│   ├── download_datasets.py         # 自动下载COCO数据集和预训练模型
│   ├── compare_100_vs_200.py        # 样本量对比分析
│   ├── plot_results.py              # 结果可视化图表
│   ├── text2img_complex_compare.py  # 复杂度对比实验
│   ├── visualize_results.py         # 可视化对比网格
│   └── logger_utils.py              # 日志工具
│
├── logs/                          # 训练和评估日志
│   ├── train_lora_sdxl_*.log        # 训练日志
│   ├── generate_*.log               # 生成日志
│   ├── evaluate*.log                # 评估日志
│   └── ...
│
├── 结果/                          # 历史实验结果
│   ├── figures/                      # 可视化图表
│   │   ├── example_comparison_grid.png  # 对比网格图
│   │   ├── clip_comparison_bar_kde.png  # CLIP分布图
│   │   └── ...
│   ├── logs/                        # 实验日志副本
│   ├── caption_image_mapping_sdxl.json  # 标注映射
│   ├── evaluation_results_sdxl.json    # 评估结果
│   └── complex_comparison_results_sdxl.json  # 对比实验结果
│
├── README.md                       # 项目说明文档
├── .gitignore                      # Git忽略配置
└── requirements.txt                # Python依赖清单
```

> ⚠️ **注意**：`data/`（COCO数据集）、`models/`（预训练模型）、`outputs/`（最新输出结果）目录因体积过大未上传至GitHub，可通过运行 `python src/download_datasets.py` 自动下载。

## ✨ 核心功能

- 🚀 **SDXL LoRA 微调**：基于COCO数据集快速训练轻量级LoRA权重
- 🎨 **批量图像生成**：支持自定义prompt、推理参数批量生成图像
- 📊 **质量量化评估**：
  - FID指标：评估生成图像与真实图像的分布相似度
  - CLIP Score：评估生成图像与文本prompt的语义匹配度
- 📈 **结果可视化**：自动生成对比网格图、指标分布曲线、样本量对比分析
- 🔬 **对比实验**：支持简单/复杂prompt效果对比、不同训练样本量效果对比

## 🛠️ 环境安装

### 依赖安装
```bash
pip install -r requirements.txt
```

### 可选优化
```bash
# 安装xformers提升显存效率
pip install xformers
```

## 🚀 使用指南

### 0. 自动下载数据集和预训练模型
```bash
python src/download_datasets.py
```
自动下载COCO2017数据集、标注文件和FID评估所需的预训练模型，支持断点续传和完整性校验。

### 1. 数据集准备
```bash
python src/prepare_lora_dataset.py
```
自动处理COCO数据集，生成训练所需的标注文件。

### 2. LoRA训练
```bash
python src/train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
  --dataset_name data/coco_lora_dataset \
  --output_dir models/sdxl_lora_coco \
  --rank 4 \
  --learning_rate 1e-4 \
  --num_train_epochs 10 \
  --train_batch_size 2
```

### 3. 批量生成图像
```bash
python src/text2img_generate_sdxl.py \
  --lora_path models/sdxl_lora_coco/pytorch_lora_weights.safetensors \
  --num_samples 100 \
  --output_dir outputs/images/results
```

### 4. 质量评估
```bash
python src/text2img_evaluate.py \
  --gen_dir outputs/images/results/generated_images \
  --real_dir data/real_images \
  --mapping_file outputs/results/caption_image_mapping.json
```
会自动计算并输出FID和CLIP Score指标。

### 5. 结果可视化
```bash
# 生成真实 vs 生成对比网格图
python src/visualize_results.py

# 生成指标分布图和对比图表
python src/plot_results.py
```

### 6. 对比实验
```bash
# 简单/复杂prompt效果对比
python src/text2img_complex_compare.py

# 100 vs 200样本量训练效果对比
python src/compare_100_vs_200.py
```

## 📊 核心指标说明

| 指标 | 取值范围 | 含义 |
|------|----------|------|
| **FID** | 0 ~ ∞ | 越低越好，衡量生成图像与真实图像的分布相似度，<br>FID < 10 表示生成质量非常优秀 |
| **CLIP Score** | 0 ~ 1 | 越高越好，衡量生成图像与文本prompt的语义匹配度，<br>通常0.3以上表示匹配度良好 |

## 📝 实验结果示例

### 样本量对比实验结果
| 配置 | FID | CLIP Score | 提升幅度 |
|------|-----|------------|----------|
| SDXL 基线 (100样本) | 28.4 | 0.3215 | - |
| LoRA 微调 (100样本) | 21.7 | 0.3582 | FID ↓23.6%, CLIP ↑11.4% |
| SDXL 基线 (200样本) | 27.9 | 0.3241 | - |
| LoRA 微调 (200样本) | 19.3 | 0.3726 | FID ↓30.8%, CLIP ↑15.0% |

## 📄 许可证

本项目仅用于学习和研究用途。
- SDXL 模型版权归 Stability AI 所有
- CLIP 模型版权归 OpenAI 所有
- COCO 数据集版权归 COCO 团队所有
