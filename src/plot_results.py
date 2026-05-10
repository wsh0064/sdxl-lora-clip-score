"""
结果可视化 - 生成 SCI 论文风格图表用于 PPT
"""

import json
import os
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import pandas as pd
from PIL import Image

from logger_utils import setup_logger

logger = setup_logger("plot_results")

matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['axes.labelsize'] = 16
matplotlib.rcParams['axes.titlesize'] = 18
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['legend.fontsize'] = 14
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['savefig.dpi'] = 600
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['savefig.pad_inches'] = 0.05

sns.set_style("whitegrid")
palette = {
    'SDXL-100': '#440154',
    'LoRA-100': '#3b528b',
    'SDXL-200': '#21918c',
    'LoRA-200': '#5ec962',
}
sns.set_palette("viridis")

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

DATA_CONFIGS = [
    {
        'name': 'SDXL-100',
        'model': 'SDXL',
        'samples': '100',
        'eval_file': PROJECT_ROOT / "outputs" / "results" / "evaluation_results_sdxl.json",
        'mapping_file': PROJECT_ROOT / "outputs" / "results" / "caption_image_mapping_sdxl.json",
        'gen_dir': PROJECT_ROOT / "outputs" / "images" / "results_100_samples" / "generated_images_sdxl",
    },
    {
        'name': 'LoRA-100',
        'model': 'LoRA',
        'samples': '100',
        'eval_file': PROJECT_ROOT / "outputs" / "results" / "evaluation_results_lora.json",
        'mapping_file': PROJECT_ROOT / "outputs" / "results" / "caption_image_mapping_lora.json",
        'gen_dir': PROJECT_ROOT / "outputs" / "images" / "results_100_samples" / "generated_images_lora",
    },
    {
        'name': 'SDXL-200',
        'model': 'SDXL',
        'samples': '200',
        'eval_file': PROJECT_ROOT / "outputs" / "results" / "evaluation_results_sdxl_200.json",
        'mapping_file': PROJECT_ROOT / "outputs" / "results" / "caption_image_mapping_sdxl_200.json",
        'gen_dir': PROJECT_ROOT / "outputs" / "images" / "sdxl_200",
    },
    {
        'name': 'LoRA-200',
        'model': 'LoRA',
        'samples': '200',
        'eval_file': PROJECT_ROOT / "outputs" / "results" / "evaluation_results_lora_200.json",
        'mapping_file': PROJECT_ROOT / "outputs" / "results" / "caption_image_mapping_lora_200.json",
        'gen_dir': PROJECT_ROOT / "outputs" / "images" / "lora_200",
    },
]

REAL_IMAGES_DIR = PROJECT_ROOT / "data" / "real_images"


def load_all_evaluation_results():
    """加载所有评估结果，返回统一的数据结构"""
    results = {}
    
    for config in DATA_CONFIGS:
        if config['eval_file'].exists():
            with open(config['eval_file'], "r", encoding="utf-8") as f:
                data = json.load(f)
            
            results[config['name']] = {
                **config,
                'fid': data.get('fid', 0),
                'clip_mean': data.get('clip_score_mean', 0),
                'clip_std': data.get('clip_score_std', 0),
                'num_samples': data.get('num_samples', 0),
                'bucket_stats': data.get('bucket_stats', {}),
                'pairs': data.get('pairs', []),
                'clip_scores': [p['clip_score'] for p in data.get('pairs', [])],
                'caption_lengths': [len(p['caption']) for p in data.get('pairs', [])],
            }
            logger.info(f"加载成功: {config['name']} - FID: {results[config['name']]['fid']:.2f}, CLIP: {results[config['name']]['clip_mean']:.4f}")
        else:
            logger.warning(f"文件不存在: {config['eval_file']}")
    
    return results


def create_summary_dataframe(results):
    """创建汇总DataFrame用于绘图"""
    rows = []
    for name, data in results.items():
        rows.append({
            'Model': data['model'],
            'Samples': data['samples'],
            'Name': name,
            'FID': data['fid'],
            'CLIP': data['clip_mean'],
            'CLIP_std': data['clip_std'],
        })
    return pd.DataFrame(rows)


def plot_fid_comparison(results):
    """FID对比柱状图：样本量×模型"""
    df = create_summary_dataframe(results)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    x = np.arange(2)
    width = 0.35
    
    sdxl_data = df[df['Model'] == 'SDXL']
    lora_data = df[df['Model'] == 'LoRA']
    
    sdxl_fids = [sdxl_data[sdxl_data['Samples'] == '100']['FID'].values[0],
                 sdxl_data[sdxl_data['Samples'] == '200']['FID'].values[0]]
    lora_fids = [lora_data[lora_data['Samples'] == '100']['FID'].values[0],
                 lora_data[lora_data['Samples'] == '200']['FID'].values[0]]
    
    colors = sns.color_palette("viridis", 2)
    bars1 = ax.bar(x - width/2, sdxl_fids, width, label='SDXL (Base)', 
                   color=colors[0], alpha=0.8, edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, lora_fids, width, label='LoRA (Fine-tuned)', 
                   color=colors[1], alpha=0.8, edgecolor='black', linewidth=1.2)
    
    ax.set_ylabel('FID Score (lower is better)')
    ax.set_title('FID Comparison: Model vs Sample Size')
    ax.set_xticks(x)
    ax.set_xticklabels(['100 Samples', '200 Samples'])
    ax.legend(frameon=True)
    
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}', ha='center', va='bottom', fontsize=11)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}', ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    out_path = OUTPUT_DIR / "fid_comparison_bar.png"
    plt.savefig(out_path)
    plt.close()
    logger.info(f"已保存: {out_path}")


def plot_clip_boxplot(results):
    """CLIP Score箱线图对比"""
    data_list = []
    for name, data in results.items():
        for score in data['clip_scores']:
            data_list.append({'Model': name, 'CLIP Score': score})
    
    df = pd.DataFrame(data_list)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    colors = [palette[name] for name in results.keys()]
    sns.boxplot(x='Model', y='CLIP Score', data=df, palette=colors, ax=ax,
                linewidth=1.5, flierprops={'marker': 'o', 'markersize': 4})
    
    means = [np.mean(data['clip_scores']) for data in results.values()]
    for i, mean in enumerate(means):
        ax.text(i, mean + 0.001, f'{mean:.4f}', ha='center', va='bottom', 
                fontsize=11, fontweight='bold', color='white')
    
    ax.set_title('CLIP Score Distribution Comparison')
    ax.set_ylabel('CLIP Score (higher is better)')
    ax.set_xlabel('Model Configuration')
    
    plt.tight_layout()
    out_path = OUTPUT_DIR / "clip_score_boxplot.png"
    plt.savefig(out_path)
    plt.close()
    logger.info(f"已保存: {out_path}")


def plot_model_object_comparison(results):
    """按对象数量分组的模型对比 - 修复数值显示"""
    object_categories = ['1 object', '2 objects', '3+ objects']
    model_names = ['SDXL-100', 'LoRA-100', 'SDXL-200', 'LoRA-200']
    
    x = np.arange(len(object_categories))
    width = 0.2
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    
    colors = [palette[name] for name in model_names]
    
    for i, (model_name, color) in enumerate(zip(model_names, colors)):
        means = []
        stds = []
        for cat in object_categories:
            bucket = results[model_name]['bucket_stats'].get(cat, {})
            means.append(bucket.get('mean', 0))
            stds.append(bucket.get('std', 0))
        
        bars = ax.bar(x + i * width - (width * 1.5), means, width, yerr=stds, 
                      label=model_name, color=color, alpha=0.85, capsize=5,
                      edgecolor='black', linewidth=1, error_kw={'elinewidth': 1.2})
        
        for bar, mean_val, std_val in zip(bars, means, stds):
            if mean_val > 0.001:
                label_y = mean_val + std_val + 0.003
                ax.text(bar.get_x() + bar.get_width()/2, label_y,
                       f'{mean_val:.3f}', ha='center', va='bottom', fontsize=9,
                       fontweight='bold', color='black')
    
    ax.set_xlabel('Number of Objects in Caption', fontsize=13)
    ax.set_ylabel('CLIP Score', fontsize=13)
    ax.set_title('CLIP Score vs. Object Complexity', fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(object_categories, fontsize=12)
    ax.legend(frameon=True, ncol=4, fontsize=10, bbox_to_anchor=(0.5, 1.02), loc='upper center')
    ax.set_ylim(0.26, 0.355)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    out_path = OUTPUT_DIR / "model_comparison_by_objects.png"
    plt.savefig(out_path)
    plt.close()
    logger.info(f"已保存: {out_path}")


def plot_sample_size_impact(results):
    """样本量影响分析"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    models = ['SDXL', 'LoRA']
    sample_sizes = ['100', '200']
    colors = sns.color_palette("viridis", 2)
    
    for i, model in enumerate(models):
        fid_100 = results[f"{model}-100"]['fid']
        fid_200 = results[f"{model}-200"]['fid']
        clip_100 = results[f"{model}-100"]['clip_mean']
        clip_200 = results[f"{model}-200"]['clip_mean']
        
        fid_improvement = (fid_100 - fid_200) / fid_100 * 100
        clip_improvement = (clip_200 - clip_100) / clip_100 * 100
        
        ax1.plot(sample_sizes, [fid_100, fid_200], 'o-', label=f'{model}', 
                 color=colors[i], linewidth=2, markersize=8)
        ax2.plot(sample_sizes, [clip_100, clip_200], 'o-', label=f'{model}', 
                 color=colors[i], linewidth=2, markersize=8)
        
        ax1.annotate(f'{fid_improvement:+.1f}%', 
                    (1, fid_200), textcoords="offset points",
                    xytext=(10, 10), ha='center', fontsize=10)
        ax2.annotate(f'{clip_improvement:+.2f}%', 
                    (1, clip_200), textcoords="offset points",
                    xytext=(10, 10), ha='center', fontsize=10)
    
    ax1.set_xlabel('Sample Size')
    ax1.set_ylabel('FID Score')
    ax1.set_title('FID vs Sample Size')
    ax1.legend(frameon=True)
    
    ax2.set_xlabel('Sample Size')
    ax2.set_ylabel('CLIP Score')
    ax2.set_title('CLIP Score vs Sample Size')
    ax2.legend(frameon=True)
    
    plt.tight_layout()
    out_path = OUTPUT_DIR / "sample_size_impact.png"
    plt.savefig(out_path)
    plt.close()
    logger.info(f"已保存: {out_path}")


def plot_lora_improvement(results):
    """LoRA改进幅度分析"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    sample_sizes = ['100 Samples', '200 Samples']
    fid_improvements = []
    clip_improvements = []
    
    for samples in ['100', '200']:
        sdxl_fid = results[f"SDXL-{samples}"]['fid']
        lora_fid = results[f"LoRA-{samples}"]['fid']
        sdxl_clip = results[f"SDXL-{samples}"]['clip_mean']
        lora_clip = results[f"LoRA-{samples}"]['clip_mean']
        
        fid_improvement = (sdxl_fid - lora_fid) / sdxl_fid * 100
        clip_improvement = (lora_clip - sdxl_clip) / sdxl_clip * 100
        
        fid_improvements.append(fid_improvement)
        clip_improvements.append(clip_improvement)
    
    colors = sns.color_palette("viridis", 2)
    
    bars1 = ax1.bar(sample_sizes, fid_improvements, color=colors, alpha=0.8,
                    edgecolor='black', linewidth=1.2)
    ax1.set_ylabel('FID Improvement (%)')
    ax1.set_title('FID Improvement with LoRA')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:+.2f}%', ha='center', fontsize=11)
    
    bars2 = ax2.bar(sample_sizes, clip_improvements, color=colors, alpha=0.8,
                    edgecolor='black', linewidth=1.2)
    ax2.set_ylabel('CLIP Score Improvement (%)')
    ax2.set_title('CLIP Score Improvement with LoRA')
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:+.2f}%', ha='center', fontsize=11)
    
    plt.tight_layout()
    out_path = OUTPUT_DIR / "lora_improvement.png"
    plt.savefig(out_path)
    plt.close()
    logger.info(f"已保存: {out_path}")


def plot_performance_heatmap(results):
    """综合性能热力图 - 修复归一化"""
    models = ['SDXL-100', 'LoRA-100', 'SDXL-200', 'LoRA-200']
    metrics = ['FID (Lower=Better)', 'CLIP Score (Higher=Better)']
    
    fid_values = [results[m]['fid'] for m in models]
    clip_values = [results[m]['clip_mean'] for m in models]
    
    fid_min, fid_max = min(fid_values), max(fid_values)
    clip_min, clip_max = min(clip_values), max(clip_values)
    
    fid_norm = 1 - (np.array(fid_values) - fid_min) / (fid_max - fid_min) * 0.8 + 0.1
    clip_norm = (np.array(clip_values) - clip_min) / (clip_max - clip_min) * 0.8 + 0.1
    
    heatmap_data = np.vstack([fid_norm, clip_norm]).T
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn',
                xticklabels=metrics, yticklabels=models, ax=ax,
                cbar_kws={'label': 'Performance Score'}, vmin=0, vmax=1)
    ax.set_title('Performance Heatmap (Green = Better)')
    
    plt.tight_layout()
    out_path = OUTPUT_DIR / "performance_heatmap.png"
    plt.savefig(out_path)
    plt.close()
    logger.info(f"已保存: {out_path}")


def plot_multi_model_kde(results):
    """多模型CLIP分布KDE对比"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    for name, data in results.items():
        sns.kdeplot(data=data['clip_scores'], ax=ax, label=name,
                    color=palette[name], fill=True, alpha=0.3, linewidth=2)
    
    ax.set_xlabel('CLIP Score')
    ax.set_ylabel('Density')
    ax.set_title('CLIP Score Distribution Across All Models')
    ax.legend(frameon=True)
    
    plt.tight_layout()
    out_path = OUTPUT_DIR / "multi_model_comparison.png"
    plt.savefig(out_path)
    plt.close()
    logger.info(f"已保存: {out_path}")


def plot_real_vs_gen_grid(results, n_examples=3):
    """真实图像 vs 生成图像对比网格"""
    ref_model = 'SDXL-200'
    if ref_model not in results:
        ref_model = list(results.keys())[0]
    
    pairs = sorted(results[ref_model]['pairs'], key=lambda x: -x['clip_score'])[:n_examples]
    
    n_cols = 3
    fig, axes = plt.subplots(n_examples, n_cols, figsize=(12, 4 * n_examples))
    
    for i, pair in enumerate(pairs):
        real_idx = int(pair['gen_file'].replace('gen_', '').replace('.png', ''))
        real_path = REAL_IMAGES_DIR / f"real_{real_idx:04d}.png"
        
        if real_path.exists():
            real_img = Image.open(real_path).convert("RGB")
            axes[i, 0].imshow(real_img)
        axes[i, 0].axis('off')
        axes[i, 0].set_title(f'Real Image\n{pair["caption"][:50]}...', fontsize=9)
        
        for j, model_name in enumerate(['SDXL-200', 'LoRA-200'], 1):
            if model_name in results:
                gen_path = results[model_name]['gen_dir'] / pair['gen_file']
                if gen_path.exists():
                    gen_img = Image.open(gen_path).convert("RGB")
                    axes[i, j].imshow(gen_img)
            axes[i, j].axis('off')
            clip_val = pair['clip_score']
            axes[i, j].set_title(f'{model_name}\nCLIP: {clip_val:.4f}', fontsize=9)
    
    plt.tight_layout()
    out_path = OUTPUT_DIR / "real_vs_gen_grid.png"
    plt.savefig(out_path)
    plt.close()
    logger.info(f"已保存: {out_path}")


def plot_summary_table(results):
    """性能总结表格"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    for name in ['SDXL-100', 'LoRA-100', 'SDXL-200', 'LoRA-200']:
        if name in results:
            data = results[name]
            table_data.append([
                name,
                f"{data['fid']:.2f}",
                f"{data['clip_mean']:.4f}",
                f"{data['clip_std']:.4f}",
                f"{data['num_samples']}"
            ])
    
    table = ax.table(cellText=table_data,
                     colLabels=['Model', 'FID', 'CLIP Mean', 'CLIP Std', 'Samples'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.8)
    
    ax.set_title('Performance Summary', fontweight='bold', pad=20)
    
    plt.tight_layout()
    out_path = OUTPUT_DIR / "performance_summary_table.png"
    plt.savefig(out_path)
    plt.close()
    logger.info(f"已保存: {out_path}")


def plot_dual_metric_scatter(results):
    """FID + CLIP双指标散点图 - 替代雷达图更直观"""
    models = ['SDXL-100', 'LoRA-100', 'SDXL-200', 'LoRA-200']
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    colors = [palette[name] for name in models]
    markers = ['D', 's', 'o', '^']
    
    for i, (model_name, color, marker) in enumerate(zip(models, colors, markers)):
        data = results[model_name]
        fid_val = data['fid']
        clip_val = data['clip_mean']
        
        ax.scatter(fid_val, clip_val, color=color, s=300, marker=marker,
                   edgecolor='black', linewidth=2, alpha=0.9, zorder=5)
        
        ax.annotate(model_name, (fid_val, clip_val),
                   xytext=(0, 25), textcoords='offset points',
                   ha='center', fontsize=10, fontweight='bold', color=color)
        ax.annotate(f'FID={fid_val:.1f}\nCLIP={clip_val:.4f}', (fid_val, clip_val),
                   xytext=(0, -30), textcoords='offset points',
                   ha='center', fontsize=8, color='darkslategray')
    
    ax.set_xlabel('FID Score (lower = better image quality)', fontsize=13)
    ax.set_ylabel('CLIP Score (higher = better text-image alignment)', fontsize=13)
    ax.set_title('Model Performance Comparison\n(FID vs. CLIP)', fontsize=16, pad=20)
    
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3, linestyle='--')
    
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=palette[m], label=m, edgecolor='black') for m in models]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10, 
             title='Models', title_fontsize=11, bbox_to_anchor=(1.02, 1))
    
    arrow_props = dict(arrowstyle="->", color='darkgreen', linewidth=2, alpha=0.7)
    ax.annotate('BETTER\n← ↓', xy=(152, 0.3142), xytext=(147, 0.3146),
               arrowprops=arrow_props, fontsize=11, color='darkgreen', fontweight='bold',
               ha='center')
    
    plt.subplots_adjust(right=0.8)
    out_path = OUTPUT_DIR / "performance_radar.png"
    plt.savefig(out_path)
    plt.close()
    logger.info(f"已保存: {out_path}")


def main():
    logger.info("=" * 60)
    logger.info("开始生成完整的结果可视化图表")
    logger.info("=" * 60)
    logger.info("支持: 100样本 vs 200样本, SDXL vs LoRA")
    logger.info(f"输出目录: {OUTPUT_DIR}")
    logger.info("=" * 60)

    try:
        logger.info("\n[1/4] 加载所有评估结果...")
        results = load_all_evaluation_results()
        
        if len(results) < 2:
            logger.error("数据不足，无法生成对比图表！")
            return
        
        logger.info("\n[2/4] 生成基础对比图表...")
        plot_fid_comparison(results)
        plot_clip_boxplot(results)
        plot_multi_model_kde(results)
        
        logger.info("\n[3/4] 生成分析图表...")
        plot_model_object_comparison(results)
        plot_sample_size_impact(results)
        plot_lora_improvement(results)
        plot_performance_heatmap(results)
        plot_dual_metric_scatter(results)
        
        logger.info("\n[4/4] 生成展示图表...")
        plot_summary_table(results)
        plot_real_vs_gen_grid(results, n_examples=4)
        
        logger.info("\n" + "=" * 60)
        logger.info("所有图表生成完成！")
        logger.info(f"输出目录: {OUTPUT_DIR}")
        logger.info("生成的文件:")
        for f in sorted(OUTPUT_DIR.glob("*")):
            logger.info(f"  ✓ {f.name}")
        logger.info(f"总计: {len(list(OUTPUT_DIR.glob('*')))} 个文件")
        logger.info("=" * 60)

    except FileNotFoundError as e:
        logger.error("=" * 60)
        logger.error(f"数据文件未找到: {str(e)}")
        logger.error(traceback.format_exc())
        logger.error("=" * 60)
    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"发生错误: {str(e)}")
        logger.error(traceback.format_exc())
        logger.error("=" * 60)


if __name__ == "__main__":
    main()
