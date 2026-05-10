"""
结果可视化 - 生成 SCI 论文风格图表用于 PPT
- 简单描述 vs 复杂描述对比（柱状图 + 密度图）
- CLIP Score vs 描述长度趋势
- 按对象数量分组对比
- caption 长度分布直方图
- 示例图像对比网格
"""

import json
import textwrap
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib
import seaborn as sns
import numpy as np
from PIL import Image

matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['axes.labelsize'] = 16
matplotlib.rcParams['axes.titlesize'] = 18
matplotlib.rcParams['axes.titleweight'] = 'bold'
matplotlib.rcParams['axes.labelweight'] = 'bold'
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['legend.fontsize'] = 13
matplotlib.rcParams['legend.framealpha'] = 0.9
matplotlib.rcParams['legend.shadow'] = True
matplotlib.rcParams['legend.edgecolor'] = '0.8'
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['savefig.dpi'] = 600
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['savefig.pad_inches'] = 0.1
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams['axes.spines.right'] = False

sns.set_style("whitegrid")
sns.set_palette("viridis")

# 全局两色方案（viridis 两端，对比最大）
PALETTE2 = sns.color_palette("viridis", 2)
PALETTE3 = sns.color_palette("viridis", 3)

SCRIPT_DIR = Path(__file__).parent
COMPLEX_RESULT_FILE = SCRIPT_DIR / "complex_comparison_results_sdxl.json"
EVAL_RESULT_FILE = SCRIPT_DIR / "evaluation_results_sdxl.json"
SIMPLE_DIR = SCRIPT_DIR / "comparison_simple_sdxl"
COMPLEX_DIR = SCRIPT_DIR / "comparison_complex_sdxl"
OUTPUT_DIR = SCRIPT_DIR / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_complex_results():
    with open(COMPLEX_RESULT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    simple_scores = [p["clip_score"] for p in data["simple"]["pairs"]]
    complex_scores = [p["clip_score"] for p in data["complex"]["pairs"]]
    simple_lengths = [len(p["caption"]) for p in data["simple"]["pairs"]]
    complex_lengths = [len(p["caption"]) for p in data["complex"]["pairs"]]

    all_scores = simple_scores + complex_scores
    all_lengths = simple_lengths + complex_lengths
    all_groups = ["Simple"] * len(simple_scores) + ["Complex"] * len(complex_scores)

    return {
        "simple": {
            "mean": data["simple"]["clip_score_mean"],
            "std": data["simple"]["clip_score_std"],
            "scores": simple_scores,
            "lengths": simple_lengths,
            "pairs": data["simple"]["pairs"],
        },
        "complex": {
            "mean": data["complex"]["clip_score_mean"],
            "std": data["complex"]["clip_score_std"],
            "scores": complex_scores,
            "lengths": complex_lengths,
            "pairs": data["complex"]["pairs"],
        },
        "delta": data["delta"],
        "delta_pct": data["delta_pct"],
        "all_scores": all_scores,
        "all_lengths": all_lengths,
        "all_groups": all_groups,
    }


def load_eval_results():
    with open(EVAL_RESULT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def plot_bar_with_kde(results):
    """(a) 柱状图 + (b) KDE 分布：Simple vs Complex CLIP Score 对比"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    means = [results["simple"]["mean"], results["complex"]["mean"]]
    stds = [results["simple"]["std"], results["complex"]["std"]]
    labels = ["Simple\n(≤ 40 chars)", "Complex\n(≥ 80 chars)"]
    colors = PALETTE2

    # 柱状图
    bars = ax1.bar(
        labels, means, yerr=stds, capsize=9,
        color=colors, alpha=0.82, edgecolor="white", linewidth=1.5,
        error_kw=dict(elinewidth=1.8, ecolor="0.3", capthick=1.8)
    )

    # 数值标签放在误差条顶端之上
    for i, (v, s) in enumerate(zip(means, stds)):
        ax1.text(i, v + s + 0.006, f"{v:.4f}", ha="center", va="bottom",
                 fontsize=12, fontweight="bold")

    ax1.set_ylabel("CLIP Score")
    ax1.set_title("(a) Average CLIP Score Comparison")
    top1 = max(m + s for m, s in zip(means, stds))
    ax1.set_ylim(min(means) - 0.06, top1 + 0.04)

    delta_text = f"Δ = +{results['delta_pct']:.2f}%"
    ax1.annotate(
        delta_text,
        xy=(0.5, (means[1] + means[0]) / 2),
        xycoords=("axes fraction", "data"),
        xytext=(0.5, top1 + 0.022),
        textcoords=("axes fraction", "data"),
        ha="center", va="bottom", fontsize=13, fontweight="bold",
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF9C4", alpha=0.85, edgecolor="#CCBB44")
    )

    # KDE 分布图
    sns.kdeplot(
        x=results["simple"]["scores"], ax=ax2,
        color=colors[0], fill=True, alpha=0.55, label="Simple", linewidth=2.2
    )
    sns.kdeplot(
        x=results["complex"]["scores"], ax=ax2,
        color=colors[1], fill=True, alpha=0.55, label="Complex", linewidth=2.2
    )

    # 均值竖线
    ax2.axvline(results["simple"]["mean"], color=colors[0], linestyle="--",
                linewidth=1.5, alpha=0.8)
    ax2.axvline(results["complex"]["mean"], color=colors[1], linestyle="--",
                linewidth=1.5, alpha=0.8)

    ax2.set_xlabel("CLIP Score")
    ax2.set_ylabel("Density")
    ax2.set_title("(b) CLIP Score Distribution")
    ax2.legend(frameon=True)
    ax2.set_xlim(0.22, 0.42)

    fig.tight_layout(pad=1.5)
    out_path = OUTPUT_DIR / "clip_comparison_bar_kde.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_scatter_trend(results):
    """散点图：CLIP Score vs Caption Length，含分组颜色与趋势线"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    x = np.array(results["all_lengths"])
    y = np.array(results["all_scores"])
    groups = results["all_groups"]

    # 使用显式 RGB 颜色列表，确保图例颜色与数据点一致
    point_colors = [PALETTE2[0] if g == "Simple" else PALETTE2[1] for g in groups]

    ax.scatter(x, y, c=point_colors, alpha=0.75, s=65,
               edgecolor="white", linewidth=0.6)

    # 趋势线
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(x.min(), x.max(), 200)
    ax.plot(x_trend, p(x_trend), color="#E63946", linestyle="--",
            linewidth=2.2, zorder=5)

    ax.set_xlabel("Caption Length (characters)")
    ax.set_ylabel("CLIP Score")
    ax.set_title("CLIP Score vs Caption Length Trend")

    # 统一构建图例（只调用一次），含趋势线
    handles = [
        mlines.Line2D([], [], marker='o', color='w',
                      markerfacecolor=PALETTE2[0], markersize=10, label='Simple (≤ 40 chars)'),
        mlines.Line2D([], [], marker='o', color='w',
                      markerfacecolor=PALETTE2[1], markersize=10, label='Complex (≥ 80 chars)'),
        mlines.Line2D([], [], color='#E63946', linestyle='--',
                      linewidth=2, label=f'Trend  (slope = {z[0]:.5f})'),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=True)

    # 分组区域背景色（淡色）
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    ax.axvspan(x.min() - 2, 60, alpha=0.04, color=PALETTE2[0], zorder=0)
    ax.axvspan(60, x.max() + 2, alpha=0.04, color=PALETTE2[1], zorder=0)

    fig.tight_layout(pad=1.5)
    out_path = OUTPUT_DIR / "clip_vs_length_trend.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_object_count_bucket(eval_data):
    """柱状图：按对象数量分组的 CLIP Score"""
    bucket_stats = eval_data["bucket_stats"]
    labels = ["1 object", "2 objects", "3+ objects"]
    means = [bucket_stats["1 object"]["mean"],
             bucket_stats["2 objects"]["mean"],
             bucket_stats["3+ objects"]["mean"]]
    stds = [bucket_stats["1 object"]["std"],
            bucket_stats["2 objects"]["std"],
            bucket_stats["3+ objects"]["std"]]
    ns = [bucket_stats["1 object"]["n"],
          bucket_stats["2 objects"]["n"],
          bucket_stats["3+ objects"]["n"]]
    colors = PALETTE3

    fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))
    ax.bar(
        labels, means, yerr=stds, capsize=9,
        color=colors, alpha=0.82, edgecolor="white", linewidth=1.5,
        error_kw=dict(elinewidth=1.8, ecolor="0.3", capthick=1.8)
    )

    # 数值标签放在误差条顶端之上，避免重叠
    for i, (v, s) in enumerate(zip(means, stds)):
        ax.text(i, v + s + 0.005, f"{v:.4f}", ha="center", va="bottom",
                fontsize=12, fontweight="bold")
        # 样本量注释
        ax.text(i, means[i] * 0.5 + min(means) * 0.5 - 0.002,
                f"n = {ns[i]}", ha="center", va="center",
                fontsize=10, color="white", fontweight="bold")

    ax.set_ylabel("CLIP Score")
    ax.set_xlabel("Number of Objects in Caption")
    ax.set_title("CLIP Score by Number of Objects")

    # 动态 ylim，确保误差条完整显示
    top = max(m + s for m, s in zip(means, stds))
    bottom = min(means) - 0.03
    ax.set_ylim(bottom, top + 0.025)

    fig.tight_layout(pad=1.5)
    out_path = OUTPUT_DIR / "clip_by_object_count.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_length_distribution(results):
    """直方图：Simple 与 Complex 的 caption 长度分布"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5.5))

    simple_len = results["simple"]["lengths"]
    complex_len = results["complex"]["lengths"]

    # 使用 PALETTE2（两端对比色）解决颜色区分度不足问题
    col_simple = PALETTE2[0]
    col_complex = PALETTE2[1]

    bins = np.linspace(0, 160, 17)
    sns.histplot(simple_len, bins=bins, kde=True, color=col_simple,
                 alpha=0.65, label="Simple (≤ 40)", edgecolor="white",
                 linewidth=1, ax=ax)
    sns.histplot(complex_len, bins=bins, kde=True, color=col_complex,
                 alpha=0.65, label="Complex (≥ 80)", edgecolor="white",
                 linewidth=1, ax=ax)

    # 阈值竖线
    ax.axvline(40, color="#E63946", linestyle="--", linewidth=2,
               alpha=0.85, label="Boundary (40 / 80 chars)")
    ax.axvline(80, color="#E63946", linestyle="--", linewidth=2, alpha=0.85)

    ax.set_xlabel("Caption Length (characters)")
    ax.set_ylabel("Count")
    ax.set_title("Caption Length Distribution")
    ax.legend(frameon=True)

    fig.tight_layout(pad=1.5)
    out_path = OUTPUT_DIR / "caption_length_distribution.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_example_comparison(results, n_examples=4):
    """图像对比网格：Simple vs Complex 最高分示例"""
    simple_pairs = sorted(results["simple"]["pairs"], key=lambda x: -x["clip_score"])[:n_examples]
    complex_pairs = sorted(results["complex"]["pairs"], key=lambda x: -x["clip_score"])[:n_examples]

    fig, axes = plt.subplots(n_examples, 2, figsize=(13, 3.6 * n_examples))
    fig.patch.set_facecolor("#F8F8F8")

    col_headers = ["Simple Caption", "Complex Caption"]
    col_colors = [PALETTE2[0], PALETTE2[1]]

    # 列标题色框（在第一行上方）
    for col_idx, (hdr, col) in enumerate(zip(col_headers, col_colors)):
        axes[0, col_idx].set_title(
            hdr,
            fontsize=15, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=col, alpha=0.9, edgecolor="none"),
            pad=8
        )

    for i, (s_pair, c_pair) in enumerate(zip(simple_pairs, complex_pairs)):
        s_img_path = SIMPLE_DIR / s_pair["gen_file"]
        c_img_path = COMPLEX_DIR / c_pair["gen_file"]

        for ax, img_path, pair, col in zip(
            [axes[i, 0], axes[i, 1]],
            [s_img_path, c_img_path],
            [s_pair, c_pair],
            col_colors
        ):
            if Path(img_path).exists():
                img = Image.open(img_path).convert("RGB")
                ax.imshow(img)
            ax.axis("off")

            # 折行显示 caption（最多 40 字符/行，最多 3 行）
            cap_wrapped = textwrap.fill(pair["caption"], width=42)
            cap_lines = cap_wrapped.split("\n")
            if len(cap_lines) > 3:
                cap_wrapped = "\n".join(cap_lines[:3]) + "…"

            meta = f"Len: {len(pair['caption'])}    CLIP: {pair['clip_score']:.4f}"
            ax.set_title(
                f"{meta}\n{cap_wrapped}",
                fontsize=9.5, ha="center", linespacing=1.4,
                color="#222222"
            )

            # 彩色边框标识类别
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(col)
                spine.set_linewidth(3)

    fig.tight_layout(pad=1.2, h_pad=1.5, w_pad=1.0)
    out_path = OUTPUT_DIR / "example_comparison_grid.png"
    plt.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {out_path}")


def main():
    print("Loading results...")
    results = load_complex_results()
    eval_data = load_eval_results()

    print("Generating plots...")
    plot_bar_with_kde(results)
    plot_scatter_trend(results)
    plot_object_count_bucket(eval_data)
    plot_length_distribution(results)
    plot_example_comparison(results)

    print("\n=== Summary ===")
    print(f"Output directory: {OUTPUT_DIR}")
    print("\nGenerated files:")
    for f in OUTPUT_DIR.glob("*.png"):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
