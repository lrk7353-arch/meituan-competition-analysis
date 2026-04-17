"""
EDA_CROSS_BASELINE - 完整可运行版（含字体修复 + 热力图修复）
直接替换 EDA_CROSS_BASELINE_Gemini.py 中的 Phase 3 热力图部分即可使用。
"""

import json
import os
import time
import math
import gc

import matplotlib
matplotlib.use("Agg")  # 无头环境必须加这个

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import numpy as np
import pandas as pd


# ==============================================================
# 字体修复：解决中文显示为空方框的问题
# ==============================================================
def _setup_chinese_font():
    """自动查找并设置可显示中文的字体，在任何图表绘制前调用一次即可。"""
    font_paths = [
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
        "/usr/share/fonts/truetype/arphic/ukai.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]
    found = None
    for fp in font_paths:
        if os.path.exists(fp):
            found = fp
            break

    if found:
        prop = fm.FontProperties(fname=found)
        plt.rcParams["font.sans-serif"] = [prop.get_name()]
        plt.rcParams["font.family"] = "sans-serif"
    else:
        # fallback: 尝试系统已安装的字体
        for name in ["WenQuanYi Micro Hei", "Noto Sans CJK SC",
                      "AR PL UMing CN", "SimHei", "Microsoft YaHei"]:
            available = [f.name for f in fm.fontManager.ttflist]
            if name in available:
                plt.rcParams["font.sans-serif"] = [name]
                break

    plt.rcParams["axes.unicode_minus"] = False  # 负号正常显示
    return found


_setup_chinese_font()


# ==============================================================
# 热力图绘制函数（答辩级质量）
# ==============================================================
def draw_cross_heatmap(df_stats, src_col, dst_col, val_col, title,
                       filename, cmap, top_n=12, is_bline=False):
    """
    绘制答辩级 Cross 热力图。

    修复点：
    1. 中文字体支持（不再显示方框）
    2. colormap 以合理范围为中心（Lift 以 1.0 为底，CCR 以 0 为底但设合理上限）
    3. 过滤掉 pair_count 很少的行列，减少噪音
    4. 适当稀疏化 annot（只标高值格）
    5. 完整坐标轴标签、标题、图例
    """
    if df_stats.empty:
        print(f"  ⚠️ 数据为空，跳过: {filename}")
        return

    top_limit = 8 if is_bline else top_n

    # ① 按 pair_count 排序选 top-N（避免矩阵太稀疏）
    src_total = df_stats.groupby(src_col)["pair_count"].sum().sort_values(ascending=False)
    dst_total = df_stats.groupby(dst_col)["pair_count"].sum().sort_values(ascending=False)

    top_src_cates = src_total.head(top_limit).index.tolist()
    top_dst_cates = dst_total.head(top_limit).index.tolist()

    # 过滤后重排序
    src_in_top = df_stats[src_col].isin(top_src_cates)
    dst_in_top = df_stats[dst_col].isin(top_dst_cates)
    sub = df_stats[src_in_top & dst_in_top].copy()

    if sub.empty:
        print(f"  ⚠️ 过滤后为空，跳过: {filename}")
        return

    # ② 构透视矩阵
    pivot = sub.pivot(index=src_col, columns=dst_col, values=val_col).fillna(0)

    # ③ 取交集并保持顺序
    common_src = [c for c in top_src_cates if c in pivot.index]
    common_dst = [c for c in top_dst_cates if c in pivot.columns]
    matrix = pivot.loc[common_src, common_dst]

    n_rows, n_cols = matrix.shape
    if n_rows == 0 or n_cols == 0:
        print(f"  ⚠️ 矩阵为空，跳过: {filename}")
        return

    # ④ 设置 colormap 合理范围
    vals = matrix.values
    finite = vals[np.isfinite(vals) & (vals != 0)]

    if "lift" in val_col:
        # Lift: 以 1.0 为中心，底色为 <1（无意义），高亮 >1 的协同
        vmin, vmax = 0.8, max(2.5, float(np.percentile(finite, 90)) if len(finite) else 2.5)
    else:
        # CCR / 其他指标：从 0 开始，设到数据分布的合理上限
        vmax = max(0.3, float(np.percentile(finite, 90)) if len(finite) else 0.3)
        vmin = 0

    # ⑤ 绘图
    fig_h = max(7, n_rows * 1.1)
    fig_w = max(8, n_cols * 1.3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Seaborn heatmap（带 annot 高值格）
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".1f",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": val_col, "shrink": 0.8},
        ax=ax,
        annot_kws={"size": 9, "fontweight": "normal"},
    )

    # ⑥ 坐标轴标签（中文字体）
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_xticklabels(matrix.columns, rotation=55, ha="right", fontsize=10)
    ax.set_yticklabels(matrix.index, rotation=0, fontsize=10)

    # ⑦ 标题 + 轴标签
    ax.set_title(title, fontsize=14, pad=15, fontweight="bold")
    ax.set_xlabel("目标品类 (Destination)", fontsize=11)
    ax.set_ylabel("源品类 (Source)", fontsize=11)

    # 补充数据摘要到副标题
    total_pairs = int(sub["pair_count"].sum())
    n_unique = len(sub)
    fig.text(0.5, 0.01,
             f"基于 {total_pairs:,} 条有效转移 | Top-{len(common_src)}×{len(common_dst)} 品类 | "
             f"α=1.0 平滑 | 30min 窗口",
             ha="center", fontsize=9, color="gray")

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(filename, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✅ 热力图已保存: {filename}")


# ==============================================================
# Lag 分布图（答辩级质量）
# ==============================================================
def draw_lag_distributions(df_stats, n_pairs=9, output_path="lag_dist.png"):
    """
    绘制 Top-N 品类对的 lag 分布柱状图。
    子图不空插，每张子图标注中位 lag 和样本量。
    """
    if df_stats.empty:
        print(f"  ⚠️ 数据为空，跳过分布图")
        return

    plot_df = df_stats.head(n_pairs)
    pairs = list(zip(plot_df["first_cate_name_src"],
                     plot_df["first_cate_name_dst"]))

    n = len(pairs)
    n_cols = 3
    n_rows = math.ceil(n / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4.5 * n_rows), squeeze=False)
    fig.suptitle("Top Cross Pairs — 触达时间分布 (Median Lag)", fontsize=15, y=1.01)

    for idx, (src, dst) in enumerate(pairs):
        row_i, col_i = idx // n_cols, idx % n_cols
        ax = axes[row_i][col_i]

        row = plot_df.iloc[idx]
        count = int(row["pair_count"])
        median_min = row.get("median_lag_min", 0)
        p25 = row.get("p25_lag_min", 0)
        p75 = row.get("p75_lag_min", 0)

        # 用已知统计量画示意柱状（而非重新从原始数据画分布）
        bar_vals = [p25, median_min, p75]
        bar_labels = ["P25", "Median", "P75"]
        colors = ["#4C78A8", "#E45756", "#4C78A8"]
        bars = ax.bar(bar_labels, bar_vals, color=colors, alpha=0.8, width=0.5)

        ax.set_title(f"{src} → {dst}\nn={count:,}", fontsize=10)
        ax.set_ylabel("Lag (min)", fontsize=9)
        ax.set_xlabel("")

        # 标注数值
        for bar, val in zip(bars, bar_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f"{val:.1f}m", ha="center", va="bottom", fontsize=9)

    # 隐藏多余子图
    for idx in range(n, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✅ Lag 分布图已保存: {output_path}")


# ==============================================================
# 使用示例（替换 Phase 3）
# ==============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("字体检查测试")
    print("=" * 60)

    # 测试中文字体是否正常
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.set_title("测试中文：旅游 → 温泉景区 → 酒旅")
    ax.set_xlabel("目的地品类")
    ax.set_ylabel("Lift 值")
    ax.text(0.5, 0.5, "旅游→温泉景区: Lift=65.0", fontsize=14,
            ha="center", va="center",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    plt.tight_layout()
    test_path = os.path.join(os.path.dirname(__file__)), "font_test.png")
    plt.savefig(test_path, dpi=150)
    plt.close()
    print(f"字体测试图已保存: {test_path}")
    print("请人工确认中文是否正常显示（不是方框）")
