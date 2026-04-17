import json
import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ==========================================
# 一级品类时间窗口 / 时间滞后分布脚本
#
# 目标：
# 1. 基于 session 内时序路径提取 category -> next_category 转移
# 2. 计算从 A 到 B 的时间滞后分布（秒 / 分钟）
# 3. 统计各品类对的 mean / median / p25 / p75 / p90 / std
# 4. 生成“推荐触达窗口”字段，供策略层使用
# 5. 导出 csv / 热力图 / 分布图
#
# 运行环境：
# - Colab 可直接运行
# - 本地存在 view_data.csv 时也可直接运行
# ==========================================


def configure_plot_style():
    plt.rcParams["font.sans-serif"] = [
        "Noto Sans CJK SC",
        "SimHei",
        "Microsoft YaHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"


def resolve_data_path(base_dir_path):
    env_path = os.getenv("DATA_PATH")
    if env_path:
        return env_path

    candidates = [
        "/content/drive/MyDrive/Colab Notebooks/view_data.csv",
        "/content/drive/MyDrive/view_data.csv",
        "/content/view_data.csv",
        os.path.join(base_dir_path, "view_data.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def format_minutes(value_minutes):
    if pd.isna(value_minutes):
        return "unknown"
    if value_minutes < 1:
        return f"{int(round(value_minutes * 60))}秒"
    if value_minutes < 60:
        return f"{value_minutes:.1f}分钟"
    return f"{value_minutes / 60:.1f}小时"


def build_window_label(p25_min, p75_min):
    if pd.isna(p25_min) or pd.isna(p75_min):
        return "unknown"
    return f"{format_minutes(p25_min)} - {format_minutes(p75_min)}"


def add_figure_text(fig, title, subtitle, footnote):
    fig.suptitle(title, x=0.02, y=0.98, ha="left", fontsize=18, fontweight="bold")
    fig.text(0.02, 0.94, subtitle, ha="left", va="top", fontsize=11, color="#444444")
    fig.text(0.02, 0.02, footnote, ha="left", va="bottom", fontsize=9, color="#555555")


def build_heatmap_from_pairs(pair_df, src_col, dst_col, value_col, weight_col, top_n_categories):
    src_order = pair_df.groupby(src_col)[weight_col].sum().sort_values(ascending=False)
    dst_order = pair_df.groupby(dst_col)[weight_col].sum().sort_values(ascending=False)
    top_src = src_order.head(top_n_categories).index
    top_dst = dst_order.head(top_n_categories).index
    heatmap_df = pair_df.pivot(index=src_col, columns=dst_col, values=value_col).reindex(
        index=top_src,
        columns=top_dst,
    )
    return heatmap_df


start_time = time.time()
configure_plot_style()
print("🚀 [Time Window] 开始计算一级品类时间窗口与时间滞后分布...")

base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = resolve_data_path(base_dir)
artifact_root = os.path.dirname(file_path) if file_path.startswith("/content/") else base_dir
output_dir = os.getenv("OUTPUT_DIR", os.path.join(artifact_root, "eda_time_window_outputs"))
os.makedirs(output_dir, exist_ok=True)

max_rows = int(os.getenv("MAX_ROWS", "0"))
min_pair_count = int(os.getenv("MIN_PAIR_COUNT", "20"))
plot_min_pair_count = int(os.getenv("PLOT_MIN_PAIR_COUNT", str(max(min_pair_count, 30))))
top_k = int(os.getenv("TOP_K", "50"))
collapse_consecutive = os.getenv("COLLAPSE_CONSECUTIVE", "1") == "1"
exclude_self_transition = os.getenv("EXCLUDE_SELF_TRANSITION", "1") == "1"
exclude_unknown_category = os.getenv("EXCLUDE_UNKNOWN_CATEGORY", "1") == "1"
unknown_category_name = os.getenv("UNKNOWN_CATEGORY_NAME", "未知品类")
top_n_categories = int(os.getenv("TOP_N_CATEGORIES", "16"))
plot_top_pairs = int(os.getenv("PLOT_TOP_PAIRS", "9"))
max_lag_hours = float(os.getenv("MAX_LAG_HOURS", "12"))
figure_dpi = int(os.getenv("FIGURE_DPI", "260"))
report_prefix = os.getenv("REPORT_PREFIX", "美团 Cross 分析")
data_date_label = os.getenv("DATA_DATE_LABEL", "2026-01-11")

print(f"📂 数据路径: {file_path}")
print(f"📁 输出目录: {output_dir}")


# ==========================================
# 1. 读取与清洗
# ==========================================
usecols = [
    "session_id",
    "event_timestamp",
    "first_cate_name",
    "event_type",
    "event_id",
]
read_kwargs = {"usecols": usecols}
if max_rows > 0:
    read_kwargs["nrows"] = max_rows

df = pd.read_csv(file_path, **read_kwargs)
raw_rows = len(df)
print(f"📄 原始行数: {raw_rows:,}")

df["event_timestamp"] = pd.to_numeric(df["event_timestamp"], errors="coerce")
df = df.dropna(subset=["session_id", "event_timestamp"])

df["first_cate_name"] = df["first_cate_name"].fillna(unknown_category_name)
df["event_type_str"] = df["event_type"].astype(str).str.strip().str.upper()
df["event_id_str"] = df["event_id"].astype(str).str.strip().str.lower()

is_pv = df["event_type_str"] == "PV"
is_mc = df["event_type_str"] == "MC"
is_order = (df["event_type_str"] == "ORDER") | (df["event_id_str"].str.contains("order", na=False))

df["unified_event"] = pd.Series(dtype="object")
df.loc[is_pv, "unified_event"] = "PV"
df.loc[is_mc, "unified_event"] = "MC"
df.loc[is_order, "unified_event"] = "ORDER"
df = df.dropna(subset=["unified_event"]).copy()
clean_rows = len(df)

print(f"🧹 清洗后行数: {clean_rows:,}")


# ==========================================
# 2. 构建 session 内品类转移与时间差
# ==========================================
df = df.sort_values(["session_id", "event_timestamp"]).copy()
sequence_df = df[["session_id", "event_timestamp", "first_cate_name"]].copy()

if collapse_consecutive:
    prev_cate = sequence_df.groupby("session_id")["first_cate_name"].shift(1)
    sequence_df = sequence_df[sequence_df["first_cate_name"] != prev_cate].copy()
    print(f"🧭 连续重复品类折叠后: {len(sequence_df):,}")

sequence_df["next_cate"] = sequence_df.groupby("session_id")["first_cate_name"].shift(-1)
sequence_df["next_ts"] = sequence_df.groupby("session_id")["event_timestamp"].shift(-1)
transitions = sequence_df.dropna(subset=["next_cate", "next_ts"]).copy()

if exclude_self_transition:
    transitions = transitions[transitions["first_cate_name"] != transitions["next_cate"]].copy()

if exclude_unknown_category:
    before_unknown_filter = len(transitions)
    transitions = transitions[
        (transitions["first_cate_name"] != unknown_category_name)
        & (transitions["next_cate"] != unknown_category_name)
    ].copy()
    print(
        f"🚫 剔除未知品类后转移数: {len(transitions):,}"
        f"（移除 {before_unknown_filter - len(transitions):,} 条）"
    )

transitions["lag_ms"] = transitions["next_ts"] - transitions["event_timestamp"]
transitions["lag_sec"] = transitions["lag_ms"] / 1000.0
transitions["lag_min"] = transitions["lag_sec"] / 60.0

max_lag_sec = max_lag_hours * 3600.0
transitions = transitions[(transitions["lag_sec"] >= 0) & (transitions["lag_sec"] <= max_lag_sec)].copy()

print(f"⏱️ 有效 category 转移数: {len(transitions):,}")

if transitions.empty:
    raise RuntimeError("没有提取到有效的时间滞后样本，无法计算时间窗口。")


# ==========================================
# 3. 统计时间窗口
# ==========================================
pair_stats = (
    transitions.groupby(["first_cate_name", "next_cate"])
    .agg(
        pair_count=("session_id", "size"),
        mean_lag_min=("lag_min", "mean"),
        median_lag_min=("lag_min", "median"),
        std_lag_min=("lag_min", "std"),
        p25_lag_min=("lag_min", lambda x: np.quantile(x, 0.25)),
        p75_lag_min=("lag_min", lambda x: np.quantile(x, 0.75)),
        p90_lag_min=("lag_min", lambda x: np.quantile(x, 0.90)),
    )
    .reset_index()
)

pair_stats["std_lag_min"] = pair_stats["std_lag_min"].fillna(0)
pair_stats["window_label"] = pair_stats.apply(
    lambda r: build_window_label(r["p25_lag_min"], r["p75_lag_min"]),
    axis=1,
)
pair_stats["window_center_min"] = (pair_stats["p25_lag_min"] + pair_stats["p75_lag_min"]) / 2.0

filtered_pair_stats = pair_stats[pair_stats["pair_count"] >= min_pair_count].copy()
filtered_pair_stats = filtered_pair_stats.sort_values(
    ["pair_count", "median_lag_min"],
    ascending=[False, True],
).reset_index(drop=True)

plot_pair_stats = filtered_pair_stats[filtered_pair_stats["pair_count"] >= plot_min_pair_count].copy()
if plot_pair_stats.empty:
    plot_pair_stats = filtered_pair_stats.copy()

median_matrix = pair_stats.pivot(
    index="first_cate_name",
    columns="next_cate",
    values="median_lag_min",
).fillna(0)
p25_matrix = pair_stats.pivot(
    index="first_cate_name",
    columns="next_cate",
    values="p25_lag_min",
).fillna(0)
p75_matrix = pair_stats.pivot(
    index="first_cate_name",
    columns="next_cate",
    values="p75_lag_min",
).fillna(0)


# ==========================================
# 4. 导出矩阵与榜单
# ==========================================
summary_path = os.path.join(output_dir, "category_transition_time_window_summary.csv")
median_matrix_path = os.path.join(output_dir, "category_transition_median_lag_matrix.csv")
p25_matrix_path = os.path.join(output_dir, "category_transition_p25_lag_matrix.csv")
p75_matrix_path = os.path.join(output_dir, "category_transition_p75_lag_matrix.csv")
top_pairs_path = os.path.join(output_dir, "category_transition_top_time_window_pairs.csv")
meta_path = os.path.join(output_dir, "time_window_metadata.json")

pair_stats.to_csv(summary_path, index=False, encoding="utf-8-sig")
median_matrix.to_csv(median_matrix_path, encoding="utf-8-sig")
p25_matrix.to_csv(p25_matrix_path, encoding="utf-8-sig")
p75_matrix.to_csv(p75_matrix_path, encoding="utf-8-sig")
filtered_pair_stats.head(top_k).to_csv(top_pairs_path, index=False, encoding="utf-8-sig")

metadata = {
    "data_path": file_path,
    "output_dir": output_dir,
    "raw_rows": int(raw_rows),
    "clean_rows": int(clean_rows),
    "transition_rows": int(len(transitions)),
    "category_count": int(len(median_matrix.index)),
    "min_pair_count": int(min_pair_count),
    "plot_min_pair_count": int(plot_min_pair_count),
    "top_k": int(top_k),
    "collapse_consecutive": bool(collapse_consecutive),
    "exclude_self_transition": bool(exclude_self_transition),
    "exclude_unknown_category": bool(exclude_unknown_category),
    "unknown_category_name": unknown_category_name,
    "max_lag_hours": float(max_lag_hours),
}
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)


# ==========================================
# 5. 中位数热力图
# ==========================================
heatmap_path = os.path.join(output_dir, "category_transition_median_lag_heatmap_topN.png")
title = f"{report_prefix} | 图3 一级品类相邻转移中位时间滞后热力图"
subtitle = "业务含义：展示从源品类 A 转移到目标品类 B 的中位等待时间，用于识别跨场景联动的最佳触达时机。"
footnote = (
    f"数据口径：{data_date_label} 单日明细日志；基于 session 内相邻一级品类转移。"
    f"{'已折叠连续重复品类；' if collapse_consecutive else ''}"
    f"{'已剔除未知品类；' if exclude_unknown_category else ''}"
    f"{'已剔除同类自环转移；' if exclude_self_transition else ''}"
    f"展示口径：仅保留 0~{max_lag_hours:.0f} 小时的有效时间差；"
    f"仅展示 pair_count≥{plot_min_pair_count} 且按总活跃度排序的 Top-{top_n_categories} 源/目标品类；"
    "颜色越深表示中位时间滞后越长，空白表示该品类对样本不足或未进入展示集合。"
)

if plot_pair_stats.empty:
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.axis("off")
    ax.text(0.5, 0.5, "过滤后暂无足够样本生成时间窗口热力图", ha="center", va="center", fontsize=15)
    add_figure_text(fig, title, subtitle, footnote)
    fig.subplots_adjust(top=0.75, bottom=0.20)
    fig.savefig(heatmap_path, dpi=figure_dpi, bbox_inches="tight")
    plt.close(fig)
else:
    heatmap_df = build_heatmap_from_pairs(
        plot_pair_stats,
        src_col="first_cate_name",
        dst_col="next_cate",
        value_col="median_lag_min",
        weight_col="pair_count",
        top_n_categories=top_n_categories,
    )

    finite_vals = heatmap_df.to_numpy(dtype=float)
    finite_vals = finite_vals[np.isfinite(finite_vals)]
    vmax = max(np.quantile(finite_vals, 0.95), 5.0) if finite_vals.size > 0 else 60.0

    fig, ax = plt.subplots(
        figsize=(max(12, 0.75 * len(heatmap_df.columns) + 6), max(8, 0.55 * len(heatmap_df.index) + 4.5))
    )
    cmap = plt.cm.get_cmap("Blues").copy()
    cmap.set_bad("#F2F2F2")

    masked_values = np.ma.masked_invalid(heatmap_df.to_numpy(dtype=float))
    im = ax.imshow(masked_values, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.02)
    cbar.set_label("Median Lag (minutes)", fontsize=11)

    ax.set_xticks(range(len(heatmap_df.columns)))
    ax.set_xticklabels(heatmap_df.columns, rotation=50, ha="right", fontsize=10)
    ax.set_yticks(range(len(heatmap_df.index)))
    ax.set_yticklabels(heatmap_df.index, fontsize=10)
    ax.set_xlabel("目标一级品类（Destination Category）", fontsize=12)
    ax.set_ylabel("起始一级品类（Source Category）", fontsize=12)

    if len(heatmap_df.index) <= 10 and len(heatmap_df.columns) <= 10:
        for i in range(len(heatmap_df.index)):
            for j in range(len(heatmap_df.columns)):
                value = heatmap_df.iat[i, j]
                if pd.notna(value):
                    text_color = "white" if value >= vmax * 0.60 else "#222222"
                    ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=8, color=text_color)

    add_figure_text(fig, title, subtitle, footnote)
    fig.subplots_adjust(left=0.22, right=0.95, top=0.80, bottom=0.24)
    fig.savefig(heatmap_path, dpi=figure_dpi, bbox_inches="tight")
    plt.close(fig)


# ==========================================
# 6. Top Pair 分布图
# ==========================================
plot_df = plot_pair_stats.head(plot_top_pairs).copy()
plot_pairs = list(zip(plot_df["first_cate_name"], plot_df["next_cate"]))
dist_plot_path = os.path.join(output_dir, "category_transition_lag_distribution_top_pairs.png")

if plot_pairs:
    n_pairs = len(plot_pairs)
    n_cols = 3
    n_rows = math.ceil(n_pairs / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4.6 * n_rows), squeeze=False)
    stats_lookup = filtered_pair_stats.set_index(["first_cate_name", "next_cate"])

    for idx, (src_cate, dst_cate) in enumerate(plot_pairs):
        ax = axes[idx // n_cols][idx % n_cols]
        pair_lags = transitions[
            (transitions["first_cate_name"] == src_cate) &
            (transitions["next_cate"] == dst_cate)
        ]["lag_min"].values

        stats_row = stats_lookup.loc[(src_cate, dst_cate)]
        median_lag = float(stats_row["median_lag_min"])
        p25_lag = float(stats_row["p25_lag_min"])
        p75_lag = float(stats_row["p75_lag_min"])
        pair_count = int(stats_row["pair_count"])

        ax.hist(pair_lags, bins=28, color="#4C78A8", alpha=0.80, edgecolor="white")
        ax.axvspan(p25_lag, p75_lag, color="#F58518", alpha=0.18, label="IQR")
        ax.axvline(median_lag, color="#E45756", linestyle="--", linewidth=1.8, label="Median")
        ax.set_title(
            f"{src_cate} → {dst_cate}\n"
            f"n={pair_count:,}, median={median_lag:.1f} min, IQR={p25_lag:.1f}-{p75_lag:.1f}",
            fontsize=11,
        )
        ax.set_xlabel("时间滞后（分钟）", fontsize=10)
        ax.set_ylabel("样本数", fontsize=10)
        ax.tick_params(labelsize=9)

    for idx in range(n_pairs, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    title = f"{report_prefix} | 图4 典型一级品类对的时间滞后分布"
    subtitle = "每个子图对应一个高支持度品类转移对；橙色阴影表示 IQR（25%~75%分位区间），红色虚线表示中位时间滞后。"
    footnote = (
        f"数据口径：{data_date_label} 单日明细日志；基于 session 内相邻一级品类转移。"
        f"展示口径：仅展示 pair_count≥{plot_min_pair_count} 的高支持度品类对中，按活跃度排序的 Top-{plot_top_pairs} 组合；"
        f"已限制时间差不超过 {max_lag_hours:.0f} 小时；"
        f"{'已剔除未知品类；' if exclude_unknown_category else ''}"
        "该图可在策略页中直接引用，用于说明不同 Cross 场景的建议触达时机。"
    )
    add_figure_text(fig, title, subtitle, footnote)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.84, bottom=0.12, hspace=0.42, wspace=0.24)
    fig.savefig(dist_plot_path, dpi=figure_dpi, bbox_inches="tight")
    plt.close(fig)


# ==========================================
# 7. 控制台摘要
# ==========================================
print("\n📊 Top Time Window Pairs:")
if filtered_pair_stats.empty:
    print("  当前在 min_pair_count 过滤下，没有满足条件的品类对。")
else:
    preview_cols = [
        "first_cate_name",
        "next_cate",
        "pair_count",
        "median_lag_min",
        "p25_lag_min",
        "p75_lag_min",
        "window_label",
    ]
    preview_df = filtered_pair_stats[preview_cols].head(min(10, len(filtered_pair_stats))).copy()
    print(preview_df.to_string(index=False))

elapsed = time.time() - start_time
print("\n✅ 时间窗口统计完成")
print(f"   summary: {summary_path}")
print(f"   median lag matrix: {median_matrix_path}")
print(f"   p25 lag matrix: {p25_matrix_path}")
print(f"   p75 lag matrix: {p75_matrix_path}")
print(f"   top pairs: {top_pairs_path}")
print(f"   heatmap: {heatmap_path}")
print(f"   distribution plot: {dist_plot_path}")
print(f"   耗时: {elapsed:.1f} 秒")
