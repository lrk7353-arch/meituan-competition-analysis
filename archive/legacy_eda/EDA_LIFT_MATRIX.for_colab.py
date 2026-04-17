import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ==========================================
# 一级品类 Lift 矩阵脚本
#
# 目标：
# 1. 从单日 session 行为序列中提取 category -> next_category 转移
# 2. 计算一级品类转移的条件概率矩阵
# 3. 计算 Lift 矩阵：P(B | A) / P(B)
# 4. 导出 csv / top pairs / 热力图
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
print("🚀 [Lift] 开始计算一级品类 Lift 矩阵...")

base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = resolve_data_path(base_dir)
artifact_root = os.path.dirname(file_path) if file_path.startswith("/content/") else base_dir
output_dir = os.getenv("OUTPUT_DIR", os.path.join(artifact_root, "eda_lift_outputs"))
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
# 2. 构建一级品类序列
# ==========================================
df = df.sort_values(["session_id", "event_timestamp"]).copy()

sequence_df = df[["session_id", "event_timestamp", "first_cate_name"]].copy()

if collapse_consecutive:
    prev_cate = sequence_df.groupby("session_id")["first_cate_name"].shift(1)
    sequence_df = sequence_df[sequence_df["first_cate_name"] != prev_cate].copy()
    print(f"🧭 连续重复品类折叠后: {len(sequence_df):,}")

sequence_df["next_cate"] = sequence_df.groupby("session_id")["first_cate_name"].shift(-1)
transitions = sequence_df.dropna(subset=["next_cate"]).copy()

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

print(f"🔀 有效 category 转移数: {len(transitions):,}")

if transitions.empty:
    raise RuntimeError("没有提取到有效的一级品类转移，无法计算 Lift 矩阵。")

# ==========================================
# 3. 统计矩阵
# ==========================================
count_matrix = pd.crosstab(transitions["first_cate_name"], transitions["next_cate"]).astype(float)

row_sum = count_matrix.sum(axis=1)
col_sum = count_matrix.sum(axis=0)
total_transitions = count_matrix.values.sum()

conditional_prob = count_matrix.div(row_sum.replace(0, np.nan), axis=0)
marginal_prob = col_sum / total_transitions
lift_matrix = conditional_prob.div(marginal_prob.replace(0, np.nan), axis=1)
support_ratio = count_matrix / total_transitions

pair_df = (
    count_matrix.stack()
    .reset_index()
    .rename(columns={"first_cate_name": "src_cate", "next_cate": "dst_cate", 0: "pair_count"})
)
pair_df["conditional_prob"] = pair_df.apply(
    lambda r: conditional_prob.loc[r["src_cate"], r["dst_cate"]],
    axis=1,
)
pair_df["marginal_prob"] = pair_df.apply(
    lambda r: marginal_prob.loc[r["dst_cate"]],
    axis=1,
)
pair_df["lift"] = pair_df.apply(
    lambda r: lift_matrix.loc[r["src_cate"], r["dst_cate"]],
    axis=1,
)
pair_df["support_ratio"] = pair_df.apply(
    lambda r: support_ratio.loc[r["src_cate"], r["dst_cate"]],
    axis=1,
)

pair_df = pair_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["lift"]).copy()
filtered_pair_df = pair_df[pair_df["pair_count"] >= min_pair_count].copy()
filtered_pair_df = filtered_pair_df.sort_values(
    ["lift", "pair_count"],
    ascending=[False, False],
).reset_index(drop=True)

plot_pair_df = filtered_pair_df[filtered_pair_df["pair_count"] >= plot_min_pair_count].copy()
if plot_pair_df.empty:
    plot_pair_df = filtered_pair_df.copy()

# ==========================================
# 4. 导出矩阵与榜单
# ==========================================
count_matrix_path = os.path.join(output_dir, "category_transition_count_matrix.csv")
conditional_path = os.path.join(output_dir, "category_transition_conditional_prob_matrix.csv")
lift_path = os.path.join(output_dir, "category_transition_lift_matrix.csv")
top_pairs_path = os.path.join(output_dir, "category_transition_top_pairs.csv")
meta_path = os.path.join(output_dir, "lift_matrix_metadata.json")

count_matrix.to_csv(count_matrix_path, encoding="utf-8-sig")
conditional_prob.to_csv(conditional_path, encoding="utf-8-sig")
lift_matrix.to_csv(lift_path, encoding="utf-8-sig")
filtered_pair_df.head(top_k).to_csv(top_pairs_path, index=False, encoding="utf-8-sig")

metadata = {
    "data_path": file_path,
    "output_dir": output_dir,
    "raw_rows": int(raw_rows),
    "clean_rows": int(clean_rows),
    "transition_rows": int(len(transitions)),
    "category_count": int(len(count_matrix.index)),
    "min_pair_count": int(min_pair_count),
    "plot_min_pair_count": int(plot_min_pair_count),
    "top_k": int(top_k),
    "collapse_consecutive": bool(collapse_consecutive),
    "exclude_self_transition": bool(exclude_self_transition),
    "exclude_unknown_category": bool(exclude_unknown_category),
    "unknown_category_name": unknown_category_name,
}
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

# ==========================================
# 5. 热力图
# ==========================================
heatmap_path = os.path.join(output_dir, "category_transition_lift_heatmap_topN.png")
title = f"{report_prefix} | 图1 一级品类相邻转移 Lift 热力图"
subtitle = (
    "业务含义：Lift(A→B)=P(B|A)/P(B)，用于识别“从源品类 A 出发后，目标品类 B 被显著放大”的"
    "高协同场景。"
)
footnote = (
    f"数据口径：{data_date_label} 单日明细日志；基于 session 内按时间排序的相邻一级品类转移。"
    f"{'已折叠连续重复品类；' if collapse_consecutive else ''}"
    f"{'已剔除未知品类；' if exclude_unknown_category else ''}"
    f"{'已剔除同类自环转移；' if exclude_self_transition else ''}"
    f"展示口径：仅展示 pair_count≥{plot_min_pair_count} 且按总活跃度排序的 Top-{top_n_categories} 源/目标品类；"
    "空白表示该品类对样本不足或未进入展示集合；颜色上限按 95 分位裁剪。"
)

if plot_pair_df.empty:
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.axis("off")
    ax.text(0.5, 0.5, "过滤后暂无足够样本生成 Lift 热力图", ha="center", va="center", fontsize=15)
    add_figure_text(fig, title, subtitle, footnote)
    fig.subplots_adjust(top=0.75, bottom=0.20)
    fig.savefig(heatmap_path, dpi=figure_dpi, bbox_inches="tight")
    plt.close(fig)
else:
    heatmap_df = build_heatmap_from_pairs(
        plot_pair_df,
        src_col="src_cate",
        dst_col="dst_cate",
        value_col="lift",
        weight_col="pair_count",
        top_n_categories=top_n_categories,
    )

    finite_vals = heatmap_df.to_numpy(dtype=float)
    finite_vals = finite_vals[np.isfinite(finite_vals)]
    vmax = max(np.quantile(finite_vals, 0.95), 1.0) if finite_vals.size > 0 else 3.0

    fig, ax = plt.subplots(
        figsize=(max(12, 0.75 * len(heatmap_df.columns) + 6), max(8, 0.55 * len(heatmap_df.index) + 4.5))
    )
    cmap = plt.cm.get_cmap("YlOrRd").copy()
    cmap.set_bad("#F2F2F2")

    masked_values = np.ma.masked_invalid(heatmap_df.to_numpy(dtype=float))
    im = ax.imshow(masked_values, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.02)
    cbar.set_label("Lift", fontsize=11)

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
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8, color=text_color)

    add_figure_text(fig, title, subtitle, footnote)
    fig.subplots_adjust(left=0.22, right=0.95, top=0.80, bottom=0.24)
    fig.savefig(heatmap_path, dpi=figure_dpi, bbox_inches="tight")
    plt.close(fig)

# ==========================================
# 6. 控制台摘要
# ==========================================
print("\n📊 Top Lift Pairs:")
if filtered_pair_df.empty:
    print("  当前在 min_pair_count 过滤下，没有满足条件的品类对。")
else:
    preview_df = filtered_pair_df.head(min(10, len(filtered_pair_df))).copy()
    print(preview_df.to_string(index=False))

elapsed = time.time() - start_time
print("\n✅ Lift 矩阵计算完成")
print(f"   count matrix: {count_matrix_path}")
print(f"   conditional prob matrix: {conditional_path}")
print(f"   lift matrix: {lift_path}")
print(f"   top pairs: {top_pairs_path}")
print(f"   heatmap: {heatmap_path}")
print(f"   耗时: {elapsed:.1f} 秒")
