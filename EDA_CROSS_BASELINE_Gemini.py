import json
import os
import time
import math
import gc
import subprocess

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import numpy as np
import pandas as pd

# 在函数调用前定义 base_dir
base_dir = os.path.dirname(os.path.abspath(__file__))


# ==============================================================
# 字体修复：解决中文显示为空方框的问题
# ==============================================================
def _setup_chinese_font():
    import matplotlib.font_manager as fm
    import os

    # 直接从 Windows 字体目录调用 Noto Sans SC（开源中文字体）
    windows_font_paths = [
        "/mnt/c/Windows/Fonts/NotoSansSC-VF.ttf",
        "/mnt/c/Windows/Fonts/msyh.ttc",      # 微软雅黑
        "/mnt/c/Windows/Fonts/simhei.ttf",    # 黑体
    ]
    
    local_font_path = None
    for fp in windows_font_paths:
        if os.path.exists(fp):
            local_font_path = fp
            break
    
    if local_font_path:
        fm.fontManager.addfont(local_font_path)
        prop = fm.FontProperties(fname=local_font_path)
        plt.rcParams["font.family"] = prop.get_name()
        plt.rcParams["axes.unicode_minus"] = False
        print(f"  ✅ 已加载 Windows 中文字体: {local_font_path}")
    else:
        print("  ⚠️ 未找到 Windows 中文字体，热力图中文可能显示为方框")

_setup_chinese_font()

# ==========================================
# 跨业务 (Cross) 核心 EDA 基线层
# 
# 一次扫表，产出两套视角的交叉组合洞察：
# 1. 严格马尔可夫口径 (Direct Next)：寻找严密的顺水推舟行为。
# 2. 前向敞口探针 (Look-ahead Window)：评估真实商业潜力。新增精细 30 分钟。
# 
# 增强功能：
# - 补全 POI -> Category 映射，解决 ORDER 品类缺失问题
# - 拉普拉斯平滑（Bayesian Smoothing），消除长尾爆炸，alpha改回默认 1.0 先验
# - 添加了聚合到业务线(Business Line)的大颗粒度 Cross 分析，完美对标答辩需要
# - 高清晰 Seaborn Heatmap, 带 annot 数值标注
# ==========================================

SEED = 42
np.random.seed(SEED)

start_time = time.time()
print("🚀 [EDA Baseline] 启动统一 EDA 分析进程...")

base_dir = os.path.dirname(os.path.abspath(__file__))

def resolve_data_path(base_dir_path):
    env_path = os.getenv("DATA_PATH")
    if env_path: return env_path
    candidates = [
        "/content/drive/MyDrive/Colab Notebooks/view_data.csv",
        "/content/drive/MyDrive/view_data.csv",
        "/content/view_data.csv",
        os.path.join(base_dir_path, "view_data.csv"),
    ]
    for path in candidates:
        if os.path.exists(path): return path
    return candidates[0]

file_path = resolve_data_path(base_dir)
artifact_root = os.path.dirname(file_path) if file_path.startswith("/content/") else base_dir
output_dir = os.getenv("OUTPUT_DIR", os.path.join(artifact_root, "eda_baseline_outputs"))
os.makedirs(output_dir, exist_ok=True)

# 参数配置
max_rows = int(os.getenv("MAX_ROWS", "0"))
min_pair_count = int(os.getenv("MIN_PAIR_COUNT", "20"))
max_lag_hours = float(os.getenv("MAX_LAG_HOURS", "4.0"))   
fine_lag_minutes = int(os.getenv("LOOKAHEAD_WINDOW_MINUTES", "30")) # 精细化约束 30 min
alpha_smooth = float(os.getenv("ALPHA_SMOOTH", "1.0"))    # 主流建议 1.0 
top_k = int(os.getenv("TOP_K", "100"))
top_n_heatmap = 15

max_lag_ms = int(max_lag_hours * 3600 * 1000)
fine_lag_ms = int(fine_lag_minutes * 60 * 1000)

print(f"📂 数据路径: {file_path}")
print(f"📁 输出目录: {output_dir}")
print(f"⚙️ 核心参数: max_lag={max_lag_hours}h, fine_lag={fine_lag_minutes}m, alpha={alpha_smooth}, min_pair={min_pair_count}")

# ---------------------------------------------------------
# 美团大单轨业务线映射字典 (粗略映射，业务方可见)
# ---------------------------------------------------------
BUSINESS_LINE_MAP = {
    "外卖": "到家",
    "买菜": "到家",
    "超市/便利店": "到家",
    "闪购": "到家",
    "医药": "到家",
    "美食": "餐饮",
    "餐饮": "餐饮",
    "甜点饮品": "餐饮",
    "酒店": "酒旅",
    "旅游": "酒旅",
    "民宿": "酒旅",
    "火车票/机票": "出行",
    "打车": "出行",
    "共享单车": "出行",
    "休闲娱乐": "到店综合",
    "电影": "到店综合",
    "KTV": "到店综合",
    "丽人": "到店综合",
    "医美": "到店综合",
    "结婚": "到店综合",
    "亲子": "到店综合",
    "周边游": "酒旅"
}

def get_business_line(cate):
    for k, v in BUSINESS_LINE_MAP.items():
        if k in str(cate): return v
    return "其他业务"

# ---------------------------------------------------------
# 1. 一次性读取、清洗、截断长尾机器刷单 session
# ---------------------------------------------------------
usecols = ["session_id", "event_timestamp", "first_cate_name", "event_type", "event_id", "poi_id"]
read_kwargs = {"usecols": usecols}
if max_rows > 0: read_kwargs["nrows"] = max_rows

df = pd.read_csv(file_path, **read_kwargs)
df["event_timestamp"] = pd.to_numeric(df["event_timestamp"], errors="coerce")
df = df.dropna(subset=["session_id", "event_timestamp"])
df["first_cate_name"] = df["first_cate_name"].fillna("未知品类").astype(str)

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

# 【修复 1】: 通过 POI->Cate 映射补全 ORDER 类型的缺失品类
print("🛠 正在通过访问记录构建 POI->Category 映射，并修复部分 ORDER 缺失的品类...")
poi_cate_map = df[df["unified_event"].isin(["PV", "MC"])].dropna(subset=["poi_id", "first_cate_name"])
poi_cate_map = poi_cate_map[poi_cate_map["first_cate_name"] != "未知品类"]
# 快速字典映射
poi_cate_dict = poi_cate_map.groupby("poi_id")["first_cate_name"].first().to_dict()

# 寻找缺失品类的订单
order_mask = (df["unified_event"] == "ORDER") & ((df["first_cate_name"] == "未知品类") | (df["first_cate_name"] == ""))
missing_poi_ids = df.loc[order_mask, "poi_id"]
df.loc[order_mask, "first_cate_name"] = missing_poi_ids.map(poi_cate_dict).fillna("未知品类")

df["business_line"] = df["first_cate_name"].apply(get_business_line)

# 防止 OOM: 过滤极端长 Session (如爬虫 / 挂机)
sess_counts = df['session_id'].value_counts()
valid_sessions = sess_counts[sess_counts <= 200].index
df = df[df['session_id'].isin(valid_sessions)].copy()

# 追加唯一索引确保合并时 src/dst 前后时间顺序唯一
df = df.sort_values(["session_id", "event_timestamp"])
df["global_idx"] = np.arange(len(df))
print(f"🧹 清洗与过滤后有效行数: {len(df):,}")

num_categories = df["first_cate_name"].nunique()
num_blines = df["business_line"].nunique()

def bayesian_lift(count_ab, count_a, count_b, total_pairs, alpha, num_vocabs):
    # Laplace(Bayesian) Smoothing
    p_b_given_a = (count_ab + alpha) / (count_a + alpha * num_vocabs)
    p_b = (count_b + alpha) / (total_pairs + alpha * num_vocabs)
    return p_b_given_a / p_b

print("\n==========================================")
print("🚀 [Phase 1] 严格相邻口径 (Direct Next Markov)")
print("==========================================")

df_direct = df.copy()
df_direct["next_cate"] = df_direct.groupby("session_id")["first_cate_name"].shift(-1)
df_direct["next_ts"] = df_direct.groupby("session_id")["event_timestamp"].shift(-1)
df_direct["next_event"] = df_direct.groupby("session_id")["unified_event"].shift(-1)

direct_trans = df_direct.dropna(subset=["next_cate", "next_ts"]).copy()
direct_trans["lag_ms"] = direct_trans["next_ts"] - direct_trans["event_timestamp"]

direct_trans = direct_trans[
    (direct_trans["first_cate_name"] != direct_trans["next_cate"]) & 
    (direct_trans["lag_ms"] >= 0) & 
    (direct_trans["lag_ms"] <= max_lag_ms)
]
print(f"🔀 有效 Direct-Next 转移数: {len(direct_trans):,}")

if not direct_trans.empty:
    direct_trans["is_order"] = (direct_trans["next_event"] == "ORDER").astype(int)
    d_stats = direct_trans.groupby(["first_cate_name", "next_cate"]).agg(
        pair_count=("session_id", "count"),
        order_count=("is_order", "sum"),
        median_lag_ms=("lag_ms", "median")
    ).reset_index()
    
    total_d_pairs = d_stats["pair_count"].sum()
    src_d_counts = d_stats.groupby("first_cate_name")["pair_count"].sum().to_dict()
    dst_d_counts = d_stats.groupby("next_cate")["pair_count"].sum().to_dict()

    d_stats["ccr"] = d_stats["order_count"] / d_stats["pair_count"]
    d_stats["lift_smoothed"] = d_stats.apply(
        lambda r: bayesian_lift(
            r["pair_count"], src_d_counts.get(r["first_cate_name"], 0),
            dst_d_counts.get(r["next_cate"], 0), total_d_pairs, alpha_smooth, num_categories
        ), axis=1
    )
    d_stats["median_lag_min"] = d_stats["median_lag_ms"] / 60000.0
    
    d_stats = d_stats[d_stats["pair_count"] >= min_pair_count].sort_values("lift_smoothed", ascending=False)
    d_stats.to_csv(os.path.join(output_dir, "DirectNext_Markov_Top_Pairs.csv"), index=False, encoding="utf-8-sig")

del df_direct, direct_trans
gc.collect()

print("\n==========================================")
print("🚀 [Phase 2] 前向敞口探针 (Look-ahead Window Potential)")
print("==========================================")

# ========== 核心优化：防 OOM 的分块处理算法 ==========
slim_df = df[["session_id", "event_timestamp", "first_cate_name", "business_line", "global_idx", "unified_event"]]
unique_sessions = slim_df["session_id"].unique()
chunk_size = 20000
print(f"📦 启动分块安全自连接机制，共有 {len(unique_sessions):,} 个 Sessions，切分为 {math.ceil(len(unique_sessions)/chunk_size)} 块...")

# 用来收集各分块的去重结果的 list
l_unique_30m_list = []
b_unique_30m_list = []
l_unique_4H_list = []
b_unique_4H_list = []

for i in range(0, len(unique_sessions), chunk_size):
    chunk_sess = unique_sessions[i:i + chunk_size]
    chunk_df = slim_df[slim_df["session_id"].isin(chunk_sess)]
    
    # 块内 self-join
    merged_chunk = pd.merge(chunk_df, chunk_df, on="session_id", suffixes=('_src', '_dst'))
    
    # 共同的基础过滤：方向向前、去自环、不超最大上限(4H)
    mask = (
        (merged_chunk["global_idx_src"] < merged_chunk["global_idx_dst"]) & 
        (merged_chunk["first_cate_name_src"] != merged_chunk["first_cate_name_dst"]) & 
        ((merged_chunk["event_timestamp_dst"] - merged_chunk["event_timestamp_src"]) <= max_lag_ms) & 
        ((merged_chunk["event_timestamp_dst"] - merged_chunk["event_timestamp_src"]) >= 0)
    )
    valid_4H = merged_chunk[mask].copy()
    del merged_chunk
    
    valid_4H["lag_ms"] = valid_4H["event_timestamp_dst"] - valid_4H["event_timestamp_src"]
    valid_4H["is_order"] = (valid_4H["unified_event_dst"] == "ORDER").astype(int)
    
    # 分化出 30min 的子集
    valid_30m = valid_4H[valid_4H["lag_ms"] <= fine_lag_ms].copy()
    
    # ---------------- 局部聚合提炼函数 ----------------
    def extract_chunk_unique(v_df):
        if v_df.empty:
            l_u = pd.DataFrame(columns=["session_id", "first_cate_name_src", "first_cate_name_dst", "has_order", "first_lag_ms"])
            b_u = pd.DataFrame(columns=["session_id", "bline_src", "bline_dst", "has_order"])
            return l_u, b_u
        # 类别级别
        l_u = v_df.groupby(["session_id", "first_cate_name_src", "first_cate_name_dst"]).agg(
            has_order=("is_order", "max"), first_lag_ms=("lag_ms", "min")
        ).reset_index()
        # 业务线级别
        v_df["bline_src"] = v_df["business_line_src"]
        v_df["bline_dst"] = v_df["business_line_dst"]
        v_b = v_df[v_df["bline_src"] != v_df["bline_dst"]]
        b_u = v_b.groupby(["session_id", "bline_src", "bline_dst"]).agg(has_order=("is_order", "max")).reset_index() if not v_b.empty else pd.DataFrame(columns=["session_id", "bline_src", "bline_dst", "has_order"])
        return l_u, b_u
    
    l_u_4H, b_u_4H = extract_chunk_unique(valid_4H)
    l_u_30m, b_u_30m = extract_chunk_unique(valid_30m)
    
    l_unique_4H_list.append(l_u_4H)
    b_unique_4H_list.append(b_u_4H)
    l_unique_30m_list.append(l_u_30m)
    b_unique_30m_list.append(b_u_30m)
    
    del valid_4H, valid_30m, chunk_df
    gc.collect()

print("🌟 所有块提取完成，开始全局并表与重去重聚集...")

# 将所有块的 unique 结果合并。注意：因为每个 session id 都只在一个 chunk 出现，所以直接 concat 即可！不会发生跨 chunk 重叠。
l_unique_4H = pd.concat(l_unique_4H_list, ignore_index=True) if l_unique_4H_list else pd.DataFrame()
b_unique_4H = pd.concat(b_unique_4H_list, ignore_index=True) if b_unique_4H_list else pd.DataFrame()
l_unique_30m = pd.concat(l_unique_30m_list, ignore_index=True) if l_unique_30m_list else pd.DataFrame()
b_unique_30m = pd.concat(b_unique_30m_list, ignore_index=True) if b_unique_30m_list else pd.DataFrame()

del l_unique_4H_list, b_unique_4H_list, l_unique_30m_list, b_unique_30m_list
gc.collect()

# 定义汇总函数
def aggregate_total_stats(l_u, b_u, suffix_name):
    if l_u.empty or b_u.empty:
        return pd.DataFrame(), pd.DataFrame()
    # Category
    l_stats = l_u.groupby(["first_cate_name_src", "first_cate_name_dst"]).agg(
        pair_count=("session_id", "count"),
        order_count=("has_order", "sum"),
        median_lag_ms=("first_lag_ms", "median"),
    ).reset_index()
    tot_pairs = l_stats["pair_count"].sum()
    src_cnt = l_stats.groupby("first_cate_name_src")["pair_count"].sum().to_dict()
    dst_cnt = l_stats.groupby("first_cate_name_dst")["pair_count"].sum().to_dict()
    l_stats["ccr"] = l_stats["order_count"] / l_stats["pair_count"]
    l_stats["lift_smoothed"] = l_stats.apply(
        lambda r: bayesian_lift(r["pair_count"], src_cnt.get(r["first_cate_name_src"], 0),
            dst_cnt.get(r["first_cate_name_dst"], 0), tot_pairs, alpha_smooth, num_categories), axis=1
    )
    l_stats["median_lag_min"] = l_stats["median_lag_ms"] / 60000.0
    l_stats = l_stats[l_stats["pair_count"] >= min_pair_count].sort_values("lift_smoothed", ascending=False)
    l_stats.to_csv(os.path.join(output_dir, f"LookAhead_Category_{suffix_name}_Pairs.csv"), index=False, encoding="utf-8-sig")

    # Business Line
    b_stats = b_u.groupby(["bline_src", "bline_dst"]).agg(
        pair_count=("session_id", "count"), order_count=("has_order", "sum")
    ).reset_index()
    tot_b_pairs = b_stats["pair_count"].sum()
    b_src_cnt = b_stats.groupby("bline_src")["pair_count"].sum().to_dict()
    b_dst_cnt = b_stats.groupby("bline_dst")["pair_count"].sum().to_dict()
    b_stats["ccr"] = b_stats["order_count"] / b_stats["pair_count"]
    b_stats["lift_smoothed"] = b_stats.apply(
        lambda r: bayesian_lift(r["pair_count"], b_src_cnt.get(r["bline_src"], 0),
            b_dst_cnt.get(r["bline_dst"], 0), tot_b_pairs, alpha_smooth, num_blines), axis=1
    )
    b_stats = b_stats[b_stats["pair_count"] >= min_pair_count].sort_values("lift_smoothed", ascending=False)
    b_stats.to_csv(os.path.join(output_dir, f"LookAhead_BusinessLine_{suffix_name}_Pairs.csv"), index=False, encoding="utf-8-sig")

    return l_stats, b_stats

print(">>> 收束计算精细短逻辑 (30min) 与宽敞口 (4H)...")
stats_4H, bstats_4H = aggregate_total_stats(l_unique_4H, b_unique_4H, "4H")
stats_30m, bstats_30m = aggregate_total_stats(l_unique_30m, b_unique_30m, "30min")

print("\n==========================================")
print("🚀 [Phase 3] 同步绘制带数值的 高清 Heatmap (Seaborn)")
print("==========================================")

def draw_cross_heatmap(df_stats, src_col, dst_col, val_col, title, filename,
                        cmap, top_n=12, is_bline=False):
    """
    答辩级 Cross 热力图。

    修复点：
    1. 中文字体支持（不再显示方框）
    2. colormap 以合理范围为中心（Lift 以 1.0 为底，CCR 设合理上限）
    3. 过滤低频行列，减少噪音
    4. 完整坐标轴标签、标题、副标题元数据
    """
    if df_stats.empty:
        print(f"  ⚠️ 数据为空，跳过: {filename}")
        return

    top_limit = 8 if is_bline else top_n

    # ① 选取 top-N（按 pair_count 总和排序）
    src_total = df_stats.groupby(src_col)["pair_count"].sum().sort_values(ascending=False)
    dst_total = df_stats.groupby(dst_col)["pair_count"].sum().sort_values(ascending=False)
    top_src_cates = src_total.head(top_limit).index.tolist()
    top_dst_cates = dst_total.head(top_limit).index.tolist()

    # ② 过滤并构建透视矩阵
    src_mask = df_stats[src_col].isin(top_src_cates)
    dst_mask = df_stats[dst_col].isin(top_dst_cates)
    sub = df_stats[src_mask & dst_mask].copy()

    if sub.empty:
        print(f"  ⚠️ 过滤后为空，跳过: {filename}")
        return

    pivot = sub.pivot(index=src_col, columns=dst_col, values=val_col).fillna(0)
    common_src = [c for c in top_src_cates if c in pivot.index]
    common_dst = [c for c in top_dst_cates if c in pivot.columns]
    matrix = pivot.loc[common_src, common_dst]

    n_rows, n_cols = matrix.shape
    if n_rows == 0 or n_cols == 0:
        return

    # ③ 合理设置 colormap 范围
    vals = matrix.values
    finite = vals[np.isfinite(vals) & (vals != 0)]

    if "lift" in val_col:
        # Lift: 以 1.0 为底，高亮 >1 的正协同
        vmax = max(2.5, float(np.percentile(finite, 90)) if len(finite) else 2.5)
        vmin = 0.8
    else:
        # CCR / 其他: vmin=0，设到数据分布的合理上限
        vmax = max(0.3, (float(np.percentile(finite, 90)) if len(finite) else 0.3))
        vmin = 0

    # ④ 绘图
    fig_h = max(7, n_rows * 1.1)
    fig_w = max(8, n_cols * 1.3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

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
        annot_kws={"size": 9},
    )

    # ⑤ 坐标轴标签（中文字体）
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_xticklabels(matrix.columns, rotation=55, ha="right", fontsize=10)
    ax.set_yticklabels(matrix.index, rotation=0, fontsize=10)

    # ⑥ 标题 + 轴标签
    ax.set_title(title, fontsize=14, pad=15, fontweight="bold")
    ax.set_xlabel("目标品类 (Destination)", fontsize=11)
    ax.set_ylabel("源品类 (Source)", fontsize=11)

    # ⑦ 副标题：数据元数据
    total_pairs = int(sub["pair_count"].sum())
    fig.text(
        0.5, 0.01,
        f"基于 {total_pairs:,} 条有效转移 | Top-{n_rows}×{n_cols} 品类 | α=1.0 平滑",
        ha="center", fontsize=9, color="gray",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(filename, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✅ 热力图已保存: {filename}")


def draw_lag_distributions(df_stats, n_pairs=9, output_path="lag_dist.png"):
    """
    绘制 Top-N 品类对的 lag 分布柱状图。
    子图不空插，每张标注中位 lag 和样本量。
    """
    if df_stats.empty:
        print(f"  ⚠️ Lag 分布数据为空，跳过")
        return

    plot_df = df_stats.head(n_pairs)
    pairs = list(zip(plot_df["first_cate_name_src"],
                     plot_df["first_cate_name_dst"]))

    n = len(pairs)
    n_cols = 3
    n_rows = math.ceil(n / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4.5 * n_rows), squeeze=False)
    fig.suptitle("Top Cross Pairs — 触达时间分布 (触达窗口 P25 / Median / P75)",
                 fontsize=14, y=1.01)

    for idx, (src, dst) in enumerate(pairs):
        row_i, col_i = idx // n_cols, idx % n_cols
        ax = axes[row_i][col_i]

        row = plot_df.iloc[idx]
        count = int(row["pair_count"])
        p25 = row.get("p25_lag_min", 0) or 0
        median_min = row.get("median_lag_min", 0) or 0
        p75 = row.get("p75_lag_min", 0) or 0

        bar_labels = ["P25", "Median", "P75"]
        bar_vals = [p25, median_min, p75]
        colors = ["#6baed6", "#e6550d", "#6baed6"]

        bars = ax.bar(bar_labels, bar_vals, color=colors, alpha=0.85, width=0.5)
        ax.set_title(f"{src} → {dst}\nn={count:,}", fontsize=10)
        ax.set_ylabel("Lag (min)", fontsize=9)

        for bar, val in zip(bars, bar_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f"{val:.1f}m", ha="center", va="bottom", fontsize=9)

    for idx in range(n, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✅ Lag 分布图已保存: {output_path}")

# 绘制 30min 版本的 Category Heatmap
draw_cross_heatmap(
    stats_30m, "first_cate_name_src", "first_cate_name_dst", "lift_smoothed",
    "Category Smoothed Lift Heatmap (30-min Window)",
    os.path.join(output_dir, "Heatmap_Cate_Lift_30min.png"),
    "YlOrRd", top_n=top_n_heatmap
)
draw_cross_heatmap(
    stats_30m, "first_cate_name_src", "first_cate_name_dst", "ccr",
    "Category CCR Heatmap (30-min Window)",
    os.path.join(output_dir, "Heatmap_Cate_CCR_30min.png"),
    "YlGnBu", top_n=top_n_heatmap
)

# 绘制 4H 版本的 Business Line 大类 Cross Heatmap
draw_cross_heatmap(
    bstats_4H, "bline_src", "bline_dst", "lift_smoothed",
    "Business Line Smoothed Lift (4H Window)",
    os.path.join(output_dir, "Heatmap_BLine_Lift_4H.png"),
    "YlOrRd", is_bline=True
)
draw_cross_heatmap(
    bstats_4H, "bline_src", "bline_dst", "ccr",
    "Business Line CCR (4H Window)",
    os.path.join(output_dir, "Heatmap_BLine_CCR_4H.png"),
    "YlGnBu", is_bline=True
)

# Lag 分布图
if "median_lag_min" in stats_30m.columns:
    draw_lag_distributions(
        stats_30m, n_pairs=9,
        output_path=os.path.join(output_dir, "Heatmap_Lag_Distribution_30min.png")
    )

# 输出摘要
print("\n📊 First 5 rows of 30min Category Potential:")
if not stats_30m.empty:
    print(stats_30m[["first_cate_name_src", "first_cate_name_dst", "pair_count", "lift_smoothed", "ccr"]].head(5).to_string(index=False))

print("\n📊 First 5 rows of 4H Business Line Potential:")
if not bstats_4H.empty:
    print(bstats_4H[["bline_src", "bline_dst", "pair_count", "lift_smoothed", "ccr"]].head(5).to_string(index=False))

elapsed = time.time() - start_time
print(f"\n✅ 统一基线 EDA 结算完毕 | 耗时: {elapsed:.1f} 秒")
print(f"📁 生成产物已保存至 {output_dir}")

metadata = {
    "run_timestamp": time.time(),
    "max_lag_hours": max_lag_hours,
    "fine_lag_minutes": fine_lag_minutes,
    "alpha_smooth": alpha_smooth,
    "total_valid_rows": len(df)
}
with open(os.path.join(output_dir, "eda_baseline_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)
