import json
import math
import os
import time
from collections import Counter, defaultdict
from itertools import permutations
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import seaborn as sns


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEED = 42
np.random.seed(SEED)


def _setup_chinese_font() -> None:
    windows_font_paths = [
        "/mnt/c/Windows/Fonts/NotoSansSC-VF.ttf",
        "/mnt/c/Windows/Fonts/msyh.ttc",
        "/mnt/c/Windows/Fonts/simhei.ttf",
    ]
    for font_path in windows_font_paths:
        if os.path.exists(font_path):
            fm.fontManager.addfont(font_path)
            prop = fm.FontProperties(fname=font_path)
            plt.rcParams["font.family"] = prop.get_name()
            plt.rcParams["axes.unicode_minus"] = False
            return


_setup_chinese_font()


def resolve_data_path(base_dir: str) -> str:
    env_path = os.getenv("DATA_PATH")
    if env_path:
        return env_path

    candidates = [
        "/content/drive/MyDrive/Colab Notebooks/view_data.v1.1.csv",
        "/content/drive/MyDrive/view_data.v1.1.csv",
        "/content/view_data.v1.1.csv",
        os.path.join(base_dir, "view_data.v1.1.csv"),
        "/content/drive/MyDrive/Colab Notebooks/view_data.v1.csv",
        "/content/drive/MyDrive/view_data.v1.csv",
        "/content/view_data.v1.csv",
        os.path.join(base_dir, "view_data.v1.csv"),
        "/content/drive/MyDrive/Colab Notebooks/view_data.csv",
        "/content/drive/MyDrive/view_data.csv",
        "/content/view_data.csv",
        os.path.join(base_dir, "view_data.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[-1]


def load_schema(file_path: str) -> List[str]:
    return list(pd.read_csv(file_path, nrows=0).columns)


def resolve_cate_column(columns: Iterable[str]) -> str:
    for candidate in [
        "resolved_first_cate_name_v1_1",
        "resolved_first_cate_name_v1",
        "first_cate_name",
    ]:
        if candidate in columns:
            return candidate
    raise ValueError("未找到品类字段。")


def resolve_bline_column(columns: Iterable[str], cate_col: str) -> str:
    for candidate in [
        "resolved_business_line_v1_1",
        "resolved_business_line_v1",
    ]:
        if candidate in columns:
            return candidate
    return "__DERIVE_FROM_CATE__"


def get_business_line(cate: str) -> str:
    cate = "" if pd.isna(cate) else str(cate)
    mapping = {
        "外卖": "到家",
        "买菜": "到家",
        "超市": "到家",
        "便利": "到家",
        "闪购": "到家",
        "医药": "到家",
        "美食": "餐饮",
        "餐饮": "餐饮",
        "甜点": "餐饮",
        "饮品": "餐饮",
        "酒店": "酒旅",
        "旅游": "酒旅",
        "民宿": "酒旅",
        "火车票": "出行",
        "机票": "出行",
        "打车": "出行",
        "单车": "出行",
        "休闲娱乐": "到店综合",
        "电影": "到店综合",
        "KTV": "到店综合",
        "丽人": "到店综合",
        "医美": "到店综合",
        "结婚": "到店综合",
        "亲子": "到店综合",
        "周边游": "酒旅",
    }
    for key, value in mapping.items():
        if key in cate:
            return value
    return "其他业务"


def city_compatible(src_city: str, dst_city: str) -> bool:
    src_city = "未知城市" if pd.isna(src_city) else str(src_city)
    dst_city = "未知城市" if pd.isna(dst_city) else str(dst_city)
    if src_city == "未知城市" or dst_city == "未知城市":
        return True
    return src_city == dst_city


def prepare_data(file_path: str, max_rows: int = 0) -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, object]]:
    columns = load_schema(file_path)
    cate_col = resolve_cate_column(columns)
    bline_col = resolve_bline_column(columns, cate_col)

    usecols = [
        "row_key",
        "user_id",
        "session_id",
        "event_timestamp",
        "event_type",
        "page_city_name",
        cate_col,
    ]
    if bline_col != "__DERIVE_FROM_CATE__" and bline_col in columns:
        usecols.append(bline_col)

    print(f"📂 数据文件: {file_path}")
    print(f"🏷 品类字段: {cate_col}")
    print(f"🏷 业务线字段: {bline_col if bline_col != '__DERIVE_FROM_CATE__' else '由品类在线推导'}")

    read_kwargs = {"usecols": usecols}
    if max_rows > 0:
        read_kwargs["nrows"] = max_rows
    df = pd.read_csv(file_path, **read_kwargs)
    df["event_timestamp"] = pd.to_numeric(df["event_timestamp"], errors="coerce")
    df = df.dropna(subset=["user_id", "session_id", "event_timestamp", cate_col]).copy()
    df["event_timestamp"] = df["event_timestamp"].astype("int64")
    df["event_type"] = df["event_type"].astype(str).str.strip().str.upper()
    df["page_city_name"] = df["page_city_name"].fillna("未知城市").astype(str)
    df[cate_col] = df[cate_col].fillna("未知品类").astype(str)

    if bline_col == "__DERIVE_FROM_CATE__":
        df["business_line"] = df[cate_col].apply(get_business_line)
    else:
        df["business_line"] = df[bline_col].fillna("其他业务").astype(str)
    df["category"] = df[cate_col]

    string_cols = ["session_id", "page_city_name", "category", "business_line", "event_type"]
    for col in string_cols:
        df[col] = df[col].astype("category")

    metadata = {
        "cate_col": cate_col,
        "bline_col": "business_line",
        "file_path": file_path,
        "rows": int(len(df)),
        "orders": int((df["event_type"] == "ORDER").sum()),
        "explicit_rows": int((df["event_type"] != "ORDER").sum()),
        "users": int(df["user_id"].nunique()),
        "sessions": int(df["session_id"].nunique()),
    }
    return df, {"cate_col": "category", "bline_col": "business_line"}, metadata


def smoothed_lift(
    joint_count: int,
    src_count: int,
    dst_count: int,
    total_baskets: int,
    alpha: float,
) -> float:
    p_b_given_a = (joint_count + alpha) / (src_count + 2 * alpha)
    p_b = (dst_count + alpha) / (total_baskets + 2 * alpha)
    return float(p_b_given_a / p_b) if p_b > 0 else 0.0


def support_shrink(count: int, scale: float) -> float:
    return float(1.0 - math.exp(-count / scale))


def pair_outer_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on_cols: List[str],
    suffixes: Tuple[str, str] = ("_x", "_y"),
) -> pd.DataFrame:
    if left.empty:
        return right.copy()
    if right.empty:
        return left.copy()
    return left.merge(right, on=on_cols, how="outer", suffixes=suffixes)


def compute_markov(
    explicit_df: pd.DataFrame,
    value_col: str,
    max_gap_ms: int,
    min_pair_count: int,
    exclude_value: str,
) -> pd.DataFrame:
    work = explicit_df[["session_id", "user_id", "event_timestamp", "page_city_name", value_col]].copy()
    work = work.sort_values(["session_id", "event_timestamp"], kind="mergesort")
    work["next_value"] = work.groupby("session_id")[value_col].shift(-1)
    work["next_ts"] = work.groupby("session_id")["event_timestamp"].shift(-1)
    work["next_city"] = work.groupby("session_id")["page_city_name"].shift(-1)
    trans = work.dropna(subset=["next_value", "next_ts"]).copy()
    trans["lag_ms"] = trans["next_ts"] - trans["event_timestamp"]

    compatible_city = (
        (trans["page_city_name"].astype(str) == trans["next_city"].astype(str))
        | (trans["page_city_name"].astype(str) == "未知城市")
        | (trans["next_city"].astype(str) == "未知城市")
    )
    trans = trans[
        (trans[value_col].astype(str) != trans["next_value"].astype(str))
        & (trans[value_col].astype(str) != exclude_value)
        & (trans["next_value"].astype(str) != exclude_value)
        & (trans["lag_ms"] >= 0)
        & (trans["lag_ms"] <= max_gap_ms)
        & compatible_city
    ].copy()

    if trans.empty:
        return pd.DataFrame(columns=["src", "dst"])

    pair_stats = (
        trans.groupby([value_col, "next_value"])
        .agg(
            pair_count=("session_id", "count"),
            unique_users=("user_id", "nunique"),
            unique_sessions=("session_id", "nunique"),
            median_lag_ms=("lag_ms", "median"),
            p90_lag_ms=("lag_ms", lambda x: float(np.quantile(x, 0.90))),
        )
        .reset_index()
        .rename(columns={value_col: "src", "next_value": "dst"})
    )
    src_counts = pair_stats.groupby("src")["pair_count"].sum().rename("src_total")
    pair_stats = pair_stats.merge(src_counts, on="src", how="left")
    pair_stats["markov_prob"] = pair_stats["pair_count"] / pair_stats["src_total"]
    pair_stats["median_lag_min"] = pair_stats["median_lag_ms"] / 60000.0
    pair_stats["p90_lag_min"] = pair_stats["p90_lag_ms"] / 60000.0
    pair_stats["markov_score"] = pair_stats["markov_prob"] * np.log1p(pair_stats["pair_count"])
    pair_stats = pair_stats[pair_stats["pair_count"] >= min_pair_count].sort_values(
        ["markov_score", "markov_prob", "pair_count"], ascending=False
    )
    return pair_stats


def compute_lift(
    explicit_df: pd.DataFrame,
    value_col: str,
    min_pair_count: int,
    exclude_value: str,
    alpha: float,
    support_scale: float,
) -> pd.DataFrame:
    work = explicit_df[["user_id", "page_city_name", value_col]].copy()
    work = work[work[value_col].astype(str) != exclude_value].copy()
    work["basket_id"] = work["user_id"].astype(str) + "||" + work["page_city_name"].astype(str)

    basket_values = (
        work.groupby("basket_id")[value_col]
        .apply(lambda x: sorted(set(map(str, x.tolist()))))
        .reset_index(name="values")
    )

    total_baskets = len(basket_values)
    value_counter: Counter = Counter()
    pair_counter: Counter = Counter()

    for values in basket_values["values"].tolist():
        for value in values:
            value_counter[value] += 1
        if len(values) >= 2:
            for src, dst in permutations(values, 2):
                pair_counter[(src, dst)] += 1

    rows = []
    for (src, dst), joint_count in pair_counter.items():
        if joint_count < min_pair_count:
            continue
        src_count = value_counter[src]
        dst_count = value_counter[dst]
        lift_value = smoothed_lift(joint_count, src_count, dst_count, total_baskets, alpha)
        shrink = support_shrink(joint_count, support_scale)
        rows.append(
            {
                "src": src,
                "dst": dst,
                "basket_pair_count": int(joint_count),
                "src_basket_count": int(src_count),
                "dst_basket_count": int(dst_count),
                "lift_smoothed": float(lift_value),
                "lift_support_shrink": float(shrink),
                "lift_score": float(lift_value * shrink),
                "cooccurrence_rate": float(joint_count / total_baskets),
            }
        )
    lift_df = pd.DataFrame(rows)
    if not lift_df.empty:
        lift_df = lift_df.sort_values(["lift_score", "basket_pair_count"], ascending=False)
    return lift_df


def compute_ccr(
    events_df: pd.DataFrame,
    value_col: str,
    conversion_window_ms: int,
    exclude_value: str,
    min_conv_sessions: int,
) -> pd.DataFrame:
    work = events_df[["session_id", "event_timestamp", "event_type", "page_city_name", value_col]].copy()
    work = work.sort_values(["session_id", "event_timestamp"], kind="mergesort")

    exposure_counter: Counter = Counter()
    pair_counter: Counter = Counter()
    pair_gap_lists: Dict[Tuple[str, str], List[int]] = defaultdict(list)

    total_sessions = work["session_id"].nunique()
    grouped = work.groupby("session_id", sort=False)
    total_groups = total_sessions

    for idx, (_, grp) in enumerate(grouped, start=1):
        values = grp[value_col].astype(str).tolist()
        events = grp["event_type"].astype(str).tolist()
        timestamps = grp["event_timestamp"].astype("int64").tolist()
        cities = grp["page_city_name"].astype(str).tolist()

        src_seen_in_session = set()
        history: List[Tuple[int, str, str]] = []
        session_pair_min_gap: Dict[Tuple[str, str], int] = {}

        for ts, event_type, value, city in zip(timestamps, events, values, cities):
            if event_type != "ORDER":
                if value != exclude_value:
                    src_seen_in_session.add(value)
                    history.append((ts, value, city))
                continue

            dst = value
            if dst == exclude_value:
                continue

            best_src_gap: Dict[str, int] = {}
            for h_ts, h_value, h_city in reversed(history):
                gap = ts - h_ts
                if gap < 0:
                    continue
                if gap > conversion_window_ms:
                    break
                if h_value == exclude_value:
                    continue
                if not city_compatible(h_city, city):
                    continue
                prev_gap = best_src_gap.get(h_value)
                if prev_gap is None or gap < prev_gap:
                    best_src_gap[h_value] = gap

            for src, gap in best_src_gap.items():
                key = (src, dst)
                prev_gap = session_pair_min_gap.get(key)
                if prev_gap is None or gap < prev_gap:
                    session_pair_min_gap[key] = gap

        for src in src_seen_in_session:
            exposure_counter[src] += 1
        for key, gap in session_pair_min_gap.items():
            pair_counter[key] += 1
            pair_gap_lists[key].append(gap)

        if idx % 50000 == 0:
            print(f"  ⏳ CCR 会话扫描进度: {idx:,}/{total_groups:,}")

    rows = []
    for (src, dst), conv_sessions in pair_counter.items():
        if src == dst or conv_sessions < min_conv_sessions:
            continue
        src_sessions = exposure_counter[src]
        gap_values = np.array(pair_gap_lists[(src, dst)], dtype=np.float64)
        rows.append(
            {
                "src": src,
                "dst": dst,
                "conv_sessions": int(conv_sessions),
                "src_sessions": int(src_sessions),
                "ccr": float(conv_sessions / src_sessions) if src_sessions > 0 else 0.0,
                "median_conv_gap_min": float(np.median(gap_values) / 60000.0),
                "p90_conv_gap_min": float(np.quantile(gap_values, 0.90) / 60000.0),
                "ccr_score": float((conv_sessions / src_sessions) * np.log1p(conv_sessions)) if src_sessions > 0 else 0.0,
            }
        )

    ccr_df = pd.DataFrame(rows)
    if not ccr_df.empty:
        ccr_df = ccr_df.sort_values(["ccr_score", "ccr", "conv_sessions"], ascending=False)
    return ccr_df


def build_master_table(
    markov_df: pd.DataFrame,
    lift_df: pd.DataFrame,
    ccr_df: pd.DataFrame,
) -> pd.DataFrame:
    on_cols = ["src", "dst"]
    master = pair_outer_merge(markov_df, lift_df, on_cols=on_cols)
    master = pair_outer_merge(master, ccr_df, on_cols=on_cols)

    if master.empty:
        return master

    for col in [
        "pair_count",
        "unique_users",
        "unique_sessions",
        "src_total",
        "markov_prob",
        "markov_score",
        "basket_pair_count",
        "src_basket_count",
        "dst_basket_count",
        "lift_smoothed",
        "lift_support_shrink",
        "lift_score",
        "cooccurrence_rate",
        "conv_sessions",
        "src_sessions",
        "ccr",
        "ccr_score",
    ]:
        if col in master.columns:
            master[col] = pd.to_numeric(master[col], errors="coerce").fillna(0)

    def pct_rank(series: pd.Series) -> pd.Series:
        valid = series.fillna(0)
        if (valid > 0).sum() == 0:
            return pd.Series(np.zeros(len(valid)), index=valid.index)
        return valid.rank(pct=True)

    master["rank_markov"] = pct_rank(master.get("markov_prob", pd.Series(0, index=master.index)))
    master["rank_lift"] = pct_rank(master.get("lift_score", pd.Series(0, index=master.index)))
    master["rank_ccr"] = pct_rank(master.get("ccr", pd.Series(0, index=master.index)))
    master["cross_baseline_score"] = (
        0.40 * master["rank_lift"] + 0.40 * master["rank_markov"] + 0.20 * master["rank_ccr"]
    )
    master = master.sort_values("cross_baseline_score", ascending=False)
    return master


def save_heatmap(
    pair_df: pd.DataFrame,
    value_col: str,
    output_path: str,
    title: str,
    subtitle: str,
    top_n: int,
) -> None:
    if pair_df.empty or value_col not in pair_df.columns:
        return

    work = pair_df.copy()
    if "src_basket_count" in work.columns:
        top_src = work.groupby("src")["src_basket_count"].max().sort_values(ascending=False).head(top_n).index.tolist()
        top_dst = work.groupby("dst")["dst_basket_count"].max().sort_values(ascending=False).head(top_n).index.tolist()
    elif "src_total" in work.columns:
        top_src = work.groupby("src")["src_total"].max().sort_values(ascending=False).head(top_n).index.tolist()
        top_dst = work.groupby("dst")["pair_count"].sum().sort_values(ascending=False).head(top_n).index.tolist()
    elif "src_sessions" in work.columns:
        top_src = work.groupby("src")["src_sessions"].max().sort_values(ascending=False).head(top_n).index.tolist()
        top_dst = work.groupby("dst")["conv_sessions"].sum().sort_values(ascending=False).head(top_n).index.tolist()
    else:
        top_src = work["src"].value_counts().head(top_n).index.tolist()
        top_dst = work["dst"].value_counts().head(top_n).index.tolist()

    heatmap_df = work[work["src"].isin(top_src) & work["dst"].isin(top_dst)].pivot_table(
        index="src", columns="dst", values=value_col, aggfunc="mean", fill_value=0.0
    )
    if heatmap_df.empty:
        return

    plt.figure(figsize=(max(10, len(heatmap_df.columns) * 0.8), max(8, len(heatmap_df.index) * 0.6)))
    sns.heatmap(
        heatmap_df,
        cmap="YlOrRd",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        linecolor="#f0f0f0",
        cbar_kws={"label": value_col},
    )
    plt.title(title, fontsize=16, fontweight="bold", pad=22)
    plt.suptitle(subtitle, y=0.96, fontsize=10, color="#555555")
    plt.xlabel("目标场景")
    plt.ylabel("起点场景")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def save_summary_markdown(
    output_path: str,
    metadata: Dict[str, object],
    markov_bline: pd.DataFrame,
    lift_bline: pd.DataFrame,
    ccr_bline: pd.DataFrame,
    master_bline: pd.DataFrame,
) -> None:
    def to_md_table(frame: pd.DataFrame) -> str:
        frame = frame.copy()
        headers = [str(col) for col in frame.columns]
        lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        for row in frame.fillna("").values.tolist():
            lines.append("| " + " | ".join(str(v) for v in row) + " |")
        return "\n".join(lines)

    lines = [
        "# Cross Baseline V2 摘要",
        "",
        "## 数据与口径",
        "",
        f"- 输入文件：`{metadata['file_path']}`",
        f"- 总行数：`{metadata['rows']:,}`，显式行为：`{metadata['explicit_rows']:,}`，订单：`{metadata['orders']:,}`",
        f"- 用户数：`{metadata['users']:,}`，会话数：`{metadata['sessions']:,}`",
        "- 本版基线默认使用严格前序归因后的订单品类，并在统计层显式区分“场景行为”与“订单结果”。",
        "",
        "## 业务线 Top 结果",
        "",
        "### Markov Top 10",
        "",
        to_md_table(markov_bline.head(10)[["src", "dst", "pair_count", "markov_prob", "median_lag_min"]]),
        "",
        "### Lift Top 10",
        "",
        to_md_table(lift_bline.head(10)[["src", "dst", "basket_pair_count", "lift_smoothed", "lift_score"]]),
        "",
        "### CCR Top 10",
        "",
        to_md_table(ccr_bline.head(10)[["src", "dst", "conv_sessions", "ccr", "median_conv_gap_min"]]),
        "",
        "### 合并总表 Top 15",
        "",
        to_md_table(master_bline.head(15)[["src", "dst", "markov_prob", "lift_score", "ccr", "cross_baseline_score"]]),
        "",
        "## 说明",
        "",
        "- Markov：严格相邻的显式场景转移概率。",
        "- Lift：基于用户-城市篮子的宏观协同提升度，附带支持度折扣。",
        "- CCR：A 场景出现后，同 session 内是否最终形成 B 订单的归因转化率。",
    ]
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    start_time = time.time()
    data_path = resolve_data_path(BASE_DIR)
    artifact_root = os.path.dirname(data_path) if data_path.startswith("/content/") else BASE_DIR
    output_dir = os.getenv("OUTPUT_DIR", os.path.join(artifact_root, "cross_baseline_v2_outputs"))
    os.makedirs(output_dir, exist_ok=True)

    max_direct_gap_min = int(os.getenv("DIRECT_MAX_GAP_MIN", "120"))
    conversion_window_min = int(os.getenv("CCR_WINDOW_MIN", "240"))
    min_markov_pair_count = int(os.getenv("MIN_MARKOV_PAIR_COUNT", "30"))
    min_lift_pair_count = int(os.getenv("MIN_LIFT_PAIR_COUNT", "30"))
    min_ccr_conv_sessions = int(os.getenv("MIN_CCR_CONV_SESSIONS", "10"))
    lift_alpha = float(os.getenv("LIFT_ALPHA", "1.0"))
    lift_support_scale = float(os.getenv("LIFT_SUPPORT_SCALE", "30.0"))
    top_n_heatmap = int(os.getenv("TOP_N_HEATMAP", "12"))
    max_rows = int(os.getenv("MAX_ROWS", "0"))

    print("🚀 [Cross Baseline V2] 启动统一 Lift / Markov / CCR 基线分析...")
    print(f"📁 输出目录: {output_dir}")
    print(
        "⚙️ 参数: "
        f"direct_gap={max_direct_gap_min}min, "
        f"ccr_window={conversion_window_min}min, "
        f"min_markov={min_markov_pair_count}, "
        f"min_lift={min_lift_pair_count}, "
        f"min_ccr={min_ccr_conv_sessions}, "
        f"lift_alpha={lift_alpha}, "
        f"lift_support_scale={lift_support_scale}, "
        f"max_rows={max_rows}"
    )

    df, cols, metadata = prepare_data(data_path, max_rows=max_rows)

    df = df.sort_values(["session_id", "event_timestamp", "row_key"]).reset_index(drop=True)
    explicit_df = df[df["event_type"].astype(str) != "ORDER"].copy()

    category_exclude = "未知品类"
    bline_exclude = "其他业务"
    direct_gap_ms = max_direct_gap_min * 60 * 1000
    conversion_window_ms = conversion_window_min * 60 * 1000

    print("🔀 正在计算 Markov 基线...")
    markov_category = compute_markov(
        explicit_df=explicit_df,
        value_col=cols["cate_col"],
        max_gap_ms=direct_gap_ms,
        min_pair_count=min_markov_pair_count,
        exclude_value=category_exclude,
    )
    markov_bline = compute_markov(
        explicit_df=explicit_df,
        value_col=cols["bline_col"],
        max_gap_ms=direct_gap_ms,
        min_pair_count=min_markov_pair_count,
        exclude_value=bline_exclude,
    )

    print("📈 正在计算 Lift 基线...")
    lift_category = compute_lift(
        explicit_df=explicit_df,
        value_col=cols["cate_col"],
        min_pair_count=min_lift_pair_count,
        exclude_value=category_exclude,
        alpha=lift_alpha,
        support_scale=lift_support_scale,
    )
    lift_bline = compute_lift(
        explicit_df=explicit_df,
        value_col=cols["bline_col"],
        min_pair_count=min_lift_pair_count,
        exclude_value=bline_exclude,
        alpha=lift_alpha,
        support_scale=lift_support_scale,
    )

    print("💰 正在计算归因 CCR 基线...")
    ccr_category = compute_ccr(
        events_df=df,
        value_col=cols["cate_col"],
        conversion_window_ms=conversion_window_ms,
        exclude_value=category_exclude,
        min_conv_sessions=min_ccr_conv_sessions,
    )
    ccr_bline = compute_ccr(
        events_df=df,
        value_col=cols["bline_col"],
        conversion_window_ms=conversion_window_ms,
        exclude_value=bline_exclude,
        min_conv_sessions=min_ccr_conv_sessions,
    )

    print("🧩 正在合并三条基线结果...")
    master_category = build_master_table(markov_category, lift_category, ccr_category)
    master_bline = build_master_table(markov_bline, lift_bline, ccr_bline)

    print("💾 正在写出 CSV 结果...")
    outputs = {
        "markov_category_pairs.csv": markov_category,
        "markov_business_line_pairs.csv": markov_bline,
        "lift_category_pairs.csv": lift_category,
        "lift_business_line_pairs.csv": lift_bline,
        "ccr_category_pairs.csv": ccr_category,
        "ccr_business_line_pairs.csv": ccr_bline,
        "cross_pair_master_table_category.csv": master_category,
        "cross_pair_master_table_business_line.csv": master_bline,
    }
    for filename, frame in outputs.items():
        frame.to_csv(os.path.join(output_dir, filename), index=False, encoding="utf-8-sig")

    print("🖼 正在生成热力图...")
    save_heatmap(
        markov_bline,
        value_col="markov_prob",
        output_path=os.path.join(output_dir, "heatmap_markov_business_line.png"),
        title="美团 Cross 基线 | Markov 业务线转移概率热力图",
        subtitle="口径：严格相邻的显式场景跳转，过滤自循环与跨城已知跳转",
        top_n=top_n_heatmap,
    )
    save_heatmap(
        lift_bline,
        value_col="lift_score",
        output_path=os.path.join(output_dir, "heatmap_lift_business_line.png"),
        title="美团 Cross 基线 | Lift 业务线协同热力图",
        subtitle="口径：用户-城市篮子共现 + Bernoulli 平滑 Lift + 支持度折扣",
        top_n=top_n_heatmap,
    )
    save_heatmap(
        ccr_bline,
        value_col="ccr",
        output_path=os.path.join(output_dir, "heatmap_ccr_business_line.png"),
        title="美团 Cross 基线 | 归因 CCR 业务线热力图",
        subtitle="口径：A 场景出现后，同 session 内是否最终形成 B 订单",
        top_n=top_n_heatmap,
    )

    save_summary_markdown(
        output_path=os.path.join(output_dir, "baseline_summary.md"),
        metadata=metadata,
        markov_bline=markov_bline,
        lift_bline=lift_bline,
        ccr_bline=ccr_bline,
        master_bline=master_bline,
    )

    metadata["config"] = {
        "direct_gap_min": max_direct_gap_min,
        "ccr_window_min": conversion_window_min,
        "min_markov_pair_count": min_markov_pair_count,
        "min_lift_pair_count": min_lift_pair_count,
        "min_ccr_conv_sessions": min_ccr_conv_sessions,
        "lift_alpha": lift_alpha,
        "lift_support_scale": lift_support_scale,
        "top_n_heatmap": top_n_heatmap,
    }
    metadata["outputs"] = list(outputs.keys()) + [
        "heatmap_markov_business_line.png",
        "heatmap_lift_business_line.png",
        "heatmap_ccr_business_line.png",
        "baseline_summary.md",
    ]
    with open(os.path.join(output_dir, "baseline_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - start_time
    print("✅ Cross Baseline V2 完成。")
    print(
        f"📊 Markov(category)={len(markov_category):,} | Lift(category)={len(lift_category):,} | CCR(category)={len(ccr_category):,}"
    )
    print(
        f"📊 Markov(bline)={len(markov_bline):,} | Lift(bline)={len(lift_bline):,} | CCR(bline)={len(ccr_bline):,}"
    )
    print(f"🕒 总耗时: {elapsed / 60:.2f} 分钟")


if __name__ == "__main__":
    main()
