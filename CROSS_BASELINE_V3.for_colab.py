import json
import math
import os
import time
from collections import Counter, defaultdict
from itertools import permutations
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEED = 42
np.random.seed(SEED)

GENERIC_CATEGORY_KEYWORDS = (
    "其他",
    "未知",
    "综合",
    "全部",
    "通用",
)

MANDATORY_BUSINESS_PAIRS = [
    ("到家", "餐饮"),
    ("餐饮", "到家"),
    ("到店综合", "餐饮"),
    ("餐饮", "到店综合"),
    ("到店综合", "酒旅"),
    ("酒旅", "到店综合"),
    ("酒旅", "餐饮"),
    ("餐饮", "酒旅"),
]


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


def resolve_bline_column(columns: Iterable[str]) -> str:
    for candidate in [
        "resolved_business_line_v1_1",
        "resolved_business_line_v1",
        "business_line",
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
        "生鲜": "到家",
        "美食": "餐饮",
        "餐饮": "餐饮",
        "甜点": "餐饮",
        "饮品": "餐饮",
        "咖啡": "餐饮",
        "酒店": "酒旅",
        "旅游": "酒旅",
        "民宿": "酒旅",
        "景区": "酒旅",
        "门票": "酒旅",
        "周边游": "酒旅",
        "休闲娱乐": "到店综合",
        "电影": "到店综合",
        "KTV": "到店综合",
        "丽人": "到店综合",
        "医美": "到店综合",
        "结婚": "到店综合",
        "亲子": "到店综合",
        "运动": "到店综合",
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


def is_generic_category(value: str) -> bool:
    text = "" if pd.isna(value) else str(value).strip()
    if not text:
        return True
    return any(keyword in text for keyword in GENERIC_CATEGORY_KEYWORDS)


def prepare_data(file_path: str, max_rows: int = 0) -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, object]]:
    columns = load_schema(file_path)
    cate_col = resolve_cate_column(columns)
    bline_col = resolve_bline_column(columns)

    usecols = [
        "user_id",
        "session_id",
        "event_timestamp",
        "event_type",
        "page_city_name",
        cate_col,
    ]
    if "row_key" in columns:
        usecols.append("row_key")
    if bline_col != "__DERIVE_FROM_CATE__" and bline_col in columns and bline_col not in usecols:
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
        "has_row_key": bool("row_key" in df.columns),
    }
    return df, {"cate_col": "category", "bline_col": "business_line"}, metadata


def smoothed_lift(joint_count: int, src_count: int, dst_count: int, total_baskets: int, alpha: float) -> float:
    p_b_given_a = (joint_count + alpha) / (src_count + 2 * alpha)
    p_b = (dst_count + alpha) / (total_baskets + 2 * alpha)
    return float(p_b_given_a / p_b) if p_b > 0 else 0.0


def support_shrink(count: int, scale: float) -> float:
    return float(1.0 - math.exp(-count / scale))


def pair_outer_merge(left: pd.DataFrame, right: pd.DataFrame, on_cols: List[str]) -> pd.DataFrame:
    if left.empty:
        return right.copy()
    if right.empty:
        return left.copy()
    return left.merge(right, on=on_cols, how="outer")


def positive_pct_rank(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    result = pd.Series(np.zeros(len(values)), index=values.index, dtype="float64")
    positive_mask = values > 0
    if positive_mask.sum() == 0:
        return result
    result.loc[positive_mask] = values.loc[positive_mask].rank(pct=True, method="average")
    return result


def compose_signal_tags(row: pd.Series) -> str:
    tags: List[str] = []
    if row.get("markov_prob", 0) > 0:
        tags.append("MARKOV")
    if row.get("lift_score", 0) > 0:
        tags.append("LIFT")
    if row.get("ccr", 0) > 0:
        tags.append("CCR")
    return "/".join(tags) if tags else "NONE"


def infer_dominant_signal(row: pd.Series) -> str:
    signal_map = {
        "MARKOV": row.get("rank_markov", 0),
        "LIFT": row.get("rank_lift", 0),
        "CCR": row.get("rank_ccr", 0),
    }
    return max(signal_map.items(), key=lambda item: item[1])[0]


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
    return pair_stats.reset_index(drop=True)


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
        lift_df = lift_df.sort_values(["lift_score", "basket_pair_count"], ascending=False).reset_index(drop=True)
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
            print(f"  ⏳ CCR 会话扫描进度: {idx:,}/{total_sessions:,}")

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
        ccr_df = ccr_df.sort_values(["ccr_score", "ccr", "conv_sessions"], ascending=False).reset_index(drop=True)
    return ccr_df


def annotate_master_table(markov_df: pd.DataFrame, lift_df: pd.DataFrame, ccr_df: pd.DataFrame) -> pd.DataFrame:
    on_cols = ["src", "dst"]
    master = pair_outer_merge(markov_df, lift_df, on_cols=on_cols)
    master = pair_outer_merge(master, ccr_df, on_cols=on_cols)

    if master.empty:
        return master

    numeric_cols = [
        "pair_count",
        "unique_users",
        "unique_sessions",
        "median_lag_ms",
        "p90_lag_ms",
        "src_total",
        "markov_prob",
        "median_lag_min",
        "p90_lag_min",
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
        "median_conv_gap_min",
        "p90_conv_gap_min",
        "ccr_score",
    ]
    for col in numeric_cols:
        if col not in master.columns:
            master[col] = 0.0
        master[col] = pd.to_numeric(master[col], errors="coerce").fillna(0.0)

    master["rank_markov"] = positive_pct_rank(master["markov_prob"])
    master["rank_lift"] = positive_pct_rank(master["lift_score"])
    master["rank_ccr"] = positive_pct_rank(master["ccr"])
    master["signal_tags"] = master.apply(compose_signal_tags, axis=1)
    master["signal_hit_count"] = (
        (master["markov_prob"] > 0).astype(int)
        + (master["lift_score"] > 0).astype(int)
        + (master["ccr"] > 0).astype(int)
    )
    master["dominant_signal"] = master.apply(infer_dominant_signal, axis=1)
    master["candidate_pool_score"] = (
        0.35 * master["rank_markov"] + 0.35 * master["rank_lift"] + 0.30 * master["rank_ccr"]
    )
    master["cross_baseline_score"] = master["candidate_pool_score"]
    master = master.sort_values(
        ["candidate_pool_score", "signal_hit_count", "rank_markov", "rank_lift", "rank_ccr"],
        ascending=False,
    ).reset_index(drop=True)
    return master


def infer_candidate_reason(row: pd.Series) -> str:
    sources = set(str(row["candidate_sources"]).split("/"))
    sources.discard("")
    core_sources = {"MARKOV", "LIFT", "CCR"} & sources

    if {"MARKOV", "CCR"} <= core_sources:
        return "顺路且带成交承接"
    if {"LIFT", "CCR"} <= core_sources:
        return "协同明显且具备转化潜力"
    if {"MARKOV", "LIFT"} <= core_sources:
        return "顺路路径上存在协同增益"
    if "MARKOV" in core_sources:
        return "短链路承接候选"
    if "LIFT" in core_sources:
        return "独特协同候选"
    if "CCR" in core_sources:
        return "交易承接候选"
    if "MANDATORY" in sources:
        return "业务必看候选"
    return "一般候选"


def infer_candidate_tier(row: pd.Series, scope: str) -> str:
    if scope == "business_line":
        if row["candidate_source_count"] >= 2 or row["candidate_sources"].endswith("MANDATORY"):
            return "CORE_CANDIDATE"
        return "WATCH_CANDIDATE"

    if row["candidate_source_count"] >= 2 and row["candidate_pool_score"] >= 0.45:
        return "CORE_CANDIDATE"
    if row["candidate_source_count"] >= 1 and row["candidate_pool_score"] >= 0.28:
        return "EXPLORE_CANDIDATE"
    return "WATCH_CANDIDATE"


def infer_candidate_role(scope: str) -> str:
    if scope == "business_line":
        return "主报告主榜候选"
    return "类目案例池候选"


def build_candidate_pool(
    master_df: pd.DataFrame,
    scope: str,
    topk_markov: int,
    topk_lift: int,
    topk_ccr: int,
    exclude_generic_category: bool,
    mandatory_pairs: List[Tuple[str, str]],
) -> pd.DataFrame:
    if master_df.empty:
        return master_df.copy()

    work = master_df.copy()
    if scope == "category" and exclude_generic_category:
        work = work[
            ~work["src"].apply(is_generic_category)
            & ~work["dst"].apply(is_generic_category)
        ].copy()

    selection_map: Dict[Tuple[str, str], Dict[str, bool]] = defaultdict(
        lambda: {"MARKOV": False, "LIFT": False, "CCR": False, "MANDATORY": False}
    )

    markov_top = work[work["markov_prob"] > 0].head(topk_markov)
    lift_top = work[work["lift_score"] > 0].head(topk_lift)
    ccr_top = work[work["ccr"] > 0].head(topk_ccr)

    for _, row in markov_top.iterrows():
        selection_map[(row["src"], row["dst"])]["MARKOV"] = True
    for _, row in lift_top.iterrows():
        selection_map[(row["src"], row["dst"])]["LIFT"] = True
    for _, row in ccr_top.iterrows():
        selection_map[(row["src"], row["dst"])]["CCR"] = True
    for src, dst in mandatory_pairs:
        if ((work["src"] == src) & (work["dst"] == dst)).any():
            selection_map[(src, dst)]["MANDATORY"] = True

    if not selection_map:
        return work.iloc[0:0].copy()

    selected_pairs = pd.DataFrame(
        [{"src": src, "dst": dst, **flags} for (src, dst), flags in selection_map.items()]
    )
    candidate_df = work.merge(selected_pairs, on=["src", "dst"], how="inner")
    flag_cols = ["MARKOV", "LIFT", "CCR", "MANDATORY"]
    for col in flag_cols:
        if col not in candidate_df.columns:
            candidate_df[col] = False
        candidate_df[col] = candidate_df[col].fillna(False).astype(bool)

    candidate_df["candidate_source_count"] = (
        candidate_df["MARKOV"].astype(int)
        + candidate_df["LIFT"].astype(int)
        + candidate_df["CCR"].astype(int)
    )
    candidate_df["candidate_sources"] = candidate_df.apply(
        lambda row: "/".join(
            [
                name
                for name in flag_cols
                if bool(row[name])
            ]
        ),
        axis=1,
    )
    candidate_df["candidate_reason"] = candidate_df.apply(infer_candidate_reason, axis=1)
    candidate_df["candidate_tier"] = candidate_df.apply(lambda row: infer_candidate_tier(row, scope), axis=1)
    candidate_df["candidate_role"] = infer_candidate_role(scope)
    candidate_df = candidate_df.sort_values(
        ["candidate_tier", "candidate_pool_score", "signal_hit_count", "candidate_source_count"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
    return candidate_df


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


def to_md_table(frame: pd.DataFrame) -> str:
    frame = frame.copy()
    headers = [str(col) for col in frame.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in frame.fillna("").values.tolist():
        values: List[str] = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def save_summary_markdown(
    output_path: str,
    metadata: Dict[str, object],
    markov_bline: pd.DataFrame,
    lift_bline: pd.DataFrame,
    ccr_bline: pd.DataFrame,
    master_bline: pd.DataFrame,
    candidate_bline: pd.DataFrame,
    candidate_category: pd.DataFrame,
) -> None:
    lines = [
        "# Cross Baseline V3 / Candidate Discovery 摘要",
        "",
        "## 数据与口径",
        "",
        f"- 输入文件：`{metadata['file_path']}`",
        f"- 总行数：`{metadata['rows']:,}`，显式行为：`{metadata['explicit_rows']:,}`，订单：`{metadata['orders']:,}`",
        f"- 用户数：`{metadata['users']:,}`，会话数：`{metadata['sessions']:,}`",
        "- 本版明确定位为 V3 的候选发现层：保留 `Markov / Lift / CCR` 三条原始信号，额外产出统一 `candidate pool`，供后续 `CrossValueScore` 和 `TaskFitScore` 消费。",
        "- 业务线层用于主报告主榜；类目层用于案例池和精细化玩法补充。",
        "",
        "## 业务线 Top 信号",
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
        "### 合并总表 Top 12",
        "",
        to_md_table(
            master_bline.head(12)[
                [
                    "src",
                    "dst",
                    "signal_tags",
                    "dominant_signal",
                    "markov_prob",
                    "lift_score",
                    "ccr",
                    "candidate_pool_score",
                ]
            ]
        ),
        "",
        "## 业务线候选池 Top 12",
        "",
        to_md_table(
            candidate_bline.head(12)[
                [
                    "src",
                    "dst",
                    "candidate_tier",
                    "candidate_sources",
                    "candidate_reason",
                    "candidate_pool_score",
                ]
            ]
        ) if not candidate_bline.empty else "当前无业务线候选池结果。",
        "",
        "## 类目候选池 Top 15",
        "",
        to_md_table(
            candidate_category.head(15)[
                [
                    "src",
                    "dst",
                    "candidate_tier",
                    "candidate_sources",
                    "candidate_reason",
                    "candidate_pool_score",
                ]
            ]
        ) if not candidate_category.empty else "当前无类目候选池结果。",
        "",
        "## 说明",
        "",
        "- Markov：严格相邻的显式场景转移概率。",
        "- Lift：基于用户-城市篮子的宏观协同提升度，附带支持度折扣。",
        "- CCR：A 场景出现后，同 session 内最终形成 B 订单的归因转化率。",
        "- Candidate Pool：并集式候选发现，不是最终价值排序，更不是最终平台动作结论。",
    ]
    with open(output_path, "w", encoding="utf-8") as file_obj:
        file_obj.write("\n".join(lines))


def main() -> None:
    start_time = time.time()
    data_path = resolve_data_path(BASE_DIR)
    artifact_root = os.path.dirname(data_path) if data_path.startswith("/content/") else BASE_DIR
    output_dir = os.getenv("OUTPUT_DIR", os.path.join(artifact_root, "cross_baseline_v3_outputs"))
    os.makedirs(output_dir, exist_ok=True)

    max_direct_gap_min = int(os.getenv("DIRECT_MAX_GAP_MIN", "120"))
    conversion_window_min = int(os.getenv("CCR_WINDOW_MIN", "240"))
    max_rows = int(os.getenv("MAX_ROWS", "0"))

    min_markov_pair_count_bline = int(os.getenv("MIN_MARKOV_PAIR_COUNT_BLINE", "20"))
    min_markov_pair_count_category = int(os.getenv("MIN_MARKOV_PAIR_COUNT_CATEGORY", "50"))
    min_lift_pair_count_bline = int(os.getenv("MIN_LIFT_PAIR_COUNT_BLINE", "20"))
    min_lift_pair_count_category = int(os.getenv("MIN_LIFT_PAIR_COUNT_CATEGORY", "80"))
    min_ccr_conv_sessions_bline = int(os.getenv("MIN_CCR_CONV_SESSIONS_BLINE", "8"))
    min_ccr_conv_sessions_category = int(os.getenv("MIN_CCR_CONV_SESSIONS_CATEGORY", "20"))

    lift_alpha = float(os.getenv("LIFT_ALPHA", "1.0"))
    lift_support_scale_bline = float(os.getenv("LIFT_SUPPORT_SCALE_BLINE", "25.0"))
    lift_support_scale_category = float(os.getenv("LIFT_SUPPORT_SCALE_CATEGORY", "60.0"))

    top_n_heatmap = int(os.getenv("TOP_N_HEATMAP", "12"))
    topk_markov_bline = int(os.getenv("TOPK_MARKOV_BLINE", "8"))
    topk_lift_bline = int(os.getenv("TOPK_LIFT_BLINE", "8"))
    topk_ccr_bline = int(os.getenv("TOPK_CCR_BLINE", "8"))
    topk_markov_category = int(os.getenv("TOPK_MARKOV_CATEGORY", "40"))
    topk_lift_category = int(os.getenv("TOPK_LIFT_CATEGORY", "40"))
    topk_ccr_category = int(os.getenv("TOPK_CCR_CATEGORY", "40"))
    exclude_generic_category = os.getenv("EXCLUDE_GENERIC_CATEGORY", "1") == "1"

    print("🚀 [Cross Baseline V3] 启动候选发现层分析...")
    print(f"📁 输出目录: {output_dir}")
    print(
        "⚙️ 参数: "
        f"direct_gap={max_direct_gap_min}min, "
        f"ccr_window={conversion_window_min}min, "
        f"markov(bline/category)={min_markov_pair_count_bline}/{min_markov_pair_count_category}, "
        f"lift(bline/category)={min_lift_pair_count_bline}/{min_lift_pair_count_category}, "
        f"ccr(bline/category)={min_ccr_conv_sessions_bline}/{min_ccr_conv_sessions_category}, "
        f"exclude_generic_category={exclude_generic_category}, "
        f"max_rows={max_rows}"
    )

    df, cols, metadata = prepare_data(data_path, max_rows=max_rows)
    sort_cols = ["session_id", "event_timestamp"]
    if "row_key" in df.columns:
        sort_cols.append("row_key")
    df = df.sort_values(sort_cols).reset_index(drop=True)
    explicit_df = df[df["event_type"].astype(str) != "ORDER"].copy()

    category_exclude = "未知品类"
    bline_exclude = "其他业务"
    direct_gap_ms = max_direct_gap_min * 60 * 1000
    conversion_window_ms = conversion_window_min * 60 * 1000

    print("🔀 正在计算 Markov 候选信号...")
    markov_category = compute_markov(
        explicit_df=explicit_df,
        value_col=cols["cate_col"],
        max_gap_ms=direct_gap_ms,
        min_pair_count=min_markov_pair_count_category,
        exclude_value=category_exclude,
    )
    markov_bline = compute_markov(
        explicit_df=explicit_df,
        value_col=cols["bline_col"],
        max_gap_ms=direct_gap_ms,
        min_pair_count=min_markov_pair_count_bline,
        exclude_value=bline_exclude,
    )

    print("📈 正在计算 Lift 候选信号...")
    lift_category = compute_lift(
        explicit_df=explicit_df,
        value_col=cols["cate_col"],
        min_pair_count=min_lift_pair_count_category,
        exclude_value=category_exclude,
        alpha=lift_alpha,
        support_scale=lift_support_scale_category,
    )
    lift_bline = compute_lift(
        explicit_df=explicit_df,
        value_col=cols["bline_col"],
        min_pair_count=min_lift_pair_count_bline,
        exclude_value=bline_exclude,
        alpha=lift_alpha,
        support_scale=lift_support_scale_bline,
    )

    print("💰 正在计算 CCR 候选信号...")
    ccr_category = compute_ccr(
        events_df=df,
        value_col=cols["cate_col"],
        conversion_window_ms=conversion_window_ms,
        exclude_value=category_exclude,
        min_conv_sessions=min_ccr_conv_sessions_category,
    )
    ccr_bline = compute_ccr(
        events_df=df,
        value_col=cols["bline_col"],
        conversion_window_ms=conversion_window_ms,
        exclude_value=bline_exclude,
        min_conv_sessions=min_ccr_conv_sessions_bline,
    )

    print("🧩 正在合并三条信号并生成 master table...")
    master_category = annotate_master_table(markov_category, lift_category, ccr_category)
    master_bline = annotate_master_table(markov_bline, lift_bline, ccr_bline)

    print("🧠 正在构建 Candidate Pool...")
    candidate_bline = build_candidate_pool(
        master_df=master_bline,
        scope="business_line",
        topk_markov=topk_markov_bline,
        topk_lift=topk_lift_bline,
        topk_ccr=topk_ccr_bline,
        exclude_generic_category=False,
        mandatory_pairs=MANDATORY_BUSINESS_PAIRS,
    )
    candidate_category = build_candidate_pool(
        master_df=master_category,
        scope="category",
        topk_markov=topk_markov_category,
        topk_lift=topk_lift_category,
        topk_ccr=topk_ccr_category,
        exclude_generic_category=exclude_generic_category,
        mandatory_pairs=[],
    )

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
        "candidate_pool_category.csv": candidate_category,
        "candidate_pool_business_line.csv": candidate_bline,
    }
    for filename, frame in outputs.items():
        frame.to_csv(os.path.join(output_dir, filename), index=False, encoding="utf-8-sig")

    print("🖼 正在生成热力图...")
    save_heatmap(
        markov_bline,
        value_col="markov_prob",
        output_path=os.path.join(output_dir, "heatmap_markov_business_line.png"),
        title="美团 Cross 候选发现 | Markov 业务线转移热力图",
        subtitle="口径：严格相邻显式场景跳转，过滤自循环与已知跨城异常跳转",
        top_n=top_n_heatmap,
    )
    save_heatmap(
        lift_bline,
        value_col="lift_score",
        output_path=os.path.join(output_dir, "heatmap_lift_business_line.png"),
        title="美团 Cross 候选发现 | Lift 业务线协同热力图",
        subtitle="口径：用户-城市篮子共现 + 平滑 Lift + 支持度折扣",
        top_n=top_n_heatmap,
    )
    save_heatmap(
        ccr_bline,
        value_col="ccr",
        output_path=os.path.join(output_dir, "heatmap_ccr_business_line.png"),
        title="美团 Cross 候选发现 | CCR 业务线承接热力图",
        subtitle="口径：A 场景出现后，同 session 内最终形成 B 订单",
        top_n=top_n_heatmap,
    )

    summary_path = os.path.join(output_dir, "baseline_summary.md")
    save_summary_markdown(
        output_path=summary_path,
        metadata=metadata,
        markov_bline=markov_bline,
        lift_bline=lift_bline,
        ccr_bline=ccr_bline,
        master_bline=master_bline,
        candidate_bline=candidate_bline,
        candidate_category=candidate_category,
    )
    save_summary_markdown(
        output_path=os.path.join(output_dir, "candidate_discovery_summary.md"),
        metadata=metadata,
        markov_bline=markov_bline,
        lift_bline=lift_bline,
        ccr_bline=ccr_bline,
        master_bline=master_bline,
        candidate_bline=candidate_bline,
        candidate_category=candidate_category,
    )

    metadata["version"] = "cross_baseline_v3"
    metadata["layer_role"] = "candidate_discovery"
    metadata["config"] = {
        "direct_gap_min": max_direct_gap_min,
        "ccr_window_min": conversion_window_min,
        "min_markov_pair_count_bline": min_markov_pair_count_bline,
        "min_markov_pair_count_category": min_markov_pair_count_category,
        "min_lift_pair_count_bline": min_lift_pair_count_bline,
        "min_lift_pair_count_category": min_lift_pair_count_category,
        "min_ccr_conv_sessions_bline": min_ccr_conv_sessions_bline,
        "min_ccr_conv_sessions_category": min_ccr_conv_sessions_category,
        "lift_alpha": lift_alpha,
        "lift_support_scale_bline": lift_support_scale_bline,
        "lift_support_scale_category": lift_support_scale_category,
        "top_n_heatmap": top_n_heatmap,
        "topk_markov_bline": topk_markov_bline,
        "topk_lift_bline": topk_lift_bline,
        "topk_ccr_bline": topk_ccr_bline,
        "topk_markov_category": topk_markov_category,
        "topk_lift_category": topk_lift_category,
        "topk_ccr_category": topk_ccr_category,
        "exclude_generic_category": exclude_generic_category,
        "max_rows": max_rows,
    }
    metadata["outputs"] = list(outputs.keys()) + [
        "heatmap_markov_business_line.png",
        "heatmap_lift_business_line.png",
        "heatmap_ccr_business_line.png",
        "baseline_summary.md",
        "candidate_discovery_summary.md",
    ]
    with open(os.path.join(output_dir, "baseline_metadata.json"), "w", encoding="utf-8") as file_obj:
        json.dump(metadata, file_obj, ensure_ascii=False, indent=2)

    elapsed = time.time() - start_time
    print("✅ Cross Baseline V3 完成。")
    print(
        f"📊 Markov(category)={len(markov_category):,} | Lift(category)={len(lift_category):,} | CCR(category)={len(ccr_category):,}"
    )
    print(
        f"📊 Markov(bline)={len(markov_bline):,} | Lift(bline)={len(lift_bline):,} | CCR(bline)={len(ccr_bline):,}"
    )
    print(
        f"🎯 CandidatePool(category)={len(candidate_category):,} | CandidatePool(bline)={len(candidate_bline):,}"
    )
    print(f"🕒 总耗时: {elapsed / 60:.2f} 分钟")


if __name__ == "__main__":
    main()
