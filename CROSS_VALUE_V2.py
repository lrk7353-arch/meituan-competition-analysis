import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_input_dir(base_dir: str) -> str:
    env_path = os.getenv("INPUT_DIR")
    if env_path:
        return env_path
    return os.path.join(base_dir, "cross_baseline_v2_outputs")


def resolve_output_dir(base_dir: str) -> str:
    env_path = os.getenv("OUTPUT_DIR")
    if env_path:
        return env_path
    return os.path.join(base_dir, "cross_value_v2_outputs")


def ensure_numeric_columns(frame: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    work = frame.copy()
    for col in columns:
        if col not in work.columns:
            work[col] = 0.0
        work[col] = pd.to_numeric(work[col], errors="coerce")
    return work


def positive_pct_rank(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    result = pd.Series(np.zeros(len(values)), index=values.index, dtype="float64")
    positive_mask = values > 0
    if positive_mask.sum() == 0:
        return result
    result.loc[positive_mask] = values.loc[positive_mask].rank(pct=True, method="average")
    return result


def positive_log_pct_rank(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    result = pd.Series(np.zeros(len(values)), index=values.index, dtype="float64")
    positive_mask = values > 0
    if positive_mask.sum() == 0:
        return result
    logged = np.log1p(values.loc[positive_mask])
    result.loc[positive_mask] = logged.rank(pct=True, method="average")
    return result


def inverse_positive_pct_rank(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    result = pd.Series(np.zeros(len(values)), index=values.index, dtype="float64")
    valid_mask = np.isfinite(values) & (values > 0)
    valid_count = int(valid_mask.sum())
    if valid_count == 0:
        return result
    ranks = values.loc[valid_mask].rank(pct=True, method="average")
    result.loc[valid_mask] = (1.0 - ranks + (1.0 / valid_count)).clip(lower=0.0, upper=1.0)
    return result


def infer_dominant_driver(row: pd.Series) -> str:
    drivers = {
        "统计协同": row["v_stat"],
        "序列承接": row["v_seq"],
        "成交归因": row["v_conv"],
        "覆盖稳定": row["v_cov"],
        "时机成熟": row["v_time"],
    }
    return max(drivers.items(), key=lambda item: item[1])[0]


def infer_value_archetype(row: pd.Series) -> str:
    if row["v_conv"] >= 0.72 and row["v_seq"] >= 0.68:
        return "即时承接价值型"
    if row["v_stat"] >= 0.76 and row["support_confidence"] < 0.55:
        return "特色协同探索型"
    if row["v_stat"] >= 0.72 and row["v_conv"] < 0.60:
        return "独特协同价值型"
    if row["v_conv"] >= 0.66 and row["v_cov"] >= 0.72:
        return "大盘转化价值型"
    if row["v_seq"] >= 0.75 and row["v_time"] >= 0.70:
        return "短链路机会型"
    if min(row["v_stat"], row["v_seq"], row["v_conv"]) >= 0.55:
        return "均衡价值型"
    return "结构性机会型"


def infer_destination_profile(row: pd.Series) -> str:
    if row["dst_popularity_rank"] >= 0.85 and row["v_stat"] < 0.60:
        return "头部承接终点"
    if row["dst_popularity_rank"] >= 0.85 and row["v_stat"] >= 0.60:
        return "头部协同终点"
    if row["dst_popularity_rank"] <= 0.35 and row["v_stat"] >= 0.70:
        return "特色协同终点"
    return "常规终点"


def infer_value_tier(row: pd.Series) -> str:
    core_signal = max(row["v_stat"], row["v_seq"], row["v_conv"])
    if row["cross_value_score"] >= 72 and row["support_confidence"] >= 0.55 and core_signal >= 0.70:
        return "HIGH_VALUE"
    if row["cross_value_score"] >= 56 and row["support_confidence"] >= 0.40 and core_signal >= 0.55:
        return "MEDIUM_VALUE"
    return "LOW_VALUE"


def build_signal_tags(row: pd.Series) -> str:
    tags: List[str] = []
    if row["v_stat"] >= 0.75:
        tags.append("STAT")
    if row["v_seq"] >= 0.75:
        tags.append("SEQ")
    if row["v_conv"] >= 0.75:
        tags.append("CONV")
    if row["support_confidence"] >= 0.70:
        tags.append("STABLE")
    if not tags:
        return "BASE"
    return "/".join(tags)


def compute_cross_value(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
    work = ensure_numeric_columns(
        frame,
        [
            "pair_count",
            "unique_users",
            "unique_sessions",
            "basket_pair_count",
            "conv_sessions",
            "src_sessions",
            "dst_basket_count",
            "markov_prob",
            "lift_support_shrink",
            "lift_score",
            "cooccurrence_rate",
            "ccr",
            "median_lag_min",
            "p90_lag_min",
            "median_conv_gap_min",
            "p90_conv_gap_min",
        ],
    )

    work["rank_lift_signal"] = positive_pct_rank(work["lift_score"])
    work["rank_cooccurrence"] = positive_pct_rank(work["cooccurrence_rate"])
    work["rank_lift_support"] = positive_pct_rank(work["lift_support_shrink"])

    work["rank_markov_signal"] = positive_pct_rank(work["markov_prob"])
    work["rank_ccr_signal"] = positive_pct_rank(work["ccr"])

    work["rank_log_pair_count"] = positive_log_pct_rank(work["pair_count"])
    work["rank_log_unique_users"] = positive_log_pct_rank(work["unique_users"])
    work["rank_log_unique_sessions"] = positive_log_pct_rank(work["unique_sessions"])
    work["rank_log_basket_pair_count"] = positive_log_pct_rank(work["basket_pair_count"])
    work["rank_log_conv_sessions"] = positive_log_pct_rank(work["conv_sessions"])
    work["rank_log_src_sessions"] = positive_log_pct_rank(work["src_sessions"])
    work["dst_popularity_rank"] = positive_log_pct_rank(work["dst_basket_count"])

    work["rank_fast_median_lag"] = inverse_positive_pct_rank(work["median_lag_min"])
    work["rank_fast_p90_lag"] = inverse_positive_pct_rank(work["p90_lag_min"])
    work["rank_fast_median_conv_gap"] = inverse_positive_pct_rank(work["median_conv_gap_min"])
    work["rank_fast_p90_conv_gap"] = inverse_positive_pct_rank(work["p90_conv_gap_min"])

    work["v_stat"] = (
        0.60 * work["rank_lift_signal"]
        + 0.25 * work["rank_cooccurrence"]
        + 0.15 * work["rank_lift_support"]
    )
    work["v_seq"] = work["rank_markov_signal"]
    work["v_conv"] = 0.75 * work["rank_ccr_signal"] + 0.25 * work["rank_log_conv_sessions"]
    work["v_cov"] = (
        0.30 * work["rank_log_unique_users"]
        + 0.25 * work["rank_log_unique_sessions"]
        + 0.25 * work["rank_log_pair_count"]
        + 0.20 * work["rank_log_basket_pair_count"]
    )
    work["v_time"] = (
        0.30 * work["rank_fast_median_lag"]
        + 0.20 * work["rank_fast_p90_lag"]
        + 0.30 * work["rank_fast_median_conv_gap"]
        + 0.20 * work["rank_fast_p90_conv_gap"]
    )
    work["support_confidence"] = (
        0.30 * work["rank_log_unique_users"]
        + 0.25 * work["rank_log_unique_sessions"]
        + 0.20 * work["rank_log_pair_count"]
        + 0.15 * work["rank_log_conv_sessions"]
        + 0.10 * work["rank_lift_support"]
    )

    work["cross_value_score"] = 100.0 * (
        0.28 * work["v_stat"]
        + 0.22 * work["v_seq"]
        + 0.24 * work["v_conv"]
        + 0.16 * work["v_cov"]
        + 0.10 * work["v_time"]
    )

    work["scope"] = scope
    work["dominant_driver"] = work.apply(infer_dominant_driver, axis=1)
    work["value_archetype"] = work.apply(infer_value_archetype, axis=1)
    work["destination_profile"] = work.apply(infer_destination_profile, axis=1)
    work["signal_tags"] = work.apply(build_signal_tags, axis=1)
    work["value_tier"] = work.apply(infer_value_tier, axis=1)

    work = work.sort_values(
        ["cross_value_score", "support_confidence", "v_conv", "v_seq", "v_stat", "v_cov"],
        ascending=False,
    ).reset_index(drop=True)
    return work


def to_md_table(frame: pd.DataFrame) -> str:
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


def build_summary_markdown(
    output_path: str,
    business_df: pd.DataFrame,
    category_df: pd.DataFrame,
    input_dir: str,
) -> None:
    top_business = business_df.head(10)[
        [
            "src",
            "dst",
            "cross_value_score",
            "value_tier",
            "value_archetype",
            "dominant_driver",
            "v_stat",
            "v_seq",
            "v_conv",
            "v_cov",
            "v_time",
            "support_confidence",
        ]
    ].copy()
    high_business = business_df[business_df["value_tier"] == "HIGH_VALUE"].head(10)[
        [
            "src",
            "dst",
            "cross_value_score",
            "value_archetype",
            "dominant_driver",
            "signal_tags",
        ]
    ].copy()
    synergy_category = (
        category_df[category_df["support_confidence"] >= 0.50]
        .sort_values(["v_stat", "cross_value_score"], ascending=False)
        .head(10)[
            [
                "src",
                "dst",
                "cross_value_score",
                "value_archetype",
                "destination_profile",
                "v_stat",
                "support_confidence",
            ]
        ]
        .copy()
    )
    conversion_category = (
        category_df[category_df["support_confidence"] >= 0.50]
        .sort_values(["v_conv", "cross_value_score"], ascending=False)
        .head(10)[
            [
                "src",
                "dst",
                "cross_value_score",
                "value_archetype",
                "destination_profile",
                "v_conv",
                "support_confidence",
            ]
        ]
        .copy()
    )
    top_category = category_df.head(10)[
        [
            "src",
            "dst",
            "cross_value_score",
            "value_tier",
            "value_archetype",
            "destination_profile",
            "support_confidence",
        ]
    ].copy()

    lines = [
        "# CrossValueScore V2 摘要",
        "",
        "## 价值层口径",
        "",
        f"- 输入目录：`{input_dir}`",
        "- 本脚本是 V3 正式价值层，不承担 `TaskFitScore` 或最终 `PushPriority` 的职责。",
        "- 它只回答一个问题：`这条 Cross 作为业务机会值不值钱。`",
        "- 五个分项定义：",
        "  - `V_stat`：统计协同价值，强调 `Lift`、共现率与支持质量",
        "  - `V_seq`：序列承接价值，对应 `Markov` 顺路程度",
        "  - `V_conv`：交易归因价值，对应 `CCR` 与转化会话规模",
        "  - `V_cov`：覆盖与稳定性，对 `users / sessions / pair_count` 做对数平滑后聚合",
        "  - `V_time`：时机成熟度，对时间滞后做反向排序",
        "- 最终得分：`CrossValueScore = 100 * (0.28*V_stat + 0.22*V_seq + 0.24*V_conv + 0.16*V_cov + 0.10*V_time)`",
        "- 额外诊断项：",
        "  - `support_confidence`：支持度与稳定性诊断，不进入最终决策层，但帮助识别长尾波动",
        "  - `destination_profile`：终点热度画像，用于提示头部承接终点与特色终点",
        "",
        "## 业务线 Top 10",
        "",
        to_md_table(top_business),
        "",
        "## HIGH_VALUE 业务线",
        "",
        to_md_table(high_business) if not high_business.empty else "当前无 HIGH_VALUE 业务线。",
        "",
        "## 类目层 Top 10",
        "",
        to_md_table(top_category),
        "",
        "## 类目层高协同案例",
        "",
        to_md_table(synergy_category) if not synergy_category.empty else "当前无满足支持度要求的高协同案例。",
        "",
        "## 类目层高转化案例",
        "",
        to_md_table(conversion_category) if not conversion_category.empty else "当前无满足支持度要求的高转化案例。",
        "",
        "## 解释提醒",
        "",
        "- `CrossValueScore` 不是最终推送优先级，不能替代 `TaskFitScore`。",
        "- 对业务线层，可将其理解为平台级 Cross 机会排序。",
        "- 对类目层，应结合 `support_confidence` 与终点画像共同判断，避免把长尾噪声或头部承接误当成最终策略结论。",
        "",
    ]

    with open(output_path, "w", encoding="utf-8") as file_obj:
        file_obj.write("\n".join(lines))


def save_metadata(output_path: str) -> None:
    metadata: Dict[str, object] = {
        "version": "cross_value_v2",
        "positioning": "value_only_layer",
        "formula": {
            "v_stat": "0.60 * rank(lift_score) + 0.25 * rank(cooccurrence_rate) + 0.15 * rank(lift_support_shrink)",
            "v_seq": "rank(markov_prob)",
            "v_conv": "0.75 * rank(ccr) + 0.25 * rank(log1p(conv_sessions))",
            "v_cov": "0.30 * rank(log1p(unique_users)) + 0.25 * rank(log1p(unique_sessions)) + 0.25 * rank(log1p(pair_count)) + 0.20 * rank(log1p(basket_pair_count))",
            "v_time": "0.30 * inv_rank(median_lag_min) + 0.20 * inv_rank(p90_lag_min) + 0.30 * inv_rank(median_conv_gap_min) + 0.20 * inv_rank(p90_conv_gap_min)",
            "cross_value_score": "100 * (0.28 * v_stat + 0.22 * v_seq + 0.24 * v_conv + 0.16 * v_cov + 0.10 * v_time)",
            "support_confidence": "0.30 * rank(log1p(unique_users)) + 0.25 * rank(log1p(unique_sessions)) + 0.20 * rank(log1p(pair_count)) + 0.15 * rank(log1p(conv_sessions)) + 0.10 * rank(lift_support_shrink)",
        },
        "tier_rules": {
            "HIGH_VALUE": "cross_value_score >= 72 and support_confidence >= 0.55 and max(v_stat, v_seq, v_conv) >= 0.70",
            "MEDIUM_VALUE": "cross_value_score >= 56 and support_confidence >= 0.40 and max(v_stat, v_seq, v_conv) >= 0.55",
            "LOW_VALUE": "otherwise",
        },
        "notes": [
            "This layer is intentionally value-only and does not include task continuity constraints.",
            "Destination heat is diagnostic only and is not used to suppress final value score.",
            "Log-smoothed coverage ranks are used to reduce head-size domination in raw count metrics.",
        ],
    }
    with open(output_path, "w", encoding="utf-8") as file_obj:
        json.dump(metadata, file_obj, ensure_ascii=False, indent=2)


def main() -> None:
    input_dir = resolve_input_dir(BASE_DIR)
    output_dir = resolve_output_dir(BASE_DIR)
    os.makedirs(output_dir, exist_ok=True)

    business_path = os.path.join(input_dir, "cross_pair_master_table_business_line.csv")
    category_path = os.path.join(input_dir, "cross_pair_master_table_category.csv")

    if not os.path.exists(business_path):
        raise FileNotFoundError(f"未找到业务线总表：{business_path}")
    if not os.path.exists(category_path):
        raise FileNotFoundError(f"未找到类目总表：{category_path}")

    business_df = pd.read_csv(business_path)
    category_df = pd.read_csv(category_path)

    scored_business = compute_cross_value(business_df, scope="business_line")
    scored_category = compute_cross_value(category_df, scope="category")

    scored_business.to_csv(
        os.path.join(output_dir, "cross_value_business_line.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    scored_category.to_csv(
        os.path.join(output_dir, "cross_value_category.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    build_summary_markdown(
        os.path.join(output_dir, "cross_value_summary.md"),
        scored_business,
        scored_category,
        input_dir,
    )
    save_metadata(os.path.join(output_dir, "cross_value_metadata.json"))

    print(f"✅ CrossValueScore V2 结果已输出到: {output_dir}")
    print("   - cross_value_business_line.csv")
    print("   - cross_value_category.csv")
    print("   - cross_value_summary.md")
    print("   - cross_value_metadata.json")


if __name__ == "__main__":
    main()
