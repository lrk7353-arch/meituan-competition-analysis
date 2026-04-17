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
    return os.path.join(base_dir, "cross_score_v1_outputs")


def positive_pct_rank(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    result = pd.Series(np.zeros(len(values)), index=values.index, dtype="float64")
    positive_mask = values > 0
    if positive_mask.sum() == 0:
        return result
    result.loc[positive_mask] = values.loc[positive_mask].rank(pct=True, method="average")
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


def ensure_numeric_columns(frame: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    work = frame.copy()
    for col in columns:
        if col not in work.columns:
            work[col] = 0.0
        work[col] = pd.to_numeric(work[col], errors="coerce")
    return work


def infer_opportunity_type(row: pd.Series) -> str:
    if row["rank_ccr"] >= 0.75 and row["rank_markov"] >= 0.75:
        return "即时转化型"
    if row["rank_lift"] >= 0.75 and row["rank_ccr"] < 0.50:
        return "协同种草型"
    if row["s_cov"] >= 0.75 and row["rank_ccr"] < 0.50:
        return "广覆盖培育型"
    if row["s_time"] >= 0.75 and row["rank_markov"] >= 0.50:
        return "短窗口触达型"
    return "常规运营型"


def infer_strategy_hint(row: pd.Series) -> str:
    opportunity_type = row["opportunity_type"]
    if opportunity_type == "即时转化型":
        return "优先做强曝光+短链路导流，适合首页入口、支付后承接和高频召回。"
    if opportunity_type == "协同种草型":
        return "适合做联动会场和内容种草，先强化认知协同，不建议直接压强转化。"
    if opportunity_type == "广覆盖培育型":
        return "适合做大盘渗透和弱触达，优先做低成本曝光，再观察转化爬升。"
    if opportunity_type == "短窗口触达型":
        return "适合做分钟级承接链路，重点优化触达时机和落地页承接。"
    return "建议先小流量试投，观察转化链路后再决定是否加大资源。"


def infer_strategy_tier(row: pd.Series) -> str:
    if (
        row["cross_score"] >= 65
        and row["s_stat"] >= 0.60
        and row["s_seq"] >= 0.60
        and row["s_cov"] >= 0.45
    ):
        return "HIGH"
    if row["cross_score"] >= 45 and (
        row["s_stat"] >= 0.45 or row["s_seq"] >= 0.45 or row["s_cov"] >= 0.45
    ):
        return "MEDIUM"
    return "LOW"


def compute_cross_score(frame: pd.DataFrame) -> pd.DataFrame:
    work = ensure_numeric_columns(
        frame,
        [
            "pair_count",
            "unique_users",
            "unique_sessions",
            "basket_pair_count",
            "conv_sessions",
            "markov_prob",
            "lift_score",
            "ccr",
            "median_lag_min",
            "p90_lag_min",
            "median_conv_gap_min",
            "p90_conv_gap_min",
        ],
    )

    work["rank_markov"] = positive_pct_rank(work["markov_prob"])
    work["rank_lift"] = positive_pct_rank(work["lift_score"])
    work["rank_ccr"] = positive_pct_rank(work["ccr"])

    work["rank_pair_count"] = positive_pct_rank(work["pair_count"])
    work["rank_unique_users"] = positive_pct_rank(work["unique_users"])
    work["rank_basket_pair_count"] = positive_pct_rank(work["basket_pair_count"])
    work["rank_conv_sessions"] = positive_pct_rank(work["conv_sessions"])

    work["rank_fast_median_lag"] = inverse_positive_pct_rank(work["median_lag_min"])
    work["rank_fast_p90_lag"] = inverse_positive_pct_rank(work["p90_lag_min"])
    work["rank_fast_median_conv_gap"] = inverse_positive_pct_rank(work["median_conv_gap_min"])
    work["rank_fast_p90_conv_gap"] = inverse_positive_pct_rank(work["p90_conv_gap_min"])

    work["s_stat"] = 0.40 * work["rank_lift"] + 0.60 * work["rank_ccr"]
    work["s_seq"] = work["rank_markov"]
    work["s_cov"] = (
        0.30 * work["rank_unique_users"]
        + 0.20 * work["rank_pair_count"]
        + 0.20 * work["rank_basket_pair_count"]
        + 0.30 * work["rank_conv_sessions"]
    )
    work["s_time"] = (
        0.25 * work["rank_fast_median_lag"]
        + 0.25 * work["rank_fast_p90_lag"]
        + 0.25 * work["rank_fast_median_conv_gap"]
        + 0.25 * work["rank_fast_p90_conv_gap"]
    )

    work["cross_score"] = 100.0 * (
        0.40 * work["s_stat"] + 0.25 * work["s_seq"] + 0.20 * work["s_cov"] + 0.15 * work["s_time"]
    )
    work["opportunity_type"] = work.apply(infer_opportunity_type, axis=1)
    work["strategy_tier"] = work.apply(infer_strategy_tier, axis=1)
    work["strategy_hint"] = work.apply(infer_strategy_hint, axis=1)
    work = work.sort_values(["cross_score", "s_stat", "s_seq", "s_cov"], ascending=False).reset_index(drop=True)
    return work


def to_md_table(frame: pd.DataFrame) -> str:
    headers = [str(col) for col in frame.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in frame.fillna("").values.tolist():
        values = []
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
        ["src", "dst", "cross_score", "strategy_tier", "opportunity_type", "s_stat", "s_seq", "s_cov", "s_time"]
    ].copy()
    high_business = business_df[business_df["strategy_tier"] == "HIGH"].head(10)[
        ["src", "dst", "cross_score", "opportunity_type", "strategy_hint"]
    ].copy()
    medium_synergy = business_df[
        (business_df["strategy_tier"] == "MEDIUM") & (business_df["rank_lift"] >= 0.75)
    ].head(10)[["src", "dst", "cross_score", "opportunity_type", "strategy_hint"]].copy()
    top_category = category_df.head(10)[
        ["src", "dst", "cross_score", "strategy_tier", "opportunity_type"]
    ].copy()

    lines = [
        "# CrossScore V1 摘要",
        "",
        "## 评分口径",
        "",
        f"- 输入目录：`{input_dir}`",
        "- 本版将现有 `Lift / Markov / CCR` 融合为四个可解释分项：",
        "  - `S_stat`：统计协同强度 = `0.40 * Lift百分位 + 0.60 * CCR百分位`",
        "  - `S_seq`：序列顺路程度 = `Markov概率百分位`",
        "  - `S_cov`：覆盖与体量 = `unique_users / pair_count / basket_pair_count / conv_sessions` 的加权百分位",
        "  - `S_time`：触达时机成熟度 = 各类时间滞后指标的反向百分位",
        "- 最终得分：`CrossScore = 100 * (0.40*S_stat + 0.25*S_seq + 0.20*S_cov + 0.15*S_time)`",
        "- 策略层级：",
        "  - `HIGH`：高协同 + 高顺路 + 体量足够，适合优先做强导流",
        "  - `MEDIUM`：存在明确机会，但更适合联动曝光、轻触达或小流量验证",
        "  - `LOW`：当前转化或覆盖不足，建议观察或仅作为补充案例",
        "",
        "## 业务线 Top 10",
        "",
        to_md_table(top_business),
        "",
        "## HIGH 优先级业务线",
        "",
        to_md_table(high_business) if not high_business.empty else "当前无 HIGH 业务线。",
        "",
        "## MEDIUM 中值得讲的协同案例",
        "",
        to_md_table(medium_synergy) if not medium_synergy.empty else "当前无符合条件的 MEDIUM 协同案例。",
        "",
        "## 类目层 Top 10",
        "",
        to_md_table(top_category),
        "",
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_metadata(output_path: str) -> None:
    metadata: Dict[str, object] = {
        "version": "cross_score_v1",
        "score_formula": {
            "s_stat": "0.40 * rank(lift_score) + 0.60 * rank(ccr)",
            "s_seq": "rank(markov_prob)",
            "s_cov": "0.30 * rank(unique_users) + 0.20 * rank(pair_count) + 0.20 * rank(basket_pair_count) + 0.30 * rank(conv_sessions)",
            "s_time": "0.25 * inv_rank(median_lag_min) + 0.25 * inv_rank(p90_lag_min) + 0.25 * inv_rank(median_conv_gap_min) + 0.25 * inv_rank(p90_conv_gap_min)",
            "cross_score": "100 * (0.40 * s_stat + 0.25 * s_seq + 0.20 * s_cov + 0.15 * s_time)",
        },
        "tier_rules": {
            "HIGH": "cross_score >= 65 and s_stat >= 0.60 and s_seq >= 0.60 and s_cov >= 0.45",
            "MEDIUM": "cross_score >= 45 and (s_stat >= 0.45 or s_seq >= 0.45 or s_cov >= 0.45)",
            "LOW": "otherwise",
        },
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


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

    scored_business = compute_cross_score(business_df)
    scored_category = compute_cross_score(category_df)

    scored_business.to_csv(
        os.path.join(output_dir, "cross_score_business_line.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    scored_category.to_csv(
        os.path.join(output_dir, "cross_score_category.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    build_summary_markdown(
        os.path.join(output_dir, "cross_score_summary.md"),
        scored_business,
        scored_category,
        input_dir,
    )
    save_metadata(os.path.join(output_dir, "cross_score_metadata.json"))

    print(f"✅ CrossScore 结果已输出到: {output_dir}")
    print("   - cross_score_business_line.csv")
    print("   - cross_score_category.csv")
    print("   - cross_score_summary.md")
    print("   - cross_score_metadata.json")


if __name__ == "__main__":
    main()
