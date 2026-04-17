import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


VALUE_COLUMNS = [
    "src",
    "dst",
    "scope",
    "cross_value_score",
    "value_tier",
    "value_archetype",
    "dominant_driver",
    "destination_profile",
    "signal_tags",
    "support_confidence",
    "v_stat",
    "v_seq",
    "v_conv",
    "v_cov",
    "v_time",
]

FIT_COLUMNS = [
    "src",
    "dst",
    "scope",
    "task_fit_score",
    "fit_tier",
    "fit_archetype",
    "risk_label",
    "constraint_action",
    "evidence_confidence",
    "planning_gap",
    "rule_source",
    "src_theme",
    "dst_theme",
    "r_intent",
    "r_chain",
    "r_semantic",
    "r_temporal",
    "r_terminal",
]

FIT_GATE_BY_TIER = {
    "HIGH_FIT": 1.00,
    "MEDIUM_FIT": 0.74,
    "LOW_FIT": 0.38,
}

RISK_PENALTY = {
    "连续性良好": 1.00,
    "时段错配风险": 0.74,
    "链路失真风险": 0.72,
    "终局质量风险": 0.68,
    "任务漂移风险": 0.50,
}

VALUE_THRESHOLD = 60.0
FIT_THRESHOLD = 60.0


def resolve_value_dir(base_dir: str) -> str:
    env_path = os.getenv("VALUE_DIR")
    if env_path:
        return env_path
    return os.path.join(base_dir, "cross_value_v2_outputs")


def resolve_fit_dir(base_dir: str) -> str:
    env_path = os.getenv("FIT_DIR")
    if env_path:
        return env_path
    return os.path.join(base_dir, "task_continuity_v1_outputs")


def resolve_output_dir(base_dir: str) -> str:
    env_path = os.getenv("OUTPUT_DIR")
    if env_path:
        return env_path
    return os.path.join(base_dir, "cross_decision_v3_outputs")


def load_layer_csv(path: str, required_columns: List[str]) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到所需输入文件：{path}")
    frame = pd.read_csv(path)
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"文件字段缺失：{path} -> {missing}")
    return frame[required_columns].copy()


def quadrant_label(value_score: float, fit_score: float) -> str:
    value_high = value_score >= VALUE_THRESHOLD
    fit_high = fit_score >= FIT_THRESHOLD
    if value_high and fit_high:
        return "Q1_高价值高连续性"
    if value_high and not fit_high:
        return "Q2_高价值低连续性"
    if (not value_high) and fit_high:
        return "Q3_低价值高连续性"
    return "Q4_低价值低连续性"


def infer_priority_style(row: pd.Series) -> str:
    if row["value_archetype"] in {"即时承接价值型", "短链路机会型"} and row["fit_archetype"] in {
        "主任务延续型",
        "短链路承接型",
    }:
        return "强承接推进型"
    if row["value_archetype"] in {"独特协同价值型", "特色协同探索型"}:
        return "协同联动型"
    if row["quadrant"] == "Q2_高价值低连续性":
        return "约束试验型"
    if row["quadrant"] == "Q3_低价值高连续性":
        return "轻触达补充型"
    return "稳健运营型"


def infer_decision_tier(row: pd.Series) -> str:
    if (
        row["quadrant"] == "Q1_高价值高连续性"
        and row["decision_confidence"] >= 0.55
        and row["risk_label"] == "连续性良好"
        and row["push_priority"] >= 58
    ):
        return "STRONG_PUSH"
    if row["quadrant"] == "Q1_高价值高连续性" and row["push_priority"] >= 42:
        return "TARGETED_PUSH"
    if row["quadrant"] == "Q2_高价值低连续性":
        return "GUARDED_TEST"
    if row["quadrant"] == "Q3_低价值高连续性":
        return "LIGHT_TOUCH"
    return "SUPPRESS"


def infer_decision_action(row: pd.Series) -> str:
    if row["decision_tier"] == "STRONG_PUSH":
        return "优先进入主资源位和重点联动清单"
    if row["decision_tier"] == "TARGETED_PUSH":
        return "建议做定向放量和重点承接页优化"
    if row["decision_tier"] == "GUARDED_TEST":
        return "建议小流量试验并加约束条件"
    if row["decision_tier"] == "LIGHT_TOUCH":
        return "建议轻触达补充，不做高压导流"
    return "建议在决策层抑制或过滤"


def infer_surface_recommendation(row: pd.Series) -> str:
    if row["decision_tier"] == "STRONG_PUSH":
        if row["priority_style"] == "强承接推进型":
            return "首页承接 / 支付后推荐 / 高频召回"
        return "首页承接 / 会场联动 / 支付后推荐"
    if row["decision_tier"] == "TARGETED_PUSH":
        if row["priority_style"] == "协同联动型":
            return "会场联动 / 搜索联动 / 内容种草"
        return "支付后推荐 / 搜索联动 / 定向召回"
    if row["decision_tier"] == "GUARDED_TEST":
        if row["risk_label"] == "时段错配风险":
            return "限时段搜索联动 / 小流量触达"
        if row["risk_label"] == "任务漂移风险":
            return "内容种草 / 会场承接 / 非强插流量位"
        return "小流量试验 / 补充流量位"
    if row["decision_tier"] == "LIGHT_TOUCH":
        return "补充流量位 / 内容种草 / 弱曝光提醒"
    return "暂不建议主动曝光"


def infer_governance_note(row: pd.Series) -> str:
    if row["risk_label"] == "任务漂移风险":
        return "需增加同主题过滤、入口限制或只在明确意图场景触发"
    if row["risk_label"] == "时段错配风险":
        return "建议限定时段、限定会话位置后再放量"
    if row["risk_label"] == "链路失真风险":
        return "建议限制触发窗口，避免多跳扩散"
    if row["risk_label"] == "终局质量风险":
        return "需补强落地页承接和成交闭环"
    if row["decision_tier"] == "STRONG_PUSH":
        return "可作为平台级重点 Cross 机会推进"
    return "建议继续观察真实曝光承接效果"


def infer_strategy_card(row: pd.Series) -> str:
    if row["decision_tier"] == "STRONG_PUSH":
        return "高价值且高连续性，可直接进入重点资源位联动"
    if row["decision_tier"] == "TARGETED_PUSH":
        return "整体可推，但更适合定向人群和承接页优化"
    if row["decision_tier"] == "GUARDED_TEST":
        return "价值可观但连续性不足，适合带约束的小流量验证"
    if row["decision_tier"] == "LIGHT_TOUCH":
        return "连续性尚可但收益有限，适合作为补充玩法"
    return "价值和连续性均不足，当前不建议主动推动"


def compute_push_priority(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["value_norm"] = work["cross_value_score"] / 100.0
    work["fit_norm"] = work["task_fit_score"] / 100.0
    work["decision_confidence"] = np.sqrt(
        work["support_confidence"].clip(lower=0.0, upper=1.0)
        * work["evidence_confidence"].clip(lower=0.0, upper=1.0)
    )
    work["fit_gate"] = work["fit_tier"].map(FIT_GATE_BY_TIER).fillna(FIT_GATE_BY_TIER["LOW_FIT"])
    work["risk_penalty"] = work["risk_label"].map(RISK_PENALTY).fillna(0.72)

    confidence_boost = 0.85 + 0.15 * work["decision_confidence"]
    raw_priority = (
        work["value_norm"]
        * work["fit_norm"]
        * work["fit_gate"]
        * work["risk_penalty"]
        * confidence_boost
    )
    work["push_priority"] = 100.0 * raw_priority
    work["quadrant"] = work.apply(
        lambda row: quadrant_label(row["cross_value_score"], row["task_fit_score"]),
        axis=1,
    )
    work["priority_style"] = work.apply(infer_priority_style, axis=1)
    work["decision_tier"] = work.apply(infer_decision_tier, axis=1)
    work["decision_action"] = work.apply(infer_decision_action, axis=1)
    work["surface_recommendation"] = work.apply(infer_surface_recommendation, axis=1)
    work["governance_note"] = work.apply(infer_governance_note, axis=1)
    work["strategy_card"] = work.apply(infer_strategy_card, axis=1)
    work = work.sort_values(
        ["push_priority", "cross_value_score", "task_fit_score", "decision_confidence"],
        ascending=False,
    ).reset_index(drop=True)
    return work


def merge_layers(value_df: pd.DataFrame, fit_df: pd.DataFrame) -> pd.DataFrame:
    merged = value_df.merge(fit_df, on=["src", "dst", "scope"], how="inner")
    if merged.empty:
        raise ValueError("CrossValueScore 与 TaskFitScore 无法成功对齐，请检查输入层输出。")
    return merged


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


def quadrant_breakdown(frame: pd.DataFrame) -> pd.DataFrame:
    summary = (
        frame.groupby(["quadrant", "decision_tier"], as_index=False)
        .size()
        .rename(columns={"size": "pair_count"})
        .sort_values(["quadrant", "decision_tier"])
        .reset_index(drop=True)
    )
    return summary


def build_summary_markdown(
    output_path: str,
    business_df: pd.DataFrame,
    category_df: pd.DataFrame,
    value_dir: str,
    fit_dir: str,
) -> None:
    top_business = business_df.head(10)[
        [
            "src",
            "dst",
            "push_priority",
            "decision_tier",
            "quadrant",
            "cross_value_score",
            "task_fit_score",
            "surface_recommendation",
        ]
    ].copy()
    strong_push_business = business_df[business_df["decision_tier"] == "STRONG_PUSH"].head(10)[
        [
            "src",
            "dst",
            "push_priority",
            "priority_style",
            "decision_action",
            "surface_recommendation",
        ]
    ].copy()
    guarded_business = business_df[business_df["decision_tier"] == "GUARDED_TEST"].head(10)[
        [
            "src",
            "dst",
            "push_priority",
            "risk_label",
            "governance_note",
            "surface_recommendation",
        ]
    ].copy()
    category_focus = category_df[
        (category_df["decision_tier"].isin(["STRONG_PUSH", "TARGETED_PUSH"]))
        & (category_df["decision_confidence"] >= 0.45)
    ].head(10)[
        [
            "src",
            "dst",
            "push_priority",
            "decision_tier",
            "quadrant",
            "priority_style",
            "surface_recommendation",
        ]
    ].copy()
    category_risky = category_df[
        (category_df["cross_value_score"] >= VALUE_THRESHOLD)
        & (category_df["task_fit_score"] < FIT_THRESHOLD)
        & (category_df["decision_confidence"] >= 0.45)
    ].head(10)[
        [
            "src",
            "dst",
            "cross_value_score",
            "task_fit_score",
            "risk_label",
            "governance_note",
        ]
    ].copy()
    business_quadrants = quadrant_breakdown(business_df)
    category_quadrants = quadrant_breakdown(category_df)

    lines = [
        "# Cross Decision V3 摘要",
        "",
        "## 决策层口径",
        "",
        f"- Value 输入目录：`{value_dir}`",
        f"- Fit 输入目录：`{fit_dir}`",
        "- 本脚本是 V3 的决策合成层，负责把 `CrossValueScore + TaskFitScore` 转成平台动作优先级。",
        "- 它不重算价值层和约束层的底层分项，只消费上一层结果并做门控式融合。",
        "- 决策逻辑：",
        "  - `PushPriority` 采用乘性融合：`Value * Fit * FitGate * RiskPenalty * ConfidenceBoost`",
        "  - `Quadrant` 采用高价值 / 高连续性四象限表达",
        "  - `DecisionTier` 负责把结果翻译成真正可执行的动作层级",
        "",
        "## 业务线 Top 10",
        "",
        to_md_table(top_business),
        "",
        "## 业务线四象限分布",
        "",
        to_md_table(business_quadrants),
        "",
        "## STRONG_PUSH 业务线",
        "",
        to_md_table(strong_push_business) if not strong_push_business.empty else "当前无 STRONG_PUSH 业务线。",
        "",
        "## GUARDED_TEST 业务线",
        "",
        to_md_table(guarded_business) if not guarded_business.empty else "当前无 GUARDED_TEST 业务线。",
        "",
        "## 类目层重点案例",
        "",
        to_md_table(category_focus) if not category_focus.empty else "当前无满足条件的重点类目案例。",
        "",
        "## 类目层高价值但需抑制的案例",
        "",
        to_md_table(category_risky) if not category_risky.empty else "当前无满足条件的高价值风险类目案例。",
        "",
        "## 类目层四象限分布",
        "",
        to_md_table(category_quadrants),
        "",
        "## 解释提醒",
        "",
        "- `PushPriority` 是决策层优先级，不等于真实线上收益承诺。",
        "- 业务线层适合做正式主榜，类目层仍应作为案例库和玩法补充。",
        "- 如果未来补齐 session 级时段、多跳和终局特征，优先增强 `TaskFitScore`，不要直接改动决策层框架。",
        "",
    ]

    with open(output_path, "w", encoding="utf-8") as file_obj:
        file_obj.write("\n".join(lines))


def save_metadata(output_path: str) -> None:
    metadata: Dict[str, object] = {
        "version": "cross_decision_v3",
        "positioning": "decision_synthesis_layer",
        "inputs": {
            "value_layer": "cross_value_v2_outputs/cross_value_{scope}.csv",
            "fit_layer": "task_continuity_v1_outputs/task_continuity_{scope}.csv",
        },
        "formula": {
            "decision_confidence": "sqrt(support_confidence * evidence_confidence)",
            "push_priority": "100 * (cross_value_score / 100) * (task_fit_score / 100) * fit_gate * risk_penalty * (0.85 + 0.15 * decision_confidence)",
        },
        "fit_gate": FIT_GATE_BY_TIER,
        "risk_penalty": RISK_PENALTY,
        "quadrant_thresholds": {
            "value_threshold": VALUE_THRESHOLD,
            "fit_threshold": FIT_THRESHOLD,
        },
        "decision_tiers": {
            "STRONG_PUSH": "Q1 and high confidence with low governance risk",
            "TARGETED_PUSH": "Q1 but still needs directional targeting or承接优化",
            "GUARDED_TEST": "Q2 high value but continuity risk remains",
            "LIGHT_TOUCH": "Q3 reasonable continuity but limited value",
            "SUPPRESS": "Q4 or severe fit suppression",
        },
        "notes": [
            "Decision layer is intentionally downstream-only and should not replace either Value or Task Continuity layers.",
            "Business-line results are the formal report mainboard; category results are filtered case pools.",
            "PushPriority is a platform action proxy rather than a causal incremental estimate.",
        ],
    }
    with open(output_path, "w", encoding="utf-8") as file_obj:
        json.dump(metadata, file_obj, ensure_ascii=False, indent=2)


def main() -> None:
    value_dir = resolve_value_dir(BASE_DIR)
    fit_dir = resolve_fit_dir(BASE_DIR)
    output_dir = resolve_output_dir(BASE_DIR)
    os.makedirs(output_dir, exist_ok=True)

    business_value = load_layer_csv(
        os.path.join(value_dir, "cross_value_business_line.csv"),
        VALUE_COLUMNS,
    )
    category_value = load_layer_csv(
        os.path.join(value_dir, "cross_value_category.csv"),
        VALUE_COLUMNS,
    )
    business_fit = load_layer_csv(
        os.path.join(fit_dir, "task_continuity_business_line.csv"),
        FIT_COLUMNS,
    )
    category_fit = load_layer_csv(
        os.path.join(fit_dir, "task_continuity_category.csv"),
        FIT_COLUMNS,
    )

    business_decision = compute_push_priority(merge_layers(business_value, business_fit))
    category_decision = compute_push_priority(merge_layers(category_value, category_fit))

    business_decision.to_csv(
        os.path.join(output_dir, "cross_decision_business_line.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    category_decision.to_csv(
        os.path.join(output_dir, "cross_decision_category.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    build_summary_markdown(
        os.path.join(output_dir, "cross_decision_summary.md"),
        business_decision,
        category_decision,
        value_dir,
        fit_dir,
    )
    save_metadata(os.path.join(output_dir, "cross_decision_metadata.json"))

    print(f"✅ Cross Decision V3 结果已输出到: {output_dir}")
    print("   - cross_decision_business_line.csv")
    print("   - cross_decision_category.csv")
    print("   - cross_decision_summary.md")
    print("   - cross_decision_metadata.json")


if __name__ == "__main__":
    main()
