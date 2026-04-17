import json
import os
from typing import Dict, List

import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


DECISION_COLUMNS = [
    "src",
    "dst",
    "scope",
    "cross_value_score",
    "task_fit_score",
    "push_priority",
    "decision_confidence",
    "quadrant",
    "decision_tier",
    "priority_style",
    "decision_action",
    "surface_recommendation",
    "governance_note",
    "strategy_card",
    "risk_label",
    "value_archetype",
    "fit_archetype",
    "dominant_driver",
    "destination_profile",
    "support_confidence",
    "evidence_confidence",
    "v_seq",
    "v_conv",
    "v_time",
    "r_chain",
    "r_temporal",
    "r_terminal",
]


SURFACE_OWNER_MAP = {
    "首页承接": "首页/推荐运营",
    "支付后推荐": "交易承接运营",
    "高频召回": "召回运营",
    "会场联动": "活动/频道运营",
    "搜索联动": "搜索运营",
    "内容种草": "内容运营",
    "定向召回": "召回运营",
    "小流量触达": "增长实验运营",
    "补充流量位": "频道补充运营",
    "弱曝光提醒": "召回/提醒运营",
    "暂不建议主动曝光": "策略观察",
}


EXECUTION_PRIORITY_ORDER = {
    "P0": 0,
    "P1": 1,
    "P2": 2,
    "P3": 3,
    "HOLD": 4,
}


def resolve_decision_dir(base_dir: str) -> str:
    env_path = os.getenv("DECISION_DIR")
    if env_path:
        return env_path
    return os.path.join(base_dir, "cross_decision_v3_outputs")


def resolve_output_dir(base_dir: str) -> str:
    env_path = os.getenv("OUTPUT_DIR")
    if env_path:
        return env_path
    return os.path.join(base_dir, "strategy_engine_v3_outputs")


def load_decision_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到所需输入文件：{path}")
    frame = pd.read_csv(path)
    missing = [column for column in DECISION_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"文件字段缺失：{path} -> {missing}")
    return frame[DECISION_COLUMNS].copy()


def split_surfaces(recommendation: str) -> List[str]:
    text = str(recommendation).strip()
    if not text or text == "nan":
        return ["暂不建议主动曝光"]
    if text == "暂不建议主动曝光":
        return [text]
    return [part.strip() for part in text.split("/") if part.strip()]


def infer_playbook_track(row: pd.Series) -> str:
    if row["decision_tier"] == "STRONG_PUSH" and row["priority_style"] == "强承接推进型":
        return "即时承接增长"
    if row["priority_style"] == "协同联动型":
        return "协同联动增长"
    if row["decision_tier"] == "TARGETED_PUSH":
        return "定向放量增长"
    if row["decision_tier"] == "GUARDED_TEST":
        return "带约束试验"
    if row["decision_tier"] == "LIGHT_TOUCH":
        return "轻触达渗透"
    return "保守观察"


def infer_execution_priority(row: pd.Series) -> str:
    if row["decision_tier"] == "STRONG_PUSH" and row["push_priority"] >= 60:
        return "P0"
    if row["decision_tier"] in {"STRONG_PUSH", "TARGETED_PUSH"} and row["push_priority"] >= 42:
        return "P1"
    if row["decision_tier"] in {"TARGETED_PUSH", "GUARDED_TEST"}:
        return "P2"
    if row["decision_tier"] == "LIGHT_TOUCH":
        return "P3"
    return "HOLD"


def infer_primary_objective(row: pd.Series) -> str:
    if row["priority_style"] == "强承接推进型":
        return f"缩短 {row['src']} -> {row['dst']} 的成交承接链路"
    if row["priority_style"] == "协同联动型":
        return f"放大 {row['src']} -> {row['dst']} 的跨场景协同转化"
    if row["decision_tier"] == "GUARDED_TEST":
        return f"验证 {row['src']} -> {row['dst']} 的价值能否在约束条件下释放"
    if row["decision_tier"] == "LIGHT_TOUCH":
        return f"以低成本补充触达验证 {row['src']} -> {row['dst']} 潜力"
    return f"持续监测 {row['src']} -> {row['dst']} 的自然协同机会"


def infer_trigger_rule(row: pd.Series) -> str:
    if row["decision_tier"] == "SUPPRESS":
        return "当前不建议主动触发"
    if row["risk_label"] == "时段错配风险":
        return "仅在高匹配时段和明确上游意图时触发"
    if row["priority_style"] == "强承接推进型":
        if row["v_time"] >= 0.72 or row["r_chain"] >= 0.75:
            return "在 src 行为后短窗口内同 session 触发"
        return "在 src 行为后同 session 或支付后承接触发"
    if row["priority_style"] == "协同联动型":
        return "围绕 src 主题会场、搜索词或专题页触发"
    if row["decision_tier"] == "GUARDED_TEST":
        return "仅在白名单场景、限定入口和收窄时机下触发"
    if row["decision_tier"] == "LIGHT_TOUCH":
        return "低频提醒或补充位触发"
    return "在明确上游行为后按规则触发"


def infer_audience_rule(row: pd.Series) -> str:
    if row["decision_tier"] == "SUPPRESS":
        return "不进入主动投放人群"
    if row["risk_label"] == "任务漂移风险":
        return "仅限对 dst 有历史兴趣或强交易信号的人群"
    if row["risk_label"] == "时段错配风险":
        return "限定匹配时段且限定高意图用户"
    if row["priority_style"] == "强承接推进型":
        return "限定最近发生 src 行为的高意图用户"
    if row["priority_style"] == "协同联动型":
        return "src 高兴趣用户 + 相邻主题偏好人群"
    if row["decision_tier"] == "LIGHT_TOUCH":
        return "白名单低频曝光人群"
    return "定向人群灰度触达"


def infer_landing_focus(row: pd.Series) -> str:
    if row["decision_tier"] == "SUPPRESS":
        return "无需配置主动承接页"
    if row["priority_style"] == "强承接推进型":
        return "短链路承接页 + 明确下单入口"
    if row["priority_style"] == "协同联动型":
        return "会场聚合页 + 组合权益承接"
    if row["decision_tier"] == "GUARDED_TEST":
        return "带过滤条件的试验承接页"
    if row["decision_tier"] == "LIGHT_TOUCH":
        return "轻量内容卡片或补充位承接"
    return "常规主题承接页"


def infer_experiment_plan(row: pd.Series) -> str:
    if row["decision_tier"] == "STRONG_PUSH":
        return "按城市/人群分层放量，验证主资源位收益提升"
    if row["decision_tier"] == "TARGETED_PUSH":
        return "做定向 A/B，测试流量面、文案和承接页组合"
    if row["decision_tier"] == "GUARDED_TEST":
        return "做入口收窄 + 时段门控 + 白名单 A/B"
    if row["decision_tier"] == "LIGHT_TOUCH":
        return "低频灰度验证补充曝光价值"
    return "先不进入实验池，仅做观察"


def infer_success_metric(row: pd.Series) -> str:
    if row["priority_style"] == "强承接推进型":
        return "同 session 下单率 / 支付后承接点击转化率"
    if row["priority_style"] == "协同联动型":
        return "跨场景进入率 / 会场转化率 / 二跳转化率"
    if row["decision_tier"] == "GUARDED_TEST":
        return "任务完成率不降前提下的增量点击与转化"
    if row["decision_tier"] == "LIGHT_TOUCH":
        return "补充曝光点击率 / 二跳进入率"
    return "自然进入率 / 自然转化率"


def infer_guardrail_metric(row: pd.Series) -> str:
    if row["risk_label"] == "任务漂移风险":
        return "会话退出率 / 主任务流失率"
    if row["risk_label"] == "时段错配风险":
        return "非匹配时段点击后退出率"
    if row["risk_label"] == "链路失真风险":
        return "多跳扩散率 / 长链无单率"
    if row["risk_label"] == "终局质量风险":
        return "承接页跳失率 / 无单率"
    return "点击后转化时延 / 会话终局质量"


def infer_rollout_phase(row: pd.Series) -> str:
    if row["decision_tier"] == "STRONG_PUSH" and row["scope"] == "business_line":
        return "平台级重点推进"
    if row["decision_tier"] in {"STRONG_PUSH", "TARGETED_PUSH"} and row["scope"] == "category":
        return "类目案例化推进"
    if row["decision_tier"] == "TARGETED_PUSH":
        return "定向放量"
    if row["decision_tier"] == "GUARDED_TEST":
        return "小流量试验"
    if row["decision_tier"] == "LIGHT_TOUCH":
        return "补充运营"
    return "观察归档"


def infer_owner_team(primary_surface: str) -> str:
    return SURFACE_OWNER_MAP.get(primary_surface, "综合策略运营")


def infer_strategy_brief(row: pd.Series) -> str:
    return (
        f"{row['playbook_track']}：以{row['primary_surface']}为主，"
        f"面向{row['audience_rule']}，重点看{row['success_metric']}，"
        f"同时盯住{row['guardrail_metric']}。"
    )


def build_strategy_frame(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["surface_list"] = work["surface_recommendation"].apply(split_surfaces)
    work["primary_surface"] = work["surface_list"].apply(lambda items: items[0] if items else "暂不建议主动曝光")
    work["secondary_surface"] = work["surface_list"].apply(
        lambda items: items[1] if len(items) > 1 else ""
    )
    work["playbook_track"] = work.apply(infer_playbook_track, axis=1)
    work["execution_priority"] = work.apply(infer_execution_priority, axis=1)
    work["primary_objective"] = work.apply(infer_primary_objective, axis=1)
    work["trigger_rule"] = work.apply(infer_trigger_rule, axis=1)
    work["audience_rule"] = work.apply(infer_audience_rule, axis=1)
    work["landing_focus"] = work.apply(infer_landing_focus, axis=1)
    work["experiment_plan"] = work.apply(infer_experiment_plan, axis=1)
    work["success_metric"] = work.apply(infer_success_metric, axis=1)
    work["guardrail_metric"] = work.apply(infer_guardrail_metric, axis=1)
    work["rollout_phase"] = work.apply(infer_rollout_phase, axis=1)
    work["owner_team"] = work["primary_surface"].apply(infer_owner_team)
    work["strategy_brief"] = work.apply(infer_strategy_brief, axis=1)
    work["strategy_card_id"] = work.apply(
        lambda row: f"{row['scope']}::{row['src']}->{row['dst']}",
        axis=1,
    )
    work["execution_priority_order"] = work["execution_priority"].map(EXECUTION_PRIORITY_ORDER).fillna(9)
    work = work.sort_values(
        ["execution_priority_order", "push_priority", "decision_confidence"],
        ascending=[True, False, False],
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


def summary_by_owner(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["owner_team", "execution_priority"], as_index=False)
        .size()
        .rename(columns={"size": "pair_count"})
        .sort_values(["execution_priority", "pair_count", "owner_team"], ascending=[True, False, True])
        .reset_index(drop=True)
    )


def build_card_json(frame: pd.DataFrame, top_n: int = 20) -> List[Dict[str, object]]:
    columns = [
        "strategy_card_id",
        "src",
        "dst",
        "scope",
        "execution_priority",
        "playbook_track",
        "decision_tier",
        "push_priority",
        "primary_surface",
        "secondary_surface",
        "owner_team",
        "primary_objective",
        "trigger_rule",
        "audience_rule",
        "landing_focus",
        "experiment_plan",
        "success_metric",
        "guardrail_metric",
        "rollout_phase",
        "strategy_brief",
        "governance_note",
    ]
    return frame.head(top_n)[columns].to_dict(orient="records")


def save_json(path: str, payload: object) -> None:
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)


def build_summary_markdown(
    output_path: str,
    business_df: pd.DataFrame,
    category_df: pd.DataFrame,
    decision_dir: str,
) -> None:
    top_business = business_df.head(10)[
        [
            "src",
            "dst",
            "execution_priority",
            "playbook_track",
            "primary_surface",
            "owner_team",
            "rollout_phase",
            "strategy_brief",
        ]
    ].copy()
    p0_p1_business = business_df[business_df["execution_priority"].isin(["P0", "P1"])].head(10)[
        [
            "src",
            "dst",
            "execution_priority",
            "playbook_track",
            "primary_objective",
            "experiment_plan",
            "success_metric",
        ]
    ].copy()
    guarded_tests = business_df[business_df["decision_tier"] == "GUARDED_TEST"].head(10)[
        [
            "src",
            "dst",
            "risk_label",
            "trigger_rule",
            "guardrail_metric",
            "governance_note",
        ]
    ].copy()
    category_showcase = category_df[
        category_df["execution_priority"].isin(["P0", "P1", "P2"])
    ].head(10)[
        [
            "src",
            "dst",
            "execution_priority",
            "playbook_track",
            "primary_surface",
            "experiment_plan",
            "strategy_brief",
        ]
    ].copy()
    watchlist = category_df[category_df["execution_priority"] == "HOLD"].head(10)[
        [
            "src",
            "dst",
            "risk_label",
            "governance_note",
            "rollout_phase",
        ]
    ].copy()
    business_owner_summary = summary_by_owner(business_df)

    lines = [
        "# Strategy Engine V3 摘要",
        "",
        "## 策略层口径",
        "",
        f"- 决策层输入目录：`{decision_dir}`",
        "- 本脚本是 V3 的策略输出层，不重算 Value / Fit / Decision 分数。",
        "- 它负责把 `PushPriority + DecisionTier + Risk` 翻译成可执行的运营打法卡。",
        "- 输出重点包括：",
        "  - `execution_priority`：运营执行优先级",
        "  - `playbook_track`：打法类型",
        "  - `primary_surface / secondary_surface`：推荐流量面",
        "  - `trigger_rule / audience_rule / landing_focus`：触发与承接规则",
        "  - `experiment_plan / success_metric / guardrail_metric`：实验与监控方案",
        "",
        "## 业务线 Top 策略卡",
        "",
        to_md_table(top_business),
        "",
        "## P0 / P1 业务线打法",
        "",
        to_md_table(p0_p1_business) if not p0_p1_business.empty else "当前无 P0 / P1 业务线打法。",
        "",
        "## 业务线需要带约束试验的案例",
        "",
        to_md_table(guarded_tests) if not guarded_tests.empty else "当前无带约束试验案例。",
        "",
        "## 类目层重点玩法案例",
        "",
        to_md_table(category_showcase) if not category_showcase.empty else "当前无重点类目玩法案例。",
        "",
        "## 类目层观察清单",
        "",
        to_md_table(watchlist) if not watchlist.empty else "当前无观察清单案例。",
        "",
        "## 业务线 Owner 分布",
        "",
        to_md_table(business_owner_summary),
        "",
        "## 解释提醒",
        "",
        "- 业务线层是正式主打法清单，类目层更适合作为案例库和精细化运营玩法。",
        "- `execution_priority` 是运营执行优先级，不是因果收益承诺。",
        "- 若后续增强 session 级上下文，优先迭代 `trigger_rule` 和 `audience_rule`，而不是重写整套策略层结构。",
        "",
    ]

    with open(output_path, "w", encoding="utf-8") as file_obj:
        file_obj.write("\n".join(lines))


def save_metadata(output_path: str) -> None:
    metadata: Dict[str, object] = {
        "version": "strategy_engine_v3",
        "positioning": "strategy_output_layer",
        "input": "cross_decision_v3_outputs/cross_decision_{scope}.csv",
        "execution_priority_order": EXECUTION_PRIORITY_ORDER,
        "surface_owner_map": SURFACE_OWNER_MAP,
        "notes": [
            "Strategy layer is downstream-only and should not mutate the decision layer scores.",
            "Business-line strategies are the formal playbook; category strategies are case-based supplements.",
            "This layer converts scoring outputs into operational hypotheses and experiment plans.",
        ],
    }
    save_json(output_path, metadata)


def main() -> None:
    decision_dir = resolve_decision_dir(BASE_DIR)
    output_dir = resolve_output_dir(BASE_DIR)
    os.makedirs(output_dir, exist_ok=True)

    business_decision = load_decision_csv(
        os.path.join(decision_dir, "cross_decision_business_line.csv")
    )
    category_decision = load_decision_csv(
        os.path.join(decision_dir, "cross_decision_category.csv")
    )

    business_strategy = build_strategy_frame(business_decision)
    category_strategy = build_strategy_frame(category_decision)

    business_strategy.to_csv(
        os.path.join(output_dir, "strategy_engine_business_line.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    category_strategy.to_csv(
        os.path.join(output_dir, "strategy_engine_category.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    save_json(
        os.path.join(output_dir, "strategy_cards_business_line.json"),
        build_card_json(business_strategy),
    )
    save_json(
        os.path.join(output_dir, "strategy_cards_category.json"),
        build_card_json(category_strategy),
    )
    build_summary_markdown(
        os.path.join(output_dir, "strategy_engine_summary.md"),
        business_strategy,
        category_strategy,
        decision_dir,
    )
    save_metadata(os.path.join(output_dir, "strategy_engine_metadata.json"))

    print(f"✅ Strategy Engine V3 结果已输出到: {output_dir}")
    print("   - strategy_engine_business_line.csv")
    print("   - strategy_engine_category.csv")
    print("   - strategy_cards_business_line.json")
    print("   - strategy_cards_category.json")
    print("   - strategy_engine_summary.md")
    print("   - strategy_engine_metadata.json")


if __name__ == "__main__":
    main()
