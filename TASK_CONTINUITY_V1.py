import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


TIME_SLOTS = [
    "breakfast",
    "lunch",
    "afternoon",
    "dinner",
    "late_night",
    "weekday_daytime",
    "weekend",
]


THEME_PROFILES: Dict[str, Dict[str, object]] = {
    "meal": {
        "planning_depth": 0.15,
        "terminal_strength": 0.92,
        "time_vector": [0.35, 1.00, 0.40, 1.00, 0.70, 0.35, 0.70],
    },
    "beverage": {
        "planning_depth": 0.12,
        "terminal_strength": 0.82,
        "time_vector": [0.30, 0.75, 0.95, 0.80, 0.45, 0.45, 0.75],
    },
    "daily_supply": {
        "planning_depth": 0.22,
        "terminal_strength": 0.84,
        "time_vector": [0.55, 0.75, 0.75, 0.88, 0.55, 0.75, 0.78],
    },
    "local_leisure": {
        "planning_depth": 0.58,
        "terminal_strength": 0.68,
        "time_vector": [0.05, 0.25, 0.75, 0.88, 0.42, 0.42, 1.00],
    },
    "travel": {
        "planning_depth": 0.86,
        "terminal_strength": 0.82,
        "time_vector": [0.08, 0.22, 0.55, 0.78, 0.40, 0.48, 1.00],
    },
    "stay": {
        "planning_depth": 0.82,
        "terminal_strength": 0.94,
        "time_vector": [0.05, 0.18, 0.38, 0.86, 0.62, 0.45, 0.95],
    },
    "beauty_service": {
        "planning_depth": 0.68,
        "terminal_strength": 0.72,
        "time_vector": [0.02, 0.18, 0.78, 0.72, 0.12, 0.72, 0.88],
    },
    "family": {
        "planning_depth": 0.66,
        "terminal_strength": 0.70,
        "time_vector": [0.12, 0.35, 0.82, 0.62, 0.10, 0.52, 1.00],
    },
    "home_service": {
        "planning_depth": 0.78,
        "terminal_strength": 0.74,
        "time_vector": [0.02, 0.15, 0.55, 0.48, 0.05, 0.82, 0.72],
    },
    "electronics": {
        "planning_depth": 0.55,
        "terminal_strength": 0.60,
        "time_vector": [0.02, 0.12, 0.48, 0.50, 0.08, 0.72, 0.68],
    },
    "health": {
        "planning_depth": 0.72,
        "terminal_strength": 0.84,
        "time_vector": [0.08, 0.25, 0.45, 0.38, 0.10, 0.88, 0.70],
    },
    "sports": {
        "planning_depth": 0.58,
        "terminal_strength": 0.66,
        "time_vector": [0.02, 0.12, 0.38, 0.82, 0.15, 0.48, 0.85],
    },
    "education": {
        "planning_depth": 0.83,
        "terminal_strength": 0.62,
        "time_vector": [0.00, 0.05, 0.30, 0.72, 0.05, 0.88, 0.78],
    },
    "pet": {
        "planning_depth": 0.54,
        "terminal_strength": 0.68,
        "time_vector": [0.10, 0.22, 0.55, 0.65, 0.20, 0.62, 0.82],
    },
    "other": {
        "planning_depth": 0.50,
        "terminal_strength": 0.56,
        "time_vector": [0.20, 0.40, 0.50, 0.55, 0.20, 0.55, 0.60],
    },
}


BUSINESS_LINE_PROFILES: Dict[str, Dict[str, object]] = {
    "到家": {
        "theme": "daily_supply",
        "planning_depth": 0.18,
        "terminal_strength": 0.82,
        "time_vector": [0.50, 0.82, 0.72, 0.95, 0.68, 0.70, 0.78],
    },
    "餐饮": {
        "theme": "meal",
        "planning_depth": 0.14,
        "terminal_strength": 0.94,
        "time_vector": [0.38, 1.00, 0.45, 1.00, 0.75, 0.38, 0.72],
    },
    "到店综合": {
        "theme": "local_leisure",
        "planning_depth": 0.55,
        "terminal_strength": 0.68,
        "time_vector": [0.05, 0.24, 0.78, 0.86, 0.35, 0.45, 1.00],
    },
    "酒旅": {
        "theme": "travel",
        "planning_depth": 0.86,
        "terminal_strength": 0.86,
        "time_vector": [0.06, 0.20, 0.55, 0.78, 0.42, 0.50, 1.00],
    },
}


KEYWORD_THEME_RULES: List[Tuple[str, Tuple[str, ...]]] = [
    ("stay", ("酒店", "住宿", "客栈", "民宿", "宾馆", "旅馆", "非标住宿")),
    ("travel", ("酒旅", "旅游", "景区", "景点", "乐园", "主题乐园", "门票", "展览", "博物馆", "动物园", "植物园", "海洋馆", "温泉", "滑雪", "公园", "人文街区")),
    ("local_leisure", ("电影", "KTV", "足疗", "按摩", "洗浴", "电玩", "桌游", "密室", "玩乐", "休闲", "演出")),
    ("meal", ("餐饮", "美食", "小吃", "快餐", "火锅", "烧烤", "夜宵", "早餐", "正餐", "面", "粉", "粥", "汉堡", "披萨")),
    ("beverage", ("饮品", "奶茶", "咖啡", "甜品", "面包", "烘焙")),
    ("daily_supply", ("超市", "便利", "买菜", "生鲜", "水果", "百货", "日化", "粮油", "家清", "日用")),
    ("beauty_service", ("美妆", "丽人", "医美", "美发", "美甲", "美容", "SPA", "养生")),
    ("family", ("亲子", "母婴", "儿童", "少儿")),
    ("home_service", ("家政", "维修", "洗衣", "搬家", "保洁", "装修", "开锁", "回收")),
    ("health", ("医疗", "药", "药店", "买药", "口腔", "体检", "医院")),
    ("sports", ("健身", "运动", "瑜伽", "球馆", "游泳", "体育")),
    ("education", ("教育", "培训", "学习", "驾校")),
    ("pet", ("宠物",)),
    ("electronics", ("数码", "家电", "手机", "电脑")),
]


THEME_COMPATIBILITY_OVERRIDES: Dict[Tuple[str, str], float] = {
    ("meal", "beverage"): 0.95,
    ("beverage", "meal"): 0.95,
    ("meal", "daily_supply"): 0.80,
    ("daily_supply", "meal"): 0.80,
    ("local_leisure", "meal"): 0.78,
    ("meal", "local_leisure"): 0.74,
    ("travel", "stay"): 0.96,
    ("stay", "travel"): 0.96,
    ("travel", "meal"): 0.72,
    ("meal", "travel"): 0.68,
    ("stay", "meal"): 0.76,
    ("meal", "stay"): 0.72,
    ("travel", "local_leisure"): 0.88,
    ("local_leisure", "travel"): 0.88,
    ("family", "travel"): 0.84,
    ("travel", "family"): 0.84,
    ("family", "local_leisure"): 0.86,
    ("local_leisure", "family"): 0.86,
    ("beauty_service", "daily_supply"): 0.58,
    ("daily_supply", "beauty_service"): 0.58,
    ("electronics", "daily_supply"): 0.48,
    ("daily_supply", "electronics"): 0.48,
    ("health", "daily_supply"): 0.64,
    ("daily_supply", "health"): 0.64,
}


TRANSITION_COMPATIBILITY_OVERRIDES: Dict[Tuple[str, str], float] = {
    ("daily_supply", "meal"): 0.86,
    ("meal", "daily_supply"): 0.80,
    ("meal", "beverage"): 0.94,
    ("beverage", "meal"): 0.92,
    ("local_leisure", "meal"): 0.80,
    ("meal", "local_leisure"): 0.72,
    ("travel", "stay"): 0.98,
    ("stay", "travel"): 0.90,
    ("travel", "meal"): 0.84,
    ("meal", "travel"): 0.62,
    ("stay", "meal"): 0.82,
    ("travel", "local_leisure"): 0.86,
    ("local_leisure", "travel"): 0.82,
    ("family", "travel"): 0.86,
    ("family", "local_leisure"): 0.88,
}


BUSINESS_LINE_DIRECTION_OVERRIDES: Dict[Tuple[str, str], float] = {
    ("到家", "餐饮"): 0.90,
    ("餐饮", "到家"): 0.84,
    ("到店综合", "餐饮"): 0.78,
    ("餐饮", "到店综合"): 0.72,
    ("到店综合", "酒旅"): 0.86,
    ("酒旅", "到店综合"): 0.88,
    ("酒旅", "餐饮"): 0.84,
    ("餐饮", "酒旅"): 0.62,
    ("到家", "到店综合"): 0.56,
    ("到店综合", "到家"): 0.60,
    ("到家", "酒旅"): 0.34,
    ("酒旅", "到家"): 0.44,
}


def resolve_input_dir(base_dir: str) -> str:
    env_path = os.getenv("INPUT_DIR")
    if env_path:
        return env_path
    return os.path.join(base_dir, "cross_baseline_v2_outputs")


def resolve_output_dir(base_dir: str) -> str:
    env_path = os.getenv("OUTPUT_DIR")
    if env_path:
        return env_path
    return os.path.join(base_dir, "task_continuity_v1_outputs")


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


def normalize_name(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def infer_theme(name: str, scope: str) -> str:
    clean = normalize_name(name)
    if scope == "business_line" and clean in BUSINESS_LINE_PROFILES:
        return str(BUSINESS_LINE_PROFILES[clean]["theme"])
    for theme, keywords in KEYWORD_THEME_RULES:
        if any(keyword in clean for keyword in keywords):
            return theme
    return "other"


def build_profile(name: str, scope: str) -> Dict[str, object]:
    clean = normalize_name(name)
    if scope == "business_line" and clean in BUSINESS_LINE_PROFILES:
        profile = BUSINESS_LINE_PROFILES[clean].copy()
        profile["theme"] = str(profile["theme"])
        profile["name"] = clean
        return profile

    theme = infer_theme(clean, scope)
    template = THEME_PROFILES.get(theme, THEME_PROFILES["other"])
    return {
        "name": clean,
        "theme": theme,
        "planning_depth": float(template["planning_depth"]),
        "terminal_strength": float(template["terminal_strength"]),
        "time_vector": list(template["time_vector"]),
    }


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    arr_a = np.array(vec_a, dtype="float64")
    arr_b = np.array(vec_b, dtype="float64")
    denom = np.linalg.norm(arr_a) * np.linalg.norm(arr_b)
    if denom == 0:
        return 0.0
    return float(np.dot(arr_a, arr_b) / denom)


def theme_semantic_similarity(src_theme: str, dst_theme: str) -> float:
    if src_theme == dst_theme:
        return 0.92
    if (src_theme, dst_theme) in THEME_COMPATIBILITY_OVERRIDES:
        return THEME_COMPATIBILITY_OVERRIDES[(src_theme, dst_theme)]
    if (dst_theme, src_theme) in THEME_COMPATIBILITY_OVERRIDES:
        return THEME_COMPATIBILITY_OVERRIDES[(dst_theme, src_theme)]

    src_profile = THEME_PROFILES.get(src_theme, THEME_PROFILES["other"])
    dst_profile = THEME_PROFILES.get(dst_theme, THEME_PROFILES["other"])
    planning_proximity = max(
        0.0,
        1.0 - abs(float(src_profile["planning_depth"]) - float(dst_profile["planning_depth"])),
    )
    time_similarity = cosine_similarity(
        list(src_profile["time_vector"]),
        list(dst_profile["time_vector"]),
    )
    return float(np.clip(0.45 * planning_proximity + 0.55 * time_similarity, 0.20, 0.88))


def transition_compatibility(
    src_name: str,
    dst_name: str,
    src_profile: Dict[str, object],
    dst_profile: Dict[str, object],
    scope: str,
) -> Tuple[float, str]:
    if scope == "business_line" and (src_name, dst_name) in BUSINESS_LINE_DIRECTION_OVERRIDES:
        return BUSINESS_LINE_DIRECTION_OVERRIDES[(src_name, dst_name)], "business_override"

    src_theme = str(src_profile["theme"])
    dst_theme = str(dst_profile["theme"])
    if (src_theme, dst_theme) in TRANSITION_COMPATIBILITY_OVERRIDES:
        return TRANSITION_COMPATIBILITY_OVERRIDES[(src_theme, dst_theme)], "theme_override"

    semantic = theme_semantic_similarity(src_theme, dst_theme)
    planning_proximity = max(
        0.0,
        1.0 - abs(float(src_profile["planning_depth"]) - float(dst_profile["planning_depth"])),
    )
    base_score = 0.60 * semantic + 0.40 * planning_proximity

    if float(src_profile["planning_depth"]) + 0.35 < float(dst_profile["planning_depth"]):
        base_score -= 0.08
    if float(src_profile["planning_depth"]) - 0.45 > float(dst_profile["planning_depth"]):
        base_score -= 0.05

    return float(np.clip(base_score, 0.20, 0.92)), "fallback_similarity"


def infer_fit_archetype(row: pd.Series) -> str:
    if row["r_intent"] >= 0.78 and row["r_chain"] >= 0.70 and row["r_terminal"] >= 0.68:
        return "主任务延续型"
    if row["r_semantic"] >= 0.78 and row["r_temporal"] >= 0.74:
        return "场景邻接型"
    if row["planning_gap"] >= 0.35 and row["r_intent"] >= 0.68:
        return "计划升级型"
    if row["r_chain"] >= 0.76 and row["r_temporal"] >= 0.74:
        return "短链路承接型"
    if row["task_fit_score"] < 55:
        return "潜在任务漂移型"
    return "谨慎探索型"


def infer_risk_label(row: pd.Series) -> str:
    if row["r_semantic"] < 0.42 and row["r_intent"] < 0.48:
        return "任务漂移风险"
    if row["r_temporal"] < 0.45:
        return "时段错配风险"
    if row["r_chain"] < 0.50:
        return "链路失真风险"
    if row["r_terminal"] < 0.50:
        return "终局质量风险"
    return "连续性良好"


def infer_fit_tier(row: pd.Series) -> str:
    if (
        row["task_fit_score"] >= 74
        and row["evidence_confidence"] >= 0.45
        and min(row["r_intent"], row["r_chain"], row["r_semantic"]) >= 0.58
    ):
        return "HIGH_FIT"
    if (
        row["task_fit_score"] >= 60
        and row["evidence_confidence"] >= 0.32
        and min(row["r_intent"], row["r_semantic"]) >= 0.45
    ):
        return "MEDIUM_FIT"
    return "LOW_FIT"


def infer_constraint_action(row: pd.Series) -> str:
    if row["fit_tier"] == "HIGH_FIT" and row["risk_label"] == "连续性良好":
        return "可进入 PushPriority 主排序"
    if row["fit_tier"] == "MEDIUM_FIT":
        return "建议与 Value 侧联判后再决定是否放量"
    return "建议在决策层抑制、降权或仅做观察"


def compute_task_continuity(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
    work = ensure_numeric_columns(
        frame,
        [
            "pair_count",
            "unique_users",
            "unique_sessions",
            "conv_sessions",
            "markov_prob",
            "ccr",
            "median_lag_min",
            "p90_lag_min",
            "median_conv_gap_min",
            "p90_conv_gap_min",
        ],
    )
    work["scope"] = scope

    work["rank_markov_signal"] = positive_pct_rank(work["markov_prob"])
    work["rank_ccr_signal"] = positive_pct_rank(work["ccr"])
    work["rank_log_pair_count"] = positive_log_pct_rank(work["pair_count"])
    work["rank_log_unique_users"] = positive_log_pct_rank(work["unique_users"])
    work["rank_log_unique_sessions"] = positive_log_pct_rank(work["unique_sessions"])
    work["rank_log_conv_sessions"] = positive_log_pct_rank(work["conv_sessions"])
    work["rank_fast_median_lag"] = inverse_positive_pct_rank(work["median_lag_min"])
    work["rank_fast_p90_lag"] = inverse_positive_pct_rank(work["p90_lag_min"])
    work["rank_fast_median_conv_gap"] = inverse_positive_pct_rank(work["median_conv_gap_min"])
    work["rank_fast_p90_conv_gap"] = inverse_positive_pct_rank(work["p90_conv_gap_min"])

    rows: List[Dict[str, object]] = []
    for _, row in work.iterrows():
        src_name = normalize_name(row["src"])
        dst_name = normalize_name(row["dst"])
        src_profile = build_profile(src_name, scope)
        dst_profile = build_profile(dst_name, scope)

        r_semantic = theme_semantic_similarity(str(src_profile["theme"]), str(dst_profile["theme"]))
        transition_score, rule_source = transition_compatibility(
            src_name,
            dst_name,
            src_profile,
            dst_profile,
            scope,
        )
        planning_gap = abs(
            float(src_profile["planning_depth"]) - float(dst_profile["planning_depth"])
        )
        planning_proximity = max(0.0, 1.0 - planning_gap)
        r_temporal = cosine_similarity(
            list(src_profile["time_vector"]),
            list(dst_profile["time_vector"]),
        )

        r_intent = float(
            np.clip(
                0.55 * transition_score
                + 0.25 * planning_proximity
                + 0.20 * float(dst_profile["terminal_strength"]),
                0.0,
                1.0,
            )
        )
        r_chain = float(
            np.clip(
                0.50 * row["rank_markov_signal"]
                + 0.20 * row["rank_fast_median_lag"]
                + 0.15 * row["rank_fast_p90_lag"]
                + 0.15 * row["rank_log_pair_count"],
                0.0,
                1.0,
            )
        )
        r_terminal = float(
            np.clip(
                0.45 * float(dst_profile["terminal_strength"])
                + 0.20 * row["rank_fast_median_conv_gap"]
                + 0.15 * row["rank_fast_p90_conv_gap"]
                + 0.20 * row["rank_ccr_signal"],
                0.0,
                1.0,
            )
        )
        evidence_confidence = float(
            np.clip(
                0.35 * row["rank_log_unique_users"]
                + 0.25 * row["rank_log_unique_sessions"]
                + 0.20 * row["rank_log_pair_count"]
                + 0.20 * row["rank_log_conv_sessions"],
                0.0,
                1.0,
            )
        )
        task_fit_score = float(
            100.0
            * (
                0.28 * r_intent
                + 0.22 * r_chain
                + 0.22 * r_semantic
                + 0.14 * r_temporal
                + 0.14 * r_terminal
            )
        )

        rows.append(
            {
                "src": src_name,
                "dst": dst_name,
                "scope": scope,
                "src_theme": src_profile["theme"],
                "dst_theme": dst_profile["theme"],
                "planning_gap": planning_gap,
                "rule_source": rule_source,
                "r_intent": r_intent,
                "r_chain": r_chain,
                "r_semantic": r_semantic,
                "r_temporal": r_temporal,
                "r_terminal": r_terminal,
                "evidence_confidence": evidence_confidence,
                "task_fit_score": task_fit_score,
            }
        )

    score_frame = pd.DataFrame(rows)
    merged = work.merge(score_frame, on=["src", "dst", "scope"], how="left")
    merged["fit_archetype"] = merged.apply(infer_fit_archetype, axis=1)
    merged["risk_label"] = merged.apply(infer_risk_label, axis=1)
    merged["fit_tier"] = merged.apply(infer_fit_tier, axis=1)
    merged["constraint_action"] = merged.apply(infer_constraint_action, axis=1)

    merged = merged.sort_values(
        ["task_fit_score", "r_intent", "r_chain", "r_semantic", "evidence_confidence"],
        ascending=False,
    ).reset_index(drop=True)
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
            "task_fit_score",
            "fit_tier",
            "fit_archetype",
            "risk_label",
            "r_intent",
            "r_chain",
            "r_semantic",
            "r_temporal",
            "r_terminal",
        ]
    ].copy()
    high_fit_business = business_df[business_df["fit_tier"] == "HIGH_FIT"].head(10)[
        [
            "src",
            "dst",
            "task_fit_score",
            "fit_archetype",
            "constraint_action",
        ]
    ].copy()
    risky_business = business_df[business_df["risk_label"] != "连续性良好"].head(10)[
        [
            "src",
            "dst",
            "task_fit_score",
            "risk_label",
            "fit_archetype",
            "constraint_action",
        ]
    ].copy()
    top_category = category_df[category_df["evidence_confidence"] >= 0.45].head(10)[
        [
            "src",
            "dst",
            "task_fit_score",
            "fit_tier",
            "fit_archetype",
            "risk_label",
            "evidence_confidence",
        ]
    ].copy()
    risky_category = category_df[
        (category_df["evidence_confidence"] >= 0.45)
        & (category_df["risk_label"] != "连续性良好")
    ].head(10)[
        [
            "src",
            "dst",
            "task_fit_score",
            "risk_label",
            "fit_archetype",
            "constraint_action",
        ]
    ].copy()

    lines = [
        "# Task Continuity V1 摘要",
        "",
        "## 约束层口径",
        "",
        f"- 输入目录：`{input_dir}`",
        "- 本脚本是 V3 的合理性约束层，不承担 `CrossValueScore` 或最终 `PushPriority` 的职责。",
        "- 它只回答一个问题：`这条 Cross 作为平台动作是否符合任务连续性。`",
        "- 五个分项定义：",
        "  - `R_intent`：任务意图保持，强调方向性任务匹配与规划深度相容性",
        "  - `R_chain`：链路连续性，强调 Markov 顺路性与短链路直接性",
        "  - `R_semantic`：场景语义邻接度，基于业务线/类目主题规则",
        "  - `R_temporal`：时段匹配度，基于场景时间画像的相似度",
        "  - `R_terminal`：终局质量代理，结合目的地终局强度与转化窗口收敛速度",
        "- 最终得分：`TaskFitScore = 100 * (0.28*R_intent + 0.22*R_chain + 0.22*R_semantic + 0.14*R_temporal + 0.14*R_terminal)`",
        "- 额外诊断项：",
        "  - `evidence_confidence`：证据强度，避免长尾弱样本把约束层结果讲重",
        "  - `risk_label`：主要风险来源，帮助后续决策层做抑制或降权",
        "",
        "## 业务线 Top 10",
        "",
        to_md_table(top_business),
        "",
        "## HIGH_FIT 业务线",
        "",
        to_md_table(high_fit_business) if not high_fit_business.empty else "当前无 HIGH_FIT 业务线。",
        "",
        "## 业务线风险提示",
        "",
        to_md_table(risky_business) if not risky_business.empty else "当前业务线结果暂无明显风险项。",
        "",
        "## 类目层高连续性案例",
        "",
        to_md_table(top_category) if not top_category.empty else "当前无满足证据强度要求的高连续性类目案例。",
        "",
        "## 类目层风险案例",
        "",
        to_md_table(risky_category) if not risky_category.empty else "当前无满足证据强度要求的风险类目案例。",
        "",
        "## 解释提醒",
        "",
        "- `TaskFitScore` 不是价值分，不应用来替代 `CrossValueScore`。",
        "- `Task Continuity V1` 当前是规则先验 + baseline 代理信号版本，后续仍可接 session 级时段、终局和多跳特征继续增强。",
        "- 在 V3 中，它应该和 `CrossValueScore` 联合使用，而不是单独作为最终平台动作排序依据。",
        "",
    ]

    with open(output_path, "w", encoding="utf-8") as file_obj:
        file_obj.write("\n".join(lines))


def save_metadata(output_path: str) -> None:
    metadata: Dict[str, object] = {
        "version": "task_continuity_v1",
        "positioning": "constraint_only_layer",
        "formula": {
            "r_intent": "0.55 * transition_compatibility + 0.25 * planning_proximity + 0.20 * dst_terminal_strength",
            "r_chain": "0.50 * rank(markov_prob) + 0.20 * inv_rank(median_lag_min) + 0.15 * inv_rank(p90_lag_min) + 0.15 * rank(log1p(pair_count))",
            "r_semantic": "theme-level semantic similarity with business-line/category overrides",
            "r_temporal": "cosine_similarity(src_time_profile, dst_time_profile)",
            "r_terminal": "0.45 * dst_terminal_strength + 0.20 * inv_rank(median_conv_gap_min) + 0.15 * inv_rank(p90_conv_gap_min) + 0.20 * rank(ccr)",
            "task_fit_score": "100 * (0.28 * r_intent + 0.22 * r_chain + 0.22 * r_semantic + 0.14 * r_temporal + 0.14 * r_terminal)",
            "evidence_confidence": "0.35 * rank(log1p(unique_users)) + 0.25 * rank(log1p(unique_sessions)) + 0.20 * rank(log1p(pair_count)) + 0.20 * rank(log1p(conv_sessions))",
        },
        "tier_rules": {
            "HIGH_FIT": "task_fit_score >= 74 and evidence_confidence >= 0.45 and min(r_intent, r_chain, r_semantic) >= 0.58",
            "MEDIUM_FIT": "task_fit_score >= 60 and evidence_confidence >= 0.32 and min(r_intent, r_semantic) >= 0.45",
            "LOW_FIT": "otherwise",
        },
        "notes": [
            "This layer is intentionally a constraint layer and does not replace CrossValueScore.",
            "V1 uses domain priors and baseline proxy signals rather than session-level multi-hop supervision.",
            "Business-line exact transition overrides are included because the project only has a small directed pair space at that layer.",
        ],
        "time_slots": TIME_SLOTS,
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

    scored_business = compute_task_continuity(business_df, scope="business_line")
    scored_category = compute_task_continuity(category_df, scope="category")

    scored_business.to_csv(
        os.path.join(output_dir, "task_continuity_business_line.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    scored_category.to_csv(
        os.path.join(output_dir, "task_continuity_category.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    build_summary_markdown(
        os.path.join(output_dir, "task_continuity_summary.md"),
        scored_business,
        scored_category,
        input_dir,
    )
    save_metadata(os.path.join(output_dir, "task_continuity_metadata.json"))

    print(f"✅ Task Continuity V1 结果已输出到: {output_dir}")
    print("   - task_continuity_business_line.csv")
    print("   - task_continuity_category.csv")
    print("   - task_continuity_summary.md")
    print("   - task_continuity_metadata.json")


if __name__ == "__main__":
    main()
