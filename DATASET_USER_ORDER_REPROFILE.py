import json
import os
import time
from typing import Dict, List

import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_data_path(base_dir: str) -> str:
    env_path = os.getenv("DATA_PATH")
    if env_path:
        return env_path
    candidates = [
        "/content/drive/MyDrive/Colab Notebooks/view_data.csv",
        "/content/drive/MyDrive/view_data.csv",
        "/content/view_data.csv",
        os.path.join(base_dir, "view_data.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[-1]


def confidence_bucket(gap_ms: float, same_session: bool, has_prev: bool) -> str:
    if not has_prev or pd.isna(gap_ms):
        return "UNATTRIBUTED"
    if same_session and gap_ms <= 60_000:
        return "HIGH"
    if same_session and gap_ms <= 300_000:
        return "MEDIUM"
    if gap_ms <= 600_000:
        return "LOW"
    return "UNCERTAIN"


def business_line_from_cate(cate: str) -> str:
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


def quantile_dict(values: pd.Series) -> Dict[str, float]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return {}
    return {
        "p50_ms": float(values.quantile(0.50)),
        "p90_ms": float(values.quantile(0.90)),
        "p95_ms": float(values.quantile(0.95)),
        "p99_ms": float(values.quantile(0.99)),
        "max_ms": float(values.max()),
    }


def write_summary_markdown(
    output_path: str,
    profile: Dict[str, object],
    order_profile: Dict[str, object],
    top_city_users: pd.DataFrame,
    top_device_users: pd.DataFrame,
    top_order_cates: pd.DataFrame,
    resolved_cate_col: str,
) -> None:
    def to_md_table(frame: pd.DataFrame) -> str:
        headers = [str(col) for col in frame.columns]
        rows = [[str(v) for v in row] for row in frame.fillna("").values.tolist()]
        sep = "| " + " | ".join(["---"] * len(headers)) + " |"
        out = ["| " + " | ".join(headers) + " |", sep]
        for row in rows:
            out.append("| " + " | ".join(row) + " |")
        return "\n".join(out)

    lines: List[str] = []
    lines.append("# 数据集重画像与 ORDER 前序归因总结")
    lines.append("")
    lines.append("## 核心结论")
    lines.append("")
    lines.append(
        f"- 总日志量为 `{profile['total_rows']:,}`，其中 `ORDER` 日志 `{profile['order_rows']:,}`，占比 `{profile['order_rate']:.2%}`。"
    )
    lines.append(
        f"- 总用户数 `{profile['total_users']:,}`，其中发生过至少一次 `ORDER` 的用户 `{profile['users_with_order']:,}`，占比 `{profile['user_conversion_rate']:.2%}`。"
    )
    lines.append(
        f"- 原始 `ORDER` 自身携带明确 `poi_id` 的数量为 `{profile['orders_with_poi']:,}`，携带明确品类的数量为 `{profile['orders_with_explicit_cate']:,}`；因此后续分析必须依赖前序归因。"
    )
    lines.append(
        f"- 采用“同一用户时间排序后的最近明确页面行为”对 `ORDER` 做严格前序归因后，可归因订单 `{order_profile['attributed_orders']:,}`，占全部 `ORDER` 的 `{order_profile['attributed_rate']:.2%}`。"
    )
    lines.append(
        f"- `ORDER` 到前一个明确行为的时间差中位数为 `{order_profile['gap_p50_sec']:.1f}` 秒，`90%` 位为 `{order_profile['gap_p90_sec']:.1f}` 秒，说明 5 分钟内前序归因具有较强现实基础。"
    )
    lines.append("")
    lines.append("## 用户与会话画像")
    lines.append("")
    lines.append(
        f"- 总会话数 `{profile['total_sessions']:,}`，发生过 `ORDER` 的会话 `{profile['sessions_with_order']:,}`，占比 `{profile['session_conversion_rate']:.2%}`。"
    )
    lines.append(
        f"- 单城市用户占比 `{profile['single_city_user_rate']:.2%}`，多城市用户占比 `{profile['multi_city_user_rate']:.2%}`。"
    )
    lines.append("")
    lines.append("### 用户城市 Top 10")
    lines.append("")
    lines.append(to_md_table(top_city_users))
    lines.append("")
    lines.append("### 用户设备 Top 10")
    lines.append("")
    lines.append(to_md_table(top_device_users))
    lines.append("")
    lines.append("## ORDER 前序归因结果")
    lines.append("")
    lines.append(
        f"- 同 session 归因占比 `{order_profile['same_session_rate']:.2%}`。"
    )
    lines.append(
        f"- 高/中/低/不确定 归因分布：`{order_profile['confidence_distribution']}`。"
    )
    lines.append("")
    lines.append("### 归因后 ORDER 品类 Top 20")
    lines.append("")
    lines.append(to_md_table(top_order_cates))
    lines.append("")
    lines.append("## 说明")
    lines.append("")
    lines.append(
        "- 本文档中的“归因订单品类”采用严格前序归因：对缺失品类的 `ORDER`，归因给同一用户时间排序下最近的明确 `PV/MC` 页面行为。"
    )
    lines.append(
        f"- 后续 Lift / CCR / Markov 的重构建议，应统一以本次产出的 `{resolved_cate_col}` 作为新的订单品类口径。"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    start = time.time()
    data_path = resolve_data_path(BASE_DIR)
    artifact_root = os.path.dirname(data_path) if data_path.startswith("/content/") else BASE_DIR
    strict_same_session = os.getenv("STRICT_SAME_SESSION", "1") == "1"
    version_tag = "v1.1" if strict_same_session else "v1"
    default_output_dir = (
        os.path.join(artifact_root, "dataset_reprofile_outputs_v1_1")
        if strict_same_session
        else os.path.join(artifact_root, "dataset_reprofile_outputs")
    )
    default_output_csv = (
        os.path.join(artifact_root, "view_data.v1.1.csv")
        if strict_same_session
        else os.path.join(artifact_root, "view_data.v1.csv")
    )
    output_dir = os.getenv("OUTPUT_DIR", default_output_dir)
    output_csv = os.getenv("OUTPUT_CSV", default_output_csv)
    chunksize = int(os.getenv("WRITE_CHUNKSIZE", "200000"))
    write_augmented = os.getenv("WRITE_AUGMENTED_CSV", "1") == "1"
    resolved_cate_col = f"resolved_first_cate_name_{version_tag.replace('.', '_')}"
    resolved_business_col = f"resolved_business_line_{version_tag.replace('.', '_')}"

    os.makedirs(output_dir, exist_ok=True)

    print("🚀 [Dataset Reprofile] 开始重新认识数据集并执行 ORDER 前序归因...")
    print(f"📂 数据路径: {data_path}")
    print(f"📁 输出目录: {output_dir}")
    print(f"📝 增强数据输出: {output_csv}")
    print(f"🔒 严格同 session 归因: {strict_same_session}")

    usecols = [
        "row_key",
        "user_id",
        "session_id",
        "event_timestamp",
        "event_type",
        "page_city_name",
        "device_type",
        "poi_id",
        "poi_name",
        "first_cate_name",
        "page_id",
        "page_name",
    ]
    df = pd.read_csv(data_path, usecols=usecols)

    df["event_timestamp"] = pd.to_numeric(df["event_timestamp"], errors="coerce")
    df = df.dropna(subset=["row_key", "user_id", "session_id", "event_timestamp"]).copy()
    df["event_timestamp"] = df["event_timestamp"].astype("int64")
    df["event_type"] = df["event_type"].astype(str).str.strip().str.upper()
    df["first_cate_name"] = df["first_cate_name"].fillna("未知品类").astype(str)
    df["page_city_name"] = df["page_city_name"].fillna("未知城市").astype(str)
    df["device_type"] = df["device_type"].fillna("未知设备").astype(str)

    is_order = df["event_type"] == "ORDER"
    known_cate_mask = df["first_cate_name"].notna() & (df["first_cate_name"] != "未知品类") & (df["first_cate_name"] != "nan")

    profile: Dict[str, object] = {
        "total_rows": int(len(df)),
        "order_rows": int(is_order.sum()),
        "order_rate": float(is_order.mean()),
        "total_users": int(df["user_id"].nunique()),
        "total_sessions": int(df["session_id"].nunique()),
        "orders_with_poi": int(df.loc[is_order, "poi_id"].notna().sum()),
        "orders_with_explicit_cate": int((is_order & known_cate_mask).sum()),
    }

    users_with_order = df.loc[is_order, "user_id"].drop_duplicates()
    sessions_with_order = df.loc[is_order, "session_id"].drop_duplicates()
    profile["users_with_order"] = int(users_with_order.shape[0])
    profile["users_without_order"] = int(profile["total_users"] - profile["users_with_order"])
    profile["user_conversion_rate"] = float(profile["users_with_order"] / profile["total_users"])
    profile["sessions_with_order"] = int(sessions_with_order.shape[0])
    profile["sessions_without_order"] = int(profile["total_sessions"] - profile["sessions_with_order"])
    profile["session_conversion_rate"] = float(profile["sessions_with_order"] / profile["total_sessions"])

    user_city_nunique = (
        df.loc[df["page_city_name"] != "未知城市", ["user_id", "page_city_name"]]
        .drop_duplicates()
        .groupby("user_id")["page_city_name"]
        .nunique()
    )
    profile["single_city_users"] = int((user_city_nunique <= 1).sum())
    profile["multi_city_users"] = int((user_city_nunique > 1).sum())
    denom_city = max(len(user_city_nunique), 1)
    profile["single_city_user_rate"] = float(profile["single_city_users"] / denom_city)
    profile["multi_city_user_rate"] = float(profile["multi_city_users"] / denom_city)

    top_city_users = (
        df[["user_id", "page_city_name"]]
        .drop_duplicates()
        .groupby("page_city_name")["user_id"]
        .nunique()
        .sort_values(ascending=False)
        .head(10)
        .reset_index(name="user_count")
    )
    top_city_rows = (
        df["page_city_name"].value_counts().head(10).rename_axis("page_city_name").reset_index(name="row_count")
    )
    top_city_stats = top_city_users.merge(top_city_rows, on="page_city_name", how="outer").fillna(0)
    top_city_stats.to_csv(
        os.path.join(output_dir, "top_city_stats.csv"), index=False, encoding="utf-8-sig"
    )

    top_device_users = (
        df[["user_id", "device_type"]]
        .drop_duplicates()
        .groupby("device_type")["user_id"]
        .nunique()
        .sort_values(ascending=False)
        .reset_index(name="user_count")
    )
    top_device_rows = (
        df["device_type"].value_counts().rename_axis("device_type").reset_index(name="row_count")
    )
    top_device_stats = top_device_users.merge(top_device_rows, on="device_type", how="outer").fillna(0)
    top_device_stats.to_csv(
        os.path.join(output_dir, "device_distribution.csv"), index=False, encoding="utf-8-sig"
    )

    print("🔄 正在按 user_id + event_timestamp 重排并构建前序显式行为索引...")
    df = df.sort_values(["user_id", "event_timestamp", "row_key"]).reset_index(drop=True)

    is_order_sorted = df["event_type"] == "ORDER"
    known_cate_mask_sorted = (
        df["first_cate_name"].notna()
        & (df["first_cate_name"] != "未知品类")
        & (df["first_cate_name"] != "nan")
    )
    explicit_mask = (~is_order_sorted) & known_cate_mask_sorted
    row_positions = np.arange(len(df), dtype=np.int64)
    explicit_pos = pd.Series(np.where(explicit_mask.to_numpy(), row_positions, np.nan), index=df.index)
    group_keys = [df["user_id"], df["session_id"]] if strict_same_session else [df["user_id"]]
    prev_explicit_pos = explicit_pos.groupby(group_keys).ffill()

    order_df = df.loc[is_order_sorted, ["row_key", "user_id", "session_id", "event_timestamp"]].copy()
    prev_pos_order = prev_explicit_pos.loc[is_order_sorted]
    order_df["has_prev_explicit"] = prev_pos_order.notna().to_numpy()

    valid_prev = prev_pos_order.dropna().astype("int64")
    prev_lookup = df.loc[
        valid_prev.to_numpy(),
        [
            "session_id",
            "event_timestamp",
            "event_type",
            "page_city_name",
            "device_type",
            "poi_id",
            "poi_name",
            "first_cate_name",
            "page_id",
            "page_name",
        ],
    ].reset_index(drop=True)

    order_attr = order_df.reset_index(drop=True)
    order_attr["order_attr_prev_session_id"] = pd.NA
    order_attr["order_attr_prev_timestamp"] = pd.NA
    order_attr["order_attr_prev_event_type"] = pd.NA
    order_attr["order_attr_prev_city"] = pd.NA
    order_attr["order_attr_prev_device_type"] = pd.NA
    order_attr["order_attr_prev_poi_id"] = pd.NA
    order_attr["order_attr_prev_poi_name"] = pd.NA
    order_attr["order_attr_prev_cate"] = pd.NA
    order_attr["order_attr_prev_page_id"] = pd.NA
    order_attr["order_attr_prev_page_name"] = pd.NA

    valid_idx = order_attr.index[order_attr["has_prev_explicit"]]
    if len(valid_idx) > 0:
        order_attr.loc[valid_idx, "order_attr_prev_session_id"] = prev_lookup["session_id"].to_numpy()
        order_attr.loc[valid_idx, "order_attr_prev_timestamp"] = prev_lookup["event_timestamp"].to_numpy()
        order_attr.loc[valid_idx, "order_attr_prev_event_type"] = prev_lookup["event_type"].to_numpy()
        order_attr.loc[valid_idx, "order_attr_prev_city"] = prev_lookup["page_city_name"].to_numpy()
        order_attr.loc[valid_idx, "order_attr_prev_device_type"] = prev_lookup["device_type"].to_numpy()
        order_attr.loc[valid_idx, "order_attr_prev_poi_id"] = prev_lookup["poi_id"].to_numpy()
        order_attr.loc[valid_idx, "order_attr_prev_poi_name"] = prev_lookup["poi_name"].to_numpy()
        order_attr.loc[valid_idx, "order_attr_prev_cate"] = prev_lookup["first_cate_name"].to_numpy()
        order_attr.loc[valid_idx, "order_attr_prev_page_id"] = prev_lookup["page_id"].to_numpy()
        order_attr.loc[valid_idx, "order_attr_prev_page_name"] = prev_lookup["page_name"].to_numpy()

    order_attr["order_attr_gap_ms"] = (
        pd.to_numeric(order_attr["event_timestamp"], errors="coerce")
        - pd.to_numeric(order_attr["order_attr_prev_timestamp"], errors="coerce")
    )
    order_attr["order_attr_same_session"] = (
        order_attr["session_id"].astype(str) == order_attr["order_attr_prev_session_id"].astype(str)
    ) & order_attr["has_prev_explicit"]
    order_attr["order_attr_confidence"] = [
        confidence_bucket(gap, same_session, has_prev)
        for gap, same_session, has_prev in zip(
            order_attr["order_attr_gap_ms"].tolist(),
            order_attr["order_attr_same_session"].tolist(),
            order_attr["has_prev_explicit"].tolist(),
        )
    ]
    order_attr[resolved_cate_col] = order_attr["order_attr_prev_cate"].fillna("未知品类")
    order_attr[resolved_business_col] = order_attr[resolved_cate_col].apply(business_line_from_cate)
    order_attr["order_attr_rule"] = (
        "STRICT_SAME_SESSION_PREV_EXPLICIT" if strict_same_session else "PREV_EXPLICIT_BY_USER"
    )

    order_attr.to_csv(
        os.path.join(output_dir, "order_attribution_lookup.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    gap_stats = quantile_dict(order_attr["order_attr_gap_ms"])
    confidence_dist = (
        order_attr["order_attr_confidence"].value_counts(dropna=False).to_dict()
    )
    top_order_cates = (
        order_attr[resolved_cate_col]
        .value_counts()
        .head(20)
        .rename_axis(resolved_cate_col)
        .reset_index(name="order_count")
    )
    top_order_cates.to_csv(
        os.path.join(output_dir, "resolved_order_category_distribution.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    top_order_cities = (
        order_attr["order_attr_prev_city"]
        .fillna("未知城市")
        .value_counts()
        .head(20)
        .rename_axis("order_attr_prev_city")
        .reset_index(name="order_count")
    )
    top_order_cities.to_csv(
        os.path.join(output_dir, "resolved_order_city_distribution.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    order_profile: Dict[str, object] = {
        "attributed_orders": int(order_attr["has_prev_explicit"].sum()),
        "attributed_rate": float(order_attr["has_prev_explicit"].mean()),
        "same_session_rate": float(order_attr["order_attr_same_session"].mean()),
        "confidence_distribution": {k: int(v) for k, v in confidence_dist.items()},
        "gap_p50_sec": gap_stats.get("p50_ms", 0.0) / 1000.0,
        "gap_p90_sec": gap_stats.get("p90_ms", 0.0) / 1000.0,
        "gap_p95_sec": gap_stats.get("p95_ms", 0.0) / 1000.0,
        "gap_p99_sec": gap_stats.get("p99_ms", 0.0) / 1000.0,
        "gap_max_sec": gap_stats.get("max_ms", 0.0) / 1000.0,
    }

    with open(os.path.join(output_dir, "dataset_profile_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"profile": profile, "order_profile": order_profile}, f, ensure_ascii=False, indent=2)

    write_summary_markdown(
        os.path.join(output_dir, "dataset_profile_summary.md"),
        profile=profile,
        order_profile=order_profile,
        top_city_users=top_city_stats.head(10),
        top_device_users=top_device_stats.head(10),
        top_order_cates=top_order_cates,
        resolved_cate_col=resolved_cate_col,
    )

    if write_augmented:
        print(f"💾 正在分块写出带归因字段的 {os.path.basename(output_csv)} ...")
        if os.path.exists(output_csv):
            os.remove(output_csv)

        lookup = order_attr[
            [
                "row_key",
                "order_attr_prev_session_id",
                "order_attr_prev_timestamp",
                "order_attr_prev_event_type",
                "order_attr_prev_city",
                "order_attr_prev_device_type",
                "order_attr_prev_poi_id",
                "order_attr_prev_poi_name",
                "order_attr_prev_cate",
                "order_attr_prev_page_id",
                "order_attr_prev_page_name",
                "order_attr_gap_ms",
                "order_attr_same_session",
                "order_attr_confidence",
                "order_attr_rule",
                resolved_cate_col,
                resolved_business_col,
            ]
        ].copy()
        first_write = True
        for chunk in pd.read_csv(data_path, chunksize=chunksize):
            chunk["event_type"] = chunk["event_type"].astype(str).str.strip().str.upper()
            merged = chunk.merge(lookup, on="row_key", how="left")
            merged[resolved_cate_col] = np.where(
                merged["event_type"] == "ORDER",
                merged[resolved_cate_col].fillna("未知品类"),
                merged["first_cate_name"].fillna("未知品类"),
            )
            merged[resolved_business_col] = np.where(
                merged["event_type"] == "ORDER",
                merged[resolved_business_col].fillna("其他业务"),
                merged[resolved_cate_col].apply(business_line_from_cate),
            )
            merged.to_csv(
                output_csv,
                index=False,
                encoding="utf-8-sig",
                mode="w" if first_write else "a",
                header=first_write,
            )
            first_write = False

    elapsed = time.time() - start
    print("✅ 数据集重画像与 ORDER 前序归因完成。")
    print(
        f"📊 用户数: {profile['total_users']:,} | 订单数: {profile['order_rows']:,} | 可归因订单占比: {order_profile['attributed_rate']:.2%}"
    )
    print(
        f"⏱ 归因时间差: p50={order_profile['gap_p50_sec']:.1f}s, p90={order_profile['gap_p90_sec']:.1f}s, p95={order_profile['gap_p95_sec']:.1f}s"
    )
    print(f"🕒 总耗时: {elapsed / 60:.2f} 分钟")


if __name__ == "__main__":
    main()
