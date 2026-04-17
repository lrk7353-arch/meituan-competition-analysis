# EDA 基线层统合与优化方案

> 本文档定义 `EDA_CROSS_BASELINE.py` 的完整重构规格。
> 整合自 Gemini + GPT 建议，并经 Claude 审查修正。

---

## 1. 现状总结与改造目标

### 1.1 现存致命弱点

| # | 问题 | 影响 |
|---|------|------|
| 1 | 三个脚本重复扫描 680 万行数据（2GB+） | 内存浪费，易 OOM，毫无必要 |
| 2 | Lift 公式无平滑，冷门品类对 Lift 可达 65+ | 榜单头部被噪音淹没，商业价值不可信 |
| 3 | `shift(-1)` 仅捕捉严格相邻转移，漏掉 look-ahead 机会 | 真实 Cross 场景如 `外卖→优惠券→酒店` 中的 `外卖→酒店` 被遗漏 |
| 4 | TIME_WINDOW 有 `max_lag_hours` 约束，LIFT 没有 | 间隔 8 小时和 5 分钟的转移权重相同，违背时效性 |
| 5 | CCR 计算中 ORDER 事件无品类信息，导致跨品类 CCR 全为 0 | CCR 模块完全失效 |
| 6 | 热力图无 colorbar 刻度、无图注、坐标轴不可读 | 完全达不到答辩展示标准 |
| 7 | 73 个细粒度品类无业务线聚合 | 答辩时无法快速回答"酒旅→休闲娱乐"等大类交叉机会 |

### 1.2 改造目标

- **一次扫描，三份输出**：680 万行只读一遍，内存峰值控制在可接受范围
- **算法可配置化**：平滑强度、窗口大小、步长限制均通过环境变量/顶部常量暴露
- **修复 CCR 数据问题**：通过 POI→category 映射表反向补全 ORDER 事件的品类
- **补充业务线聚合矩阵**：额外输出一级业务（外卖/到店/酒旅/闪购等）之间的 Cross 统计
- **答辩级图表**：热力图含数值标注、colorbar 含刻度、子图不空插
- **统一 metadata schema**：所有输出 JSON 共享同一套字段定义

---

## 2. 数据流架构

```
view_data.csv (680万行)
        │
        ▼
  ┌─────────────┐
  │  One-Pass   │
  │  读取与清洗  │
  └─────────────┘
        │
        ├──→ session_df        （session 序列，已折叠连续重复）
        │
        ├──→ poi_cate_map       （poi_id → first_cate_name 映射表）
        │
        └──→ order_session_cate （session → 是否含 ORDER + 品类集合）
                │
                ▼
        ┌──────────────────────┐
        │  核心统计引擎        │
        │  ① Lift 矩阵        │
        │  ② CCR 矩阵         │
        │  ③ Time Window 矩阵 │
        │  ④ Markov P(B|A)    │
        │  ⑤ 业务线聚合矩阵   │
        └──────────────────────┘
                │
                ▼
        ┌──────────────────────┐
        │  图表导出与文件输出  │
        └──────────────────────┘
```

---

## 3. 全局 One-Pass 读取与清洗

### 3.1 读取字段

```python
usecols = [
    "user_id", "session_id", "event_timestamp",
    "poi_id", "poi_name", "first_cate_name",
    "event_type", "event_id", "page_city_name",
]
```

### 3.2 事件类型统一

```python
# PV / MC / ORDER 统一映射
df["unified_event"] = ...
df["first_cate_name"] = df["first_cate_name"].fillna("未知品类")
df = df.dropna(subset=["session_id", "event_timestamp"])
```

### 3.3 session 内连续重复品类折叠

```python
df = df.sort_values(["session_id", "event_timestamp"])
prev_cate = df.groupby("session_id")["first_cate_name"].shift(1)
df = df[df["first_cate_name"] != prev_cate].copy()
```

### 3.4 POI→Category 映射表（用于 CCR 修复）

```python
# 从 PV/MC 事件中提取 poi_id → first_cate_name 的稳定映射
poi_cate_map = (
    df[df["unified_event"].isin(["PV", "MC"])]
    .groupby("poi_id")["first_cate_name"]
    .agg(lambda x: x.value_counts().index[0])  # 取最频繁的品类
    .to_dict()
)
```

### 3.5 ORDER 事件品类反向补全

```python
# 将 ORDER 事件的 first_cate_name 替换为真实 POI 对应品类
order_mask = df["unified_event"] == "ORDER"
df.loc[order_mask, "first_cate_name"] = df.loc[order_mask, "poi_id"].map(poi_cate_map).fillna("未知品类")
```

---

## 4. 核心算法规格

### 4.1 转移提取：多步滑动窗口（Look-ahead）

**配置常量**（全部可由环境变量覆盖）：

```python
MAX_INTERVAL_MINUTES = int(os.getenv("MAX_INTERVAL_MINUTES", "240"))   # 4 小时：同一 session 内 A→B 超过此值不算
LOOKAHEAD_WINDOW_MINUTES = int(os.getenv("LOOKAHEAD_WINDOW_MINUTES", "30"))  # 向前看窗口：默认 30 分钟
MIN_PAIR_COUNT = int(os.getenv("MIN_PAIR_COUNT", "20"))
TOP_K = int(os.getenv("TOP_K", "50"))
EXCLUDE_SELF_TRANSITION = os.getenv("EXCLUDE_SELF_TRANSITION", "1") == "1"
```

**Look-ahead 逻辑**：

```
给定 session 内已按时间排序的品类序列：
[A, C, B, D]

对于每个品类 A，在 A 之后 30 分钟内（含）的所有品类：
  - 如果出现 B，记为 A→B 成功转移
  - 不要求 A 和 B 严格相邻

这意味着：
  A→B（间隔 < 30min）：✓ 计入
  A→C（间隔 < 30min）：✓ 计入
  A→D（间隔 > 30min）：✗ 不计入
```

**实现思路**：

```python
def extract_lookahead_transitions(session_df, max_interval_min=30):
    """
    session_df: 已按时间排序的单个 session DataFrame
    返回: list of (src_cate, dst_cate, lag_sec) 元组
    """
    cates = session_df["first_cate_name"].values
    times = session_df["event_timestamp"].values
    transitions = []

    for i, (src_cate, src_ts) in enumerate(zip(cates[:-1], times[:-1])):
        for j in range(i + 1, len(cates)):
            dst_cate, dst_ts = cates[j], times[j]
            lag_min = (dst_ts - src_ts) / 60000.0
            if lag_min > max_interval_min:
                break  # 超出窗口，后面的更远，不再继续
            if src_cate == dst_cate:
                continue
            transitions.append((src_cate, dst_cate, lag_min))
    return transitions
```

> **为什么选 A 而非 B**：用户在 A 之后 30 分钟内的所有跨品类行为都是"受 A 激发"的后续兴趣，这种协同才是真实 Cross 推荐价值。B 方案（严格相邻）本质上是 Markov 一阶，彻底丢弃了跨品类桥接机会。

### 4.2 Lift 矩阵（含拉普拉斯平滑）

**原始公式（无平滑）**：

```
Lift(A→B) = P(B|A) / P(B) = count(A→B) / count(A)  ÷  count(B) / N_total
```

**修正后公式（分子加法平滑）**：

```python
ALPHA = float(os.getenv("LIFT_SMOOTHING_ALPHA", "1.0"))  # 平滑系数

# 条件概率平滑
p_ab_smoothed = (count_ab + ALPHA) / (count_a + ALPHA * num_categories)

# Marginal 概率保持原始（不需要平滑）
p_b = count_b / total_count

lift_smoothed = p_ab_smoothed / p_b
```

> **为什么只平滑分子**：分母 P(B) 是 B 的整体热度，平滑分母会把热门品类的有效 Lift 压扁，失去区分度。只平滑分子可以让低频品类对的 Lift 从异常高值回落到 1.x 区间，同时保持热门品类的原始排序。

**预期效果**：

| 品类对 | 原始 Lift | α=1.0 平滑后 |
|--------|-----------|--------------|
| 旅游→温泉景区（2967对） | 65.0 | ~12-15（仍是很高的显著协同） |
| 某冷门品类对（21对） | 100+ | ~1.2-1.5（被压制到接近 1） |

### 4.3 CCR 矩阵（Cross Conversion Rate）

**定义**：

```
CCR(A→B) = 在发生 A→B 转移的 session 中，B 被后续 ORDER 的比例
```

**修复后的计算逻辑**：

```python
# Step 1: 从 PV/MC 中建立 poi→真实品类映射
poi_cate_map = build_poi_cate_map(df)

# Step 2: 将 ORDER 事件的品类补全
df = fill_order_category(df, poi_cate_map)

# Step 3: 判断每个 session 中每个品类是否被 ORDER
session_cate_order = (
    df.assign(is_order=(df["unified_event"] == "ORDER").astype(int))
    .groupby(["session_id", "first_cate_name"])["is_order"]
    .max()
    .reset_index()
)

# Step 4: 计算 A→B 转移中 B 最终被下单的比例
transitions = transitions.merge(session_cate_order, on=["session_id", "next_cate"], how="left")
transitions["b_ordered"] = transitions["b_ordered"].fillna(0)
ccr = transitions.groupby(["src_cate", "dst_cate"])["b_ordered"].mean()
```

### 4.4 Markov 转移矩阵 P(B|A)

```python
# 条件概率矩阵（无平滑版本，用于 CrossScore S2）
markov_prob = transitions.groupby(["src_cate", "next_cate"]).size().unstack(fill_value=0)
markov_prob = markov_prob.div(markov_prob.sum(axis=1), axis=0)  # 行归一化
```

### 4.5 时间窗口矩阵

```python
# 统计每个品类对的 lag 分布
pair_stats = transitions.groupby(["src_cate", "dst_cate"]).agg(
    pair_count=("session_id", "count"),
    mean_lag_min=("lag_min", "mean"),
    median_lag_min=("lag_min", "median"),
    p25_lag_min=("lag_min", lambda x: np.percentile(x, 25)),
    p75_lag_min=("lag_min", lambda x: np.percentile(x, 75)),
).reset_index()
```

---

## 5. 业务线聚合矩阵

### 5.1 业务线定义

```python
BUSINESS_LINE_MAPPING = {
    "外卖": ["美食", "生鲜果蔬", "超市便利", "医药健康", "鲜花绿植"],
    "到店餐饮": ["餐饮", "甜点", "饮品", "休闲食品", "小吃快餐"],
    "酒旅": ["酒店", "旅游", "度假景区", "温泉景区", "滑雪景区", "主题乐园", "自然景观", "人文古迹"],
    "休闲娱乐": ["休闲娱乐", "丽人", "K歌", "电影演出赛事", "宠物", "运动健身", "亲子"],
    "闪购": ["数码家电", "美妆日化", "日用百货", "母婴玩具", "家居"],
    "到店其他": ["生活服务", "结婚", "教育培训", "养车/用车", "购物"],
}
```

> 实际映射需对照 view_data.csv 中真实出现的一级品类名称进行调整。

### 5.2 聚合方式

将细粒度品类矩阵按业务线合并：

```python
def aggregate_to_business_line(fine_matrix, mapping_dict):
    """
    将细粒度品类矩阵按业务线聚合
    矩阵值 = sum（不丢失信息量）
    """
    bl_matrix = {}
    for bl_name, cates in mapping_dict.items():
        bl_matrix[bl_name] = fine_matrix.loc[
            fine_matrix.index.isin(cates), :
        ].sum(axis=0)
    return pd.DataFrame(bl_matrix).T
```

聚合后单独输出：
- `business_line_lift_matrix.csv`
- `business_line_ccr_matrix.csv`
- `business_line_markov_matrix.csv`

---

## 6. 未知品类处理策略

| 处理位置 | 策略 |
|---------|------|
| 转移提取阶段 | **保留但不计入榜单**——在统计结果中单独标记 `is_unknown_dst=True` |
| 热力图输出 | 未知品类列/行**加粗红色**标注，便于判断噪声占比 |
| Top pairs 榜单 | 过滤条件 `src_cate != "未知品类" AND dst_cate != "未知品类"` |
| 业务线矩阵 | 未知品类**不参与**业务线聚合（不在任何业务线内） |

---

## 7. 输出文件规范

### 7.1 统一目录结构

```
eda_cross_outputs/
├── lift/
│   ├── category_transition_lift_matrix.csv
│   ├── category_transition_lift_top_pairs.csv
│   ├── category_transition_lift_heatmap.png
│   ├── business_line_lift_matrix.csv
│   └── lift_metadata.json
├── ccr/
│   ├── category_transition_ccr_matrix.csv
│   ├── category_transition_ccr_top_pairs.csv
│   ├── category_transition_ccr_heatmap.png
│   ├── business_line_ccr_matrix.csv
│   └── ccr_metadata.json
├── time_window/
│   ├── category_transition_time_window_summary.csv
│   ├── category_transition_median_lag_heatmap.png
│   ├── category_transition_lag_distribution.png
│   └── time_window_metadata.json
├── markov/
│   ├── category_markov_matrix.csv
│   ├── business_line_markov_matrix.csv
│   └── markov_metadata.json
└── metadata/
    ├── global_config.json          # 统一配置快照
    ├── poi_cate_map.json           # POI→品类映射（供下游使用）
    └── data_quality_report.json    # 数据质量报告（未知品类占比等）
```

### 7.2 统一 JSON metadata Schema

```json
{
  "schema_version": "2.0",
  "output_type": "lift | ccr | time_window | markov",
  "data_path": "/path/to/view_data.csv",
  "raw_rows": 6807745,
  "cleaned_rows": 6500000,
  "session_count": 950000,
  "unique_categories": 73,
  "collapse_consecutive": true,
  "exclude_self_transition": true,
  "max_interval_minutes": 240,
  "lookahead_window_minutes": 30,
  "lift_smoothing_alpha": 1.0,
  "min_pair_count": 20,
  "top_k": 50,
  "order_category_fill_rate": 0.XX,
  "unknown_category_ratio": 0.XX,
  "generated_at": "2026-04-06TXX:XX:XX"
}
```

---

## 8. 热力图规格（答辩级）

所有热力图必须满足以下标准，否则视为不合格：

### 8.1 Lift 热力图

```python
# 数值标注（每个格子内显示 Lift 值）
annot_matrix = heatmap_df.fillna(0).values
annot_text = np.where(annot_matrix > 0, np.round(annot_matrix, 1).astype(str), "")

fig, ax = plt.subplots(figsize=(14, 12))
im = ax.imshow(heatmap_df.fillna(0).values, cmap="YlOrRd", aspect="auto")

# Colorbar 必须有刻度
cbar = fig.colorbar(im, fraction=0.025, pad=0.02)
cbar.set_label("Lift", fontsize=12)
cbar.ax.tick_params(labelsize=10)

# 数值标注
for i in range(len(top_src)):
    for j in range(len(top_dst)):
        text = ax.text(j, i, annot_text[i, j],
                       ha="center", va="center", color="black", fontsize=8)

# 坐标轴
ax.set_xticks(range(len(top_dst)))
ax.set_yticks(range(len(top_src)))
ax.set_xticklabels(top_dst, rotation=60, ha="right", fontsize=10)
ax.set_yticklabels(top_src, fontsize=10)
ax.set_title(f"Category Transition Lift Heatmap\n(n={transition_count:,}, α={ALPHA}, window={LOOKAHEAD_WINDOW_MINUTES}min)",
             fontsize=14, pad=20)
ax.set_xlabel("Destination Category", fontsize=12)
ax.set_ylabel("Source Category", fontsize=12)
plt.tight_layout()
```

### 8.2 CCR 热力图

使用 `cmap="YlGnBu"`，其余规格同上。颜色范围 0~max(0.3, 95th_percentile)。

### 8.3 Lag 分布图

```python
# 不允许空子图：n_rows = ceil(n_pairs / n_cols)
n_pairs = plot_top_pairs  # 默认 12
n_cols = 3
n_rows = math.ceil(n_pairs / n_cols)

# 每个子图必须标注：
#   ① 品类对名称（父标题）
#   ② 样本量 n=xxx
#   ③ 中位数 lag
for idx, (src, dst) in enumerate(top_pairs):
    pair_data = filtered[src, dst]
    ax.hist(pair_data, bins=40, color=f"C{idx}", alpha=0.8)
    ax.set_title(f"{src} → {dst}\nn={len(pair_data)}, median={pair_data.median():.1f}min",
                 fontsize=9)
    ax.set_xlabel("Lag (minutes)", fontsize=8)
    ax.set_ylabel("Count", fontsize=8)
```

---

## 9. 要删除的旧文件

| 文件 | 删除原因 |
|------|---------|
| `EDA_LIFT_MATRIX.for_colab.py` | 功能已被整合进 `EDA_CROSS_BASELINE.py` |
| `EDA_CCR_MATRIX.for_colab.py` | 功能已被整合，且原版 CCR 有数据缺陷 |
| `EDA_TIME_WINDOW.for_colab.py` | 功能已被整合进 `EDA_CROSS_BASELINE.py` |
| `eda_lift_outputs/` | 输出目录已废弃，由 `eda_cross_outputs/` 替代 |
| `eda_ccr_outputs/` | 同上 |
| `eda_time_window_outputs/` | 同上 |

---

## 10. 验证标准

### 10.1 功能验证

- [ ] 脚本单次运行成功，无 OOM
- [ ] Lift Top-50 榜单中无未知品类
- [ ] 冷门品类对（pair_count≈20）的 Lift 平滑后 ≤ 2.0
- [ ] 热门品类对（pair_count≥1000）的 Lift 平滑后排序与原始基本一致
- [ ] CCR Top-50 榜单中出现真实跨品类对（如"旅游→酒店"），不再是未知品类霸榜
- [ ] 热力图数值标注清晰可读，colorbar 有刻度

### 10.2 数据质量验证

```bash
# 检查未知品类占比
python -c "
import pandas as pd
df = pd.read_csv('view_data.csv', usecols=['first_cate_name'])
unknown_ratio = (df['first_cate_name'] == '未知品类').mean()
print(f'未知品类占比: {unknown_ratio:.2%}')
"
```

### 10.3 性能验证

- [ ] 内存峰值 < 16GB（可在 Colab 15GB 限制内运行）
- [ ] 单次运行耗时 < 15 分钟（680 万行数据量）
- [ ] Look-ahead 窗口提取不超过总时间的 50%

---

## 11. 开发顺序建议

```
Phase 1: 数据读取与 POI→Category 映射表构建
Phase 2: Look-ahead 转移提取引擎
Phase 3: Lift 平滑算法 + 矩阵导出
Phase 4: CCR 修复 + 矩阵导出
Phase 5: Markov P(B|A) 矩阵导出
Phase 6: Time Window 矩阵导出
Phase 7: 业务线聚合矩阵
Phase 8: 答辩级热力图（含数值标注）
Phase 9: 统一 metadata 导出
Phase 10: 清理旧脚本与旧输出目录
```

---

## 12. Open Question 结论

**选择：A（算，30 分钟内看到 B 就计为 A→B）**

默认 `LOOKAHEAD_WINDOW_MINUTES = 30`，理由：
- 符合真实用户 Cross 转化心智（被 A 吸引后 30 分钟内的其他兴趣也是有效推荐候选）
- 比 2 小时更精准，避免跨时段噪音
- 可通过环境变量覆盖，团队可根据数据分布做调参实验

---

## 13. 后续依赖

`EDA_CROSS_BASELINE.py` 的输出资产是以下模块的输入：

```
EDA_CROSS_BASELINE.py
    ├── lift_matrix.csv      →  CrossScore S1
    ├── ccr_matrix.csv        →  CrossScore S1（CCR 分项）
    ├── markov_matrix.csv     →  CrossScore S2_seq
    ├── time_window_summary   →  CrossScore S4（触达时机）
    └── poi_cate_map.json    →  GNN / 推理模块
```

---

*文档版本：v1.0 | 审查：Claude | 日期：2026-04-06*
