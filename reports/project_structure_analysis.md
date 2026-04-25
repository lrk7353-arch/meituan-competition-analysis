# meituan-competition-analysis 项目结构分析文稿

## 一、项目整体定位

`meituan-competition-analysis` 是一个面向美团商业分析竞赛的项目，核心任务是基于单日用户行为数据，识别平台内部跨业务场景之间的协同关系，并进一步评估这些 Cross 导流链路的价值。

项目可以概括为：

> 基于单日用户决策链的跨业务导流有效性评估框架。

其分析目标不是单纯做热力图或模型训练，而是完成一条相对完整的商业分析链路：

```text
用户行为数据清洗与订单归因
  → 跨业务关系识别
  → 协同强度与转化价值评估
  → CrossScore 综合评分
  → 策略分层与报告输出
```

从现有项目上下文看，项目曾经尝试以 GNN 为核心方法，但后续逐渐转向以 `Lift + Markov + CCR` 为主线的统计与序列分析框架。这一转向是合理的，因为当前数据只有单日行为记录，图结构较薄，POI 粒度又较细，GNN 容易过拟合或学习到伪结构。

---

## 二、项目方法结构

当前项目大致可以分成四层：

```text
EDA 统计量化层
  ↓
GNN 候选发现层
  ↓
价值评估与策略层
  ↓
Agent 交互展示层
```

但从实际推进状态看，项目主线已经发生变化：

| 层级 | 当前状态 | 说明 |
|---|---|---|
| EDA / 统计量化层 | 主线 | 以 Lift、Markov、CCR 为核心，承担主要解释功能 |
| GNN 候选发现层 | 基本弃用 | 单日薄图、POI 粒度过细，GNN 泛化效果不足 |
| 价值评估与策略层 | 待完善 | 需要继续设计 CrossScore 和策略分层 |
| Agent / Demo 展示层 | 待开发 | 可用于自然语言问答、策略报告生成和演示 |

因此，当前最稳妥的技术路线应当是：

```text
Lift + Markov + CCR → CrossScore → 策略建议 → 报告 / PPT / Demo
```

---

## 三、核心数据处理链路

项目的数据处理链路可以整理为：

```text
原始数据 view_data.csv
  ↓
ORDER 事件归因补全
  ↓
增强数据 view_data.v1.1.csv
  ↓
Lift / Markov / CCR 基线分析
  ↓
cross_baseline_v2_outputs
  ↓
CrossScore / 策略分层 / 项目报告
```

### 1. 原始数据问题

原始数据中的核心问题是：`ORDER` 事件缺少明确的 `poi_id` 和品类信息。

这会直接影响 CCR 等转化类指标。如果订单没有品类或业务线，那么就无法判断“从 A 场景到 B 订单”的跨业务转化关系，导致跨品类转化结果大面积为 0 或失真。

### 2. 订单归因处理

项目中用于订单归因补全的关键脚本是：

```text
DATASET_USER_ORDER_REPROFILE.py
```

该脚本的核心思想是：

> 对 ORDER 事件使用同一用户、同一 session 内最近的明确 PV / MC 行为进行前序归因。

增强后的主数据文件是：

```text
view_data.v1.1.csv
```

新增字段包括：

```text
resolved_first_cate_name_v1_1
resolved_business_line_v1_1
order_attr_prev_cate
order_attr_gap_ms
order_attr_same_session
order_attr_confidence
```

这一处理非常关键，因为它把原本无法参与转化分析的订单事件转化为可归因、可统计、可解释的业务事件。

---

## 四、核心分析脚本

当前最重要的统一基线脚本是：

```text
CROSS_BASELINE_V2.for_colab.py
```

该脚本统一产出三类核心指标：

| 指标 | 含义 | 主要作用 |
|---|---|---|
| Lift | 衡量两个业务或类目是否经常共同出现 | 识别宏观协同关系 |
| Markov | 衡量 session 内严格相邻的场景跳转 | 识别用户即时路径和承接关系 |
| CCR | 衡量 A 场景出现后是否形成 B 订单 | 识别跨业务导流的转化价值 |

该脚本支持本地与 Colab 环境切换，并会优先读取 `view_data.v1.1.csv`。它还支持通过环境变量控制运行参数，例如：

```text
DATA_PATH
OUTPUT_DIR
MAX_ROWS
DIRECT_MAX_GAP_MIN
CCR_WINDOW_MIN
MIN_MARKOV_PAIR_COUNT
MIN_LIFT_PAIR_COUNT
MIN_CCR_CONV_SESSIONS
```

这说明项目已经具备一定的可复现基础，后续只需要进一步整理目录结构和运行说明即可。

---

## 五、主要输出目录

当前最核心的输出目录是：

```text
cross_baseline_v2_outputs/
```

其中包括：

```text
baseline_summary.md
cross_pair_master_table_business_line.csv
cross_pair_master_table_category.csv
markov_business_line_pairs.csv
lift_business_line_pairs.csv
ccr_business_line_pairs.csv
heatmap_markov_business_line.png
heatmap_lift_business_line.png
heatmap_ccr_business_line.png
```

这些文件应当被视为项目报告和 PPT 的主要素材来源。

其他相关输出目录包括：

```text
dataset_reprofile_outputs_v1_1/
cross_score_v1_outputs/
dataset_reprofile_outputs/
eda_baseline_outputs/
export_assets_v1_0/
```

其中，`dataset_reprofile_outputs_v1_1/` 主要对应订单归因和数据画像，`cross_score_v1_outputs/` 则可以作为后续综合评分和策略分层的输出位置。

---

## 六、当前业务结论

根据现有分析结果，业务线层面最值得进入报告的 Cross 关系包括：

| Cross 关系 | 业务解释 |
|---|---|
| 到店综合 → 餐饮 | 到店综合场景后容易被餐饮承接，适合做即时推荐 |
| 到家 → 餐饮 | 到家与餐饮之间存在即时转化关系 |
| 餐饮 → 到家 | 生活服务闭环明显，可用于复购和补充需求推荐 |
| 到店综合 ↔ 酒旅 | 宏观协同强，适合做场景联动和组合推荐 |
| 酒旅 → 餐饮 | 酒旅后餐饮承接能力较强，符合出游后的消费链路 |

当前可以形成三个主要判断：

1. `餐饮` 是主要承接场景；
2. `到店综合 ↔ 酒旅` 是较强的宏观协同场景；
3. `到家 → 餐饮` 与 `餐饮 → 到家` 构成较典型的生活服务闭环。

这些结论适合作为报告中的核心业务洞察。

---

## 七、GNN 路线状态与方法选择理由

项目中保留了若干 GNN 相关脚本，例如：

```text
GAT_CLAUDE.V1.1.py
cross_analysis_fullgraph_training_v2.py
cross_analysis_training_local.py
```

但当前不建议继续把 GNN 作为主方法。原因包括：

- 数据只有单日，图结构较薄；
- POI 粒度过细，节点过多但有效结构不足；
- POI 输入特征维度较低，模型容量容易过大；
- GNN 可能记忆节点 ID，而不是学习可迁移规律；
- 训练 AUC 提升有限，且存在明显过拟合风险；
- 跨城伪跳转等问题会干扰图结构学习。

因此，在当前数据条件下，统计和序列方法比 GNN 更适合作为主线。这一选择并不是方法退化，而是对数据条件、业务解释性和竞赛报告可讲性的综合权衡。

---

## 八、当前仓库结构评价

目前仓库更像是一个竞赛分析项目的工作区快照，而不是标准工程化项目。

### 优点

- 业务问题比较清楚；
- 数据处理链路已经形成；
- 已有订单归因增强数据；
- 已有统一基线脚本；
- 已有输出结果和报告素材；
- 对 GNN 路线失败原因有清晰反思。

### 问题

- 缺少标准 `README.md`；
- 缺少 `requirements.txt`；
- 脚本命名较临时，例如 `for_colab.py`、`V2`、`V1.1`；
- 数据、输出、实验文件之间的目录边界不够清楚；
- 方法主线已经变化，但文件结构仍保留较多旧 GNN 痕迹；
- 尚未形成清晰的 `src/`、`notebooks/`、`outputs/`、`reports/` 分层。

---

## 九、建议整理后的目录结构

建议后续将项目整理为：

```text
meituan-competition-analysis/
├─ README.md
├─ CLAUDE.md
├─ requirements.txt
├─ .gitignore
├─ data/
│  ├─ README.md
│  └─ sample/
├─ src/
│  ├─ data_reprofile.py
│  ├─ baseline_cross.py
│  ├─ cross_score.py
│  └─ visualization.py
├─ notebooks/
│  ├─ 01_data_profile.ipynb
│  ├─ 02_baseline_analysis.ipynb
│  └─ 03_report_figures.ipynb
├─ outputs/
│  ├─ dataset_reprofile/
│  ├─ cross_baseline/
│  └─ cross_score/
├─ reports/
│  ├─ project_structure_analysis.md
│  ├─ report.md
│  ├─ slides_outline.md
│  └─ defense_qa.md
└─ archive/
   └─ gnn_experiments/
```

各目录建议职责如下：

| 目录 | 作用 |
|---|---|
| `src/` | 放稳定脚本和可复用代码 |
| `notebooks/` | 放 Colab / Jupyter 实验和可视化过程 |
| `outputs/` | 放可复现生成的表格、图片、模型结果 |
| `reports/` | 放报告正文、PPT 大纲、答辩问答等文稿 |
| `archive/` | 放弃用但有解释价值的旧实验，例如 GNN |

---

## 十、下一步优先事项

后续建议按以下顺序推进：

1. 补充标准 `README.md`，说明项目目标、数据来源、运行方式和核心输出；
2. 补充 `requirements.txt`，固定运行环境；
3. 将订单归因、基线分析、CrossScore 分别整理成稳定脚本；
4. 将 GNN 相关脚本移动到 `archive/gnn_experiments/`，在报告中作为方法比较和失败实验解释；
5. 设计并固定 `CrossScore` 公式；
6. 输出 `reports/report.md`、`reports/slides_outline.md` 和 `reports/defense_qa.md`；
7. 后续再接入飞书文档，用于同步项目周报、会议纪要和最终展示材料。

---

## 十一、结论

`meituan-competition-analysis` 当前已经具备一条完整的分析主线：

```text
订单归因补全
  → 用户决策链分析
  → Lift / Markov / CCR
  → CrossScore
  → 策略分层与报告输出
```

当前项目最需要的不是继续堆模型，而是完成工程化整理和报告化表达。

更准确地说，下一阶段目标应当是把它从“能跑的分析工作区”整理成：

> 一个可复现、可解释、可展示、可答辩的商业分析系统。
