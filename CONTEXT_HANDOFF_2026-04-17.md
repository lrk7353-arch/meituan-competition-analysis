# Cross 项目上下文压缩总结（2026-04-17）

## 1. 当前项目定位

项目已从“GNN 主导的薄图预测”转为“基于用户决策链的 Cross 导流有效性评估框架”。

当前主线：

- `Markov`：刻画用户在单日 session 内的严格相邻场景跳转。
- `Lift`：刻画用户-城市篮子层面的宏观协同强度。
- `CCR`：刻画 A 场景出现后，同 session 内最终形成 B 订单的归因转化率。
- `GNN`：已基本决定弃用，不再作为主方法推进。

核心原因：

- 数据只有单天，图过薄。
- POI 粒度太细。
- GNN 训练长期卡在 `AUC ~0.65` 左右，且出现跨城/伪结构问题。
- 用户链路分析更适合 `Lift + Markov + CCR` 这种统计/序列方法。

## 2. 关键文件

### 2.1 原始数据

- `/home/konglingrui/meituan_project/view_data.csv`

### 2.2 增强后的严格归因数据

- `/home/konglingrui/meituan_project/view_data.v1.1.csv`

说明：

- 这是目前后续所有分析的主底座。
- 使用严格前序归因：
  - 同一 `user_id`
  - 同一 `session_id`
  - 最近的明确 `PV/MC`
  - 仅在同 session 内归因

新增字段：

- `resolved_first_cate_name_v1_1`
- `resolved_business_line_v1_1`
- `order_attr_prev_cate`
- `order_attr_gap_ms`
- `order_attr_same_session`
- `order_attr_confidence`

### 2.3 ORDER 归因脚本

- `/home/konglingrui/meituan_project/DATASET_USER_ORDER_REPROFILE.py`

当前默认行为：

- 默认开启 `STRICT_SAME_SESSION=1`
- 默认输出：
  - `/home/konglingrui/meituan_project/view_data.v1.1.csv`
  - `/home/konglingrui/meituan_project/dataset_reprofile_outputs_v1_1/`

### 2.4 新的统一基线脚本

- `/home/konglingrui/meituan_project/CROSS_BASELINE_V2.for_colab.py`

特点：

- 统一产出 `Lift / Markov / CCR`
- 默认优先读取 `view_data.v1.1.csv`
- 适配 Colab 和本地
- 支持环境变量：
  - `DATA_PATH`
  - `OUTPUT_DIR`
  - `MAX_ROWS`
  - `DIRECT_MAX_GAP_MIN`
  - `CCR_WINDOW_MIN`
  - `MIN_MARKOV_PAIR_COUNT`
  - `MIN_LIFT_PAIR_COUNT`
  - `MIN_CCR_CONV_SESSIONS`

### 2.5 全量基线输出目录

- `/home/konglingrui/meituan_project/cross_baseline_v2_outputs`

关键文件：

- `baseline_summary.md`
- `cross_pair_master_table_business_line.csv`
- `cross_pair_master_table_category.csv`
- `markov_business_line_pairs.csv`
- `lift_business_line_pairs.csv`
- `ccr_business_line_pairs.csv`
- `heatmap_markov_business_line.png`
- `heatmap_lift_business_line.png`
- `heatmap_ccr_business_line.png`

## 3. ORDER 严格前序归因的关键结论

已经确认：

- 原始 `ORDER` 自带 `poi_id` 数量：`0`
- 原始 `ORDER` 自带明确品类数量：`0`
- 因此订单品类必须依赖前序归因

严格版 v1.1 归因结果：

- 总 `ORDER`：`243,054`
- 同 session 可归因 `ORDER`：`236,767`
- 归因覆盖率：`97.41%`
- 跨 session 归因数：`0`
- 无法归因订单：`6,287`

归因时间差分布：

- `p50 = 58.7s`
- `p90 = 236.4s`
- `p95 = 361.2s`

解读：

- “最近明确页面行为归因给订单”是有很强现实基础的。
- `5 分钟` 可以作为主归因置信边界。

## 4. 数据集画像关键结论

来自 `dataset_reprofile_outputs_v1_1/`：

- 总日志量：`6,807,745`
- 总用户数：`494,433`
- 发生过至少一次 `ORDER` 的用户：`177,367`
- 用户转化率：`35.87%`
- 总会话数：`496,872`
- 发生过 `ORDER` 的会话：`177,610`
- 会话转化率：`35.75%`

城市结论：

- 单城市用户占比：`98.38%`
- 发生过订单的用户中，单城市占比约 `98.68%`

方法含义：

- 城市不应主导 Cross 跳转学习。
- 城市更适合作为：
  - 过滤条件
  - 分层切片
  - 偏好背景变量

设备结论：

- 主体设备是 `android` 和 `iphone`
- `HarmonyOS` 体量可观但不是主流

## 5. 新基线结果的审阅结论

### 5.1 总体评价

这版结果已经从“能跑”进化到“可讲、可用、可写报告”。

优点：

- 三条线口径统一
- 不再被 `其他业务` 或 `未知品类` 严重污染
- 业务线层面的结构很清晰
- CCR 不再大面积塌成 0

### 5.2 业务线层面结论

从 `cross_pair_master_table_business_line.csv` 与热力图看：

最值得讲的结构：

- `到店综合 -> 餐饮`
- `到家 -> 餐饮`
- `餐饮 -> 到家`
- `到店综合 <-> 酒旅`
- `酒旅 -> 餐饮`

理解：

- `Markov` 显示“餐饮”是最强承接场景，很多业务最后都会落到餐饮。
- `Lift` 显示“到店综合 <-> 酒旅”是真正的高协同关系，属于宏观联动最强的一组。
- `CCR` 显示“到家/到店综合/酒旅 -> 餐饮”是即时成交能力最强的链路。

因此：

- `餐饮` 是主要承接场景
- `到店综合 <-> 酒旅` 是最强协同场景
- `到家 -> 餐饮`、`餐饮 -> 到家` 是最强生活服务闭环之一

### 5.3 类目层面结论

类目层结果有信息，但需要更谨慎：

- `Markov(category)` 有不少合理结果
- `CCR(category)` 也明显变健康了，如：
  - `美妆日化 -> 超市便利`
  - `日用百货 -> 超市便利`
  - `餐饮 -> 美食`
  - `水果 -> 美食`
- `Lift(category)` 仍然会被旅游/景区/展馆等强细分协同占据头部

这不是错误，但说明：

- 类目级 `Lift` 不能直接拿来做最终业务结论
- 需要二次筛选或类目归并

建议：

- 报告主图用业务线层
- 类目层用于补充案例

## 6. 当前对 GNN 的最终态度

目前建议：

- 正式放弃 GNN 作为主方法
- 不再继续投入主要精力优化 GNN

原因：

- 单日数据 + 薄图 + POI 粒度过细
- AUC 提升空间有限
- 跨城伪跳转问题严重
- 与“基于单个用户决策链”的分析目标不匹配

现在更合理的技术栈：

- 主线：`Lift + Markov + CCR`
- 输出：`CrossScore / 策略建议 / 评估框架`

## 7. 项目宏观定位（当前建议）

不再把项目仅仅理解为“Cross 分析”。

更好的表达是：

> 基于单日用户决策链的跨业务导流有效性评估框架

或者：

> 对平台内部跨业务引导机制的离线诊断与优化建议系统

这个表述比“做了几个热力图”更高级，也更符合竞赛答辩。

## 8. 下一步最合理的工作

按优先级：

1. 基于当前 `cross_baseline_v2_outputs` 继续做“结果审阅 + 报告用结论筛选”
2. 设计 `CrossScore`
3. 基于 `CrossScore` 形成策略分层（HIGH / MEDIUM / LOW）
4. 生成报告/PPT 话术与案例分析

## 9. 给下一窗口的直接提示

如果在下一窗口继续，请直接从这里接：

- 底座数据：`view_data.v1.1.csv`
- 主脚本：`CROSS_BASELINE_V2.for_colab.py`
- 主输出目录：`cross_baseline_v2_outputs`

优先事项：

- 继续审阅这版 `Lift / CCR / Markov`
- 筛出可直接进报告的业务线结论
- 设计 `CrossScore`

