# 美团 Cross 项目上下文汇总

本文档用于汇总当前项目讨论中已经明确的关键结论，方便后续继续开发、和队友同步、准备答辩与汇报。

---

## 1. 项目目标与总体判断

本项目不是做一个单独的“GNN 预测 demo”，而是要做成一个完整的竞赛级 Cross 决策系统。

核心目标包括：

1. 识别高潜力 Cross 场景。
2. 量化业务/品类之间的协同效应、引流效率和转化潜力。
3. 输出可以落地的运营策略建议。
4. 用 Agent / Demo 强化展示力与答辩表现。

已明确的总框架为四层：

```text
EDA 统计量化层
-> GNN 候选发现层
-> 价值评估与策略层
-> Agent 交互展示层
```

这个框架已经保存为：

- [CROSS_FRAMEWORK_V2.md](/home/konglingrui/meituan_project/CROSS_FRAMEWORK_V2.md)

---

## 2. 关于现有项目文件的总体结论

当前项目里几个核心 Python 文件的定位已经明确：

### 正式主干候选

- [`GAT_CLAUDE.py`](/home/konglingrui/meituan_project/GAT_CLAUDE.py)
- [`cross_analysis_fullgraph_training_v2.py`](/home/konglingrui/meituan_project/cross_analysis_fullgraph_training_v2.py)

### 本地调试 / 辅助脚本

- [`cross_analysis_training_local.py`](/home/konglingrui/meituan_project/cross_analysis_training_local.py)
- [`inference_sample.py`](/home/konglingrui/meituan_project/inference_sample.py)

### 演示片段 / 需要重写

- [`cross_scene_topk_inference.py`](/home/konglingrui/meituan_project/cross_scene_topk_inference.py)

### 历史实验版 / 不建议做主干

- [`GAT_GPT.py`](/home/konglingrui/meituan_project/GAT_GPT.py)

### 故障记录

- [`training.log`](/home/konglingrui/meituan_project/training.log)

---

## 3. 各脚本角色总结

## 3.1 `GAT_CLAUDE.py`

定位：

- 当前最适合作为正式主模型训练脚本
- 在整体框架中承担 `GNN 候选发现层`
- 同时是后续 `CrossScore`、策略层、推理层、Agent 层的上游资产生产器

优点：

- 明确修复了三类关键 bug
- 训练流程更稳健
- 具备 target leakage 防护
- 具备 early stopping / scheduler / 最优模型保存
- 更适合多人协作和 Colab 复现

结论：

- 建议作为主干保留和继续开发

---

## 3.2 `cross_analysis_fullgraph_training_v2.py`

定位：

- 正式训练主干的备选版本
- 更像“实验增强版 / 结构尝试版”

优点：

- 有 `log1p` 特征平滑
- 有非对称投影打分头
- 有全图训练思路

不足：

- 可配置性和工程稳健性略弱于 `GAT_CLAUDE.py`

结论：

- 不作为主干
- 作为对照实验版和组件来源保留
- 值得吸收的内容：
  - `log1p` 长尾平滑
  - 非对称双线性 / 投影打分头

---

## 3.3 `cross_analysis_training_local.py`

定位：

- 本地采样调试版

问题：

- 训练日志中出现了明显报错
- 当前不适合作为正式实验主线

结论：

- 保留
- 但只作为本地 debug 脚本使用

---

## 3.4 `cross_scene_topk_inference.py`

定位：

- 演示型脚本 / notebook 片段风格

问题：

- 依赖外部上下文变量
- 不能作为正式推理程序

结论：

- 保留思路，不保留原样
- 后续建议重写为正式推理脚本

---

## 3.5 `inference_sample.py`

定位：

- 较稳定的推理模板

结论：

- 保留
- 可以作为未来正式推理模块的基础版

---

## 3.6 `GAT_GPT.py`

定位：

- 较早的实验版 / 历史版本

核心问题：

- 有较强的 ID Embedding 记忆风险
- 工程稳健性不如 `GAT_CLAUDE.py`
- 更像“提分尝试版”而不是正式主干

结论：

- 不是废稿
- 但不建议作为主干
- 可作为历史实验版 / 对照参考保留

---

## 3.7 `training.log`

定位：

- 训练运行日志文件

当前意义：

- 主要记录了一次失败的本地训练过程
- 更像故障记录，而不是成果记录

结论：

- 不是最终实验结果展示材料
- 对排查 `cross_analysis_training_local.py` 的问题有帮助

---

## 4. 主干选择最终结论

当前已经明确：

### 主干脚本

- [`GAT_CLAUDE.py`](/home/konglingrui/meituan_project/GAT_CLAUDE.py)

### 辅助 / 对照脚本

- [`cross_analysis_fullgraph_training_v2.py`](/home/konglingrui/meituan_project/cross_analysis_fullgraph_training_v2.py)

### 主干吸收内容

从 `cross_analysis_fullgraph_training_v2.py` 中吸收到主干的建议包括：

1. `log1p` 特征平滑
2. 非对称投影打分头，作为可选预测头做对照实验

---

## 5. 关于项目整体框架的最终共识

主框架不改，继续采用四层结构：

1. EDA 统计量化层
2. GNN 候选发现层
3. 价值评估与策略层
4. Agent 交互展示层

额外吸收的可取思想：

- `CrossScore(A, B)` 统一评分公式
- `StrategyTier(A, B)` 分层策略
- Agent 工具接口化

不采用的偏差点：

- 不把 Agent 当成核心分析层
- 不省略价值评估层
- 不把权重拍脑袋写死

---

## 6. `GAT_CLAUDE.py` 在整体系统中的角色

它的系统角色已经明确：

> `GAT_CLAUDE.py` 是项目的异构图主模型训练模块，负责从用户、商家、品类的多关系行为图中学习 Cross 迁移结构，并输出高潜 Cross 候选关系，为后续的 CrossScore 价值评估和策略生成提供核心结构信号。

它不是最终完整系统本身，而是整套系统的“引擎”。

---

## 7. 关于 `GAT_CLAUDE.py` 的后续优化共识

针对 `GAT_CLAUDE.py`，已经形成的共识是：

### 需要补充的能力

1. 训练结束后导出下游推理所需资产
2. 支持后续同城过滤
3. 支持 category / business 级聚合
4. 为后续 `inference_cross_score.py` 提供 embedding 和元数据

### 可取的改进方向

1. 导出 `poi_embeddings`
2. 导出 `category_embeddings`
3. 导出 `poi_idx_to_city`
4. 导出 `poi_idx_to_cate_idx`
5. 导出 `cate_idx_to_name`
6. 导出 `active_poi_mask`
7. 导出实验配置

### 对城市信息的处理原则

当前共识是：

- 第一阶段先不把城市直接入模
- 先作为推理过滤和 EDA 切片分析的元数据导出

---

## 8. 关于 Gemini 提出的“终极资产导出”建议的最终判断

Gemini 对 `GAT_CLAUDE.py` 末尾增加资产导出代码的建议，整体结论是：

### 总体判断

- 方向正确
- 工程上很有价值
- 但初版不能原样直接使用，需要修正

### 之前已经指出的关键修正点

1. 导出 embedding 时必须加载 best model，而不是直接使用当前内存模型
2. `active_poi_mask` 必须做成长度严格等于 `num_pois` 的全长 mask
3. `save_json` 必须支持嵌套 dict 和 numpy / tensor 的递归安全转换
4. 城市映射需要防空值，且建议基于 `train_df`

### 后续 Gemini 根据这些意见给出了修正版

这份修正版已经达到“基本可用”的程度，整体结论是：

- 可以加
- 没有原则性反对意见

### 额外补充的小建议

1. 在加载 best model 前增加文件存在性检查
2. 如果未来要做更友好的 Demo，可以额外导出 `poi_idx_to_name.json`

---

## 9. 资产导出模块的最终共识

建议在 `GAT_CLAUDE.py` 训练完成后，追加“阶段 7：终极资产导出”，用于把训练产物和下游推理衔接起来。

建议导出的文件包括：

- `best_cross_model.pth`
- `poi_embeddings.pt`
- `category_embeddings.pt`
- `active_poi_mask.pt`
- `poi_idx_to_id.json`
- `cate_idx_to_name.json`
- `poi_idx_to_cate_idx.json`
- `poi_idx_to_city.json`
- `poi_stats.json`
- `cate_stats.json`
- `experiment_config.json`

可选增强：

- `poi_idx_to_name.json`

---

## 10. 对 Category Embedding 的共识

已经明确：

- 第一版可以先用 `PV-weighted mean pooling`
- 不建议只停留在简单平均
- 后续可以做对照实验：
  - mean pooling
  - PV-weighted pooling
  - order-weighted pooling

这样更有竞赛答辩价值。

---

## 11. 对 CrossScore 的共识

统一评分公式可以保留，但不能拍脑袋硬设权重。

建议表达为：

```text
CrossScore(A, B) = α * S1 + β * S2 + γ * S3 + δ * S4
```

其中：

- `S1`：统计协同强度
- `S2`：图结构潜力
- `S3`：时机成熟度
- `S4`：稳定性与覆盖性

统计项的主信号建议优先采用：

- Lift
- CCR
- 样本量平滑

不建议仅用简单共现频次作为主项。

---

## 12. 下一步开发顺序的共识

当前已经形成明确结论：

### 推荐顺序

1. 在 `GAT_CLAUDE.py` 中补充资产导出
2. 新建 `inference_cross_score.py`
3. 在推理脚本里接入：
   - embedding 加载
   - 同城过滤
   - 冷启动过滤
   - category / business 级 Top-K
   - CrossScore 排序
4. 然后再补 EDA 报表
5. 再做策略层
6. 最后做 Agent / Demo

### 为什么先做推理底座

因为这样最快形成闭环：

```text
训练
-> 导出资产
-> 推理
-> 排序
-> 策略
-> Demo
```

EDA 虽然同样重要，但从工程推进上，先搭好推理底座更能把模型成果落地。

---

## 13. 现阶段最重要的开发目标

当前最适合立刻进入开发的模块是：

### `inference_cross_score.py`

建议承担的职责：

1. 加载 `poi_embeddings.pt` 与 `category_embeddings.pt`
2. 加载映射字典和 `active_poi_mask`
3. 实现同城过滤
4. 实现冷启动 / 活跃度过滤
5. 实现 category / business 级候选排序
6. 实现 `CrossScore` 打分
7. 输出策略层可以直接使用的排序结果

---

## 14. 本项目当前成熟度判断

根据已有讨论，当前项目已经从“代码能不能跑通”的阶段，升级到了“系统怎么组织、结果怎么落地”的阶段。

当前成熟度可以概括为：

- 主框架已定
- 主干脚本已选
- 角色分工已初步清晰
- 训练资产导出方案已明确
- 下一步推理基座开发方向已明确

真正还没完成的，不是方向，而是工程落地。

---

## 15. 当前最重要的文件

后续应重点围绕以下文件继续推进：

- [CROSS_FRAMEWORK_V2.md](/home/konglingrui/meituan_project/CROSS_FRAMEWORK_V2.md)
- [PROJECT_CONTEXT_SUMMARY.md](/home/konglingrui/meituan_project/PROJECT_CONTEXT_SUMMARY.md)
- [`GAT_CLAUDE.py`](/home/konglingrui/meituan_project/GAT_CLAUDE.py)
- [`cross_analysis_fullgraph_training_v2.py`](/home/konglingrui/meituan_project/cross_analysis_fullgraph_training_v2.py)
- [`inference_sample.py`](/home/konglingrui/meituan_project/inference_sample.py)

---

## 16. 一句话结论

当前项目的最终共识是：

> 以 `GAT_CLAUDE.py` 作为异构图主模型训练主干，补充资产导出能力，随后优先开发 `inference_cross_score.py` 作为推理与 CrossScore 底座，再逐步接入 EDA、策略层和 Agent 展示层，完成一个竞赛级的 Cross 决策系统。

