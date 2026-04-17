# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

**美团商业分析竞赛项目**——Cross 跨业务协同机会发现与价值评估系统。

- 项目路径: `/home/konglingrui/meituan_project/`
- 数据: 680万条用户行为记录（`view_data.csv`），2026年1月11日单日数据
- 核心任务: 发现高潜力 Cross 场景 → 量化协同价值 → 输出运营策略

## 系统架构（四层）

```
EDA 统计量化层（Lift/CCR/Markov/Time Window）
    ↓
GNN 候选发现层（异构图注意力网络）
    ↓
价值评估与策略层（CrossScore + StrategyTier）
    ↓
Agent 交互展示层（自然语言问答 + 策略报告）
```

**技术路线 2.0 结论**：以统计方法（Lift/CCR/Markov）为主线，GNN 为辅助信号。原因：单日薄图数据上 GNN 容易过拟合（Train AUC 0.85 / Val AUC 0.65，gap = 0.21）。

## 常用命令

```bash
# 进入项目目录
cd ~/meituan_project

# 激活 conda 环境
source ~/miniconda3/etc/profile.d/conda.sh && conda activate meituan_gnn

# 使用指定环境运行（避免路径问题）
/home/konglingrui/miniconda3/envs/meituan_gnn/bin/python <script.py>

# 后台训练 GNN（显存占用约 8-10GB）
nohup /home/konglingrui/miniconda3/envs/meituan_gnn/bin/python GAT_CLAUDE.V1.1.py > training.log 2>&1 &

# 查看训练日志
tail -f training.log
```

## 核心文件角色

| 文件 | 角色 | 备注 |
|------|------|------|
| `GAT_CLAUDE.V1.1.py` | GNN 主训练脚本 | 异构 GAT，hidden=64，heads=2，含早停/资产导出 |
| `EDA_CROSS_BASELINE_Gemini.py` | EDA 基线脚本 | Lift/CCR/Markov/Lag，一次扫描，含防OOM分块 |
| `fix_font.py` | 字体修复 | 解决 matplotlib 中文显示方框问题 |
| `cross_analysis_fullgraph_training_v2.py` | 队友 Colab 版 GNN | 60GB 显存版，hidden=128，heads=4 |
| `inference_sample.py` | 推理模板 | 较稳定的推理基础 |
| `cross_analysis_training_local.py` | 本地调试版 | 手动 scatter，含 debug 痕迹 |

## 关键经验教训

### 1. 显存约束
- GPU: RTX 5060 Ti（15.9 GB），需预留 6-8GB 给其他任务
- `hidden=128, heads=4` 在全图 GAT 下会爆 80GB 显存
- 安全的 GNN 配置: `hidden=64, heads=2`

### 2. OOM 风险点
- EDA 的 session 自连接（`pd.merge(df, df, on="session_id")`）会产生亿行笛卡尔积
- 解决：分块处理（chunk=20000）或改用逐 session 遍历聚合

### 3. 中文图表字体
- matplotlib 默认不支持中文，会显示为空方框
- 解决：在 `fix_font.py` 中查找 WQY/Noto/UMing 字体并注册

### 4. GNN 过拟合根因
- POI 输入特征仅 3 维（pv/mc/order），hidden=64 容量过大
- 特征维度信息量撑不起模型宽度 → 模型记忆节点 ID，而非学习迁移规律
- 修复：特征增强（9维）+ 降低 hidden 或加强正则

### 5. ORDER 事件品类缺失
- view_data.csv 的 ORDER 事件没有 poi_id/品类信息
- 影响：CCR 计算时跨品类转化几乎全为 0
- 修复：通过 PV/MC 事件的 POI→Category 映射表反向补全

## 当前进展

- ✅ EDA 基线层完成（Lift/CCR/Markov/Lag + 热力图）
- ⏳ GNN 训练（等 Colab 跑完导出资产）
- ⏳ inference_cross_score.py（融合层，未开发）
- ⏳ 策略层 + Agent/Demo

## 输出目录

- `eda_baseline_outputs/` — EDA 基线结果（Lift 矩阵、CCR 矩阵、热力图）
- `export_assets_v1_0/` — GNN 训练产物（embeddings、映射表、active mask）
