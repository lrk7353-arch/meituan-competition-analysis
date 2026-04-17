import torch
import numpy as np
import pandas as pd

print("🚀 正在启动工业级 Top-K 推荐推理引擎...")

# ==========================================
# 1. 加载最佳模型与全图前向推断
# ==========================================
# 确保模型在验证模式
model.load_state_dict(torch.load("best_cross_model.pth"))
model.eval()

with torch.no_grad():
    print("🧠 正在提取全网商家的终极特征向量 (Embedding)...")
    # 拿到包含 140 万个商家最终特征的巨型矩阵
    out_x = model(x_dict, edge_index_dict, edge_attr_dict)
    poi_embeddings = out_x['poi']  # 维度: [num_pois, hidden_channels]

# ==========================================
# 2. 准备品类映射字典 (为了让结果能看懂)
# ==========================================
# 获取 POI 到 Category 的映射关系
poi_to_cate_map = dict(zip(train_df['poi_idx'], train_df['first_cate_name']))

# ==========================================
# 3. 从“温区”中抽取一个验证集里的真实浏览轨迹
# ==========================================
print("正在清洗测试考卷，过滤纯冷启动节点...")

# 获取训练集中活跃的商家集合
active_poi_set = set(np.where(data['poi'].x[:, 0].cpu().numpy() > 0)[0])

# 过滤验证集：只保留起点和终点都在“温区”里的跳转边
warm_val_edges = [
    edge for edge in val_pos_edges
    if edge[0] in active_poi_set and edge[1] in active_poi_set
]

if len(warm_val_edges) == 0:
    print("⚠️ 警告：找不到全活跃的测试样本！")
else:
    # 从洗好的“温区”样本中随机挑一个测试
    sample_idx = np.random.randint(0, len(warm_val_edges))
    src_poi_idx = int(warm_val_edges[sample_idx][0])
    true_dst_poi_idx = int(warm_val_edges[sample_idx][1])

    full_poi_to_cate_map = dict(zip(df['poi_idx'], df['first_cate_name']))
    src_category = full_poi_to_cate_map.get(src_poi_idx, "未知品类")
    true_dst_category = full_poi_to_cate_map.get(true_dst_poi_idx, "未知品类")

    print("\n" + "=" * 50)
    print(f"🎯 测试起点商家 ID: {src_poi_idx} [品类: {src_category}]")
    print(f"✅ 顾客真实的下一跳: {true_dst_poi_idx} [品类: {true_dst_category}]")
    print("=" * 50)

# ==========================================
# 4. 向量化极速打分 (核心算法 - 加入冷启动掩码)
# ==========================================
start_time = time.time()
with torch.no_grad():
    src_emb = poi_embeddings[src_poi_idx].unsqueeze(0)
    src_h = model.src_proj(src_emb)
    dst_h = model.dst_proj(poi_embeddings)

    # 全量矩阵打分
    scores = (src_h * dst_h).sum(dim=-1)

# 💡 核心修复：工业界候选池过滤规则
# 找出在训练集里 PV > 0 的活跃商家（抛弃死节点和纯冷节点）
active_mask = (data['poi'].x[:, 0] > 0)

# 把自己，以及所有死节点的得分强制设为负无穷
scores[~active_mask] = -float('inf')
scores[src_poi_idx] = -float('inf')

# ==========================================
# 5. 提取 Top-K 推荐名单
# ==========================================
K = 15
topk_scores, topk_indices = torch.topk(scores, k=K)

print(f"\n⚡ 矩阵打分耗时: {(time.time() - start_time) * 1000:.2f} ms")
print(f"🏆 模型推荐的 Top-{K} 跨店引流名单：\n")

topk_indices = topk_indices.cpu().numpy()
topk_scores = topk_scores.cpu().numpy()

# 为了防止测试集的新节点找不到品类，我们从原始全量 df 里构建更全的字典
full_poi_to_cate_map = dict(zip(df['poi_idx'], df['first_cate_name']))
src_category = full_poi_to_cate_map.get(src_poi_idx, "未知品类")
true_dst_category = full_poi_to_cate_map.get(true_dst_poi_idx, "未知品类")

# 更新一下头部打印，让品类更准
print(f"🎯 测试起点商家 ID: {src_poi_idx} [品类: {src_category}]")
print(f"✅ 顾客真实的下一跳: {true_dst_poi_idx} [品类: {true_dst_category}]\n")

df_res = []
hit = False
for i, (poi, score) in enumerate(zip(topk_indices, topk_scores)):
    cate = full_poi_to_cate_map.get(poi, "未知品类")
    is_true = "⭐⭐⭐ 命中!" if poi == true_dst_poi_idx else ""
    if poi == true_dst_poi_idx: hit = True
    df_res.append({'排名': i + 1, '推荐商家ID': poi, '所属品类': cate, '预测得分': f"{score:.4f}", '备注': is_true})

display_df = pd.DataFrame(df_res)
print(display_df.to_string(index=False))

if not hit:
    true_score = scores[true_dst_poi_idx].item()
    if true_score == -float('inf'):
        print(f"\n💡 真实的下家商家 (ID:{true_dst_poi_idx}) 是一个绝对的死节点/冷启动节点，已被推荐池过滤。")
    else:
        rank = (scores > true_score).sum().item() + 1
        print(
            f"\n💡 真实的下家商家 (ID:{true_dst_poi_idx}) 预测得分为 {true_score:.4f}，在全网活跃商家中排名第 {rank} 位。")