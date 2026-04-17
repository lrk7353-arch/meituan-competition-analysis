"""
Cross Scene Top-K Inference (采样版 - 适配16GB显存)
基于队友的 cross_scene_topk_inference.py 修改
"""
import torch
import numpy as np
import pandas as pd
import sys
sys.stdout.reconfigure(encoding='utf-8')

print("🚀 启动采样版Top-K推理引擎 (适配16GB显存)")
print("=" * 60)

# ==========================================
# 1. 加载数据（同训练脚本）
# ==========================================
print("\n📂 [阶段1] 加载数据...")
file_path = '/home/konglingrui/meituan_project/view_data.csv'
usecols = ['user_id', 'poi_id', 'first_cate_name', 'event_type', 'event_id',
           'session_id', 'event_timestamp', 'device_type', 'page_city_name']
# event_id is needed for ORDER detection
df = pd.read_csv(file_path, usecols=usecols)

df['event_timestamp'] = pd.to_numeric(df['event_timestamp'], errors='coerce')
df = df.dropna(subset=['user_id', 'poi_id', 'event_timestamp', 'session_id'])

df['event_type_str'] = df['event_type'].astype(str).str.strip().str.upper()
df['event_id_str'] = df['event_id'].astype(str).str.strip().str.lower()
is_pv = df['event_type_str'] == 'PV'
is_mc = df['event_type_str'] == 'MC'
is_order = (df['event_type_str'] == 'ORDER') | (df['event_id_str'].str.contains('order', na=False))

df['unified_event'] = pd.Series(dtype='object')
df.loc[is_pv, 'unified_event'] = 'PV'
df.loc[is_mc, 'unified_event'] = 'MC'
df.loc[is_order, 'unified_event'] = 'ORDER'

df['first_cate_name'] = df['first_cate_name'].fillna('未知品类')
df = df.dropna(subset=['unified_event'])

# ID映射
df['user_id'] = df['user_id'].astype('category')
df['poi_id'] = df['poi_id'].astype('category')
df['first_cate_name'] = df['first_cate_name'].astype('category')

df['user_idx'] = df['user_id'].cat.codes
df['poi_idx'] = df['poi_id'].cat.codes
df['cate_idx'] = df['first_cate_name'].cat.codes

num_users = df['user_idx'].max() + 1
num_pois = df['poi_idx'].max() + 1
num_cates = df['cate_idx'].max() + 1
print(f"  用户数: {num_users:,}, POI数: {num_pois:,}, 品类数: {num_cates}")

# 时序切分
sorted_times = df['event_timestamp'].sort_values().values
split_time = sorted_times[int(len(sorted_times) * 0.8)]
train_df = df[df['event_timestamp'] <= split_time].copy()
print(f"  训练集: {len(train_df):,}行")

# ==========================================
# 2. 特征工程（与训练一致）
# ==========================================
print("\n📊 [阶段2] 特征工程...")
from sklearn.preprocessing import MinMaxScaler

poi_stats = train_df.groupby('poi_idx')['unified_event'].value_counts().unstack(fill_value=0)
poi_stats['total_pv'] = poi_stats['PV'] if 'PV' in poi_stats.columns else pd.Series(0, index=poi_stats.index)
poi_stats['total_order'] = poi_stats['ORDER'] if 'ORDER' in poi_stats.columns else pd.Series(0, index=poi_stats.index)
poi_stats['cvr'] = (poi_stats['total_order'] / (poi_stats['total_pv'] + 1e-5)).clip(upper=1.0)
poi_stats['log_pv'] = np.log1p(poi_stats['total_pv'])
poi_stats['log_order'] = np.log1p(poi_stats['total_order'])

poi_x_full = np.zeros((num_pois, 3))
poi_x_full[poi_stats.index.values] = poi_stats[['log_pv', 'log_order', 'cvr']].values
poi_x = MinMaxScaler().fit_transform(poi_x_full)

user_stats = train_df.groupby('user_idx')['unified_event'].value_counts().unstack(fill_value=0)
user_stats['total_actions'] = user_stats.sum(axis=1)
user_stats['has_ordered'] = (user_stats['ORDER'] > 0).astype(int) if 'ORDER' in user_stats.columns else 0
user_stats['log_actions'] = np.log1p(user_stats['total_actions'])

user_x_full = np.zeros((num_users, 2))
user_x_full[user_stats.index.values] = user_stats[['log_actions', 'has_ordered']].values
user_x = MinMaxScaler().fit_transform(user_x_full)

cate_stats = train_df.groupby('cate_idx').size()
cate_x_full = np.zeros((num_cates, 1))
cate_x_full[cate_stats.index.values] = np.log1p(cate_stats.values).reshape(-1, 1)
cate_x = MinMaxScaler().fit_transform(cate_x_full)

print(f"  特征维度: user={user_x.shape}, poi={poi_x.shape}, cate={cate_x.shape}")

# ==========================================
# 3. 构建图（同训练）
# ==========================================
print("\n🕸️ [阶段3] 构建异构图...")
from torch_geometric.data import HeteroData

data = HeteroData()
data['user'].x = torch.from_numpy(user_x).to(torch.float32)
data['poi'].x = torch.from_numpy(poi_x).to(torch.float32)
data['category'].x = torch.from_numpy(cate_x).to(torch.float32)

# 构建边字典并存储到data对象
edge_index_dict = {}
edge_attr_dict = {}

for event in ['PV', 'MC', 'ORDER']:
    sub_df = train_df[train_df['unified_event'] == event]
    if sub_df.empty:
        continue
    agg = sub_df.groupby(['user_idx', 'poi_idx']).size().reset_index(name='weight')
    edge_index = torch.from_numpy(agg[['user_idx', 'poi_idx']].values.T).to(torch.long)
    edge_attr = torch.log1p(torch.from_numpy(agg['weight'].values).to(torch.float32))
    edge_index_dict[('user', event.lower(), 'poi')] = edge_index
    edge_attr_dict[('user', event.lower(), 'poi')] = edge_attr
    edge_index_dict[('poi', f'rev_{event.lower()}', 'user')] = edge_index.flip([0])
    edge_attr_dict[('poi', f'rev_{event.lower()}', 'user')] = edge_attr
    data['user', event.lower(), 'poi'].edge_index = edge_index
    data['user', event.lower(), 'poi'].edge_attr = edge_attr
    data['poi', f'rev_{event.lower()}', 'user'].edge_index = edge_index.flip([0])
    data['poi', f'rev_{event.lower()}', 'user'].edge_attr = edge_attr

poi_cate_df = train_df[['poi_idx', 'cate_idx']].drop_duplicates()
poi_cate_edge_index = torch.from_numpy(poi_cate_df.values.T).to(torch.long)
edge_index_dict[('poi', 'belongs_to', 'category')] = poi_cate_edge_index
edge_index_dict[('category', 'rev_belongs_to', 'poi')] = poi_cate_edge_index.flip([0])
data['poi', 'belongs_to', 'category'].edge_index = poi_cate_edge_index
data['category', 'rev_belongs_to', 'poi'].edge_index = poi_cate_edge_index.flip([0])

# POI跳转边（用于生成测试集）
train_df_sorted = train_df.sort_values(by=['session_id', 'event_timestamp'])
max_time_train = train_df_sorted['event_timestamp'].max()
train_df_sorted['days_ago'] = (max_time_train - train_df_sorted['event_timestamp']) / (1000 * 60 * 60 * 24)
train_df_sorted['decay_weight'] = np.exp(-0.1 * train_df_sorted['days_ago'])

train_df_sorted['next_poi_idx'] = train_df_sorted.groupby('session_id')['poi_idx'].shift(-1)
trans_poi_df = train_df_sorted.dropna(subset=['next_poi_idx']).copy()
trans_poi_df = trans_poi_df[trans_poi_df['poi_idx'] != trans_poi_df['next_poi_idx']]
agg_poi = trans_poi_df.groupby(['poi_idx', 'next_poi_idx'])['decay_weight'].sum().reset_index()

if not agg_poi.empty:
    trans_poi_index = torch.from_numpy(agg_poi[['poi_idx', 'next_poi_idx']].values.T).to(torch.long)
    trans_poi_attr = torch.log1p(torch.from_numpy(agg_poi['decay_weight'].values).to(torch.float32))
    edge_index_dict[('poi', 'transition_to', 'poi')] = trans_poi_index
    edge_attr_dict[('poi', 'transition_to', 'poi')] = trans_poi_attr
    edge_index_dict[('poi', 'rev_transition_to', 'poi')] = trans_poi_index.flip([0])
    edge_attr_dict[('poi', 'rev_transition_to', 'poi')] = trans_poi_attr
    data['poi', 'transition_to', 'poi'].edge_index = trans_poi_index
    data['poi', 'transition_to', 'poi'].edge_attr = trans_poi_attr
    data['poi', 'rev_transition_to', 'poi'].edge_index = trans_poi_index.flip([0])
    data['poi', 'rev_transition_to', 'poi'].edge_attr = trans_poi_attr

print(f"  节点类型: {data.node_types}")
print(f"  边类型: {data.edge_types}")

# ==========================================
# 4. 加载模型
# ==========================================
print("\n🧠 [阶段4] 加载GNN模型...")
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, to_hetero

class BaseGAT(nn.Module):
    def __init__(self, hidden_channels, out_channels, heads=2):
        super().__init__()
        self.conv1 = GATv2Conv((-1, -1), hidden_channels, heads=heads, add_self_loops=False, edge_dim=1)
        self.conv2 = GATv2Conv((-1, -1), out_channels, heads=1, add_self_loops=False, edge_dim=1)
        self.dropout = nn.Dropout(p=0.4)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        return x

class FeatureEncoder(nn.Module):
    def __init__(self, hidden_channels, num_cates):
        super().__init__()
        self.user_lin = nn.Linear(2, hidden_channels)
        self.poi_lin = nn.Linear(3, hidden_channels)
        self.category_lin = nn.Linear(1, hidden_channels)
        self.cate_emb = nn.Embedding(num_cates, hidden_channels)
        nn.init.xavier_uniform_(self.cate_emb.weight)

    def forward(self, x_dict):
        return {
            'user': F.relu(self.user_lin(x_dict['user'])),
            'poi': F.relu(self.poi_lin(x_dict['poi'])),
            'category': F.relu(self.category_lin(x_dict['category'])) + self.cate_emb.weight
        }

class CrossAnalysisModel(nn.Module):
    def __init__(self, hidden_channels, metadata, num_cates):
        super().__init__()
        self.encoder = FeatureEncoder(hidden_channels, num_cates)
        base_model = BaseGAT(hidden_channels, hidden_channels, heads=2)
        self.gnn = to_hetero(base_model, metadata, aggr='mean')
        self.src_proj = nn.Linear(hidden_channels, hidden_channels)
        self.dst_proj = nn.Linear(hidden_channels, hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        x_dict = self.encoder(x_dict)
        if edge_attr_dict is not None:
            formatted_edge_attr = {k: (v.view(-1, 1) if v.dim() == 1 else v) for k, v in edge_attr_dict.items()}
            x_dict = self.gnn(x_dict, edge_index_dict, formatted_edge_attr)
        else:
            x_dict = self.gnn(x_dict, edge_index_dict)
        return x_dict

    def predict_link(self, src_x, dst_x):
        src_h = self.src_proj(src_x)
        dst_h = self.dst_proj(dst_x)
        return (src_h * dst_h).sum(dim=-1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  设备: {device}")

model = CrossAnalysisModel(
    hidden_channels=64,
    metadata=data.metadata(),
    num_cates=num_cates
).to(device)

# 尝试加载队友的模型
model_path = '/home/konglingrui/meituan_project/best_cross_model.pth'
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"  ✅ 成功加载模型: {model_path}")
except Exception as e:
    print(f"  ⚠️ 加载失败: {e}")
    print("  将使用随机初始化的模型进行演示")

model.eval()  # 确保推理时是eval模式

# ==========================================
# 5. 采样推理（适配16GB显存）
# ==========================================
print("\n⚡ [阶段5] 采样推理...")
SAMPLE_SIZE = 500  # 采样500个POI进行推理演示

# 获取活跃POI（训练集中有数据的POI）
active_mask = data['poi'].x[:, 0] > 0
active_poi_indices = np.where(active_mask.cpu().numpy())[0]
print(f"  活跃POI总数: {len(active_poi_indices):,}")

# 随机采样
np.random.seed(42)
sample_poi_indices = np.random.choice(active_poi_indices, size=min(SAMPLE_SIZE, len(active_poi_indices)), replace=False)
print(f"  采样POI数: {len(sample_poi_indices)}")

# 提取POI Embeddings（这是显存瓶颈，只对采样POI做）
x_dict = {
    'user': data['user'].x.to(device),
    'poi': data['poi'].x.to(device),
    'category': data['category'].x.to(device)
}
edge_index_dict_dev = {k: v.to(device) for k, v in edge_index_dict.items()}
edge_attr_dict_dev = {k: v.to(device) for k, v in edge_attr_dict.items()}

print("  正在提取POI Embeddings（显存友好模式）...")
with torch.no_grad():
    out_x = model(x_dict, edge_index_dict_dev, edge_attr_dict_dev)
    poi_embeddings = out_x['poi'].cpu()  # 转回CPU节省显存

# POI->品类映射
poi_to_cate = dict(zip(train_df['poi_idx'], train_df['first_cate_name']))

# ==========================================
# 6. Top-K推荐演示
# ==========================================
print("\n🏆 [阶段6] Top-K跨店引流推荐结果")
print("=" * 60)

K = 15
results = []

for poi_idx in sample_poi_indices[:10]:  # 展示前10个
    src_emb = poi_embeddings[poi_idx].unsqueeze(0).to(device)
    src_h = model.src_proj(src_emb)
    dst_h = model.dst_proj(poi_embeddings.to(device))

    scores = (src_h * dst_h).sum(dim=-1).cpu()

    # 过滤自己和非活跃POI
    scores[~active_mask] = -float('inf')
    scores[poi_idx] = -float('inf')

    # Top-K
    topk_scores, topk_indices = torch.topk(scores, k=K)

    src_cate = poi_to_cate.get(poi_idx, '未知')
    print(f"\n起点POI {poi_idx} [品类: {src_cate}] 的Top-{K}推荐:")

    topk_indices = topk_indices.detach().numpy()
    topk_scores = topk_scores.detach().numpy()

    for i, (tgt_idx, score) in enumerate(zip(topk_indices, topk_scores)):
        tgt_cate = poi_to_cate.get(tgt_idx, '未知')
        cross_mark = " ⬅️ 同品类" if tgt_cate == src_cate else " 🔄 跨品类"
        print(f"  {i+1:2d}. POI {tgt_idx:>8} [{tgt_cate:<6}] 得分:{score:.4f}{cross_mark}")
        results.append({
            'src_poi': poi_idx,
            'src_cate': src_cate,
            'tgt_poi': tgt_idx,
            'tgt_cate': tgt_cate,
            'score': score,
            'is_cross': tgt_cate != src_cate
        })

# ==========================================
# 7. Cross场景统计
# ==========================================
print("\n\n📈 [阶段7] Cross场景统计分析")
print("=" * 60)

results_df = pd.DataFrame(results)
print(f"\n样本总数: {len(results_df)}")
print(f"跨品类推荐占比: {results_df['is_cross'].mean()*100:.1f}%")

print("\n品类组合频次 (Top 10):")
cate_pairs = results_df[results_df['is_cross']].groupby(['src_cate', 'tgt_cate']).size().sort_values(ascending=False)
for (src, tgt), count in cate_pairs.head(10).items():
    print(f"  {src} → {tgt}: {count}")

print("\n跨品类推荐得分 vs 同品类推荐得分:")
print(f"  跨品类平均得分: {results_df[results_df['is_cross']]['score'].mean():.4f}")
print(f"  同品类平均得分: {results_df[~results_df['is_cross']]['score'].mean():.4f}")

print("\n✅ 采样推理完成！")
