import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATv2Conv, to_hetero
from torch_geometric.loader import LinkNeighborLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import time
import os

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def sample_negative_edges(active_nodes, num_samples, forbidden_edges, rng, multiplier=3):
    neg_edges = []
    sampled = set()
    active_nodes = np.asarray(active_nodes)
    while len(neg_edges) < num_samples:
        batch_size = max((num_samples - len(neg_edges)) * multiplier, 1024)
        src_batch = rng.choice(active_nodes, size=batch_size, replace=True)
        dst_batch = rng.choice(active_nodes, size=batch_size, replace=True)
        for s, d in zip(src_batch, dst_batch):
            edge = (int(s), int(d))
            if s == d or edge in forbidden_edges or edge in sampled:
                continue
            sampled.add(edge)
            neg_edges.append(edge)
            if len(neg_edges) == num_samples:
                break
    return np.asarray(neg_edges, dtype=np.int64)


start_time = time.time()
print("🚀 [阶段 1/6] 开始读取清洗全量数据...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"💻 当前运行设备: {device}")

# ==========================================
# 1. 数据读取与防警告清洗
# ==========================================
file_path = '/content/drive/MyDrive/Colab Notebooks/view_data.csv' 
usecols = ['user_id', 'poi_id', 'first_cate_name', 'event_type', 'event_id', 
           'session_id', 'event_timestamp', 'device_type', 'page_city_name']
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

# ==========================================
# 2. 全局 ID 映射与时序切分
# ==========================================
print("🚀 [阶段 2/6] 全局节点映射与时序防泄漏切分...")
df['user_id'] = df['user_id'].astype('category')
df['poi_id'] = df['poi_id'].astype('category')
df['first_cate_name'] = df['first_cate_name'].astype('category')

df['user_idx'] = df['user_id'].cat.codes
df['poi_idx'] = df['poi_id'].cat.codes
df['cate_idx'] = df['first_cate_name'].cat.codes

num_users = df['user_idx'].max() + 1
num_pois = df['poi_idx'].max() + 1
num_cates = df['cate_idx'].max() + 1

sorted_times = df['event_timestamp'].sort_values().values
split_time = sorted_times[int(len(sorted_times) * 0.8)]

train_df = df[df['event_timestamp'] <= split_time].copy()
test_df = df[df['event_timestamp'] > split_time].copy()

# ==========================================
# 3. 高阶特征工程 (仅依赖 train_df)
# ==========================================
print("🚀 [阶段 3/6] 计算高阶特征与冷启动对齐...")
poi_stats = train_df.groupby('poi_idx')['unified_event'].value_counts().unstack(fill_value=0)
poi_stats['total_pv'] = poi_stats['PV'] if 'PV' in poi_stats.columns else pd.Series(0, index=poi_stats.index)
poi_stats['total_order'] = poi_stats['ORDER'] if 'ORDER' in poi_stats.columns else pd.Series(0, index=poi_stats.index)
poi_stats['cvr'] = (poi_stats['total_order'] / (poi_stats['total_pv'] + 1e-5)).clip(upper=1.0)

poi_x_full = np.zeros((num_pois, 3))  
poi_x_full[poi_stats.index.values] = poi_stats[['total_pv', 'total_order', 'cvr']].values
poi_x = MinMaxScaler().fit(poi_stats[['total_pv', 'total_order', 'cvr']].values).transform(poi_x_full)

user_stats = train_df.groupby('user_idx')['unified_event'].value_counts().unstack(fill_value=0)
user_stats['total_actions'] = user_stats.sum(axis=1)
user_stats['has_ordered'] = (user_stats['ORDER'] > 0).astype(int) if 'ORDER' in user_stats.columns else 0

user_x_full = np.zeros((num_users, 2))  
user_x_full[user_stats.index.values] = user_stats[['total_actions', 'has_ordered']].values
user_x = MinMaxScaler().fit(user_stats[['total_actions', 'has_ordered']].values).transform(user_x_full)

cate_stats = train_df.groupby('cate_idx').size()
cate_x_full = np.zeros((num_cates, 1))
cate_x_full[cate_stats.index.values] = cate_stats.values.reshape(-1, 1)
cate_x = MinMaxScaler().fit(cate_stats.values.reshape(-1, 1)).transform(cate_x_full)

data = HeteroData()
data['user'].x = torch.from_numpy(user_x).to(torch.float)
data['poi'].x = torch.from_numpy(poi_x).to(torch.float)
data['category'].x = torch.from_numpy(cate_x).to(torch.float)

# ==========================================
# 4. 构建异构关系网络图
# ==========================================
print("🚀 [阶段 4/6] 组装多关系时序拓扑图...")
for event in ['PV', 'MC', 'ORDER']:
    sub_df = train_df[train_df['unified_event'] == event]
    if sub_df.empty: continue
    agg = sub_df.groupby(['user_idx', 'poi_idx']).size().reset_index(name='weight')
    edge_index = torch.from_numpy(agg[['user_idx', 'poi_idx']].values.T).to(torch.long)
    edge_attr = torch.log1p(torch.from_numpy(agg['weight'].values).to(torch.float))
    
    data['user', event.lower(), 'poi'].edge_index = edge_index
    data['user', event.lower(), 'poi'].edge_attr = edge_attr
    data['poi', f'rev_{event.lower()}', 'user'].edge_index = edge_index.flip([0])
    data['poi', f'rev_{event.lower()}', 'user'].edge_attr = edge_attr

poi_cate_df = train_df[['poi_idx', 'cate_idx']].drop_duplicates()
poi_cate_edge_index = torch.from_numpy(poi_cate_df.values.T).to(torch.long)
data['poi', 'belongs_to', 'category'].edge_index = poi_cate_edge_index
data['category', 'rev_belongs_to', 'poi'].edge_index = poi_cate_edge_index.flip([0])

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
    trans_poi_attr = torch.log1p(torch.from_numpy(agg_poi['decay_weight'].values).to(torch.float))
    data['poi', 'transition_to', 'poi'].edge_index = trans_poi_index
    data['poi', 'transition_to', 'poi'].edge_attr = trans_poi_attr
    data['poi', 'rev_transition_to', 'poi'].edge_index = trans_poi_index.flip([0])
    data['poi', 'rev_transition_to', 'poi'].edge_attr = trans_poi_attr

# 边特征二次全局归一化
for edge_type in data.edge_types:
    if 'edge_attr' in data[edge_type]:
        attr = data[edge_type].edge_attr
        if attr.max() > attr.min():
            data[edge_type].edge_attr = (attr - attr.min()) / (attr.max() - attr.min() + 1e-8)

# ==========================================
# 5. GNN 模型定义与实例化 (V7.0 终极提分版)
# ==========================================
print("🚀 [阶段 5/6] 实例化异构图注意力网络 (引入 CF ID Embedding)...")
class BaseGAT(nn.Module):
    def __init__(self, hidden_channels, out_channels, heads=2):
        super().__init__()
        self.conv1 = GATv2Conv((-1, -1), hidden_channels, heads=heads, add_self_loops=False, edge_dim=1)
        self.conv2 = GATv2Conv((-1, -1), out_channels, heads=1, add_self_loops=False, edge_dim=1)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        return x

class FeatureEncoder(nn.Module):
    # 💡 杀手锏 1：引入全量节点的 ID 维度，构建 Embedding 层
    def __init__(self, hidden_channels, num_users, num_pois, num_cates):
        super().__init__()
        self.user_lin = nn.Linear(2, hidden_channels)
        self.poi_lin = nn.Linear(3, hidden_channels)
        self.category_lin = nn.Linear(1, hidden_channels)
        
        # 可学习的专属 ID 向量，这是推荐系统记住用户和商家偏好的核心！
        self.user_emb = nn.Embedding(num_users, hidden_channels)
        self.poi_emb = nn.Embedding(num_pois, hidden_channels)
        self.cate_emb = nn.Embedding(num_cates, hidden_channels)
        
        # 初始化权重
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.poi_emb.weight)
        nn.init.xavier_uniform_(self.cate_emb.weight)

    def forward(self, x_dict):
        # 💡 将统计特征的映射结果，与专属的 ID Embedding 相加
        return {
            'user': F.relu(self.user_lin(x_dict['user'])) + self.user_emb.weight,
            'poi': F.relu(self.poi_lin(x_dict['poi'])) + self.poi_emb.weight,
            'category': F.relu(self.category_lin(x_dict['category'])) + self.cate_emb.weight
        }

class CrossAnalysisModel(nn.Module):
    def __init__(self, hidden_channels, metadata, num_users, num_pois, num_cates):
        super().__init__()
        self.encoder = FeatureEncoder(hidden_channels, num_users, num_pois, num_cates)
        base_model = BaseGAT(hidden_channels, hidden_channels, heads=2)
        self.gnn = to_hetero(base_model, metadata, aggr='mean')
        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_channels * 4, hidden_channels),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        x_dict = self.encoder(x_dict)
        if edge_attr_dict is not None:
            formatted_edge_attr = {k: (v.view(-1, 1) if v.dim() == 1 else v) for k, v in edge_attr_dict.items()}
            x_dict = self.gnn(x_dict, edge_index_dict, formatted_edge_attr)
        else:
            x_dict = self.gnn(x_dict, edge_index_dict)
        return x_dict

    def predict_link(self, src_x, dst_x):
        pair_x = torch.cat([src_x, dst_x, torch.abs(src_x - dst_x), src_x * dst_x], dim=-1)
        return self.link_predictor(pair_x).squeeze(-1)

# 注意这里传入了 num_users, num_pois, num_cates
model = CrossAnalysisModel(
    hidden_channels=64, 
    metadata=data.metadata(),
    num_users=num_users,
    num_pois=num_pois,
    num_cates=num_cates
).to(device)

x_dict = {node_type: data[node_type].x.to(device) for node_type in data.node_types}
edge_index_dict = {edge_type: data[edge_type].edge_index.to(device) for edge_type in data.edge_types}
edge_attr_dict = {
    edge_type: data[edge_type].edge_attr.to(device)
    for edge_type in data.edge_types
    if 'edge_attr' in data[edge_type]
}


# ==========================================
# 6. 全图模式硬核训练 (修复分布偏移的真·终极版)
# ==========================================
print("🚀 [阶段 6/6] 启动全图模式训练...")

target_edge_type = ('poi', 'transition_to', 'poi')
reverse_target_edge_type = ('poi', 'rev_transition_to', 'poi')
if target_edge_type not in data.edge_types:
    raise RuntimeError("缺少 poi->poi 的 transition_to 边，无法进行当前链路预测任务。")

full_target_edge_index = data[target_edge_type].edge_index
full_target_edge_attr = data[target_edge_type].edge_attr
num_target_edges = full_target_edge_index.size(1)
perm = torch.randperm(num_target_edges)
message_edge_count = max(int(num_target_edges * 0.8), 1)
message_perm = perm[:message_edge_count]
supervision_perm = perm[message_edge_count:]
if supervision_perm.numel() == 0:
    supervision_perm = message_perm[-1:].clone()
    message_perm = message_perm[:-1]

message_edge_index = full_target_edge_index[:, message_perm]
message_edge_attr = full_target_edge_attr[message_perm]
train_pos_edge_index = full_target_edge_index[:, supervision_perm].to(device)

edge_index_dict[target_edge_type] = message_edge_index.to(device)
edge_attr_dict[target_edge_type] = message_edge_attr.to(device)
edge_index_dict[reverse_target_edge_type] = message_edge_index.flip([0]).to(device)
edge_attr_dict[reverse_target_edge_type] = message_edge_attr.to(device)

active_pois = torch.unique(full_target_edge_index.flatten()).to(device)
active_pois_np = active_pois.cpu().numpy()
rng = np.random.default_rng(SEED)

print("正在抽取同分布的验证集负样本...")
test_df_sorted = test_df.sort_values(by=['session_id', 'event_timestamp'])
test_df_sorted['next_poi_idx'] = test_df_sorted.groupby('session_id')['poi_idx'].shift(-1)
val_pos_df = test_df_sorted.dropna(subset=['next_poi_idx']).copy()
val_pos_edges = val_pos_df[val_pos_df['poi_idx'] != val_pos_df['next_poi_idx']][['poi_idx', 'next_poi_idx']].drop_duplicates().values

num_val_pos = len(val_pos_edges)
train_message_set = set(map(tuple, message_edge_index.t().tolist()))
train_pos_set = set(map(tuple, train_pos_edge_index.t().tolist()))
val_pos_set = set(map(tuple, val_pos_edges.tolist()))
all_train_positive_set = train_message_set | train_pos_set

val_neg_edges = sample_negative_edges(
    active_pois_np,
    num_val_pos,
    all_train_positive_set | val_pos_set,
    rng
)

val_edge_index = torch.tensor(np.vstack([val_pos_edges, val_neg_edges]).T, dtype=torch.long).to(device)
val_edge_label = torch.tensor([1]*num_val_pos + [0]*num_val_pos, dtype=torch.float).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

best_auc = 0
epochs = 40 

pos_edge_index = train_pos_edge_index
num_pos_edges = pos_edge_index.size(1)

print("\n🔥 >>>>> 开启满血全图训练 (修复分布偏差，双 AUC 监控) <<<<< 🔥")
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    out_x = model(x_dict, edge_index_dict, edge_attr_dict)
    
    # 提取正样本预测
    src_pos = pos_edge_index[0]
    dst_pos = pos_edge_index[1]
    pos_preds = model.predict_link(out_x['poi'][src_pos], out_x['poi'][dst_pos])
    
    neg_edges = sample_negative_edges(
        active_pois_np,
        num_pos_edges,
        all_train_positive_set,
        rng
    )
    neg_edge_index = torch.from_numpy(neg_edges.T).to(device)
    neg_src = neg_edge_index[0]
    neg_dst = neg_edge_index[1]
    
    neg_preds = model.predict_link(out_x['poi'][neg_src], out_x['poi'][neg_dst])
    
    preds = torch.cat([pos_preds, neg_preds])
    labels = torch.cat([torch.ones_like(pos_preds), torch.zeros_like(neg_preds)])
    loss = F.binary_cross_entropy_with_logits(preds, labels)
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    # 计算当前 Epoch 的 Train AUC
    with torch.no_grad():
        train_preds_prob = torch.sigmoid(preds).cpu().numpy()
        train_labels_np = labels.cpu().numpy()
        train_auc = roc_auc_score(train_labels_np, train_preds_prob)
    
    # 评估验证集 Val AUC
    model.eval()
    with torch.no_grad():
        eval_out_x = model(x_dict, edge_index_dict, edge_attr_dict)
        val_src = val_edge_index[0]
        val_dst = val_edge_index[1]
        val_preds = model.predict_link(eval_out_x['poi'][val_src], eval_out_x['poi'][val_dst])
        
        val_preds_prob = torch.sigmoid(val_preds).cpu().numpy()
        val_auc = roc_auc_score(val_edge_label.cpu().numpy(), val_preds_prob)
        
    print(f"👉 Epoch {epoch+1:02d}/{epochs} | Loss: {loss.item():.4f} | Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f}")
    scheduler.step(val_auc)
    
    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.state_dict(), "best_cross_model.pth")

print(f"\n🎉 恭喜！全流程跑通，最高验证集 AUC 达到: {best_auc:.4f}")
