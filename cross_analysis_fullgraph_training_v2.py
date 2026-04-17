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

start_time = time.time()
print("🚀 [阶段 1/6] 开始读取清洗全量数据...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"💻 当前运行设备: {device}")

# ==========================================
# 1. 数据读取与防警告清洗
# ==========================================
file_path = '/home/konglingrui/meituan_project/view_data.csv'
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
# 3. 高阶特征工程 (V2.0 Log平滑破除长尾挤压)
# ==========================================
print("🚀 [阶段 3/6] 计算高阶特征 (引入 Log 平滑破除长尾挤压)...")
poi_stats = train_df.groupby('poi_idx')['unified_event'].value_counts().unstack(fill_value=0)
poi_stats['total_pv'] = poi_stats['PV'] if 'PV' in poi_stats.columns else pd.Series(0, index=poi_stats.index)
poi_stats['total_order'] = poi_stats['ORDER'] if 'ORDER' in poi_stats.columns else pd.Series(0, index=poi_stats.index)
poi_stats['cvr'] = (poi_stats['total_order'] / (poi_stats['total_pv'] + 1e-5)).clip(upper=1.0)

# 💡 核心修复：对长尾分布进行 Log1p 对数变换，放大中小商家的差异！
poi_stats['log_pv'] = np.log1p(poi_stats['total_pv'])
poi_stats['log_order'] = np.log1p(poi_stats['total_order'])

poi_x_full = np.zeros((num_pois, 3))
poi_x_full[poi_stats.index.values] = poi_stats[['log_pv', 'log_order', 'cvr']].values
poi_x = MinMaxScaler().fit(poi_x_full).transform(poi_x_full)

# 用户特征同理平滑处理
user_stats = train_df.groupby('user_idx')['unified_event'].value_counts().unstack(fill_value=0)
user_stats['total_actions'] = user_stats.sum(axis=1)
user_stats['has_ordered'] = (user_stats['ORDER'] > 0).astype(int) if 'ORDER' in user_stats.columns else 0

user_stats['log_actions'] = np.log1p(user_stats['total_actions'])
user_x_full = np.zeros((num_users, 2))
user_x_full[user_stats.index.values] = user_stats[['log_actions', 'has_ordered']].values
user_x = MinMaxScaler().fit(user_x_full).transform(user_x_full)

# 品类特征同样平滑
cate_stats = train_df.groupby('cate_idx').size()
cate_x_full = np.zeros((num_cates, 1))
cate_x_full[cate_stats.index.values] = np.log1p(cate_stats.values).reshape(-1, 1)
cate_x = MinMaxScaler().fit(cate_x_full).transform(cate_x_full)

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

# 构建边字典（Colab notebook中预定义，本地脚本需显式构造）
edge_index_dict = {et: data[et].edge_index for et in data.edge_types}
edge_attr_dict = {}
for et in data.edge_types:
    if hasattr(data[et], 'edge_attr') and data[et].edge_attr is not None:
        edge_attr_dict[et] = data[et].edge_attr
    else:
        edge_attr_dict[et] = torch.ones(data[et].edge_index.size(1))

# ==========================================
# 5. GNN 模型定义 (V10.0 终极版：双线性内积抗过拟合)
# ==========================================
print("🚀 [阶段 5/6] 实例化异构图注意力网络 (启用双线性内积预测头)...")


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

        # 💡 核心大招：非对称投影层！
        # 让源节点(起点)和目标节点(终点)先经过各自的专属通道，打破对称性
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
        # 💡 双线性点积：既保留了数学上的空间相似度，又兼顾了方向差异！
        src_h = self.src_proj(src_x)
        dst_h = self.dst_proj(dst_x)
        return (src_h * dst_h).sum(dim=-1)


model = CrossAnalysisModel(
    hidden_channels=64,
    metadata=data.metadata(),
    num_cates=num_cates
).to(device)

# ==========================================
# 6. 全图模式硬核训练 (彻底解决 Target Leakage)
# ==========================================
print("🚀 [阶段 6/6] 启动全图模式训练 (应用 Edge Split 阻断信息穿越)...")

target_edge_type = ('poi', 'transition_to', 'poi')
rev_target_edge_type = ('poi', 'rev_transition_to', 'poi')

# 提取全体活跃 POI
pos_edge_index = edge_index_dict[target_edge_type]
active_pois = torch.unique(pos_edge_index.flatten()).to(device)
active_pois_np = active_pois.cpu().numpy()

# --------------------------------------------------
# 1. 极速构建验证集
# --------------------------------------------------
test_df_sorted = test_df.sort_values(by=['session_id', 'event_timestamp'])
test_df_sorted['next_poi_idx'] = test_df_sorted.groupby('session_id')['poi_idx'].shift(-1)
val_pos_df = test_df_sorted.dropna(subset=['next_poi_idx']).copy()
val_pos_edges = val_pos_df[val_pos_df['poi_idx'] != val_pos_df['next_poi_idx']][
    ['poi_idx', 'next_poi_idx']].drop_duplicates().values

num_val_pos = len(val_pos_edges)
train_pos_set = set(map(tuple, pos_edge_index.t().tolist()))
val_pos_set = set(map(tuple, val_pos_edges.tolist()))

val_neg_src = np.random.choice(active_pois_np, size=num_val_pos * 3)
val_neg_dst = np.random.choice(active_pois_np, size=num_val_pos * 3)

val_neg_edges = []
for s, d in zip(val_neg_src, val_neg_dst):
    if s == d or (s, d) in train_pos_set or (s, d) in val_pos_set:
        continue
    val_neg_edges.append((s, d))
    if len(val_neg_edges) == num_val_pos:
        break

val_edge_index = torch.tensor(np.vstack([val_pos_edges, np.array(val_neg_edges)]).T, dtype=torch.long).to(device)
val_edge_label = torch.tensor([1] * num_val_pos + [0] * num_val_pos, dtype=torch.float).to(device)

# --------------------------------------------------
# 💡 2. 核心修复：防止信息泄露 (切割 70% 结构边 / 30% 监督边)
# --------------------------------------------------
num_pos_edges = pos_edge_index.size(1)
perm = torch.randperm(num_pos_edges, device=device)
split_idx = int(num_pos_edges * 0.7)
msg_idx = perm[:split_idx]  # 70% 消息传递
sup_idx = perm[split_idx:]  # 30% 目标预测

# 【模型看的图】构建训练时使用的 70% 消息传递图
train_msg_edge_index_dict = {k: v.to(device) for k, v in edge_index_dict.items()}
train_msg_edge_attr_dict = {k: v.to(device) for k, v in edge_attr_dict.items()}
pos_edge_index = pos_edge_index.to(device)

train_msg_edge_index_dict[target_edge_type] = pos_edge_index[:, msg_idx]
train_msg_edge_attr_dict[target_edge_type] = edge_attr_dict[target_edge_type][msg_idx]
# 反向边必须同步隐藏！
train_msg_edge_index_dict[rev_target_edge_type] = edge_index_dict[rev_target_edge_type][:, msg_idx]
train_msg_edge_attr_dict[rev_target_edge_type] = edge_attr_dict[rev_target_edge_type][msg_idx]

# 【模型要猜的边】用于计算 Loss 的 30% 真实目标边
sup_pos_edge_index = pos_edge_index[:, sup_idx]
num_sup_edges = sup_pos_edge_index.size(1)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

best_auc = 0
epochs = 40

print("\n🔥 >>>>> 开启满血全图训练 (没收答案，硬核协同过滤) <<<<< 🔥")
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    # 💡 亮点 1：前向传播只输入 70% 的边，强迫模型跨越隐藏图谱学习特征！
    out_x = model(x_dict, train_msg_edge_index_dict, train_msg_edge_attr_dict)

    # 对被隐藏的 30% 边进行预测打分
    src_pos = sup_pos_edge_index[0]
    dst_pos = sup_pos_edge_index[1]
    pos_preds = model.predict_link(out_x['poi'][src_pos], out_x['poi'][dst_pos])

    # 负采样 (数量与 30% 的监督边对齐)
    rand_idx_src = torch.randint(0, len(active_pois), (num_sup_edges,), device=device)
    rand_idx_dst = torch.randint(0, len(active_pois), (num_sup_edges,), device=device)
    neg_src = active_pois[rand_idx_src]
    neg_dst = active_pois[rand_idx_dst]

    neg_preds = model.predict_link(out_x['poi'][neg_src], out_x['poi'][neg_dst])

    preds = torch.cat([pos_preds, neg_preds])
    labels = torch.cat([torch.ones_like(pos_preds), torch.zeros_like(neg_preds)])
    loss = F.binary_cross_entropy_with_logits(preds, labels)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    with torch.no_grad():
        train_preds_prob = torch.sigmoid(preds).cpu().numpy()
        train_labels_np = labels.cpu().numpy()
        train_auc = roc_auc_score(train_labels_np, train_preds_prob)

    # 💡 亮点 2：验证阶段恢复 100% 训练边作为结构图，去预测全新的验证边
    model.eval()
    with torch.no_grad():
        val_out_x = model(x_dict, edge_index_dict, edge_attr_dict)
        val_src = val_edge_index[0]
        val_dst = val_edge_index[1]
        val_preds = model.predict_link(val_out_x['poi'][val_src], val_out_x['poi'][val_dst])

        val_preds_prob = torch.sigmoid(val_preds).cpu().numpy()
        val_auc = roc_auc_score(val_edge_label.cpu().numpy(), val_preds_prob)

    print(
        f"👉 Epoch {epoch + 1:02d}/{epochs} | Loss: {loss.item():.4f} | Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f}")
    scheduler.step(val_auc)

    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.state_dict(), "best_cross_model.pth")

print(f"\n🎉 恭喜！跨越所有陷阱，最高验证集 AUC 达到: {best_auc:.4f}")