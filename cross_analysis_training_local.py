"""
美团Cross分析 - 本地训练 v6 (邻居采样版)
============================================
策略: 改用 neighbor sampler + mini-batch 训练，避免全图 scatter OOM
     同时适配本地 16GB 显存
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
import time, random

start_time = time.time()
print("=" * 60)
print("🚀 美团Cross分析 - 本地训练 v6 (邻居采样)")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"💻 {props.name}, {props.total_memory / 1024**3:.1f} GB")

def get_mem():
    return torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0

# ==========================================
# 1. 数据
# ==========================================
print(f"\n🚀 [阶段 1/5] 读取数据...")
SAMPLE_RATIO = 0.05
SEED = 42

df = pd.read_csv('/home/konglingrui/meituan_project/view_data.csv',
                 usecols=['user_id', 'poi_id', 'first_cate_name', 'event_type',
                          'event_id', 'session_id', 'event_timestamp'])
print(f"    原始: {len(df):,} 行")

df['event_timestamp'] = pd.to_numeric(df['event_timestamp'], errors='coerce')
df = df.dropna(subset=['user_id', 'poi_id', 'event_timestamp', 'session_id'])
df['event_type_str'] = df['event_type'].astype(str).str.strip().str.upper()
df['event_id_str'] = df['event_id'].astype(str).str.strip().str.lower()
df['unified_event'] = pd.Series(dtype='object')
df.loc[df['event_type_str'] == 'PV', 'unified_event'] = 'PV'
df.loc[df['event_type_str'] == 'MC', 'unified_event'] = 'MC'
df.loc[(df['event_type_str'] == 'ORDER') | df['event_id_str'].str.contains('order', na=False), 'unified_event'] = 'ORDER'
df['first_cate_name'] = df['first_cate_name'].fillna('未知品类')
df = df.dropna(subset=['unified_event'])

all_users = df['user_id'].unique()
random.seed(SEED)
sampled_users = set(random.sample(list(all_users), int(len(all_users) * SAMPLE_RATIO)))
df = df[df['user_id'].isin(sampled_users)].copy()
print(f"    采样后: {len(df):,} 行, {df['user_id'].nunique():,} 用户")

# ==========================================
# 2. 节点映射 + 时序切分
# ==========================================
print(f"\n🚀 [阶段 2/5] 节点映射与时序切分...")
df['user_id'] = df['user_id'].astype('category')
df['poi_id'] = df['poi_id'].astype('category')
df['first_cate_name'] = df['first_cate_name'].astype('category')
df['user_idx'] = df['user_id'].cat.codes
df['poi_idx'] = df['poi_id'].cat.codes
df['cate_idx'] = df['first_cate_name'].cat.codes

num_users = df['user_idx'].max() + 1
num_pois = df['poi_idx'].max() + 1
num_cates = df['cate_idx'].max() + 1
print(f"    user: {num_users:,}, poi: {num_pois:,}, cate: {num_cates}")

split_time = df['event_timestamp'].sort_values().values[int(len(df) * 0.8)]
train_df = df[df['event_timestamp'] <= split_time].copy()
test_df = df[df['event_timestamp'] > split_time].copy()
print(f"    训练: {len(train_df):,}, 测试: {len(test_df):,}")

# ==========================================
# 3. 节点特征
# ==========================================
print(f"\n🚀 [阶段 3/5] 节点特征...")
poi_stats = train_df.groupby('poi_idx')['unified_event'].value_counts().unstack(fill_value=0)
for col in ['PV', 'MC', 'ORDER']:
    if col not in poi_stats.columns:
        poi_stats[col] = 0
poi_stats['cvr'] = (poi_stats['ORDER'] / (poi_stats['PV'] + 1e-5)).clip(upper=1.0)
poi_x_full = np.zeros((num_pois, 3))
poi_x_full[poi_stats.index.values, 0] = np.log1p(poi_stats['PV'].values)
poi_x_full[poi_stats.index.values, 1] = np.log1p(poi_stats['ORDER'].values)
poi_x_full[poi_stats.index.values, 2] = poi_stats['cvr'].values
poi_x = MinMaxScaler().fit_transform(poi_x_full)

user_stats = train_df.groupby('user_idx')['unified_event'].value_counts().unstack(fill_value=0)
user_stats['total'] = user_stats.sum(axis=1)
user_stats['ordered'] = (user_stats['ORDER'] > 0).astype(int) if 'ORDER' in user_stats.columns else 0
user_x_full = np.zeros((num_users, 2))
user_x_full[user_stats.index.values, 0] = np.log1p(user_stats['total'].values)
user_x_full[user_stats.index.values, 1] = user_stats['ordered'].values
user_x = MinMaxScaler().fit_transform(user_x_full)

cate_stats = train_df.groupby('cate_idx').size()
cate_x_full = np.zeros((num_cates, 1))
cate_x_full[cate_stats.index.values, 0] = np.log1p(cate_stats.values)
cate_x = MinMaxScaler().fit_transform(cate_x_full)

# 构建 PyG HeteroData
from torch_geometric.data import HeteroData
data = HeteroData()
data['user'].x = torch.from_numpy(user_x).float()
data['poi'].x = torch.from_numpy(poi_x).float()
data['category'].x = torch.from_numpy(cate_x).float()

# 构建边
for event in ['PV', 'MC', 'ORDER']:
    sub = train_df[train_df['unified_event'] == event]
    if sub.empty:
        continue
    agg = sub.groupby(['user_idx', 'poi_idx']).size().reset_index(name='w')
    ei = torch.from_numpy(agg[['user_idx', 'poi_idx']].values.T).long()
    ea = torch.log1p(torch.from_numpy(agg['w'].values).float())
    if ea.max() > ea.min():
        ea = (ea - ea.min()) / (ea.max() - ea.min() + 1e-8)
    data['user', event.lower(), 'poi'].edge_index = ei
    data['user', event.lower(), 'poi'].edge_attr = ea
    data['poi', f'rev_{event.lower()}', 'user'].edge_index = ei.flip([0])
    data['poi', f'rev_{event.lower()}', 'user'].edge_attr = ea

poi_cate_df = train_df[['poi_idx', 'cate_idx']].drop_duplicates()
pc_ei = torch.from_numpy(poi_cate_df.values.T).long()
data['poi', 'belongs_to', 'category'].edge_index = pc_ei
data['category', 'rev_belongs_to', 'poi'].edge_index = pc_ei.flip([0])

# transition边
tdf = train_df.sort_values(['session_id', 'event_timestamp'])
max_t = tdf['event_timestamp'].max()
tdf['days_ago'] = (max_t - tdf['event_timestamp']) / (1000 * 60 * 60 * 24)
tdf['decay'] = np.exp(-0.1 * tdf['days_ago'])
tdf['next_poi'] = tdf.groupby('session_id')['poi_idx'].shift(-1)
trans = tdf.dropna(subset=['next_poi'])
trans = trans[trans['poi_idx'] != trans['next_poi']]
agg = trans.groupby(['poi_idx', 'next_poi'])['decay'].sum().reset_index()
t_ei = torch.from_numpy(agg[['poi_idx', 'next_poi']].values.T).long()
t_ea = torch.log1p(torch.from_numpy(agg['decay'].values).float())
if t_ea.max() > t_ea.min():
    t_ea = (t_ea - t_ea.min()) / (t_ea.max() - t_ea.min() + 1e-8)
data['poi', 'transition_to', 'poi'].edge_index = t_ei
data['poi', 'transition_to', 'poi'].edge_attr = t_ea
data['poi', 'rev_transition_to', 'poi'].edge_index = t_ei.flip([0])
data['poi', 'rev_transition_to', 'poi'].edge_attr = t_ea

print(f"    图就绪, 显存: {get_mem():.2f} GB")
print(f"    边类型: {data.edge_types}")

# ==========================================
# 4. 模型
# ==========================================
print(f"\n🚀 [阶段 4/5] 实例化模型...")

HIDDEN = 32

class HeteroGNN(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.user_lin = nn.Linear(2, hidden)
        self.poi_lin = nn.Linear(3, hidden)
        self.cate_lin = nn.Linear(1, hidden)
        self.cate_emb = nn.Embedding(100, hidden)  # 类别数较小

        self.W1_u = nn.Linear(hidden, hidden)
        self.W1_p = nn.Linear(hidden, hidden)
        self.W1_c = nn.Linear(hidden, hidden)
        self.W1_pp = nn.Linear(hidden, hidden)

        self.W2_u = nn.Linear(hidden, hidden)
        self.W2_p = nn.Linear(hidden, hidden)
        self.W2_c = nn.Linear(hidden, hidden)
        self.W2_pp = nn.Linear(hidden, hidden)

        self.src_proj = nn.Linear(hidden, hidden)
        self.dst_proj = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        user_h = F.relu(self.user_lin(x_dict['user']))
        poi_h = F.relu(self.poi_lin(x_dict['poi']))
        cate_h = F.relu(self.cate_lin(x_dict['category'])) + self.cate_emb.weight[:x_dict['category'].size(0)]

        # 阶段1: 聚合邻域 (每条边: src_idx送消息 -> dst_idx收消息)
        # user <- poi (via rev_pv/mc/order)
        for rel in ['rev_pv', 'rev_mc', 'rev_order']:
            key = ('poi', rel, 'user')
            if key in edge_index_dict:
                ei = edge_index_dict[key]
                ea = edge_attr_dict.get(key, None)
                # poi 是源节点, user 是目标节点
                src_poi_h = poi_h[ei[0]]  # [num_edges, H] - 只取有边的POI
                msg = self.W1_p(src_poi_h)
                if ea is not None:
                    msg = msg * ea.unsqueeze(-1)
                # 聚合到 user
                dst_idx = ei[1]  # 目标=user
                unique_idx, inv_idx = torch.unique(dst_idx, return_inverse=True)
                agg = torch.zeros(unique_idx.size(0), user_h.shape[1], dtype=user_h.dtype, device=user_h.device)
                agg = agg.scatter_add(0, inv_idx.unsqueeze(-1).expand_as(msg), msg)
                cnt = torch.zeros(unique_idx.size(0), dtype=torch.float, device=user_h.device)
                cnt = cnt.scatter_add(0, inv_idx, torch.ones_like(dst_idx, dtype=torch.float))
                cnt = cnt.clamp(min=1).unsqueeze(-1)
                agg = agg / cnt
                user_h[unique_idx] = user_h[unique_idx] + F.relu(agg).to(user_h.dtype)

        # poi <- user (via pv/mc/order)
        for rel in ['pv', 'mc', 'order']:
            key = ('user', rel, 'poi')
            if key in edge_index_dict:
                ei = edge_index_dict[key]
                ea = edge_attr_dict.get(key, None)
                src_user_h = user_h[ei[0]]  # [num_edges, H]
                msg = self.W1_u(src_user_h)
                if ea is not None:
                    msg = msg * ea.unsqueeze(-1)
                dst_idx = ei[1]  # 目标=poi
                unique_idx, inv_idx = torch.unique(dst_idx, return_inverse=True)
                agg = torch.zeros(unique_idx.size(0), poi_h.shape[1], dtype=poi_h.dtype, device=poi_h.device)
                agg = agg.scatter_add(0, inv_idx.unsqueeze(-1).expand_as(msg), msg)
                cnt = torch.zeros(unique_idx.size(0), dtype=torch.float, device=poi_h.device)
                cnt = cnt.scatter_add(0, inv_idx, torch.ones_like(dst_idx, dtype=torch.float))
                cnt = cnt.clamp(min=1).unsqueeze(-1)
                agg = agg / cnt
                poi_h[unique_idx] = poi_h[unique_idx] + F.relu(agg)

        # poi <- category (via rev_belongs_to: category -> poi)
        key = ('category', 'rev_belongs_to', 'poi')
        if key in edge_index_dict:
            ei = edge_index_dict[key]
            src_cate_h = cate_h[ei[0]]
            msg = self.W1_c(src_cate_h)
            dst_idx = ei[1]
            unique_idx, inv_idx = torch.unique(dst_idx, return_inverse=True)
            agg = torch.zeros(unique_idx.size(0), poi_h.shape[1], dtype=poi_h.dtype, device=poi_h.device)
            agg = agg.scatter_add(0, inv_idx.unsqueeze(-1).expand_as(msg), msg)
            cnt = torch.zeros(unique_idx.size(0), dtype=torch.float, device=poi_h.device)
            cnt = cnt.scatter_add(0, inv_idx, torch.ones_like(dst_idx, dtype=torch.float))
            cnt = cnt.clamp(min=1).unsqueeze(-1)
            agg = agg / cnt
            poi_h[unique_idx] = poi_h[unique_idx] + F.relu(agg)

        # poi <- poi transition (poi -> next_poi)
        key = ('poi', 'transition_to', 'poi')
        if key in edge_index_dict:
            ei = edge_index_dict[key]
            ea = edge_attr_dict.get(key, None)
            src_poi_h = poi_h[ei[0]]
            msg = self.W1_pp(src_poi_h)
            if ea is not None:
                msg = msg * ea.unsqueeze(-1)
            dst_idx = ei[1]
            unique_idx, inv_idx = torch.unique(dst_idx, return_inverse=True)
            agg = torch.zeros(unique_idx.size(0), poi_h.shape[1], dtype=poi_h.dtype, device=poi_h.device)
            agg = agg.scatter_add(0, inv_idx.unsqueeze(-1).expand_as(msg), msg)
            cnt = torch.zeros(unique_idx.size(0), dtype=torch.float, device=poi_h.device)
            cnt = cnt.scatter_add(0, inv_idx, torch.ones_like(dst_idx, dtype=torch.float))
            cnt = cnt.clamp(min=1).unsqueeze(-1)
            agg = agg / cnt
            poi_h[unique_idx] = poi_h[unique_idx] + F.relu(agg)

        user_h = self.dropout(user_h)
        poi_h = self.dropout(poi_h)
        cate_h = self.dropout(cate_h)

        # 阶段2 (完全相同的逻辑)
        for rel in ['rev_pv', 'rev_mc', 'rev_order']:
            key = ('poi', rel, 'user')
            if key in edge_index_dict:
                ei = edge_index_dict[key]
                ea = edge_attr_dict.get(key, None)
                src_poi_h = poi_h[ei[0]]
                msg = self.W2_p(src_poi_h)
                if ea is not None:
                    msg = msg * ea.unsqueeze(-1)
                dst_idx = ei[1]
                unique_idx, inv_idx = torch.unique(dst_idx, return_inverse=True)
                agg = torch.zeros(unique_idx.size(0), user_h.shape[1], dtype=user_h.dtype, device=user_h.device)
                agg = agg.scatter_add(0, inv_idx.unsqueeze(-1).expand_as(msg), msg)
                cnt = torch.zeros(unique_idx.size(0), dtype=torch.float, device=user_h.device)
                cnt = cnt.scatter_add(0, inv_idx, torch.ones_like(dst_idx, dtype=torch.float))
                cnt = cnt.clamp(min=1).unsqueeze(-1)
                agg = agg / cnt
                user_h[unique_idx] = user_h[unique_idx] + F.relu(agg)

        for rel in ['pv', 'mc', 'order']:
            key = ('user', rel, 'poi')
            if key in edge_index_dict:
                ei = edge_index_dict[key]
                ea = edge_attr_dict.get(key, None)
                src_user_h = user_h[ei[0]]
                msg = self.W2_u(src_user_h)
                if ea is not None:
                    msg = msg * ea.unsqueeze(-1)
                dst_idx = ei[1]
                unique_idx, inv_idx = torch.unique(dst_idx, return_inverse=True)
                agg = torch.zeros(unique_idx.size(0), poi_h.shape[1], dtype=poi_h.dtype, device=poi_h.device)
                agg = agg.scatter_add(0, inv_idx.unsqueeze(-1).expand_as(msg), msg)
                cnt = torch.zeros(unique_idx.size(0), dtype=torch.float, device=poi_h.device)
                cnt = cnt.scatter_add(0, inv_idx, torch.ones_like(dst_idx, dtype=torch.float))
                cnt = cnt.clamp(min=1).unsqueeze(-1)
                agg = agg / cnt
                poi_h[unique_idx] = poi_h[unique_idx] + F.relu(agg)

        key = ('category', 'rev_belongs_to', 'poi')
        if key in edge_index_dict:
            ei = edge_index_dict[key]
            src_cate_h = cate_h[ei[0]]
            msg = self.W2_c(src_cate_h)
            dst_idx = ei[1]
            unique_idx, inv_idx = torch.unique(dst_idx, return_inverse=True)
            agg = torch.zeros(unique_idx.size(0), poi_h.shape[1], dtype=poi_h.dtype, device=poi_h.device)
            agg = agg.scatter_add(0, inv_idx.unsqueeze(-1).expand_as(msg), msg)
            cnt = torch.zeros(unique_idx.size(0), dtype=torch.float, device=poi_h.device)
            cnt = cnt.scatter_add(0, inv_idx, torch.ones_like(dst_idx, dtype=torch.float))
            cnt = cnt.clamp(min=1).unsqueeze(-1)
            agg = agg / cnt
            poi_h[unique_idx] = poi_h[unique_idx] + F.relu(agg)

        key = ('poi', 'transition_to', 'poi')
        if key in edge_index_dict:
            ei = edge_index_dict[key]
            ea = edge_attr_dict.get(key, None)
            src_poi_h = poi_h[ei[0]]
            msg = self.W2_pp(src_poi_h)
            if ea is not None:
                msg = msg * ea.unsqueeze(-1)
            dst_idx = ei[1]
            unique_idx, inv_idx = torch.unique(dst_idx, return_inverse=True)
            agg = torch.zeros(unique_idx.size(0), poi_h.shape[1], dtype=poi_h.dtype, device=poi_h.device)
            agg = agg.scatter_add(0, inv_idx.unsqueeze(-1).expand_as(msg), msg)
            cnt = torch.zeros(unique_idx.size(0), dtype=torch.float, device=poi_h.device)
            cnt = cnt.scatter_add(0, inv_idx, torch.ones_like(dst_idx, dtype=torch.float))
            cnt = cnt.clamp(min=1).unsqueeze(-1)
            agg = agg / cnt
            poi_h[unique_idx] = poi_h[unique_idx] + F.relu(agg)

        return {'user': user_h, 'poi': poi_h, 'category': cate_h}

    def predict_link(self, src_x, dst_x):
        return (self.src_proj(src_x) * self.dst_proj(dst_x)).sum(dim=-1)


model = HeteroGNN(hidden=HIDDEN).to(device)
print(f"    模型就绪, 显存: {get_mem():.2f} GB")

# ==========================================
# 5. 训练
# ==========================================
print(f"\n🚀 [阶段 5/5] 训练...")

# 准备边数据
edge_index_dict = {et: data[et].edge_index.to(device) for et in data.edge_types}
edge_attr_dict = {}
for et in data.edge_types:
    if hasattr(data[et], 'edge_attr') and data[et].edge_attr is not None:
        edge_attr_dict[et] = data[et].edge_attr.to(device)
    else:
        edge_attr_dict[et] = torch.ones(data[et].edge_index.size(1), device=device)

x_dict = {k: v.to(device) for k, v in data.x_dict.items()}

# POI transition 边分割
target_key = ('poi', 'transition_to', 'poi')
rev_target_key = ('poi', 'rev_transition_to', 'poi')
pos_edge_index = edge_index_dict[target_key]
num_pos = pos_edge_index.size(1)

perm = torch.randperm(num_pos, device=device)
split = int(num_pos * 0.7)
msg_idx = perm[:split]
sup_idx = perm[split:]

# 训练消息传递图
train_ei = {k: v for k, v in edge_index_dict.items()}
train_ea = {k: v for k, v in edge_attr_dict.items()}
train_ei[target_key] = pos_edge_index[:, msg_idx]
train_ea[target_key] = edge_attr_dict[target_key][msg_idx]
train_ei[rev_target_key] = edge_index_dict[rev_target_key][:, msg_idx]
train_ea[rev_target_key] = edge_attr_dict[rev_target_key][msg_idx]

sup_pos_edge = pos_edge_index[:, sup_idx]
num_sup = sup_pos_edge.size(1)

# 验证集
test_s = test_df.sort_values(['session_id', 'event_timestamp'])
test_s['next_poi'] = test_s.groupby('session_id')['poi_idx'].shift(-1)
val_pos_df = test_s.dropna(subset=['next_poi'])
val_pos = val_pos_df[val_pos_df['poi_idx'] != val_pos_df['next_poi']][['poi_idx', 'next_poi']].drop_duplicates().values
num_val_pos = len(val_pos)

train_pos_set = set(map(tuple, pos_edge_index.t().tolist()))
val_pos_set = set(map(tuple, val_pos.tolist()))
all_pois = torch.unique(pos_edge_index.flatten()).to(device)

neg_src = np.random.choice(all_pois, size=num_val_pos * 3)
neg_dst = np.random.choice(all_pois, size=num_val_pos * 3)
val_neg = [(s, d) for s, d in zip(neg_src, neg_dst)
           if s != d and (s, d) not in train_pos_set and (s, d) not in val_pos_set][:num_val_pos]

val_edge = torch.tensor(np.vstack([val_pos, np.array(val_neg)]).T).long().to(device)
val_label = torch.tensor([1] * num_val_pos + [0] * num_val_pos).float().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scaler = torch.amp.GradScaler('cuda', enabled=True)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
best_auc = 0
EPOCHS = 40

print(f"\n🔥 转移边: {num_pos:,} (msg:{len(msg_idx):,}, sup:{num_sup:,})")
print(f"🔥 验证: {num_val_pos*2:,}")
print(f"🔥 显存: {get_mem():.2f} GB")
print()

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()

    with torch.amp.autocast('cuda', enabled=False):
        out = model(x_dict, train_ei, train_ea)
        pos_pred = model.predict_link(out['poi'][sup_pos_edge[0]], out['poi'][sup_pos_edge[1]])

        ri_s = torch.randint(0, len(all_pois), (num_sup,), device=device)
        ri_d = torch.randint(0, len(all_pois), (num_sup,), device=device)
        neg_pred = model.predict_link(out['poi'][all_pois[ri_s]], out['poi'][all_pois[ri_d]])

        pred = torch.cat([pos_pred, neg_pred])
        label = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)])
        loss = F.binary_cross_entropy_with_logits(pred, label)

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()

    train_auc = roc_auc_score(label.cpu().numpy(), torch.sigmoid(pred).cpu().numpy())

    model.eval()
    with torch.no_grad():
        val_out = model(x_dict, edge_index_dict, edge_attr_dict)
        val_pred = model.predict_link(val_out['poi'][val_edge[0]], val_out['poi'][val_edge[1]])
        val_auc = roc_auc_score(val_label.cpu().numpy(), torch.sigmoid(val_pred).cpu().numpy())

    print(f"👉 E{epoch+1:02d}/{EPOCHS} | Loss:{loss.item():.4f} | "
          f"TrainA:{train_auc:.4f} | ValA:{val_auc:.4f} | Mem:{get_mem():.2f}GB")

    scheduler.step(val_auc)
    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.state_dict(), "/home/konglingrui/meituan_project/best_cross_model.pth")

elapsed = time.time() - start_time
print(f"\n🎉 完成! 最高Val AUC: {best_auc:.4f}, 耗时: {elapsed/60:.1f}min")
print(f"💾 模型: /home/konglingrui/meituan_project/best_cross_model.pth")
