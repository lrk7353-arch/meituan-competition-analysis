import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATv2Conv, to_hetero
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
import time
import os

# ==========================================
# 修复说明（对应三个核心Bug）
#
# [Fix 1 - 致命] 删除 FeatureEncoder 中的 ID Embedding
#   原代码用 self.poi_emb.weight（全量矩阵）作为残差加入节点特征，
#   导致模型记住训练集中所有 (poi_src, poi_dst) 配对的 ID，
#   而非学习结构性规律，测试集 AUC 因此退化为 0.5（随机水平）。
#   修复：移除三个 Embedding 层，改用 BatchNorm + 线性映射。
#
# [Fix 2 - Bug] MinMaxScaler fit/transform 不一致
#   原代码对「非零训练统计量子集」fit，再对「含冷启动零值的全量数组」transform，
#   使冷启动节点的特征值变为负数，污染 GNN 输入。
#   修复：对全量 zero-padded 数组直接 fit_transform。
#
# [Fix 3 - Bug] edge_attr_dict 不完整导致注意力计算崩溃
#   原代码只把有 edge_attr 的边类型加入字典，
#   但 GATv2Conv(edge_dim=1) 要求所有边类型都必须传入 edge_attr，
#   缺失类型在 to_hetero 内部会静默地接收 None，破坏注意力机制。
#   修复：对无权重的边类型填充全 1 向量作为默认权重。
#
# [改进] 早停机制 / BatchNorm / Dropout 0.3→0.4
# ==========================================

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def sample_negative_edges(active_nodes, num_samples, forbidden_edges, rng, multiplier=5):
    neg_edges = []
    sampled = set()
    active_nodes = np.asarray(active_nodes)
    while len(neg_edges) < num_samples:
        batch_size = max((num_samples - len(neg_edges)) * multiplier, 2048)
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
base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.getenv('DATA_PATH', os.path.join(base_dir, 'view_data.csv'))
max_rows = int(os.getenv('MAX_ROWS', '0'))
epochs = int(os.getenv('EPOCHS', '60'))
early_stop_patience = int(os.getenv('EARLY_STOP_PATIENCE', '10'))
hidden_channels = int(os.getenv('HIDDEN_CHANNELS', '64'))
model_path = os.getenv('MODEL_PATH', os.path.join(base_dir, 'best_cross_model.pth'))

# ==========================================
# 1. 数据读取与清洗
# ==========================================
usecols = ['user_id', 'poi_id', 'first_cate_name', 'event_type', 'event_id',
           'session_id', 'event_timestamp', 'device_type', 'page_city_name']
read_kwargs = {'usecols': usecols}
if max_rows > 0:
    read_kwargs['nrows'] = max_rows
df = pd.read_csv(file_path, **read_kwargs)

df['event_timestamp'] = pd.to_numeric(df['event_timestamp'], errors='coerce')
df = df.dropna(subset=['user_id', 'poi_id', 'event_timestamp', 'session_id'])

df['event_type_str'] = df['event_type'].astype(str).str.strip().str.upper()
df['event_id_str'] = df['event_id'].astype(str).str.strip().str.lower()
is_pv    = df['event_type_str'] == 'PV'
is_mc    = df['event_type_str'] == 'MC'
is_order = (df['event_type_str'] == 'ORDER') | (df['event_id_str'].str.contains('order', na=False))

df['unified_event'] = pd.Series(dtype='object')
df.loc[is_pv,    'unified_event'] = 'PV'
df.loc[is_mc,    'unified_event'] = 'MC'
df.loc[is_order, 'unified_event'] = 'ORDER'

df['first_cate_name'] = df['first_cate_name'].fillna('未知品类')
df = df.dropna(subset=['unified_event'])

# ==========================================
# 2. 全局 ID 映射与时序切分
# ==========================================
print("🚀 [阶段 2/6] 全局节点映射与时序防泄漏切分...")
df['user_id']        = df['user_id'].astype('category')
df['poi_id']         = df['poi_id'].astype('category')
df['first_cate_name'] = df['first_cate_name'].astype('category')

df['user_idx'] = df['user_id'].cat.codes
df['poi_idx']  = df['poi_id'].cat.codes
df['cate_idx'] = df['first_cate_name'].cat.codes

num_users = int(df['user_idx'].max()) + 1
num_pois  = int(df['poi_idx'].max()) + 1
num_cates = int(df['cate_idx'].max()) + 1

sorted_times = df['event_timestamp'].sort_values().values
split_time   = sorted_times[int(len(sorted_times) * 0.8)]

train_df = df[df['event_timestamp'] <= split_time].copy()
test_df  = df[df['event_timestamp'] >  split_time].copy()

# ==========================================
# 3. 特征工程（仅依赖 train_df）
#    [Fix 2] MinMaxScaler 改为对 zero-padded 全量数组 fit_transform
# ==========================================
print("🚀 [阶段 3/6] 计算高阶特征与冷启动对齐...")

# 统一补齐事件列，避免某些采样/切分下缺列导致特征退化
def ensure_event_columns(stats_df):
    for col in ['PV', 'MC', 'ORDER']:
        if col not in stats_df.columns:
            stats_df[col] = 0
    return stats_df


# POI 特征：仅用 PV/ORDER 太薄，这里补上 MC、覆盖度和行为漏斗特征
poi_stats = train_df.groupby('poi_idx')['unified_event'].value_counts().unstack(fill_value=0)
poi_stats = ensure_event_columns(poi_stats)
poi_stats['total_pv'] = poi_stats['PV']
poi_stats['total_mc'] = poi_stats['MC']
poi_stats['total_order'] = poi_stats['ORDER']
poi_stats['unique_users'] = train_df.groupby('poi_idx')['user_idx'].nunique().reindex(poi_stats.index, fill_value=0)
poi_stats['unique_sessions'] = train_df.groupby('poi_idx')['session_id'].nunique().reindex(poi_stats.index, fill_value=0)
poi_stats['ctr'] = (poi_stats['total_mc'] / (poi_stats['total_pv'] + 1e-5)).clip(upper=1.0)
poi_stats['view_cvr'] = (poi_stats['total_order'] / (poi_stats['total_pv'] + 1e-5)).clip(upper=1.0)
poi_stats['click_cvr'] = (poi_stats['total_order'] / (poi_stats['total_mc'] + 1e-5)).clip(upper=1.0)
poi_stats['log_pv'] = np.log1p(poi_stats['total_pv'])
poi_stats['log_mc'] = np.log1p(poi_stats['total_mc'])
poi_stats['log_order'] = np.log1p(poi_stats['total_order'])
poi_stats['log_unique_users'] = np.log1p(poi_stats['unique_users'])
poi_stats['log_unique_sessions'] = np.log1p(poi_stats['unique_sessions'])

poi_feature_cols = [
    'log_pv', 'log_mc', 'log_order',
    'ctr', 'view_cvr', 'click_cvr',
    'log_unique_users', 'log_unique_sessions'
]
poi_x_full = np.zeros((num_pois, len(poi_feature_cols)))
poi_x_full[poi_stats.index.values] = poi_stats[poi_feature_cols].values
# [Fix 2] fit_transform 作用于全量数组，冷启动节点保持在合理归一化范围内
poi_x = MinMaxScaler().fit_transform(poi_x_full)

# User 特征：增加活跃广度与行为倾向，减轻“只有总行为量”的信息瓶颈
user_stats = train_df.groupby('user_idx')['unified_event'].value_counts().unstack(fill_value=0)
user_stats = ensure_event_columns(user_stats)
user_stats['total_actions'] = user_stats.sum(axis=1)
user_stats['has_ordered'] = (user_stats['ORDER'] > 0).astype(int)
user_stats['unique_pois'] = train_df.groupby('user_idx')['poi_idx'].nunique().reindex(user_stats.index, fill_value=0)
user_stats['unique_cates'] = train_df.groupby('user_idx')['cate_idx'].nunique().reindex(user_stats.index, fill_value=0)
user_stats['pv_ratio'] = user_stats['PV'] / (user_stats['total_actions'] + 1e-5)
user_stats['mc_ratio'] = user_stats['MC'] / (user_stats['total_actions'] + 1e-5)
user_stats['order_ratio'] = user_stats['ORDER'] / (user_stats['total_actions'] + 1e-5)
user_stats['log_total_actions'] = np.log1p(user_stats['total_actions'])
user_stats['log_unique_pois'] = np.log1p(user_stats['unique_pois'])
user_stats['log_unique_cates'] = np.log1p(user_stats['unique_cates'])

user_feature_cols = [
    'log_total_actions', 'has_ordered',
    'pv_ratio', 'mc_ratio', 'order_ratio',
    'log_unique_pois', 'log_unique_cates'
]
user_x_full = np.zeros((num_users, len(user_feature_cols)))
user_x_full[user_stats.index.values] = user_stats[user_feature_cols].values
user_x = MinMaxScaler().fit_transform(user_x_full)

# Category 特征：加入覆盖规模与转化倾向，避免只有单一体量特征
cate_stats = train_df.groupby('cate_idx')['unified_event'].value_counts().unstack(fill_value=0)
cate_stats = ensure_event_columns(cate_stats)
cate_stats['total_events'] = cate_stats.sum(axis=1)
cate_stats['unique_pois'] = train_df.groupby('cate_idx')['poi_idx'].nunique().reindex(cate_stats.index, fill_value=0)
cate_stats['unique_users'] = train_df.groupby('cate_idx')['user_idx'].nunique().reindex(cate_stats.index, fill_value=0)
cate_stats['order_rate'] = cate_stats['ORDER'] / (cate_stats['total_events'] + 1e-5)
cate_stats['log_total_events'] = np.log1p(cate_stats['total_events'])
cate_stats['log_unique_pois'] = np.log1p(cate_stats['unique_pois'])
cate_stats['log_unique_users'] = np.log1p(cate_stats['unique_users'])
cate_stats['mc_rate'] = cate_stats['MC'] / (cate_stats['total_events'] + 1e-5)

cate_feature_cols = [
    'log_total_events', 'log_unique_pois',
    'log_unique_users', 'order_rate', 'mc_rate'
]
cate_x_full = np.zeros((num_cates, len(cate_feature_cols)))
cate_x_full[cate_stats.index.values] = cate_stats[cate_feature_cols].values
cate_x = MinMaxScaler().fit_transform(cate_x_full)

USER_FEAT_DIM = user_x.shape[1]
POI_FEAT_DIM = poi_x.shape[1]
CATE_FEAT_DIM = cate_x.shape[1]

data = HeteroData()
data['user'].x     = torch.from_numpy(user_x).float()
data['poi'].x      = torch.from_numpy(poi_x).float()
data['category'].x = torch.from_numpy(cate_x).float()

# ==========================================
# 4. 构建异构关系网络图
# ==========================================
print("🚀 [阶段 4/6] 组装多关系时序拓扑图...")
for event in ['PV', 'MC', 'ORDER']:
    sub_df = train_df[train_df['unified_event'] == event]
    if sub_df.empty:
        continue
    agg = sub_df.groupby(['user_idx', 'poi_idx']).size().reset_index(name='weight')
    edge_index = torch.from_numpy(agg[['user_idx', 'poi_idx']].values.T).long()
    edge_attr  = torch.log1p(torch.from_numpy(agg['weight'].values).float())

    data['user', event.lower(), 'poi'].edge_index = edge_index
    data['user', event.lower(), 'poi'].edge_attr  = edge_attr
    data['poi', f'rev_{event.lower()}', 'user'].edge_index = edge_index.flip([0])
    data['poi', f'rev_{event.lower()}', 'user'].edge_attr  = edge_attr

poi_cate_df        = train_df[['poi_idx', 'cate_idx']].drop_duplicates()
poi_cate_edge_index = torch.from_numpy(poi_cate_df.values.T).long()
data['poi', 'belongs_to', 'category'].edge_index        = poi_cate_edge_index
data['category', 'rev_belongs_to', 'poi'].edge_index    = poi_cate_edge_index.flip([0])

train_df_sorted = train_df.sort_values(by=['session_id', 'event_timestamp'])
max_time_train  = train_df_sorted['event_timestamp'].max()
train_df_sorted['days_ago']     = (max_time_train - train_df_sorted['event_timestamp']) / (1000 * 60 * 60 * 24)
train_df_sorted['decay_weight'] = np.exp(-0.1 * train_df_sorted['days_ago'])

train_df_sorted['next_poi_idx'] = train_df_sorted.groupby('session_id')['poi_idx'].shift(-1)
trans_poi_df = train_df_sorted.dropna(subset=['next_poi_idx']).copy()
trans_poi_df = trans_poi_df[trans_poi_df['poi_idx'] != trans_poi_df['next_poi_idx']]
agg_poi      = trans_poi_df.groupby(['poi_idx', 'next_poi_idx'])['decay_weight'].sum().reset_index()

if not agg_poi.empty:
    trans_poi_index = torch.from_numpy(agg_poi[['poi_idx', 'next_poi_idx']].values.T).long()
    trans_poi_attr  = torch.log1p(torch.from_numpy(agg_poi['decay_weight'].values).float())
    data['poi', 'transition_to',     'poi'].edge_index = trans_poi_index
    data['poi', 'transition_to',     'poi'].edge_attr  = trans_poi_attr
    data['poi', 'rev_transition_to', 'poi'].edge_index = trans_poi_index.flip([0])
    data['poi', 'rev_transition_to', 'poi'].edge_attr  = trans_poi_attr

# 边权重归一化
for edge_type in data.edge_types:
    if hasattr(data[edge_type], 'edge_attr') and data[edge_type].edge_attr is not None:
        attr = data[edge_type].edge_attr
        if attr.max() > attr.min():
            data[edge_type].edge_attr = (attr - attr.min()) / (attr.max() - attr.min() + 1e-8)

# ==========================================
# 5. 模型定义
# ==========================================
print("🚀 [阶段 5/6] 实例化异构图注意力网络...")


class BaseGAT(nn.Module):
    """
    双层 GATv2Conv。
    edge_attr 必须传入（edge_dim=1），由外部保证所有边类型都有 edge_attr。
    """
    def __init__(self, hidden_channels, out_channels, heads=2):
        super().__init__()
        self.conv1   = GATv2Conv((-1, -1), hidden_channels, heads=heads,
                                 add_self_loops=False, edge_dim=1)
        self.conv2   = GATv2Conv((-1, -1), out_channels, heads=1,
                                 add_self_loops=False, edge_dim=1)
        # [改进] Dropout 0.3 → 0.4
        self.dropout = nn.Dropout(p=0.4)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        return x


class FeatureEncoder(nn.Module):
    """
    [Fix 1] 删除 ID Embedding，改用 Linear + BatchNorm。
    原设计将 self.xxx_emb.weight（全量矩阵）加到每个节点的特征上，
    等价于给每个节点一个可学习的唯一 ID 向量；模型会利用它来
    记住训练集中的 (poi_src, poi_dst) ID 配对，而非学习结构规律。
    删除后模型只能依赖统计特征+图结构来泛化，Val AUC 将显著提升。
    """
    def __init__(self, hidden_channels):
        super().__init__()
        self.user_lin  = nn.Linear(USER_FEAT_DIM, hidden_channels)
        self.poi_lin   = nn.Linear(POI_FEAT_DIM, hidden_channels)
        self.cate_lin  = nn.Linear(CATE_FEAT_DIM, hidden_channels)
        # BatchNorm 代替 ID Embedding，稳定各节点类型的特征尺度
        self.user_bn   = nn.BatchNorm1d(hidden_channels)
        self.poi_bn    = nn.BatchNorm1d(hidden_channels)
        self.cate_bn   = nn.BatchNorm1d(hidden_channels)

    def forward(self, x_dict):
        return {
            'user':     F.relu(self.user_bn(self.user_lin(x_dict['user']))),
            'poi':      F.relu(self.poi_bn(self.poi_lin(x_dict['poi']))),
            'category': F.relu(self.cate_bn(self.cate_lin(x_dict['category']))),
        }


class CrossAnalysisModel(nn.Module):
    # [Fix 1] 构造函数不再需要 num_users / num_pois / num_cates
    def __init__(self, hidden_channels, metadata):
        super().__init__()
        self.encoder   = FeatureEncoder(hidden_channels)
        base_model     = BaseGAT(hidden_channels, hidden_channels, heads=2)
        self.gnn       = to_hetero(base_model, metadata, aggr='mean')
        self.src_proj  = nn.Linear(hidden_channels, hidden_channels)
        self.dst_proj  = nn.Linear(hidden_channels, hidden_channels)
        self.link_pred = nn.Sequential(
            nn.Linear(hidden_channels * 4, hidden_channels * 2),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_channels, 1),
        )
        # 让模型自行调节 MLP 打分与非对称双线性打分的融合强度
        self.bilinear_weight = nn.Parameter(torch.tensor(0.2))

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        x_dict = self.encoder(x_dict)
        # edge_attr 保证每种边类型都有值（见下方 edge_attr_dict 构造逻辑）
        fmt_attr = {k: (v.view(-1, 1) if v.dim() == 1 else v)
                    for k, v in edge_attr_dict.items()}
        x_dict   = self.gnn(x_dict, edge_index_dict, fmt_attr)
        return x_dict

    def predict_link(self, src_x, dst_x):
        pair_x = torch.cat([src_x, dst_x,
                             torch.abs(src_x - dst_x),
                             src_x * dst_x], dim=-1)
        mlp_score = self.link_pred(pair_x).squeeze(-1)
        bilinear_score = (
            self.src_proj(src_x) * self.dst_proj(dst_x)
        ).sum(dim=-1) / (src_x.size(-1) ** 0.5)
        return mlp_score + self.bilinear_weight * bilinear_score


model = CrossAnalysisModel(
    hidden_channels=hidden_channels,
    metadata=data.metadata(),
)

# 先在 CPU 上准备一份输入，触发 to_hetero/GATv2Conv 的惰性参数初始化；
# 若直接 model.to(cuda) 后首次前向，部分惰性参数仍可能留在 CPU，导致设备不一致报错
cpu_x_dict = {nt: data[nt].x for nt in data.node_types}
cpu_edge_index_dict = {et: data[et].edge_index for et in data.edge_types}
cpu_edge_attr_dict = {}
for edge_type in data.edge_types:
    if hasattr(data[edge_type], 'edge_attr') and data[edge_type].edge_attr is not None:
        cpu_edge_attr_dict[edge_type] = data[edge_type].edge_attr
    else:
        num_edges = data[edge_type].edge_index.size(1)
        cpu_edge_attr_dict[edge_type] = torch.ones(num_edges)

with torch.no_grad():
    _ = model(cpu_x_dict, cpu_edge_index_dict, cpu_edge_attr_dict)

model = model.to(device)

# 节点特征
x_dict = {nt: cpu_x_dict[nt].to(device) for nt in data.node_types}

# 边索引
edge_index_dict = {et: cpu_edge_index_dict[et].to(device) for et in data.edge_types}

# [Fix 3] 所有边类型都必须提供 edge_attr（GATv2Conv edge_dim=1 强依赖）
#         无权重的边类型（poi→category 等）用全 1 向量作为默认权重
edge_attr_dict = {et: cpu_edge_attr_dict[et].to(device) for et in data.edge_types}

# ==========================================
# 6. 训练（全图模式 + 早停）
# ==========================================
print("🚀 [阶段 6/6] 启动全图模式训练...")

target_edge_type         = ('poi', 'transition_to',     'poi')
reverse_target_edge_type = ('poi', 'rev_transition_to', 'poi')
if target_edge_type not in data.edge_types:
    raise RuntimeError("缺少 poi->poi 的 transition_to 边，无法进行链路预测。")

full_target_edge_index = data[target_edge_type].edge_index
full_target_edge_attr  = data[target_edge_type].edge_attr
num_target_edges       = full_target_edge_index.size(1)

perm               = torch.randperm(num_target_edges)
message_edge_count = max(int(num_target_edges * 0.8), 1)
message_perm       = perm[:message_edge_count]
supervision_perm   = perm[message_edge_count:]
if supervision_perm.numel() == 0:
    supervision_perm = message_perm[-1:].clone()
    message_perm     = message_perm[:-1]

message_edge_index  = full_target_edge_index[:, message_perm]
message_edge_attr   = full_target_edge_attr[message_perm]
train_pos_edge_index = full_target_edge_index[:, supervision_perm].to(device)

# 用消息边覆盖字典中的 transition_to（避免标签泄漏）
edge_index_dict[target_edge_type]         = message_edge_index.to(device)
edge_attr_dict[target_edge_type]          = (message_edge_attr.view(-1, 1) if message_edge_attr.dim() == 1 else message_edge_attr).to(device)
edge_index_dict[reverse_target_edge_type] = message_edge_index.flip([0]).to(device)
edge_attr_dict[reverse_target_edge_type]  = edge_attr_dict[target_edge_type]

active_pois    = torch.unique(full_target_edge_index.flatten())
active_pois_np = active_pois.cpu().numpy()
rng = np.random.default_rng(SEED)

# 构建验证集（来自 test_df 的 poi 转移正样本）
print("正在抽取同分布的验证集负样本...")
test_df_sorted              = test_df.sort_values(by=['session_id', 'event_timestamp'])
test_df_sorted['next_poi_idx'] = test_df_sorted.groupby('session_id')['poi_idx'].shift(-1)
val_pos_df  = test_df_sorted.dropna(subset=['next_poi_idx']).copy()
val_pos_edges = (val_pos_df[val_pos_df['poi_idx'] != val_pos_df['next_poi_idx']]
                 [['poi_idx', 'next_poi_idx']].drop_duplicates().values.astype(np.int64))

num_val_pos       = len(val_pos_edges)
train_message_set = set(map(tuple, message_edge_index.t().tolist()))
train_pos_set     = set(map(tuple, train_pos_edge_index.t().tolist()))
val_pos_set       = set(map(tuple, val_pos_edges.tolist()))
all_train_pos_set = train_message_set | train_pos_set

val_neg_edges = sample_negative_edges(
    active_pois_np, num_val_pos,
    all_train_pos_set | val_pos_set, rng
)

val_edge_index = torch.tensor(
    np.vstack([val_pos_edges, val_neg_edges]).T, dtype=torch.long
).to(device)
val_edge_label = torch.tensor(
    [1] * num_val_pos + [0] * num_val_pos, dtype=torch.float
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=5
)

# [改进] 早停：连续 patience 个 epoch val_auc 不提升则终止
best_auc       = 0.0
no_improve_cnt = 0

pos_edge_index = train_pos_edge_index
num_pos_edges  = pos_edge_index.size(1)

print("\n🔥 >>>>> 全图训练（已修复 ID 记忆/特征缩放/边属性三大 Bug）<<<<< 🔥")
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    out_x = model(x_dict, edge_index_dict, edge_attr_dict)

    src_pos  = pos_edge_index[0]
    dst_pos  = pos_edge_index[1]
    pos_preds = model.predict_link(out_x['poi'][src_pos], out_x['poi'][dst_pos])

    neg_edges     = sample_negative_edges(active_pois_np, num_pos_edges, all_train_pos_set, rng)
    neg_edge_index = torch.from_numpy(neg_edges.T).to(device)
    neg_preds      = model.predict_link(
        out_x['poi'][neg_edge_index[0]], out_x['poi'][neg_edge_index[1]]
    )

    preds  = torch.cat([pos_preds, neg_preds])
    labels = torch.cat([torch.ones_like(pos_preds), torch.zeros_like(neg_preds)])
    loss   = F.binary_cross_entropy_with_logits(preds, labels)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    with torch.no_grad():
        train_auc = roc_auc_score(
            labels.cpu().numpy(),
            torch.sigmoid(preds).cpu().numpy()
        )

    model.eval()
    with torch.no_grad():
        eval_out  = model(x_dict, edge_index_dict, edge_attr_dict)
        val_preds = model.predict_link(
            eval_out['poi'][val_edge_index[0]],
            eval_out['poi'][val_edge_index[1]]
        )
        val_auc = roc_auc_score(
            val_edge_label.cpu().numpy(),
            torch.sigmoid(val_preds).cpu().numpy()
        )

    print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {loss.item():.4f} "
          f"| Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f}")

    scheduler.step(val_auc)

    if val_auc > best_auc:
        best_auc       = val_auc
        no_improve_cnt = 0
        torch.save(model.state_dict(), model_path)
        print(f"  ✅ 最优 Val AUC 更新: {best_auc:.4f}，已保存模型")
    else:
        no_improve_cnt += 1
        if no_improve_cnt >= early_stop_patience:
            print(f"\n⏹  早停触发：{early_stop_patience} 个 epoch 无提升，终止训练")
            break

elapsed = time.time() - start_time
print(f"\n🎉 训练完成！最高验证集 AUC: {best_auc:.4f} | 总耗时: {elapsed/60:.1f} min")
