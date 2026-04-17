import json
import os
import time
import gc

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATv2Conv, to_hetero


# ==========================================
# V1.2 核心优化点
#
# 1. 特征增强：补强 POI / User / Category 的行为画像
# 2. 过拟合控制：保留正则，但从“过强压制”回调到更平衡配置
# 3. 监督增强：transition_to 从 80/20 改为 70/30
# 4. 训练负样本课程学习：先随机负样本，再逐步混入 hard negatives
# 5. 验证集改回稳定随机负样本，避免“评估口径太难”掩盖真实优化
# 6. 混合预测头：MLP + 非对称双线性分支，用 sigmoid gate 自适应融合
# 7. 训练产物导出：embedding / 映射字典 / 活跃 mask / 配置
# ==========================================

SEED = int(os.getenv("SEED", "42"))
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

start_time = time.time()
print("🚀 [阶段 1/7] 开始读取清洗全量数据...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"💻 当前运行设备: {device}")

base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.getenv("DATA_PATH", os.path.join(base_dir, "view_data.csv"))
model_path = os.getenv("MODEL_PATH", os.path.join(base_dir, "best_cross_model_v1_2.pth"))
export_dir = os.getenv("EXPORT_DIR", os.path.join(base_dir, "export_assets_v1_2"))

max_rows = int(os.getenv("MAX_ROWS", "0"))
epochs = int(os.getenv("EPOCHS", "120"))
early_stop_patience = int(os.getenv("EARLY_STOP_PATIENCE", "30"))
# 全图异构 GAT 的显存成本极高，这里默认使用验证过更稳的组合。
# 若后续要冲更高容量，可通过环境变量显式覆盖。
hidden_channels = int(os.getenv("HIDDEN_CHANNELS", "64"))
gat_heads = int(os.getenv("GAT_HEADS", "2"))
learning_rate = float(os.getenv("LEARNING_RATE", "5e-4"))
weight_decay = float(os.getenv("WEIGHT_DECAY", "2e-4"))
scheduler_patience = int(os.getenv("SCHEDULER_PATIENCE", "10"))
message_ratio = float(os.getenv("MESSAGE_RATIO", "0.7"))
train_random_negative_ratio = float(os.getenv("TRAIN_RANDOM_NEGATIVE_RATIO", "0.75"))
val_random_negative_ratio = float(os.getenv("VAL_RANDOM_NEGATIVE_RATIO", "1.0"))
hard_negative_warmup_epochs = int(os.getenv("HARD_NEGATIVE_WARMUP_EPOCHS", "8"))


def ensure_event_columns(stats_df: pd.DataFrame) -> pd.DataFrame:
    for col in ["PV", "MC", "ORDER"]:
        if col not in stats_df.columns:
            stats_df[col] = 0
    return stats_df


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


def sample_hard_negative_edges(
    active_nodes,
    num_samples,
    forbidden_edges,
    rng,
    poi_to_cate,
    random_ratio=0.5,
):
    active_nodes = np.asarray(active_nodes)
    random_count = int(num_samples * random_ratio)
    hard_count = num_samples - random_count

    random_negs = sample_negative_edges(active_nodes, random_count, forbidden_edges, rng)

    cate_to_pois = {}
    for poi in active_nodes:
        cate = poi_to_cate.get(int(poi))
        if cate is None:
            continue
        cate_to_pois.setdefault(int(cate), []).append(int(poi))

    # 过滤掉不足 2 个 POI 的品类，避免无效循环；同时转为 ndarray 提升索引效率
    cate_to_pois = {k: np.asarray(v, dtype=np.int64) for k, v in cate_to_pois.items() if len(v) >= 2}
    valid_src_nodes = np.asarray(
        [int(p) for p in active_nodes if poi_to_cate.get(int(p)) in cate_to_pois],
        dtype=np.int64,
    )

    hard_negs = []
    sampled = set(map(tuple, random_negs.tolist()))

    if len(valid_src_nodes) > 0 and hard_count > 0:
        multiplier = 5
        while len(hard_negs) < hard_count:
            batch_size = max((hard_count - len(hard_negs)) * multiplier, 2048)
            src_batch = rng.choice(valid_src_nodes, size=batch_size, replace=True)

            for src in src_batch:
                src = int(src)
                cate = poi_to_cate[src]
                candidates = cate_to_pois[cate]
                dst = int(candidates[rng.integers(0, len(candidates))])
                edge = (src, dst)
                if src == dst or edge in forbidden_edges or edge in sampled:
                    continue
                sampled.add(edge)
                hard_negs.append(edge)
                if len(hard_negs) == hard_count:
                    break

    if len(hard_negs) < hard_count:
        fill_count = hard_count - len(hard_negs)
        fill_negs = sample_negative_edges(active_nodes, fill_count, forbidden_edges | sampled, rng)
        hard_array = np.asarray(hard_negs, dtype=np.int64) if hard_negs else np.empty((0, 2), dtype=np.int64)
        return np.vstack([random_negs, hard_array, fill_negs]).astype(np.int64)

    hard_array = np.asarray(hard_negs, dtype=np.int64)
    return np.vstack([random_negs, hard_array]).astype(np.int64)


def convert_to_json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): convert_to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [convert_to_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return convert_to_json_safe(obj.tolist())
    if isinstance(obj, torch.Tensor):
        return convert_to_json_safe(obj.detach().cpu().numpy())
    if pd.isna(obj):
        return None
    return str(obj)


def save_json(data_dict, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(convert_to_json_safe(data_dict), f, ensure_ascii=False, indent=2)


# ==========================================
# 1. 数据读取与清洗
# ==========================================
usecols = [
    "user_id",
    "poi_id",
    "poi_name",
    "first_cate_name",
    "event_type",
    "event_id",
    "session_id",
    "event_timestamp",
    "device_type",
    "page_city_name",
]
read_kwargs = {"usecols": usecols}
if max_rows > 0:
    read_kwargs["nrows"] = max_rows
df = pd.read_csv(file_path, **read_kwargs)

df["event_timestamp"] = pd.to_numeric(df["event_timestamp"], errors="coerce")
df = df.dropna(subset=["user_id", "poi_id", "event_timestamp", "session_id"])

df["event_type_str"] = df["event_type"].astype(str).str.strip().str.upper()
df["event_id_str"] = df["event_id"].astype(str).str.strip().str.lower()

is_pv = df["event_type_str"] == "PV"
is_mc = df["event_type_str"] == "MC"
is_order = (df["event_type_str"] == "ORDER") | (df["event_id_str"].str.contains("order", na=False))

df["unified_event"] = pd.Series(dtype="object")
df.loc[is_pv, "unified_event"] = "PV"
df.loc[is_mc, "unified_event"] = "MC"
df.loc[is_order, "unified_event"] = "ORDER"

df["first_cate_name"] = df["first_cate_name"].fillna("未知品类")
df["page_city_name"] = df["page_city_name"].fillna("unknown")
df["device_type"] = df["device_type"].fillna("unknown")
df["poi_name"] = df["poi_name"].fillna("unknown_poi")
df = df.dropna(subset=["unified_event"])


# ==========================================
# 2. 全局 ID 映射与时序切分
# ==========================================
print("🚀 [阶段 2/7] 全局节点映射与时序防泄漏切分...")
df["user_id"] = df["user_id"].astype("category")
df["poi_id"] = df["poi_id"].astype("category")
df["first_cate_name"] = df["first_cate_name"].astype("category")

df["user_idx"] = df["user_id"].cat.codes
df["poi_idx"] = df["poi_id"].cat.codes
df["cate_idx"] = df["first_cate_name"].cat.codes

num_users = int(df["user_idx"].max()) + 1
num_pois = int(df["poi_idx"].max()) + 1
num_cates = int(df["cate_idx"].max()) + 1

sorted_times = df["event_timestamp"].sort_values().values
split_time = sorted_times[int(len(sorted_times) * 0.8)]

train_df = df[df["event_timestamp"] <= split_time].copy()
test_df = df[df["event_timestamp"] > split_time].copy()


# ==========================================
# 3. 特征工程
# ==========================================
print("🚀 [阶段 3/7] 计算增强特征与冷启动对齐...")

# POI 特征
poi_stats = train_df.groupby("poi_idx")["unified_event"].value_counts().unstack(fill_value=0)
poi_stats = ensure_event_columns(poi_stats)
poi_stats["total_pv"] = poi_stats["PV"]
poi_stats["total_mc"] = poi_stats["MC"]
poi_stats["total_order"] = poi_stats["ORDER"]
poi_stats["unique_users"] = train_df.groupby("poi_idx")["user_idx"].nunique().reindex(poi_stats.index, fill_value=0)
poi_stats["unique_sessions"] = train_df.groupby("poi_idx")["session_id"].nunique().reindex(poi_stats.index, fill_value=0)
poi_stats["ctr"] = (poi_stats["total_mc"] / (poi_stats["total_pv"] + 1e-5)).clip(upper=1.0)
poi_stats["view_cvr"] = (poi_stats["total_order"] / (poi_stats["total_pv"] + 1e-5)).clip(upper=1.0)
poi_stats["click_cvr"] = (poi_stats["total_order"] / (poi_stats["total_mc"] + 1e-5)).clip(upper=1.0)
poi_stats["order_share"] = poi_stats["total_order"] / (
    poi_stats["total_pv"] + poi_stats["total_mc"] + poi_stats["total_order"] + 1e-5
)
poi_stats["log_pv"] = np.log1p(poi_stats["total_pv"])
poi_stats["log_mc"] = np.log1p(poi_stats["total_mc"])
poi_stats["log_order"] = np.log1p(poi_stats["total_order"])
poi_stats["log_unique_users"] = np.log1p(poi_stats["unique_users"])
poi_stats["log_unique_sessions"] = np.log1p(poi_stats["unique_sessions"])

poi_feature_cols = [
    "log_pv",
    "log_mc",
    "log_order",
    "ctr",
    "view_cvr",
    "click_cvr",
    "order_share",
    "log_unique_users",
    "log_unique_sessions",
]
poi_x_full = np.zeros((num_pois, len(poi_feature_cols)), dtype=np.float32)
poi_x_full[poi_stats.index.values] = poi_stats[poi_feature_cols].values
poi_x = MinMaxScaler().fit_transform(poi_x_full)

# User 特征
user_stats = train_df.groupby("user_idx")["unified_event"].value_counts().unstack(fill_value=0)
user_stats = ensure_event_columns(user_stats)
user_stats["total_actions"] = user_stats.sum(axis=1)
user_stats["has_ordered"] = (user_stats["ORDER"] > 0).astype(int)
user_stats["unique_pois"] = train_df.groupby("user_idx")["poi_idx"].nunique().reindex(user_stats.index, fill_value=0)
user_stats["unique_cates"] = train_df.groupby("user_idx")["cate_idx"].nunique().reindex(user_stats.index, fill_value=0)
user_stats["unique_sessions"] = train_df.groupby("user_idx")["session_id"].nunique().reindex(user_stats.index, fill_value=0)
user_stats["pv_ratio"] = user_stats["PV"] / (user_stats["total_actions"] + 1e-5)
user_stats["mc_ratio"] = user_stats["MC"] / (user_stats["total_actions"] + 1e-5)
user_stats["order_ratio"] = user_stats["ORDER"] / (user_stats["total_actions"] + 1e-5)
user_stats["log_total_actions"] = np.log1p(user_stats["total_actions"])
user_stats["log_unique_pois"] = np.log1p(user_stats["unique_pois"])
user_stats["log_unique_cates"] = np.log1p(user_stats["unique_cates"])
user_stats["log_unique_sessions"] = np.log1p(user_stats["unique_sessions"])

user_feature_cols = [
    "log_total_actions",
    "has_ordered",
    "pv_ratio",
    "mc_ratio",
    "order_ratio",
    "log_unique_pois",
    "log_unique_cates",
    "log_unique_sessions",
]
user_x_full = np.zeros((num_users, len(user_feature_cols)), dtype=np.float32)
user_x_full[user_stats.index.values] = user_stats[user_feature_cols].values
user_x = MinMaxScaler().fit_transform(user_x_full)

# Category 特征
cate_stats = train_df.groupby("cate_idx")["unified_event"].value_counts().unstack(fill_value=0)
cate_stats = ensure_event_columns(cate_stats)
cate_stats["total_events"] = cate_stats.sum(axis=1)
cate_stats["unique_pois"] = train_df.groupby("cate_idx")["poi_idx"].nunique().reindex(cate_stats.index, fill_value=0)
cate_stats["unique_users"] = train_df.groupby("cate_idx")["user_idx"].nunique().reindex(cate_stats.index, fill_value=0)
cate_stats["order_rate"] = cate_stats["ORDER"] / (cate_stats["total_events"] + 1e-5)
cate_stats["mc_rate"] = cate_stats["MC"] / (cate_stats["total_events"] + 1e-5)
cate_stats["log_total_events"] = np.log1p(cate_stats["total_events"])
cate_stats["log_unique_pois"] = np.log1p(cate_stats["unique_pois"])
cate_stats["log_unique_users"] = np.log1p(cate_stats["unique_users"])

cate_feature_cols = [
    "log_total_events",
    "log_unique_pois",
    "log_unique_users",
    "order_rate",
    "mc_rate",
]
cate_x_full = np.zeros((num_cates, len(cate_feature_cols)), dtype=np.float32)
cate_x_full[cate_stats.index.values] = cate_stats[cate_feature_cols].values
cate_x = MinMaxScaler().fit_transform(cate_x_full)

USER_FEAT_DIM = user_x.shape[1]
POI_FEAT_DIM = poi_x.shape[1]
CATE_FEAT_DIM = cate_x.shape[1]

data = HeteroData()
data["user"].x = torch.from_numpy(user_x).float()
data["poi"].x = torch.from_numpy(poi_x).float()
data["category"].x = torch.from_numpy(cate_x).float()


# ==========================================
# 4. 构建异构关系网络图
# ==========================================
print("🚀 [阶段 4/7] 组装多关系时序拓扑图...")
for event in ["PV", "MC", "ORDER"]:
    sub_df = train_df[train_df["unified_event"] == event]
    if sub_df.empty:
        continue
    agg = sub_df.groupby(["user_idx", "poi_idx"]).size().reset_index(name="weight")
    edge_index = torch.from_numpy(agg[["user_idx", "poi_idx"]].values.T).long()
    edge_attr = torch.log1p(torch.from_numpy(agg["weight"].values).float())

    data["user", event.lower(), "poi"].edge_index = edge_index
    data["user", event.lower(), "poi"].edge_attr = edge_attr
    data["poi", f"rev_{event.lower()}", "user"].edge_index = edge_index.flip([0])
    data["poi", f"rev_{event.lower()}", "user"].edge_attr = edge_attr

poi_cate_df = train_df[["poi_idx", "cate_idx"]].drop_duplicates()
poi_to_cate_idx = {int(p): int(c) for p, c in poi_cate_df.values.tolist()}
poi_cate_edge_index = torch.from_numpy(poi_cate_df.values.T).long()
data["poi", "belongs_to", "category"].edge_index = poi_cate_edge_index
data["category", "rev_belongs_to", "poi"].edge_index = poi_cate_edge_index.flip([0])

train_df_sorted = train_df.sort_values(by=["session_id", "event_timestamp"])
max_time_train = train_df_sorted["event_timestamp"].max()
train_df_sorted["days_ago"] = (max_time_train - train_df_sorted["event_timestamp"]) / (1000 * 60 * 60 * 24)
train_df_sorted["decay_weight"] = np.exp(-0.1 * train_df_sorted["days_ago"])

train_df_sorted["next_poi_idx"] = train_df_sorted.groupby("session_id")["poi_idx"].shift(-1)
trans_poi_df = train_df_sorted.dropna(subset=["next_poi_idx"]).copy()
trans_poi_df = trans_poi_df[trans_poi_df["poi_idx"] != trans_poi_df["next_poi_idx"]]
agg_poi = trans_poi_df.groupby(["poi_idx", "next_poi_idx"])["decay_weight"].sum().reset_index()

if agg_poi.empty:
    raise RuntimeError("训练集里没有足够的 poi->poi transition 样本，无法继续训练。")

trans_poi_index = torch.from_numpy(agg_poi[["poi_idx", "next_poi_idx"]].values.T).long()
trans_poi_attr = torch.log1p(torch.from_numpy(agg_poi["decay_weight"].values).float())
data["poi", "transition_to", "poi"].edge_index = trans_poi_index
data["poi", "transition_to", "poi"].edge_attr = trans_poi_attr
data["poi", "rev_transition_to", "poi"].edge_index = trans_poi_index.flip([0])
data["poi", "rev_transition_to", "poi"].edge_attr = trans_poi_attr

for edge_type in data.edge_types:
    if hasattr(data[edge_type], "edge_attr") and data[edge_type].edge_attr is not None:
        attr = data[edge_type].edge_attr
        if attr.max() > attr.min():
            data[edge_type].edge_attr = (attr - attr.min()) / (attr.max() - attr.min() + 1e-8)


# ==========================================
# 5. 模型定义
# ==========================================
print("🚀 [阶段 5/7] 实例化异构图注意力网络...")


class BaseGAT(nn.Module):
    def __init__(self, hidden_channels, out_channels, heads=4):
        super().__init__()
        self.conv1 = GATv2Conv(
            (-1, -1),
            hidden_channels,
            heads=heads,
            add_self_loops=False,
            edge_dim=1,
        )
        self.conv2 = GATv2Conv(
            (-1, -1),
            out_channels,
            heads=1,
            add_self_loops=False,
            edge_dim=1,
        )
        self.dropout = nn.Dropout(p=0.35)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        return x


class FeatureEncoder(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.user_lin = nn.Linear(USER_FEAT_DIM, hidden_channels)
        self.poi_lin = nn.Linear(POI_FEAT_DIM, hidden_channels)
        self.cate_lin = nn.Linear(CATE_FEAT_DIM, hidden_channels)
        self.user_bn = nn.BatchNorm1d(hidden_channels)
        self.poi_bn = nn.BatchNorm1d(hidden_channels)
        self.cate_bn = nn.BatchNorm1d(hidden_channels)
        self.dropout = nn.Dropout(p=0.10)

    def forward(self, x_dict):
        user_h = self.dropout(F.relu(self.user_bn(self.user_lin(x_dict["user"]))))
        poi_h = self.dropout(F.relu(self.poi_bn(self.poi_lin(x_dict["poi"]))))
        cate_h = self.dropout(F.relu(self.cate_bn(self.cate_lin(x_dict["category"]))))
        return {"user": user_h, "poi": poi_h, "category": cate_h}


class CrossAnalysisModel(nn.Module):
    def __init__(self, hidden_channels, metadata, heads=4):
        super().__init__()
        self.encoder = FeatureEncoder(hidden_channels)
        base_model = BaseGAT(hidden_channels, hidden_channels, heads=heads)
        self.gnn = to_hetero(base_model, metadata, aggr="mean")
        self.src_proj = nn.Linear(hidden_channels, hidden_channels)
        self.dst_proj = nn.Linear(hidden_channels, hidden_channels)
        self.link_pred = nn.Sequential(
            nn.Linear(hidden_channels * 4, hidden_channels * 2),
            nn.ReLU(),
            nn.Dropout(p=0.30),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(p=0.10),
            nn.Linear(hidden_channels, 1),
        )
        # 初始更偏向稳健的 MLP 分支，训练中再自适应放大 bilinear 分支
        self.gate = nn.Parameter(torch.tensor(-1.0))

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        x_dict = self.encoder(x_dict)
        fmt_attr = {k: (v.view(-1, 1) if v.dim() == 1 else v) for k, v in edge_attr_dict.items()}
        return self.gnn(x_dict, edge_index_dict, fmt_attr)

    def predict_link(self, src_x, dst_x):
        pair_x = torch.cat([src_x, dst_x, torch.abs(src_x - dst_x), src_x * dst_x], dim=-1)
        mlp_score = self.link_pred(pair_x).squeeze(-1)
        bilinear_score = (
            self.src_proj(src_x) * self.dst_proj(dst_x)
        ).sum(dim=-1) / (src_x.size(-1) ** 0.5)
        g = torch.sigmoid(self.gate)
        return (1.0 - g) * mlp_score + g * bilinear_score


model = CrossAnalysisModel(
    hidden_channels=hidden_channels,
    metadata=data.metadata(),
    heads=gat_heads,
)

cpu_x_dict = {nt: data[nt].x for nt in data.node_types}
cpu_edge_index_dict = {et: data[et].edge_index for et in data.edge_types}
cpu_edge_attr_dict = {}
for edge_type in data.edge_types:
    if hasattr(data[edge_type], "edge_attr") and data[edge_type].edge_attr is not None:
        cpu_edge_attr_dict[edge_type] = data[edge_type].edge_attr
    else:
        num_edges = data[edge_type].edge_index.size(1)
        cpu_edge_attr_dict[edge_type] = torch.ones(num_edges)

with torch.no_grad():
    _ = model(cpu_x_dict, cpu_edge_index_dict, cpu_edge_attr_dict)

model = model.to(device)
x_dict = {nt: cpu_x_dict[nt].to(device) for nt in data.node_types}
edge_index_dict = {et: cpu_edge_index_dict[et].to(device) for et in data.edge_types}
edge_attr_dict = {et: cpu_edge_attr_dict[et].to(device) for et in data.edge_types}


# ==========================================
# 6. 训练
# ==========================================
print("🚀 [阶段 6/7] 启动全图模式训练...")

target_edge_type = ("poi", "transition_to", "poi")
reverse_target_edge_type = ("poi", "rev_transition_to", "poi")

full_target_edge_index = data[target_edge_type].edge_index
full_target_edge_attr = data[target_edge_type].edge_attr
num_target_edges = full_target_edge_index.size(1)

perm = torch.randperm(num_target_edges)
message_edge_count = max(int(num_target_edges * message_ratio), 1)
message_perm = perm[:message_edge_count]
supervision_perm = perm[message_edge_count:]
if supervision_perm.numel() == 0:
    supervision_perm = message_perm[-1:].clone()
    message_perm = message_perm[:-1]

message_edge_index = full_target_edge_index[:, message_perm]
message_edge_attr = full_target_edge_attr[message_perm]
train_pos_edge_index = full_target_edge_index[:, supervision_perm].to(device)

edge_index_dict[target_edge_type] = message_edge_index.to(device)
edge_attr_dict[target_edge_type] = (
    message_edge_attr.view(-1, 1) if message_edge_attr.dim() == 1 else message_edge_attr
).to(device)
edge_index_dict[reverse_target_edge_type] = message_edge_index.flip([0]).to(device)
edge_attr_dict[reverse_target_edge_type] = edge_attr_dict[target_edge_type]

active_pois = torch.unique(full_target_edge_index.flatten())
active_pois_np = active_pois.cpu().numpy()
rng = np.random.default_rng(SEED)

print("正在抽取同分布的验证集负样本...")
test_df_sorted = test_df.sort_values(by=["session_id", "event_timestamp"])
test_df_sorted["next_poi_idx"] = test_df_sorted.groupby("session_id")["poi_idx"].shift(-1)
val_pos_df = test_df_sorted.dropna(subset=["next_poi_idx"]).copy()
val_pos_edges = (
    val_pos_df[val_pos_df["poi_idx"] != val_pos_df["next_poi_idx"]][["poi_idx", "next_poi_idx"]]
    .drop_duplicates()
    .values.astype(np.int64)
)

num_val_pos = len(val_pos_edges)
train_message_set = set(map(tuple, message_edge_index.t().tolist()))
train_pos_set = set(map(tuple, train_pos_edge_index.t().tolist()))
val_pos_set = set(map(tuple, val_pos_edges.tolist()))
all_train_pos_set = train_message_set | train_pos_set

val_neg_edges = sample_hard_negative_edges(
    active_pois_np,
    num_val_pos,
    all_train_pos_set | val_pos_set,
    rng,
    poi_to_cate_idx,
    random_ratio=val_random_negative_ratio,
)

val_edge_index = torch.tensor(np.vstack([val_pos_edges, val_neg_edges]).T, dtype=torch.long).to(device)
val_edge_label = torch.tensor([1] * num_val_pos + [0] * num_val_pos, dtype=torch.float).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=scheduler_patience,
)

best_auc = 0.0
no_improve_cnt = 0
pos_edge_index = train_pos_edge_index
num_pos_edges = pos_edge_index.size(1)

print("\n🔥 >>>>> V1.2 全图训练（平衡正则 + 课程学习 Hard Negative）<<<<< 🔥")
print(
    f"    hidden={hidden_channels}, heads={gat_heads}, lr={learning_rate}, wd={weight_decay}, "
    f"message_ratio={message_ratio}, epochs={epochs}, train_random_ratio={train_random_negative_ratio}, "
    f"val_random_ratio={val_random_negative_ratio}"
)

if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    out_x = model(x_dict, edge_index_dict, edge_attr_dict)
    src_pos = pos_edge_index[0]
    dst_pos = pos_edge_index[1]
    pos_preds = model.predict_link(out_x["poi"][src_pos], out_x["poi"][dst_pos])

    current_random_ratio = 1.0 if epoch < hard_negative_warmup_epochs else train_random_negative_ratio
    neg_edges = sample_hard_negative_edges(
        active_pois_np,
        num_pos_edges,
        all_train_pos_set,
        rng,
        poi_to_cate_idx,
        random_ratio=current_random_ratio,
    )
    neg_edge_index = torch.from_numpy(neg_edges.T).to(device)
    neg_preds = model.predict_link(
        out_x["poi"][neg_edge_index[0]],
        out_x["poi"][neg_edge_index[1]],
    )

    preds = torch.cat([pos_preds, neg_preds])
    labels = torch.cat([torch.ones_like(pos_preds), torch.zeros_like(neg_preds)])
    loss = F.binary_cross_entropy_with_logits(preds, labels)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    with torch.no_grad():
        train_auc = roc_auc_score(labels.cpu().numpy(), torch.sigmoid(preds).cpu().numpy())

    model.eval()
    with torch.no_grad():
        eval_out = model(x_dict, edge_index_dict, edge_attr_dict)
        val_preds = model.predict_link(
            eval_out["poi"][val_edge_index[0]],
            eval_out["poi"][val_edge_index[1]],
        )
        val_auc = roc_auc_score(val_edge_label.cpu().numpy(), torch.sigmoid(val_preds).cpu().numpy())

    current_lr = optimizer.param_groups[0]["lr"]
    gate_value = torch.sigmoid(model.gate).item()
    print(
        f"Epoch {epoch+1:03d}/{epochs} | Loss: {loss.item():.4f} | "
        f"Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f} | "
        f"LR: {current_lr:.2e} | Gate: {gate_value:.3f} | NegRnd: {current_random_ratio:.2f}"
    )

    scheduler.step(val_auc)

    if val_auc > best_auc:
        best_auc = val_auc
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


# ==========================================
# 7. 终极资产导出
# ==========================================
print("\n🚀 [阶段 7/7] 开始导出下游推理所需的终极资产...")
os.makedirs(export_dir, exist_ok=True)

if not os.path.exists(model_path):
    raise FileNotFoundError(f"未找到最优模型权重: {model_path}")

print("📥 正在加载最优模型权重...")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

with torch.no_grad():
    out_x = model(x_dict, edge_index_dict, edge_attr_dict)
    poi_embeddings = out_x["poi"].cpu()

torch.save(poi_embeddings, os.path.join(export_dir, "poi_embeddings.pt"))

print("📍 正在构建稳健映射字典...")
poi_city_series = train_df.groupby("poi_idx")["page_city_name"].apply(
    lambda x: x.fillna("unknown").mode()[0] if not x.dropna().empty else "unknown"
)
poi_idx_to_city = poi_city_series.to_dict()

poi_idx_to_id = dict(enumerate(df["poi_id"].cat.categories))
cate_idx_to_name = dict(enumerate(df["first_cate_name"].cat.categories))
poi_idx_to_name = (
    train_df.drop_duplicates("poi_idx").set_index("poi_idx")["poi_name"].fillna("unknown_poi").to_dict()
)

active_poi_mask = torch.zeros(num_pois, dtype=torch.bool)
active_poi_mask[poi_stats.index.values] = torch.tensor((poi_stats["total_pv"] > 0).values, dtype=torch.bool)
torch.save(active_poi_mask, os.path.join(export_dir, "active_poi_mask.pt"))

poi_stat_dict = poi_stats[["total_pv", "total_mc", "total_order", "ctr", "view_cvr", "click_cvr"]].to_dict(
    orient="index"
)

print("🧠 正在生成第一版加权 Category Embeddings...")
category_embeddings = torch.zeros((num_cates, hidden_channels))
cate_stats_dict = {}

for c_idx in range(num_cates):
    pois_in_cate = [p_idx for p_idx, c in poi_to_cate_idx.items() if c == c_idx]
    if not pois_in_cate:
        continue

    pois_tensor = torch.tensor(pois_in_cate, dtype=torch.long)
    cate_embs = poi_embeddings[pois_tensor]
    weights = torch.tensor(
        [poi_stat_dict.get(p, {}).get("total_pv", 0) + 1 for p in pois_in_cate],
        dtype=torch.float32,
    )
    weights = weights / weights.sum()
    weighted_emb = (cate_embs * weights.unsqueeze(1)).sum(dim=0)
    category_embeddings[c_idx] = weighted_emb

    cate_stats_dict[int(c_idx)] = {
        "total_pois": len(pois_in_cate),
        "total_pv": int(sum([poi_stat_dict.get(p, {}).get("total_pv", 0) for p in pois_in_cate])),
    }

torch.save(category_embeddings, os.path.join(export_dir, "category_embeddings.pt"))

save_json(poi_idx_to_id, os.path.join(export_dir, "poi_idx_to_id.json"))
save_json(poi_idx_to_name, os.path.join(export_dir, "poi_idx_to_name.json"))
save_json(cate_idx_to_name, os.path.join(export_dir, "cate_idx_to_name.json"))
save_json(poi_to_cate_idx, os.path.join(export_dir, "poi_idx_to_cate_idx.json"))
save_json(poi_idx_to_city, os.path.join(export_dir, "poi_idx_to_city.json"))
save_json(poi_stat_dict, os.path.join(export_dir, "poi_stats.json"))
save_json(cate_stats_dict, os.path.join(export_dir, "cate_stats.json"))

experiment_config = {
    "seed": SEED,
    "num_users": num_users,
    "num_pois": num_pois,
    "num_cates": num_cates,
    "split_time": float(split_time),
    "hidden_channels": hidden_channels,
    "gat_heads": gat_heads,
    "epochs": epochs,
    "early_stop_patience": early_stop_patience,
    "learning_rate": learning_rate,
    "weight_decay": weight_decay,
    "scheduler_patience": scheduler_patience,
    "message_ratio": message_ratio,
    "train_random_negative_ratio": train_random_negative_ratio,
    "val_random_negative_ratio": val_random_negative_ratio,
    "hard_negative_warmup_epochs": hard_negative_warmup_epochs,
    "best_val_auc": best_auc,
    "model_path": model_path,
}
save_json(experiment_config, os.path.join(export_dir, "experiment_config.json"))

print(f"✅ 所有资产已成功稳健导出至: {export_dir}")
