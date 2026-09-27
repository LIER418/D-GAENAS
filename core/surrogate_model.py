
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from torch_geometric.nn import GINEConv
from torch_geometric.data import Data, Batch


def precision_at_k(pred, true, k):
    pred     = np.asarray(pred, dtype=np.float64)
    true     = np.asarray(true, dtype=np.float64)
    true_top = set(np.argsort(true)[::-1][:k].tolist())
    pred_top = set(np.argsort(pred)[::-1][:k].tolist())
    return len(true_top & pred_top) / k


NASBENCH_OPS = ['input', 'conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3', 'output']
NUM_OPS      = len(NASBENCH_OPS)
OP_TO_IDX    = {op: i for i, op in enumerate(NASBENCH_OPS)}
MAX_NODES    = 7

NUM_NODE_FEATURES = NUM_OPS + 8
NUM_EDGE_FEATURES = 4


class GINEncoder(nn.Module):

    def __init__(self, node_channels, edge_channels, hidden_channels, out_channels):
        super().__init__()

        mlp1 = nn.Sequential(
            nn.Linear(node_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.conv1 = GINEConv(mlp1, edge_dim=edge_channels)

        mlp2 = nn.Sequential(
            nn.Linear(hidden_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )
        self.conv2 = GINEConv(mlp2, edge_dim=edge_channels)

        self.input_projection = (
            nn.Linear(node_channels, hidden_channels)
            if node_channels != hidden_channels else None
        )
        self.hidden_projection = (
            nn.Linear(hidden_channels, out_channels)
            if hidden_channels != out_channels else None
        )

        self.layer_norm1 = nn.LayerNorm(hidden_channels)
        self.layer_norm2 = nn.LayerNorm(out_channels)

    def forward(self, x, edge_index, edge_attr):
        identity1 = x

        if edge_index.shape[1] == 0:
            x = self.conv1.nn(x)
        else:
            x = self.conv1(x, edge_index, edge_attr)

        if self.input_projection is not None:
            identity1 = self.input_projection(identity1)
        x = self.layer_norm1(x + identity1)
        x = F.relu(x)

        identity2 = x

        if edge_index.shape[1] == 0:
            x = self.conv2.nn(x)
        else:
            x = self.conv2(x, edge_index, edge_attr)

        if self.hidden_projection is not None:
            identity2 = self.hidden_projection(identity2)
        x = self.layer_norm2(x + identity2)

        return x


class SurrogateModel(nn.Module):

    def __init__(self, num_nodes, node_channels, edge_channels,
                 hidden_channels, latent_channels):
        super().__init__()
        self.num_nodes = num_nodes

        self.encoder = GINEncoder(node_channels, edge_channels,
                                  hidden_channels, latent_channels)

        self.node_mlps = nn.ModuleList([
            nn.Linear(latent_channels, 1)
            for _ in range(num_nodes)
        ])

        self.aggregate = nn.Linear(num_nodes, 1)

    def forward(self, x, edge_index, edge_attr, batch=None):
        z = self.encoder(x, edge_index, edge_attr)

        if batch is None:
            z = z.unsqueeze(0)
        else:
            batch_size = int(batch.max().item()) + 1
            z = z.view(batch_size, self.num_nodes, -1)

        node_scalars = torch.cat(
            [self.node_mlps[i](z[:, i, :]) for i in range(self.num_nodes)],
            dim=-1,
        )

        out = self.aggregate(node_scalars)
        return out.squeeze(-1)


def nasbench_arch_to_graph(ops, adjacency_matrix, accuracy):
    N   = len(ops)
    adj = np.array(adjacency_matrix, dtype=np.float32)


    onehot = np.zeros((N, NUM_OPS), dtype=np.float32)
    for i, op in enumerate(ops):
        idx = OP_TO_IDX.get(op)
        if idx is not None:
            onehot[i, idx] = 1.0

    depth = np.zeros(N, dtype=np.float32)
    for i in range(1, N):
        preds = np.where(adj[:i, i] > 0)[0]
        if len(preds) > 0:
            depth[i] = depth[preds].max() + 1
    max_depth = depth.max() if depth.max() > 0 else 1.0
    depth_norm = (depth / max_depth).reshape(-1, 1)

    in_deg  = adj.sum(axis=0)
    out_deg = adj.sum(axis=1)
    max_deg = max(in_deg.max(), out_deg.max(), 1.0)
    in_norm  = (in_deg  / max_deg).reshape(-1, 1)
    out_norm = (out_deg / max_deg).reshape(-1, 1)

    pos_norm = np.array([i / (MAX_NODES - 1) for i in range(N)],
                        dtype=np.float32).reshape(-1, 1)

    paths_to   = np.zeros(N, dtype=np.float64)
    paths_from = np.zeros(N, dtype=np.float64)
    paths_to[0] = 1.0
    for v in range(1, N):
        preds = np.where(adj[:v, v] > 0)[0]
        paths_to[v] = paths_to[preds].sum()
    paths_from[N - 1] = 1.0
    for u in range(N - 2, -1, -1):
        succs = np.where(adj[u, u + 1:] > 0)[0] + u + 1
        paths_from[u] = paths_from[succs].sum()
    total_paths = max(paths_to[N - 1], 1.0)

    path_node_cov = (paths_to * paths_from / total_paths).astype(np.float32).reshape(-1, 1)

    bfs_depth = np.full(N, -1, dtype=np.float32)
    bfs_depth[0] = 0.0
    q = deque([0])
    while q:
        u = q.popleft()
        for v in range(N):
            if adj[u, v] > 0 and bfs_depth[v] < 0:
                bfs_depth[v] = bfs_depth[u] + 1
                q.append(v)
    bfs_depth = np.maximum(bfs_depth, 0)
    max_bfs = bfs_depth.max() if bfs_depth.max() > 0 else 1.0
    bfs_depth_norm = (bfs_depth / max_bfs).reshape(-1, 1)

    bfs_to_out = np.full(N, -1, dtype=np.float32)
    bfs_to_out[N - 1] = 0.0
    q = deque([N - 1])
    adj_T = adj.T
    while q:
        v = q.popleft()
        for u in range(N):
            if adj_T[v, u] > 0 and bfs_to_out[u] < 0:
                bfs_to_out[u] = bfs_to_out[v] + 1
                q.append(u)
    bfs_to_out = np.maximum(bfs_to_out, 0)
    max_bfs_out = bfs_to_out.max() if bfs_to_out.max() > 0 else 1.0
    bfs_to_out_norm = (bfs_to_out / max_bfs_out).reshape(-1, 1)

    fan_in_ratio = (in_deg / (in_deg + out_deg + 1e-6)).astype(np.float32).reshape(-1, 1)

    node_feats = np.concatenate([
        onehot, depth_norm, in_norm, out_norm,
        pos_norm, path_node_cov, bfs_depth_norm, bfs_to_out_norm, fan_in_ratio,
    ], axis=1)
    x = np.zeros((MAX_NODES, NUM_NODE_FEATURES), dtype=np.float32)
    x[:N] = node_feats

    src_arr, dst_arr = np.where(adj > 0)
    if len(src_arr) > 0:
        edge_index = torch.tensor(
            np.stack([src_arr, dst_arr], axis=0), dtype=torch.long
        )

        span        = (dst_arr - src_arr).astype(np.float32) / (MAX_NODES - 1)
        is_direct   = (dst_arr == src_arr + 1).astype(np.float32)
        path_edge_cov = np.array(
            [paths_to[u] * paths_from[v] / total_paths
             for u, v in zip(src_arr, dst_arr)],
            dtype=np.float32,
        )
        is_bridge   = (path_edge_cov >= 1.0 - 1e-9).astype(np.float32)

        edge_attr = torch.tensor(
            np.stack([span, is_direct, path_edge_cov, is_bridge], axis=1),
            dtype=torch.float,
        )
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr  = torch.empty((0, NUM_EDGE_FEATURES), dtype=torch.float)

    return Data(
        x=torch.tensor(x, dtype=torch.float),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([accuracy], dtype=torch.float),
    )


def load_nasbench_dataset(data_file, epochs=108):
    try:
        import sys
        import types

        for _mod in ('nasbench.lib.evaluate',
                     'nasbench.lib.model_builder',
                     'nasbench.lib.training_time'):
            if _mod not in sys.modules:
                sys.modules[_mod] = types.ModuleType(_mod)

        import tensorflow as tf
        if not hasattr(tf, 'python_io'):
            tf.python_io = tf.compat.v1.python_io
        tf.get_logger().setLevel('ERROR')

        from nasbench import api
    except ImportError:
        raise ImportError(
            "NAS-Bench-101 not installed. Run:\n"
            "  pip install git+https://github.com/google-research/nasbench.git\n"
            "See https://github.com/google-research/nasbench"
        )

    print(f"Loading NAS-Bench-101: {data_file}")
    nasbench = api.NASBench(data_file)

    data_list = []
    for h in nasbench.hash_iterator():
        fixed_stats, computed_stats = nasbench.get_metrics_from_hash(h)
        ops = fixed_stats['module_operations']
        adj = fixed_stats['module_adjacency']

        metrics  = computed_stats[epochs]
        accuracy = float(np.mean([m['final_validation_accuracy'] for m in metrics]))

        data_list.append(nasbench_arch_to_graph(ops, adj, accuracy))

    print(f"Loaded — {len(data_list)} architectures")
    return data_list


def train_surrogate(model, data_list, optimizer, criterion=None):
    if criterion is None:
        criterion = nn.MSELoss()

    model.train()
    batch = Batch.from_data_list(data_list).to(next(model.parameters()).device)
    optimizer.zero_grad()

    pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    loss = criterion(pred, batch.y.squeeze())

    loss.backward()
    optimizer.step()
    return loss.item()


def train_surrogate_ranking(model, data_list, costs, optimizer,
                            margin=1.0, n_pairs=None):
    model.train()

    n = len(data_list)
    if n_pairs is None:
        n_pairs = n

    costs_arr = np.array(costs, dtype=np.float64)

    idx_a = np.random.randint(0, n, size=n_pairs)
    idx_b = np.random.randint(0, n, size=n_pairs)

    mask  = (idx_a != idx_b) & (costs_arr[idx_a] != costs_arr[idx_b])
    idx_a, idx_b = idx_a[mask], idx_b[mask]

    if len(idx_a) == 0:
        return 0.0, 1.0

    targets = np.where(costs_arr[idx_a] > costs_arr[idx_b], 1.0, -1.0)
    device  = next(model.parameters()).device
    target_tensor = torch.tensor(targets, dtype=torch.float, device=device)

    batch_a = Batch.from_data_list([data_list[i] for i in idx_a]).to(device)
    batch_b = Batch.from_data_list([data_list[i] for i in idx_b]).to(device)

    pred_a = model(batch_a.x, batch_a.edge_index, batch_a.edge_attr, batch_a.batch)
    pred_b = model(batch_b.x, batch_b.edge_index, batch_b.edge_attr, batch_b.batch)

    optimizer.zero_grad()
    criterion = nn.MarginRankingLoss(margin=margin)
    loss = criterion(pred_a, pred_b, target_tensor)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        diff    = pred_a - pred_b
        correct = ((diff > 0) & (target_tensor == 1)) | ((diff < 0) & (target_tensor == -1))
        accuracy = correct.float().mean().item()

    return loss.item(), accuracy


@torch.no_grad()
def predict_surrogate(model, data_list):
    model.eval()
    device = next(model.parameters()).device
    batch  = Batch.from_data_list(data_list).to(device)
    pred   = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    return pred.tolist()


from nasbench201_utils import EDGES, N_NODES as _NB201_N_NODES, N_OPS as _NB201_N_OPS

_NB201_EDGE_INDEX = torch.tensor(
    [[u for u, _ in EDGES],
     [v for _, v in EDGES]],
    dtype=torch.long,
)

_NB201_POS_ENC = torch.eye(_NB201_N_NODES)


def _nb201_arch_to_pyg(edge_feats_1arch: torch.Tensor) -> Data:
    return Data(
        x          = _NB201_POS_ENC.clone(),
        edge_index = _NB201_EDGE_INDEX.clone(),
        edge_attr  = edge_feats_1arch.cpu().float(),
    )


class GINESurrogate201(nn.Module):

    def __init__(self, hidden_channels: int = 64, latent_channels: int = 64):
        super().__init__()

        self.encoder = GINEncoder(
            node_channels   = _NB201_N_NODES,
            edge_channels   = _NB201_N_OPS,
            hidden_channels = hidden_channels,
            out_channels    = latent_channels,
        )

        self.node_mlps = nn.ModuleList([
            nn.Linear(latent_channels, 1) for _ in range(_NB201_N_NODES)
        ])
        self.aggregate = nn.Linear(_NB201_N_NODES, 1)

    def forward(self, edge_feats: torch.Tensor) -> torch.Tensor:
        squeeze = edge_feats.dim() == 2
        if squeeze:
            edge_feats = edge_feats.unsqueeze(0)
        B      = edge_feats.shape[0]
        device = edge_feats.device

        batch = Batch.from_data_list(
            [_nb201_arch_to_pyg(edge_feats[i]) for i in range(B)]
        ).to(device)

        z = self.encoder(batch.x, batch.edge_index, batch.edge_attr)
        z = z.view(B, _NB201_N_NODES, -1)

        node_scalars = torch.cat(
            [self.node_mlps[i](z[:, i, :]) for i in range(_NB201_N_NODES)],
            dim=-1,
        )

        out = self.aggregate(node_scalars).squeeze(-1)
        return out.squeeze(0) if squeeze else out


from nasbench301_utils import N_OPS_301, EDGES_301, N_EDGES_DAG_301

_NB301_N_NODES = 6
_NB301_POS_ENC = torch.eye(_NB301_N_NODES)


def _nb301_cell_to_pyg(cell_feat: torch.Tensor) -> Data:
    edge_src, edge_dst, edge_ops = [], [], []
    for e in range(N_EDGES_DAG_301):
        if cell_feat[e, 0] > 0.5:
            src, dst = EDGES_301[e]
            edge_src.append(src)
            edge_dst.append(dst)
            edge_ops.append(cell_feat[e, 1:])

    if len(edge_src) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr  = torch.empty((0, N_OPS_301), dtype=torch.float)
    else:
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_attr  = torch.stack(edge_ops)

    return Data(
        x          = _NB301_POS_ENC.clone(),
        edge_index = edge_index,
        edge_attr  = edge_attr,
    )


class GINESurrogate301(nn.Module):

    def __init__(self, hidden_channels: int = 64, latent_channels: int = 64):
        super().__init__()
        enc_kwargs = dict(
            node_channels   = _NB301_N_NODES,
            edge_channels   = N_OPS_301,
            hidden_channels = hidden_channels,
            out_channels    = latent_channels,
        )
        self.enc_normal = GINEncoder(**enc_kwargs)
        self.enc_reduce = GINEncoder(**enc_kwargs)

        self.node_mlps_normal = nn.ModuleList([
            nn.Linear(latent_channels, 1) for _ in range(_NB301_N_NODES)
        ])
        self.node_mlps_reduce = nn.ModuleList([
            nn.Linear(latent_channels, 1) for _ in range(_NB301_N_NODES)
        ])

        self.aggregate = nn.Linear(_NB301_N_NODES * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze = x.dim() == 3
        if squeeze:
            x = x.unsqueeze(0)
        B      = x.shape[0]
        device = x.device

        normal_batch = Batch.from_data_list(
            [_nb301_cell_to_pyg(x[b, 0].cpu()) for b in range(B)]
        ).to(device)
        reduce_batch = Batch.from_data_list(
            [_nb301_cell_to_pyg(x[b, 1].cpu()) for b in range(B)]
        ).to(device)

        z_n = self.enc_normal(
            normal_batch.x, normal_batch.edge_index, normal_batch.edge_attr
        ).view(B, _NB301_N_NODES, -1)
        z_r = self.enc_reduce(
            reduce_batch.x, reduce_batch.edge_index, reduce_batch.edge_attr
        ).view(B, _NB301_N_NODES, -1)

        n_scalars = torch.cat(
            [self.node_mlps_normal[i](z_n[:, i, :]) for i in range(_NB301_N_NODES)],
            dim=-1,
        )
        r_scalars = torch.cat(
            [self.node_mlps_reduce[i](z_r[:, i, :]) for i in range(_NB301_N_NODES)],
            dim=-1,
        )

        out = self.aggregate(
            torch.cat([n_scalars, r_scalars], dim=-1)
        ).squeeze(-1)

        return out.squeeze(0) if squeeze else out
