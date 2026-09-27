
from __future__ import annotations

import torch
import torch.nn.functional as F

from nas_graph import NASGraph


_OP_INPUT   = 0
_OP_OUTPUT  = 4
_INTER_OPS  = [1, 2, 3]
_MAX_EDGES  = 9


def _bfs(adj: torch.Tensor, src: int) -> set:
    N       = adj.size(0)
    visited = {src}
    queue   = [src]
    while queue:
        n = queue.pop(0)
        for j in range(N):
            if adj[n, j] > 0.5 and j not in visited:
                visited.add(j)
                queue.append(j)
    return visited


def _bfs_rev(adj: torch.Tensor, dst: int) -> set:
    N       = adj.size(0)
    visited = {dst}
    queue   = [dst]
    while queue:
        n = queue.pop(0)
        for i in range(N):
            if adj[i, n] > 0.5 and i not in visited:
                visited.add(i)
                queue.append(i)
    return visited


def repair_101(
    x:   torch.Tensor,
    adj: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    N, C    = x.shape
    device  = x.device
    x_rep   = x.clone()
    adj_rep = adj.clone()

    x_rep[0]     = F.one_hot(torch.tensor(_OP_INPUT),  num_classes=C).float().to(device)
    x_rep[N - 1] = F.one_hot(torch.tensor(_OP_OUTPUT), num_classes=C).float().to(device)

    for i in range(1, N - 1):
        if int(x_rep[i].argmax().item()) not in _INTER_OPS:
            nc       = _INTER_OPS[torch.randint(0, len(_INTER_OPS), (1,)).item()]
            x_rep[i] = F.one_hot(torch.tensor(nc), num_classes=C).float().to(device)

    n_edges = int(adj_rep.triu(diagonal=1).sum().item())
    if n_edges > _MAX_EDGES:
        existing = [
            (i, j)
            for i in range(N) for j in range(i + 1, N)
            if adj_rep[i, j].item() > 0.5
        ]
        perm = torch.randperm(len(existing)).tolist()
        for k in perm[: n_edges - _MAX_EDGES]:
            ei, ej = existing[k]
            adj_rep[ei, ej] = 0.0

    for _ in range(N):
        from_input = _bfs(adj_rep, 0)
        if (N - 1) in from_input:
            break

        to_output  = _bfs_rev(adj_rep, N - 1)

        candidates = []
        for s in sorted(from_input):
            for d in range(s + 1, N):
                if d not in from_input and adj_rep[s, d].item() < 0.5:
                    priority = 1 if d in to_output else 0
                    candidates.append((priority, s, d))

        if not candidates:
            break

        candidates.sort(key=lambda t: -t[0])

        if int(adj_rep.triu(diagonal=1).sum().item()) >= _MAX_EDGES:
            non_critical = [
                (i, j)
                for i in range(N) for j in range(i + 1, N)
                if adj_rep[i, j] > 0.5
                and not (i in from_input and j in to_output)
            ]
            if not non_critical:
                break
            ri, rj = non_critical[torch.randint(0, len(non_critical), (1,)).item()]
            adj_rep[ri, rj] = 0.0

        _, s, d = candidates[0]
        adj_rep[s, d] = 1.0

    return x_rep, adj_rep


def repair_fn_101(graph: NASGraph) -> NASGraph:
    B              = graph.batch_size
    node_feat_list = []
    adj_list       = []

    for b in range(B):
        x_rep, adj_rep = repair_101(graph.node_feat[b], graph.adj[b])
        node_feat_list.append(x_rep)
        adj_list.append(adj_rep)

    return NASGraph(
        node_feat  = torch.stack(node_feat_list),
        edge_feat  = graph.edge_feat,
        edge_exist = graph.edge_exist,
        adj        = torch.stack(adj_list),
        edge_index = graph.edge_index,
        n_nodes    = graph.n_nodes,
    )


_DST_GROUPS_301 = [
    [0, 1],
    [2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12, 13],
]


def repair_301_topology(act_samp: torch.Tensor) -> torch.Tensor:
    repaired   = torch.zeros_like(act_samp)
    noisy_samp = act_samp + torch.rand_like(act_samp) * 1e-5

    for indices in _DST_GROUPS_301:
        local_vals        = noisy_samp[..., indices]
        _, top2_local_idx = torch.topk(local_vals, k=2, dim=-1)
        top2_global_idx   = top2_local_idx + indices[0]
        repaired.scatter_(-1, top2_global_idx, 1.0)

    return repaired


def repair_fn_301(graph: NASGraph) -> NASGraph:
    edge_exist_rep = repair_301_topology(graph.edge_exist)

    return NASGraph(
        node_feat  = graph.node_feat,
        edge_feat  = graph.edge_feat,
        edge_exist = edge_exist_rep,
        adj        = graph.adj,
        edge_index = graph.edge_index,
        n_nodes    = graph.n_nodes,
    )
