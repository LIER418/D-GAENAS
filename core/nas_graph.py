
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class NASGraph:

    node_feat  : torch.Tensor | None
    edge_feat  : torch.Tensor | None
    edge_exist : torch.Tensor | None
    adj        : torch.Tensor | None
    edge_index : torch.Tensor
    n_nodes    : int


    @property
    def batch_size(self) -> int:
        for t in (self.node_feat, self.edge_feat, self.adj):
            if t is not None:
                return t.shape[0]
        raise ValueError("NASGraph contains no tensors — cannot infer batch size.")

    @property
    def device(self) -> torch.device:
        return self.edge_index.device


    @staticmethod
    def from_101(x: torch.Tensor, adj: torch.Tensor) -> NASGraph:
        if x.dim() == 2:
            x   = x.unsqueeze(0)
            adj = adj.unsqueeze(0)

        N              = x.shape[1]
        rows, cols     = torch.triu_indices(N, N, offset=1, device=x.device)
        edge_index     = torch.stack([rows, cols])

        return NASGraph(
            node_feat  = x,
            edge_feat  = None,
            edge_exist = None,
            adj        = adj,
            edge_index = edge_index,
            n_nodes    = N,
        )

    @staticmethod
    def from_201(edge_feats: torch.Tensor) -> NASGraph:
        from nasbench201_utils import EDGES, N_NODES

        if edge_feats.dim() == 2:
            edge_feats = edge_feats.unsqueeze(0)

        B         = edge_feats.shape[0]
        node_feat = (
            torch.eye(N_NODES, device=edge_feats.device)
            .unsqueeze(0)
            .expand(B, -1, -1)
        )

        srcs       = torch.tensor([u for u, _ in EDGES], dtype=torch.long, device=edge_feats.device)
        dsts       = torch.tensor([v for _, v in EDGES], dtype=torch.long, device=edge_feats.device)
        edge_index = torch.stack([srcs, dsts])

        return NASGraph(
            node_feat  = node_feat,
            edge_feat  = edge_feats,
            edge_exist = None,
            adj        = None,
            edge_index = edge_index,
            n_nodes    = N_NODES,
        )

    @staticmethod
    def from_301(tensor: torch.Tensor) -> NASGraph:
        from nasbench301_utils import EDGES_301

        B          = tensor.shape[0]
        flat       = tensor.reshape(B * 2, 14, 8)
        edge_exist = flat[..., 0]
        edge_feat  = flat[..., 1:]

        srcs       = torch.tensor([u for u, _ in EDGES_301], dtype=torch.long, device=tensor.device)
        dsts       = torch.tensor([v for _, v in EDGES_301], dtype=torch.long, device=tensor.device)
        edge_index = torch.stack([srcs, dsts])

        return NASGraph(
            node_feat  = None,
            edge_feat  = edge_feat,
            edge_exist = edge_exist,
            adj        = None,
            edge_index = edge_index,
            n_nodes    = 6,
        )

    @staticmethod
    def from_rnas(tensor: torch.Tensor) -> NASGraph:
        from nasbench301_utils import EDGES_301

        B          = tensor.shape[0]
        flat       = tensor.reshape(B * 2, 14, 9)
        edge_exist = flat[..., 0]
        edge_feat  = flat[..., 1:]

        srcs       = torch.tensor([u for u, _ in EDGES_301], dtype=torch.long, device=tensor.device)
        dsts       = torch.tensor([v for _, v in EDGES_301], dtype=torch.long, device=tensor.device)
        edge_index = torch.stack([srcs, dsts])

        return NASGraph(
            node_feat  = None,
            edge_feat  = edge_feat,
            edge_exist = edge_exist,
            adj        = None,
            edge_index = edge_index,
            n_nodes    = 6,
        )

    @staticmethod
    def unflatten_301(tensor: torch.Tensor, B: int) -> torch.Tensor:
        return tensor.reshape(B, 2, *tensor.shape[1:])
