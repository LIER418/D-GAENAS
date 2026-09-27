from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from nas_graph import NASGraph


def make_noise_fn_101(
    node_prob: float = 0.1,
    edge_prob: float = 0.1,
) -> Callable[[NASGraph], NASGraph]:
    def _noise(graph: NASGraph) -> NASGraph:
        x   = graph.node_feat.clone()
        adj = graph.adj.clone()

        if node_prob > 0.0:
            B, N, C = x.shape
            mask    = torch.rand(B, N, device=x.device) < node_prob
            orig    = x.argmax(dim=-1)
            shift   = torch.randint(1, C, (B, N), device=x.device)
            new_cls = (orig + shift) % C
            noisy_x = x.clone()
            noisy_x[mask] = F.one_hot(new_cls[mask], num_classes=C).float()
            x = noisy_x

        if edge_prob > 0.0:
            B, N, _ = adj.shape
            triu    = torch.triu(
                torch.ones(N, N, device=adj.device, dtype=torch.bool), diagonal=1
            )
            flip    = (torch.rand(B, N, N, device=adj.device) < edge_prob) & triu
            adj[flip] = 1.0 - adj[flip]

        return NASGraph(
            node_feat  = x,
            edge_feat  = graph.edge_feat,
            edge_exist = graph.edge_exist,
            adj        = adj,
            edge_index = graph.edge_index,
            n_nodes    = graph.n_nodes,
        )

    return _noise


def make_noise_fn_201(prob: float = 0.15) -> Callable[[NASGraph], NASGraph]:
    def _noise(graph: NASGraph) -> NASGraph:
        ef = graph.edge_feat.clone()
        if prob > 0.0:
            B, E, C = ef.shape
            mask    = torch.rand(B, E, device=ef.device) < prob
            curr    = ef.argmax(dim=-1)
            shift   = torch.randint(1, C, (B, E), device=ef.device)
            new_op  = (curr + shift) % C
            ef[mask] = F.one_hot(new_op[mask], num_classes=C).float()

        return NASGraph(
            node_feat  = graph.node_feat,
            edge_feat  = ef,
            edge_exist = graph.edge_exist,
            adj        = graph.adj,
            edge_index = graph.edge_index,
            n_nodes    = graph.n_nodes,
        )

    return _noise


def make_noise_fn_301(prob: float = 0.15) -> Callable[[NASGraph], NASGraph]:
    def _noise(graph: NASGraph) -> NASGraph:
        ef = graph.edge_feat.clone()
        ee = graph.edge_exist.clone()

        if prob > 0.0:
            B, E, C = ef.shape

            mask_act      = torch.rand(B, E, device=ee.device) < prob
            ee[mask_act]  = 1.0 - ee[mask_act]

            mask_op = torch.rand(B, E, device=ef.device) < prob
            curr    = ef.argmax(dim=-1)
            shift   = torch.randint(1, C, (B, E), device=ef.device)
            new_op  = (curr + shift) % C
            ef[mask_op] = F.one_hot(new_op[mask_op], num_classes=C).float()

        return NASGraph(
            node_feat  = graph.node_feat,
            edge_feat  = ef,
            edge_exist = ee,
            adj        = graph.adj,
            edge_index = graph.edge_index,
            n_nodes    = graph.n_nodes,
        )

    return _noise


class GINLayer(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, mlp_hidden: int):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(in_dim,    mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, out_dim),
        )

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        agg = adj @ h
        return self.mlp((1.0 + self.eps) * h + agg)


class GINELayer(nn.Module):

    def __init__(
        self,
        node_in_dim:   int,
        edge_feat_dim: int,
        out_dim:       int,
        mlp_hidden:    int,
    ):
        super().__init__()
        self.eps       = nn.Parameter(torch.zeros(1))
        self.edge_proj = nn.Linear(edge_feat_dim, node_in_dim, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(node_in_dim, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden,  out_dim),
        )

    def forward(
        self,
        h:          torch.Tensor,
        edge_feat:  torch.Tensor,
        edge_index: torch.Tensor,
        edge_exist: torch.Tensor | None,
    ) -> torch.Tensor:
        B, N, D = h.shape
        E       = edge_feat.shape[1]
        srcs    = edge_index[0]
        dsts    = edge_index[1]

        h_src = h[:, srcs, :]
        msg   = F.relu(h_src + self.edge_proj(edge_feat))

        if edge_exist is not None:
            msg = msg * edge_exist.unsqueeze(-1)

        dsts_exp = dsts.view(1, -1, 1).expand(B, E, D)
        agg      = torch.zeros(B, N, D, device=h.device)
        agg.scatter_add_(1, dsts_exp, msg)

        return self.mlp((1.0 + self.eps) * h + agg)


class GINEncoder(nn.Module):

    def __init__(
        self,
        node_feat_dim: int,
        hidden_dim:    int,
        latent_dim:    int,
        dropout:       float = 0.0,
    ):
        super().__init__()
        self.gin1    = GINLayer(node_feat_dim, hidden_dim, hidden_dim)
        self.gin2    = GINLayer(hidden_dim,    latent_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph: NASGraph) -> torch.Tensor:
        h       = graph.node_feat
        adj_sym = (graph.adj + graph.adj.transpose(-1, -2)).clamp(0.0, 1.0)
        h       = self.gin1(h, adj_sym)
        h       = self.dropout(h)
        return self.gin2(h, adj_sym)


class GINEEncoder(nn.Module):

    def __init__(
        self,
        edge_feat_dim: int,
        hidden_dim:    int,
        latent_dim:    int,
        node_feat_dim: int | None = None,
        n_nodes:       int | None = None,
        dropout:       float = 0.0,
    ):
        super().__init__()

        if node_feat_dim is None:
            if n_nodes is None:
                raise ValueError("n_nodes must be provided when node_feat_dim is None.")
            self.node_emb = nn.Parameter(torch.randn(n_nodes, hidden_dim))
            in_node_dim   = hidden_dim
        else:
            self.node_emb = None
            in_node_dim   = node_feat_dim

        self.gine1   = GINELayer(in_node_dim, edge_feat_dim, hidden_dim, hidden_dim)
        self.gine2   = GINELayer(hidden_dim,  edge_feat_dim, latent_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph: NASGraph) -> torch.Tensor:
        B  = graph.edge_feat.shape[0]
        h  = (
            self.node_emb.unsqueeze(0).expand(B, -1, -1)
            if self.node_emb is not None
            else graph.node_feat
        )
        ef = graph.edge_feat
        ei = graph.edge_index
        ee = graph.edge_exist

        h = self.gine1(h, ef, ei, ee)
        h = self.dropout(h)
        return self.gine2(h, ef, ei, ee)


class NodeFeatureDecoder(nn.Module):

    def __init__(self, latent_dim: int, hidden_dim: int, node_feat_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim,  hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_feat_dim),
        )

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.mlp(z)


class EdgeExistDecoder(nn.Module):

    def __init__(self, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        srcs  = edge_index[0]
        dsts  = edge_index[1]
        z_src = z[:, srcs, :]
        z_dst = z[:, dsts, :]
        pair  = torch.cat([z_src, z_dst], dim=-1)
        return self.mlp(pair).squeeze(-1)


class EdgeFeatureDecoder(nn.Module):

    def __init__(self, latent_dim: int, hidden_dim: int, edge_feat_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, edge_feat_dim),
        )

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        srcs  = edge_index[0]
        dsts  = edge_index[1]
        z_src = z[:, srcs, :]
        z_dst = z[:, dsts, :]
        pair  = torch.cat([z_src, z_dst], dim=-1)
        return self.mlp(pair)


class UnifiedNASGAE(nn.Module):

    def __init__(
        self,
        encoder:          nn.Module,
        decoders:         dict[str, nn.Module],
        noise_fn:         Callable[[NASGraph], NASGraph] | None = None,
        repair_fn:        Callable[[NASGraph], NASGraph] | None = None,
        decoder_weights:  dict[str, float] | None = None,
    ):
        super().__init__()
        self.encoder          = encoder
        self.decoders         = nn.ModuleDict(decoders)
        self.noise_fn         = noise_fn
        self.repair_fn        = repair_fn
        self.decoder_weights  = decoder_weights or {}


    def encode(self, graph: NASGraph) -> torch.Tensor:
        return self.encoder(graph)


    def compute_loss(
        self,
        graph: NASGraph,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        in_graph = self.noise_fn(graph) if (self.training and self.noise_fn) else graph
        z        = self.encode(in_graph)

        total = z.new_zeros(1).squeeze()
        info: dict[str, float] = {}

        if 'node_feat' in self.decoders and graph.node_feat is not None:
            node_logits  = self.decoders['node_feat'](z, graph.edge_index)
            B, N, C      = node_logits.shape
            node_targets = graph.node_feat.argmax(dim=-1)
            loss_nf      = F.cross_entropy(
                node_logits.reshape(B * N, C),
                node_targets.reshape(B * N),
            )
            total = total + self.decoder_weights.get('node_feat', 1.0) * loss_nf
            info['node_feat_loss'] = loss_nf.item()

        if 'edge_exist' in self.decoders:
            exist_logits = self.decoders['edge_exist'](z, graph.edge_index)
            if graph.adj is not None:
                srcs, dsts    = graph.edge_index
                exist_targets = graph.adj[:, srcs, dsts]
            else:
                exist_targets = graph.edge_exist
            loss_ee = F.binary_cross_entropy_with_logits(exist_logits, exist_targets)
            total   = total + self.decoder_weights.get('edge_exist', 1.0) * loss_ee
            info['edge_exist_loss'] = loss_ee.item()

        if 'edge_feat' in self.decoders and graph.edge_feat is not None:
            feat_logits  = self.decoders['edge_feat'](z, graph.edge_index)
            B, E, C      = feat_logits.shape
            feat_targets = graph.edge_feat.argmax(dim=-1)

            if graph.edge_exist is not None:
                mask = graph.edge_exist > 0.5
                if mask.any():
                    loss_ef = F.cross_entropy(feat_logits[mask], feat_targets[mask])
                else:
                    loss_ef = z.new_zeros(1).squeeze()
            else:
                loss_ef = F.cross_entropy(
                    feat_logits.reshape(B * E, C),
                    feat_targets.reshape(B * E),
                )
            total = total + self.decoder_weights.get('edge_feat', 1.0) * loss_ef
            info['edge_feat_loss'] = loss_ef.item()

        return total, info


    @torch.no_grad()
    def reconstruct(self, graph: NASGraph) -> NASGraph:
        was_training = self.training
        self.eval()

        z = self.encode(graph)
        B  = graph.batch_size
        ei = graph.edge_index

        node_feat_out = None
        if 'node_feat' in self.decoders:
            logits        = self.decoders['node_feat'](z, ei)
            C             = logits.shape[-1]
            probs         = torch.softmax(logits, dim=-1)
            idx           = torch.multinomial(
                probs.reshape(-1, C), num_samples=1
            ).reshape(B, -1)
            node_feat_out = F.one_hot(idx, num_classes=C).float()

        edge_exist_out = None
        adj_out        = None
        if 'edge_exist' in self.decoders:
            logits         = self.decoders['edge_exist'](z, ei)
            edge_exist_out = torch.bernoulli(torch.sigmoid(logits))
            if graph.adj is not None:
                N       = graph.n_nodes
                adj_out = z.new_zeros(B, N, N)
                srcs, dsts = ei
                adj_out[:, srcs, dsts] = edge_exist_out

        edge_feat_out = None
        if 'edge_feat' in self.decoders:
            logits        = self.decoders['edge_feat'](z, ei)
            C             = logits.shape[-1]
            probs         = torch.softmax(logits, dim=-1)
            idx           = torch.multinomial(
                probs.reshape(-1, C), num_samples=1
            ).reshape(B, -1)
            edge_feat_out = F.one_hot(idx, num_classes=C).float()

        out = NASGraph(
            node_feat  = node_feat_out,
            edge_feat  = edge_feat_out,
            edge_exist = edge_exist_out,
            adj        = adj_out,
            edge_index = ei,
            n_nodes    = graph.n_nodes,
        )

        if self.repair_fn is not None:
            out = self.repair_fn(out)

        if was_training:
            self.train()

        return out
