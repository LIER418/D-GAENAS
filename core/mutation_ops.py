
from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from repair_ops import repair_101, repair_301_topology
from nasbench301_utils import (
    Genotype301, PRIMITIVES_301, MAX_SRC_301, CONCAT_301, N_OPS_301,
    EDGES_301, N_EDGES_DAG_301,
    genotype_to_tensor, tensor_to_genotype,
    mutate_arch_301,
)
from nasbench201_utils import N_OPS
from ea import N_OPS as N_OPS_101


def mutate_101(
    x:         torch.Tensor,
    adj:       torch.Tensor,
    node_prob: float = 0.15,
    edge_prob: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor]:
    x   = x.clone().cpu()
    adj = adj.clone().cpu()
    N, C = x.shape

    if node_prob > 0.0:
        for i in range(N):
            if random.random() < node_prob:
                orig   = int(x[i].argmax().item())
                shift  = random.randint(1, C - 1)
                new_c  = (orig + shift) % C
                x[i]   = F.one_hot(torch.tensor(new_c), num_classes=C).float()

    if edge_prob > 0.0:
        for i in range(N):
            for j in range(i + 1, N):
                if random.random() < edge_prob:
                    adj[i, j] = 1.0 - adj[i, j]

    return repair_101(x, adj)


def mutate_batch_101(
    pop:       list,
    node_prob: float = 0.15,
    edge_prob: float = 0.15,
) -> list:
    return [mutate_101(ind[0], ind[1], node_prob, edge_prob) for ind in pop]


def mutate_201(
    edge_feats: torch.Tensor,
    prob:       float = 0.15,
) -> torch.Tensor:
    ef  = edge_feats.clone().cpu()
    E, C = ef.shape

    for e in range(E):
        if random.random() < prob:
            orig  = int(ef[e].argmax().item())
            shift = random.randint(1, C - 1)
            new_c = (orig + shift) % C
            ef[e] = F.one_hot(torch.tensor(new_c), num_classes=C).float()

    return ef


def mutate_batch_201(
    pop:  list,
    prob: float = 0.15,
) -> list:
    return [mutate_201(ind[0], prob) for ind in pop]


def mutate_301_tensor(
    tensor: torch.Tensor,
    prob:   float = 0.15,
) -> torch.Tensor:
    t = tensor.clone().cpu()
    N_CELLS, N_EDGES, FEAT = t.shape
    N_OPS = FEAT - 1

    for c in range(N_CELLS):
        for e in range(N_EDGES):
            if random.random() < prob:
                t[c, e, 0] = 1.0 - t[c, e, 0]
            if random.random() < prob:
                orig   = int(t[c, e, 1:].argmax().item())
                shift  = random.randint(1, N_OPS - 1)
                new_op = (orig + shift) % N_OPS
                t[c, e, 1:] = F.one_hot(torch.tensor(new_op), num_classes=N_OPS).float()

    act = t[:, :, 0]
    act_rep = repair_301_topology(act)
    t[:, :, 0] = act_rep

    return t


def mutate_301_geno(
    genotype: Genotype301,
    prob:     float = 0.15,
) -> Genotype301:
    return mutate_arch_301(genotype, mut_prob=prob)


def mutate_batch_301_tensor(
    pop:  list,
    prob: float = 0.15,
) -> list:
    return [mutate_301_tensor(ind[0], prob) for ind in pop]


def mutate_batch_301_geno(
    pop:  list,
    prob: float = 0.15,
) -> list:
    return [mutate_301_geno(ind[1], prob) for ind in pop]


def batch_noise_101(
    x:         torch.Tensor,
    adj:       torch.Tensor,
    node_prob: float = 0.15,
    edge_prob: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor]:
    x   = x.clone()
    adj = adj.clone()

    if node_prob > 0.0:
        B, N, C = x.shape
        mask    = torch.rand(B, N, device=x.device) < node_prob
        orig    = x.argmax(dim=-1)
        shift   = torch.randint(1, C, (B, N), device=x.device)
        new_cls = (orig + shift) % C
        x[mask] = F.one_hot(new_cls[mask], num_classes=C).float()

    if edge_prob > 0.0:
        B, N, _ = adj.shape
        triu    = torch.triu(torch.ones(N, N, device=adj.device, dtype=torch.bool), diagonal=1)
        flip    = (torch.rand(B, N, N, device=adj.device) < edge_prob) & triu
        adj[flip] = 1.0 - adj[flip]

    return x, adj


def batch_noise_201(
    edge_feats: torch.Tensor,
    prob:       float = 0.15,
) -> torch.Tensor:
    ef = edge_feats.clone()
    if prob > 0.0:
        B, E, C = ef.shape
        mask   = torch.rand(B, E, device=ef.device) < prob
        curr   = ef.argmax(dim=-1)
        shift  = torch.randint(1, C, (B, E), device=ef.device)
        new_op = (curr + shift) % C
        ef[mask] = F.one_hot(new_op[mask], num_classes=C).float()
    return ef


def batch_noise_301(
    tensor: torch.Tensor,
    prob:   float = 0.15,
) -> torch.Tensor:
    t  = tensor.clone()
    B, CELLS, E, FEAT = t.shape
    N_OPS = FEAT - 1

    act = t[:, :, :, 0].reshape(B * CELLS, E)
    ops = t[:, :, :, 1:].reshape(B * CELLS, E, N_OPS)

    if prob > 0.0:
        BC = B * CELLS
        mask_act = torch.rand(BC, E, device=t.device) < prob
        act[mask_act] = 1.0 - act[mask_act]

        mask_op = torch.rand(BC, E, device=t.device) < prob
        curr    = ops.argmax(dim=-1)
        shift   = torch.randint(1, N_OPS, (BC, E), device=t.device)
        new_op  = (curr + shift) % N_OPS
        ops[mask_op] = F.one_hot(new_op[mask_op], num_classes=N_OPS).float()

    t[:, :, :, 0]  = act.reshape(B, CELLS, E)
    t[:, :, :, 1:] = ops.reshape(B, CELLS, E, N_OPS)
    return t
