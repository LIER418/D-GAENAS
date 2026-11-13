"""NAS adaptation of Probst and Rothlauf's DAE-based EDA.

The implementation retains tournament selection, a fresh tied-weight DAE per
generation, salt-and-pepper corruption, and a ten-step sampling chain.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .nb101 import Architecture, EDGE_POSITIONS, Oracle


def _reachable(edges: set[int], source: int, reverse: bool = False) -> set[int]:
    """Reachable NB101 nodes under the fixed upper-triangular edge ordering."""
    visited = {source}
    queue = [source]
    while queue:
        node = queue.pop(0)
        for index, (left, right) in enumerate(EDGE_POSITIONS):
            if index not in edges:
                continue
            predecessor, successor = (right, left) if reverse else (left, right)
            if predecessor == node and successor not in visited:
                visited.add(successor)
                queue.append(successor)
    return visited


def random_valid_architecture(
    rng: random.Random, min_edges: int = 1, max_edges: int = 9
) -> Architecture:
    """Sample and repair an NB101 cell using D-GAENAS initialization rules."""
    if not 0 <= min_edges <= max_edges <= 9:
        raise ValueError("edge range must satisfy 0 <= min_edges <= max_edges <= 9")
    operations = (0, *(rng.randint(1, 3) for _ in range(5)), 4)
    edge_count = rng.randint(min_edges, max_edges)
    present = set(rng.sample(range(len(EDGE_POSITIONS)), edge_count))
    edge_map = {edge: index for index, edge in enumerate(EDGE_POSITIONS)}

    # Match repair() in D-GAENAS-release/core/ea.py: iteratively bridge the
    # input-reachable component toward nodes that can already reach output.
    for _ in range(7):
        from_input = _reachable(present, 0)
        if 6 in from_input:
            break
        to_output = _reachable(present, 6, reverse=True)
        candidates = []
        for source in sorted(from_input):
            for target in range(source + 1, 7):
                index = edge_map[(source, target)]
                if target not in from_input and index not in present:
                    candidates.append((1 if target in to_output else 0,
                                       source, target, index))
        if not candidates:
            break
        candidates.sort(key=lambda item: -item[0])
        if len(present) >= 9:
            non_critical = [
                index for index in present
                if not (EDGE_POSITIONS[index][0] in from_input
                        and EDGE_POSITIONS[index][1] in to_output)
            ]
            if not non_critical:
                break
            present.remove(rng.choice(non_critical))
        present.add(candidates[0][3])

    return Architecture(
        tuple(operations),
        tuple(int(index in present) for index in range(len(EDGE_POSITIONS))),
    )


def decode_valid(vector: np.ndarray, rng: random.Random) -> Architecture:
    """Project a 36-dimensional operation/edge proposal into NB101 validity."""
    values = np.asarray(vector, dtype=np.float32).copy()
    edge_scores = values[15:]
    present = np.flatnonzero(edge_scores >= 0.5).tolist()
    if len(present) > 9:
        present = sorted(present, key=lambda k: float(edge_scores[k]), reverse=True)[:9]
    edge_map = {position: index for index, position in enumerate(EDGE_POSITIONS)}

    # Guarantee an input-to-output path without always inserting 0 -> 6. A
    # forced shortcut makes NASBench prune most intermediate nodes and turns
    # many distinct binary proposals into the same canonical architecture.
    path_candidates: list[tuple[float, list[int]]] = []
    for mask in range(1 << 5):
        nodes = [0, *(node for node in range(1, 6) if mask & (1 << (node - 1))), 6]
        indices = [edge_map[(a, b)] for a, b in zip(nodes, nodes[1:])]
        path_candidates.append((float(edge_scores[indices].sum()), indices))
    best_score = max(score for score, _ in path_candidates)
    tied = [indices for score, indices in path_candidates if score == best_score]
    protected = set(tied[rng.randrange(len(tied))])
    present = list(set(present) | protected)
    if len(present) > 9:
        optional = [index for index in present if index not in protected]
        rng.shuffle(optional)
        optional.sort(key=lambda index: float(edge_scores[index]), reverse=True)
        present = list(protected) + optional[:9 - len(protected)]
    values[15:] = 0
    values[15 + np.asarray(present, dtype=int)] = 1
    return Architecture.from_vector(values)


class TiedDAE(nn.Module):
    def __init__(self, dimension: int = 36):
        super().__init__()
        # EDA-suite uses randn(...) * 0.005 for weights and both biases.
        self.weight = nn.Parameter(torch.randn(dimension, dimension) * 0.005)
        self.encoder_bias = nn.Parameter(torch.randn(dimension) * 0.005)
        self.decoder_bias = nn.Parameter(torch.randn(dimension) * 0.005)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.sigmoid(F.linear(x, self.weight, self.encoder_bias))
        return torch.sigmoid(F.linear(h, self.weight.T, self.decoder_bias))


def corrupt(x: torch.Tensor, rate: float = 0.1) -> torch.Tensor:
    mask = torch.rand_like(x) < rate
    salt = torch.randint(0, 2, x.shape, device=x.device).float()
    return torch.where(mask, salt, x)


@dataclass
class DAEConfig:
    # The published combinatorial experiments use 2000. Fifty is the only
    # deliberate budget adaptation: otherwise a 150/400/1000-query NAS run
    # would end during initialization. The original 1:1 parent/candidate ratio
    # is retained.
    population: int = 50
    min_epochs: int = 50
    max_epochs: int = 2000
    min_updates: int = 200
    stats_interval: int = 5
    batch_size: int = 100
    lr_weight: float = 0.1
    lr_bias: float = 0.5
    momentum_weight: float = 0.3
    weight_decay: float = 0.001
    corruption: float = 0.1
    sampling_steps: int = 10
    max_generations: int = 150
    use_bit_flip_mask: bool = True
    max_proposals_per_evaluation: int = 200


def _smoothed_errors(errors: list[float]) -> list[float]:
    """Reproduce the adaptive moving average used by EDA-suite."""
    count = len(errors)
    radius = 2 if count < 20 else 3 if count < 30 else 4
    if count <= radius:
        return []
    smoothed = []
    for center in range(count - radius):
        values = [errors[max(0, center + offset)] for offset in range(-radius, radius + 1)]
        smoothed.append(float(np.mean(values)))
    return smoothed


def fit_dae(population: list[Architecture], cfg: DAEConfig,
            bit_flip_mask: np.ndarray | None = None) -> TiedDAE:
    model = TiedDAE()
    model.train()
    encoded = np.stack([a.vector() for a in population])
    if bit_flip_mask is not None:
        encoded = np.logical_xor(encoded >= 0.5, bit_flip_mask).astype(np.float32)
    data = torch.tensor(encoded, dtype=torch.float32)
    # The Octave implementation uses min(configured batch, floor(n/5)).
    batch_size = min(cfg.batch_size, max(1, len(data) // 5))
    optimizer = torch.optim.SGD([
        {"params": [model.weight], "lr": cfg.lr_weight, "momentum": cfg.momentum_weight,
         "weight_decay": cfg.weight_decay},
        {"params": [model.encoder_bias, model.decoder_bias], "lr": cfg.lr_bias},
    ])
    errors: list[float] = []
    best_state = None
    best_loss = float("inf")
    updates = 0
    recon_data = data[:min(len(data), 3 * batch_size)]
    recon_corrupted = corrupt(recon_data, cfg.corruption)
    for epoch in range(cfg.max_epochs):
        shuffled = data[torch.randperm(len(data))]
        corrupted = corrupt(shuffled, cfg.corruption)
        full = (len(data) // batch_size) * batch_size
        for start in range(0, full, batch_size):
            batch = shuffled[start:start + batch_size]
            pred = model(corrupted[start:start + batch_size])
            loss = F.binary_cross_entropy(pred.clamp(1e-6, 1-1e-6), batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            updates += 1
        if (epoch + 1) % cfg.stats_interval == 0 or epoch + 1 == cfg.max_epochs:
            with torch.no_grad():
                current = float(F.binary_cross_entropy(
                    model(recon_corrupted).clamp(1e-6, 1 - 1e-6), recon_data))
            errors.append(current)
            if current < best_loss:
                best_loss = current
                best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            if epoch + 1 > cfg.min_epochs and updates > cfg.min_updates:
                smoothed = _smoothed_errors(errors)
                if len(smoothed) >= 2:
                    total_gain = smoothed[0] - smoothed[-1]
                    if total_gain <= 0:
                        break
                    earlier = smoothed[math.ceil(2 * len(smoothed) / 3) - 1]
                    if (earlier - smoothed[-1]) / total_gain < 0.05:
                        break
    assert best_state is not None
    model.load_state_dict(best_state)
    return model.eval()


def sample_dae(model: TiedDAE, cfg: DAEConfig, rng: random.Random,
               bit_flip_mask: np.ndarray | None = None) -> Architecture:
    with torch.no_grad():
        state = torch.randint(0, 2, (1, 36)).float()
        for _ in range(cfg.sampling_steps):
            state = model(corrupt(state, cfg.corruption))
        state = torch.bernoulli(state)
    vector = state[0].numpy()
    if bit_flip_mask is not None:
        vector = np.logical_xor(vector >= 0.5, bit_flip_mask).astype(np.float32)
    return decode_valid(vector, rng)


def tournament_half(population: list[tuple[Architecture, float]],
                    rng: random.Random) -> list[tuple[Architecture, float]]:
    """Random-pair tournament selection from the released EDA-suite."""
    indices = list(range(len(population)))
    rng.shuffle(indices)
    half = len(indices) // 2
    winners = []
    for left, right in zip(indices[:half], indices[half:2 * half]):
        # The original strict comparison chooses the second item on a tie.
        winners.append(population[left] if population[left][1] > population[right][1]
                       else population[right])
    return winners


def run_dae_eda(oracle: Oracle, rng: random.Random, cfg: DAEConfig) -> dict:
    """Adapted DAE-EDA search; all oracle labels are acquired online."""
    population: list[tuple[Architecture, float]] = []
    proposal_limit = cfg.max_proposals_per_evaluation * oracle.budget
    proposals = 0
    initial_count = min(cfg.population, oracle.budget)
    while len(population) < initial_count:
        arch = random_valid_architecture(rng, min_edges=6, max_edges=9)
        score = oracle.evaluate(arch)
        if score is not None:
            population.append((arch, score[0]))
    bit_flip_mask = None
    if cfg.use_bit_flip_mask:
        bit_flip_mask = np.asarray([rng.randint(0, 1) for _ in range(36)], dtype=bool)
    generations = 0
    while (len(oracle.cache) < oracle.budget and proposals < proposal_limit
           and generations < cfg.max_generations):
        generations += 1
        selected = tournament_half(population, rng)
        model = fit_dae([arch for arch, _ in selected], cfg, bit_flip_mask)
        before = len(oracle.cache)
        offspring: list[tuple[Architecture, float]] = []
        target = cfg.population // 2
        while (len(offspring) < target and len(oracle.cache) < oracle.budget
               and proposals < proposal_limit):
            proposals += 1
            arch = sample_dae(model, cfg, rng, bit_flip_mask)
            result = oracle.evaluate(arch)
            if result is not None:
                offspring.append((arch, result[0]))
        population = selected + offspring
        # If the last partial generation exhausted the query budget, its size
        # need not equal the configured population.
        if len(oracle.cache) == before and not offspring:
            break
    if len(oracle.cache) == oracle.budget:
        status = "budget_complete"
    elif generations >= cfg.max_generations:
        status = "generation_limit"
    else:
        status = "proposal_limit"
    return {"proposals": proposals, "generations": generations, "status": status,
            "population_size_adaptation": "2000 in source configuration; 50 for NAS query budgets"}
