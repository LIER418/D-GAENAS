"""Visualize NB101 populations for D-GAENAS and its strict mutation ablation.

Both searches share the evaluated initial population, trained surrogate, and
post-initialization RNG state. They differ only in ``generation_method``.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from saea_gae_nb101 import (
    CFG, SAEAEvolution, GINEncoder, NodeFeatureDecoder, EdgeExistDecoder,
    UnifiedNASGAE, make_noise_fn_101, repair_fn_101, N_OPS,
    SurrogateModel, SM_MAX_NODES, NUM_NODE_FEATURES, NUM_EDGE_FEATURES,
    load_nasbench, set_seed,
)

OP_SHORT = ["in", "1x1", "3x3", "MP3", "out"]


def load_config(path, method):
    cfg = dict(CFG)
    with open(path, "r", encoding="utf-8") as handle:
        cfg.update(json.load(handle))
    cfg.update(generation_method=method, n_runs=1, n_jobs=1)
    return cfg


def build_dgae(cfg, device):
    return UnifiedNASGAE(
        encoder=GINEncoder(N_OPS, cfg["hidden_dim"], cfg["latent_dim"],
                           cfg["dropout"]),
        decoders={
            "node_feat": NodeFeatureDecoder(
                cfg["latent_dim"], cfg["node_dec_hidden"], N_OPS),
            "edge_exist": EdgeExistDecoder(
                cfg["latent_dim"], cfg["edge_dec_hidden"]),
        },
        noise_fn=make_noise_fn_101(cfg["node_noise"], cfg["edge_noise"]),
        repair_fn=repair_fn_101,
    ).to(device)


def build_gine_surrogate(cfg, device):
    """Build the same GINE surrogate used by the NB101 experiments."""
    return SurrogateModel(
        SM_MAX_NODES,
        NUM_NODE_FEATURES,
        NUM_EDGE_FEATURES,
        cfg["surr_hidden"],
        cfg["surr_latent"],
    ).to(device)


def clone_initial_state(source, target):
    target.population = [
        [ind[0].clone(), ind[1].clone(), float(ind[2]), float(ind[3])]
        for ind in source.population
    ]
    target.train_archive = copy.deepcopy(source.train_archive)
    target.train_costs = list(source.train_costs)
    target.cache._store = source.cache._store.copy()
    target.cache.hits, target.cache.misses = source.cache.hits, source.cache.misses
    target.surrogate.load_state_dict(source.surrogate.state_dict())
    if source.surr_opt is not None and target.surr_opt is not None:
        target.surr_opt.load_state_dict(source.surr_opt.state_dict())
    target._best_test_acc = source._best_test_acc
    target._query_curve = list(source._query_curve)
    target._incumbent_curve = list(source._incumbent_curve)
    target._global_best = max(target.population, key=lambda ind: ind[2])


def capture_rng():
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return random.getstate(), np.random.get_state(), torch.get_rng_state(), cuda


def restore_rng(state):
    py_state, np_state, torch_state, cuda_state = state
    random.setstate(py_state)
    np.random.set_state(np_state)
    torch.set_rng_state(torch_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)


def node_freq(pop):
    ops = pop.argmax(axis=-1)
    result = np.zeros((pop.shape[1], pop.shape[2]))
    for node in range(pop.shape[1]):
        for op in range(pop.shape[2]):
            result[node, op] = np.mean(ops[:, node] == op)
    return result


def edge_freq(pop):
    return np.mean(pop > 0.5, axis=0)


def snapshot(ea, label, ratio):
    population = sorted(ea.population, key=lambda ind: -ind[2])
    k = max(1, math.ceil(ratio * len(population)))
    return {
        "label": label,
        "elite": np.arange(len(population)) < k,
        "x": torch.stack([ind[0] for ind in population]).cpu().numpy(),
        "adj": torch.stack([ind[1] for ind in population]).cpu().numpy(),
    }


def run_to_budget(ea, budget, max_generations, label):
    base_ratio = float(ea.cfg["eval_elite_ratio"])
    while len(ea.train_archive) < budget and ea.generation < max_generations:
        remaining = budget - len(ea.train_archive)
        ea.cfg["eval_elite_ratio"] = min(
            base_ratio, remaining / len(ea.population))
        stats = ea.step()
        best_observed = ea._global_best[2] if ea._global_best is not None else float("nan")
        print(f"{label} | gen={ea.generation:4d} | "
              f"queries={len(ea.train_archive):4d}/{budget} | "
              f"best_observed={best_observed:.4f} | "
              f"new_eval={stats['n_real_eval']}")
    ea.cfg["eval_elite_ratio"] = base_ratio
    if len(ea.train_archive) < budget:
        print(f"Warning: {label} stopped at {len(ea.train_archive)}/{budget} "
              f"evaluations after {max_generations} generations.")


def plot_rows(rows, output):
    labels = [f"N{i}" for i in range(7)]
    mask = np.triu(np.ones((7, 7), dtype=bool), 1)
    columns = [
        ("en", "Operator distribution / elites", "plasma"),
        ("pn", "Operator distribution / population", "plasma"),
        ("ee", "Edge distribution / elites", "YlOrRd"),
        ("pe", "Edge distribution / population", "YlOrRd"),
    ]
    fig, axes = plt.subplots(len(rows), 4, figsize=(24, 5.5 * len(rows)),
                             squeeze=False)
    fig.subplots_adjust(hspace=0.38, wspace=0.34)
    for row_no, snap in enumerate(rows):
        elite = snap["elite"]
        matrices = {
            "en": node_freq(snap["x"][elite]), "pn": node_freq(snap["x"]),
            "ee": edge_freq(snap["adj"][elite]), "pe": edge_freq(snap["adj"]),
        }
        for col_no, (key, title, cmap) in enumerate(columns):
            ax = axes[row_no, col_no]
            matrix = matrices[key]
            if key in {"ee", "pe"}:
                image = ax.imshow(np.where(mask, matrix, np.nan), cmap=cmap,
                                  vmin=0, vmax=1, aspect="auto")
                ax.set_xticks(range(7), labels); ax.set_yticks(range(7), labels)
                ax.set_xlabel("Target node"); ax.set_ylabel("Source node")
            else:
                image = ax.imshow(matrix.T, cmap=cmap, vmin=0, vmax=1,
                                  aspect="auto")
                ax.set_xticks(range(7), labels)
                ax.set_yticks(range(len(OP_SHORT)), OP_SHORT)
                ax.set_xlabel("Node position"); ax.set_ylabel("Operation")
            if row_no == 0:
                ax.set_title(title, fontsize=14)
            if col_no == 0:
                ax.text(-0.34, 0.5, snap["label"], transform=ax.transAxes,
                        rotation=90, va="center", ha="center", fontsize=15,
                        fontweight="bold")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/nb101_budget400.json")
    parser.add_argument("--mutation-config", default=(
        "configs/ablation_random_mutation_nb101_budget400.json"))
    parser.add_argument("--tfrecord", default=None)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--max-generations", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=(
        "results/nb101/population_visualization_400.png"))
    return parser.parse_args()


def main():
    args = parse_args()
    dgae_cfg = load_config(args.config, "dgae")
    mutation_cfg = dict(dgae_cfg)
    mutation_cfg.update(load_config(
        args.mutation_config, "population_random_mutation"))
    for cfg in (dgae_cfg, mutation_cfg):
        if args.tfrecord is not None: cfg["data_file"] = args.tfrecord
        if args.budget is not None: cfg["eval_budget"] = args.budget
        if args.seed is not None: cfg["seed"] = args.seed
        if args.device is not None: cfg["device"] = args.device
        # Mechanism-analysis protocol: discard parents after every generation
        # so that old/new elitist merging cannot create distribution alignment.
        cfg["population_update"] = "offspring_only"

    budget = int(dgae_cfg["eval_budget"])
    generations = int(args.max_generations or dgae_cfg["n_generations"])
    set_seed(int(dgae_cfg["seed"]))
    nb = load_nasbench(dgae_cfg["data_file"])
    device = torch.device(dgae_cfg["device"])

    dgae = SAEAEvolution(nb, build_dgae(dgae_cfg, device),
                         build_gine_surrogate(dgae_cfg, device), dgae_cfg)
    dgae.pretrain()
    initial = snapshot(dgae,
        f"Initial population\n({len(dgae.train_archive)} evaluations)",
        float(dgae_cfg["gae_elite_ratio"]))

    mutation = SAEAEvolution(nb, build_dgae(mutation_cfg, device),
        build_gine_surrogate(mutation_cfg, device), mutation_cfg)
    clone_initial_state(dgae, mutation)
    shared_rng = capture_rng()

    restore_rng(shared_rng)
    run_to_budget(dgae, budget, generations, "D-GAENAS")
    dgae_final = snapshot(dgae, "D-GAENAS\nAfter search",
                          float(dgae_cfg["gae_elite_ratio"]))

    restore_rng(shared_rng)
    run_to_budget(mutation, budget, generations, "Random mutation")
    mutation_final = snapshot(mutation, "Random mutation\nAfter search",
                              float(mutation_cfg["gae_elite_ratio"]))
    plot_rows([initial, dgae_final, mutation_final], Path(args.out))


if __name__ == "__main__":
    main()
