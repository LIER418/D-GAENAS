"""Compare the initial, DGAE, and random-mutation populations at 1000 queries."""

import argparse
import math
import os
import random
import sys
import types

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from saea_gae_nb101 import (
    CFG as SAEA_CFG,
    SAEAEvolution,
    GINEncoder,
    NodeFeatureDecoder,
    EdgeExistDecoder,
    UnifiedNASGAE,
    make_noise_fn_101,
    repair_fn_101,
    N_OPS,
    SurrogateModel,
    SM_MAX_NODES,
    NUM_NODE_FEATURES,
    NUM_EDGE_FEATURES,
)
from ablation_nb101 import (
    CFG as ABLATION_CFG,
    AblationMutation,
    tensor_to_pyg as ablation_tensor_to_pyg,
    train_surrogate_ranking as train_ablation_surrogate,
    individual_hash as ablation_individual_hash,
)


QUERY_BUDGET = 1000
ELITE_RATIO = 0.1
OP_SHORT = ["in", "1x1", "3x3", "MP3", "out"]


def _node_op_freq(x_pop: np.ndarray) -> np.ndarray:
    """Return the operation frequency at each node position."""
    ops = x_pop.argmax(axis=-1)
    n_nodes = x_pop.shape[1]
    n_ops = x_pop.shape[2]
    freq = np.zeros((n_nodes, n_ops), dtype=float)
    for pos in range(n_nodes):
        for op in range(n_ops):
            freq[pos, op] = (ops[:, pos] == op).mean()
    return freq


def _edge_freq(adj_pop: np.ndarray) -> np.ndarray:
    """Return the fraction of architectures containing each directed edge."""
    return (adj_pop > 0.5).astype(float).mean(axis=0)


def snapshot(ea, label, elite_ratio=ELITE_RATIO):
    """Capture the current sorted population and its elite membership."""
    if hasattr(ea, "_sort_population"):
        ea._sort_population()
    else:
        ea.population.sort(key=lambda ind: -ind[2])
    k = max(1, math.ceil(elite_ratio * len(ea.population)))
    is_elite = np.zeros(len(ea.population), dtype=bool)
    is_elite[:k] = True
    return {
        "label": label,
        "queries": ea.cache.misses,
        "is_elite": is_elite,
        "x_pop": torch.stack([ind[0] for ind in ea.population]).cpu().numpy(),
        "adj_pop": torch.stack([ind[1] for ind in ea.population]).cpu().numpy(),
    }


def run_saea_to_budget(ea, query_budget, max_generations=500):
    """Run SAEAEvolution unchanged while preventing final-step overshoot."""
    stagnation_limit = ea.cfg.get("restart_stagnation")
    stagnation_count = 0
    original_eval_ratio = ea.cfg["eval_elite_ratio"]
    while (len(ea.train_archive) < query_budget
           and ea.generation < max_generations):
        remaining = query_budget - len(ea.train_archive)
        # SAEAEvolution.step computes ceil(ratio * population size). Restrict
        # that value only when fewer than the normal five evaluations remain.
        ea.cfg["eval_elite_ratio"] = min(
            original_eval_ratio, remaining / len(ea.population)
        )
        stats = ea.step()
        ea.cfg["eval_elite_ratio"] = original_eval_ratio

        if stagnation_limit is not None:
            if len({ind[2] for ind in ea.population}) == 1:
                stagnation_count += 1
            else:
                stagnation_count = 0
            if (stagnation_count >= stagnation_limit
                    and len(ea.train_archive) < query_budget):
                stagnation_count = 0
                ea._restart()

        print(
            f"D-GAENAS | gen={ea.generation:4d} | "
            f"queries={len(ea.train_archive):4d}/{query_budget} | "
            f"best={stats['best_fitness']:.4f}"
        )
    ea.cfg["eval_elite_ratio"] = original_eval_ratio
    if len(ea.train_archive) < query_budget:
        print(
            f"D-GAENAS stopped at the {max_generations}-generation limit "
            f"with {len(ea.train_archive)} evaluations."
        )


def initialise_ablation_from_population(algo, initial_population):
    """Apply AblationMutation pretraining to the shared D-GAENAS population."""
    algo.population = []
    for source in initial_population:
        x, adj, val_acc, test_acc = source
        x = x.clone().to(algo.device)
        adj = adj.clone().to(algo.device)
        ind = [x, adj, float(val_acc), float(test_acc)]
        algo.population.append(ind)
        algo.cache.put(ablation_individual_hash(x, adj), float(val_acc))
        algo.train_archive.append(
            ablation_tensor_to_pyg(x, adj, accuracy=float(val_acc))
        )
        algo.train_costs.append(float(val_acc))
        algo._best_test_acc = max(algo._best_test_acc, float(test_acc))
        algo._query_curve.append((len(algo.train_archive), algo._best_test_acc))

    for _ in range(algo.cfg["pretrain_surr_epochs"]):
        train_ablation_surrogate(
            algo.surrogate,
            algo.train_archive,
            algo.train_costs,
            algo.surr_opt,
            margin=algo.cfg["surr_margin"],
        )
    algo.population.sort(key=lambda ind: -ind[2])


def run_ablation_to_budget(algo, query_budget, max_generations=500):
    """Run AblationMutation and stop exactly at the requested archive size."""
    original_eval_ratio = algo.cfg["eval_elite_ratio"]
    while (len(algo.train_archive) < query_budget
           and algo.generation < max_generations):
        remaining = query_budget - len(algo.train_archive)
        algo.cfg["eval_elite_ratio"] = min(
            original_eval_ratio, remaining / len(algo.population)
        )
        stats = algo.step()
        algo.cfg["eval_elite_ratio"] = original_eval_ratio
        print(
            f"Mutation | gen={algo.generation:4d} | "
            f"queries={len(algo.train_archive):4d}/{query_budget} | "
            f"best={stats['best_fitness']:.4f}"
        )
    algo.cfg["eval_elite_ratio"] = original_eval_ratio
    if len(algo.train_archive) < query_budget:
        print(
            f"Mutation ablation stopped at the {max_generations}-generation "
            f"limit with {len(algo.train_archive)} evaluations."
        )


def plot_three_by_four(rows, out_path):
    """Rows are experiment stages; columns are the four requested views."""
    node_labels = [f"N{i}" for i in range(7)]
    upper_mask = np.triu(np.ones((7, 7), dtype=bool), k=1)
    columns = [
        ("elite_node", "Operator distribution / elites", "plasma"),
        ("pop_node", "Operator distribution / population", "plasma"),
        ("elite_edge", "Edge distribution / elites", "YlOrRd"),
        ("pop_edge", "Edge distribution / population", "YlOrRd"),
    ]

    fig, axes = plt.subplots(3, 4, figsize=(48, 30), squeeze=False)
    fig.subplots_adjust(hspace=0.42, wspace=0.38)

    for row_index, snap in enumerate(rows):
        pop_node = _node_op_freq(snap["x_pop"])
        elite_node = _node_op_freq(snap["x_pop"][snap["is_elite"]])
        pop_edge = _edge_freq(snap["adj_pop"])
        elite_edge = _edge_freq(snap["adj_pop"][snap["is_elite"]])
        matrices = {
            "elite_node": elite_node,
            "pop_node": pop_node,
            "elite_edge": elite_edge,
            "pop_edge": pop_edge,
        }

        for col_index, (key, heading, cmap) in enumerate(columns):
            ax = axes[row_index, col_index]
            matrix = matrices[key]
            is_edge = key.endswith("edge")
            if is_edge:
                matrix = np.where(upper_mask, matrix, np.nan)
                image = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
                ax.set_xticks(range(7), node_labels, fontsize=28)
                ax.set_yticks(range(7), node_labels, fontsize=28)
                ax.set_xlabel("Target node", fontsize=32)
                ax.set_ylabel("Source node", fontsize=32)
            else:
                image = ax.imshow(matrix.T, cmap=cmap, vmin=0, vmax=1, aspect="auto")
                ax.set_xticks(range(7), node_labels, fontsize=28)
                ax.set_yticks(range(len(OP_SHORT)), OP_SHORT, fontsize=28)
                ax.set_xlabel("Node position", fontsize=32)
                ax.set_ylabel("Operation", fontsize=32)

            title = heading if row_index == 0 else ""
            ax.set_title(title, fontsize=38, pad=20)
            if col_index == 0:
                ax.text(
                    -0.34, 0.5, snap["label"], transform=ax.transAxes,
                    rotation=90, va="center", ha="center", fontsize=40,
                    fontweight="bold",
                )
            colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            colorbar.ax.tick_params(labelsize=24)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="3x4 population comparison at 1000 queries")
    parser.add_argument("--tfrecord", default="data/nasbench_only108.tfrecord")
    parser.add_argument("--budget", type=int, default=QUERY_BUDGET)
    parser.add_argument("--max_generations", type=int, default=500)
    parser.add_argument(
        "--seed", type=int, default=None,
        help="override both algorithms' native seeds",
    )
    parser.add_argument("--out", default="viz_output")
    parser.add_argument("--device", default=None, help="override configured device")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_saea_gae(cfg, device):
    """Build the exact GAE architecture configured by saea_gae_nb101.py."""
    return UnifiedNASGAE(
        encoder=GINEncoder(
            node_feat_dim=N_OPS,
            hidden_dim=cfg["hidden_dim"],
            latent_dim=cfg["latent_dim"],
            dropout=cfg["dropout"],
        ),
        decoders={
            "node_feat": NodeFeatureDecoder(
                cfg["latent_dim"], cfg["node_dec_hidden"], N_OPS
            ),
            "edge_exist": EdgeExistDecoder(
                cfg["latent_dim"], cfg["edge_dec_hidden"]
            ),
        },
        noise_fn=make_noise_fn_101(cfg["node_noise"], cfg["edge_noise"]),
        repair_fn=repair_fn_101,
    ).to(device)


def main():
    args = parse_args()
    dgae_seed = SAEA_CFG["seed"] if args.seed is None else args.seed
    ablation_seed = ABLATION_CFG["seed"] if args.seed is None else args.seed
    run_device = SAEA_CFG["device"] if args.device is None else args.device
    min_budget = SAEA_CFG["n_init"]
    if args.budget < min_budget:
        raise ValueError(f"--budget must be at least the initial population size ({min_budget}).")
    if args.max_generations < 1:
        raise ValueError("--max_generations must be at least 1.")
    os.makedirs(args.out, exist_ok=True)
    set_seed(dgae_seed)

    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    for module_name in (
        "nasbench.lib.evaluate", "nasbench.lib.model_builder", "nasbench.lib.training_time"
    ):
        sys.modules.setdefault(module_name, types.ModuleType(module_name))

    import tensorflow as tf
    if not hasattr(tf, "python_io"):
        setattr(tf, "python_io", tf.compat.v1.python_io)
    tf.get_logger().setLevel("ERROR")
    from nasbench import api as nb_api

    print(f"Loading {args.tfrecord} ...")
    nb = nb_api.NASBench(args.tfrecord)
    # Full D-GAENAS implementation from saea_gae_nb101.py.
    dgae_cfg = dict(SAEA_CFG)
    dgae_cfg.update({
        "data_file": args.tfrecord,
        "device": run_device,
        "seed": dgae_seed,
        "eval_budget": args.budget,
    })
    device = torch.device(run_device)
    gae = build_saea_gae(dgae_cfg, device)
    surrogate = SurrogateModel(
        num_nodes=SM_MAX_NODES,
        node_channels=NUM_NODE_FEATURES,
        edge_channels=NUM_EDGE_FEATURES,
        hidden_channels=dgae_cfg["surr_hidden"],
        latent_channels=dgae_cfg["surr_latent"],
    ).to(device)
    dgae = SAEAEvolution(nb, gae, surrogate, dgae_cfg)
    dgae.pretrain()
    initial_population = [
        (ind[0].clone(), ind[1].clone(), ind[2], ind[3])
        for ind in dgae.population
    ]
    initial = snapshot(
        dgae,
        f"Initial population\n({len(dgae.train_archive)} evaluations)",
        elite_ratio=dgae_cfg["pretrain_elite_ratio"],
    )

    run_saea_to_budget(dgae, args.budget, args.max_generations)
    dgae_final = snapshot(
        dgae,
        "D-GAENAS\nAfter Experiment",
        elite_ratio=dgae_cfg["gae_elite_ratio"],
    )

    set_seed(ablation_seed)
    ablation_cfg = dict(ABLATION_CFG)
    ablation_cfg.update({
        "data_file": args.tfrecord,
        "device": run_device,
        "seed": ablation_seed,
        "eval_budget": args.budget,
    })
    ablation_surrogate = SurrogateModel(
        num_nodes=SM_MAX_NODES,
        node_channels=NUM_NODE_FEATURES,
        edge_channels=NUM_EDGE_FEATURES,
        hidden_channels=ablation_cfg["surr_hidden"],
        latent_channels=ablation_cfg["surr_latent"],
    ).to(device)
    ablation = AblationMutation(nb, ablation_surrogate, ablation_cfg)
    initialise_ablation_from_population(ablation, initial_population)
    run_ablation_to_budget(ablation, args.budget, args.max_generations)
    ablation_final = snapshot(
        ablation,
        "Mutation Ablation\nAfter Experiment",
        elite_ratio=ablation_cfg["eval_elite_ratio"],
    )

    output_path = os.path.join(args.out, f"population_comparison_{args.budget}_evals.png")
    plot_three_by_four([initial, dgae_final, ablation_final], output_path)


if __name__ == "__main__":
    main()
