
import argparse
import json
import multiprocessing as mp
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import sys, types, random, math
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "core"))
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import torch.optim as optim

for _mod in ("nasbench.lib.evaluate", "nasbench.lib.model_builder",
             "nasbench.lib.training_time"):
    sys.modules.setdefault(_mod, types.ModuleType(_mod))

import tensorflow as tf
if not hasattr(tf, "python_io"):
    setattr(tf, "python_io", tf.compat.v1.python_io)
tf.get_logger().setLevel("ERROR")

from nasbench import api as nb_api

from nas_gae import (
    UnifiedNASGAE, GINEncoder, NodeFeatureDecoder, EdgeExistDecoder,
    make_noise_fn_101, VectorDAE101,
)
from nas_graph import NASGraph
from repair_ops import repair_fn_101
from mutation_ops import mutate_101
from ea import N_OPS, repair, is_valid, random_individual, individual_hash, EvalCache
from surrogate_model import (
    SurrogateModel, MAX_NODES as SM_MAX_NODES,
    NUM_NODE_FEATURES, NUM_EDGE_FEATURES,
    nasbench_arch_to_graph,
    train_surrogate_ranking, predict_surrogate,
)


CFG = {
    "data_file"     : "data/nasbench_only108.tfrecord",
    "target_epochs" : 108,
    "n_runs"        : 10,
    "n_jobs"        : 1,

    "n_init"               : 100,
    "pretrain_gae_epochs"  : 10,
    "pretrain_surr_epochs" : 100,
    "pretrain_elite_ratio" : 0.1,

    "pop_size"      : 100,
    "n_generations" : 500,

    "gae_elite_ratio"  : 0.1,
    "eval_elite_ratio" : 0.1,

    "surr_online_epochs" : 50,
    "surr_margin"        : 1.0,

    "gae_online_epochs" : 10,

    "hidden_dim"      : 128,
    "latent_dim"      : 64,
    "node_dec_hidden" : 128,
    "edge_dec_hidden" : 128,
    "dropout"         : 0.1,
    "node_noise"      : 0.15,
    "edge_noise"      : 0.15,
    "gae_lr"          : 1e-3,

    "surr_hidden"  : 64,
    "surr_latent"  : 64,
    "surr_lr"      : 1e-3,
    "device"      : "cpu",
    "seed"        : 17,

    "eval_budget" : 1000,

    "mutation_prob": 0.1,
    "generation_method": "dgae",
    "autoencoder_type": "graph_dgae",
}

OP_STRINGS = ["input", "conv1x1-bn-relu", "conv3x3-bn-relu", "maxpool3x3", "output"]


def set_seed(s: int) -> None:
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def load_nasbench(path: str) -> nb_api.NASBench:
    print(f"Loading {path} …")
    nb = nb_api.NASBench(path)
    print(f"Loaded — {len(nb.computed_statistics)} unique architectures.\n")
    return nb


def query_avg(nb: nb_api.NASBench, spec: nb_api.ModelSpec, epochs: int = 108) -> dict:
    _, computed = nb.get_metrics_from_spec(spec)
    runs = computed[epochs]
    return {
        "validation_accuracy": float(np.mean([r["final_validation_accuracy"] for r in runs])),
        "test_accuracy":       float(np.mean([r["final_test_accuracy"]       for r in runs])),
    }


def eval_arch(nb: nb_api.NASBench, x: torch.Tensor, adj: torch.Tensor):
    ops = [OP_STRINGS[i] for i in x.argmax(dim=-1).tolist()]
    mat = adj.cpu().numpy().astype(int)
    try:
        spec = nb_api.ModelSpec(matrix=mat, ops=ops)
        if not nb.is_valid(spec):
            return None, None
        result = query_avg(nb, spec)
        return result["validation_accuracy"], result["test_accuracy"]
    except Exception:
        return None, None


def tensor_to_pyg(x: torch.Tensor, adj: torch.Tensor, accuracy: float = 0.0):
    ops = [OP_STRINGS[i] for i in x.argmax(dim=-1).tolist()]
    mat = adj.cpu().numpy().astype(int)
    return nasbench_arch_to_graph(ops, mat, accuracy)


def build_surrogate_101(cfg: dict, device):
    return SurrogateModel(
        SM_MAX_NODES, NUM_NODE_FEATURES, NUM_EDGE_FEATURES,
        cfg["surr_hidden"], cfg["surr_latent"],
    ).to(device)


def build_autoencoder_101(cfg: dict, device):
    """Build the graph DGAE or its flattened-vector DAE ablation."""
    kind = str(cfg.get("autoencoder_type", "graph_dgae")).lower()
    if kind == "graph_dgae":
        return UnifiedNASGAE(
            encoder=GINEncoder(
                N_OPS, cfg["hidden_dim"], cfg["latent_dim"], cfg["dropout"]
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
    if kind == "vector_dae":
        return VectorDAE101(
            hidden_dim=cfg["hidden_dim"],
            latent_dim=cfg["latent_dim"],
            node_decoder_hidden=cfg["node_dec_hidden"],
            edge_decoder_hidden=cfg["edge_dec_hidden"],
            dropout=cfg["dropout"],
            node_noise=cfg["node_noise"],
            edge_noise=cfg["edge_noise"],
            repair_fn=repair_fn_101,
        ).to(device)
    raise ValueError(
        "autoencoder_type must be one of: graph_dgae, vector_dae"
    )


class SAEAEvolution:

    def __init__(self, nb, gae: UnifiedNASGAE, surrogate: SurrogateModel, cfg: dict):
        self.nb        = nb
        self.surrogate = surrogate
        self.cfg       = cfg
        self.device    = torch.device(cfg["device"])

        self.gae = gae.to(self.device)
        self.gae_opt = optim.Adam(self.gae.parameters(), lr=cfg["gae_lr"])

        self.surr_opt = optim.Adam(surrogate.parameters(), lr=cfg["surr_lr"])

        self.population: list = []
        self.train_archive: list = []
        self.train_costs:   list = []

        self.cache      = EvalCache(max_size=20_000)
        self.generation = 0
        self._best_test_acc: float = 0.0
        self._query_curve:   list  = []
        self._incumbent_curve: list = []
        self._global_best:   list | None = None


    def pretrain(self) -> None:
        cfg = self.cfg
        n   = cfg["n_init"]
        print(f"Pre-training: sampling {n} valid architectures for real evaluation …")

        pool = []
        while len(pool) < n:
            # Common initialization for all NB101 experiments: uniformly
            # sample a 6--9-edge graph and apply the standard validity repair.
            x, adj = random_individual(
                device=self.device, min_edges=6, max_edges=9
            )
            key    = individual_hash(x, adj)
            if self.cache.get(key) is not None:
                continue
            val_acc, test_acc = eval_arch(self.nb, x, adj)
            if val_acc is None:
                continue
            self.cache.put(key, val_acc)
            t = test_acc if test_acc is not None else 0.0
            ind = [x, adj, val_acc, t]
            pool.append(ind)
            if self._global_best is None or val_acc > self._global_best[2]:
                self._global_best = ind
            if t > self._best_test_acc:
                self._best_test_acc = t
            self._query_curve.append((len(pool), self._best_test_acc))
            self._incumbent_curve.append(
                (len(pool), self._global_best[2], self._global_best[3])
            )

        for x, adj, val_acc, _ in pool:
            self.train_archive.append(tensor_to_pyg(x, adj, accuracy=val_acc))
            self.train_costs.append(val_acc)

        elite_ratio = cfg.get("pretrain_elite_ratio", 1.0)
        k_elite     = max(1, int(elite_ratio * len(pool)))
        pool_sorted = sorted(pool, key=lambda ind: -ind[2])
        gae_pool    = pool_sorted[:k_elite]
        print(f"Pre-training GAE ({cfg['pretrain_gae_epochs']} steps"
              f", top {k_elite}/{len(pool)} individuals) …")
        x_batch   = torch.stack([ind[0] for ind in gae_pool]).to(self.device)
        adj_batch = torch.stack([ind[1] for ind in gae_pool]).to(self.device)
        self.gae.train()
        for _ in range(cfg["pretrain_gae_epochs"]):
            self.gae_opt.zero_grad()
            loss, _ = self.gae.compute_loss(NASGraph.from_101(x_batch, adj_batch))
            loss.backward()
            self.gae_opt.step()
        print(f"  DGAE pre-training done, final loss={loss.item():.4f}")

        print(f"Pre-training surrogate model ({cfg['pretrain_surr_epochs']} epochs) …")
        for _ in range(cfg["pretrain_surr_epochs"]):
            train_surrogate_ranking(
                self.surrogate, self.train_archive, self.train_costs,
                self.surr_opt, margin=cfg["surr_margin"],
            )

        self.population = sorted(pool, key=lambda ind: -ind[2])
        best = self.population[0][2]
        print(f"Initial population best val_acc = {best:.4f}\n")


    def step(self) -> dict:
        cfg    = self.cfg
        k_gae  = max(1, math.ceil(cfg["gae_elite_ratio"]  * len(self.population)))
        k_eval = max(1, math.ceil(cfg["eval_elite_ratio"] * len(self.population)))

        candidates = []
        n_repaired = 0
        n_mutated = 0
        generation_method = cfg.get("generation_method", "dgae")
        if generation_method == "population_random_mutation":
            prob = float(cfg.get("random_generation_prob", 0.15))
            if not 0.0 <= prob <= 1.0:
                raise ValueError("random_generation_prob must be between 0 and 1")
            for parent_x, parent_adj, *_ in self.population:
                x_mut, adj_mut = mutate_101(parent_x, parent_adj, prob, prob)
                if random.random() < float(cfg.get("mutation_prob", 0.1)):
                    if random.random() < 0.5:
                        node = random.randrange(1, x_mut.shape[0] - 1)
                        current = int(x_mut[node].argmax().item())
                        op = random.choice([v for v in (1, 2, 3) if v != current])
                        x_mut[node].zero_(); x_mut[node, op] = 1.0
                    else:
                        src, dst = random.choice([
                            (i, j) for i in range(adj_mut.shape[0])
                            for j in range(i + 1, adj_mut.shape[1])
                        ])
                        adj_mut[src, dst] = 1.0 - adj_mut[src, dst]
                    x_mut, adj_mut = repair(x_mut, adj_mut)
                candidates.append((x_mut.to(self.device), adj_mut.to(self.device)))
                n_mutated += int(
                    not torch.equal(x_mut.cpu(), parent_x.cpu())
                    or not torch.equal(adj_mut.cpu(), parent_adj.cpu())
                )
        elif generation_method == "best_probabilistic_mutation":
            parent_x, parent_adj = self.population[0][0], self.population[0][1]
            prob = float(cfg.get("mutation_prob", 0.15))
            if not 0.0 <= prob <= 1.0:
                raise ValueError("mutation_prob must be between 0 and 1")
            for _ in range(len(self.population)):
                x_mut, adj_mut = mutate_101(parent_x, parent_adj, prob, prob)
                candidates.append((x_mut.to(self.device), adj_mut.to(self.device)))
                n_mutated += int(
                    not torch.equal(x_mut.cpu(), parent_x.cpu())
                    or not torch.equal(adj_mut.cpu(), parent_adj.cpu())
                )
        elif generation_method == "dgae":
            train_list = [(ind[0], ind[1]) for ind in self.population[:k_gae]]
            x_batch = torch.stack([t[0] for t in train_list]).to(self.device)
            adj_batch = torch.stack([t[1] for t in train_list]).to(self.device)
            drop_count = int(cfg.get("dgae_elite_drop_count", 1))
            if not 0 <= drop_count < x_batch.shape[0]:
                raise ValueError(
                    "dgae_elite_drop_count must be non-negative and smaller "
                    "than the number of DGAE training elites"
                )
            if drop_count:
                dropped = set(random.sample(range(x_batch.shape[0]), drop_count))
                kept = [i for i in range(x_batch.shape[0]) if i not in dropped]
                x_train, adj_train = x_batch[kept], adj_batch[kept]
            else:
                x_train, adj_train = x_batch, adj_batch
            self.gae.train()
            for _ in range(cfg["gae_online_epochs"]):
                self.gae_opt.zero_grad()
                loss, _ = self.gae.compute_loss(
                    NASGraph.from_101(x_train, adj_train)
                )
                loss.backward()
                self.gae_opt.step()
            self.gae.eval()
            x_pop = torch.stack([ind[0] for ind in self.population]).to(self.device)
            adj_pop = torch.stack([ind[1] for ind in self.population]).to(self.device)
            with torch.no_grad():
                rec_graph = self.gae.reconstruct(NASGraph.from_101(x_pop, adj_pop))
            assert rec_graph.node_feat is not None and rec_graph.adj is not None
            for b in range(x_pop.shape[0]):
                x_rec, adj_rec = rec_graph.node_feat[b], rec_graph.adj[b]
                if not is_valid(x_rec, adj_rec):
                    x_rec, adj_rec = repair(x_rec, adj_rec)
                    n_repaired += 1
                if random.random() < float(cfg.get("mutation_prob", 0.1)):
                    n_mutated += 1
                    if random.random() < 0.5:
                        node = random.randrange(1, x_rec.shape[0] - 1)
                        current = int(x_rec[node].argmax().item())
                        op = random.choice([v for v in (1, 2, 3) if v != current])
                        x_rec[node].zero_(); x_rec[node, op] = 1.0
                    else:
                        src, dst = random.choice([
                            (i, j) for i in range(adj_rec.shape[0])
                            for j in range(i + 1, adj_rec.shape[1])
                        ])
                        adj_rec[src, dst] = 1.0 - adj_rec[src, dst]
                    x_rec, adj_rec = repair(x_rec, adj_rec)
                candidates.append((x_rec.to(self.device), adj_rec.to(self.device)))
        else:
            raise ValueError(
                "Unknown generation_method: "
                f"{generation_method}. Expected one of dgae, "
                "population_random_mutation or "
                "best_probabilistic_mutation."
            )

        pyg_list    = [tensor_to_pyg(x, adj) for x, adj in candidates]
        surr_scores = predict_surrogate(self.surrogate, pyg_list)
        ranked      = sorted(range(len(surr_scores)), key=lambda i: -surr_scores[i])
        top_indices = ranked[:k_eval]

        new_inds   = []
        n_real_eval = 0
        for i in top_indices:
            x, adj = candidates[i]
            key    = individual_hash(x, adj)
            cached = self.cache.get(key)
            if cached is not None:
                ind = [x, adj, cached, 0.0]
                new_inds.append(ind)
                if self._global_best is None or cached > self._global_best[2]:
                    self._global_best = ind
                continue

            budget = cfg.get("eval_budget")
            if budget is not None and len(self.train_archive) >= int(budget):
                break

            val_acc, test_acc = eval_arch(self.nb, x, adj)
            if val_acc is None:
                val_acc, test_acc = 0.0, 0.0
            self.cache.put(key, val_acc)
            n_real_eval += 1

            self.train_archive.append(tensor_to_pyg(x, adj, accuracy=val_acc))
            self.train_costs.append(val_acc)
            ind = [x, adj, val_acc, test_acc]
            new_inds.append(ind)
            if self._global_best is None or val_acc > self._global_best[2]:
                self._global_best = ind
            if (test_acc or 0.0) > self._best_test_acc:
                self._best_test_acc = float(test_acc or 0.0)
            self._query_curve.append((len(self.train_archive), self._best_test_acc))
            self._incumbent_curve.append(
                (len(self.train_archive), self._global_best[2], self._global_best[3])
            )

        for _ in range(cfg["surr_online_epochs"]):
            train_surrogate_ranking(
                self.surrogate, self.train_archive, self.train_costs,
                self.surr_opt, margin=cfg["surr_margin"],
            )

        population_update = cfg.get("population_update", "mu_plus_lambda")
        if population_update == "offspring_only":
            # Diagnostic (mu, lambda)-style update: discard all parents and
            # retain the complete generated population.  GINE scores define
            # the offspring ordering; ground-truth evaluations still update
            # only the archive and the surrogate training data above.
            self.population = sorted(
                [
                    [x, adj, float(surr_scores[i]), 0.0]
                    for i, (x, adj) in enumerate(candidates)
                ],
                key=lambda ind: -ind[2],
            )[:cfg["pop_size"]]
        elif population_update == "mu_plus_lambda":
            combined = self.population + new_inds
            combined.sort(key=lambda ind: -ind[2])
            self.population = combined[:cfg["pop_size"]]
        else:
            raise ValueError(
                "population_update must be one of: mu_plus_lambda, offspring_only"
            )

        self.generation += 1
        fitnesses = [ind[2] for ind in self.population]
        return {
            "generation"   : self.generation,
            "best_fitness" : self.population[0][2],
            "mean_fitness" : float(np.mean(fitnesses)),
            "n_repaired"   : n_repaired,
            "n_mutated"    : n_mutated,
            "n_real_eval"  : n_real_eval,
            "archive_size" : len(self.train_archive),
        }


    def run(self, n_generations: int) -> list:
        if not self.population:
            raise RuntimeError("Call pretrain() first to initialize the population.")
        budget = self.cfg.get("eval_budget")
        history = []

        for _ in range(n_generations):
            if budget is not None and len(self.train_archive) >= budget:
                print(f"  [Stop] Real eval reached budget {budget}, terminating.")
                break
            stats = self.step()
            history.append(stats)

            print(
                f"  Gen {stats['generation']:3d}/{n_generations}"
                f" | best={stats['best_fitness']:.4f}"
                f" | mean={stats['mean_fitness']:.4f}"
                f" | repaired={stats['n_repaired']}"
                f" | real_eval={stats['n_real_eval']}"
                + f" | archive={stats['archive_size']}"
                + (f" | budget_left={budget - len(self.train_archive)}" if budget is not None else "")
            )
        return history

    @property
    def best(self):
        if self._global_best is not None:
            return self._global_best
        return self.population[0]


def plot_evolution_curve(
    all_curves: list,
    save_path: str,
    title: str = "NAS-Bench-101",
    xlabel: str = "# Queries",
    ylabel: str = "Accuracy",
    errorbar_every: int = 50,
) -> None:
    if not all_curves:
        return

    max_q = max(curve[-1][0] for curve in all_curves if curve)
    x_grid = np.arange(1, max_q + 1)

    interp_mat = []
    for curve in all_curves:
        if not curve:
            continue
        xs = np.array([p[0] for p in curve])
        ys = np.array([p[1] for p in curve])
        y_full = np.zeros(len(x_grid))
        j = 0
        cur_best = 0.0
        for i, xv in enumerate(x_grid):
            while j < len(xs) and xs[j] <= xv:
                cur_best = ys[j]
                j += 1
            y_full[i] = cur_best
        interp_mat.append(y_full)

    mat  = np.array(interp_mat)
    mean = mat.mean(axis=0)
    std  = mat.std(axis=0)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(x_grid, mean, color="#1f77b4", linewidth=1.5, label="Mean")

    eb_idx = np.arange(errorbar_every - 1, len(x_grid), errorbar_every)
    ax.errorbar(
        x_grid[eb_idx], mean[eb_idx], yerr=std[eb_idx],
        fmt="none", ecolor="#1f77b4", elinewidth=1.2,
        capsize=3, capthick=1.2,
    )

    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlim(0, 1000)
    ax.set_ylim(0.930, 0.944)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Evolution curve saved to: {save_path}")


def _run_one_parallel(run: int, nb, cfg: dict) -> dict:
    seed = cfg["seed"] + run
    cfg = dict(cfg, seed=seed)
    set_seed(seed)
    device = torch.device(cfg["device"])
    print(f"\nRun {run + 1}/{cfg['n_runs']} (seed={seed})", flush=True)
    gae = build_autoencoder_101(cfg, device)
    surrogate = build_surrogate_101(cfg, device)
    saea = SAEAEvolution(nb, gae, surrogate, cfg)
    saea.pretrain()
    history = saea.run(n_generations=cfg["n_generations"])
    best = saea.best
    ops = [OP_STRINGS[i] for i in best[0].argmax(dim=-1).tolist()]
    try:
        result = query_avg(nb, nb_api.ModelSpec(
            matrix=best[1].cpu().numpy().astype(int), ops=ops
        ))
        val_acc, test_acc = result["validation_accuracy"], result["test_accuracy"]
    except Exception:
        val_acc, test_acc = best[2], float("nan")
    real_evals = len(saea.train_archive)
    checkpoint = next((x for x in saea._incumbent_curve if x[0] == 150), None)
    print(f"Run {run + 1} final val={val_acc:.4f} test={test_acc:.4f} "
          f"real_evals={real_evals}", flush=True)
    return {"run": run + 1, "seed": seed, "val_acc": val_acc,
            "test_acc": test_acc, "real_evals": real_evals,
            "history": history,
            "query_curve": saea._query_curve,
            "validation_curve": [(q, v) for q, v, _ in saea._incumbent_curve],
            "val_at_150": checkpoint[1] if checkpoint else float("nan"),
            "test_at_150": checkpoint[2] if checkpoint else float("nan")}


_WORKER_NB = None
_WORKER_CFG = None


def _init_worker(data_file: str, cfg: dict) -> None:
    global _WORKER_NB, _WORKER_CFG
    _WORKER_CFG = cfg
    _WORKER_NB = load_nasbench(data_file)


def _run_worker(run: int) -> dict:
    return _run_one_parallel(run, _WORKER_NB, _WORKER_CFG)


def main() -> None:
    parser = argparse.ArgumentParser(description="D-GAENAS on NAS-Bench-101")
    parser.add_argument("--config", help="JSON file whose values override CFG")
    parser.add_argument("--n-jobs", type=int, default=None,
                        help="parallel independent runs (overrides config)")
    args = parser.parse_args()
    if args.config:
        with open(args.config, encoding="utf-8") as handle:
            CFG.update(json.load(handle))
    if args.n_jobs is not None:
        CFG["n_jobs"] = args.n_jobs
    n_runs = int(CFG.get("n_runs", 1))
    n_jobs = CFG.get("n_jobs", 1)
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or n_jobs < 1:
        raise ValueError("n_jobs must be a positive integer")
    n_jobs = min(n_jobs, n_runs)
    CFG["n_jobs"] = n_jobs
    print(f"Device: {CFG['device']} | runs={n_runs} | n_jobs={n_jobs}")
    if n_jobs == 1:
        nb = load_nasbench(CFG["data_file"])
        results = [_run_one_parallel(i, nb, CFG) for i in range(n_runs)]
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=n_jobs, mp_context=ctx, initializer=_init_worker,
            initargs=(CFG["data_file"], CFG),
        ) as pool:
            results = list(pool.map(_run_worker, range(n_runs)))
    results.sort(key=lambda x: x["run"])
    histories, curves, validation_curves, summary = [], [], [], []
    for result in results:
        histories.extend({"run": result["run"], "seed": result["seed"], **row}
                         for row in result.pop("history"))
        curves.append(result.pop("query_curve"))
        validation_curves.append(result.pop("validation_curve"))
        summary.append(result)
    val = np.asarray([x["val_acc"] for x in summary], dtype=float)
    test = np.asarray([x["test_acc"] for x in summary], dtype=float)
    queries = np.asarray([x["real_evals"] for x in summary], dtype=float)
    val150 = np.asarray([x["val_at_150"] for x in summary], dtype=float)
    test150 = np.asarray([x["test_at_150"] for x in summary], dtype=float)
    val150_ok = val150[np.isfinite(val150)]
    test150_ok = test150[np.isfinite(test150)]
    val150_mean = val150_ok.mean() if len(val150_ok) else np.nan
    val150_std = val150_ok.std(ddof=0) if len(val150_ok) else np.nan
    test150_mean = test150_ok.mean() if len(test150_ok) else np.nan
    test150_std = test150_ok.std(ddof=0) if len(test150_ok) else np.nan
    print(f"val={val.mean():.6f} +/- {val.std(ddof=0):.6f}")
    print(f"test={np.nanmean(test):.6f} +/- {np.nanstd(test):.6f}")
    if np.isfinite(val150).any():
        print(f"@150 val={val150_mean:.6f} +/- {val150_std:.6f}")
        print(f"@150 test={test150_mean:.6f} +/- {test150_std:.6f}")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = str(CFG.get("experiment_name", "dgaenas"))
    experiment_name = "".join(
        ch if ch.isalnum() or ch in "-_" else "_" for ch in experiment_name
    ).strip("_") or "dgaenas"
    stem = os.path.join(
        _PROJECT_ROOT, "results", "nb101", f"{experiment_name}_nb101_{ts}"
    )
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    stat_rows = summary + [
        {"run": "mean", "val_acc": val.mean(), "test_acc": np.nanmean(test),
         "real_evals": queries.mean(), "val_at_150": val150_mean,
         "test_at_150": test150_mean},
        {"run": "std", "val_acc": val.std(ddof=0), "test_acc": np.nanstd(test),
         "real_evals": queries.std(ddof=0), "val_at_150": val150_std,
         "test_at_150": test150_std},
    ]
    with pd.ExcelWriter(stem + ".xlsx", engine="openpyxl") as writer:
        pd.DataFrame(stat_rows).to_excel(writer, "summary", index=False)
        pd.DataFrame(histories).to_excel(writer, "history", index=False)
        pd.DataFrame([{"key": k, "value": str(v)} for k, v in CFG.items()]).to_excel(
            writer, "config", index=False)
    np.savez(
        stem + "_curves.npz",
        saea_curves=np.array(curves, dtype=object),
        validation_curves=np.array(validation_curves, dtype=object),
    )
    print(f"Results saved under {stem}.*")


if __name__ == "__main__":
    main()
