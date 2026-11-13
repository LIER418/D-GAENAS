
import argparse
import json
import math
import multiprocessing as mp
import os
import random
import sys
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "core"))
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from nasbench201_utils import (
    random_arch, individual_hash,
    tensor_to_ops_list, ops_to_arch_str,
    load_nasbench201, query_nasbench201, set_seed,
)
from surrogate_model import GINESurrogate201
from nas_graph import NASGraph
from mutation_ops import mutate_201
from nas_gae import (
    GINEEncoder,
    EdgeFeatureDecoder,
    UnifiedNASGAE,
    make_noise_fn_201,
)


class _UnifiedGAE201Adapter:

    def __init__(self, model: UnifiedNASGAE):
        self._model = model

    def to(self, device):
        self._model.to(device)
        return self

    def train(self):
        self._model.train()

    def eval(self):
        self._model.eval()

    def parameters(self):
        return self._model.parameters()

    def compute_loss(self, ops_batch: torch.Tensor):
        graph = NASGraph.from_201(ops_batch)
        loss, _ = self._model.compute_loss(graph)
        return loss, {}

    @torch.no_grad()
    def reconstruct(self, ops_pop: torch.Tensor) -> torch.Tensor:
        graph = NASGraph.from_201(ops_pop)
        out   = self._model.reconstruct(graph)
        assert out.edge_feat is not None
        return out.edge_feat


def build_unified_gae_201(cfg: dict) -> _UnifiedGAE201Adapter:
    from nasbench201_utils import N_NODES, N_OPS
    enc   = GINEEncoder(
        edge_feat_dim = N_OPS,
        hidden_dim    = cfg["hidden_dim"],
        latent_dim    = cfg["latent_dim"],
        node_feat_dim = N_NODES,
        dropout       = cfg["dropout"],
    )
    model = UnifiedNASGAE(
        encoder  = enc,
        decoders = {"edge_feat": EdgeFeatureDecoder(cfg["latent_dim"], cfg["dec_hidden"], N_OPS)},
        noise_fn = make_noise_fn_201(cfg["noise_prob"]),
    )
    return _UnifiedGAE201Adapter(model)


CFG = {
    "data_file"     : "data/NATS-tss-v1_0-3ffb9.pickle.pbz2",
    "dataset"       : "cifar10-valid",

    "n_init"               : 50,
    "pretrain_gae_epochs"  : 10,
    "pretrain_surr_epochs" : 50,
    "pretrain_elite_ratio" : 0.1,

    "pop_size"      : 100,
    "n_generations" : 200,

    "gae_elite_ratio"  : 0.01,
    "eval_elite_ratio" : 0.01,

    "surr_online_epochs" : 50,
    "surr_margin"        : 1.0,
    "surr_lr"            : 1e-3,
    "surr_hidden"        : 64,
    "surr_latent"        : 64,

    "gae_online_epochs"  : 10,

    "hidden_dim"  : 4,
    "latent_dim"  : 2,
    "dec_hidden"  : 2,
    "dropout"     : 0.1,
    "noise_prob"  : 0.15,
    "gae_lr"      : 1e-3,

    "device" : "cpu",
    "seed"   : 1738,
    "n_runs" : 10,
    "n_jobs" : 1,

    "run_all_datasets" : True,

    "eval_budget" : 100,

    "mutation_prob": 0.1,
    "generation_method": "dgae",

}


def train_surrogate_ranking(
    surrogate:  nn.Module,
    archive:    list,
    scores:     list,
    optimizer:  optim.Optimizer,
    margin:     float = 0.0,
    n_pairs:    int   = 256,
) -> float:
    if len(archive) < 2:
        return 0.0

    surrogate.train()
    device    = next(surrogate.parameters()).device

    n = len(archive)
    idx_a, idx_b = [], []
    attempts = 0
    max_attempts = n_pairs * 20

    while len(idx_a) < n_pairs and attempts < max_attempts:
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if scores[i] != scores[j]:
            idx_a.append(i)
            idx_b.append(j)
        attempts += 1

    if not idx_a:
        return 0.0

    ops_a = torch.stack([archive[i] for i in idx_a]).to(device)
    ops_b = torch.stack([archive[i] for i in idx_b]).to(device)
    s_a   = torch.tensor([scores[i] for i in idx_a], device=device)
    s_b   = torch.tensor([scores[i] for i in idx_b], device=device)

    pred_a = surrogate(ops_a)
    pred_b = surrogate(ops_b)

    y = (s_a > s_b).float() * 2 - 1

    optimizer.zero_grad()
    criterion = nn.MarginRankingLoss(margin=margin)
    loss = criterion(pred_a, pred_b, y)
    loss.backward()
    optimizer.step()

    return loss.item()


def predict_surrogate(
    surrogate:  nn.Module,
    candidates: list,
) -> list:
    surrogate.eval()
    device = next(surrogate.parameters()).device
    with torch.no_grad():
        ops_batch = torch.stack(candidates).to(device)
        scores    = surrogate(ops_batch).cpu().tolist()
    return scores


def build_surrogate_201(cfg: dict):
    return GINESurrogate201(
        hidden_channels=cfg["surr_hidden"], latent_channels=cfg["surr_latent"]
    )


class SAEAEvolution201:

    def __init__(
        self,
        gae,
        surrogate: GINESurrogate201,
        api,
        cfg:       dict,
    ):
        self.surrogate = surrogate
        self.api       = api
        self.cfg       = cfg
        self.device    = torch.device(cfg["device"])
        self.dataset   = cfg["dataset"]

        self.gae = gae.to(self.device)
        self.surrogate.to(self.device)

        self.gae_opt = optim.Adam(self.gae.parameters(), lr=cfg["gae_lr"])
        self.surr_opt = optim.Adam(surrogate.parameters(), lr=cfg["surr_lr"])

        self.population:    list  = []
        self.train_archive: list  = []
        self.train_scores:  list  = []
        self.eval_cache:    dict  = {}
        self.gen:           int   = 0
        self._best_test_acc: float = 0.0
        self._query_curve:   list  = []
        self._incumbent_curve: list = []
        self._global_best:   tuple | None = None


    def _evaluate(self, ops_tensor: torch.Tensor) -> tuple:
        key = individual_hash(ops_tensor)
        if key in self.eval_cache:
            return self.eval_cache[key]
        val_acc, test_acc = query_nasbench201(self.api, ops_tensor, self.dataset)
        if val_acc is not None:
            self.eval_cache[key] = (val_acc, test_acc)
        return val_acc, test_acc


    def pretrain(self) -> None:
        cfg = self.cfg

        print(f"Sampling and evaluating {cfg['n_init']} initial architectures …")
        pool = []
        while len(pool) < cfg["n_init"]:
            ops = random_arch()
            val_acc, test_acc = self._evaluate(ops)
            if val_acc is None:
                continue
            ind = (ops, val_acc, test_acc)
            pool.append(ind)
            if self._global_best is None or val_acc > self._global_best[1]:
                self._global_best = ind
            t = test_acc if test_acc is not None else 0.0
            if t > self._best_test_acc:
                self._best_test_acc = t
            self._query_curve.append((len(self.eval_cache), self._best_test_acc))
            self._incumbent_curve.append(
                (len(self.eval_cache), self._global_best[1], self._global_best[2])
            )

        self.train_archive = [ind[0] for ind in pool]
        self.train_scores  = [ind[1] for ind in pool]

        elite_ratio = cfg.get("pretrain_elite_ratio", 1.0)
        k_elite     = max(1, int(elite_ratio * len(pool)))
        pool_sorted = sorted(pool, key=lambda t: -t[1])
        gae_pool    = pool_sorted[:k_elite]
        print(f"Pre-training GAE ({cfg['pretrain_gae_epochs']} steps"
              f", top {k_elite}/{len(pool)} individuals) …")
        ops_batch = torch.stack([ind[0] for ind in gae_pool]).to(self.device)
        self.gae.train()
        for _ in range(cfg["pretrain_gae_epochs"]):
            self.gae_opt.zero_grad()
            loss, _ = self.gae.compute_loss(ops_batch)
            loss.backward()
            self.gae_opt.step()
        print(f"  DGAE pre-training done, final loss = {loss.item():.4f}")

        print(f"Pre-training surrogate model ({cfg['pretrain_surr_epochs']} epochs) …")
        for _ in range(cfg["pretrain_surr_epochs"]):
            train_surrogate_ranking(
                self.surrogate, self.train_archive, self.train_scores,
                self.surr_opt, margin=cfg["surr_margin"],
            )

        self.population = sorted(pool, key=lambda t: -t[1])
        print(f"Initial population best val_acc = {self.population[0][1]:.4f}\n")


    def step(self) -> dict:
        cfg    = self.cfg
        k_gae  = max(1, math.ceil(cfg["gae_elite_ratio"]  * len(self.population)))
        k_eval = max(1, math.ceil(cfg["eval_elite_ratio"] * len(self.population)))

        mutation_prob = float(cfg.get("mutation_prob", 0.1))
        if not 0.0 <= mutation_prob <= 1.0:
            raise ValueError("mutation probabilities must be between 0 and 1")
        n_mutated = 0
        dgae_loss = float("nan")
        generation_method = cfg.get("generation_method", "dgae")
        if generation_method == "population_random_mutation":
            generation_prob = float(cfg.get("random_generation_prob", 0.15))
            if not 0.0 <= generation_prob <= 1.0:
                raise ValueError("random_generation_prob must be between 0 and 1")
            candidates = [
                mutate_201(ind[0], generation_prob) for ind in self.population
            ]
            for i, candidate in enumerate(candidates):
                if random.random() < mutation_prob:
                    values = tensor_to_ops_list(candidate)
                    edge = random.randrange(len(values))
                    values[edge] = random.choice(
                        [op for op in range(5) if op != values[edge]]
                    )
                    candidates[i] = torch.nn.functional.one_hot(
                        torch.tensor(values), num_classes=5
                    ).float()
            n_mutated = sum(
                not torch.equal(candidate, parent[0].cpu())
                for candidate, parent in zip(candidates, self.population)
            )
        elif generation_method == "best_probabilistic_mutation":
            parent = self.population[0][0]
            candidates = [mutate_201(parent, mutation_prob) for _ in self.population]
            n_mutated = sum(not torch.equal(c, parent.cpu()) for c in candidates)
        elif generation_method == "dgae":
            ops_list = [ind[0] for ind in self.population[:k_gae]]
            ops_batch = torch.stack(ops_list).to(self.device)
            drop_count = int(cfg.get("dgae_elite_drop_count", 1))
            if not 0 <= drop_count < ops_batch.shape[0]:
                raise ValueError(
                    "dgae_elite_drop_count must be non-negative and smaller "
                    "than the number of DGAE training elites"
                )
            if drop_count:
                dropped = set(random.sample(range(ops_batch.shape[0]), drop_count))
                kept = [i for i in range(ops_batch.shape[0]) if i not in dropped]
                ops_train = ops_batch[kept]
            else:
                ops_train = ops_batch
            self.gae.train()
            dgae_losses = []
            for _ in range(cfg["gae_online_epochs"]):
                self.gae_opt.zero_grad()
                loss, _ = self.gae.compute_loss(ops_train)
                loss.backward()
                self.gae_opt.step()
                dgae_losses.append(float(loss.detach().cpu().item()))
            if dgae_losses:
                dgae_loss = float(np.mean(dgae_losses))
            self.gae.eval()
            ops_pop = torch.stack([ind[0] for ind in self.population]).to(self.device)
            cand_batch = self.gae.reconstruct(ops_pop)
            candidates = [cand_batch[i].cpu() for i in range(cand_batch.shape[0])]
            for i, candidate in enumerate(candidates):
                if random.random() < mutation_prob:
                    values = tensor_to_ops_list(candidate)
                    edge = random.randrange(len(values))
                    values[edge] = random.choice([op for op in range(5) if op != values[edge]])
                    candidates[i] = torch.nn.functional.one_hot(
                        torch.tensor(values), num_classes=5
                    ).float()
                    n_mutated += 1
        else:
            raise ValueError(f"Unknown generation_method: {generation_method}")

        surr_scores = predict_surrogate(self.surrogate, candidates)
        ranked_idx  = sorted(range(len(surr_scores)), key=lambda i: -surr_scores[i])
        top_cands   = [candidates[i] for i in ranked_idx[:k_eval]]

        queries_before = len(self.eval_cache)
        new_evals = []
        for ops_t in top_cands:
            is_new    = individual_hash(ops_t) not in self.eval_cache
            val_acc, test_acc = self._evaluate(ops_t)
            if val_acc is None:
                continue
            ind = (ops_t, val_acc, test_acc)
            new_evals.append(ind)
            if self._global_best is None or val_acc > self._global_best[1]:
                self._global_best = ind
            if is_new:
                self.train_archive.append(ops_t)
                self.train_scores.append(val_acc)
                t = test_acc if test_acc is not None else 0.0
                if t > self._best_test_acc:
                    self._best_test_acc = t
                self._query_curve.append((len(self.eval_cache), self._best_test_acc))
                self._incumbent_curve.append(
                    (len(self.eval_cache), self._global_best[1], self._global_best[2])
                )

        for _ in range(cfg["surr_online_epochs"]):
            train_surrogate_ranking(
                self.surrogate, self.train_archive, self.train_scores,
                self.surr_opt, margin=cfg["surr_margin"],
            )

        combined = self.population + new_evals
        combined.sort(key=lambda t: -t[1])
        self.population = combined[:cfg["pop_size"]]

        n_real_eval = len(self.eval_cache) - queries_before
        self.gen += 1
        fitnesses = [t[1] for t in self.population]
        return {
            "gen":          self.gen,
            "best_fitness": fitnesses[0],
            "mean_fitness": sum(fitnesses) / len(fitnesses),
            "dgae_loss":    dgae_loss,
            "n_mutated":     n_mutated,
            "n_real_eval":  n_real_eval,
            "archive_size": len(self.train_archive),
        }


    def run(self, n_generations: int) -> list:
        budget = self.cfg.get("eval_budget")
        history = []

        for _ in range(n_generations):
            if budget is not None and len(self.eval_cache) >= budget:
                print(f"  [Stop] Real eval reached budget {budget}, terminating.")
                break
            stats = self.step()
            history.append(stats)

            print(
                f"  Gen {stats['gen']:3d}/{n_generations}"
                f" | best={stats['best_fitness']:.4f}"
                f" | mean={stats['mean_fitness']:.4f}"
                + (f" | dgae_loss={stats['dgae_loss']:.6f}"
                   if np.isfinite(stats['dgae_loss']) else " | dgae_loss=N/A")
                + f" | real_eval={stats['n_real_eval']}"
                f" | mutated={stats['n_mutated']}"
                + f" | archive={stats['archive_size']}"
                + (f" | budget_left={budget - len(self.eval_cache)}" if budget is not None else "")
            )
        return history

    @property
    def best(self) -> tuple:
        if self._global_best is not None:
            return self._global_best
        return self.population[0]


def _run_one_201(run: int, api, cfg: dict) -> dict:
    dataset = cfg["dataset"]
    seed = cfg["seed"] + run
    cfg = dict(cfg, seed=seed)
    set_seed(seed)
    print(f"\n[{dataset}] Run {run + 1}/{cfg['n_runs']} (seed={seed})", flush=True)
    gae = build_unified_gae_201(cfg)
    surrogate = build_surrogate_201(cfg)
    saea = SAEAEvolution201(gae=gae, surrogate=surrogate, api=api, cfg=cfg)
    saea.pretrain()
    history = saea.run(n_generations=cfg["n_generations"])
    _, val_acc, test_acc = saea.best
    real_evals = len(saea.eval_cache)
    checkpoint = next((x for x in saea._incumbent_curve if x[0] == 150), None)
    print(f"[{dataset}] Run {run + 1} final val={val_acc:.4f} "
          f"test={test_acc:.4f} real_evals={real_evals}", flush=True)
    return {"run": run + 1, "seed": seed, "val": val_acc, "test": test_acc,
            "real_evals": real_evals,
            "history": history, "query_curve": saea._query_curve,
            "val_at_150": checkpoint[1] if checkpoint else float("nan"),
            "test_at_150": checkpoint[2] if checkpoint else float("nan")}


_WORKER_API201 = None
_WORKER_CFG201 = None


def _init_worker_201(data_file: str, cfg: dict) -> None:
    global _WORKER_API201, _WORKER_CFG201
    _WORKER_CFG201 = cfg
    _WORKER_API201 = load_nasbench201(data_file)


def _run_worker_201(run: int) -> dict:
    return _run_one_201(run, _WORKER_API201, _WORKER_CFG201)


def run_one_dataset(api, cfg: dict) -> dict:
    n_runs = int(cfg.get("n_runs", 1))
    n_jobs = min(int(cfg.get("n_jobs", 1)), n_runs)
    if n_jobs < 1:
        raise ValueError("n_jobs must be a positive integer")
    print(f"\nDataset={cfg['dataset']} | runs={n_runs} | n_jobs={n_jobs}")
    if n_jobs == 1:
        results = [_run_one_201(i, api, cfg) for i in range(n_runs)]
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=n_jobs, mp_context=ctx, initializer=_init_worker_201,
            initargs=(cfg["data_file"], cfg),
        ) as pool:
            results = list(pool.map(_run_worker_201, range(n_runs)))
    results.sort(key=lambda x: x["run"])
    all_val = [x["val"] for x in results]
    all_test = [x["test"] for x in results]
    all_cache = [x["real_evals"] for x in results]
    all_val150 = [x["val_at_150"] for x in results]
    all_test150 = [x["test_at_150"] for x in results]
    histories = [x["history"] for x in results]
    curves = [x["query_curve"] for x in results]
    val_arr, test_arr = np.asarray(all_val), np.asarray(all_test)
    cache_arr = np.asarray(all_cache)
    print(f"[{cfg['dataset']}] val={val_arr.mean():.6f} +/- {val_arr.std(ddof=0):.6f}")
    print(f"[{cfg['dataset']}] test={test_arr.mean():.6f} +/- {test_arr.std(ddof=0):.6f}")
    if np.isfinite(np.asarray(all_val150)).any():
        print(f"[{cfg['dataset']}] @150 val={np.nanmean(all_val150):.6f} "
              f"+/- {np.nanstd(all_val150):.6f}")
        print(f"[{cfg['dataset']}] @150 test={np.nanmean(all_test150):.6f} "
              f"+/- {np.nanstd(all_test150):.6f}")
    return {"dataset": cfg["dataset"], "all_val": all_val, "all_test": all_test,
            "all_cache": all_cache,
            "all_val_at_150": all_val150, "all_test_at_150": all_test150,
            "all_histories": histories, "all_query_curves": curves,
            "val_arr": val_arr, "test_arr": test_arr, "cache_arr": cache_arr}


def plot_evolution_curve_201(
    all_curves: list,
    save_path: str,
    title: str = "NAS-Bench-201",
    xlabel: str = "# Queries",
    ylabel: str = "Accuracy",
    errorbar_every: int = 20,
) -> None:
    if not all_curves:
        return

    max_q  = max(curve[-1][0] for curve in all_curves if curve)
    x_grid = np.arange(1, max_q + 1)

    interp_mat = []
    for curve in all_curves:
        if not curve:
            continue
        xs = np.array([p[0] for p in curve])
        ys = np.array([p[1] for p in curve])
        y_full = np.zeros(len(x_grid))
        j, cur = 0, 0.0
        for i, xv in enumerate(x_grid):
            while j < len(xs) and xs[j] <= xv:
                cur = ys[j]; j += 1
            y_full[i] = cur
        interp_mat.append(y_full)

    mat  = np.array(interp_mat)
    mean = mat.mean(axis=0)
    std  = mat.std(axis=0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_grid, mean, color="#1f77b4", linewidth=1.5, label="D-GAENAS")
    eb_idx = np.arange(errorbar_every - 1, len(x_grid), errorbar_every)
    ax.errorbar(x_grid[eb_idx], mean[eb_idx], yerr=std[eb_idx],
                fmt="none", ecolor="#1f77b4", elinewidth=1.2,
                capsize=3, capthick=1.2)

    y_lo = max(0.0, (mean - std).min() - std.mean())
    y_hi = min(1.0, (mean + std).max() + std.mean())
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlim(left=1)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Evolution curve saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="D-GAENAS on NAS-Bench-201")
    parser.add_argument("--config", help="JSON file whose values override CFG")
    parser.add_argument("--n-jobs", type=int, default=None,
                        help="parallel independent runs (overrides config)")
    args = parser.parse_args()
    if args.config:
        with open(args.config, "r", encoding="utf-8") as handle:
            CFG.update(json.load(handle))
    if args.n_jobs is not None:
        CFG["n_jobs"] = args.n_jobs
    n_jobs = CFG.get("n_jobs", 1)
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or n_jobs < 1:
        raise ValueError("n_jobs must be a positive integer")
    CFG["n_jobs"] = min(n_jobs, int(CFG.get("n_runs", 1)))

    api = load_nasbench201(CFG["data_file"]) if CFG["n_jobs"] == 1 else None

    ALL_DATASETS = ["cifar10-valid", "cifar100", "ImageNet16-120"]
    if CFG.get("run_all_datasets", False):
        datasets_to_run = ALL_DATASETS
    else:
        datasets_to_run = [CFG["dataset"]]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = str(CFG.get("experiment_name", "dgaenas"))
    experiment_name = "".join(
        ch if ch.isalnum() or ch in "-_" else "_" for ch in experiment_name
    ).strip("_") or "dgaenas"
    output_dir = os.path.join(_PROJECT_ROOT, "results", "nb201")
    os.makedirs(output_dir, exist_ok=True)
    xlsx_path = os.path.join(output_dir, f"{experiment_name}_nb201_{ts}.xlsx")

    all_results: list = []

    for ds in datasets_to_run:
        ds_cfg = {**CFG, "dataset": ds}
        result = run_one_dataset(api, ds_cfg)
        all_results.append(result)

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:

        for res in all_results:
            ds        = res["dataset"]
            sheet_pfx = ds[:10]

            history_rows = []
            for run_idx, hist in enumerate(res["all_histories"]):
                for row in hist:
                    history_rows.append({
                        "dataset": ds,
                        "run":     run_idx + 1,
                        "seed":    CFG["seed"] + run_idx,
                        **row,
                    })
            pd.DataFrame(history_rows).to_excel(
                writer, sheet_name=f"{sheet_pfx}_hist", index=False)

            val_arr  = res["val_arr"]
            test_arr = res["test_arr"]
            cache_arr= res["cache_arr"]
            val150_arr = np.asarray(res["all_val_at_150"], dtype=float)
            test150_arr = np.asarray(res["all_test_at_150"], dtype=float)
            val150_ok = val150_arr[np.isfinite(val150_arr)]
            test150_ok = test150_arr[np.isfinite(test150_arr)]
            checkpoint_stats = {
                "mean": (val150_ok.mean(), test150_ok.mean()) if len(val150_ok) else (np.nan, np.nan),
                "std": (val150_ok.std(), test150_ok.std()) if len(val150_ok) else (np.nan, np.nan),
                "best": (val150_ok.max(), test150_ok.max()) if len(val150_ok) else (np.nan, np.nan),
            }
            summary_rows = [
                {"dataset": ds, "run": i + 1, "seed": CFG["seed"] + i,
                 "val_acc": v, "test_acc": t, "real_evals": c,
                 "val_at_150": v150, "test_at_150": t150}
                for i, (v, t, c, v150, t150) in enumerate(zip(
                    res["all_val"], res["all_test"], res["all_cache"],
                    res["all_val_at_150"], res["all_test_at_150"]))
            ]
            for label, arr in [
                ("mean", np.array([val_arr.mean(), test_arr.mean(), cache_arr.mean(),
                                    *checkpoint_stats["mean"]])),
                ("std",  np.array([val_arr.std(), test_arr.std(), cache_arr.std(),
                                    *checkpoint_stats["std"]])),
                ("best", np.array([val_arr.max(), test_arr.max(), cache_arr.max(),
                                    *checkpoint_stats["best"]])),
            ]:
                summary_rows.append({"dataset": ds, "run": label, "seed": "",
                                     "val_acc": arr[0], "test_acc": arr[1],
                                     "real_evals": arr[2],
                                     "val_at_150": arr[3], "test_at_150": arr[4]})
            pd.DataFrame(summary_rows).to_excel(
                writer, sheet_name=f"{sheet_pfx}_summ", index=False)

        pd.DataFrame([{"key": k, "value": str(v)} for k, v in CFG.items()]).to_excel(
            writer, sheet_name="config", index=False)

    print(f"\nResults saved to: {xlsx_path}")

    for res in all_results:
        ds     = res["dataset"]
        curves = res["all_query_curves"]

        max_len = max((len(c) for c in curves), default=0)
        if max_len > 0:
            last_val = lambda c: c[-1][1] if c else 0.0
            padded = [
                c + [(c[-1][0] + k + 1, last_val(c)) for k in range(max_len - len(c))]
                for c in curves
            ]
            npz_path = xlsx_path.replace(".xlsx", f"_{ds.replace('-','_')}_curves.npz")
            np.savez(npz_path, dgaenas_curves=np.array(padded, dtype=float))
            print(f"[{ds}] Curve data saved to: {npz_path}")

        png_path = xlsx_path.replace(".xlsx", f"_{ds.replace('-','_')}_curve.png")
        plot_evolution_curve_201(
            curves, save_path=png_path,
            title=f"NAS-Bench-201 ({ds})",
        )

