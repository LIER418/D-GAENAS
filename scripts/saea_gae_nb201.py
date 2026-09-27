
import argparse
import json
import math
import os
import random
import sys
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "core"))
from datetime import datetime

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
    "dataset"       : "cifar100",

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
    "seed"   : 29,
    "n_runs" : 30,

    "run_all_datasets" : True,

    "eval_budget" : 100,

    "restart_stagnation": 10,

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
        self.surr_opt = optim.Adam(
            surrogate.parameters(), lr=cfg["surr_lr"],
        )

        self.population:    list  = []
        self.train_archive: list  = []
        self.train_scores:  list  = []
        self.eval_cache:    dict  = {}
        self.gen:           int   = 0
        self._best_test_acc: float = 0.0
        self._query_curve:   list  = []
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

        ops_list  = [ind[0] for ind in self.population[:k_gae]]
        ops_batch = torch.stack(ops_list).to(self.device)
        n_train   = ops_batch.shape[0]
        if n_train > 1:
            drop  = random.randrange(n_train)
            keep  = [i for i in range(n_train) if i != drop]
            ops_b = ops_batch[keep]
        else:
            ops_b = ops_batch
        self.gae.train()
        for _ in range(cfg["gae_online_epochs"]):
            self.gae_opt.zero_grad()
            loss, _ = self.gae.compute_loss(ops_b)
            loss.backward()
            self.gae_opt.step()

        self.gae.eval()
        ops_pop    = torch.stack([ind[0] for ind in self.population]).to(self.device)
        cand_batch = self.gae.reconstruct(ops_pop)
        candidates = [cand_batch[i].cpu() for i in range(cand_batch.shape[0])]

        surr_scores = predict_surrogate(self.surrogate, candidates)
        ranked_idx  = sorted(range(len(surr_scores)), key=lambda i: -surr_scores[i])
        top_cands   = [candidates[i] for i in ranked_idx[:k_eval]]

        new_evals = []
        budget = cfg.get("eval_budget")
        for ops_t in top_cands:
            is_new    = individual_hash(ops_t) not in self.eval_cache
            if is_new and budget is not None and len(self.eval_cache) >= budget:
                break
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

        for _ in range(cfg["surr_online_epochs"]):
            train_surrogate_ranking(
                self.surrogate, self.train_archive, self.train_scores,
                self.surr_opt, margin=cfg["surr_margin"],
            )

        combined = self.population + new_evals
        combined.sort(key=lambda t: -t[1])
        self.population = combined[:cfg["pop_size"]]

        self.gen += 1
        fitnesses = [t[1] for t in self.population]
        return {
            "gen":          self.gen,
            "best_fitness": fitnesses[0],
            "mean_fitness": sum(fitnesses) / len(fitnesses),
            "n_real_eval":  len(new_evals),
            "archive_size": len(self.train_archive),
        }


    def _restart(self) -> None:
        cfg = self.cfg
        print(f"  [Restart] Re-sampling population and resetting GAE …")

        model = getattr(self.gae, "_model", self.gae)
        for layer in model.modules():
            if callable(getattr(layer, "reset_parameters", None)):
                layer.reset_parameters()
        self.gae_opt = optim.Adam(self.gae.parameters(), lr=cfg["gae_lr"])

        n = cfg["n_init"]
        pool = []
        while len(pool) < n:
            if cfg.get("eval_budget") is not None and len(self.eval_cache) >= cfg["eval_budget"]:
                break
            ops    = random_arch()
            is_new = individual_hash(ops) not in self.eval_cache
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
            if is_new:
                self.train_archive.append(ops)
                self.train_scores.append(val_acc)

        pool_sorted = sorted(pool, key=lambda t: -t[1])
        k_elite     = max(1, int(cfg.get("pretrain_elite_ratio", 1.0) * len(pool_sorted)))
        ops_batch   = torch.stack([ind[0] for ind in pool_sorted[:k_elite]]).to(self.device)
        self.gae.train()
        for _ in range(cfg["pretrain_gae_epochs"]):
            self.gae_opt.zero_grad()
            loss, _ = self.gae.compute_loss(ops_batch)
            loss.backward()
            self.gae_opt.step()

        self.population = pool_sorted[:cfg["pop_size"]]
        print(f"  [Restart] Done, new population best val_acc={self.population[0][1]:.4f}"
              f"  eval_cache={len(self.eval_cache)}")


    def run(self, n_generations: int) -> list:
        budget         = self.cfg.get("eval_budget")
        stagnation_limit = self.cfg.get("restart_stagnation", None)
        history          = []
        stagnation_count = 0

        for _ in range(n_generations):
            if budget is not None and len(self.eval_cache) >= budget:
                print(f"  [Stop] Real eval reached budget {budget}, terminating.")
                break
            stats = self.step()
            history.append(stats)

            if stagnation_limit is not None:
                if len({ind[1] for ind in self.population}) == 1:
                    stagnation_count += 1
                else:
                    stagnation_count = 0
                budget_available = budget is None or len(self.eval_cache) < budget
                if stagnation_count >= stagnation_limit and budget_available:
                    stagnation_count = 0
                    self._restart()

            print(
                f"  Gen {stats['gen']:3d}/{n_generations}"
                f" | best={stats['best_fitness']:.4f}"
                f" | mean={stats['mean_fitness']:.4f}"
                f" | real_eval={stats['n_real_eval']}"
                f" | archive={stats['archive_size']}"
                + (f" | budget_left={budget - len(self.eval_cache)}" if budget is not None else "")
            )
        return history

    @property
    def best(self) -> tuple:
        if self._global_best is not None:
            return self._global_best
        return self.population[0]


def run_one_dataset(api, cfg: dict) -> dict:
    dataset  = cfg["dataset"]
    n_runs   = cfg.get("n_runs", 1)
    budget   = cfg.get("eval_budget")
    n_elite  = max(1, math.ceil(cfg["eval_elite_ratio"] * cfg["pop_size"]))
    theo_max = cfg["n_init"] + cfg["n_generations"] * n_elite
    budget_str = str(budget) if budget is not None else f"unlimited (theoretical max {theo_max})"

    print(f"\n{'★'*60}")
    print(f"  Dataset: {dataset}  |  Real eval budget: {budget_str}  |  Runs: {n_runs}")
    print(f"{'★'*60}\n")

    all_val:            list = []
    all_test:           list = []
    all_cache:          list = []
    all_histories:      list = []
    all_query_curves:   list = []

    for run in range(n_runs):
        seed = cfg["seed"] + run
        set_seed(seed)
        print(f"{'─'*60}")
        print(f"  [{dataset}] Run {run + 1}/{n_runs}  (seed={seed})")
        print(f"{'─'*60}")

        gae = build_unified_gae_201(cfg)
        surrogate = GINESurrogate201(
            hidden_channels = cfg["surr_hidden"],
            latent_channels = cfg["surr_latent"],
        )

        saea = SAEAEvolution201(
            gae       = gae,
            surrogate = surrogate,
            api       = api,
            cfg       = cfg,
        )

        print("=== Pre-training ===")
        saea.pretrain()
        print(f"=== Evolution ({cfg['n_generations']} generations) ===\n")
        history = saea.run(n_generations=cfg["n_generations"])
        all_histories.append(history)

        best_ops, best_val_r, best_test_r = saea.best
        real_evals = len(saea.eval_cache)

        all_val.append(best_val_r)
        all_test.append(best_test_r)
        all_cache.append(real_evals)
        all_query_curves.append(saea._query_curve)

        ops_idx = tensor_to_ops_list(best_ops)
        print(f"  Run {run+1} result — val={best_val_r:.4f}  test={best_test_r:.4f}"
              f"  real_evals={real_evals}")
        print(f"  Best arch: {ops_to_arch_str(ops_idx)}")

    val_arr   = np.array(all_val)
    test_arr  = np.array(all_test)
    cache_arr = np.array(all_cache)
    print(f"\n{'═'*60}")
    print(f"  Summary ({n_runs} runs, dataset={dataset})")
    print(f"{'═'*60}")
    for i, (v, t, c) in enumerate(zip(all_val, all_test, all_cache)):
        print(f"  Run {i+1:2d}: val={v:.4f}  test={t:.4f}  real_evals={c}")
    print(f"{'─'*60}")
    print(f"  val_acc   mean={val_arr.mean():.4f}  std={val_arr.std():.4f}"
          f"  best={val_arr.max():.4f}")
    print(f"  test_acc  mean={test_arr.mean():.4f}  std={test_arr.std():.4f}"
          f"  best={test_arr.max():.4f}")
    print(f"  real_evals  mean={cache_arr.mean():.1f}  std={cache_arr.std():.1f}"
          f"  total={cache_arr.sum()}")
    print(f"{'═'*60}")

    return {
        "dataset":           dataset,
        "all_val":           all_val,
        "all_test":          all_test,
        "all_cache":         all_cache,
        "all_histories":     all_histories,
        "all_query_curves":  all_query_curves,
        "val_arr":           val_arr,
        "test_arr":          test_arr,
        "cache_arr":         cache_arr,
    }


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
    args = parser.parse_args()
    if args.config:
        with open(args.config, "r", encoding="utf-8") as handle:
            CFG.update(json.load(handle))

    api = load_nasbench201(CFG["data_file"])

    ALL_DATASETS = ["cifar10-valid", "cifar100", "ImageNet16-120"]
    if CFG.get("run_all_datasets", False):
        datasets_to_run = ALL_DATASETS
    else:
        datasets_to_run = [CFG["dataset"]]

    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    xlsx_path = f"saea_gae_201_results_{ts}.xlsx"

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
            summary_rows = [
                {"dataset": ds, "run": i + 1, "seed": CFG["seed"] + i,
                 "val_acc": v, "test_acc": t, "real_evals": c}
                for i, (v, t, c) in enumerate(
                    zip(res["all_val"], res["all_test"], res["all_cache"]))
            ]
            for label, arr in [
                ("mean", np.array([val_arr.mean(),  test_arr.mean(),  cache_arr.mean()])),
                ("std",  np.array([val_arr.std(),   test_arr.std(),   cache_arr.std()])),
                ("best", np.array([val_arr.max(),   test_arr.max(),   cache_arr.max()])),
            ]:
                summary_rows.append({"dataset": ds, "run": label, "seed": "",
                                     "val_acc": arr[0], "test_acc": arr[1],
                                     "real_evals": arr[2]})
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
