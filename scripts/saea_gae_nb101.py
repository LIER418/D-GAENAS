
import argparse
import json
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import sys, types, random, math
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "core"))
from datetime import datetime
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
    make_noise_fn_101,
)
from nas_graph import NASGraph
from repair_ops import repair_fn_101
from ea import (N_OPS, repair, is_valid, random_individual,
                individual_hash, EvalCache)
from surrogate_model import (
    SurrogateModel, MAX_NODES as SM_MAX_NODES,
    NUM_NODE_FEATURES, NUM_EDGE_FEATURES,
    nasbench_arch_to_graph,
    train_surrogate_ranking, predict_surrogate,
)


CFG = {
    "data_file"     : "data/nasbench_only108.tfrecord",
    "target_epochs" : 108,
    "n_runs"        : 30,

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
    "seed"        : 1749,

    "eval_budget" : 1000,

    "restart_stagnation": 10,
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


class SAEAEvolution:

    def __init__(self, nb, gae: UnifiedNASGAE, surrogate: SurrogateModel, cfg: dict):
        self.nb        = nb
        self.surrogate = surrogate
        self.cfg       = cfg
        self.device    = torch.device(cfg["device"])

        self.gae = gae.to(self.device)
        self.gae_opt = optim.Adam(self.gae.parameters(), lr=cfg["gae_lr"])

        self.surr_opt = optim.Adam(
            surrogate.parameters(), lr=cfg["surr_lr"]
        )

        self.population: list = []
        self.train_archive: list = []
        self.train_costs:   list = []

        self.cache      = EvalCache(max_size=20_000)
        self.generation = 0
        self._best_test_acc: float = 0.0
        self._query_curve:   list  = []
        self._global_best:   list | None = None


    def pretrain(self) -> None:
        cfg = self.cfg
        n   = cfg["n_init"]
        print(f"Pre-training: sampling {n} valid architectures for real evaluation …")

        pool = []
        while len(pool) < n:
            x, adj = random_individual(device=self.device)
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

        train_list = [(ind[0], ind[1]) for ind in self.population[:k_gae]]

        x_batch   = torch.stack([t[0] for t in train_list]).to(self.device)
        adj_batch = torch.stack([t[1] for t in train_list]).to(self.device)
        n_train   = x_batch.shape[0]
        if n_train > 1:
            drop  = random.randrange(n_train)
            keep  = [i for i in range(n_train) if i != drop]
            xb    = x_batch[keep]
            ab    = adj_batch[keep]
        else:
            xb, ab = x_batch, adj_batch
        self.gae.train()
        for _ in range(cfg["gae_online_epochs"]):
            self.gae_opt.zero_grad()
            loss, _ = self.gae.compute_loss(NASGraph.from_101(xb, ab))
            loss.backward()
            self.gae_opt.step()

        self.gae.eval()
        x_pop   = torch.stack([ind[0] for ind in self.population]).to(self.device)
        adj_pop = torch.stack([ind[1] for ind in self.population]).to(self.device)
        with torch.no_grad():
            rec_graph = self.gae.reconstruct(NASGraph.from_101(x_pop, adj_pop))

        assert rec_graph.node_feat is not None and rec_graph.adj is not None
        B = x_pop.shape[0]
        candidates = []
        n_repaired = 0
        for b in range(B):
            x_rec   = rec_graph.node_feat[b]
            adj_rec = rec_graph.adj[b]
            if not is_valid(x_rec, adj_rec):
                x_rec, adj_rec = repair(x_rec, adj_rec)
                n_repaired += 1
            candidates.append((x_rec.to(self.device), adj_rec.to(self.device)))

        pyg_list    = [tensor_to_pyg(x, adj) for x, adj in candidates]
        surr_scores = predict_surrogate(self.surrogate, pyg_list)
        ranked      = sorted(range(len(surr_scores)), key=lambda i: -surr_scores[i])
        top_indices = ranked[:k_eval]

        new_inds   = []
        n_real_eval = 0
        budget = cfg.get("eval_budget")
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

            if budget is not None and len(self.train_archive) >= budget:
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

        for _ in range(cfg["surr_online_epochs"]):
            train_surrogate_ranking(
                self.surrogate, self.train_archive, self.train_costs,
                self.surr_opt, margin=cfg["surr_margin"],
            )

        combined = self.population + new_inds
        combined.sort(key=lambda ind: -ind[2])
        self.population = combined[:cfg["pop_size"]]

        self.generation += 1
        fitnesses = [ind[2] for ind in self.population]
        return {
            "generation"   : self.generation,
            "best_fitness" : self.population[0][2],
            "mean_fitness" : float(np.mean(fitnesses)),
            "n_repaired"   : n_repaired,
            "n_real_eval"  : n_real_eval,
            "archive_size" : len(self.train_archive),
        }


    def _restart(self) -> None:
        cfg = self.cfg
        print(f"  [Restart] Re-sampling population and resetting GAE …")

        for layer in self.gae.modules():
            if callable(getattr(layer, "reset_parameters", None)):
                layer.reset_parameters()
        self.gae_opt = optim.Adam(self.gae.parameters(), lr=cfg["gae_lr"])

        n = cfg["n_init"]
        pool = []
        while len(pool) < n:
            if cfg.get("eval_budget") is not None and len(self.train_archive) >= cfg["eval_budget"]:
                break
            x, adj = random_individual(device=self.device)
            key    = individual_hash(x, adj)
            cached = self.cache.get(key)
            if cached is not None:
                pool.append([x, adj, cached, 0.0])
                continue
            val_acc, test_acc = eval_arch(self.nb, x, adj)
            if val_acc is None:
                continue
            self.cache.put(key, val_acc)
            t   = test_acc if test_acc is not None else 0.0
            ind = [x, adj, val_acc, t]
            pool.append(ind)
            if self._global_best is None or val_acc > self._global_best[2]:
                self._global_best = ind
            self.train_archive.append(tensor_to_pyg(x, adj, accuracy=val_acc))
            self.train_costs.append(val_acc)
            if t > self._best_test_acc:
                self._best_test_acc = t
            self._query_curve.append((len(self.train_archive), self._best_test_acc))

        pool_sorted = sorted(pool, key=lambda ind: -ind[2])
        k_elite     = max(1, int(cfg.get("pretrain_elite_ratio", 1.0) * len(pool_sorted)))
        gae_pool    = pool_sorted[:k_elite]
        x_batch     = torch.stack([ind[0] for ind in gae_pool]).to(self.device)
        adj_batch   = torch.stack([ind[1] for ind in gae_pool]).to(self.device)
        self.gae.train()
        for _ in range(cfg["pretrain_gae_epochs"]):
            self.gae_opt.zero_grad()
            loss, _ = self.gae.compute_loss(NASGraph.from_101(x_batch, adj_batch))
            loss.backward()
            self.gae_opt.step()

        self.population = pool_sorted[:cfg["pop_size"]]
        print(f"  [Restart] Done, new population best val_acc={self.population[0][2]:.4f}"
              f"  archive={len(self.train_archive)}")


    def run(self, n_generations: int) -> list:
        if not self.population:
            raise RuntimeError("Call pretrain() first to initialize the population.")
        budget            = self.cfg.get("eval_budget")
        stagnation_limit  = self.cfg.get("restart_stagnation", None)
        history           = []

        stagnation_count = 0

        for _ in range(n_generations):
            if budget is not None and len(self.train_archive) >= budget:
                print(f"  [Stop] Real eval reached budget {budget}, terminating.")
                break
            stats = self.step()
            history.append(stats)

            if stagnation_limit is not None:
                if len({ind[2] for ind in self.population}) == 1:
                    stagnation_count += 1
                else:
                    stagnation_count = 0
                budget_available = budget is None or len(self.train_archive) < budget
                if stagnation_count >= stagnation_limit and budget_available:
                    stagnation_count = 0
                    self._restart()

            print(
                f"  Gen {stats['generation']:3d}/{n_generations}"
                f" | best={stats['best_fitness']:.4f}"
                f" | mean={stats['mean_fitness']:.4f}"
                f" | repaired={stats['n_repaired']}"
                f" | real_eval={stats['n_real_eval']}"
                f" | archive={stats['archive_size']}"
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="D-GAENAS on NAS-Bench-101")
    parser.add_argument("--config", help="JSON file whose values override CFG")
    args = parser.parse_args()
    if args.config:
        with open(args.config, "r", encoding="utf-8") as handle:
            CFG.update(json.load(handle))

    device = torch.device(CFG["device"])
    print(f"Device : {device}")

    n_elite  = max(1, math.ceil(CFG["eval_elite_ratio"] * CFG["pop_size"]))
    theo_max = CFG["n_init"] + CFG["n_generations"] * n_elite
    budget   = CFG.get("eval_budget")
    n_runs   = CFG.get("n_runs", 1)
    budget_str = str(budget) if budget is not None else f"unlimited (theoretical max {theo_max})"
    print(f"Real eval budget: {budget_str}  |  runs: {n_runs}\n")

    nb = load_nasbench(CFG["data_file"])

    all_val           = []
    all_test          = []
    all_cache         = []
    all_histories     = []
    all_query_curves  = []

    for run in range(n_runs):
        seed = CFG["seed"] + run
        set_seed(seed)
        print(f"{'─'*55}")
        print(f"  Run {run + 1}/{n_runs}  (seed={seed})")
        print(f"{'─'*55}")

        def _build_gae():
            return UnifiedNASGAE(
                encoder  = GINEncoder(
                    node_feat_dim = N_OPS,
                    hidden_dim    = CFG["hidden_dim"],
                    latent_dim    = CFG["latent_dim"],
                    dropout       = CFG["dropout"],
                ),
                decoders = {
                    "node_feat"  : NodeFeatureDecoder(CFG["latent_dim"], CFG["node_dec_hidden"], N_OPS),
                    "edge_exist" : EdgeExistDecoder(CFG["latent_dim"], CFG["edge_dec_hidden"]),
                },
                noise_fn  = make_noise_fn_101(CFG["node_noise"], CFG["edge_noise"]),
                repair_fn = repair_fn_101,
            ).to(device)

        gae = _build_gae()

        surrogate = SurrogateModel(
            num_nodes       = SM_MAX_NODES,
            node_channels   = NUM_NODE_FEATURES,
            edge_channels   = NUM_EDGE_FEATURES,
            hidden_channels = CFG["surr_hidden"],
            latent_channels = CFG["surr_latent"],
        ).to(device)

        saea = SAEAEvolution(nb, gae, surrogate, CFG)
        saea.pretrain()
        history = saea.run(n_generations=CFG["n_generations"])
        all_histories.append(history)

        best     = saea.best
        best_ops = [OP_STRINGS[i] for i in best[0].argmax(dim=-1).tolist()]
        best_mat = best[1].cpu().numpy().astype(int)

        try:
            spec     = nb_api.ModelSpec(matrix=best_mat, ops=best_ops)
            result   = query_avg(nb, spec)
            val_acc  = result["validation_accuracy"]
            test_acc = result["test_accuracy"]
        except Exception:
            val_acc, test_acc = best[2], float("nan")

        evals_used = len(saea.train_archive)

        all_val.append(val_acc)
        all_test.append(test_acc)
        all_cache.append(evals_used)
        all_query_curves.append(saea._query_curve)
        print(f"  Run {run+1} result — val={val_acc:.4f}  test={test_acc:.4f}"
              f"  real_evals={evals_used}")

    val_arr   = np.array(all_val)
    test_arr  = np.array([v for v in all_test if not math.isnan(v)])
    cache_arr = np.array(all_cache)
    print(f"\n{'═'*55}")
    print(f"  Summary ({n_runs} runs)")
    print(f"{'═'*55}")
    for i, (v, t, c) in enumerate(zip(all_val, all_test, all_cache)):
        print(f"  Run {i+1:2d}: val={v:.4f}  test={t:.4f}"
              f"  real_evals={c}")
    print(f"{'─'*55}")
    print(f"  val_acc   mean={val_arr.mean():.4f}  std={val_arr.std():.4f}"
          f"  best={val_arr.max():.4f}")
    if len(test_arr):
        print(f"  test_acc  mean={test_arr.mean():.4f}  std={test_arr.std():.4f}"
              f"  best={test_arr.max():.4f}")
    print(f"  real_evals  mean={cache_arr.mean():.1f}  std={cache_arr.std():.1f}"
          f"  total={cache_arr.sum()}")
    print(f"{'═'*55}")

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    xlsx_path = f"saea_gae_results_{ts}.xlsx"

    history_rows = []
    for run_idx, history in enumerate(all_histories):
        for row in history:
            history_rows.append({"run": run_idx + 1, "seed": CFG["seed"] + run_idx, **row})
    df_history = pd.DataFrame(history_rows)

    summary_rows = [
        {"run": i + 1, "seed": CFG["seed"] + i,
         "val_acc": v, "test_acc": t, "real_evals": c}
        for i, (v, t, c) in enumerate(zip(all_val, all_test, all_cache))
    ]
    t_mean = float(test_arr.mean()) if len(test_arr) else float("nan")
    t_std  = float(test_arr.std())  if len(test_arr) else float("nan")
    t_best = float(test_arr.max())  if len(test_arr) else float("nan")
    for label, v, t, c in [
        ("mean", float(val_arr.mean()), t_mean, float(cache_arr.mean())),
        ("std",  float(val_arr.std()),  t_std,  float(cache_arr.std())),
        ("best", float(val_arr.max()),  t_best, float(cache_arr.max())),
    ]:
        summary_rows.append({"run": label, "seed": "",
                              "val_acc": v, "test_acc": t, "real_evals": c})
    df_summary = pd.DataFrame(summary_rows)

    cfg_rows = [{"key": k, "value": str(v)} for k, v in CFG.items()]
    df_cfg = pd.DataFrame(cfg_rows)

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df_history.to_excel(writer, sheet_name="history",  index=False)
        df_summary.to_excel(writer, sheet_name="summary",  index=False)
        df_cfg.to_excel(    writer, sheet_name="config",   index=False)

    print(f"\nResults saved to: {xlsx_path}")

    npy_path = xlsx_path.replace(".xlsx", "_curves.npz")
    max_len = max((len(c) for c in all_query_curves), default=0)
    if max_len > 0:
        padded = []
        for c in all_query_curves:
            last = c[-1][1] if c else 0.0
            c_pad = c + [(c[-1][0] + k + 1, last) for k in range(max_len - len(c))]
            padded.append(c_pad)
        np.savez(npy_path, saea_curves=np.array(padded, dtype=float))
        print(f"Curve data saved to: {npy_path}")
        print(f"  Joint plot: python baselines_nb101.py {npy_path}")

    png_path = xlsx_path.replace(".xlsx", "_curve.png")
    plot_evolution_curve(
        all_query_curves,
        save_path=png_path,
        title="NAS-Bench-101",
        xlabel="# Queries",
        ylabel="Accuracy",
        errorbar_every=50,
    )
