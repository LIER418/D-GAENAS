import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import sys, types, random, math, argparse
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "core"))
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

for _mod in ("nasbench.lib.evaluate", "nasbench.lib.model_builder",
             "nasbench.lib.training_time"):
    sys.modules.setdefault(_mod, types.ModuleType(_mod))

import tensorflow as tf
if not hasattr(tf, "python_io"):
    setattr(tf, "python_io", tf.compat.v1.python_io)
tf.get_logger().setLevel("ERROR")

from nasbench import api as nb_api

from ea import (N_OPS, mutate, repair, is_valid, random_individual,
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
    "pretrain_surr_epochs" : 100,

    "pop_size"      : 100,
    "n_generations" : 300,
    "eval_elite_ratio" : 1,

    "node_mut_prob" : 0.5,
    "edge_add_prob" : 0.5,
    "edge_del_prob" : 0.5,

    "surr_online_epochs" : 50,
    "surr_margin"        : 1.0,
    "surr_hidden"        : 64,
    "surr_latent"        : 64,
    "surr_lr"            : 1e-3,

    "device"      : "cpu",
    "seed"        : 38,
    "eval_budget" : 1000,
}

OP_STRINGS   = ["input", "conv1x1-bn-relu", "conv3x3-bn-relu", "maxpool3x3", "output"]
N_NODES      = 7
N_EDGES_TRIU = 21
N_INTER      = 5


def set_seed(s: int) -> None:
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def load_nasbench(path: str) -> nb_api.NASBench:
    print(f"Loading {path} …")
    nb = nb_api.NASBench(path)
    print(f"Loaded — {len(nb.computed_statistics)} unique architectures.\n")
    return nb


def query_avg(nb, spec, epochs=108) -> dict:
    _, computed = nb.get_metrics_from_spec(spec)
    runs = computed[epochs]
    return {
        "validation_accuracy": float(np.mean([r["final_validation_accuracy"] for r in runs])),
        "test_accuracy":       float(np.mean([r["final_test_accuracy"]       for r in runs])),
    }


def eval_arch(nb, x: torch.Tensor, adj: torch.Tensor):
    ops = [OP_STRINGS[i] for i in x.argmax(dim=-1).tolist()]
    mat = adj.cpu().numpy().astype(int)
    try:
        spec = nb_api.ModelSpec(matrix=mat, ops=ops)
        if not nb.is_valid(spec):
            return None, None
        r = query_avg(nb, spec)
        return r["validation_accuracy"], r["test_accuracy"]
    except Exception:
        return None, None


def tensor_to_pyg(x: torch.Tensor, adj: torch.Tensor, accuracy: float = 0.0):
    ops = [OP_STRINGS[i] for i in x.argmax(dim=-1).tolist()]
    mat = adj.cpu().numpy().astype(int)
    return nasbench_arch_to_graph(ops, mat, accuracy)


def _encode(x: torch.Tensor, adj: torch.Tensor) -> list:
    triu  = torch.triu(torch.ones(N_NODES, N_NODES, dtype=torch.bool), diagonal=1)
    edges = adj[triu].long().tolist()
    ops   = x[1:N_NODES - 1].argmax(dim=-1).tolist()
    return edges + ops


def _decode(genes: list, device) -> tuple:
    edges = genes[:N_EDGES_TRIU]
    ops   = genes[N_EDGES_TRIU:]

    adj = torch.zeros(N_NODES, N_NODES, dtype=torch.float32)
    triu_idx = torch.triu(torch.ones(N_NODES, N_NODES, dtype=torch.bool), diagonal=1)
    adj[triu_idx] = torch.tensor(edges, dtype=torch.float32)

    node_ops = [0] + [max(1, min(3, o)) for o in ops] + [4]
    x = torch.zeros(N_NODES, len(OP_STRINGS))
    for i, op in enumerate(node_ops):
        x[i, op] = 1.0

    return x.to(device), adj.to(device)


def _uniform_crossover(genes_a: list, genes_b: list) -> list:
    return [a if random.random() < 0.5 else b for a, b in zip(genes_a, genes_b)]


class _AblationBase:

    def __init__(self, nb, surrogate: SurrogateModel, cfg: dict):
        self.nb        = nb
        self.surrogate = surrogate
        self.cfg       = cfg
        self.device    = torch.device(cfg["device"])

        self.surr_opt = optim.Adam(surrogate.parameters(), lr=cfg["surr_lr"])

        self.population:    list = []
        self.train_archive: list = []
        self.train_costs:   list = []
        self.cache           = EvalCache(max_size=20_000)
        self.generation      = 0
        self._best_test_acc: float = 0.0
        self._query_curve:   list  = []

    def _generate_candidates(self) -> tuple:
        raise NotImplementedError

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
            pool.append([x, adj, val_acc, t])
            if t > self._best_test_acc:
                self._best_test_acc = t
            self._query_curve.append((len(pool), self._best_test_acc))

        for x, adj, val_acc, _ in pool:
            self.train_archive.append(tensor_to_pyg(x, adj, accuracy=val_acc))
            self.train_costs.append(val_acc)

        print(f"Pre-training surrogate model ({cfg['pretrain_surr_epochs']} epochs) …")
        for _ in range(cfg["pretrain_surr_epochs"]):
            train_surrogate_ranking(
                self.surrogate, self.train_archive, self.train_costs,
                self.surr_opt, margin=cfg["surr_margin"],
            )

        self.population = sorted(pool, key=lambda ind: -ind[2])
        print(f"Initial population best val_acc={self.population[0][2]:.4f}  "
              f"test_acc={self.population[0][3]:.4f}\n")

    def step(self) -> dict:
        cfg    = self.cfg
        k_eval = max(1, math.ceil(cfg["eval_elite_ratio"] * len(self.population)))

        candidates, n_repaired = self._generate_candidates()

        pyg_list    = [tensor_to_pyg(x, adj) for x, adj in candidates]
        surr_scores = predict_surrogate(self.surrogate, pyg_list)
        ranked      = sorted(range(len(surr_scores)), key=lambda i: -surr_scores[i])
        top_indices = ranked[:k_eval]

        new_inds    = []
        n_real_eval = 0
        for i in top_indices:
            x, adj = candidates[i]
            key    = individual_hash(x, adj)
            cached = self.cache.get(key)
            if cached is not None:
                new_inds.append([x, adj, cached, 0.0])
                continue
            val_acc, test_acc = eval_arch(self.nb, x, adj)
            if val_acc is None:
                val_acc, test_acc = 0.0, 0.0
            self.cache.put(key, val_acc)
            n_real_eval += 1
            self.train_archive.append(tensor_to_pyg(x, adj, accuracy=val_acc))
            self.train_costs.append(val_acc)
            new_inds.append([x, adj, val_acc, test_acc])
            t = test_acc if test_acc is not None else 0.0
            if t > self._best_test_acc:
                self._best_test_acc = t
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
            "generation"    : self.generation,
            "best_fitness"  : self.population[0][2],
            "best_test_acc" : self.population[0][3],
            "mean_fitness"  : float(np.mean(fitnesses)),
            "n_repaired"    : n_repaired,
            "n_real_eval"   : n_real_eval,
            "archive_size"  : len(self.train_archive),
        }

    def run(self, n_generations: int) -> list:
        if not self.population:
            raise RuntimeError("Call pretrain() first to initialize the population.")
        budget  = self.cfg.get("eval_budget")
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
                f" | archive={stats['archive_size']}"
                + (f" | budget_left={budget - len(self.train_archive)}"
                   if budget is not None else "")
            )
        return history

    @property
    def best(self):
        return max(self.population, key=lambda ind: ind[2])


class AblationMutation(_AblationBase):

    def _generate_candidates(self) -> tuple:
        cfg = self.cfg
        candidates, n_repaired = [], 0
        for ind in self.population:
            x_mut, adj_mut = mutate(
                ind[0], ind[1],
                node_mut_prob=cfg["node_mut_prob"],
                edge_add_prob=cfg["edge_add_prob"],
                edge_del_prob=cfg["edge_del_prob"],
            )
            if not is_valid(x_mut, adj_mut):
                x_mut, adj_mut = repair(x_mut, adj_mut)
                n_repaired += 1
            candidates.append((x_mut.to(self.device), adj_mut.to(self.device)))
        return candidates, n_repaired


class AblationCrossover(_AblationBase):

    def _generate_candidates(self) -> tuple:
        pop_genes  = [_encode(ind[0], ind[1]) for ind in self.population]
        candidates, n_repaired = [], 0
        for i, genes_a in enumerate(pop_genes):
            j = random.choice([k for k in range(len(pop_genes)) if k != i])
            child_genes = _uniform_crossover(genes_a, pop_genes[j])
            x_c, adj_c = _decode(child_genes, self.device)
            if not is_valid(x_c, adj_c):
                x_c, adj_c = repair(x_c, adj_c)
                n_repaired += 1
            candidates.append((x_c, adj_c))

        unique = {tuple(_encode(ind[0], ind[1])) for ind in self.population}
        self._converged = len(unique) == 1
        return candidates, n_repaired

    def run(self, n_generations: int) -> list:
        self._converged = False
        if not self.population:
            raise RuntimeError("Call pretrain() first to initialize the population.")
        budget  = self.cfg.get("eval_budget")
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
                f" | archive={stats['archive_size']}"
                + (f" | budget_left={budget - len(self.train_archive)}"
                   if budget is not None else "")
            )
            if self._converged:
                print(f"  [Stop] Population converged to a single architecture, terminating.")
                break
        return history


class AblationCrossoverMutation(_AblationBase):

    def _generate_candidates(self) -> tuple:
        cfg        = self.cfg
        pop_genes  = [_encode(ind[0], ind[1]) for ind in self.population]
        candidates, n_repaired = [], 0
        for i, genes_a in enumerate(pop_genes):
            j = random.choice([k for k in range(len(pop_genes)) if k != i])
            child_genes = _uniform_crossover(genes_a, pop_genes[j])
            x_c, adj_c = _decode(child_genes, self.device)
            x_c, adj_c = mutate(
                x_c, adj_c,
                node_mut_prob=cfg["node_mut_prob"],
                edge_add_prob=cfg["edge_add_prob"],
                edge_del_prob=cfg["edge_del_prob"],
            )
            if not is_valid(x_c, adj_c):
                x_c, adj_c = repair(x_c, adj_c)
                n_repaired += 1
            candidates.append((x_c.to(self.device), adj_c.to(self.device)))
        return candidates, n_repaired


COLORS = {
    "D-GAENAS"  : "#1f77b4",
    "Mutation"  : "#d62728",
    "Crossover" : "#9467bd",
    "Cross+Mut" : "#8c564b",
    "RS"        : "#999999",
    "RE"        : "#2ca02c",
    "CMA-ES"    : "#e377c2",
    "TPE"       : "#ff7f0e",
}

N_QUERIES = CFG.get("eval_budget", 1000)



def _to_matrix(curves: list, n_queries: int) -> np.ndarray:
    mat = []
    for curve in curves:
        ys = np.zeros(n_queries)
        ci, cur = 0, 0.0
        for qi in range(n_queries):
            xv = qi + 1
            while ci < len(curve) and curve[ci][0] <= xv:
                cur = curve[ci][1]; ci += 1
            ys[qi] = cur
        mat.append(ys)
    return np.array(mat)


def plot_ablation(
    results:          dict,
    saea_curves:      list | None = None,
    baseline_results: dict | None = None,
    save_path:        str  = "ablation_nb101.png",
    n_queries:        int  = N_QUERIES,
    errorbar_every:   int  = 50,
    y_min:            float = 0.930,
    y_max:            float = 0.944,
) -> None:
    x_grid = np.arange(1, n_queries + 1)
    eb_idx = np.arange(errorbar_every - 1, n_queries, errorbar_every)
    fig, ax = plt.subplots(figsize=(9, 5))

    if saea_curves:
        mat  = _to_matrix(saea_curves, n_queries)
        mean, std = mat.mean(axis=0), mat.std(axis=0)
        c = COLORS["D-GAENAS"]
        ax.plot(x_grid, mean, color=c, linewidth=2.2, label="D-GAENAS", zorder=5)
        ax.errorbar(x_grid[eb_idx], mean[eb_idx], yerr=std[eb_idx],
                    fmt="none", ecolor=c, elinewidth=1.3, capsize=3, capthick=1.3, zorder=5)

    for name, curves in results.items():
        if not curves:
            continue
        mat  = _to_matrix(curves, n_queries)
        mean, std = mat.mean(axis=0), mat.std(axis=0)
        c = COLORS.get(name, "#333333")
        ax.plot(x_grid, mean, color=c, linewidth=2.0, label=name, zorder=4)
        ax.errorbar(x_grid[eb_idx], mean[eb_idx], yerr=std[eb_idx],
                    fmt="none", ecolor=c, elinewidth=1.2, capsize=3, capthick=1.2, zorder=4)

    if baseline_results:
        for name, curves in baseline_results.items():
            if not curves:
                continue
            mat  = _to_matrix(curves, n_queries)
            mean, std = mat.mean(axis=0), mat.std(axis=0)
            c = COLORS.get(name, "#333333")
            ax.plot(x_grid, mean, color=c, linewidth=1.5, linestyle="--", label=name)
            ax.errorbar(x_grid[eb_idx], mean[eb_idx], yerr=std[eb_idx],
                        fmt="none", ecolor=c, elinewidth=1.0, capsize=3, capthick=1.0)

    ax.set_xlim(0, n_queries)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("# Queries", fontsize=22)
    ax.set_ylabel("Test Accuracy", fontsize=22)
    ax.tick_params(axis="both", labelsize=20)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(fontsize=18, loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"\nAblation plot saved to: {save_path}")


VARIANTS = [
    ("Mutation",   AblationMutation),
    # ("Crossover",  AblationCrossover),
    # ("Cross+Mut",  AblationCrossoverMutation),
]


def _run_variant(name, cls, nb, device, n_runs) -> tuple:
    print(f"\n{'★'*55}")
    print(f"  Variant: {name}")
    print(f"{'★'*55}\n")

    all_val, all_test, all_cache, all_histories, all_query_curves = [], [], [], [], []

    for run in range(n_runs):
        seed = CFG["seed"] + run
        set_seed(seed)
        print(f"{'─'*55}")
        print(f"  {name} — Run {run + 1}/{n_runs}  (seed={seed})")
        print(f"{'─'*55}")

        surrogate = SurrogateModel(
            num_nodes       = SM_MAX_NODES,
            node_channels   = NUM_NODE_FEATURES,
            edge_channels   = NUM_EDGE_FEATURES,
            hidden_channels = CFG["surr_hidden"],
            latent_channels = CFG["surr_latent"],
        ).to(device)

        algo = cls(nb, surrogate, CFG)
        algo.pretrain()
        history = algo.run(n_generations=CFG["n_generations"])
        all_histories.append(history)

        best     = algo.best
        best_ops = [OP_STRINGS[i] for i in best[0].argmax(dim=-1).tolist()]
        best_mat = best[1].cpu().numpy().astype(int)
        try:
            spec     = nb_api.ModelSpec(matrix=best_mat, ops=best_ops)
            result   = query_avg(nb, spec)
            val_acc  = result["validation_accuracy"]
            test_acc = result["test_accuracy"]
        except Exception:
            val_acc, test_acc = best[2], float("nan")

        all_val.append(val_acc)
        all_test.append(test_acc)
        all_cache.append(len(algo.train_archive))
        all_query_curves.append(algo._query_curve)
        print(f"  Run {run+1} — val={val_acc:.4f}  test={test_acc:.4f}"
              f"  real_evals={len(algo.train_archive)}")

    return all_val, all_test, all_cache, all_histories, all_query_curves


def _print_summary(name, all_val, all_test, all_cache) -> None:
    val_arr   = np.array(all_val)
    test_list = [v for v in all_test if not math.isnan(v)]
    test_arr  = np.array(test_list) if test_list else np.array([float("nan")])
    cache_arr = np.array(all_cache)
    n_runs    = len(all_val)

    print(f"\n{'═'*55}")
    print(f"  [{name}] Summary ({n_runs} runs)")
    print(f"{'═'*55}")
    for i, (v, t, c) in enumerate(zip(all_val, all_test, all_cache)):
        print(f"  Run {i+1:2d}: val={v:.4f}  test={t:.4f}  real_evals={c}")
    print(f"{'─'*55}")
    print(f"  val_acc   mean={val_arr.mean():.4f}  std={val_arr.std():.4f}"
          f"  best={val_arr.max():.4f}")
    if not math.isnan(test_arr[0]):
        print(f"  test_acc  mean={test_arr.mean():.4f}  std={test_arr.std():.4f}"
              f"  best={test_arr.max():.4f}")
    print(f"  real_evals  mean={cache_arr.mean():.1f}  std={cache_arr.std():.1f}"
          f"  total={cache_arr.sum()}")
    print(f"{'═'*55}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--saea",      default=None, help="SAEA curves .npz")
    parser.add_argument("--baselines", default=None, help="baselines_nb101 .npz")
    parser.add_argument("--ablation",  default=None, help="existing ablation curves .npz, skip re-running")
    parser.add_argument("--variants",  default="all",
                        help="variants to run (comma-separated): Mutation,Crossover,Cross+Mut or all")
    _args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    variant_curves: dict = {}

    if _args.ablation and os.path.exists(_args.ablation):
        print(f"Loading existing ablation data: {_args.ablation}")
        _ad = np.load(_args.ablation, allow_pickle=True)
        for key in _ad.files:
            name = key.replace("_plus_", "+").replace("_", " ").strip()
            variant_curves[name] = [
                [(int(p[0]), float(p[1])) for p in run]
                for run in _ad[key]
            ]
        print(f"Loaded variants: {list(variant_curves.keys())}")
    else:
        device = torch.device(CFG["device"])
        n_runs = CFG.get("n_runs", 1)
        budget = CFG.get("eval_budget")
        print(f"Device: {device}  |  runs: {n_runs}  |  budget: {budget}\n")

        nb = load_nasbench(CFG["data_file"])

        if _args.variants == "all":
            run_variants = VARIANTS
        else:
            req = {v.strip() for v in _args.variants.split(",")}
            run_variants = [(n, c) for n, c in VARIANTS if n in req]

        all_results = {}

        for name, cls in run_variants:
            all_val, all_test, all_cache, all_histories, all_query_curves = _run_variant(
                name, cls, nb, device, n_runs
            )
            _print_summary(name, all_val, all_test, all_cache)
            all_results[name]    = (all_val, all_test, all_cache, all_histories)
            variant_curves[name] = all_query_curves

        xlsx_path = f"ablation_nb101_results_{ts}.xlsx"
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            for name, (all_val, all_test, all_cache, all_histories) in all_results.items():
                pfx = name[:8]

                history_rows = []
                for run_idx, history in enumerate(all_histories):
                    for row in history:
                        history_rows.append({"variant": name, "run": run_idx + 1,
                                             "seed": CFG["seed"] + run_idx, **row})
                pd.DataFrame(history_rows).to_excel(
                    writer, sheet_name=f"{pfx}_hist", index=False)

                val_arr   = np.array(all_val)
                test_list = [v for v in all_test if not math.isnan(v)]
                test_arr  = np.array(test_list) if test_list else np.array([float("nan")])
                cache_arr = np.array(all_cache)
                summary_rows = [
                    {"variant": name, "run": i + 1, "seed": CFG["seed"] + i,
                     "val_acc": v, "test_acc": t, "real_evals": c}
                    for i, (v, t, c) in enumerate(zip(all_val, all_test, all_cache))
                ]
                for label, v, t, c in [
                    ("mean", float(val_arr.mean()),
                             float(test_arr.mean()) if not math.isnan(test_arr[0]) else float("nan"),
                             float(cache_arr.mean())),
                    ("std",  float(val_arr.std()),
                             float(test_arr.std())  if not math.isnan(test_arr[0]) else float("nan"),
                             float(cache_arr.std())),
                    ("best", float(val_arr.max()),
                             float(test_arr.max())  if not math.isnan(test_arr[0]) else float("nan"),
                             float(cache_arr.max())),
                ]:
                    summary_rows.append({"variant": name, "run": label, "seed": "",
                                         "val_acc": v, "test_acc": t, "real_evals": c})
                pd.DataFrame(summary_rows).to_excel(
                    writer, sheet_name=f"{pfx}_summ", index=False)

            pd.DataFrame([{"key": k, "value": str(v)} for k, v in CFG.items()]).to_excel(
                writer, sheet_name="config", index=False)

        print(f"\nResults saved to: {xlsx_path}")

        npz_data = {}
        for name, curves in variant_curves.items():
            key = name.replace("+", "_plus_").replace(" ", "_")
            npz_data[key] = np.array([np.array(c, dtype=float) for c in curves], dtype=object)
        npz_path = f"ablation_nb101_{ts}_curves.npz"
        np.savez(npz_path, **npz_data)
        print(f"Curve data saved to: {npz_path}")

    saea_curves: list = []
    if _args.saea and os.path.exists(_args.saea):
        _sd = np.load(_args.saea, allow_pickle=True)
        saea_curves = [[(int(p[0]), float(p[1])) for p in r] for r in _sd["saea_curves"]]

    baseline_results: dict = {}
    if _args.baselines and os.path.exists(_args.baselines):
        _bd = np.load(_args.baselines, allow_pickle=True)
        baseline_results = {n: [list(map(tuple, r)) for r in _bd[n]] for n in _bd.files}

    png_path = f"ablation_nb101_{ts}.png"
    plot_ablation(
        variant_curves,
        saea_curves      or None,
        baseline_results or None,
        save_path = png_path,
    )
