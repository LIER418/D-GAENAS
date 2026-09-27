
import os, sys, random, math, types, warnings
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "core"))
from datetime import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
warnings.filterwarnings("ignore")

for _mod in ("nasbench.lib.evaluate", "nasbench.lib.model_builder",
             "nasbench.lib.training_time"):
    sys.modules.setdefault(_mod, types.ModuleType(_mod))

import tensorflow as tf
if not hasattr(tf, "python_io"):
    setattr(tf, "python_io", tf.compat.v1.python_io)
tf.get_logger().setLevel("ERROR")

from nasbench import api as nb_api
from ea import random_individual, individual_hash, repair, is_valid, INTER_OPS
from saea_gae_nb101 import load_nasbench, query_avg, set_seed, OP_STRINGS, CFG


N_QUERIES = 1000
N_RUNS    = CFG.get("n_runs", 30)
BASE_SEED = CFG.get("seed", 37)
DATA_FILE = CFG["data_file"]
DEVICE    = torch.device(CFG["device"])

EDGES   = [(i, j) for i in range(7) for j in range(i + 1, 7)]
N_EDGES = len(EDGES)
OPS_INTER = ["conv1x1-bn-relu", "conv3x3-bn-relu", "maxpool3x3"]

NNI_SEARCH_SPACE = {
    **{f"e{k}": {"_type": "randint", "_value": [0, 2]} for k in range(N_EDGES)},
    **{f"op{k}": {"_type": "randint", "_value": [0, 3]} for k in range(5)},
}


def _lookup(nb_obj, mat: np.ndarray, op_list: list):
    try:
        spec = nb_api.ModelSpec(matrix=mat, ops=op_list)
        if not nb_obj.is_valid(spec):
            return None, None
        r = query_avg(nb_obj, spec)
        return r["validation_accuracy"], r["test_accuracy"]
    except Exception:
        return None, None


def _decode_params(params: dict):
    edges = [params[f"e{k}"] for k in range(N_EDGES)]
    ops   = [params[f"op{k}"] for k in range(5)]
    mat   = np.zeros((7, 7), dtype=int)
    for k, (i, j) in enumerate(EDGES):
        mat[i, j] = int(edges[k])
    op_list = ["input"] + [OPS_INTER[int(o) % 3] for o in ops] + ["output"]
    return mat, op_list


def incumbent_curve(log: list, n_queries: int) -> list:
    curve, best, q = [], 0.0, 0
    for _, test_acc in log:
        q += 1
        t = test_acc if test_acc is not None else 0.0
        if t > best:
            best = t
        curve.append((q, best))
        if q >= n_queries:
            break
    while len(curve) < n_queries:
        curve.append((len(curve) + 1, best))
    return curve


def run_nni_tuner(nb_obj, tuner, n_queries: int) -> list:
    tuner.update_search_space(NNI_SEARCH_SPACE)
    log = []
    for pid in range(n_queries):
        params = tuner.generate_parameters(pid)
        mat, op_list = _decode_params(params)
        v, t = _lookup(nb_obj, mat, op_list)
        if v is None:
            v, t = 0.0, 0.0
        log.append((v, t))
        tuner.receive_trial_result(pid, params, 1.0 - v)
    return log


def run_rs(nb_obj, n_queries: int, seed: int) -> list:
    from nni.algorithms.hpo.random_tuner import RandomTuner
    set_seed(seed)
    tuner = RandomTuner()
    return run_nni_tuner(nb_obj, tuner, n_queries)


def _mutate(nb_obj, mat: np.ndarray, ops: list, max_tries: int = 20):
    import torch, torch.nn.functional as F
    from ea import repair, is_valid
    for _ in range(max_tries):
        m2 = mat.copy(); o2 = ops[:]
        if random.random() < 0.5:
            node = random.randint(1, 5)
            o2[node] = random.choice(OPS_INTER)
        else:
            i, j = random.choice(EDGES)
            m2[i, j] = 1 - m2[i, j]
        x_t   = torch.zeros(7, 5); adj_t = torch.tensor(m2, dtype=torch.float32)
        for ni, op in enumerate(o2):
            x_t[ni, ["input","conv1x1-bn-relu","conv3x3-bn-relu","maxpool3x3","output"].index(op)] = 1
        if not is_valid(x_t, adj_t):
            x_t, adj_t = repair(x_t, adj_t)
            m2 = adj_t.numpy().astype(int)
            argmax = x_t.argmax(dim=-1).tolist()
            all_ops = ["input","conv1x1-bn-relu","conv3x3-bn-relu","maxpool3x3","output"]
            o2 = [all_ops[i] for i in argmax]
        v, t = _lookup(nb_obj, m2, o2)
        if v is not None:
            return m2, o2, v, t
    return None, None, None, None


def run_re(nb_obj, n_queries: int, seed: int,
           pop_size: int = 50, tournament: int = 25) -> list:
    from regularized_evolution import Population as _REPopulation

    set_seed(seed)
    log = []

    def _rand_eval():
        import torch
        from ea import random_individual
        ALL_OPS = ["input","conv1x1-bn-relu","conv3x3-bn-relu","maxpool3x3","output"]
        for _ in range(200):
            x, adj = random_individual(device=torch.device("cpu"))
            ops = [ALL_OPS[i] for i in x.argmax(dim=-1).tolist()]
            mat = adj.numpy().astype(int)
            v, t = _lookup(nb_obj, mat, ops)
            if v is not None:
                return mat, ops, v, t
        return None, None, None, None

    class _NB101Population(_REPopulation):
        def create_initial_population(self):
            while len(self) < self._population_size:
                mat, ops, v, t = _rand_eval()
                if v is None:
                    continue
                log.append((v, t))
                self.add_to_population((mat, ops), fitness=1.0 - v)

    pop = _NB101Population(
        population_size   = pop_size,
        tournament_size   = tournament,
        mutation_probability = 1.0,
    )

    while len(log) < n_queries:
        parent_gene = pop.get_parent()
        parent_mat, parent_ops = parent_gene
        m2, o2, v, t = _mutate(nb_obj, parent_mat, parent_ops)
        if v is None:
            continue
        log.append((v, t))
        pop.add_to_population((m2, o2), fitness=1.0 - v)

    return log


def run_cmaes(nb_obj, n_queries: int, seed: int) -> list:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    log = []

    def objective(trial):
        edges = [trial.suggest_int(f"e{k}", 0, 1) for k in range(N_EDGES)]
        ops   = [trial.suggest_int(f"op{k}", 0, 2) for k in range(5)]
        mat   = np.zeros((7, 7), dtype=int)
        for k, (i, j) in enumerate(EDGES):
            mat[i, j] = int(edges[k])
        op_list = ["input"] + [OPS_INTER[int(o)] for o in ops] + ["output"]
        v, t = _lookup(nb_obj, mat, op_list)
        if v is None:
            log.append((0.0, 0.0)); return 1.0
        log.append((v, t)); return 1.0 - v

    sampler = optuna.samplers.CmaEsSampler(seed=seed, n_startup_trials=20,
                                           with_margin=True)
    study   = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=n_queries, show_progress_bar=False)
    return log[:n_queries]


def run_tpe(nb_obj, n_queries: int, seed: int) -> list:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    log = []

    def objective(trial):
        edges = [trial.suggest_int(f"e{k}", 0, 1) for k in range(N_EDGES)]
        ops   = [trial.suggest_int(f"op{k}", 0, 2) for k in range(5)]
        mat   = np.zeros((7, 7), dtype=int)
        for k, (i, j) in enumerate(EDGES):
            mat[i, j] = int(edges[k])
        op_list = ["input"] + [OPS_INTER[int(o)] for o in ops] + ["output"]
        v, t = _lookup(nb_obj, mat, op_list)
        if v is None:
            log.append((0.0, 0.0)); return 1.0
        log.append((v, t)); return 1.0 - v

    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True,
                                         n_startup_trials=20)
    study   = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=n_queries, show_progress_bar=False)
    return log[:n_queries]


ALGORITHMS = {
    "RS":      run_rs,
    "RE":      run_re,
    "CMA-ES":  run_cmaes,
    "TPE":     run_tpe,
}

COLORS = {
    "D-GAENAS": "#1f77b4",
    "RS":      "#999999",
    "RE":      "#2ca02c",
    "CMA-ES":  "#e377c2",
    "TPE":     "#ff7f0e",
}


def run_all(nb_obj) -> dict:
    results = {}
    for name, fn in ALGORITHMS.items():
        print(f"\n{'═'*52}")
        print(f"  {name}  ({N_RUNS} runs × {N_QUERIES} queries)")
        print(f"{'═'*52}")
        curves = []
        for run in range(N_RUNS):
            seed = BASE_SEED + run
            print(f"  Run {run+1:2d}/{N_RUNS}  seed={seed} …", end=" ", flush=True)
            try:
                log   = fn(nb_obj, N_QUERIES, seed)
                curve = incumbent_curve(log, N_QUERIES)
                print(f"best_test={curve[-1][1]:.4f}")
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"ERROR: {e}")
                curve = [(q, 0.0) for q in range(1, N_QUERIES + 1)]
            curves.append(curve)
        results[name] = curves
    return results


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


def plot_comparison(
    baseline_results: dict,
    saea_curves: list = None,
    save_path: str = "comparison_nb101.png",
    n_queries: int = N_QUERIES,
    errorbar_every: int = 50,
    y_min: float = 0.930,
    y_max: float = 0.944,
) -> None:
    x_grid = np.arange(1, n_queries + 1)
    eb_idx = np.arange(errorbar_every - 1, n_queries, errorbar_every)

    fig, ax = plt.subplots(figsize=(9, 5))

    if saea_curves:
        mat   = _to_matrix(saea_curves, n_queries)
        mean  = mat.mean(axis=0)
        std   = mat.std(axis=0)
        color = COLORS["D-GAENAS"]
        ax.plot(x_grid, mean, color=color, linewidth=2.2,
                label="D-GAENAS", zorder=5)
        ax.errorbar(x_grid[eb_idx], mean[eb_idx], yerr=std[eb_idx],
                    fmt="none", ecolor=color, elinewidth=1.3,
                    capsize=3, capthick=1.3, zorder=5)

    for name, curves in baseline_results.items():
        if not curves:
            continue
        mat   = _to_matrix(curves, n_queries)
        mean  = mat.mean(axis=0)
        std   = mat.std(axis=0)
        color = COLORS.get(name, "#333333")
        ax.plot(x_grid, mean, color=color, linewidth=1.5, label=name)
        ax.errorbar(x_grid[eb_idx], mean[eb_idx], yerr=std[eb_idx],
                    fmt="none", ecolor=color, elinewidth=1.0,
                    capsize=3, capthick=1.0)

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
    print(f"\nComparison plot saved to: {save_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--saea",      default=None, help="SAEA curves npz (*_curves.npz)")
    parser.add_argument("--baselines", default=None, help="existing baselines npz, skip re-running")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.baselines and os.path.exists(args.baselines):
        print(f"Loading existing baseline data: {args.baselines}")
        _d = np.load(args.baselines, allow_pickle=True)
        baseline_results = {}
        for name in _d.files:
            raw = _d[name]
            baseline_results[name] = [
                [(int(p[0]), float(p[1])) for p in run]
                for run in raw
            ]
    else:
        nb = load_nasbench(DATA_FILE)
        baseline_results = run_all(nb)
        npz_path = f"baselines_nb101_{ts}.npz"
        np.savez(npz_path, **{
            name: np.array([[(p[0], p[1]) for p in curve] for curve in curves], dtype=float)
            for name, curves in baseline_results.items()
        })
        print(f"\nBaseline data saved to: {npz_path}")

    saea_curves = []
    saea_file = args.saea
    if saea_file and os.path.exists(saea_file):
        _sd = np.load(saea_file, allow_pickle=True)
        raw = _sd["saea_curves"]
        saea_curves = [
            [(int(p[0]), float(p[1])) for p in run]
            for run in raw
        ]
        print(f"Loaded SAEA curves: {saea_file}  ({len(saea_curves)} runs)")

    png_path = f"comparison_nb101_{ts}.png"
    plot_comparison(
        baseline_results,
        saea_curves=saea_curves,
        save_path=png_path,
    )
