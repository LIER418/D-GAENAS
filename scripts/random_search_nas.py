import os, sys, random, argparse, types
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "core"))

import numpy as np
import torch
from datetime import datetime


def set_seed(s: int) -> None:
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


NB101_CFG = {
    "data_file"    : "data/nasbench_only108.tfrecord",
    "eval_budget"  : 423,
    "n_runs"       : 30,
    "seed"         : 38,
}

NB201_CFG = {
    "data_file"    : "data/NATS-tss-v1_0-3ffb9.pickle.pbz2",
    "eval_budget"  : 100,
    "n_runs"       : 30,
    "seed"         : 38,
    "datasets"     : ["cifar10-valid", "cifar100", "ImageNet16-120"],
}


def run_nb101(cfg: dict) -> list:
    for _mod in ("nasbench.lib.evaluate", "nasbench.lib.model_builder",
                 "nasbench.lib.training_time"):
        sys.modules.setdefault(_mod, types.ModuleType(_mod))

    import tensorflow as tf
    if not hasattr(tf, "python_io"):
        setattr(tf, "python_io", tf.compat.v1.python_io)
    tf.get_logger().setLevel("ERROR")

    from nasbench import api as nb_api
    import torch.nn.functional as F

    def _random_individual():
        n_nodes = 7
        inter_ops = [1, 2, 3]
        cls = torch.zeros(n_nodes, dtype=torch.long)
        cls[0] = 0; cls[-1] = 4
        for i in range(1, n_nodes - 1):
            cls[i] = inter_ops[torch.randint(0, 3, (1,)).item()]
        x = F.one_hot(cls, num_classes=5).float()
        adj = torch.zeros(n_nodes, n_nodes)
        edges = [(i, j) for i in range(n_nodes) for j in range(i + 1, n_nodes)]
        n_edges = torch.randint(1, len(edges) + 1, (1,)).item()
        perm = torch.randperm(len(edges)).tolist()
        for k in perm[:n_edges]:
            i, j = edges[k]
            adj[i, j] = 1.0
        return x, adj

    OP_STRINGS = ["input", "conv1x1-bn-relu", "conv3x3-bn-relu", "maxpool3x3", "output"]

    def _eval(nb, x, adj):
        ops = [OP_STRINGS[i] for i in x.argmax(dim=-1).tolist()]
        mat = adj.cpu().numpy().astype(int)
        try:
            spec = nb_api.ModelSpec(matrix=mat, ops=ops)
            if not nb.is_valid(spec):
                return None, None
            _, computed = nb.get_metrics_from_spec(spec)
            runs = computed[108]
            val  = float(np.mean([r["final_validation_accuracy"] for r in runs]))
            test = float(np.mean([r["final_test_accuracy"]       for r in runs]))
            return val, test
        except Exception:
            return None, None

    print(f"Loading {cfg['data_file']} …")
    nb = nb_api.NASBench(cfg["data_file"])
    print(f"Loaded — {len(nb.computed_statistics)} unique architectures.\n")

    all_curves = []

    for run in range(cfg["n_runs"]):
        seed = cfg["seed"] + run
        set_seed(seed)
        print(f"  Run {run+1}/{cfg['n_runs']}  seed={seed}", end=" … ", flush=True)

        budget = cfg["eval_budget"]
        best_val, best_test = 0.0, 0.0
        curve = []
        evals = 0

        while evals < budget:
            x, adj = _random_individual()
            val, test = _eval(nb, x, adj)
            evals += 1
            if val is not None and val > best_val:
                best_val, best_test = val, test
            curve.append((evals, best_test))

        print(f"best_val={best_val:.4f}  best_test={best_test:.4f}")
        all_curves.append((curve, best_val, best_test))

    finals_val  = np.array([r[1] for r in all_curves])
    finals_test = np.array([r[2] for r in all_curves])
    print(f"\n  val_acc   mean={finals_val.mean():.4f}  std={finals_val.std():.4f}  best={finals_val.max():.4f}")
    print(f"  test_acc  mean={finals_test.mean():.4f}  std={finals_test.std():.4f}  best={finals_test.max():.4f}")

    return [r[0] for r in all_curves]


def run_nb201(cfg: dict) -> dict:
    from nasbench201_utils import random_arch, query_nasbench201, load_nasbench201

    api = load_nasbench201(cfg["data_file"])
    results = {ds: [] for ds in cfg["datasets"]}

    for ds in cfg["datasets"]:
        print(f"\n{'─'*50}")
        print(f"  Dataset: {ds}")
        print(f"{'─'*50}")

        for run in range(cfg["n_runs"]):
            seed = cfg["seed"] + run
            set_seed(seed)
            print(f"  Run {run+1}/{cfg['n_runs']}  seed={seed}", end=" … ", flush=True)

            budget = cfg["eval_budget"]
            best_val, best_test = 0.0, 0.0
            curve = []
            evals = 0

            while evals < budget:
                ops = random_arch()
                val, test = query_nasbench201(api, ops, dataset=ds)
                if val is None:
                    continue
                evals += 1
                if val > best_val:
                    best_val, best_test = val, test
                curve.append((evals, best_test))

            print(f"best_val={best_val:.4f}  best_test={best_test:.4f}")
            results[ds].append((curve, best_val, best_test))

        finals_val  = np.array([r[1] for r in results[ds]])
        finals_test = np.array([r[2] for r in results[ds]])
        print(f"\n  [{ds}] val_acc   mean={finals_val.mean():.4f}  std={finals_val.std():.4f}  best={finals_val.max():.4f}")
        print(f"  [{ds}] test_acc  mean={finals_test.mean():.4f}  std={finals_test.std():.4f}  best={finals_test.max():.4f}")
        results[ds] = [r[0] for r in results[ds]]

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", choices=["101", "201", "both"], default="both",
                        help="which benchmark to run")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.bench in ("101", "both"):
        print("\n" + "═"*55)
        print("  NAS-Bench-101  Random Search")
        print("═"*55 + "\n")
        curves_101 = run_nb101(NB101_CFG)
        npz_path = f"rs_nb101_{ts}_curves.npz"
        np.savez(npz_path,
                 saea_curves=np.array([np.array(c, dtype=float) for c in curves_101], dtype=object))
        print(f"\nSaved: {npz_path}")

    if args.bench in ("201", "both"):
        print("\n" + "═"*55)
        print("  NAS-Bench-201  Random Search")
        print("═"*55)
        curves_201 = run_nb201(NB201_CFG)
        npz_path = f"rs_nb201_{ts}_curves.npz"
        save_dict = {ds.replace("-", "_"): np.array([np.array(c, dtype=float) for c in crvs], dtype=object)
                     for ds, crvs in curves_201.items()}
        np.savez(npz_path, **save_dict)
        print(f"\nSaved: {npz_path}")
