
import argparse
import os
import sys
import types
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "core"))

import numpy as np
import torch
from scipy.stats import kendalltau

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
for _mod in ("nasbench.lib.evaluate", "nasbench.lib.model_builder",
             "nasbench.lib.training_time"):
    sys.modules.setdefault(_mod, types.ModuleType(_mod))

import tensorflow as tf
if not hasattr(tf, "python_io"):
    setattr(tf, "python_io", tf.compat.v1.python_io)
tf.get_logger().setLevel("ERROR")

from nasbench import api as nb_api
from torch_geometric.data import Batch

from saea_gae_nb101 import load_nasbench, query_avg, set_seed, OP_STRINGS
from ea import random_individual
from surrogate_model import (
    SurrogateModel,
    nasbench_arch_to_graph,
    train_surrogate_ranking,
    NUM_NODE_FEATURES,
    NUM_EDGE_FEATURES,
)

DEFAULT_DATA_FILE   = "data/nasbench_only108.tfrecord"
DEFAULT_N_ARCHS     = 1000
DEFAULT_SEED        = 37
DEFAULT_DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

SM_MAX_NODES    = 7
SURR_HIDDEN     = 128
SURR_LATENT     = 64


def eval_surrogate_ktau(
    nb_obj,
    surrogate   : SurrogateModel,
    n_archs     : int,
    device      : torch.device,
    seed        : int,
) -> dict:
    set_seed(seed)
    surrogate.eval()

    print(f"Sampling {n_archs} valid random architectures …")
    t0 = time.time()
    archs = []
    while len(archs) < n_archs:
        x, adj = random_individual(device=torch.device("cpu"))
        ops    = [OP_STRINGS[i] for i in x.argmax(dim=-1).tolist()]
        mat    = adj.numpy().astype(int)
        try:
            spec = nb_api.ModelSpec(matrix=mat, ops=ops)
            if not nb_obj.is_valid(spec):
                continue
            r = query_avg(nb_obj, spec)
            archs.append((ops, mat, r["validation_accuracy"]))
        except Exception:
            continue
        if len(archs) % 50 == 0:
            print(f"  Sampled {len(archs)}/{n_archs}  ({time.time()-t0:.1f}s)")
    print(f"Sampling done, elapsed {time.time()-t0:.1f}s")

    graphs = [nasbench_arch_to_graph(ops, mat, acc) for ops, mat, acc in archs]
    batch  = Batch.from_data_list(graphs).to(device)

    print("Surrogate inference …")
    t0 = time.time()
    with torch.no_grad():
        preds = surrogate(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    preds = preds.cpu().numpy()
    print(f"  Elapsed {time.time()-t0:.2f}s")

    gt = np.array([a[2] for a in archs])
    tau, pval = kendalltau(gt, preds)

    return {
        "n_archs"   : n_archs,
        "ktau"      : tau,
        "pval"      : pval,
        "gt_mean"   : gt.mean(),
        "gt_std"    : gt.std(),
        "pred_mean" : preds.mean(),
        "pred_std"  : preds.std(),
        "gt"        : gt,
        "preds"     : preds,
    }


def train_surrogate_standalone(
    nb_obj,
    surrogate   : SurrogateModel,
    n_train     : int   = 400,
    epochs      : int   = 200,
    lr          : float = 1e-3,
    margin      : float = 1.0,
    seed        : int   = DEFAULT_SEED,
    save_path   : str | None = None,
) -> SurrogateModel:
    set_seed(seed)
    optimizer = torch.optim.Adam(surrogate.parameters(), lr=lr)

    print(f"Sampling {n_train} training architectures …")
    train_data, costs = [], []
    while len(train_data) < n_train:
        x, adj = random_individual(device=torch.device("cpu"))
        ops    = [OP_STRINGS[i] for i in x.argmax(dim=-1).tolist()]
        mat    = adj.numpy().astype(int)
        try:
            spec = nb_api.ModelSpec(matrix=mat, ops=ops)
            if not nb_obj.is_valid(spec):
                continue
            r   = query_avg(nb_obj, spec)
            acc = r["validation_accuracy"]
            train_data.append(nasbench_arch_to_graph(ops, mat, acc))
            costs.append(acc)
        except Exception:
            continue
    print(f"Training set ready, starting {epochs} epochs …")

    for ep in range(1, epochs + 1):
        loss, rank_acc = train_surrogate_ranking(
            surrogate, train_data, costs, optimizer, margin=margin
        )
        if ep % 50 == 0 or ep == 1:
            print(f"  Epoch {ep:4d}/{epochs}  loss={loss:.4f}  rank_acc={rank_acc:.3f}")

    if save_path:
        torch.save({"surrogate": surrogate.state_dict()}, save_path)
        print(f"Weights saved to: {save_path}")

    return surrogate


def print_result(label: str, res: dict) -> None:
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"  # architectures : {res['n_archs']}")
    print(f"  Kendall τ       : {res['ktau']:+.4f}  (p={res['pval']:.3e})")
    print(f"{'='*55}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kendall τ: GNN surrogate vs NAS-Bench-101")
    parser.add_argument("--checkpoint",  type=str,   default=None,
                        help="trained surrogate weights (.pt); if omitted, trains standalone first")
    parser.add_argument("--data_file",   type=str,   default=DEFAULT_DATA_FILE)
    parser.add_argument("--n_archs",     type=int,   default=DEFAULT_N_ARCHS,
                        help="number of architectures for ktau evaluation")
    parser.add_argument("--n_train",     type=int,   default=200,
                        help="number of architectures to sample for standalone training")
    parser.add_argument("--train_epochs",type=int,   default=200,
                        help="number of training epochs for standalone training")
    parser.add_argument("--save",        type=str,   default=None,
                        help="path to save trained weights (.pt)")
    parser.add_argument("--seed",        type=int,   default=DEFAULT_SEED)
    parser.add_argument("--eval_seed",   type=int,   default=DEFAULT_SEED,
                        help="fixed seed for eval set (shared across all runs)")
    parser.add_argument("--n_runs",      type=int,   default=30)
    parser.add_argument("--device",      type=str,   default=DEFAULT_DEVICE)
    parser.add_argument("--surr_hidden", type=int,   default=SURR_HIDDEN)
    parser.add_argument("--surr_latent", type=int,   default=SURR_LATENT)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device : {device}")

    print(f"Loading NAS-Bench-101: {args.data_file}")
    nb = load_nasbench(args.data_file)

    import pandas as pd
    from datetime import datetime
    all_rows = []

    for run in range(args.n_runs):
        seed = args.seed + run
        print(f"\n{'─'*50}")
        print(f"  Run {run+1}/{args.n_runs}  seed={seed}")
        print(f"{'─'*50}")

        surrogate = SurrogateModel(
            num_nodes       = SM_MAX_NODES,
            node_channels   = NUM_NODE_FEATURES,
            edge_channels   = NUM_EDGE_FEATURES,
            hidden_channels = args.surr_hidden,
            latent_channels = args.surr_latent,
        ).to(device)

        if args.checkpoint and os.path.exists(args.checkpoint):
            ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=True)
            state = ckpt.get("surrogate", ckpt) if isinstance(ckpt, dict) else ckpt
            surrogate.load_state_dict(state)
            if run == 0:
                print(f"Loaded weights: {args.checkpoint}")
        else:
            train_surrogate_standalone(
                nb_obj    = nb,
                surrogate = surrogate,
                n_train   = args.n_train,
                epochs    = args.train_epochs,
                seed      = seed,
                save_path = args.save if run == 0 else None,
            )

        res = eval_surrogate_ktau(nb, surrogate, args.n_archs, device, args.eval_seed)
        print_result(f"Run {run+1}", res)
        all_rows.append({"run": run+1, "seed": seed,
                         "n_archs": res["n_archs"],
                         "ktau": res["ktau"], "pval": res["pval"]})

    taus = [r["ktau"] for r in all_rows]
    print(f"\n{'='*55}")
    print(f"  Summary ({args.n_runs} runs)")
    print(f"  Kendall τ  mean={np.mean(taus):+.4f}  std={np.std(taus):.4f}"
          f"  max={np.max(taus):+.4f}")
    print(f"{'='*55}")

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    df       = pd.DataFrame(all_rows)
    summary  = pd.DataFrame([{
        "run": "mean", "seed": "", "n_archs": all_rows[0]["n_archs"],
        "ktau": np.mean(taus), "pval": float("nan"),
    }, {
        "run": "std",  "seed": "", "n_archs": all_rows[0]["n_archs"],
        "ktau": np.std(taus),  "pval": float("nan"),
    }, {
        "run": "max",  "seed": "", "n_archs": all_rows[0]["n_archs"],
        "ktau": np.max(taus),  "pval": float("nan"),
    }])
    xlsx_path = f"ktau_surrogate_{ts}.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        pd.concat([df, summary], ignore_index=True).to_excel(
            writer, sheet_name="ktau", index=False)
    print(f"Results saved to: {xlsx_path}")
