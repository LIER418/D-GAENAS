
import argparse
import os
import random
import sys
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "core"))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import kendalltau

from nasbench201_utils import (
    load_nasbench201, set_seed,
    N_OPS,
)
from surrogate_model import GINESurrogate201

DEFAULT_API_PATH   = "data/NATS-tss-v1_0-3ffb9.pickle.pbz2"
DEFAULT_N_ARCHS    = 1000
DEFAULT_SEED       = 37
DEFAULT_DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
SURR_HIDDEN        = 64
SURR_LATENT        = 64
DATASET            = "cifar10-valid"


def _sample_by_random_arch(api, n: int, seed: int) -> list:
    import torch.nn.functional as F
    from nasbench201_utils import arch_str_to_ops
    set_seed(seed)
    total   = 15625
    indices = random.sample(range(total), min(n, total))
    result  = []
    for idx in indices:
        try:
            val_info   = api.get_more_info(idx, 'cifar10-valid', hp=200, is_random=False)
            val_acc    = val_info['valid-accuracy'] / 100.0
            cfg        = api.get_net_config(idx, 'cifar10-valid')
            arch_str   = cfg['arch_str']
            ops_list   = arch_str_to_ops(arch_str)
            edge_feats = F.one_hot(torch.tensor(ops_list), num_classes=N_OPS).float()
            result.append((edge_feats, val_acc))
        except Exception:
            continue
    return result


def eval_surrogate_ktau_201(
    api,
    surrogate : GINESurrogate201,
    n_archs   : int,
    device    : torch.device,
    seed      : int,
) -> dict:
    set_seed(seed)
    surrogate.eval()

    print(f"Sampling {n_archs} architectures from NAS-Bench-201 ({DATASET}) …")
    t0    = time.time()
    archs = _sample_by_random_arch(api, n_archs, seed)
    print(f"Sampling done, {len(archs)} architectures, elapsed {time.time()-t0:.1f}s")

    edge_feats_list = [a[0] for a in archs]
    gt              = np.array([a[1] for a in archs])

    print("Surrogate inference …")
    t0 = time.time()
    with torch.no_grad():
        batch_size = 256
        preds = []
        for i in range(0, len(edge_feats_list), batch_size):
            batch_t = torch.stack(edge_feats_list[i:i+batch_size]).to(device)
            preds.append(surrogate(batch_t).cpu())
    preds = torch.cat(preds).numpy()
    print(f"  Elapsed {time.time()-t0:.2f}s")

    tau, pval = kendalltau(gt, preds)
    return {
        "n_archs"   : len(archs),
        "ktau"      : tau,
        "pval"      : pval,
        "gt_mean"   : gt.mean(),
        "gt_std"    : gt.std(),
        "pred_mean" : preds.mean(),
        "pred_std"  : preds.std(),
        "gt"        : gt,
        "preds"     : preds,
    }


def print_result(label: str, res: dict) -> None:
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"  # architectures : {res['n_archs']}")
    print(f"  Kendall τ       : {res['ktau']:+.4f}  (p={res['pval']:.3e})")
    print(f"  GT val_acc      : {res['gt_mean']:.4f} ± {res['gt_std']:.4f}")
    print(f"  Surrogate score : {res['pred_mean']:.4f} ± {res['pred_std']:.4f}")
    print(f"{'='*55}")


def train_surrogate_standalone_201(
    api,
    surrogate   : GINESurrogate201,
    n_train     : int   = 400,
    epochs      : int   = 200,
    lr          : float = 1e-3,
    margin      : float = 1.0,
    n_pairs     : int   = 256,
    seed        : int   = DEFAULT_SEED,
    save_path   : str | None = None,
) -> GINESurrogate201:
    set_seed(seed)
    optimizer = optim.Adam(surrogate.parameters(), lr=lr)
    criterion = nn.MarginRankingLoss(margin=margin)
    device    = next(surrogate.parameters()).device

    print(f"Sampling {n_train} training architectures …")
    archs  = _sample_by_random_arch(api, n_train, seed)
    archive = [a[0] for a in archs]
    scores  = [a[1] for a in archs]
    print(f"Training set ready ({len(archive)} architectures), starting {epochs} epochs …")

    for ep in range(1, epochs + 1):
        surrogate.train()
        n = len(archive)

        idx_a, idx_b = [], []
        attempts, max_att = 0, n_pairs * 20
        while len(idx_a) < n_pairs and attempts < max_att:
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if scores[i] != scores[j]:
                idx_a.append(i); idx_b.append(j)
            attempts += 1

        if not idx_a:
            continue

        ops_a = torch.stack([archive[i] for i in idx_a]).to(device)
        ops_b = torch.stack([archive[i] for i in idx_b]).to(device)
        s_a   = torch.tensor([scores[i] for i in idx_a], device=device)
        s_b   = torch.tensor([scores[i] for i in idx_b], device=device)

        pred_a = surrogate(ops_a)
        pred_b = surrogate(ops_b)
        y      = (s_a > s_b).float() * 2 - 1

        optimizer.zero_grad()
        loss = criterion(pred_a, pred_b, y)
        loss.backward()
        optimizer.step()

        if ep % 50 == 0 or ep == 1:
            rank_acc = ((pred_a > pred_b) == (s_a > s_b)).float().mean().item()
            print(f"  Epoch {ep:4d}/{epochs}  loss={loss.item():.4f}  rank_acc={rank_acc:.3f}")

    if save_path:
        torch.save({"surrogate": surrogate.state_dict()}, save_path)
        print(f"Weights saved to: {save_path}")

    return surrogate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Kendall τ：GINESurrogate201 vs NAS-Bench-201 CIFAR-10"
    )
    parser.add_argument("--checkpoint",   type=str,   default=None,
                        help="trained weights (.pt); if omitted, trains standalone first")
    parser.add_argument("--api_path",     type=str,   default=DEFAULT_API_PATH,
                        help="path to NAS-Bench-201 API file")
    parser.add_argument("--n_archs",      type=int,   default=DEFAULT_N_ARCHS,
                        help="number of architectures for ktau evaluation (max 15625)")
    parser.add_argument("--n_train",      type=int,   default=400,
                        help="number of architectures to sample for standalone training")
    parser.add_argument("--train_epochs", type=int,   default=200)
    parser.add_argument("--save",         type=str,   default=None,
                        help="path to save trained weights (.pt)")
    parser.add_argument("--seed",         type=int,   default=DEFAULT_SEED)
    parser.add_argument("--device",       type=str,   default=DEFAULT_DEVICE)
    parser.add_argument("--surr_hidden",  type=int,   default=SURR_HIDDEN)
    parser.add_argument("--surr_latent",  type=int,   default=SURR_LATENT)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device : {device}")

    print(f"Loading NAS-Bench-201: {args.api_path}")
    api = load_nasbench201(args.api_path)

    surrogate = GINESurrogate201(
        hidden_channels = args.surr_hidden,
        latent_channels = args.surr_latent,
    ).to(device)

    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=True)
        state = ckpt.get("surrogate", ckpt) if isinstance(ckpt, dict) else ckpt
        surrogate.load_state_dict(state)
        print(f"Loaded weights: {args.checkpoint}")
    else:
        print("No checkpoint provided, starting standalone training …")
        train_surrogate_standalone_201(
            api         = api,
            surrogate   = surrogate,
            n_train     = args.n_train,
            epochs      = args.train_epochs,
            seed        = args.seed,
            save_path   = args.save,
        )

    res = eval_surrogate_ktau_201(api, surrogate, args.n_archs, device, args.seed)
    label = args.checkpoint if args.checkpoint else f"standalone({args.n_train}archs/{args.train_epochs}epochs)"
    print_result(label, res)
