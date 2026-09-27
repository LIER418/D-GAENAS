
from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset


_ROOT_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SUPERNET_DIR = os.path.join(_ROOT_DIR, "SuperNet")
_CORE_DIR     = os.path.join(_ROOT_DIR, "core")
for _d in (_SUPERNET_DIR, _CORE_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from supernet_model import Network as _SuperNetwork
from generator import (
    node         as _SN_NODE,
    layer_type   as _SN_LAYER_TYPE,
    type_number  as _SN_TYPE_NUM,
    supernet_generator,
)
from utils import (
    _data_transforms_cifar10,
    AverageMeter,
    accuracy,
)


from nasbench301_utils import (
    Genotype301, PRIMITIVES_301, CONCAT_301,
    random_arch_301, genotype_hash,
    genotype_to_tensor, tensor_to_genotype,
    set_seed_301,
)
from nas_gae    import (
    GINEEncoder, EdgeExistDecoder, EdgeFeatureDecoder,
    UnifiedNASGAE, make_noise_fn_301,
)
from nas_graph    import NASGraph
from repair_ops   import repair_fn_301


_PRIM301_TO_SN: list[int] = [_SN_LAYER_TYPE.index(p) for p in PRIMITIVES_301]
_SN_TO_PRIM301: list[int] = [PRIMITIVES_301.index(op) for op in _SN_LAYER_TYPE]

assert _PRIM301_TO_SN == [1, 0, 2, 3, 4, 5, 6], (
    f"op mapping mismatch (SuperNet layer_type changed?): {_PRIM301_TO_SN}"
)


def genotype_to_supernet_masks(g: Genotype301) -> tuple[list, list]:
    n_nodes = _SN_NODE
    n_ops   = _SN_TYPE_NUM

    def _cell_to_mask(cell: list) -> list:
        mask = [
            [[0.0] * n_ops for _ in range(n_nodes + 2)]
            for _ in range(n_nodes)
        ]
        for edge_pos, (op_name, src) in enumerate(cell):
            node_i       = edge_pos // 2
            j            = src
            op_idx_301   = PRIMITIVES_301.index(op_name)
            op_idx_sn    = _PRIM301_TO_SN[op_idx_301]
            mask[node_i][j][op_idx_sn] = 1.0
        return mask

    return _cell_to_mask(g.normal), _cell_to_mask(g.reduce)


class SuperNetEvaluator:

    def __init__(
        self,
        checkpoint_path: str,
        data_dir:        str,
        device:          str       = "cuda",
        init_channels:   int       = 16,
        layers:          int       = 8,
        num_classes:     int       = 10,
        valid_batches:   int | None = None,
    ):
        self.device        = torch.device(device)
        self.valid_batches = valid_batches

        self.criterion = nn.CrossEntropyLoss().to(self.device)
        sn = supernet_generator(_SN_NODE, _SN_LAYER_TYPE)
        sr = supernet_generator(_SN_NODE, _SN_LAYER_TYPE)
        self.model = _SuperNetwork(
            sn, sr, _SN_LAYER_TYPE,
            init_channels, num_classes, layers,
            self.criterion,
            steps=len(sn),
        ).to(self.device)

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        print(f"[SuperNetEvaluator] Loaded {checkpoint_path}")

        _, valid_tf = _data_transforms_cifar10(cutout=False, cutout_length=16)
        valid_data  = dset.CIFAR10(
            root=data_dir, train=False, download=True, transform=valid_tf
        )
        self._valid_loader = torch.utils.data.DataLoader(
            valid_data, batch_size=64, shuffle=False,
            pin_memory=(device != "cpu"), num_workers=2,
        )
        self._cached_inputs:  torch.Tensor | None = None
        self._cached_targets: torch.Tensor | None = None
        n_batches = valid_batches if valid_batches is not None else len(self._valid_loader)
        print(
            f"[SuperNetEvaluator] validation set {len(valid_data)} images"
            f", using {n_batches} batches per eval"
        )

    def _ensure_cache(self) -> None:
        if self._cached_inputs is not None:
            return
        inputs_list, targets_list = [], []
        for step, (x, y) in enumerate(self._valid_loader):
            if self.valid_batches is not None and step >= self.valid_batches:
                break
            inputs_list.append(x)
            targets_list.append(y)
        self._cached_inputs  = torch.cat(inputs_list,  dim=0)
        self._cached_targets = torch.cat(targets_list, dim=0)
        print(
            f"[SuperNetEvaluator] validation cache ready: {self._cached_inputs.shape[0]} images"
        )

    def evaluate(self, genotype: Genotype301) -> float | None:
        try:
            normal_mask, reduce_mask = genotype_to_supernet_masks(genotype)
        except Exception as e:
            print(f"[SuperNetEvaluator] mask conversion failed: {e}")
            return None

        self.model.change_masks(normal_mask, reduce_mask)
        self._ensure_cache()

        top1 = AverageMeter()
        self.model.eval()
        bs = 256
        n  = self._cached_inputs.shape[0]
        with torch.no_grad():
            for start in range(0, n, bs):
                x = self._cached_inputs[start:start + bs].to(self.device)
                y = self._cached_targets[start:start + bs].to(self.device, non_blocking=True)
                logits = self.model(x)
                prec1, _ = accuracy(logits, y, topk=(1, 5))
                top1.update(prec1.item(), x.size(0))
        return top1.avg

    def evaluate_batch(self, genotypes: list) -> list:
        return [self.evaluate(g) for g in genotypes]

class _UnifiedGAE301Adapter(nn.Module):

    def __init__(self, model: UnifiedNASGAE):
        super().__init__()
        self.model = model

    def compute_loss(self, x: torch.Tensor):
        graph      = NASGraph.from_301(x)
        loss, _    = self.model.compute_loss(graph)
        with torch.no_grad():
            z = self.model.encode(graph)
        return loss, z

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor):
        B     = x.shape[0]
        graph = NASGraph.from_301(x)
        out   = self.model.reconstruct(graph)

        assert out.edge_exist is not None and out.edge_feat is not None
        act  = out.edge_exist.reshape(B, 2, 14)
        feat = out.edge_feat.reshape(B, 2, 14, -1)
        full = torch.cat([act.unsqueeze(-1), feat], dim=-1)

        return [tensor_to_genotype(full[b]) for b in range(B)]

    def train(self, mode: bool = True):
        self.model.train(mode)
        return self

    def eval(self):
        self.model.eval()
        return self

    def parameters(self, recurse: bool = True):
        return self.model.parameters(recurse)


def _build_unified_gae(cfg: dict) -> _UnifiedGAE301Adapter:
    enc = GINEEncoder(
        edge_feat_dim = 7,
        hidden_dim    = cfg["hidden_dim"],
        latent_dim    = cfg["latent_dim"],
        n_nodes       = 6,
        dropout       = cfg["dropout"],
    )
    model = UnifiedNASGAE(
        encoder  = enc,
        decoders = {
            "edge_exist": EdgeExistDecoder(cfg["latent_dim"], cfg["hidden_dim"]),
            "edge_feat" : EdgeFeatureDecoder(cfg["latent_dim"], cfg["hidden_dim"], 7),
        },
        noise_fn  = make_noise_fn_301(cfg["noise_prob"]),
        repair_fn = repair_fn_301,
    )
    return _UnifiedGAE301Adapter(model)


CFG: dict = {
    "supernet_path"  : "SuperNet/supernet-logs/latest_model.pt",
    "data_dir"       : "data",

    "init_channels"  : 48,
    "layers"         : 8,
    "num_classes"    : 10,

    "valid_batches"  : None,

    "n_init"         : 100,
    "pop_size"       : 100,
    "n_generations"  : 200,

    "pretrain_gae_epochs" : 10,
    "gae_online_epochs"   : 10,
    "gae_elite_ratio"     : 0.1,
    "gae_lr"              : 1e-3,

    "hidden_dim"      : 128,
    "latent_dim"      : 64,
    "dropout"         : 0.1,
    "noise_prob"      : 0.15,

    "early_stop_patience" : 10,
    "early_stop_tol"      : 1e-4,

    "final_eval_preset": "standard",

    "device" : "cuda" if torch.cuda.is_available() else "cpu",
    "seed"   : 42,
    "n_runs" : 1,
}


class SAEAEvolutionSuperNet:

    def __init__(
        self,
        evaluator : SuperNetEvaluator,
        gae       : _UnifiedGAE301Adapter,
        cfg       : dict,
    ):
        self.evaluator = evaluator
        self.cfg       = cfg
        self.device    = torch.device(cfg["device"])

        self.gae = gae
        self.gae.to(self.device)
        self.gae_opt = optim.Adam(gae.parameters(), lr=cfg["gae_lr"])

        self.population: list = []
        self.eval_cache: dict = {}
        self.gen:        int  = 0


    def _evaluate(self, g: Genotype301) -> float | None:
        key = genotype_hash(g)
        if key in self.eval_cache:
            return self.eval_cache[key]
        acc = self.evaluator.evaluate(g)
        if acc is not None:
            self.eval_cache[key] = acc
        return acc

    def _evaluate_batch(self, genotypes: list) -> list:
        uncached_idx = [
            i for i, g in enumerate(genotypes)
            if genotype_hash(g) not in self.eval_cache
        ]
        for i in uncached_idx:
            acc = self.evaluator.evaluate(genotypes[i])
            if acc is not None:
                self.eval_cache[genotype_hash(genotypes[i])] = acc
        return [self.eval_cache.get(genotype_hash(g)) for g in genotypes]


    def pretrain(self) -> None:
        cfg = self.cfg
        print(f"SuperNet evaluating {cfg['n_init']} initial random architectures …")

        pool = []
        while len(pool) < cfg["n_init"]:
            remaining    = cfg["n_init"] - len(pool)
            candidates_g = [random_arch_301() for _ in range(remaining)]
            accs         = self._evaluate_batch(candidates_g)
            for g, acc in zip(candidates_g, accs):
                if acc is not None:
                    pool.append((genotype_to_tensor(g), g, acc))

        pool_sorted = sorted(pool, key=lambda t: -t[2])
        k_elite     = max(1, int(cfg.get("pretrain_elite_ratio", 0.2) * len(pool)))
        gae_pool    = pool_sorted[:k_elite]
        print(f"Pre-training GAE ({cfg['pretrain_gae_epochs']} epochs, top {k_elite} elite) …")

        ops_batch = torch.stack([ind[0] for ind in gae_pool]).to(self.device)
        self.gae.train()
        loss_val = 0.0
        for _ in range(cfg["pretrain_gae_epochs"]):
            self.gae_opt.zero_grad()
            loss, _ = self.gae.compute_loss(ops_batch)
            loss.backward()
            self.gae_opt.step()
            loss_val = loss.item()
        print(f"  GAE pre-training done, final loss = {loss_val:.4f}")

        self.population = pool_sorted[:cfg["pop_size"]]
        print(f"Initial population best acc = {self.population[0][2]:.2f}%\n")


    def step(self) -> dict:
        cfg   = self.cfg
        k_gae = max(1, math.ceil(cfg["gae_elite_ratio"] * len(self.population)))

        train_inds = self.population[:k_gae]
        ops_batch  = torch.stack([ind[0] for ind in train_inds]).to(self.device)
        self.gae.train()
        gae_loss = 0.0
        for _ in range(cfg["gae_online_epochs"]):
            self.gae_opt.zero_grad()
            loss, _ = self.gae.compute_loss(ops_batch)
            loss.backward()
            self.gae_opt.step()
            gae_loss = loss.item()

        self.gae.eval()
        ops_pop   = torch.stack([ind[0] for ind in self.population]).to(self.device)
        new_genos = self.gae.reconstruct(ops_pop)

        is_new = [genotype_hash(g) not in self.eval_cache for g in new_genos]
        accs   = self._evaluate_batch(new_genos)

        new_evals   = []
        n_truly_new = 0
        for g, new, acc in zip(new_genos, is_new, accs):
            if acc is None:
                continue
            new_evals.append((genotype_to_tensor(g).cpu(), g, acc))
            if new:
                n_truly_new += 1

        combined = self.population + new_evals
        combined.sort(key=lambda t: -t[2])
        self.population = combined[:cfg["pop_size"]]

        self.gen += 1
        fitnesses = [t[2] for t in self.population]
        return {
            "gen"          : self.gen,
            "best_fitness" : fitnesses[0],
            "mean_fitness" : sum(fitnesses) / len(fitnesses),
            "n_new_eval"   : n_truly_new,
            "cache_size"   : len(self.eval_cache),
            "gae_loss"     : gae_loss,
        }


    def run(self, n_generations: int) -> list:
        patience     = int(self.cfg.get("early_stop_patience", 0))
        tol          = float(self.cfg.get("early_stop_tol", 1e-4))
        history      = []
        converge_cnt = 0
        prev_mean    = None

        for _ in range(n_generations):
            stats = self.step()
            history.append(stats)
            gae_str = f"{stats['gae_loss']:.4f}"
            print(
                f"  Gen {stats['gen']:4d}/{n_generations}"
                f" | best={stats['best_fitness']:.2f}%"
                f" | mean={stats['mean_fitness']:.2f}%"
                f" | GAE={gae_str}"
                f" | new_evals={stats['n_new_eval']}"
                f" | cache={stats['cache_size']}"
            )

            if patience > 0:
                mean = stats["mean_fitness"]
                if prev_mean is not None and abs(mean - prev_mean) < tol:
                    converge_cnt += 1
                    if converge_cnt >= patience:
                        print(
                            f"  [Early Stop] Population mean unchanged for {converge_cnt} generations (< {tol}), "
                            f"terminating (Gen {stats['gen']})"
                        )
                        break
                else:
                    converge_cnt = 0
                prev_mean = mean

        return history

    @property
    def best(self) -> tuple:
        return self.population[0]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SAEA + SuperNet direct evaluation")
    p.add_argument("--supernet_path", default=CFG["supernet_path"])
    p.add_argument("--data_dir",      default=CFG["data_dir"])
    p.add_argument("--device",        default=CFG["device"])
    p.add_argument("--valid_batches", type=int,  default=CFG["valid_batches"])
    p.add_argument("--n_init",        type=int,  default=CFG["n_init"])
    p.add_argument("--pop_size",      type=int,  default=CFG["pop_size"])
    p.add_argument("--n_generations", type=int,  default=CFG["n_generations"])
    p.add_argument("--n_runs",        type=int,  default=CFG["n_runs"])
    p.add_argument("--seed",          type=int,  default=CFG["seed"])
    return p.parse_args()


def _geno_str(g: Genotype301) -> tuple[str, str]:
    normal_str = " | ".join(f"{op}←{src}" for op, src in g.normal)
    reduce_str = " | ".join(f"{op}←{src}" for op, src in g.reduce)
    return normal_str, reduce_str


if __name__ == "__main__":
    args = _parse_args()

    CFG.update({
        "supernet_path" : args.supernet_path,
        "data_dir"      : args.data_dir,
        "device"        : args.device,
        "valid_batches" : args.valid_batches,
        "n_init"        : args.n_init,
        "pop_size"      : args.pop_size,
        "n_generations" : args.n_generations,
        "n_runs"        : args.n_runs,
        "seed"          : args.seed,
    })

    print(f"SAEA + SuperNet direct evaluation — DARTS search space")
    print(f"  supernet weights : {CFG['supernet_path']}")
    print(f"  valid_batches={CFG['valid_batches']}  pop_size={CFG['pop_size']}"
          f"  n_init={CFG['n_init']}  n_gen={CFG['n_generations']}"
          f"  device={CFG['device']}\n")

    evaluator = SuperNetEvaluator(
        checkpoint_path = CFG["supernet_path"],
        data_dir        = CFG["data_dir"],
        device          = CFG["device"],
        init_channels   = CFG["init_channels"],
        layers          = CFG["layers"],
        num_classes     = CFG["num_classes"],
        valid_batches   = CFG["valid_batches"],
    )

    all_best       = []
    all_cache      = []
    all_histories  = []
    all_best_genos = []

    for run in range(CFG["n_runs"]):
        seed = CFG["seed"] + run
        set_seed_301(seed)
        print(f"{'─' * 60}")
        print(f"  Run {run + 1}/{CFG['n_runs']}  (seed={seed})")
        print(f"{'─' * 60}")

        gae = _build_unified_gae(CFG)

        saea = SAEAEvolutionSuperNet(evaluator=evaluator, gae=gae, cfg=CFG)

        _search_start = time.time()
        print("=== Pre-training ===")
        saea.pretrain()
        print(f"=== Evolution ({CFG['n_generations']} generations) ===\n")
        history = saea.run(n_generations=CFG["n_generations"])

        best_t, best_g, best_acc = saea.best
        _acc_tag = f"{CFG['valid_batches']} batch"

        _search_sec = time.time() - _search_start
        print(f"Search time: {_search_sec:.1f} s  ({_search_sec / 86400:.4f} GPU-days)")
        all_histories.append(history)

        n_evals = len(saea.eval_cache)
        all_best.append(best_acc)
        all_cache.append(n_evals)
        all_best_genos.append(best_g)
        print(f"  Run {run + 1} — best={best_acc:.2f}% ({_acc_tag})  total_evals={n_evals}")
        n_str, r_str = _geno_str(best_g)
        print(f"  Normal : {n_str}")
        print(f"  Reduce : {r_str}\n")

    best_arr  = np.array(all_best)
    cache_arr = np.array(all_cache)

    print(f"\n{'═' * 60}")
    print(f"  Summary ({CFG['n_runs']} runs)")
    print(f"{'═' * 60}")
    for i, (b, c) in enumerate(zip(all_best, all_cache)):
        print(f"  Run {i + 1:2d}: best={b:.2f}%  n_evals={c}")
    print(f"{'─' * 60}")
    print(f"  best  mean={best_arr.mean():.2f}%  std={best_arr.std():.2f}%"
          f"  max={best_arr.max():.2f}%")
    print(f"  evals mean={cache_arr.mean():.1f}  total={cache_arr.sum()}")
    print(f"{'═' * 60}")

    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    xlsx_path = f"saea_supernet_results_{ts}.xlsx"

    history_rows = []
    for run_idx, hist in enumerate(all_histories):
        for row in hist:
            history_rows.append({"run": run_idx + 1, "seed": CFG["seed"] + run_idx, **row})

    summary_rows = []
    for i, (b, c, g) in enumerate(zip(all_best, all_cache, all_best_genos)):
        n_str, r_str = _geno_str(g)
        summary_rows.append({
            "run"          : i + 1,
            "seed"         : CFG["seed"] + i,
            "best_acc"     : b,
            "n_evals"      : c,
            "normal_cell"  : n_str,
            "reduce_cell"  : r_str,
        })

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        pd.DataFrame(history_rows).to_excel(writer, sheet_name="history",  index=False)
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="summary",  index=False)
        pd.DataFrame([CFG]).to_excel(writer,         sheet_name="config",   index=False)

    print(f"\nResults saved to {xlsx_path}")

    if CFG.get("final_eval_preset") is not None:
        _overall_best_g = all_best_genos[int(np.argmax(all_best))]
        print(f"\n{'═'*60}")
        print(f"  Final standard evaluation (darts-master/cnn/train.py protocol)")
        n_str, r_str = _geno_str(_overall_best_g)
        print(f"  Normal : {n_str}")
        print(f"  Reduce : {r_str}")
        print(f"{'═'*60}")

        _geno_name   = "SAEA_SUPERNET_BEST"
        _geno_normal = list(_overall_best_g.normal)
        _geno_reduce = list(_overall_best_g.reduce)
        _geno_line   = (
            f"\n{_geno_name} = Genotype("
            f"normal={_geno_normal}, normal_concat={list(_overall_best_g.normal_concat)}, "
            f"reduce={_geno_reduce}, reduce_concat={list(_overall_best_g.reduce_concat)})\n"
        )
        _BASE      = os.path.dirname(os.path.abspath(__file__))
        _geno_file = os.path.join(_BASE, "darts-master", "cnn", "genotypes.py")
        with open(_geno_file, "r") as _f:
            _lines = [l for l in _f if not l.startswith(f"{_geno_name} =")]
        with open(_geno_file, "w") as _f:
            _f.writelines(_lines)
            _f.write(_geno_line)
        print(f"  Genotype '{_geno_name}' written to {_geno_file}")

        _darts_cnn = os.path.join(_BASE, "darts-master", "cnn")
        _data_dir  = os.path.abspath(CFG.get("data_dir", "data"))
        print(f"  Launching darts-master/cnn/train.py …")
        os.system(
            f'cd /d "{_darts_cnn}" && python train.py'
            f' --data "{_data_dir}"'
            f' --arch {_geno_name}'
            f' --auxiliary'
            f' --cutout'
        )
        print(f"  Training complete, results in darts-master/cnn/eval-EXP-*/")
