"""D-GAENAS on the NAS-Bench-301 v1.0 surrogate benchmark.

NAS-Bench-301 predicts CIFAR-10 validation accuracy for DARTS genotypes. One
prediction is charged for every distinct architecture queried by the search.
The selected architecture is additionally reported with the deterministic
ensemble prediction; that final report is not fed back into the search and is
not counted as another query.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "core"))

from nas_graph import NASGraph
from nas_gae import (
    EdgeExistDecoder,
    EdgeFeatureDecoder,
    GINEEncoder,
    UnifiedNASGAE,
    make_noise_fn_301,
)
from nasbench301_utils import (
    CONCAT_301,
    N_OPS_301,
    Genotype301,
    genotype_hash,
    genotype_to_tensor,
    load_nasbench301,
    mutate_arch_301,
    mutate_one_arch_301,
    query_nasbench301,
    random_arch_301,
    set_seed_301,
    tensor_to_genotype,
)
from repair_ops import repair_fn_301
from surrogate_model import GINESurrogate301


DEFAULT_CFG = {
    "model_dir": "data/nb_models_1.0/xgb_v1.0",
    "eval_budget": 400,
    "n_init": 50,
    "n_runs": 10,
    "n_jobs": 1,
    "n_generations": 500,
    "pop_size": 100,
    "pretrain_gae_epochs": 10,
    "pretrain_surr_epochs": 100,
    "pretrain_elite_ratio": 0.1,
    "gae_online_epochs": 10,
    "surr_online_epochs": 50,
    "gae_elite_ratio": 0.02,
    "eval_elite_ratio": 0.02,
    "generation_multiplier": 1,
    "hidden_dim": 32,
    "latent_dim": 16,
    "dec_hidden": 16,
    "dropout": 0.1,
    "noise_prob": 0.15,
    "gae_lr": 1e-3,
    "surr_hidden": 64,
    "surr_latent": 64,
    "surr_lr": 1e-3,
    "surr_margin": 1.0,
    "query_with_noise": False,
    "mutation_prob": 0.1,
    "generation_method": "dgae",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 1800,
}


class _UnifiedGAE301Adapter:
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

    def compute_loss(self, batch: torch.Tensor):
        return self._model.compute_loss(NASGraph.from_301(batch))

    @torch.no_grad()
    def reconstruct(self, batch: torch.Tensor) -> torch.Tensor:
        bsz = batch.shape[0]
        out = self._model.reconstruct(NASGraph.from_301(batch))
        assert out.edge_exist is not None and out.edge_feat is not None
        flat = torch.cat((out.edge_exist.unsqueeze(-1), out.edge_feat), dim=-1)
        return flat.reshape(bsz, 2, 14, 1 + N_OPS_301)


def build_dgae(cfg: dict) -> _UnifiedGAE301Adapter:
    encoder = GINEEncoder(
        edge_feat_dim=N_OPS_301,
        hidden_dim=cfg["hidden_dim"],
        latent_dim=cfg["latent_dim"],
        n_nodes=6,
        dropout=cfg["dropout"],
    )
    model = UnifiedNASGAE(
        encoder=encoder,
        decoders={
            "edge_exist": EdgeExistDecoder(cfg["latent_dim"], cfg["dec_hidden"]),
            "edge_feat": EdgeFeatureDecoder(
                cfg["latent_dim"], cfg["dec_hidden"], N_OPS_301
            ),
        },
        noise_fn=make_noise_fn_301(cfg["noise_prob"]),
        repair_fn=repair_fn_301,
    )
    return _UnifiedGAE301Adapter(model)


def train_ranking(model, archive, scores, optimizer, margin, n_pairs=256):
    if len(archive) < 2:
        return 0.0
    pairs = []
    for _ in range(n_pairs * 20):
        i, j = random.randrange(len(archive)), random.randrange(len(archive))
        if scores[i] != scores[j]:
            pairs.append((i, j))
            if len(pairs) == n_pairs:
                break
    if not pairs:
        return 0.0
    device = next(model.parameters()).device
    a = torch.stack([archive[i] for i, _ in pairs]).to(device)
    b = torch.stack([archive[j] for _, j in pairs]).to(device)
    target = torch.tensor(
        [1.0 if scores[i] > scores[j] else -1.0 for i, j in pairs], device=device
    )
    model.train()
    optimizer.zero_grad()
    loss = nn.MarginRankingLoss(margin=margin)(model(a), model(b), target)
    loss.backward()
    optimizer.step()
    return float(loss.item())


@torch.no_grad()
def predict(model, candidates):
    model.eval()
    device = next(model.parameters()).device
    return model(torch.stack(candidates).to(device)).cpu().tolist()


class DGAENAS301:
    def __init__(self, benchmark, cfg):
        self.benchmark = benchmark
        self.cfg = cfg
        self.device = torch.device(cfg["device"])
        self.dgae = build_dgae(cfg).to(self.device)
        self.surrogate = GINESurrogate301(
            hidden_channels=cfg["surr_hidden"], latent_channels=cfg["surr_latent"]
        ).to(self.device)
        self.dgae_opt = optim.Adam(self.dgae.parameters(), lr=cfg["gae_lr"])
        self.surr_opt = optim.Adam(self.surrogate.parameters(), lr=cfg["surr_lr"])
        self.population = []
        self.archive = []
        self.scores = []
        self.cache = {}
        self.query_curve = []
        self.best = None

        multiplier = cfg.get("generation_multiplier", 1)
        if isinstance(multiplier, bool) or not isinstance(multiplier, int) or multiplier < 1:
            raise ValueError("generation_multiplier must be a positive integer")

    def evaluate(self, genotype):
        key = genotype_hash(genotype)
        if key in self.cache:
            return self.cache[key], False
        score = query_nasbench301(
            self.benchmark, genotype, with_noise=self.cfg["query_with_noise"]
        )
        if score is None:
            return None, False
        self.cache[key] = score
        if self.best is None or score > self.best[2]:
            self.best = (genotype_to_tensor(genotype), genotype, score)
        self.query_curve.append((len(self.cache), self.best[2]))
        return score, True

    def pretrain(self):
        while len(self.cache) < self.cfg["n_init"]:
            genotype = random_arch_301()
            score, is_new = self.evaluate(genotype)
            if score is None or not is_new:
                continue
            tensor = genotype_to_tensor(genotype)
            self.population.append((tensor, genotype, score))
            self.archive.append(tensor)
            self.scores.append(score)

        self.population.sort(key=lambda item: -item[2])
        k = max(1, math.ceil(self.cfg["pretrain_elite_ratio"] * len(self.population)))
        elite = torch.stack([item[0] for item in self.population[:k]]).to(self.device)
        self.dgae.train()
        for _ in range(self.cfg["pretrain_gae_epochs"]):
            self.dgae_opt.zero_grad()
            loss, _ = self.dgae.compute_loss(elite)
            loss.backward()
            self.dgae_opt.step()
        for _ in range(self.cfg["pretrain_surr_epochs"]):
            train_ranking(
                self.surrogate, self.archive, self.scores, self.surr_opt,
                self.cfg["surr_margin"],
            )

    def step(self, generation):
        k_gae = max(1, math.ceil(self.cfg["gae_elite_ratio"] * len(self.population)))
        k_eval = max(1, math.ceil(self.cfg["eval_elite_ratio"] * len(self.population)))
        # k_gae and k_eval above are intentionally computed from the actual
        # population size. The multiplier only expands the generated candidate
        # pool: every parent receives this many independent stochastic DGAE
        # reconstructions.
        multiplier = self.cfg.get("generation_multiplier", 1)
        n_mutated = 0
        generation_method = self.cfg.get("generation_method", "dgae")
        mutation_prob = float(self.cfg.get("mutation_prob", 0.1))
        if not 0.0 <= mutation_prob <= 1.0:
            raise ValueError("mutation_prob must be between 0 and 1")
        if generation_method == "population_random_mutation":
            generation_prob = float(
                self.cfg.get("random_generation_prob", 0.15)
            )
            if not 0.0 <= generation_prob <= 1.0:
                raise ValueError("random_generation_prob must be between 0 and 1")
            candidates = []
            for _, parent, _ in self.population:
                parent_key = genotype_hash(parent)
                for _ in range(multiplier):
                    genotype = mutate_arch_301(parent, generation_prob)
                    if random.random() < mutation_prob:
                        genotype = mutate_one_arch_301(genotype)
                    candidates.append(genotype_to_tensor(genotype))
                    n_mutated += int(genotype_hash(genotype) != parent_key)
        elif generation_method == "best_probabilistic_mutation":
            parent = self.population[0][1]
            parent_key = genotype_hash(parent)
            candidates = []
            for _ in range(len(self.population) * multiplier):
                genotype = mutate_arch_301(parent, mutation_prob)
                candidates.append(genotype_to_tensor(genotype))
                n_mutated += int(genotype_hash(genotype) != parent_key)
        elif generation_method == "dgae":
            elite = torch.stack([item[0] for item in self.population[:k_gae]]).to(self.device)
            drop_count = int(self.cfg.get("dgae_elite_drop_count", 0))
            if not 0 <= drop_count < elite.shape[0]:
                raise ValueError(
                    "dgae_elite_drop_count must be non-negative and smaller "
                    "than the number of DGAE training elites"
                )
            if drop_count:
                dropped = set(random.sample(range(elite.shape[0]), drop_count))
                kept = [i for i in range(elite.shape[0]) if i not in dropped]
                elite_train = elite[kept]
            else:
                elite_train = elite
            self.dgae.train()
            for _ in range(self.cfg["gae_online_epochs"]):
                self.dgae_opt.zero_grad()
                loss, _ = self.dgae.compute_loss(elite_train)
                loss.backward()
                self.dgae_opt.step()
            population_tensor = torch.stack(
                [item[0] for item in self.population]
            ).repeat_interleave(multiplier, dim=0).to(self.device)
            self.dgae.eval()
            generated = self.dgae.reconstruct(population_tensor).cpu()
            candidates = [generated[i] for i in range(generated.shape[0])]
            for i, candidate in enumerate(candidates):
                if random.random() < mutation_prob:
                    genotype = mutate_one_arch_301(tensor_to_genotype(candidate))
                    candidates[i] = genotype_to_tensor(genotype)
                    n_mutated += 1
        else:
            raise ValueError(f"Unknown generation_method: {generation_method}")
        ranking = np.argsort(-np.asarray(predict(self.surrogate, candidates)))

        additions = []
        for index in ranking[:k_eval]:
            if len(self.cache) >= self.cfg["eval_budget"]:
                break
            tensor = candidates[int(index)]
            genotype = tensor_to_genotype(tensor)
            score, is_new = self.evaluate(genotype)
            if score is None or not is_new:
                continue
            canonical_tensor = genotype_to_tensor(genotype)
            additions.append((canonical_tensor, genotype, score))
            self.archive.append(canonical_tensor)
            self.scores.append(score)

        for _ in range(self.cfg["surr_online_epochs"]):
            train_ranking(
                self.surrogate, self.archive, self.scores, self.surr_opt,
                self.cfg["surr_margin"],
            )
        self.population.extend(additions)
        self.population.sort(key=lambda item: -item[2])
        self.population = self.population[:self.cfg["pop_size"]]
        return {
            "generation": generation,
            "queries": len(self.cache),
            "new_queries": len(additions),
            "generated_candidates": len(candidates),
            "mutated_candidates": n_mutated,
            "best_observed_validation": self.best[2],
            "population_mean": float(np.mean([item[2] for item in self.population])),
        }

    def run(self):
        self.pretrain()
        history = []
        for generation in range(1, self.cfg["n_generations"] + 1):
            if len(self.cache) >= self.cfg["eval_budget"]:
                break
            row = self.step(generation)
            history.append(row)
            print(
                f"gen={generation:03d} queries={row['queries']:04d} "
                f"new={row['new_queries']:02d} mutated={row['mutated_candidates']:02d} "
                + f"best={row['best_observed_validation']:.4f}"
            )
            if row["new_queries"] == 0:
                print(
                    "No unseen valid candidate was produced in this generation; "
                    "continuing until the query budget or generation limit is reached."
                )
        tensor, genotype, noisy_score = self.best
        deterministic = query_nasbench301(self.benchmark, genotype, with_noise=False)
        return history, genotype, noisy_score, deterministic


def genotype_to_json(genotype: Genotype301):
    return {
        "normal": [list(edge) for edge in genotype.normal],
        "normal_concat": list(CONCAT_301),
        "reduce": [list(edge) for edge in genotype.reduce],
        "reduce_concat": list(CONCAT_301),
    }


def _run_one(run: int, benchmark, cfg: dict) -> dict:
    seed = cfg["seed"] + run
    cfg = dict(cfg, seed=seed)
    set_seed_301(seed)
    print(f"\nRun {run + 1}/{cfg['n_runs']} (seed={seed})", flush=True)
    search = DGAENAS301(benchmark, cfg)
    history, genotype, observed, deterministic = search.run()
    print(
        f"Run {run + 1} final observed={observed:.4f} "
        f"deterministic={deterministic:.4f}",
        flush=True,
    )
    return {
        "run": run + 1,
        "seed": seed,
        "queries": len(search.cache),
        "best_observed_validation": observed,
        "deterministic_validation": deterministic,
        "genotype": json.dumps(genotype_to_json(genotype)),
        "history": history,
        "query_curve": search.query_curve,
    }


_WORKER_BENCHMARK = None
_WORKER_CFG = None


def _init_worker(model_dir: str, cfg: dict) -> None:
    """Load one benchmark ensemble per persistent worker process."""
    global _WORKER_BENCHMARK, _WORKER_CFG
    _WORKER_CFG = cfg
    _WORKER_BENCHMARK = load_nasbench301(model_dir)


def _run_one_worker(run: int) -> dict:
    if _WORKER_BENCHMARK is None or _WORKER_CFG is None:
        raise RuntimeError("NAS-Bench-301 worker was not initialized")
    return _run_one(run, _WORKER_BENCHMARK, _WORKER_CFG)


def main():
    parser = argparse.ArgumentParser(description="D-GAENAS on NAS-Bench-301")
    parser.add_argument("--config", required=True, help="JSON configuration file")
    parser.add_argument("--smoke-test", action="store_true", help="use a 3-query quick run")
    parser.add_argument(
        "--n-jobs", type=int, default=None,
        help="number of independent runs executed concurrently (overrides config)",
    )
    args = parser.parse_args()
    cfg = dict(DEFAULT_CFG)
    with open(args.config, encoding="utf-8") as handle:
        cfg.update(json.load(handle))
    if args.n_jobs is not None:
        cfg["n_jobs"] = args.n_jobs
    if args.smoke_test:
        cfg.update(n_init=2, eval_budget=3, n_runs=1, n_generations=2,
                   pretrain_gae_epochs=1, pretrain_surr_epochs=1,
                   gae_online_epochs=1, surr_online_epochs=1, n_jobs=1)
    n_jobs = cfg.get("n_jobs", 1)
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or n_jobs < 1:
        raise ValueError("n_jobs must be a positive integer")
    n_jobs = min(n_jobs, cfg["n_runs"])
    cfg["n_jobs"] = n_jobs
    model_dir = Path(cfg["model_dir"])
    if not model_dir.is_absolute():
        model_dir = _PROJECT_ROOT / model_dir
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"NAS-Bench-301 model directory not found: {model_dir}\n"
            "Download and extract the official v1.0 model bundle so that "
            "this directory contains the XGBoost ensemble files."
        )
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = _PROJECT_ROOT / "results" / "nb301"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Executing {cfg['n_runs']} run(s) with n_jobs={n_jobs}")
    if n_jobs == 1:
        benchmark = load_nasbench301(str(model_dir))
        run_results = [_run_one(run, benchmark, cfg) for run in range(cfg["n_runs"])]
    else:
        # Explicit spawn is safe on Windows and avoids inheriting initialized
        # CUDA, TensorFlow, or XGBoost state from the parent process.
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=n_jobs,
            mp_context=context,
            initializer=_init_worker,
            initargs=(str(model_dir), cfg),
        ) as executor:
            run_results = list(executor.map(_run_one_worker, range(cfg["n_runs"])))

    run_results.sort(key=lambda row: row["run"])
    summary, histories, curves = [], [], []
    for result in run_results:
        histories.extend(
            {"run": result["run"], "seed": result["seed"], **row}
            for row in result.pop("history")
        )
        curves.append(result.pop("query_curve"))
        summary.append(result)

    observed_values = np.asarray(
        [row["best_observed_validation"] for row in summary], dtype=float
    )
    deterministic_values = np.asarray(
        [row["deterministic_validation"] for row in summary], dtype=float
    )
    query_values = np.asarray([row["queries"] for row in summary], dtype=float)
    aggregate = {
        "n_runs": len(summary),
        "best_observed_validation_mean": float(observed_values.mean()),
        "best_observed_validation_std": float(observed_values.std(ddof=0)),
        "deterministic_validation_mean": float(deterministic_values.mean()),
        "deterministic_validation_std": float(deterministic_values.std(ddof=0)),
        "queries_mean": float(query_values.mean()),
        "queries_std": float(query_values.std(ddof=0)),
    }

    print("\n" + "=" * 64)
    print(f"Summary over {aggregate['n_runs']} independent run(s)")
    print(
        "best observed validation : "
        f"{aggregate['best_observed_validation_mean']:.4f} "
        f"± {aggregate['best_observed_validation_std']:.4f}"
    )
    print(
        "deterministic validation : "
        f"{aggregate['deterministic_validation_mean']:.4f} "
        f"± {aggregate['deterministic_validation_std']:.4f}"
    )
    print(
        f"queries                  : {aggregate['queries_mean']:.2f} "
        f"± {aggregate['queries_std']:.2f}"
    )
    print("=" * 64)

    summary_with_stats = summary + [
        {
            "run": "mean",
            "seed": "",
            "queries": aggregate["queries_mean"],
            "best_observed_validation": aggregate["best_observed_validation_mean"],
            "deterministic_validation": aggregate["deterministic_validation_mean"],
            "genotype": "",
        },
        {
            "run": "std",
            "seed": "",
            "queries": aggregate["queries_std"],
            "best_observed_validation": aggregate["best_observed_validation_std"],
            "deterministic_validation": aggregate["deterministic_validation_std"],
            "genotype": "",
        },
    ]

    experiment_name = str(cfg.get("experiment_name", "dgaenas"))
    experiment_name = "".join(
        ch if ch.isalnum() or ch in "-_" else "_" for ch in experiment_name
    ).strip("_") or "dgaenas"
    stem = out_dir / f"{experiment_name}_nb301_{stamp}"
    with pd.ExcelWriter(str(stem) + ".xlsx", engine="openpyxl") as writer:
        pd.DataFrame(summary_with_stats).to_excel(writer, sheet_name="summary", index=False)
        pd.DataFrame(histories).to_excel(writer, sheet_name="history", index=False)
        pd.DataFrame([{"key": k, "value": str(v)} for k, v in cfg.items()]).to_excel(
            writer, sheet_name="config", index=False
        )
    np.savez(str(stem) + "_curves.npz", curves=np.array(curves, dtype=object))
    with open(str(stem) + ".json", "w", encoding="utf-8") as handle:
        json.dump({"config": cfg, "runs": summary, "aggregate": aggregate}, handle, indent=2)
    print(f"Results saved under {stem}.*")


if __name__ == "__main__":
    main()
