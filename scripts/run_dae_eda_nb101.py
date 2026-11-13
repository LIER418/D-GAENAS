"""Run the independent DAE-EDA adaptation used in the NB101 ablation."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.dae_eda.adapted_models import DAEConfig, run_dae_eda
from baselines.dae_eda.nb101 import Oracle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = ROOT / "results" / "nb101" / "dae_eda"
    output_dir.mkdir(parents=True, exist_ok=True)
    tfrecord = Path(cfg["tfrecord"])
    if not tfrecord.is_absolute():
        tfrecord = ROOT / tfrecord

    for run in range(int(cfg["n_runs"])):
        seed = int(cfg["seed"]) + run
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        oracle = Oracle(int(cfg["budget"]), tfrecord)
        dae_cfg = DAEConfig()
        result = run_dae_eda(oracle, random.Random(seed), dae_cfg)
        payload = {
            "method": "dae_eda_nb101",
            "seed": seed,
            "budget": int(cfg["budget"]),
            "config": asdict(dae_cfg),
            "query_count": len(oracle.events),
            "best_validation": oracle.events[-1]["best_validation"],
            "best_test_of_best_validation": oracle.events[-1]["best_test_of_best_validation"],
            "events": oracle.events,
            **result,
        }
        output = output_dir / f"dae_eda_nb101_400_{seed}.json"
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(
            f"run={run + 1}/{cfg['n_runs']} seed={seed} "
            f"validation={payload['best_validation']:.6f} "
            f"test={payload['best_test_of_best_validation']:.6f}"
        )


if __name__ == "__main__":
    main()
