"""Significance tests for the final NB101 ablation runs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "nb101"

XLSX = {
    "D-GAENAS": RESULTS / "dgaenas" / "dgaenas_nb101_20261111_140431.xlsx",
    "Random mutation": RESULTS / "random_mutation" / "random_mutation_ablation_nb101_20261111_141128.xlsx",
    "Best-centered mutation": RESULTS / "best_parent_mutation" / "best_parent_mutation_ablation_nb101_20261111_144417.xlsx",
    "GAE": RESULTS / "gae" / "gae_ablation_nb101_20261111_142450.xlsx",
    "Full-population DGAE": RESULTS / "full_population" / "full_population_ablation_nb101_20261111_143604.xlsx",
    "GA-dA": RESULTS / "ga_da" / "ga_da_vector_ablation_nb101_20261111_145726.xlsx",
}

DAE_EDA = RESULTS / "dae_eda"


def load_xlsx(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name="summary")
    aliases = {
        "run": ("run",),
        "seed": ("seed",),
        "validation": ("val_acc", "validation", "best_validation"),
        "test": ("test_acc", "test", "best_test_of_best_validation"),
    }
    output = {}
    for target, candidates in aliases.items():
        source = next((column for column in candidates if column in frame.columns), None)
        if source is None:
            raise KeyError(f"{path}: none of {candidates} found; columns={list(frame.columns)}")
        output[target] = frame[source]
    result = pd.DataFrame(output)
    result["seed"] = pd.to_numeric(result["seed"], errors="coerce")
    result = result.dropna(subset=["seed", "validation", "test"]).copy()
    result["seed"] = result["seed"].astype(int)
    if result["seed"].duplicated().any():
        raise ValueError(f"{path}: duplicate run seeds in summary sheet")
    return result.sort_values("seed").reset_index(drop=True)


def load_dae_eda(path: Path) -> pd.DataFrame:
    rows = []
    for file in sorted(path.glob("*.json")):
        data = json.loads(file.read_text(encoding="utf-8"))
        rows.append({
            "run": len(rows),
            "seed": int(data["seed"]),
            "validation": float(data["best_validation"]),
            "test": float(data["best_test_of_best_validation"]),
        })
    return pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)


def holm_adjust(p_values: list[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    count = len(p)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * p[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def compare(reference: pd.DataFrame, candidate: pd.DataFrame, metric: str) -> dict:
    paired = reference[["seed", metric]].merge(
        candidate[["seed", metric]], on="seed", suffixes=("_ref", "_candidate")
    )
    x = paired[f"{metric}_ref"].to_numpy(float)
    y = paired[f"{metric}_candidate"].to_numpy(float)
    difference = x - y
    paired_t = stats.ttest_rel(x, y)
    welch = stats.ttest_ind(x, y, equal_var=False)
    mann_whitney = stats.mannwhitneyu(x, y, alternative="two-sided")
    try:
        wilcoxon = stats.wilcoxon(x, y, alternative="two-sided", zero_method="wilcox")
        wilcoxon_p = float(wilcoxon.pvalue)
    except ValueError:
        wilcoxon_p = 1.0
    dz = float(difference.mean() / difference.std(ddof=1)) if difference.std(ddof=1) else np.inf
    return {
        "n": len(paired),
        "reference_mean": x.mean(),
        "reference_std": x.std(ddof=1),
        "candidate_mean": y.mean(),
        "candidate_std": y.std(ddof=1),
        "mean_difference": difference.mean(),
        "paired_t_p": float(paired_t.pvalue),
        "wilcoxon_p": wilcoxon_p,
        "welch_p": float(welch.pvalue),
        "mann_whitney_p": float(mann_whitney.pvalue),
        "cohen_dz": dz,
    }


def main() -> None:
    data = {name: load_xlsx(path) for name, path in XLSX.items()}
    data["DAE-EDA"] = load_dae_eda(DAE_EDA)
    reference = data.pop("D-GAENAS")

    for metric in ("validation", "test"):
        rows = []
        for name, candidate in data.items():
            rows.append({"variant": name, **compare(reference, candidate, metric)})
        # D-GAENAS-derived variants share the same initialization process and
        # seeds, whereas the independently implemented DAE-EDA does not form
        # meaningful per-seed pairs.  Use the corresponding non-parametric
        # test for each comparison, then correct the complete test family.
        for row in rows:
            row["primary_test"] = (
                "Mann-Whitney U" if row["variant"] == "DAE-EDA"
                else "paired Wilcoxon"
            )
            row["primary_p"] = (
                row["mann_whitney_p"] if row["variant"] == "DAE-EDA"
                else row["wilcoxon_p"]
            )
        adjusted = holm_adjust([row["paired_t_p"] for row in rows])
        adjusted_w = holm_adjust([row["primary_p"] for row in rows])
        for row, p_t, p_w in zip(rows, adjusted, adjusted_w):
            row["paired_t_holm"] = p_t
            row["primary_holm"] = p_w
        frame = pd.DataFrame(rows)
        print(f"\n=== {metric.upper()} ===")
        print(frame.to_string(index=False, float_format=lambda value: f"{value:.8g}"))


if __name__ == "__main__":
    main()
