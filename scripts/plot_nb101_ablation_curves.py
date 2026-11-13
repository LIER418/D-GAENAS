"""Plot NB101 D-GAENAS ablation and DAE-EDA trajectories from NPZ files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


def clean_cli_path(path: str) -> str:
    """Remove control characters accidentally introduced by shell line wrapping."""
    return path.replace("\r", "").replace("\n", "").strip()


def _validation_curves_from_xlsx(npz_path: str) -> list[list[tuple[int, float]]]:
    path = Path(npz_path)
    suffix = "_curves.npz"
    if not path.name.endswith(suffix):
        raise KeyError(f"Cannot infer the matching XLSX file from {path}")
    xlsx = path.with_name(path.name[:-len(suffix)] + ".xlsx")
    if not xlsx.exists():
        raise KeyError(
            f"{path} has no saved validation curves and matching file {xlsx} is absent"
        )
    history = __import__("pandas").read_excel(xlsx, sheet_name="history")
    required = {"run", "archive_size", "best_fitness"}
    if not required.issubset(history.columns):
        raise KeyError(f"{xlsx} history sheet lacks columns {sorted(required)}")
    curves = []
    for _, frame in history.groupby("run", sort=True):
        points = []
        for _, row in frame.sort_values(["archive_size", "generation"]).iterrows():
            point = (int(row["archive_size"]), float(row["best_fitness"]))
            if points and points[-1][0] == point[0]:
                points[-1] = point
            else:
                points.append(point)
        curves.append(points)
    return curves


def load_curves(path: str, metric: str) -> list[list[tuple[int, float]]]:
    path = clean_cli_path(path)
    with np.load(path, allow_pickle=True) as data:
        if metric == "validation":
            for key in ("validation_curves", "best_validation_curves"):
                if key in data:
                    raw = data[key]
                    break
            else:
                return _validation_curves_from_xlsx(path)
        else:
            if "saea_curves" not in data:
                raise KeyError(f"{path} does not contain 'saea_curves'; keys={data.files}")
            raw = data["saea_curves"]
    return [
        [(int(point[0]), float(point[1])) for point in run]
        for run in raw
    ]


def to_matrix(curves, n_queries: int) -> np.ndarray:
    matrix = []
    for curve in curves:
        values = np.full(n_queries, np.nan, dtype=float)
        index = 0
        incumbent = np.nan
        for query in range(1, n_queries + 1):
            while index < len(curve) and curve[index][0] <= query:
                incumbent = curve[index][1]
                index += 1
            values[query - 1] = incumbent
        matrix.append(values)
    return np.asarray(matrix, dtype=float)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True)
    parser.add_argument("--gae", required=True)
    parser.add_argument("--full-population", required=True)
    parser.add_argument("--random-mutation", required=True)
    parser.add_argument("--dae-eda", default=None)
    parser.add_argument("--vector-dae", default=None)
    parser.add_argument("--best-parent-mutation", default=None)
    parser.add_argument("--n-queries", type=int, default=400)
    parser.add_argument(
        "--initial-queries", type=int, default=1,
        help=("First query displayed in the curve (default: 1). This option is "
              "provided only for optional visualization cropping."),
    )
    parser.add_argument("--y-min", type=float, default=0.90)
    parser.add_argument("--y-max", type=float, default=0.944)
    parser.add_argument("--out", required=True)
    parser.add_argument("--metric", choices=["test", "validation"], default="test")
    args = parser.parse_args()

    specifications = [
        (args.original, "D-GAENAS", "#1f77b4", "-", 2.5),
        (args.gae, "GAE (without denoising)", "#d62728", "--", 2.0),
        (args.full_population, "Full-population DGAE training", "#2ca02c", "-.", 2.0),
        (args.random_mutation, "Random-mutation generation", "#9467bd", ":", 2.3),
    ]
    if args.dae_eda:
        specifications.append(
            (args.dae_eda, "DAE-EDA (adapted)", "#ff7f0e", (0, (5, 2)), 2.0)
        )
    if args.vector_dae:
        specifications.append(
            (args.vector_dae, "Vector DAE (GA-dA-style)", "#8c564b", (0, (3, 1)), 2.0)
        )
    if args.best_parent_mutation:
        specifications.append(
            (args.best_parent_mutation, "Best-parent mutation", "#e377c2", (0, (1, 1)), 2.0)
        )
    if not 1 <= args.initial_queries <= args.n_queries:
        parser.error("--initial-queries must be between 1 and --n-queries")

    fig, axis = plt.subplots(figsize=(9.2, 5.5))
    # Preserve the per-query incumbent trajectory by default.  Cropping is an
    # optional visualization choice and does not alter the underlying curves.
    x = np.arange(args.initial_queries, args.n_queries + 1)
    for path, label, color, linestyle, width in specifications:
        path = clean_cli_path(path)
        curves = load_curves(path, args.metric)
        matrix = to_matrix(curves, args.n_queries)[:, args.initial_queries - 1:]
        mean = np.nanmean(matrix, axis=0)
        std = np.nanstd(matrix, axis=0, ddof=0)
        axis.plot(
            x, mean, label=label, color=color, linestyle=linestyle,
            linewidth=width,
        )
        axis.fill_between(
            x, mean - std, mean + std, color=color, alpha=0.10, linewidth=0,
        )
        print(
            f"{label}: runs={len(curves)}, "
            f"final={mean[-1]:.6f} +/- {std[-1]:.6f}"
        )

    axis.set_xlim(args.initial_queries, args.n_queries)
    axis.set_ylim(args.y_min, args.y_max)
    axis.set_xlabel("Number of ground-truth evaluations", fontsize=14)
    ylabel = "Best validation accuracy (%)" if args.metric == "validation" else "Best test accuracy (%)"
    axis.set_ylabel(ylabel, fontsize=14)
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
    axis.tick_params(axis="both", labelsize=12)
    axis.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    axis.legend(fontsize=9.5, loc="upper left", ncol=2)
    fig.tight_layout()

    output = Path(clean_cli_path(args.out))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to: {output}")


if __name__ == "__main__":
    main()
