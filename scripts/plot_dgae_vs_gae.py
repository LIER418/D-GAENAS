import os, sys, argparse
import numpy as np
import matplotlib.pyplot as plt


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


def load_curves(path: str) -> list:
    d = np.load(path, allow_pickle=True)
    raw = d["saea_curves"]
    return [[(int(p[0]), float(p[1])) for p in run] for run in raw]


def plot(dgae_curves, gae_curves, save_path, n_queries,
         errorbar_every=50, y_min=0.930, y_max=0.944):
    x_grid = np.arange(1, n_queries + 1)
    eb_idx = np.arange(errorbar_every - 1, n_queries, errorbar_every)

    fig, ax = plt.subplots(figsize=(9, 5))

    for curves, label, color, lw, zorder in [
        (dgae_curves, "D-GAENAS (DGAE)", "#1f77b4", 2.2, 5),
        (gae_curves,  "D-GAENAS (GAE)",  "#d62728", 2.0, 4),
    ]:
        mat  = _to_matrix(curves, n_queries)
        mean = mat.mean(axis=0)
        std  = mat.std(axis=0)
        ax.plot(x_grid, mean, color=color, linewidth=lw, label=label, zorder=zorder)
        ax.errorbar(x_grid[eb_idx], mean[eb_idx], yerr=std[eb_idx],
                    fmt="none", ecolor=color, elinewidth=1.3,
                    capsize=3, capthick=1.3, zorder=zorder)

    ax.set_xlim(0, n_queries)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("# Queries", fontsize=22)
    ax.set_ylabel("Test Accuracy", fontsize=22)
    ax.tick_params(axis="both", labelsize=20)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(fontsize=20, loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dgae", required=True, help="DGAE curves .npz")
    parser.add_argument("--gae",  required=True, help="GAE curves .npz")
    parser.add_argument("--out",  default=None,  help="output png path")
    parser.add_argument("--n_queries", type=int, default=1000)
    parser.add_argument("--y_min",     type=float, default=0.930)
    parser.add_argument("--y_max",     type=float, default=0.944)
    args = parser.parse_args()

    dgae_curves = load_curves(args.dgae)
    gae_curves  = load_curves(args.gae)

    print(f"DGAE runs: {len(dgae_curves)}  GAE runs: {len(gae_curves)}")

    save_path = args.out or f"dgae_vs_gae_comparison.png"
    plot(dgae_curves, gae_curves, save_path,
         n_queries=args.n_queries, y_min=args.y_min, y_max=args.y_max)
