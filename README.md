# D-GAENAS

Official experiment code for **Denoising Graph Autoencoder-Based Evolutionary Neural Architecture Search**.

D-GAENAS trains a denoising graph autoencoder (DGAE) online on elite architectures and uses reconstruction as a learned, structure-aware generation operation. The repository contains the experiments reported for NAS-Bench-101, NAS-Bench-201, and the DARTS open-domain search space, together with the principal ablations and plotting utilities.

## Repository layout

```text
.
|-- configs/                 Paper-aligned JSON configurations
|-- core/                    DGAE, graph representations, repair, mutation, surrogate
|-- darts/                   DARTS evaluation network utilities
|-- SuperNet/                Weight-sharing supernet used by open-domain search
|-- scripts/                 Search, ablation, baseline, evaluation, and plotting entry points
|-- data/                    Place downloaded benchmarks here (not tracked)
|-- results/                 Location for generated or exported curve data
|-- pyproject.toml
`-- README.md
```

Generated logs, downloaded datasets, checkpoints, build products, caches, and large intermediate files are intentionally excluded.

## Installation

Python 3.10 or newer is required. Create an isolated environment, then install the project from the repository root:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -e .
```

For NAS-Bench-101 experiments, install the optional dependencies and the official NAS-Bench package:

```bash
python -m pip install -e ".[nb101]"
python -m pip install "git+https://github.com/google-research/nasbench.git"
```

GPU-enabled PyTorch should be installed using the command appropriate for the local CUDA version before installing this project.

## Benchmark data

The benchmark files are not redistributed. Download them from their official projects and place them as follows:

```text
data/
|-- nasbench_only108.tfrecord
`-- NATS-tss-v1_0-3ffb9.pickle.pbz2
```

- NAS-Bench-101: <https://github.com/google-research/nasbench>
- NATS-Bench / NAS-Bench-201: <https://github.com/D-X-Y/NATS-Bench>

CIFAR-10 is downloaded automatically by the supernet training script unless `--no_download` is specified.

## Reproducing the tabular benchmark experiments

Run commands from the repository root. The JSON files override the in-script defaults and correspond to the budget-dependent settings reported in the paper.

### NAS-Bench-101

```bash
python scripts/saea_gae_nb101.py --config configs/nb101_budget150.json
python scripts/saea_gae_nb101.py --config configs/nb101_budget400.json
python scripts/saea_gae_nb101.py --config configs/nb101_budget1000.json
```

### NAS-Bench-201

Each supplied configuration runs CIFAR-10, CIFAR-100, and ImageNet-16-120.

```bash
python scripts/saea_gae_nb201.py --config configs/nb201_budget100.json
python scripts/saea_gae_nb201.py --config configs/nb201_budget200.json
python scripts/saea_gae_nb201.py --config configs/nb201_budget400.json
```

The main scripts save timestamped Excel workbooks and curve files in the current working directory. Each workbook records the effective configuration and per-run results.

## Open-domain DARTS search

Train the single-path weight-sharing supernet first:

```bash
cd SuperNet
python train.py --dataset CIFAR10 --epochs 500 --cuda 0 --data_dir ../data
cd ..
```

This produces `SuperNet/supernet-logs/latest_model.pt`. Then run the search:

```bash
python scripts/saea_gae_openspace.py \
  --supernet_path SuperNet/supernet-logs/latest_model.pt \
  --data_dir data \
  --n_init 100 \
  --pop_size 100 \
  --n_generations 200 \
  --n_runs 1
```

The final genotype can be retrained with the standard DARTS evaluation pipeline in `darts/`.

## Ablations, baselines, and figures

Search-mechanism ablation and NAS-Bench-101 baselines:

```bash
python scripts/ablation_nb101.py
python scripts/baselines_nb101.py
```

DGAE versus standard-GAE curve using exported 30-run curve data (see
`results/README.md` for the expected filenames):

```bash
python scripts/plot_dgae_vs_gae.py \
  --dgae results/dgae_nb101_curves.npz \
  --gae results/gae_nb101_curves.npz \
  --out results/dgae_vs_gae.png
```

Population-distribution visualization:

```bash
python scripts/viz_population_1000.py \
  --tfrecord data/nasbench_only108.tfrecord \
  --budget 1000 \
  --out results
```

Surrogate Kendall correlation:

```bash
python scripts/eval_ktau_surrogate.py --data_file data/nasbench_only108.tfrecord
python scripts/eval_ktau_surrogate_201.py --api_path data/NATS-tss-v1_0-3ffb9.pickle.pbz2
```

Use `python <script> --help` to inspect the options exposed by a utility.

## Reproducibility notes

- Search selection uses validation accuracy; test accuracy is reported only for the selected incumbent architecture.
- The reported tabular benchmark results use 30 independent runs.
- Seeds are derived deterministically from the base seed recorded in each configuration.
- GPU-day measurements are hardware- and implementation-dependent. The open-domain number reported in the paper was measured on one NVIDIA RTX 5090 GPU and excludes final-architecture retraining.
- Large benchmark data and the trained supernet checkpoint are excluded because of their size and third-party distribution terms.

## Citation

The final BibTeX citation will be added after publication.

## License

This project is released under the [MIT License](LICENSE).
