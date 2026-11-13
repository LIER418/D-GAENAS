# D-GAENAS

Official experiment code for **Denoising Graph Autoencoder-Based Evolutionary Neural Architecture Search**.

D-GAENAS trains a denoising graph autoencoder online on elite architectures and applies graph reconstruction as a learned generation operator. This release is limited to the experiments reported in the paper: NAS-Bench-101, NAS-Bench-201, NAS-Bench-301, the NAS-Bench-101 component ablations, and the population-distribution analysis.

## Layout

```text
.
|-- baselines/              DAE-EDA adaptation used in the paper
|-- configs/                Paper-aligned experiment configurations
|-- core/                   DGAE, graph, mutation, repair, and surrogate modules
|-- data/                    Downloaded benchmark data (ignored by Git)
|-- results/                 Paper result records and compact curve data
|-- scripts/                 Search, analysis, and plotting entry points
|-- LICENSE
|-- pyproject.toml
`-- README.md
```

## Installation

Python 3.10 or newer is recommended. Install GPU-enabled PyTorch separately when required, then run:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install "git+https://github.com/google-research/nasbench.git"
```

For NAS-Bench-301, also install the official API version compatible with the released v1.0 surrogate models.

## Benchmark data

The benchmark files are not redistributed. Download them from their official projects and place them at:

```text
data/
|-- nasbench_only108.tfrecord
|-- NATS-tss-v1_0-3ffb9.pickle.pbz2
`-- nb_models_1.0/
    `-- xgb_v1.0/
```

- NAS-Bench-101 data: <https://storage.googleapis.com/nasbench/nasbench_only108.tfrecord>
- NATS-Bench topology data: <https://drive.google.com/file/d/1vzyK0UVH2D3fTpa1_dSWnp1gvGpAxRul/view?usp=sharing>
- NAS-Bench-301 v1.0 models: <https://figshare.com/articles/software/nasbench301_models_v1_0_zip/13061510>

See [`data/README.md`](data/README.md) for exact extraction and placement instructions.

## Main experiments

Run commands from the repository root. Each supplied configuration performs the 10 independent runs reported in the paper and writes its outputs below `results/`.

```bash
python scripts/saea_gae_nb101.py --config configs/nb101_budget400.json
python scripts/saea_gae_nb201.py --config configs/nb201_budget192.json
python scripts/saea_gae_nb301.py --config configs/nb301_budget192.json
```

Use `--n-jobs N` to override the number of concurrently executed repetitions.

## NAS-Bench-101 ablations

```bash
python scripts/saea_gae_nb101.py --config configs/ablation_random_mutation_nb101_budget400.json
python scripts/saea_gae_nb101.py --config configs/ablation_best_parent_mutation_nb101_budget400.json
python scripts/saea_gae_nb101.py --config configs/ablation_gae_nb101_budget400.json
python scripts/saea_gae_nb101.py --config configs/ablation_full_population_nb101_budget400.json
python scripts/saea_gae_nb101.py --config configs/ablation_ga_da_nb101_budget400.json
python scripts/run_dae_eda_nb101.py --config configs/dae_eda_nb101_budget400.json
```

The DAE-EDA code is an independent NAS adaptation of the released DAE-EDA procedure, not an official NAS implementation by its original authors.

## Paper analyses

Generate the component-ablation plot using the retained curve files:

```bash
python scripts/plot_nb101_ablation_curves.py \
  --original "results/nb101/dgaenas/dgaenas_nb101_20261111_140431_curves.npz" \
  --gae "results/nb101/gae/gae_ablation_nb101_20261111_142450_curves.npz" \
  --full-population "results/nb101/full_population/full_population_ablation_nb101_20261111_143604_curves.npz" \
  --random-mutation "results/nb101/random_mutation/random_mutation_ablation_nb101_20261111_141128_curves.npz" \
  --best-parent-mutation "results/nb101/best_parent_mutation/best_parent_mutation_ablation_nb101_20261111_144417_curves.npz" \
  --vector-dae "results/nb101/ga_da/ga_da_vector_ablation_nb101_20261111_145726_curves.npz" \
  --dae-eda "results/nb101/dae_eda/dae_eda_nb101_b400_dense_curves.npz" \
  --metric validation --n-queries 400 --y-min 0.94 --y-max 0.951 \
  --out "results/nb101/nb101_ablation.png"
```

Run the statistical tests with:

```bash
python scripts/analyze_nb101_ablation_significance.py
```

Generate the population-distribution analysis with:

```bash
python scripts/viz_population_1000.py \
  --config configs/nb101_budget400.json \
  --mutation-config configs/ablation_random_mutation_nb101_budget400.json \
  --tfrecord data/nasbench_only108.tfrecord \
  --budget 400 \
  --out results/nb101/population_distribution.png
```

## Reproducibility notes

- Validation accuracy is used for search selection. Test accuracy is reported only for the selected NAS-Bench-101/201 incumbent.
- NAS-Bench-301 reports the deterministic validation prediction of the official v1.0 surrogate ensemble.
- All D-GAENAS and reproduced-algorithm results reported in the paper use 10 independent runs.
- Downloaded benchmarks and generated caches are excluded from version control.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
