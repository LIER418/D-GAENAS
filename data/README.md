# Benchmark data

No third-party benchmark data are distributed with this repository. Download
the following files from their official sources and place them exactly as
shown:

```text
data/
|-- nasbench_only108.tfrecord
|-- NATS-tss-v1_0-3ffb9.pickle.pbz2
`-- nb_models_1.0/
    `-- xgb_v1.0/
```

## NAS-Bench-101

- Required file: `nasbench_only108.tfrecord`
- Direct download: <https://storage.googleapis.com/nasbench/nasbench_only108.tfrecord>
- Official repository: <https://github.com/google-research/nasbench>

PowerShell example:

```powershell
Invoke-WebRequest https://storage.googleapis.com/nasbench/nasbench_only108.tfrecord -OutFile data/nasbench_only108.tfrecord
```

## NAS-Bench-201 / NATS-Bench

- Required file: `NATS-tss-v1_0-3ffb9.pickle.pbz2`
- Official Google Drive file: <https://drive.google.com/file/d/1vzyK0UVH2D3fTpa1_dSWnp1gvGpAxRul/view?usp=sharing>
- Official repository and mirrors: <https://github.com/D-X-Y/NATS-Bench#preparation-and-download>

Download the topology-search-space benchmark file, retain its original file
name, and place it directly under `data/`.

## NAS-Bench-301

- Required model directory: `nb_models_1.0/xgb_v1.0/`
- Official v1.0 model bundle: <https://figshare.com/articles/software/nasbench301_models_v1_0_zip/13061510>
- Official repository: <https://github.com/automl/nasbench301>

Download and extract the v1.0 model archive. Copy the extracted `xgb_v1.0`
directory to `data/nb_models_1.0/xgb_v1.0/`. The runtime and GNN ensembles are
not required by the supplied configuration.

The benchmark files retain their original licenses and are excluded from Git.
