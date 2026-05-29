# Experimental Study: DBSCAN for Multi-Instance Learning

> Reproduction package for the Bachelor's Thesis *"Adapting DBSCAN to the Multi-Instance Learning Paradigm"* — Universidad de Córdoba / University of Cordoba (UCO), 2026.

This repository contains all scripts, configuration, and instructions needed to reproduce the experimental results reported in the thesis. The clustering and classification algorithms are implemented in the companion library [`miclustering`](https://github.com/Andmo2004/MIClustering), listed as a dependency and imported here without modification.

---

## Table of contents

- [Overview](#overview)
- [Repository structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Datasets](#datasets)
- [Reproducing the experiments](#reproducing-the-experiments)
- [Results](#results)
- [Integration tests](#integration-tests)
- [Citation](#citation)
- [License](#license)

---

## Overview

The thesis investigates whether DBSCAN, adapted to operate directly on Multi-Instance Learning (MIL) bags via bag-level distance functions, can match the classification performance of supervised MIL baselines while requiring no label information at training time.

The experimental pipeline covers five phases:

<p align="center">
  <img src="data/img_phases.jpg" alt="Proceso de exeperimentación" width="600">
</p>

| Phase | Script | Purpose |
|---|---|---|
| 0 | `00_precompute_matrices.py` | Cache all distance matrices to disk |
| 1 | `01_eda.py` | Dataset characterisation and separability analysis |
| 2 | `02_hyperparameter_tuning.py` | Optuna-based search over (scaler, metric, ε, min\_pts) |
| 3 | `03_clustering_quality.py` | Internal CVIs evaluated without ground-truth labels |
| 4 | `04_comparison_vs_baseline.py` | MIDBSCAN vs MIKnn, MIKMeans, MIKMedoids on held-out test |
| 5 | `05_statistical_tests.py` | Wilcoxon signed-rank tests and effect-size estimation |

The companion Kaggle notebook [`notebooks/Experiments_notebook.ipynb`](notebooks/Experiments_notebook.ipynb) executes all five phases end-to-end and records the full console output used in the thesis.

---

## Repository structure

```
tfg-experiments/
├ config/
│   └ settings.py              # Dataset paths, optimal hyperparameters, scaler map
├ data/
│   ├ datasets/                # .arff files — not tracked by git (see Datasets)
│   └ README.md                # Dataset provenance, versions, and licences
├ experiments/
│   ├ 00_precompute_matrices.py
│   ├ 01_eda.py
│   ├ 02_hyperparameter_tuning.py
│   ├ 03_clustering_quality.py
│   ├ 04_comparison_vs_baseline.py
│   └ 05_statistical_tests.py
├ notebooks/
│   └ Experiments_notebook.ipynb
│   └ Experiments_notebook_2.ipynb
├ optimization/
│   ├ best_params.py           # Optuna objective and study runner
│   ├ grid_search.py           # Grid search wrapper for MIDBSCAN
│   └ knn_dist_eps.py          # k-NN distance plot and knee detection
├ results/                     # Generated outputs — (*) not tracked by git
│   ├ distance_matrices/       # Cached .npy distance matrices
├ visualization/
│   ├ boxplots.py
│   ├ heatmap.py
│   └ plotter.py
├ .gitignore
├ README.md
├ README_en.md
├ requirements.txt
└ run.py                       # CLI entry point for single-experiment runs
```

> Unit tests for the distance functions and clustering algorithms live in [`miclustering/tests/`](https://github.com/Andmo2004/MIClustering/tree/main/tests) and are maintained alongside the library.

---

## Requirements

- Python ≥ 3.10
- The `miclustering` library (installed automatically via `requirements.txt`)
- The ARFF datasets listed in `data/README.md`

---

## Installation

### With uv (recommended)

```bash
# 1. Clone this repository
git clone https://github.com/Andmo2004/tfg-experiments.git
cd tfg-experiments

# 2. Create a virtual environment and install dependencies
uv venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows

uv pip install -r requirements.txt
```

### With pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The `miclustering` package is installed directly from its GitHub repository:

```
miclustering @ git+https://github.com/Andmo2004/MIClustering.git
```

No local copy of the library source is required.

---

## Datasets

The ten ARFF datasets used in the thesis are **not included** in this repository due to size and licensing constraints. Place each file under `data/datasets/` before running any experiment.

| Dataset | Bags | Instances | Features | Source |
|---|---|---|---|---|
| Musk1 | 92 | 476 | 166 | [UCI ML Repository](https://archive.ics.uci.edu/dataset/74/musk+version+1) |
| Musk2 | 102 | 6 598 | 166 | [UCI ML Repository](https://archive.ics.uci.edu/dataset/75/musk+version+2) |
| ImageElephant | 200 | 1 391 | 230 | [Andrews et al., 2003](https://dl.acm.org/doi/10.5555/2981345.2981352) |
| BirdsChestnut | 548 | 10 232 | 38 | [Briggs et al., 2012](https://dl.acm.org/doi/10.1145/2339530.2339745) |
| BirdsHammonds | 548 | 10 232 | 38 | [Briggs et al., 2012](https://dl.acm.org/doi/10.1145/2339530.2339745) |
| Harddrive1 | 369 | 68 411 | 61 | [Krause et al., 2016](https://www.usenix.org/conference/fast16) |
| Mutagenesis (atoms) | 188 | 1 618 | 10 | [Srinivasan et al., 1996](https://link.springer.com/article/10.1007/BF00114804) |
| Mutagenesis (chains) | 188 | 5 349 | 24 | [Srinivasan et al., 1996](https://link.springer.com/article/10.1007/BF00114804) |
| Newsgroups1 | 100 | 5 443 | 200 | [Zhou et al., 2009](https://dl.acm.org/doi/10.1145/1553374.1553534) |
| Thioredoxin | 193 | 26 611 | 8 | [Gärtner et al., 2002](https://link.springer.com/chapter/10.1007/3-540-36755-1_12) |

See `data/README.md` for download instructions and licence details for each dataset.

---

## Reproducing the experiments

All phases assume that datasets are present in `data/datasets/` and that the virtual environment is activated. Run the phases in order; each phase reads the outputs of the previous ones from `results/`.

**Phase 0 — Precompute distance matrices** *(run once; ~30 min on first execution)*

```bash
python experiments/00_precompute_matrices.py
```

Computes and caches all (dataset × scaler × metric) distance matrices under `results/distance_matrices/`. Subsequent phases load from cache and skip recomputation.

**Phase 1 — Exploratory data analysis**

```bash
python experiments/01_eda.py
```

Outputs separability ratios, instance-count boxplots, and distance heatmaps to `results/figures/` and a summary CSV to `results/tables/eda_summary.csv`.

**Phase 2 — Hyperparameter tuning**

```bash
python experiments/02_hyperparameter_tuning.py
```

Runs 100 Optuna trials per dataset (TPE sampler, seed 42) and writes the best (scaler, metric, ε, min\_pts) configuration to `results/tables/optuna_best_params_<timestamp>.csv`. The file is automatically picked up by `config/settings.py` on the next run.

**Phase 3 — Internal clustering quality**

```bash
python experiments/03_clustering_quality.py
```

Evaluates SED, DD, Hc, VRC, and I indices for each dataset's optimal configuration and two ε perturbations (×0.5 and ×2.0). Outputs `results/tables/cvi_comparative_<timestamp>.csv` and Spearman correlation between VRC/I and F1.

**Phase 4 — Comparison vs baselines**

```bash
python experiments/04_comparison_vs_baseline.py
```

Trains MIDBSCAN, MIKMeans, MIKMedoids, and MIKnn (best k via inner validation) on a 70/30 stratified split and evaluates all models on the held-out test set. Writes `results/tables/full_eval_<timestamp>.csv` and confusion matrix figures for representative datasets.

**Phase 5 — Statistical validation**

```bash
python experiments/05_statistical_tests.py
```

Reads the latest `full_eval_*.csv` and runs Wilcoxon signed-rank tests (α = 0.05) with effect-size estimation (r = Z / √N) for global and subgroup comparisons.

### Running all phases at once (Kaggle / CI)

```bash
for phase in 00 01 02 03 04 05; do
    python experiments/${phase}_*.py
done
```

The Kaggle notebook replicates this sequence and archives the full console output as part of the submission record.

---

## Results

Key results from the thesis (70/30 split, seed 42, Hungarian label mapping):

| Dataset | MIDBSCAN F1 | MIKnn F1 | Δ F1 |
|---|---|---|---|
| Musk1 | 0.788 | 0.849 | −0.061 |
| Musk2 | 0.640 | 0.828 | −0.188 |
| ImageElephant | 0.729 | 0.773 | −0.044 |
| BirdsChestnut | 0.678 | 0.686 | −0.008 |
| BirdsHammonds | 0.984 | 0.984 | +0.000 |
| Harddrive1 | 0.983 | 0.957 | +0.026 |
| Mutagenesis (atoms) | 0.881 | 0.872 | +0.009 |
| Mutagenesis (chains) | 0.800 | 0.868 | −0.068 |
| Newsgroups1 | 0.786 | 0.417 | +0.369 |
| Thioredoxin | 0.182 | 0.316 | −0.134 |

Wilcoxon signed-rank test (global F1): W = 14.0, p = 0.359, r = 0.29. The null hypothesis of equal performance between MIDBSCAN and MIKnn is not rejected at α = 0.05.

Full tables, figures, and per-dataset confusion matrices are available in `results/` after running the pipeline.

---

## Citation

If you use this code or the experimental setup in your own work, please cite:

```bibtex
@bachelorsthesis{mors2026midbscan,
  author  = {Moros Rincón, Andrés},
  title   = {Adapting {DBSCAN} to the Multi-Instance Learning Paradigm},
  school  = {Universidad de Córdoba / University of Cordoba},
  year    = {2026},
  type    = {Bachelor's Thesis},
}
```