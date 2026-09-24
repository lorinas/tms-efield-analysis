# TMS E-field analysis

A deliberately simple two-script workflow for batch SimNIBS TMS simulations and spatial E-field/effect-size analysis.

## The only two Python scripts

1. **`1_TMS_batch_simulations.py`** — reads the study CSV, runs SimNIBS study by study, and writes `<Study_ID>_study/` simulation folders.
2. **`2_TMS_efield_analysis.py`** — finds those simulation outputs, extracts gray-matter `magnE`, verifies mesh alignment, correlates E-fields with effect sizes using Spearman rho, runs efficient/resumable permutations, computes max-statistic FWER p-values, optionally performs cluster permutation inference, and exports CSV/mesh maps.

The old manual scripts for extraction, combining CSVs, correlations, and mapping are no longer required.

## Files

- `studies.csv` — your working table used by both stages (ignored by Git).
- `studies_template.csv` — clean GitHub template.
- `config.yaml` — your working local configuration (ignored by Git).
- `config.example.yaml` — clean GitHub configuration template.
- `simulations/`, `work/`, `results/` — generated/local folders ignored by Git.

## Setup

Use the SimNIBS Python interpreter you already use in PyCharm. A ready-to-edit `config.yaml` and `studies.csv` are already included in the downloaded folder. Edit those two files with your real local paths and study rows. Their clean GitHub templates are kept separately as `config.example.yaml` and `studies_template.csv`.

## Stage 1 — simulations

Safety check without launching simulations:

```bash
python 1_TMS_batch_simulations.py --config config.yaml --dry-run
```

To actually simulate, set `simulation.dry_run: false` in `config.yaml` and run:

```bash
python 1_TMS_batch_simulations.py --config config.yaml
```

## Stage 2 — analysis

The normal command is simply:

```bash
python 2_TMS_efield_analysis.py --config config.yaml
```

For the first reproduction pass, keep `statistics.n_permutations: 0`. This produces the observed rho map without permutation inference. After validating that result, set the intended permutation count and `independent_studies_confirmed: true`.

Optional input check only:

```bash
python 2_TMS_efield_analysis.py --validate-only --config config.yaml
```

Synthetic software test:

```bash
python 2_TMS_efield_analysis.py --demo --output work/demo --n-permutations 128
```

## Analysis logic

Stage 2 performs, in order:

1. study/effect-size validation by `Study_ID`;
2. one-and-only-one scalar mesh resolution per study;
3. GM (`tissue_tag: 2`) `magnE` extraction;
4. element ID, tetrahedral topology, and coordinate alignment checks across studies;
5. cached E-field matrix construction;
6. tied-rank Spearman rho at every valid element;
7. batched, seeded, resumable effect-size permutations;
8. uncorrected permutation p-values and max-statistic FWER p-values from the same shuffles;
9. optional sign-separated face-connected cluster inference;
10. tabular and SimNIBS mesh-map export.

## GitHub

`config.yaml`, simulation outputs, caches, results, and local data are ignored. Commit `config.example.yaml`, the two scripts, the template CSV, README, and `.gitignore`.

## Scientific validation

Synthetic tests check software behavior only. Before scientific use, reproduce your previously accepted rho map on real data and verify study/effect-size definitions, mesh correspondence, and inference assumptions.
