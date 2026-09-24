# TMS E-field Analysis Pipeline

A 3-step Python pipeline for SimNIBS TMS electric-field simulation, element-wise association with study effect sizes, permutation testing, and result reporting.

## 1. TMS simulations

`1_TMS_batch_simulations.py`

Runs the SimNIBS TMS simulations for all studies.

```bash
python3 1_TMS_batch_simulations.py
```

If simulations already exist, this step can be skipped.

## 2. E-field analysis

`2_TMS_efield_analysis.py`

Extracts gray-matter E-fields and performs element-wise Spearman correlations, permutation testing, max-statistic FWER correction, connected-region identification, and result export.

First create the local configuration:

```bash
cp config.example.yaml config.yaml
```

Then run, for example with 10,000 permutations:

```bash
python3 2_TMS_efield_analysis.py \
  --config config.yaml \
  --n-permutations 10000
```

## 3. Results report

`3_TMS_analyses_results.py`

Creates the final HTML report and summary tables.

```bash
python3 3_TMS_analyses_results.py \
  --results-dir "/path/to/permutations_10000"
```

To run without Gmsh figures:

```bash
python3 3_TMS_analyses_results.py \
  --results-dir "/path/to/permutations_10000" \
  --no-figures
```

## Input

See `studies_template.csv` for the required study-table format.

## Validated example

Tested with:

- Cohort: CUD
- Head model: `sub-002`
- Studies: 11
- GM elements: 1,041,320
- Permutations: 10,000
- Runtime: approximately 1 min 21 s with compatible cached data available

Generated simulations, caches, results, and local configuration files are excluded from GitHub through `.gitignore`.

Reported region p-values are element-wise permutation p-values, not cluster-level p-values.
