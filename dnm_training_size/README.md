# DNM training-set size vs. Gnocchi's GC bias

Regime 1 of the training-set-size experiment: shrink the DNM training set (both the mutated `dnm1`
sites and the non-mutated `dnm0` background, at the same rate), refit the whole per-context model,
re-derive genome-wide expected counts, and watch what happens to GC-content bias.

Renamed from `dnm_training_experiment/` (2026-08-04) when the fitting code moved into the shared
`gnocchi_bias/` package.

```
dnm_training_set_size.ipynb    plots the headline result from already-computed refit tables
run_dnm_training_experiment.py the CLI: -mode validate | refit | reliability
plot_dnm_bias_comparison.py    the CLI equivalent of the notebook
plot_reliability_gap.py        standalone calibration-gap plot from a binned reliability table
output/                        refit tables, selected features, reliability tables, plots
```

The model itself — training-data loading, subsampling, univariate selection, the multivariate
PCA+logit fit, genome-wide apply — lives in `gnocchi_bias/dnm_model.py`, shared with `fig3/`.

## The three modes

```bash
# 0. Does the reimplemented feature selection reproduce the published coefficients?
python run_dnm_training_experiment.py -mode validate

# 1. Refit at a given training-set size and re-derive genome-wide expected counts (slow)
python run_dnm_training_experiment.py -mode refit -subsample_frac 0.01
python run_dnm_training_experiment.py -mode refit -subsample_frac 0.1
python run_dnm_training_experiment.py -mode refit -subsample_frac 1.0 -tag full

# 2. Training-set reliability diagram (fast -- never touches the genome-wide files)
python run_dnm_training_experiment.py -mode reliability -subsample_frac 1.0 -tag full
```

`-mode reliability` at `-subsample_frac 1.0 -tag full` is what produces
`output/training_reliability_binned.dnm_refit_full.txt`, which **`fig3/` reads for its panel B**.

## Validation status

- `-mode validate` reproduces the published
  `dnm01_10x_ft_logit_regularized_coef_z_3mer_context_flnk_1k-1M.txt` across all 1,664
  (context, window, feature) rows, max |coef diff| = 2.6e-4.
- `-mode refit -subsample_frac 1.0` reproduces the published Gnocchi `expected` column at
  Pearson r = 1.0 over 1,984,900 windows — the end-to-end check on the reconstructed multivariate
  step, which has no published source anywhere.

## Result

The 1% curve is indistinguishable from the context-only (step 1) model at every GC bin; the 10% curve
sits smoothly between; the full-data curve reproduces published Gnocchi. A second, independent line of
evidence: the number of contexts in which `GC_content` itself clears Bonferroni significance is
**0/4** at 1%, **7/21** at 10%, **23/32** at full scale — at 1% the model has no statistical power to
detect a GC effect to adjust for at all.

## Known issue

`-downsample_frac` / `-downsample_n` (in `plot_dnm_bias_comparison.py`, and the same options in
`compute_gc_bias_step1_vs_step2.py`) are **not reproducible across runs even with a fixed
`-random_seed`**: duckdb's parallel join emits rows in a nondeterministic order, so `polars.sample()`
picks a different subset each time. Verified directly — two identical invocations of the unmodified
pre-refactor script gave different per-bin counts. The full, non-downsampled path is deterministic
(byte-identical across runs). Use downsampling only for rough iteration, never for a reported number.
