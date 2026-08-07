# DNM training-set SIZE vs. Gnocchi's GC bias

Shrink the DNM training set — both the mutated `dnm1` sites and the non-mutated `dnm0`
background, at the same rate — refit the whole per-context model, re-derive genome-wide
expected counts, and watch what happens to GC-content bias.

```
dnm_training_set_size.ipynb     the result: bias curves at 1% / 10% / 100%
run_dnm_training_experiment.py  refit at one training-set size (slow; run once per size)
output/                         expected-count and selected-feature tables, the figure
```

The model itself — training-data loading, subsampling, univariate selection, the
multivariate PCA+logit fit, genome-wide apply — is in `gnocchi_bias/dnm_model.py`, shared
with `fig5/`. The full-scale refit is produced by `fig5/refit.py -population full` into
the shared repo-root `refits/`, which is where the notebook reads it from.

```bash
.venv/bin/python dnm_training_size/run_dnm_training_experiment.py -subsample_frac 0.01
.venv/bin/python dnm_training_size/run_dnm_training_experiment.py -subsample_frac 0.1
```

## What this shows that `fig5/` does not

**1. A dose-response in training-set size.** The 1% curve is indistinguishable from the
context-only (step 1) model at every GC bin; the 10% curve sits smoothly between; the
full-data curve reproduces published Gnocchi. This tests `chen_formula.tex`'s original
prediction — that sparse feature tails collapse `r_c(x)` toward 1 — and it holds across
the *entire* GC range, not just at the endpoints the tex described.

**The contrast with `fig5/` is the point of keeping both.** Shrinking the training set
moves Gnocchi *toward* the context-only model but **never past it**. `fig5/`'s
intervention — changing the training *population* at fixed size — goes past it
(mean |rank − 0.5| 0.046, against the context-only model's 0.093). So the population fix
differs in kind from merely having less data or more shrinkage. `fig5/`'s size-matched
control tests one size and establishes that size does not explain *that* result; it
cannot show the dose-response.

**2. The mechanism, from feature selection.** The number of contexts in which
`GC_content` itself clears Bonferroni significance grows with training-set size:

| training set | contexts fit | of which `GC_content` selected |
|---|---|---|
| 1% | 4 | **0** |
| 10% | 21 | 7 |
| full | 32 | 23 |

At 1% the model has no statistical power to detect a GC effect to adjust for at all,
regardless of whether one exists. Read from `output/selected.dnm_refit_*.txt` and
`refits/selected.full.txt`.

## Not here any more

- **Validation of the reimplementation** moved to `preconditions/`. It is a
  precondition for this experiment, not a finding of it.
- **The training-set reliability diagram and calibration gap** are superseded by `fig5/`
  panel D, which is the same diagram on the populations that matter. The gap in
  particular measures a *level* error, which cancels in `r = σ(β₀+β·z)/σ(β₀)` and never
  reaches Gnocchi.
- **The pooled GC-only logistic diagnostic** was never a faithful stand-in for the
  per-context multivariate model (it collapses several PCA-whitened features and all 32
  contexts into one curve). The selection-frequency table above and the bias curves are
  the real evidence.

## Known issue, still live elsewhere

`-downsample_frac` / `-downsample_n` in `compute_gc_bias_step1_vs_step2.py` are **not
reproducible across runs even with a fixed `-random_seed`**: duckdb's parallel join emits
rows in a nondeterministic order, so `polars.sample()` picks a different subset each time.
Verified directly — two identical invocations of the unmodified pre-refactor script gave
different per-bin counts. The full, non-downsampled path is deterministic (byte-identical
across runs). Use downsampling for rough iteration only, never for a reported number.
