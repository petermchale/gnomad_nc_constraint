---
type: Code Component
title: Repurposable code in run_nc_constraint_gnomad_v31_main.py
description: >
  Which lines of the published pipeline script can be reused as-is (or
  near-as-is) to apply a refitted regional-adjustment model genome-wide,
  once a new model exists.
resource: /Users/petermchale/gnomad_nc_constraint/run_nc_constraint_gnomad_v31_main.py
tags: [gnocchi, code, reusable]
timestamp: 2026-07-20T00:00:00Z
---

# Reusable code

Verified by direct line-by-line reading of `run_nc_constraint_gnomad_v31_main.py`
(line numbers below match the file as of this writing).

## Fully reusable, unchanged

- **Lines 219–229**: per-context loop over `contexts`; selects that context's
  features from a `genomic_features13_sel`-shaped table (`context, feature,
  window`), including the CpG special case (line 227: for contexts
  `ACG/CCG/GCG/TCG`, exclude `ft_corr_met = ['GC_content','SINE','met_sperm',
  'Nucleosome','CpG_island']` — methylation-correlated features). This code
  doesn't care whether the selection table came from the published
  `misc/genomic_features13_sel.txt` or a freshly-refit one — just point it at
  the new selection output (see [pipeline](pipeline.md) step 2).
- **Lines 236–237, 241–249**: standardize the selected features
  (`ft_zscore2`), PCA-transform (`pca.transform(df_x)`), get the model's
  predicted probability at the window (`logit.predict(...)`) and at the
  training-population mean (`ave`), compute `rr = pred/ave`. Unchanged — only
  needs `logit` and `pca` to be real fitted objects, doesn't care how they
  were obtained.
- **Lines 250–270**: merge each context's `rr` into
  `expected_counts_per_context_methyl_genome_1kb.txt`
  (**not** the already-summed `expected_counts_by_context_methyl_genome_1kb.txt`
  — `rr` varies by context, so it must be applied before the context sum),
  multiply `expected * rr`, sum over context to get final per-window
  `possible`/`predicted`. Fully reusable unchanged.

## Needs replacement

- **Lines 231–234**:
  ```python
  logit = pickle.load(open(f'{model}.pkl', 'rb'))
  pca = pickle.load(open(f'{model}.pca.pkl', 'rb'))
  ```
  These *load* pre-fitted objects. Retraining means replacing this with a real
  fit call — see [missing-code](missing-code.md).
- **Lines 238–240**: `ft_mean`/`ft_std` are read from a published
  `{model}.ft_mean_std.txt`. Retraining should standardize against the
  *resampled* training set's own mean/std, not the original published ones —
  so this needs to become an in-memory computation from the new training data
  instead of a file read.

## Not needed at all

Everything upstream of line 199 (mutation-rate fitting, `possible`/`expected`
computation) is untouched by this experiment — the context-only (step 1)
model doesn't depend on the DNM training set at all, only the regional
adjustment `r` does.
