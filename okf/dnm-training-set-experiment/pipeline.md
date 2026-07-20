---
type: Plan
title: Implementation pipeline
description: >
  Concrete step-by-step plan to resize the DNM training set, refit the
  regional-adjustment logistic regression, and recompute local GC-content
  bias genome-wide. Not yet implemented — no script in this repo does this.
tags: [gnocchi, gc-bias, dnm, plan, not-implemented]
timestamp: 2026-07-20T00:00:00Z
---

# Pipeline

0. **Validate the fitting code first, before touching training-set size at
   all.** Run `analyze_individual_feature_effects.py` unmodified, on the
   full, unmodified [training-data](training-data.md) files, and diff its
   output against the published
   `genomic_features/dnm01_10x_ft_logit_regularized_coef_z_3mer_context_flnk_1k-1M.txt`
   — confirmed to be that script's exact published output (see
   [training-data](training-data.md)). Only proceed to steps 1+ once this
   matches; otherwise a resized-training-set refit could just as easily be
   diverging from the published pipeline for an unrelated reason.

Start with regime 1 from [training-data](training-data.md) (shrink both
dnm0+dnm1) — simplest, no new data sourcing needed, most directly comparable
to the existing step-1/step-2 baseline in root `CLAUDE.md`.

1. **Subsample.** Download the four files in [training-data](training-data.md).
   Join dnm1 sites ↔ their features on `locus`/`element_id`; same for dnm0
   (dropping `chrX`, per `analyze_individual_feature_effects.py:18`). Take a
   random subsample of both at a chosen rate (e.g. 10%, 1%) — or, for regime 3,
   source additional dnm0 sites from `context_prepared.ht` instead of
   subsampling.

2. **Refit feature selection**, per context, per `(feature, window)`:
   z-score the feature over the subsampled dnm0+dnm1 pool, fit
   `sm.Logit(group, sm.add_constant(z_feature)).fit_regularized()`
   (`analyze_individual_feature_effects.py:38–57` almost verbatim, just fed
   the subsampled data instead of the full dnm0/dnm1), Bonferroni-select
   (`analyze_individual_feature_effects.py:61–68`, same per-context threshold
   logic: `0.05/4/8` for CpG contexts `ACG/CCG/GCG/TCG`, `0.05/4/13`
   otherwise). Output: a new `context, feature, window, coef, se, pval` table,
   shaped like `misc/genomic_features13_sel.txt` but from the resized set —
   likely a *different* selected-feature set per context than published,
   especially for small subsamples.

3. **Refit the multivariate model**, per context (the actual gap — see
   [missing-code](missing-code.md)): take that context's newly-selected
   features from the subsampled training pool, standardize (mean/std from
   the subsampled data itself), fit `IncrementalPCA()` (default components —
   verified to equal the full feature count, not a reduction), transform,
   then `sm.Logit(group, sm.add_constant(pca_components)).fit_regularized()`.
   Output: new `(logit, pca, ft_mean, ft_std)` per context, replacing the
   published `.pkl`/`.pca.pkl`/`.ft_mean_std.txt` trio.

4. **Apply genome-wide**: reuse `run_nc_constraint_gnomad_v31_main.py`
   lines 236–270 unchanged (see [reusable-code](reusable-code.md)), feeding
   in the new per-context `(logit, pca, ft_mean, ft_std)` from step 3 instead
   of the published ones, against the unchanged genome-wide feature table
   `misc/genomic_features13_genome_1kb.txt`. Output: new
   `rr_{context}` per window, then new `expected` (post-adjustment) per
   window, using the **per-context** expected file
   `expected_counts_per_context_methyl_genome_1kb.txt` (not the pre-summed
   `expected_counts_by_context_methyl_genome_1kb.txt` — see
   [reusable-code](reusable-code.md)).

5. **Recompute local bias**: join the new per-window `expected` (post-new-`r`)
   against `observed_counts_genome_1kb.txt` and GC content from
   `misc/genomic_features13_genome_1kb.txt`'s `GC_content_1k`, bin by GC,
   compute mean `(expected − observed)` per bin — same procedure as
   `compute_gc_bias_step1_vs_step2.py`, ideally reusing that script's binning/
   plotting code directly, just swapping in the new `expected` column as a
   third curve.

6. **Compare** three curves on one plot: step-1 (context-only, unaffected by
   any of this), published step-2 (real Gnocchi, from the existing
   `compute_gc_bias_step1_vs_step2.py` run), and new-step-2 (this
   experiment's resized-training-set refit). Does the resized-training-set
   curve move toward step-1 (regime 1, testing
   [hypothesis](hypothesis.md) claim 1) or does tail bias shrink specifically
   (regime 3, testing claim 3)?

## Cost/scope note

Step 3 (refitting per-context multivariate models) means re-running steps
2–4 once *per subsample size tested* — not a single script run. Suggest
starting with one aggressive subsample (e.g. 1% of dnm0+dnm1) to get a clear
directional signal before investing in a full sweep across subsample sizes.
