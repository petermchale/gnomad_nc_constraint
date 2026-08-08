---
type: Open Question
title: Unresolved methodological details before refitting
description: >
  Hyperparameters and criteria the published code/data don't reveal, that a
  faithful refit would need to either recover, assume, or probe for.
tags: [gnocchi, gc-bias, dnm, risk, unresolved]
timestamp: 2026-07-20T00:00:00Z
---

# Open questions

Anyone implementing [pipeline](pipeline.md) should resolve or explicitly
assume these first — none are answered by code or data currently in this
repo or bucket.

1. **Regularization strength.** Both `analyze_individual_feature_effects.py`
   (univariate selection) and the presumed multivariate refit call
   `sm.Logit(...).fit_regularized()` with no explicit `alpha`/`L1_wt`
   arguments shown — so the published values are statsmodels' defaults. If
   the actual original fit used non-default regularization, refits here
   would silently diverge in a way that's hard to detect. Not checked: does
   the fitted `L1BinaryResultsWrapper` object expose its fit call's
   hyperparameters after the fact (e.g. via `.mle_settings` or similar)? If
   so, this is directly probeable from the published `.pkl` files, same
   technique as `verify_logit_predict_behavior.py`.

2. **`IncrementalPCA` settings beyond component count.**
   [missing-code](missing-code.md) confirmed `n_components_` equals the full
   feature count for context `AAA` — but not `whiten`, `batch_size`, or
   whether other contexts (especially ones with more selected features, e.g.
   `AAT` with 6) behave the same way. Should check a second, higher-feature-
   count context's `.pca.pkl` before assuming this generalizes.

3. **Standardization convention.** `ft_zscore2` in
   `run_nc_constraint_gnomad_v31_main.py:209–212` just subtracts a mean and
   divides by a std read from a file — doesn't reveal whether that std was
   computed with `ddof=0` or `ddof=1`. `analyze_individual_feature_effects.py`
   uses `scipy.stats.zscore` for the univariate step (default `ddof=0`) —
   reasonable to assume the same for the multivariate refit, but unconfirmed.

4. **What filtering built the original 10x dnm0 pool.** If regime 3 (grow
   dnm0 from `context_prepared.ht`, see [training-data](training-data.md))
   is attempted, the new sites need to match whatever criteria produced the
   published `context_prefiltered_nonmutated-dnm_sites10xdnm.mutation_rate.txt`
   — the filename says "context_prefiltered" (echoing `context_prefiltered.ht`
   from `run_nc_constraint_gnomad_v31_main.py:88`, i.e. coverage 30–32× +
   black-region filtered) but the exact "non-mutated" exclusion criterion
   (excluded from dnm1 only? from all known DNM call sets? from gnomAD
   variants too?) isn't shown anywhere in this repo.

5. **Whether `fit_regularized()` on a much smaller subsample even converges.**
   Aggressive subsampling (e.g. 1%) could leave some `(context, feature,
   window)` cells with very few positive (dnm1) examples, especially for
   less common trinucleotide contexts — worth checking per-context dnm1
   counts survive subsampling before committing to a specific rate.
