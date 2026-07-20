---
type: Code Component
title: The missing multivariate-fit step, and what narrows it
description: >
  The one part of the published pipeline with no fitting code in this repo —
  only the apply/predict side is published — plus a verified finding
  (published PCA keeps all components) that makes reconstructing it tractable.
resource: /Users/petermchale/gnomad_nc_constraint/analyze_individual_feature_effects.py
tags: [gnocchi, code, gap, verified]
timestamp: 2026-07-20T00:00:00Z
---

# The gap

`run_nc_constraint_gnomad_v31_main.py` lines 231–234 *load* a pre-fitted
`L1BinaryResultsWrapper` (`{model}.pkl`) and a pre-fitted `sklearn.
IncrementalPCA` (`{model}.pca.pkl`) per trinucleotide context. No script in
this repo shows the code that produced those two objects — i.e. the
multivariate, PCA-reduced logistic regression that actually computes the
real Gnocchi `r(w)`. This is the same "missing downstream code" situation
documented for several other files in root `CLAUDE.md`.

What *is* in this repo, and is a close analogue: `analyze_individual_feature_
effects.py:38–57` fits one logistic regression *per feature* (univariate,
`sm.Logit(df_y, sm.add_constant(df_x[[ft]])).fit_regularized()`) to select
which features are significant. The missing step is the same idea, but fit
once per context on *all* that context's selected features together
(multivariate), after a PCA transform.

# What narrows the gap: PCA keeps all components

Verified by directly downloading and inspecting a real fitted PCA object,
not assumed:

```python
import pickle
pca = pickle.load(open('logit_regularized_dnm01_AAA_pbonf_pca.pca.pkl', 'rb'))
pca.n_components_   # -> 3
pca.n_features_in_  # -> 3
```

Context `AAA` has exactly 3 selected features (matches `misc/
genomic_features13_sel.txt`'s 3 `AAA` rows), and the fitted PCA also has
exactly 3 components — **it keeps every component**. So this "PCA" step is
not doing dimensionality reduction at all; it's an orthogonal whitening/
decorrelation of the standardized selected features before the regularized
fit (useful when selected features are correlated, e.g. `GC_content` and
`CpG_island`), not a feature-count reduction. That means reconstructing it
faithfully just requires `sklearn.decomposition.IncrementalPCA()` with
default (= all) components fit on the standardized training-feature matrix —
no component-count hyperparameter to guess. (Not yet checked whether this
holds for *every* context, only `AAA` — worth confirming for a context with
more selected features, e.g. `AAT` which has 6, before relying on it broadly.)

# What's still genuinely unknown

See [open-questions](open-questions.md) for the hyperparameters this
narrowing doesn't resolve (regularization strength, exact `IncrementalPCA`
batch-size/whitening settings, whether standardization uses population or
sample std).
