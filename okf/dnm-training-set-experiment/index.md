---
type: Experiment Plan
title: DNM training-set size vs. Gnocchi's local (GC-content) bias
description: >
  Plan to empirically test whether Gnocchi's GC-content bias is caused by
  sparseness of the de novo mutation (DNM) training set used to fit the
  regional-feature adjustment r(w), by resizing that training set, refitting,
  and recomputing local bias genome-wide. Implemented and run 2026-07-21 for
  regime 1 (shrink both dnm0+dnm1); confirms hypothesis claim 1 genome-wide.
tags: [gnocchi, gnomad-nc-constraint, gc-bias, dnm, logistic-regression, implemented]
timestamp: 2026-07-20T00:00:00Z
---

# DNM training-set size vs. Gnocchi's local bias

This started as a planning record for an experiment that was not yet implemented; as of
2026-07-21, regime 1 (shrink both dnm0+dnm1) is implemented and run — see "Results" below.
The plan/rationale docs below are kept as-is so a future session (human or agent) can
verify the implementation's provenance against the original reasoning, and so a
reviewer's agent can probe this bundle end-to-end.

## Results (2026-07-21)

- `/Users/petermchale/gnomad_nc_constraint/dnm_training_size/run_dnm_training_experiment.py`
  — implements pipeline.md steps 0-4 (validate, subsample, refit univariate selection,
  refit the missing multivariate PCA+logit step, apply genome-wide); also reports, per
  context, exactly which features were selected (plus a selection-frequency table) and
  writes a training-set GC-distribution diagnostic plot.
- `/Users/petermchale/gnomad_nc_constraint/dnm_training_size/plot_dnm_bias_comparison.py`
  — implements pipeline.md steps 5-6 (GC-binned rank-bias comparison across curves).
- Both scripts (and all of this run's outputs) were moved into their own directory,
  `dnm_training_size/`, on 2026-07-21 — code + results live together, separate from
  generic `tmp/` scratch downloads (which still hold the large shared bucket-file cache,
  reused across scripts via `-cache_dir`). See each script's own docstring for the
  cache_dir/output_dir split.
- [log](log.md)'s 2026-07-21 entries have full run provenance, validation numbers,
  per-GC-bin results, feature-selection-frequency tables, and GC-diagnostic-plot
  findings. Root `CLAUDE.md`, section "Implementation and results (2026-07-21)" (under
  "The next experiment...") has the same narrative for rebuttal/paper use.
- One-line summary: genome-wide, at every GC bin, a 1%-of-published DNM training set
  collapses Gnocchi's bias curve onto the context-only (step-1) curve; a 10% subsample
  sits smoothly in between; the full (100%) training set, run through this same
  reimplemented pipeline, reproduces published Gnocchi almost exactly (Pearson r=1.0).
  Confirms [hypothesis](hypothesis.md) claim 1. A second, independent line of evidence:
  the fraction of contexts in which `GC_content` itself clears Bonferroni significance
  grows with training-set size (0/4 -> 7/21 -> 23/32 selected-anything contexts, at
  1%/10%/100%). Regime 3 (densify background only) not attempted — needs Hail access to
  `context_prepared.ht`.
- A third line of evidence (`-mode reliability`, added 2026-07-21 later the same day): a
  training-set-only reliability diagram (real per-context multivariate model, evaluated
  in-sample against actual dnm0/dnm1 labels, binned by site-level GC — sidesteps the
  case-control training design's intercept bias by staying entirely within the training
  population). At full training-set size, fitted and empirical probability track almost
  exactly through the dense bulk of the GC range but the fitted curve visibly overshoots
  in the sparse high-GC tail (e.g. GC≈77%: predicted 0.291 vs empirical 0.153, n=649) —
  direct, in-sample confirmation of "sensitive to the tails, doesn't fit them well."

## Why

[hypothesis](hypothesis.md) — the specific claim from `chen_formula/chen_formula.tex`
this experiment tests, and what the rebuttal's (unpublished-methods) red text
claims was already shown informally.

## What exists already (verified, not assumed)

- [training-data](training-data.md) — the real dnm0/dnm1 site tables and their
  regional-feature joins, with confirmed row counts and schemas, that the
  experiment would subsample.
- [reusable-code](reusable-code.md) — which lines of
  `run_nc_constraint_gnomad_v31_main.py` can be repurposed as-is to apply a
  refitted model genome-wide.
- [missing-code](missing-code.md) — the one real gap (the multivariate
  PCA + logistic fit itself isn't published) and a concrete finding that
  narrows it: the published PCA step keeps *all* components, i.e. it is a
  whitening rotation, not a dimensionality reduction.
- [validation-data](validation-data.md) — DNM-based possible/expected/observed
  tables already in the bucket, found via `list_bucket_files.py`, that look
  like a pre-existing (but uncoded, unconfirmed) DNM validation of the
  context-only model — background context, not required for this experiment.

## The plan itself

- [pipeline](pipeline.md) — the concrete step-by-step implementation plan.
- [open-questions](open-questions.md) — hyperparameters and methodological
  details that aren't recoverable from the published code/data and would need
  to be assumed or probed for.

## Provenance

- [log](log.md) — when this bundle was created and by what conversation.
- Root context: `/Users/petermchale/gnomad_nc_constraint/CLAUDE.md`, section
  "The next experiment: DNM training-set size vs. Gnocchi's local bias".
