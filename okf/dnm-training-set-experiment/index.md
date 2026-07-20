---
type: Experiment Plan
title: DNM training-set size vs. Gnocchi's local (GC-content) bias
description: >
  Plan to empirically test whether Gnocchi's GC-content bias is caused by
  sparseness of the de novo mutation (DNM) training set used to fit the
  regional-feature adjustment r(w), by resizing that training set, refitting,
  and recomputing local bias genome-wide. Not yet implemented.
tags: [gnocchi, gnomad-nc-constraint, gc-bias, dnm, logistic-regression, planned]
timestamp: 2026-07-20T00:00:00Z
---

# DNM training-set size vs. Gnocchi's local bias

This is the planning record for an experiment not yet implemented. It exists so
that a future session (human or agent) can resume implementation without
re-deriving the reasoning below, and so that a reviewer's agent can probe this
bundle to verify the plan's provenance against the actual code and data.

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
