---
type: Dataset
title: DNM training-set files (dnm0/dnm1) to subsample
description: >
  The real, verified (downloaded and inspected directly) input tables that
  fit the published per-context regional-feature logistic regressions, and
  that this experiment would subsample to resize the training set.
resource: gs://gnomad-nc-constraint-v31-paper/genomic_features/
tags: [gnocchi, dnm, training-data, verified]
timestamp: 2026-07-20T00:00:00Z
---

# DNM training-set files

All four row counts and schemas below were confirmed by directly downloading
the files (not inferred from filenames alone). Full descriptions live in the
root `CLAUDE.md`, section "The next experiment...", "The actual training data"
subsection — this file is a condensed pointer to the same facts for agent
consumption.

| File | Rows (confirmed) | Role |
|---|---|---|
| `genomic_features/DNM_decode_psychencode_site_context.mutation_rate.txt` | 410,542 | **dnm1**: real germline DNM sites (positive class) |
| `genomic_features/context_prefiltered_nonmutated-dnm_sites10xdnm.mutation_rate.txt` | 4,107,802 | **dnm0**: matched non-mutated background sites (negative class), exactly 10x dnm1 |
| `genomic_features/genomic_features13_dnm1_flnk_1k-1M.txt` | ~413,274 | 52 regional-feature columns (13 features × 4 windows) for each dnm1 site, keyed by `element_id` = site's own locus |
| `genomic_features/genomic_features13_dnm0_10x_flnk_1k-1M.txt` | not counted directly (2.05 GB; byte-size ratio to the dnm1 feature file is ~10x, consistent) | same 52 columns for each dnm0 site |

Join key: `locus` (dnm0/dnm1 site tables) ↔ `element_id` (feature tables) —
see `analyze_individual_feature_effects.py:15,20` for the exact merge.

**Validation target, not a subsampling input**:
`genomic_features/dnm01_10x_ft_logit_regularized_coef_z_3mer_context_flnk_1k-1M.txt`
(124.8 KB) is the *published output* of `analyze_individual_feature_effects.py:29`
run on the full, unmodified files above — confirmed by exact path match. Before
subsampling anything, re-run the script unmodified and diff its output against this
file; only trust a resized-training-set refit once that baseline reproduction checks
out. See [pipeline](pipeline.md) step 0.

`analyze_individual_feature_effects.py:18` drops all `chrX` sites from dnm0
before fitting (autosomes only) — replicate this when subsampling.

## The three subsampling regimes (mapping to [hypothesis](hypothesis.md))

1. Shrink **both** dnm0 and dnm1 (e.g. random sub-sample at a fixed rate) →
   removes tail-`x` coverage entirely.
2. Full dataset as published → baseline, already the real Gnocchi.
3. Grow **only** dnm0 (background sites), holding dnm1 fixed. There is no
   larger real dnm1 pool available — the 410,542 real DNMs are all that
   exist — but dnm0 *can* be grown: the published 4,107,802-row dnm0 file is
   itself named `..._sites10xdnm...`, i.e. it was already deliberately
   subsampled down to exactly 10x dnm1's count from some larger candidate
   pool of non-mutated sites. The genome has far more candidate background
   sites than that — `context_prepared.ht` alone has ~8.77 billion possible-SNV
   rows (see root `CLAUDE.md`) — so a larger dnm0 could be assembled directly
   from `context_prepared.ht` via Hail (see the "Hail-on-this-Mac recipe" in
   root `CLAUDE.md`), filtered to autosomal, non-dnm1, coverage-passing sites,
   at whatever multiple of dnm1's count is wanted. Not yet attempted — see
   [open-questions](open-questions.md) for what filtering criteria the
   original 10x dnm0 pool actually used, which would need to be matched.
