# Preconditions — what had to be true before any of this meant anything

Every figure here rests on claims about Chen et al.'s published pipeline that the paper
does not state, or states differently from how the code behaves. This directory holds the
checks on those claims, kept apart from the analyses that assume them, so the foundation
can be audited without reading the figures and nothing gets quietly re-assumed.

Each script downloads the real artifact and inspects it. None trusts CLAUDE.md, and none
needs Hail, a JVM, or credentials — the bucket is public.

## Two directions, and the filenames encode which

**`verify_*` — is what we BELIEVE about their artifact true?** Claims about the published
data, code and formula. These come first: they establish what the pipeline *is*.

**`validate.py` — is OUR code faithful to theirs?** The reimplementation in
`gnocchi_bias/dnm_model.py` against Chen et al.'s own outputs. Last, and it matters only
*because* `verify_missing_utils_files.py` establishes that reimplementing was necessary.

| | script | claim it checks | who depends on it |
|---|---|---|---|
| verify | `verify_expected_r1.py` | `expected_counts_by_context_methyl_genome_1kb.txt` really is the context-only, pre-adjustment (r ≡ 1) table, *and* it describes the same windows with the same `possible` denominators as the published constraint table | **everything** — fig5's step-1 curve (panel A), E₁ denominator (B), context-only baseline (E) |
| verify | `verify_logit_predict_behavior.py` | the operative adjustment is `r = σ(β₀+β·z)/σ(β₀)`, a ratio of *probabilities*, not the logit ratio the Methods state | fig5's Notation cell; panel B's "a level error cancels" argument |
| verify | `verify_missing_utils_files.py` | the PCA+logit fit behind `r(w)` is genuinely absent from the bucket (only the apply side is published), and `misc/generic.py` et al. are *not* missing | the premise of `validate.py` |
| verify | `verify_training_set_counts.py` | the four shipped training tables really are the training set the paper describes — both published counts reproduced, and the join `load_training_data` performs loses nothing | every fig5 claim about *what step 2 was fit on*: panels C, D and E, and the whole population argument. Also `dnm_training_size/` |
| validate | `validate.py` | the reimplementation reproduces Chen et al.'s, at the univariate parameters and end to end | fig5 and dnm_training_size both refit; if this fails, both measure their own bug |

## Running them

```bash
.venv/bin/python preconditions/verify_expected_r1.py              # ~8 min cold (3.3 GB), 3s cached
.venv/bin/python preconditions/verify_logit_predict_behavior.py   # seconds
.venv/bin/python preconditions/verify_missing_utils_files.py      # seconds
.venv/bin/python preconditions/verify_training_set_counts.py      # 5s cached (421 MB)
.venv/bin/python preconditions/validate.py -check expected        # seconds
.venv/bin/python preconditions/validate.py -check coefficients    # ~2 min cached (2.5 GB)
```

Timings measured 2026-08-10. "Cached" means the inputs are already in `published/`; the
first run of anything pays its download instead. `-check coefficients` is compute, not
I/O — 1,664 L1-logit fits, 77s of the 109s total.

All default `-dest_dir` to the repo-root `published/`, resolved from `__file__`, so running
one from inside this directory reuses the shared cache instead of refetching multi-GB files.
Downloads go through `gnocchi_bias.windows.download` (`curl -fL`, checked exit status,
`.part` sidecar) rather than a bare `curl`: a silent download failure would otherwise let an
empty file read as a passing check.

## Status, last run

- **`verify_expected_r1`** — `possible` exact on all 2,575,299 rows; `expected` within 4.6e-5
  relative. The residual is the two source files coming from pipeline runs 277 days apart
  (GCS `customTime`), each with its own random-downsample mutation-rate refit — not the
  r ≡ 1 interpretation. It also checks the r ≡ 1 table against the published constraint
  table's `possible`: equal on all **1,984,900** joined windows, max diff 0, so panel A's
  two curves count the same sequence and differ only in `expected`. (The z computed from
  those counts has no published counterpart; `fig5/README.md`, "How panel A's `r = 1`
  curve is validated", sets out the three layers and which check covers which.)
- **`verify_logit_predict_behavior`** — on the real `AAA` model, `predict(zero_row)` returns
  `0.0394`, a probability; `which="linear"` returns `-3.1948`, the intercept. The paper's
  stated formula is not what produced the published scores, and the one Author Correction
  (Nature 626:E1) says nothing about it.
- **`verify_missing_utils_files`** — the three modules exist at `misc/*.py`, are exactly what
  `run_nc_constraint_gnomad_v31_main.py:23–25` imports, and contain no `PCA`,
  `IncrementalPCA` or `fit_regularized`. The gap is real.
- **`verify_training_set_counts`** — both published counts reproduced, by different tables
  of the pair. The paper's **413,304** DNMs = the dnm1 *feature* table's 413,273 rows plus
  the 31 loci carrying two DNMs each (a locus-keyed table collapses those pairs). The
  **4,104,879** background = the dnm0 *site* table's 4,107,802 rows minus its 2,924 chrX
  rows, to within one row. The join loses nothing: 0 of 410,542 sites lack features. And
  `3mer` is the step-1 context-only rate — `fitted_po` summed over the three alts — to
  2.2e-16 across all 92 (context, methylation) combinations.
- **`validate -check expected`** — Pearson r = 1.000000 over 1,984,900 windows, median
  relative difference 3.8e-6.
- **`validate -check coefficients`** — **our feature selection reproduces theirs exactly**,
  against Chen et al.'s own published selected-feature file
  (`misc/genomic_features13_sel.txt`): 239 rows each, none
  in one and not the other, no significance verdict flipped, though 249 rows sit within a
  decade of the Bonferroni threshold. That is the result to quote — it is a comparison
  against their selection *output*, and the selected set is what each context's multivariate
  model is then fit on.
  On the coefficients themselves, all 1,664 rows agree to within **0.021 of that row's own
  published standard error** (median 0.0008). Prefer that to the absolute figure (max |coef
  diff| 2.6e-4, 100% under 1e-3), which cannot be read without knowing the scale — the
  coefficients are not order 1, median |coef| is 0.027. Relative error is no better,
  reaching 172% on coefficients of order 1e-5 that are indistinguishable from zero.
  All consistent with solver-tolerance and library-version noise against a 2022
  `statsmodels` L1 logit.

## Two further checks live in `fig5/`, deliberately

Properties of the figure, printed on every run, so copies here would drift: refit-vs-published
`r_eff` per GC bin (max 1.0e-4, median 3.9e-6, `data.r_eff_by_gc`), and the full-population
refit landing on published Gnocchi in panel E (mean |rank − 0.5| 0.212 vs 0.212) — which is
also what makes the retrained result attributable to the intervention rather than to the
reimplementation.

## Not here

`verify_comparisons_tables.py` (repo root) checks why the data behind Extended Data Fig. 6
cannot answer the GC-bias question. Nothing depends on it, so it records a road not taken
rather than a precondition.
