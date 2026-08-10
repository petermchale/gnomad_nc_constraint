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
| verify | `verify_expected_r1.py` | `expected_counts_by_context_methyl_genome_1kb.txt` really is the context-only, pre-adjustment (r ≡ 1) table | **everything** — fig5's step-1 curve (panel A), E₁ denominator (B), context-only baseline (E) |
| verify | `verify_logit_predict_behavior.py` | the operative adjustment is `r = σ(β₀+β·z)/σ(β₀)`, a ratio of *probabilities*, not the logit ratio the Methods state | fig5's Notation cell; panel B's "a level error cancels" argument |
| verify | `verify_missing_utils_files.py` | the PCA+logit fit behind `r(w)` is genuinely absent from the bucket (only the apply side is published), and `misc/generic.py` et al. are *not* missing | the premise of `validate.py` |
| validate | `validate.py` | the reimplementation reproduces Chen et al.'s, at the univariate parameters and end to end | fig5 and dnm_training_size both refit; if this fails, both measure their own bug |

## Running them

```bash
.venv/bin/python preconditions/verify_expected_r1.py              # ~8 min, 3.3 GB download
.venv/bin/python preconditions/verify_logit_predict_behavior.py   # seconds
.venv/bin/python preconditions/verify_missing_utils_files.py      # seconds
.venv/bin/python preconditions/validate.py -check expected        # seconds
.venv/bin/python preconditions/validate.py -check coefficients    # ~10 min
```

All default `-dest_dir` to the repo-root `published/`, resolved from `__file__`, so running
one from inside this directory reuses the shared cache instead of refetching multi-GB files.
Downloads go through `gnocchi_bias.windows.download` (`curl -fL`, checked exit status,
`.part` sidecar) rather than a bare `curl`: a silent download failure would otherwise let an
empty file read as a passing check.

## Status, last run

- **`verify_expected_r1`** — `possible` exact on all 2,575,299 rows; `expected` within 4.6e-5
  relative. The residual is the two source files coming from pipeline runs 277 days apart
  (GCS `customTime`), each with its own random-downsample mutation-rate refit — not the
  r ≡ 1 interpretation.
- **`verify_logit_predict_behavior`** — on the real `AAA` model, `predict(zero_row)` returns
  `0.0394`, a probability; `which="linear"` returns `-3.1948`, the intercept. The paper's
  stated formula is not what produced the published scores, and the one Author Correction
  (Nature 626:E1) says nothing about it.
- **`verify_missing_utils_files`** — the three modules exist at `misc/*.py`, are exactly what
  `run_nc_constraint_gnomad_v31_main.py:23–25` imports, and contain no `PCA`,
  `IncrementalPCA` or `fit_regularized`. The gap is real.
- **`validate -check expected`** — Pearson r = 1.000000 over 1,984,900 windows, median
  relative difference 3.8e-6.
- **`validate -check coefficients`** — all 1,664 `(context, window, feature)` rows comparable,
  max |coef diff| 2.6e-4, 100% agreeing to <1e-3. Plausibly solver-tolerance and
  library-version noise against a 2022 `statsmodels` L1 logit.

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
