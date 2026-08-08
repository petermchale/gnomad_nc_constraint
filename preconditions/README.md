# Preconditions — what had to be true before any of this meant anything

Every figure in this repo rests on claims about Chen et al.'s published pipeline that are
**not** stated in the paper, or that the paper states differently from how the code
behaves. This directory holds the checks on those claims, kept separate from the analyses
that assume them, so a reader can audit the foundation without reading the figures and a
future session cannot quietly re-assume something that was only ever asserted.

Each script downloads the real published artifact and inspects it. None takes anything on
trust from CLAUDE.md, and none needs Hail, a JVM, or credentials — the bucket is public.

## Two directions, and the filenames encode which

**`verify_*` — is what we BELIEVE about their artifact actually true?**
Claims about the published data, code and formula. These come first: they establish what
the pipeline *is*.

**`validate.py` — is OUR code faithful to theirs?**
The reimplementation in `gnocchi_bias/dnm_model.py` against Chen et al.'s own outputs.
This comes last, and only matters *because* `verify_missing_utils_files.py` establishes
that reimplementation was necessary at all.

| | script | claim it checks | who depends on it |
|---|---|---|---|
| verify | `verify_expected_r1.py` | `expected_counts_by_context_methyl_genome_1kb.txt` really is the context-only, pre-adjustment (r ≡ 1) expected-count table | **everything.** It is fig5's step-1 curve in panel A, its E₁ denominator in panel B, and its context-only baseline in panel E (`gnocchi_bias/windows.py`, `fig5/data.py`) |
| verify | `verify_logit_predict_behavior.py` | the operative adjustment is `r = σ(β₀+β·z)/σ(β₀)`, a ratio of *probabilities* — not the ratio of *logits* the paper's Methods state | fig5's Notation cell states this formula; panel B's whole "a level error cancels" argument needs `r` to be that ratio |
| verify | `verify_missing_utils_files.py` | the multivariate PCA+logit fit that produces `r(w)` is genuinely absent from the bucket (only the apply side is published), and `misc/generic.py` et al. are *not* missing | it is the premise of `validate.py` — the reason there is anything to reimplement |
| validate | `validate.py` | the reimplemented pipeline reproduces Chen et al.'s, at the univariate parameters and end to end | fig5 and dnm_training_size both refit; if this fails, both are measuring their own bug |

## Running them

```bash
.venv/bin/python preconditions/verify_expected_r1.py              # ~8 min, 3.3 GB download
.venv/bin/python preconditions/verify_logit_predict_behavior.py   # seconds
.venv/bin/python preconditions/verify_missing_utils_files.py      # seconds
.venv/bin/python preconditions/validate.py -check expected        # seconds
.venv/bin/python preconditions/validate.py -check coefficients    # ~10 min
```

All default `-dest_dir` to the repo-root `published/`, resolved from `__file__` rather than as a
relative path, so running one from inside this directory reuses the shared download cache
instead of fetching multi-GB files again.

## Status, last run

- **`verify_expected_r1`** — `possible` matches exactly on all 2,575,299 rows; `expected`
  to within 4.6e-5 relative. The residual is explained by the two source files coming from
  separate pipeline runs 277 days apart (GCS `customTime` metadata), each with its own
  random-downsample mutation-rate refit — not by the r ≡ 1 interpretation.
- **`verify_logit_predict_behavior`** — on the real `AAA` model, `logit.predict(zero_row)`
  returns `0.0394`, a probability; `linear=True` returns `-3.1948`, the intercept. The
  paper's stated logit-ratio formula is not what produced the published scores. This is an
  uncorrected discrepancy: the one published Author Correction (Nature 626:E1) says
  nothing about it.
- **`verify_missing_utils_files`** — the three modules exist at `misc/*.py` and are exactly
  what `run_nc_constraint_gnomad_v31_main.py:23–25` imports; none of them contains a `PCA`,
  `IncrementalPCA` or `fit_regularized` reference. The gap is real.
- **`validate -check expected`** — Pearson r = 1.000000 over 1,984,900 windows, median
  relative difference 3.8e-6.
- **`validate -check coefficients`** — all 1,664 `(context, window, feature)` rows
  comparable, max |coef diff| 2.6e-4, 100% agreeing to <1e-3. Not bit-identical; plausibly
  solver-tolerance and library-version noise against a 2022 `statsmodels` L1 logit.

## Two further checks live in `fig5/`, deliberately

They are properties of the figure and are printed on every run, so duplicating them here
would let the copies drift:

- refit-vs-published `r_eff` per GC bin — max 1.0e-4, median 3.9e-6 (`data.r_eff_by_gc`);
- the full-population refit landing exactly on published Gnocchi in panel E,
  mean |rank − 0.5| 0.212 vs 0.212 — which is also the control that makes the retrained
  result attributable to the intervention rather than to the reimplementation.

## Not here

`verify_comparisons_tables.py` (repo root) checks why the data behind Extended Data Fig. 6
cannot answer the GC-bias question. It is a record of a road not taken — nothing depends
on it — so it is not a precondition.
