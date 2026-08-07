# Is the reimplemented fitting pipeline faithful to Chen et al.'s?

Every result in `fig5/` and `dnm_training_size/` refits Chen et al.'s per-context
regional-adjustment model, because **the fitting code for it was never published** — the
bucket ships the fitted `.pkl` models and the apply-side code, not the multivariate
PCA+logit fit that produced them. So the reimplementation in `gnocchi_bias/dnm_model.py`
is load-bearing for everything downstream, and it needs its own validation, kept separate
from the experiments that assume it.

```
.venv/bin/python validate_reimplementation/validate.py            # both checks
.venv/bin/python validate_reimplementation/validate.py -check expected      # ~seconds
.venv/bin/python validate_reimplementation/validate.py -check coefficients  # ~10 min
```

## The two checks, and why both are needed

| check | against | covers |
|---|---|---|
| `coefficients` | `genomic_features/dnm01_10x_ft_logit_regularized_coef_z_3mer_context_flnk_1k-1M.txt` | the **univariate** feature-selection stage |
| `expected` | the published `expected` column of `fig_tables/constraint_z_genome_1kb.annot.txt` | the whole pipeline **end to end** |

`coefficients` is the only check anywhere in this repo against a published *fitted
parameter* file rather than a downstream output. That is what makes it say the fitting
code agrees, rather than that the numbers happen to come out the same. But it covers a
stage whose output the figures never use directly.

`expected` covers the multivariate step, which has no published parameters to diff
against and so can only be validated through its output — but it validates a composite,
so on its own it cannot localize a discrepancy.

Neither substitutes for the other.

## Status, last run

- **coefficients**: all 1,664 `(context, window, feature)` rows comparable, max
  |coef diff| = 2.6e-4, 100% agreeing to <1e-3. Not bit-identical — plausibly
  solver-tolerance and library-version noise, since the published fit is a
  `statsmodels` L1 logit from 2022.
- **expected**: Pearson r = 1.0 over 1,984,900 windows, median relative difference 4e-6.

## Two further downstream checks live in `fig5/`, not here

They are properties of the figure and are printed on every run, so repeating them would
mean they could drift:

- refit-vs-published `r_eff` per GC bin — max 1.0e-4, median 3.9e-6 (`data.r_eff_by_gc`);
- the full-population refit landing exactly on published Gnocchi in panel E —
  mean |rank − 0.5| 0.212 vs 0.212, which is also the control that makes the retrained
  result attributable to the intervention rather than to the reimplementation.

## Prerequisite

`-check expected` reads `refits/expected_counts_by_context_methyl_genome_1kb.full.txt`,
written by `fig5/refit.py -population full`. It raises with that command if missing.
`-check coefficients` needs only the training data, downloaded on demand into `tmp/`.
