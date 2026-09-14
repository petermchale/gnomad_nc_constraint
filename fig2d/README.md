# fig2d/ — McHale et al. Fig. 2D with the decontaminated Gnocchi

McHale et al.'s Fig. 2D is the histogram of Gnocchi over their putatively neutral windows
whose GC content lies within 0.1 sd of the mean, drawn against a standard normal scaled to
the window count. It shows over-dispersion at typical GC. This directory redraws it with the
decontaminated Gnocchi (the `scored.neutral` refit from `fig5/refit.py`) beside the published
one, on the same windows, and asks only that question: does retraining `r` change the spread
in the central GC slice? It is not about the GC bias itself, which is Fig. 5's.

`fig2d_decontaminated.py` follows the original, `plot_gnocchi_distribution_vs_standard_normal`
in quinlan-lab/constraint-tools `papers/neutral_models_are_biased/10.residuals-are-over-dispersed.ipynb`,
step for step (the script's docstring lists each step), and checks that its slice holds the
58,286 windows that notebook printed. It needs McHale et al.'s window file and the
`scored.neutral` refit, so it runs on the constraint-tools HPC path:

```
python fig2d/fig2d_decontaminated.py
```

Outputs go to `output/fig2d_decontaminated.neutral.{png,pdf,tsv}`. The only import from
`fig5/` is `config.NEUTRAL_WINDOWS_BED`, deliberately: it is what the `scored` refit was fit on.

## Result (HPC run, 2026-09-14)

**Conclusion: decontaminating the training set does not remove the over-dispersion McHale
et al.'s Fig. 2D reports.** Near mean GC the published score is already roughly unbiased on
average, so the fix for the GC bias has little to act on there, and the excess spread is
variation within a GC slice that neither model captures. McHale et al.'s point that Gnocchi
misses neutral variation even at typical GC therefore stands after decontamination. It is a
separate defect from the GC bias Fig. 5 dissects, not a consequence of it.

The slice REPRODUCES the original: GC mean 0.3933, sd 0.0573 over 693,270 windows,
0.3876 < GC < 0.3990, **58,286 windows**, every one carrying both scores.

| | published | decontaminated | N(0,1) |
|---|---|---|---|
| sd | 2.003 | 1.909 | 1 |
| variance | 4.01 | 3.65 | 1 |
| sd from the IQR (ignores tails) | 1.705 | 1.584 | 1 |
| P(\|z\| > 2) | 0.250 | 0.218 | 0.046 |
| mean / median | -0.468 / -0.215 | -0.294 / +0.007 | 0 / 0 |

Retraining trims the sd by under 5% (variance by 9%); the over-dispersion -- a variance about
3.6x the standard normal's -- stays. The two scores correlate at 0.954 on these windows and
the decontaminated one sits +0.174 higher on average. It is also better centred: its median at
mean GC is +0.007 against the published score's -0.215, though its mean stays at -0.294
because of the heavy left tail. 16 decontaminated windows fall outside
the histogram's [-10, 10] bins; no published ones do.

**Which observed count.** The window file's `N_observed` differs from Chen et al.'s
`observed` in 691,490 of 693,270 windows, so it is not the count Gnocchi was computed from.
The constraint table's `observed` is: through the pipeline's z formula with the published
expected count it reproduces the file's `gnocchi` to 1.8e-15. The decontaminated z uses that
count. What `N_observed` measures was not investigated.
