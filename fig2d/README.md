# fig2d/ — McHale et al. Fig. 2D with the decontaminated Gnocchi

McHale et al.'s Fig. 2D is the histogram of Gnocchi over their putatively neutral windows
whose GC content lies within 0.1 sd of the mean, drawn against a standard normal scaled to
the window count. It shows over-dispersion at typical GC. This directory redraws it with the
decontaminated Gnocchi (the `scored` refit from `fig5/refit.py`) beside the published one,
on the same windows, and asks only that question: does retraining `r` change the spread in
the central GC slice? It is not about the GC bias itself, which is Fig. 5's.

`fig2d_decontaminated.py` follows the original, `plot_gnocchi_distribution_vs_standard_normal`
in quinlan-lab/constraint-tools `papers/neutral_models_are_biased/10.residuals-are-over-dispersed.ipynb`,
step for step (the script's docstring lists each step), and checks that its slice holds the
58,286 windows that notebook printed.

```
.venv/bin/python fig2d/fig2d_decontaminated.py           # their 693,270 windows; HPC path + scored.neutral refit
.venv/bin/python fig2d/fig2d_decontaminated.py -wider    # the 1,843,559-window reproduction; runs offline
```

Outputs go to `output/fig2d_decontaminated{,.neutral}.{png,pdf,tsv}`. The only import from
`fig5/` is `config.NEUTRAL_WINDOWS_BED`, deliberately: it is what the `scored` refit was fit on.

**Wider run (offline, not their figure)**: 144,576 windows in the slice; sd 1.96 published
against 1.86 decontaminated (N(0,1): 1), IQR-based sd 1.63 against 1.50, P(|z| > 2) 0.233
against 0.207 (0.046). Retraining trims the spread by about 5%; the over-dispersion stays.
The narrowed run has not been done.
