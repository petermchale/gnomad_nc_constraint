"""
Shared library for the Gnocchi GC-bias analyses in this repo.

Two independent figure directories import from here, so that the fitting and
window-table code lives in exactly one place (see CLAUDE.md):

  fig3/               -- the new Fig. 3 for McHale, Goldberg & Quinlan: GC-binned
                         rank bias (step 1 vs full Gnocchi vs depletion rank)
                         over the DNM model's own calibration gap.
  dnm_training_size/  -- the DNM training-set-size experiment (regime 1):
                         does Gnocchi's bias collapse toward the context-only
                         model's as the training set shrinks?

Modules:
  windows.py    genome-wide 1kb window table -- download, duckdb join, the
                McHale-et-al window filters, GC units, z-scores, standardized
                ranks, GC binning, and shared plot style constants.
  dnm_model.py  the DNM training set and the per-context mutation model --
                loading, regime-1 subsampling, univariate feature selection,
                the (unpublished) multivariate PCA+logit fit, and training-set
                prediction for reliability diagrams.

Neither module sets a matplotlib backend: that is a caller's decision, so that
these import cleanly inside a Jupyter notebook with an interactive backend.
The command-line entry points set "Agg" inside their own main().
"""
