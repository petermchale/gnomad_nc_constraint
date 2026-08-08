# `gnocchi_bias` — shared library

Imported by both figure directories so the window-table and model-fitting code exists in exactly one
place.

| Module | Contents |
|---|---|
| `windows.py` | Genome-wide 1kb window table: bucket download, duckdb column-pruned join, the McHale et al. window filters (sex chromosomes, noncoding, `pass_qc`, GeneHancer), GC units, the Gnocchi z-score, standardized ranks, GC binning, shared plot-style constants |
| `dnm_model.py` | The DNM training set and the per-context mutation model: loading, regime-1 subsampling, univariate feature selection, the multivariate PCA+logit fit, genome-wide `r(w)` apply, and training-set prediction |

## Two rules

1. **Neither module sets a matplotlib backend, and neither should.** `windows.py` doesn't import
   pyplot at all; `dnm_model.py` is the model, not the plots. The CLI entry points call
   `matplotlib.use("Agg")` inside their own `main()`. This is what lets the notebooks import these
   modules and still render inline — moving those calls to module scope silently breaks both
   notebooks.
2. **Docstrings here carry the citation trail** into CLAUDE.md, the Chen et al. source line numbers,
   and McHale et al.'s Methods. They are the provenance for the manuscript. Don't trim them.

## Provenance of the code itself

- `windows.py` was extracted verbatim from `compute_gc_bias_step1_vs_step2.py`, which was the CLI
  entry point until it was deleted (2026-08-07, superseded by fig5 panel A; recoverable from git).
  The extraction was verified behavior-preserving at the time: byte-identical binned output on the
  full 1,843,559-window path before and after.
- `dnm_model.py` was extracted verbatim from `dnm_training_size/run_dnm_training_experiment.py`.
  Validation of the reimplementation against Chen et al.'s own published outputs
  lives in `preconditions/`.
- The only genuinely new code is the N-curve generalization of the z/rank computation
  (`add_z_column`, `add_rank_columns`, `binned_rank_curves`), lifted from what
  the training-set-size experiment had already worked out.

`fit_multivariate_context` in `dnm_model.py` is the one step with **no published source anywhere** —
reconstructed from the apply-side code in `run_nc_constraint_gnomad_v31_main.py:231–249`. It is
validated end-to-end rather than line-by-line: at `frac=1.0` it reproduces the published Gnocchi
`expected` column at Pearson r = 1.0 over 1,984,900 windows.
