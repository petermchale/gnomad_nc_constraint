---
type: Log
title: Change log
description: Chronological record of when this bundle was created/updated and why.
tags: [gnocchi, log]
timestamp: 2026-07-20T00:00:00Z
---

# Log

- **2026-07-20** — Bundle created. Preceded by (in the same conversation):
  finding the DNM training-set files during a bucket scan prompted by
  investigating `expected_counts_by_context_methyl_dnm_1M.txt`; documenting
  them in root `CLAUDE.md`; then this planning pass answering "can we
  retrain the logistic regression on resized DNM data and recompute local
  bias?" and "can `run_nc_constraint_gnomad_v31_main.py` be repurposed?".
  Nothing in [pipeline](pipeline.md) has been implemented or run yet — this
  is a plan, not a result.

- **2026-07-20 (later same day)** — Prompted by a direct question ("how do we
  know [training-data](training-data.md)'s list is exhaustive?"), fully
  listed `misc/` for the first time (previously only spot-checked). Found:
  (a) `misc/generic.py`, `misc/constraint_basics.py`, `misc/nc_constraint_
  utils.py` exist and are exactly what `run_nc_constraint_gnomad_v31_main.py`
  imports — root `CLAUDE.md` had wrongly called these "missing"; corrected
  there, and [missing-code](missing-code.md) here updated to note they were
  checked directly and confirmed to *not* contain the multivariate PCA fit
  either, which strengthens (doesn't weaken) that specific gap claim.
  (b) A separate, unrelated Random Forest DNM-prediction approach
  (`misc/RF_f18_dnm_1M.pkl` + `fig_tables_init/rf_f18_*`), on a 17-feature
  superset panel (`misc/genomic_features17_*`) — documented in root
  `CLAUDE.md` only, not folded into this bundle's pipeline since it's a
  different modeling approach, not a resizing of the same training set.
