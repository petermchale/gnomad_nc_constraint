---
type: Dataset
title: DNM-based validation files (background context, not required)
description: >
  Files found in the bucket during an earlier full-listing scan that look
  like a pre-existing, uncoded validation of the context-only model against
  real DNMs. Not training inputs, and not required for this experiment — kept
  here only so a future session doesn't re-discover and re-investigate them.
resource: gs://gnomad-nc-constraint-v31-paper/
tags: [gnocchi, dnm, validation, unconfirmed, background]
timestamp: 2026-07-20T00:00:00Z
---

# DNM validation files (background only)

None of these are inputs to [pipeline](pipeline.md) — they're outputs of some
DNM-based validation of the fitted context-only model, inferred from naming
and structure, **not confirmed by any code in this repo** (no script here
produces or consumes them).

| File | Contents |
|---|---|
| `expected_counts_by_context_methyl_dnm_1M.txt` | `element_id, possible, expected` at 1Mb resolution for the DNM cohort — same shape as the genome-wide 1kb file, confirmed by direct download. |
| `observed_counts_dnm_1M.txt` | `element_id, variant_count` — observed DNM counts per 1Mb window, same `element_id`s. |
| `possible_counts_by_context_methyl_dnm.ht/`, `observed_counts_by_context_methyl_dnm.ht/`, `proportion_observed_by_context_methyl_dnm.ht/` (+ a `_.ht` duplicate) | Hail tables, presumably per-context (not binned) versions. |
| `possible_counts_by_context_methyl_dnm_1M.ht/`, `observed_counts_by_context_methyl_dnm_1M.ht/`, `expected__counts_by_context_methyl_dnm_1M.ht/` (double underscore is real, in the bucket) | 1Mb-binned Hail-table versions. |
| `possible_counts_by_context_heptamer_methyl_dnm_1M.ht/` | Same idea at heptamer context resolution — presumably related to the `z_heptamer` model in Extended Data Fig. 6 (see root `CLAUDE.md`), but unconfirmed. |

Full context: root `CLAUDE.md`, section "The next experiment...",
subsection "Files that look like outputs, not inputs, of a DNM-based
validation". Browse any of these with
`list_bucket_files.py -prefix genomic_features/` or `-prefix <name>.ht/`.
