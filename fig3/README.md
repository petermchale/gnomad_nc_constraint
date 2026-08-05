# Fig. 3 — GC bias is introduced by the regional adjustment, and tracks the DNM model's miscalibration

`fig3.ipynb` builds the figure. Everything it imports lives either in this directory or in
`gnocchi_bias/` at the repo root.

```
fig3.ipynb          the figure; run top to bottom
panels.py           the panels, as ax-accepting functions (no figure, no file I/O)
depletion_rank.py   loader for the Halldorsson depletion-rank window set (panel A, third curve)
r_eff.py            genome-wide r_eff = E2/E1, decomposed by CpG status
empirical_r.py      the adjustment the observed de novo mutations actually support
make_r_figures.py   driver: builds both adjustment-factor figures end to end
output/             figures, plus cached binned/intermediate tables
```

## The two adjustment-factor figures (added 2026-08-05)

`python fig3/make_r_figures.py` builds both, ~1 min warm / ~3 min cold. They supersede
the training-set calibration panel as panel-B candidates — see CLAUDE.md, "What Gnocchi
applies, and whether it is right".

**`r_eff_decomposition.pdf` — what Gnocchi applies.** `r_eff = E2/E1` versus GC, split
into CpG and non-CpG parts (an exact identity: `r_eff = Pi*r_CpG + (1-Pi)*r_non`). The
non-CpG curve runs 0.95 → 1.78; the CpG curve is flat at 0.98–1.00 at every GC; and the
counterfactual holding non-CpG `r` at 1 is flat within 0.6% across the whole range. The
GC dependence of what Gnocchi actually applies is wholly non-CpG, even though CpG
contexts carry 43% of the expected-count weight at GC 0.75.

**`r_non_vs_empirical.pdf` — whether it is right.** The fitted non-CpG `r` against the
adjustment the observed DNMs support, `DNMs / opportunities` per (context, GC bin). Both
sides are normalized per context to mean 1, mirroring `r`'s own definition as
"rate here / rate at the average", so only GC *shape* is compared. The fitted
curve climbs to 1.55; the observed one is flat near 1.0 until GC ≈ 0.55. Panel B is their
ratio: it crosses 1 at GC ≈ 0.40 and reaches **1.22–1.26** by GC 0.61–0.68.

## What the panels show

**A.** Mean standardized rank of three constraint metrics, binned by GC content. Full Gnocchi's rank
climbs steeply with GC; the context-only step of the *same* model is biased about as little as
depletion rank. So the bias is introduced by the regional-feature adjustment, not inherited from the
sequence-context model underneath it.

**B.** The multiplicative error in the adjustment factor the pipeline applies,
`r_model/r_true = mean_pred / empirical`, **stratified by CpG status**. CpG contexts are the only ones
denied `GC_content` (and four correlated features) by `FT_CORR_MET`, and they rise from 0.9% of sites
at GC 0.25 to 32% at GC 0.74 — so the high-GC signal in the earlier pooled version was almost entirely
a CpG effect.

> **Read before writing the caption.** Panel B does **not** explain panel A in the GC bulk. A uniform
> 10% inflation of `r` already moves mean rank from 0.500 to 0.687, and panel A's r-adjustment
> contribution at GC 0.57 needs `f ≈ 1.10`; panel B measures `f ≈ 0.98–1.00` there, and *wrong-signed*
> across GC 0.55–0.66. Only the top two GC bins are large enough to matter. Panel B is in-sample, on
> case-control-sampled sites, at site-level features; `r(w)` is applied out-of-sample, genome-wide, at
> window-aggregated features. See `panel_calibration_ratio`'s docstring and CLAUDE.md.

## Two inputs you must supply by hand

Neither is fetchable in this environment; both are set in the notebook's config cell.

| Constant | What it needs | Effect if left `None` |
|---|---|---|
| `DEPLETION_RANK_BED` | `41586_2022_4965_MOESM3_ESM.noncoding.enhancer.BGS.gBGC.GC_content.bed` from the constraint-tools `CONSTRAINT_TOOLS_DATA` path | The figure builds with two curves instead of three |
| `GENEHANCER_BED` | A licensed GeneHancer BED | "Neutral" is noncoding + `pass_qc` + autosome/PAR only, without the enhancer exclusion |

`depletion_rank.py` has been exercised against synthetic input (column resolution, GC unit
auto-detection, the `1 - DR` complement, error paths) but **not** against the real file. Check its
printed summary the first time it runs.

## Things worth knowing before citing numbers from this

- **The depletion-rank curve comes from a different window set** (Halldorsson windows, a different
  window size) than the two Gnocchi curves. They are not joined; each is ranked within itself. That is
  fine for a conditional-mean-rank comparison — the rank is uniform on (0,1) by construction for every
  curve — but the caption should say so.
- **Both panels' x-axis is the GC content of a 1 kb window**, but over different populations: panel A
  bins Chen windows by their own GC; panel B bins DNM training *sites* by the `GC_content_1k` regional
  feature at that site. Same quantity, two populations.
- Panel A's y-range is fixed to `[0, 1]` to match Fig. 2A; the curves only occupy ~0.26–0.88 of it.
  Pass `yrange=` to `panel_rank_bias` to tighten it.
- Panel B uses a **symlog** y-axis, not log. The gap changes sign across GC, and a log axis would
  silently drop exactly the bulk-GC bins that establish the model is well calibrated where data is
  dense.
- **The adjustment-factor figures use the refit's per-context `r`**, because the published pipeline
  writes its own to a local `output_dir` and never uploaded it (checked: no such bucket object under
  any prefix). This is validated, not assumed — the published `r_eff` for the "all" curve is directly
  computable as `expected_step2/expected_step1`, and refit vs published agree to max 1.0e-4 across all
  20 GC bins (median 3.9e-6). `make_r_figures.py` prints that check on every run; read it.
- **The observed curve's denominator is an opportunity count** (`possible`, the number of possible SNV
  sites of that context in those windows). Using the step-1 expected count `E1` instead gives the same
  curve to <0.1% — within a non-CpG context there is one methylation level, so `E1 = opportunities ×
  const`, and the constant cancels in the per-context normalization. Replacing it with a real sample of
  dnm0 background sites changes it by 1–2% through GC 0.65. What it *is* sensitive to
  is the window population: rates must be measured over the same analyzed windows the E1 weights and
  the per-context normalization come from. `load_training_by_context_bin` (whole genome) violates that
  and is documented as mis-specified.
- **Callability is the largest known uncertainty, and it is quantified.** DNMs are counted anywhere in
  a window, but `possible` counts only gnomAD-callable positions — a fraction that falls from 0.905 at
  GC 0.30 to 0.749 at GC 0.61 as short-read coverage drops in GC-rich sequence. If the trio call sets'
  callability tracks gnomAD's, no correction applies (over-adjustment 1.22 at GC 0.61); if they are
  closer to complete, dividing by the callable fraction gives 1.44. The finding holds either way, but
  **do not quote 1.22 as a tight number** — see `callable_fraction_by_bin()`. If instead trio calling is
  *less* sensitive at high GC than gnomAD, the bias runs the other way and part of the over-adjustment
  is technical. Both directions belong in the caption.
