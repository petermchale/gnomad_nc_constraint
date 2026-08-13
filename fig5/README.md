# Fig. 5 — Gnocchi's GC bias comes from its regional adjustment, fit on the wrong population

`fig5.ipynb` builds the figure and writes each panel to `output/fig5{A..E}.pdf` as a
standalone vector file for assembly in Illustrator, plus one supporting figure,
`output/supp_fig7.pdf` — **Supporting Figure 7** in the manuscript, whose four rows are
cited there as 7A-7D. Run it top to bottom.

**Panel C is two rows** sharing a GC axis: the composition of the background training
sites (how much of the training set is outside the scored population), and each other
stratum's DNM rate relative to the QC-pass noncoding one (whether the territory outside
is also *different*). The upper row alone shows only an absence. Its three bands are
*QC-pass noncoding*, *QC-pass coding* and *QC-fail* — only the first two are split by
coding status, because `coding_prop` comes from the constraint table and a QC-fail window
has no row in it; measured separately, that band is 6.9% coding-overlapping against the
QC-pass windows' 7.1%.

**Supporting Figure 7** backs panel B's claim that `r_CpG ~ 1` is *correct*: the
methylation effect step 1 absorbs (3.0-4.3x within one trinucleotide, against 9.7-15.2x
pre-saturation), the CpG-island character of high-GC CpGs (2.5% hypomethylated in the
GC bulk rising to 90-100% above GC 0.70), and the resulting DNM-rate collapse
(0.53 -> 0.195, a 2.7x fall). Its bin floor is 100 sites, not the main figure's 500:
the top two GC bins hold 356 and 169 sites and they ARE the claim, so they are drawn
with error bars rather than dropped. **A fourth row** carries `Pi`, the CpG share of a
bin's step-1 expected counts (0.025 -> 0.426), which is why that claim matters rather
than merely holds: it is the weight in panel B's identity, so a GC trend in `r_CpG`
would have reached the applied multiplier scaled by up to 0.43 rather than erased. It is
binned over *windows* (`binned_b`, floor 100 windows) where the rows above are binned
over *sites*, and it is drawn on this figure's wider 0.2-0.8 axis, so it shows the two
highest-GC bins that panel B's own 0.2-0.73 range cuts off.

```
fig5.ipynb          the figure: LaTeX derivation of each plotted quantity, then the panels
config.py           the two hand-supplied inputs, and the refit provenance stamp
data.py             one builder per panel, each cached as parquet in output/
diagnostics.py      the measurements panels B and C state in prose but do not plot
panels.py           the five panels as ax-accepting functions (no figure, no file I/O)
refit.py            the intervention and its two controls (must run before the notebook)
depletion_rank.py   loader for the Halldorsson depletion-rank window set (panel A, third curve)
output/             panel PDFs, the supporting figure, and this figure's own caches
../refits/          the refit tables, shared with dnm_training_size/
```

Shared with `dnm_training_size/`, so deliberately outside this directory:
`gnocchi_bias/windows.py` (the window table, z-scores, ranks, GC binning) and
`gnocchi_bias/dnm_model.py` (the DNM training set and the per-context refit pipeline).

## Prerequisite: three refits

```
.venv/bin/python fig5/refit.py -population full          # ~6 min, control + panel B's r
.venv/bin/python fig5/refit.py -population scored        # the intervention
.venv/bin/python fig5/refit.py -population sizematched   # the sample-size control
```

Each writes ~4 GB into the **repo-root `refits/`** (gitignored) as
`{table}.{population}.txt`. That directory holds one copy of each table, read directly by
`fig5/` and `dnm_training_size/`.

`data.refit_path` raises with the exact command if one is missing. The `full` refit is
needed even though it changes nothing:
the published pipeline writes its per-context `r` to a local directory and never
uploaded it, so panel B's CpG/non-CpG split uses the reimplementation's — a substitution
the notebook validates per GC bin against the published `E2/E1`, which needs no refit.

## Two inputs supplied by hand

Neither is fetchable here; both are `None` in **`config.py`** — set them there, not in
the notebook.

| Constant | What it needs | Effect if left `None` |
|---|---|---|
| `DEPLETION_RANK_BED` | `41586_2022_4965_MOESM3_ESM.noncoding.enhancer.BGS.gBGC.GC_content.bed` from the constraint-tools `CONSTRAINT_TOOLS_DATA` path | Panel A builds with two curves instead of three |
| `GENEHANCER_BED` | A licensed GeneHancer BED | "Neutral" is noncoding + `pass_qc` + autosome/PAR without the enhancer exclusion |

`GENEHANCER_BED` lives in a module, not the notebook, because it defines the analyzed
window set — which `refit.py` uses to decide what the model is **fit** on and the notebook
uses to decide what the panels are **evaluated** on. Disagreement would train on one
population and score on another, the defect this figure is about. `config.py`'s docstring
has the full argument, including how `refits/provenance.json` turns a post-refit edit into
a loud error rather than a silent mismatch.

`depletion_rank.py` has been exercised against synthetic input (column resolution, GC
unit detection, the `1 - DR` complement, error paths) but **never against the real
file**. Check its printed summary the first time it runs.

## How panel A's `r = 1` curve is validated

The context-only curve carries the comparison the whole figure rests on (0.093 against
0.212), and Chen et al. never published anything to check it against directly — their
pipeline computes no step-1 z. So it is validated in three separable pieces, two of them
runnable checks and one an inheritance argument:

| what | how | result |
|---|---|---|
| the **expected counts** really are pre-adjustment (`r ≡ 1`) | `preconditions/verify_expected_r1.py` regenerates the file genome-wide from `expected_counts_per_context_methyl_genome_1kb.txt`, whose provenance is confirmed — it is the literal `hl.export()` at `run_nc_constraint_gnomad_v31_main.py:191–197`, written *before* any r code runs | `possible` exact on all 2,575,299 rows; `expected` within 4.6e-5 relative, explained by two pipeline runs 277 days apart |
| the two curves describe the **same windows** | same script: `possible` in the r≡1 table against `possible` in the published constraint table, which the r-adjustment multiplies `expected` but never touches | equal on all **1,984,900** joined windows, max diff 0 — so the curves differ only in `expected` |
| the **z and rank** computed from them | cannot be checked directly, since no published step-1 z exists. Instead `gnocchi_bias/windows.py` runs the *identical* code path on the step-2 curve, which does have a published counterpart (`check_z_against_published`) | max \|z − z_published\| = **0.0** over 1,843,559 windows |

The third row is the load-bearing one and worth stating plainly to a reviewer: `z_step1`
is self-computed, and what licenses it is that the same `z_expr`, the same joint filter
and the same within-curve ranking reproduce Chen et al.'s own `z` exactly wherever a
published value exists. Both curves are also z-filtered *jointly* and ranked *after* that
filter, so neither is advantaged by its own window set.

## Things to know before quoting numbers

- **Panels A and E are the same statistic on the same windows**, so they read as
  before/after. They differ only in that E's inner join against the retrained expected
  counts drops a handful of windows, and the joint `z` filter then applies to five
  curves rather than two.
- **The depletion-rank curve is a different window set** (Halldorsson windows, a
  different window size). It is ranked within itself and overlaid, never joined on
  `element_id`. Legitimate for a conditional-mean-rank plot — the rank is uniform on
  (0,1) by construction for every curve — but the caption must say so.
- **Panel B is a decomposition identity, not a fit.** `r_eff = Pi*r_CpG + (1-Pi)*r_non`
  holds bin by bin because each bin aggregates ratios of *summed* expected counts, not
  means of per-window ratios.
- **Two claims are made in the text and not plotted**, so `diagnostics.py` computes them
  and the notebook prints them: the QC-fail stratum's non-CpG DNM rate (1.55x the
  noncoding rate in the GC bulk, **4.06x by GC 0.61**, while coding/noncoding stays at
  0.90–0.99 and flat), and the CpG-island character of high-GC CpGs (90–100%
  hypomethylated above GC 0.70, DNM rate 2.7x lower than the bulk, against a 3.0–4.3x
  methylation effect that step 1 already absorbs). Migrated from `fig3/` when that
  directory was retired, and drawn in `output/supp_fig7.pdf`.
- **That stratum is QC failure, not absent sequence** — it was called "no gnomAD coverage"
  here until it was measured. Every one of the 587,902 windows has its QC inputs on file;
  70.9% fail Chen et al.'s ≥80%-PASS rule against 3.3% failing the 25–35× coverage band.
  `preconditions/verify_qc_filter.py` also confirms the filter forwards (all 1,984,900
  scored windows satisfy all three conditions) and records both denominators, since
  **86.6% of background training sites are in QC-pass windows** — the 87.8% quoted for the
  PASS rule is *within* the 13.3% that are not.
- **Panel D measures a level error**, and levels cancel in `r = sigma(b0+b.z)/sigma(b0)`.
  It diagnoses the fit; panel E is the measurement of the bias. Its y-axis is also not a
  mutation rate — the ~0.07 baseline reflects the 10:1 case-control design.
- **Panel D is in-sample**; panel E is the out-of-sample confirmation, on gnomAD
  polymorphism counts the DNM model never sees. A held-out DNM split for panel D has not
  been run.
- **`output/` holds only figures and this figure's own caches.** The refit tables live in
  the repo-root `refits/` (see above), so there is exactly one copy of each on disk.
