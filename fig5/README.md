# Fig. 5 — Gnocchi's GC bias comes from its regional adjustment, fit on the wrong population

`fig5.ipynb` builds the figure and writes each panel to `output/fig5{A..E}.pdf` as a
standalone vector file for assembly in Illustrator, plus one supporting figure,
`output/fig5_supp_cpg.pdf`. Run it top to bottom.

**Panel C is two rows** sharing a GC axis: the composition of the background training
sites (how much of the training set is outside the scored population), and each
excluded stratum's DNM rate relative to the noncoding one (whether the excluded
territory is also *different*). The upper row alone shows only an absence.

**The supporting figure** backs panel B's claim that `r_CpG ~ 1` is *correct*: the
methylation effect step 1 absorbs (3.0-4.3x within one trinucleotide, against 9.7-15.2x
pre-saturation), the CpG-island character of high-GC CpGs (2.5% hypomethylated in the
GC bulk rising to 90-100% above GC 0.70), and the resulting DNM-rate collapse
(0.53 -> 0.195, a 2.7x fall). Its bin floor is 100 sites, not the main figure's 500:
the top two GC bins hold 356 and 169 sites and they ARE the claim, so they are drawn
with error bars rather than dropped.

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

`GENEHANCER_BED` is in a module rather than the notebook because it defines the analyzed
window set, and that set is used by two separate processes: `refit.py -population scored`
uses it to decide what the model is **fit** on, the notebook uses it to decide what the
panels are **evaluated** on. Disagreement means training on one population and scoring on
another — the defect this figure is about. Both read `config.py`, so they cannot disagree
within a run; and because refits persist on disk across edits, `refit.py` stamps the
value it used into `refits/provenance.json` and `data.refit_path` refuses a refit built
under a different setting. Edit `config.py` and you get a loud error naming the refit to
rerun, rather than a silent mismatch.

`depletion_rank.py` has been exercised against synthetic input (column resolution, GC
unit detection, the `1 - DR` complement, error paths) but **never against the real
file**. Check its printed summary the first time it runs.

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
  and the notebook prints them: the no-coverage stratum's non-CpG DNM rate (1.55x the
  noncoding rate in the GC bulk, **4.06x by GC 0.61**, while coding/noncoding stays at
  0.90–0.99 and flat), and the CpG-island character of high-GC CpGs (90–100%
  hypomethylated above GC 0.70, DNM rate 2.7x lower than the bulk, against a 3.0–4.3x
  methylation effect that step 1 already absorbs). Migrated from `fig3/` when that
  directory was retired, and drawn in `output/fig5_supp_cpg.pdf`.
- **Panel D measures a level error**, and levels cancel in `r = sigma(b0+b.z)/sigma(b0)`.
  It diagnoses the fit; panel E is the measurement of the bias. Its y-axis is also not a
  mutation rate — the ~0.07 baseline reflects the 10:1 case-control design.
- **Panel D is in-sample**; panel E is the out-of-sample confirmation, on gnomAD
  polymorphism counts the DNM model never sees. A held-out DNM split for panel D has not
  been run.
- **`output/` holds only figures and this figure's own caches.** The refit tables live in
  the repo-root `refits/` (see above), so there is exactly one copy of each on disk.
