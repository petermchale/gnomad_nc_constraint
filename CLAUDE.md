# Context for this fork

This is a fork (`petermchale/gnomad_nc_constraint`, upstream `atgu/gnomad_nc_constraint`)
of the code behind Chen et al. 2024 Nature ("A genomic mutational constraint map using
variation in 76,156 human genomes", DOI 10.1038/s41586-023-06045-0), which built the
Gnocchi noncoding constraint score.

**Why this fork exists**: Peter is a co-author of McHale, Goldberg & Quinlan ("The
performance of genetic-constraint metrics varies significantly across the human
noncoding genome"), responding to a peer reviewer who asked for a mechanistic dissection
of GC-content bias in Gnocchi's two-step model (step 1: sequence-context-only mutation
rate; step 2: regional-feature adjustment `r`). 


## Repository layout

```
fig5/                    THE manuscript figure. Five panels, one argument; each panel a
  fig5.ipynb             standalone PDF for Illustrator. Start here.
  make_fig5_nb.py        generates fig5.ipynb -- edit prose/code HERE, not the notebook
  config.py, data.py, panels.py, refit.py, depletion_rank.py

dnm_training_size/       the training-set-SIZE dose-response, and only that
preconditions/           what had to be true about Chen et al.'s pipeline for any of the
                         above to mean anything: verify_* (is what we believe about their
                         artifact true?) and validate.py (is our code faithful to theirs?)
gnocchi_bias/            shared library: windows.py (window table, z, ranks, GC binning)
                         and dnm_model.py (training set, per-context refit pipeline)

published/               Chen et al.'s data as downloaded (gitignored, ~8 GB; set
                         $GNOCCHI_PUBLISHED_DIR to relocate it -- one definition, in
                         gnocchi_bias/windows.py, that every entry point defaults to)
refits/                  one copy of each regional-adjustment refit (gitignored, ~12 GB)

METHODS.md               the rank statistic's methods narrative -- extractable paper text,
                         and the citation trail 9 of windows.py's docstrings point into.
                         Kept at the root, not under fig5/, because windows.py is imported
                         from preconditions/ and dnm_training_size/ too.
```

Each directory has its own README with operational detail, and METHODS.md holds the
methods narrative. **This file holds only what those do not**: the bucket inventory, the
settled findings below, and what the paper's Methods get wrong about its own code.

Deleted, recoverable from git history: `fig3/` (superseded by fig5; preserved wholesale at
`070fee9`), `compute_gc_bias_step1_vs_step2.py` (its reusable logic is `gnocchi_bias/
windows.py`), and `chen_formula/` (the LaTeX write-up of the model; its
sections 1-5 are migrated into `fig5/fig5.ipynb`).

## Settled findings — do not re-derive these

**Two window sets, and which numbers go with which.** `NEUTRAL_WINDOWS_BED` in
`fig5/config.py` decides: unset, the analyzed set is this repo's **1,843,559-window
reproduction** (noncoding + `pass_qc` + autosome/PAR, built from the bucket); set, it is
McHale et al.'s own **693,270** putatively neutral windows. **The committed figure is the
narrowed run** -- `fig5/fig5.neutral.png`, `fig5/output/*.neutral.*`, and the executed
`fig5/fig5.ipynb`, whose prose quotes it throughout -- and so are the two manuscript
files, `fig5/captions.txt` and `fig5/methods.txt`. Numbers below therefore give the
**narrowed run first, the wider one in parentheses**. A few measurements exist only on the
wider set, because the neutral BED lives on the constraint-tools HPC path and is not
available offline; each of those says so.

Numbers are over the GC bins the panels actually draw (n >= 100 windows), which is what
`fig5` reports. On the wider set the same statistic over all 20 bins gives
0.130 / 0.221 / 0.079 instead of 0.093 / 0.212 / 0.046; both are correct, so quote the
filtered set to match the figure.

**The causal chain, panel by panel.**

1. The GC bias is **introduced by the regional adjustment**, not inherited from the
   context-only model: mean |rank - 0.5| is **0.046** for `r == 1` against **0.168** for
   published Gnocchi (wider run: 0.093 against 0.212). Depletion rank, overlaid on its own
   windows as an external comparison, sits at **0.096**.
2. That adjustment's GC dependence is **wholly non-CpG**. `r_non` runs **0.96 -> 1.45**
   while `r_CpG` stays **0.997-1.014**, and the counterfactual holding non-CpG `r` at 1 is
   flat within **0.4%** even though CpG contexts carry **26%** of the expected-count weight
   in the highest bin drawn (wider run: 0.95 -> 1.79, 0.98-1.00, 0.6%, 43% at GC 0.75).
   This is a decomposition identity, not a fit.
3. The training set **is not the scored population**: the QC-pass noncoding share of the
   training sites falls **0.82 -> 0.27** across GC, and the scored band -- the part of that
   territory McHale et al. call putatively neutral -- peaks at **0.36** near GC 0.35 and is
   down to **0.007** by GC 0.68 (wider run, where the scored band *is* QC-pass noncoding:
   0.84 -> 0.28). The excluded
   territory is *different*, not merely absent -- the QC-failing stratum's non-CpG DNM
   rate runs **1.50-1.63x** the scored rate through the GC bulk and **3.39x by GC 0.58**,
   while coding/noncoding stays flat at **0.86-1.00** (wider run: 1.55x, 4.06x by GC 0.61,
   coding 0.90-0.99). The *QC-pass putatively nonneutral noncoding* stratum -- the half of
   QC-pass noncoding the narrowing gives up, and a band that exists only on the narrowed
   run -- is flat too, at **0.94-1.03**. That is the measurement saying the narrowing costs
   sample size and nothing else, which is why the whole figure carries over between the two
   window sets.
   *Name that stratum carefully.* It is the windows with no row in the published
   constraint table, and until measured it was called "no gnomAD coverage" here, which is
   wrong: all 587,902 of them have their QC inputs on file, and they are absent because
   they failed Chen et al.'s window filter -- 70.9% the `>= 80%` of observed variants PASS
   rule, 43.3% the `>= 1000 possible variants` rule, only 3.3% the 25-35x coverage band
   (weighted by training sites: 87.8 / 14.3 / 1.0%). A residual 1.9% pass all three and
   are unexplained. Relatedly, `pass_qc` is **True on all 1,984,900 rows** of that table,
   so filtering on it is a no-op and a QC failure is only ever visible as an absent row.
   The filter also holds in the forward direction, re-evaluated from the raw
   `pass`/`coverage`/`possible` inputs rather than from that flag: **all 1,984,900 scored
   windows satisfy it, 0 violations**, with the pass fraction bottoming out at exactly
   0.8000 (1,723 windows sit there, fixing the comparison as `>=`), coverage spanning
   25.003-34.862 and `possible` at exactly 1,000. `preconditions/verify_qc_filter.py`.
   Note which file is the scored set: `constraint_z_genome_1kb.annot.txt`, not
   `expected_counts_by_context_methyl_genome_1kb.txt` -- the latter is the 2,575,299-window
   step-1 universe and still contains every QC failure. The QC-fail stratum is a **mixture
   of coding and noncoding** windows (6.9% coding-overlapping, against 7.1% among the
   QC-pass ones, so QC failure is near-independent of coding status); panel C therefore
   draws the scored population as its bottom band and names the territory outside it
   *QC-pass coding*, *QC-pass putatively nonneutral noncoding* and *QC-fail*, splitting
   only the QC-pass ones by coding status. The genome splits three ways -- QC-pass noncoding, QC-pass
   coding, QC-fail -- and the fourth band exists only when the scored population is
   narrower than QC-pass noncoding, cutting that category into McHale et al.'s set and
   the rest. The bottom band is defined by MEMBERSHIP in the analyzed window table,
   not by re-deriving its filters in SQL, so it follows `NEUTRAL_WINDOWS_BED` the moment
   that file is supplied; the `other_noncoding` band is empty and undrawn until then.
   That band was called `non_neutral` until 2026-08-18 and has read *QC-pass putatively
   nonneutral noncoding* since 2026-08-25. The bare `non_neutral` asserted more than the
   data does, since being outside a set McHale et al. call putatively neutral is not
   evidence of selection, and whether those windows differ at all is the open question
   the band exists to answer; the legend now carries *putatively* on both sides -- the
   bottom band reads *QC-pass putatively neutral noncoding* -- so the pair reads as one
   partition of QC-pass noncoding territory rather than as a verdict on the upper half.
   The stratum's column name is still `other_noncoding`.
4. **Restricting** the training set to the scored population shrinks the empirical GC
   dependence of P(DNM) from 2.45x (and non-monotonic -- it collapses above GC 0.66) to a
   smooth **1.60x** (wider run: 1.57x), and the logistic regression can then track it --
   to within 6% through GC 0.58, its final 670-site bin excepted, where it is 28% low --
   instead of missing by 26% and 29% in opposite directions.
5. **Refitting `r` there removes the bias**: **0.168 -> 0.026**, below the context-only
   model's own 0.046 (wider run: 0.212 -> 0.046, below 0.093). Two controls make this the
   population and not something else: the full-population refit through the same code
   lands at **0.168** (so it is not the reimplementation), and a size-matched random
   subsample lands at **0.162** (so it is not less data). Wider run: 0.212 and 0.210.

**The adjustment is wrong, not merely present.** *(Wider run only -- this came from
`fig3/`, which was built and deleted before `NEUTRAL_WINDOWS_BED` existed, so none of it
has been recomputed on McHale et al.'s 693,270 windows. The direction is not in doubt, but
do not pair these magnitudes with the committed panels.)* Measured against the adjustment
the observed DNMs support -- `DNMs / opportunities` per (context, GC bin), both sides
normalized per context -- the fitted non-CpG `r` climbs monotonically to 1.55 while the
observed one stays near 1.0 until GC ~0.55. Over-adjustment reaches **1.22-1.26** at
GC 0.61-0.68, many SEs from 1. Retraining on the scored population brings it to 0.92-0.97.
An independent 1 Mb ground-truth test agrees in direction: the model's `r` rises with GC
while the real residual DNM rate falls, and at GC 51% `r` is too high by 1.24x.
*Caveat, and it is the largest known one:* the DNM numerator counts anywhere in a window
but the denominator counts only gnomAD-callable positions, and that fraction falls
0.905 -> 0.749 across GC. If trio callability tracks gnomAD's, no correction applies
(1.22); if the trio sets are closer to complete, the correction gives 1.44. Quote the
range. Code for this figure went with `fig3/`; it is at `070fee9`.

**Ruled out — these are dead ends, with the measurement that closed each.**

- **Methylation is not the cause.** Chen et al. model it carefully and correctly, in step
  1. Step 2 *is* methylation-blind, which looks like a suspect, but CpG-context `r` is flat
  at ~1.00 across the entire GC range, so those models contribute no GC-dependent
  adjustment at all. Two independent reasons: `FT_CORR_MET` strips their GC-correlated
  features, and `r` is a ratio in which a level error cancels.
- **`r_CpG = 1` is not just inert but correct.** Step 1's `fitted_po` is already keyed by
  methylation, so the low rate at hypomethylated CpG-island sites is already in E1. The
  apparent residual CpG decline is a `fitted_po` saturation artifact; corrected, true
  `r_CpG` is flat within +/-11% with no trend.
- **The CpG mechanism Peter proposed**: steps 1-4 confirmed (CpG models are fit without
  GC/methylation; high-GC CpGs are **92% hypomethylated in the top GC bin with a 1.9x
  lower DNM rate**, 0.532 -> 0.283; the model over-predicts there). Wider run: 90-100%
  above GC 0.70 and 2.7x, 0.53 -> 0.195 -- the bins differ because the GC edges span the
  window set's own range. Step 5 refuted -- the counterfactual holding non-CpG `r` at 1
  is flat, so none of it reaches Gnocchi.
- **The dnm0 background sample is not the cause.** Building the same empirical curve four
  ways, one ingredient at a time: denominator 2.4%, aggregation 4.3%, **window population
  37.6%**. The background sample IS non-uniform in GC (2.0-fold within a context) -- it is
  just not what changes the curve. *(Wider run; not recomputed on the narrowed set.)*
- **Training-set *size* is not the explanation either**, though it is a real effect:
  shrinking moves Gnocchi *toward* the context-only model (1% is indistinguishable from
  it) but never past it. The population fix goes past it. `dnm_training_size/` keeps that
  contrast. *(Wider run; not recomputed on the narrowed set.)*
- **Calibration-gap / reliability panels measure a LEVEL error**, which cancels in
  `r = sigma(b0 + b.z)/sigma(b0)` and never reaches the score. They diagnose the fit; only
  panel E measures the bias.
- **`fig_tables/comparisons_*.txt` (Extended Data Fig. 6) cannot answer the question.**
  Confirmed by downloading it: the files are a curated variant-classification set (GWAS /
  fine-mapped / pathogenic positives against AF-matched TOPMed negatives), keyed by
  `locus` not `element_id`, with no GC column, and scored in `z` rather than the residual.
  Ascertainment alone disqualifies it. `verify_comparisons_tables.py` reproduces this, and
  `verify_comparisons_tables.log` beside it is a real run's transcript — read that rather
  than re-downloading the tarball.

## The paper's Methods do not match the code — and the code is what ran

The published Methods state the adjustment factor as a ratio of raw **logits**,
`r = beta.x(w) / beta.xbar`, with the intercept excluded. The code
(`run_nc_constraint_gnomad_v31_main.py:209-249`) computes `logit.predict()` on a
`statsmodels` L1 result, which returns `sigma(linear predictor)` -- a **probability**.
So the operative formula is

    r(w) = sigma(b0 + b.z(w)) / sigma(b0)

a ratio of predicted probabilities, where `z(w)` is the standardized, PCA-transformed
feature vector for window `w`'s trinucleotide context, and the denominator is the model's
probability at the population mean (z = 0).

Confirmed empirically on a real fitted model, not just read from source:
`preconditions/verify_logit_predict_behavior.py` downloads one per-context `.pkl` and gets
`predict(zero_row) = 0.0394` (a probability) against `-3.1948` (the intercept) with
`which="linear"`. This is uncorrected: the one published Author Correction
(Nature 626:E1, DOI 10.1038/s41586-024-07050-7) fixes missing points in Supplementary
Figs 6-8 and says nothing about the formula. **Treat the code's probability-ratio formula
as ground truth**, and note that everything downstream depends on `r` being a ratio, since
that is what makes a level error cancel.

Two further consequences of `r`'s actual form, both load-bearing:

- `r` is fit **per trinucleotide context only**, never per (context, methylation).
  Methylation enters in step 1 alone. This is why `r_CpG ~ 1` is correct.
- The multivariate PCA+logit fit that produces `r` has **no published source anywhere** --
  the bucket ships fitted `.pkl`s and the apply side only. Everything here reimplements it;
  `preconditions/` is where that reimplementation is validated.
## Public data inventory (bucket `gs://gnomad-nc-constraint-v31-paper`, world-readable,
no auth needed — also fetchable via `https://storage.googleapis.com/gnomad-nc-constraint-v31-paper/<path>`)

Naming note: `genomic_features13` names the fixed panel of 13 candidate regional
features (`dist2telo, dist2cent, LCR, SINE, LINE, GC_content, recomb_male,
recomb_female, met_sperm, Nucleosome, CpG_island, cDNM_maternal_05M,
cDNM_paternal_05M`) — it does not imply the files are keyed by `feature` alone. The two
`*_sel*` files below are actually row-keyed by `(context, feature, window)`, since
selection is per-trinucleotide-context.

| File | Size | Contents |
|---|---|---|
| `misc/genomic_features13_genome_1kb.txt` | 1.44 GB | Raw x(w): 13 features × 4 window scales (1k/10k/100k/1M) = 52 columns, one row per 1kb `element_id` genome-wide. Includes `GC_content_1k`, `GC_content_10k`, etc. |
| `misc/genomic_features13_sel.txt` | 19 KB | One row per `(context, feature, window)` triple that survived Bonferroni selection for that trinucleotide context's L1-logit model (line ~209ff of `run_nc_constraint_gnomad_v31_main.py`) — i.e. the regional features actually used to compute `x(w)`/`x̄` and thus `r(w)` for that context. Columns: `context, feature, window, coef, se, pval` (`coef`/`se`/`pval` are the fitted logistic-regression coefficient, its standard error, and p-value). A context can have multiple rows (e.g. `AAA` has 3: `cDNM_maternal_05M`@1k, `dist2telo`@1k, `recomb_male`@1k; `AAT` has 6, spanning windows from 1k to 1M). |
| `fig_tables/genomic_features13_sel.annot.txt` | small | The full univariate table underlying the row above, not a strict superset of it: all 13 features × 4 window scales (52 rows) for every one of the 32 contexts (1664 rows + header), columns `context, feature, window, coef, ft_sel, label` (drops `se`/`pval`, adds `ft_sel`/`label`). `ft_sel` (bool) / `label` (`"x"` or empty) flag exactly the rows that survived Bonferroni selection — that subset is what `misc/genomic_features13_sel.txt` contains. E.g. context `AAT` has 52 rows here (13 features × {1k,10k,100k,1M}), of which 6 have `ft_sel=True` — matching the 6 `AAT` rows in the selected-only file. |
| `fig_tables/mutation_rate_by_context_methyl.txt` | 12.5 KB | Per-`(context, ref, alt, methylation_level)` mutation-rate table, **156 rows** — 96 `(context, ref, alt)` triples, of which only the four CpG contexts' C>T carry the 16 methylation levels (`run_nc_constraint_gnomad_v31_main.py` lines 86–148). `possible` = genome-wide count of such sites after coverage (30–32×) and black-region filtering; `observed` = those carrying a rare (AF ≤ 0.001) PASS variant in the 76,156-genome callset; `proportion_observed` = the ratio, which saturates below 1; `mu` = **not a measured rate** but the polymorphism proportion in a 1,000-genome downsample times one global constant (`s = 8.849e-7`, set so the per-base genome-wide mean is `total_mu = 1.2e-08`); `fitted_po` = `1 − exp(B)·exp(A·mu)` from regressing `log(1 − proportion_observed)` on `mu` (lines 137–141; `A = -1.885e7, B = -7.32e-5`, weighted R² = 0.9987). **`fitted_po` is the per-site step-1 probability the pipeline actually uses** — `expected = possible × fitted_po` at line 188. Its coalescent reading (`fitted_po = 1 − exp(−u·L_n)`, so the fit is a branch-length ratio) is derived in `fig5/fig5.ipynb`, Supporting Figure 7's section. |
| `fig_tables/constraint_z_genome_1kb.annot.txt` | 325 MB | Real, final (step-2, r-adjusted) genome-wide 1kb table: `element_id, possible, expected, observed, oe, z, pass_qc, coding_prop` + functional annotation columns (ENCODE cCREs, FANTOM enhancers, GWAS Catalog, etc.). `expected` here is **post-r-adjustment**. |
| `logit_pickles/logit_regularized_dnm01_{context}_pbonf_pca.pkl` | ~15–20 MB each | Fitted L1-logit model, one per trinucleotide context (32 contexts). |
| `logit_pickles/logit_regularized_dnm01_{context}_pbonf_pca.pca.pkl` | ~1 KB each | Fitted PCA transform (sklearn `IncrementalPCA`) per context. |
| `logit_pickles/logit_regularized_dnm01_{context}_pbonf_pca.ft_mean_std.txt` | ~150 B each | Per-context, per-selected-feature mean/std (this mean is x̄) used to standardize features before PCA. |
| `context_prepared.ht` | ~578 GB (38,029 partitions, 8,771,192,175 rows) | Hail `Table` keyed `(locus, alleles)`, **one row per *possible* SNV** — 3 per covered reference position genome-wide, polymorphic or not — carrying `context, ref, alt, coverage_mean, methyl_level, cpg` and a large unused `vep` struct. It is what gets grouped to produce the `possible` denominator (line 111). The actual call set is a separate table, `genome_prepared.ht` (line 38). **Superseded here** by `expected_counts_by_context_methyl_genome_1kb.txt` below — no longer needed; the evidence for the one-row-per-possible-SNV reading is in `preconditions/README.md`. |
| `expected_counts_per_context_methyl_genome_1kb.txt` | 3.3 GB (bucket root) | This *is* the exact `hl.export()` at `run_nc_constraint_gnomad_v31_main.py` lines 191–197: `expected_ht = possible_ht.group_by(key=(element_id, context)).aggregate(possible=sum, expected=sum)`, one row per `(element_id, context)` pair — multiple rows per window, one for each trinucleotide context that occurs in it (e.g. `chr1-10000-11000` has 4: `ACC, CCC, TAA, TAG`). Columns `element_id, context, possible, expected`, both **summed over every `(ref, alt, methylation_level)` combination sharing that context**: `possible` = count of possible SNV sites of this context in the window (after coverage/black-region filtering, lines 159–166); `expected` = `possible × fitted_po` per `(ref,alt,methylation_level)` (line 188, `fitted_po` from `fig_tables/mutation_rate_by_context_methyl.txt`), i.e. genome-wide expected counts from sequence context alone, `r ≡ 1`, computed *before* the regional-feature adjustment in lines 209–249. Sample: `chr1-10000-11000 / ACC → possible=3, expected=0.31501`. |
| `expected_counts_by_context_methyl_genome_1kb.txt` | 107 MB (bucket root) | **The step-1 (context-only, r ≡ 1) expected-count table**, one row per `element_id`: `element_id, possible, expected` — the row above summed over all 32 contexts, so `possible` matches the meaning it has in `constraint_z_genome_1kb.annot.txt`. No published script produces it (the pipeline only writes the per-context file), but `preconditions/verify_expected_r1.py` regenerates it genome-wide from that file and confirms the r ≡ 1 reading: `possible` exact on all 2,575,299 rows, `expected` to 4.6e-5 relative, with the residual explained by two pipeline runs — see `preconditions/README.md`. A Hail-native counterpart `.ht/` exists with the same 3-column schema. **Use this directly.** |
| `observed_counts_genome_1kb.txt` | 71 MB (bucket root) | Standalone observed-variant-count table, `element_id, variant_count`. Same numbers as the `observed` column of `fig_tables/constraint_z_genome_1kb.annot.txt` below, but much smaller if `pass_qc`/`coding_prop`/functional annotations aren't needed. |

Bucket contents are listable without `gsutil`/auth via the JSON API, e.g.:
```
curl -s "https://storage.googleapis.com/storage/v1/b/gnomad-nc-constraint-v31-paper/o?prefix=logit_pickles/&maxResults=50"
```

Reading `context_prepared.ht`, or any other `.ht`/`.mt`, needs a specific Hail/Java/backend
setup: the recipe and its two gotchas are in `preconditions/README.md`. Nothing in the
current analysis needs it.

## Methods narrative — moved

`METHODS.md` (repo root) holds the canonical methods text for the Figure-2A rank
statistic: what the statistic is, the GC units, chromosome/noncoding filters, the neutral
window set and the join that supplies it, axis ranges, and the window-count gap against
McHale et al. **Read it before writing any methods or rebuttal prose about the rank
statistic**, and before changing anything in `gnocchi_bias/windows.py`.

## Where to pick up

1. **Fig. 5 and Supporting Fig. 8 are built, run and captioned end to end.** As of
   `213d551` (2026-09-11) every panel is regenerated, every code cell in the notebook carries
   real output, and no placeholder remains in any manuscript file. The one thing outstanding
   is the Illustrator resize recorded below. `fig5/README.md` has the operational detail;
   `fig5/fig5.ipynb` carries the derivation of every plotted quantity and is committed with
   the **narrowed** run's outputs.
   **The committed notebook and PNG are one restyle behind `fig5/panels.py`.** The panels
   were made monochrome on 2026-08-27 -- A, D and E carry identity in marker shape, fill,
   dash pattern and presence of error bars rather than in colour, with two exemptions
   (panel C, which recolours to green-for-QC-pass / red-for-QC-fail, and panel B's
   R_eff-plus-counterfactual pair) -- panel D lost the size-matched control's two curves,
   and panel E now prints each curve's mean |rank - 0.5| in its legend. No number changed;
   `captions.txt` and `methods.txt` are already updated. **Panel D was then split into two
   stacked rows on 2026-09-01** -- empirical curves above, fitted below, over a shared x
   axis and a single shared y range, one curve per training population in each row, a
   population's marker and dash pattern the same in both. Its shape is now
   `panel_dnm_probability_pairs(ax_empirical, ax_fitted, binned, ...)`, the notebook cell
   builds a 2x1 like panel C's, and `_grouped_legend` went with the change (no other
   caller). Again no number changed, and `captions.txt`/`methods.txt` are updated.
   **Fig. 5F's axis became a percentile on 2026-09-05** -- detail below, and again no number
   changed, only which end of one is labelled.
   **The notebook is 41 cells, ALL 23 code cells carry real outputs, and no placeholder
   remains anywhere.** It was executed end to end on the HPC path at `035e144`
   (2026-09-11) after the left-tail rework, so every number in it is real; the count read 37
   here until that day and was stale by two even then.
   **HOW TO EDIT IT WITHOUT LOSING THAT.** Running `make_fig5_nb.py` over `fig5.ipynb`
   directly discards every output. The reliable way, used repeatedly on 2026-09-11: generate
   into a SCRATCH directory, then copy `outputs`/`execution_count`/`id` onto every cell whose
   SOURCE is byte-identical, and write that. A markdown-only prose pass then costs nothing --
   the last one rewrote two md cells and kept all 23 outputs -- while touching one comment
   inside a code cell costs that cell its output. Verify afterwards that the generator and
   the notebook agree cell-for-cell; that check has caught a mismatch more than once.
   The 2026-09-09 run cleared the
   rebuild the TWO 2026-09-08 revisions had left outstanding (Supporting Fig. 8, two panels
   -> four -> five, the last moving every cross-bin reading onto LR+); see WHAT THE RERUN
   SETTLED below for what it did and did not fill. The 37 count is after THREE cells were
   retired on 2026-09-09, all of them computed-but-undrawn diagnostics, and in every case
   THE BUILDER IS UNTOUCHED -- only the call site is gone, so each is one line to restore.
     * the `recall = lift x k` identity check, which stopped licensing its own conclusion
       once panel D moved to the odds ratio (see the RECALL IS DRAWN ONLY IN A AND B block
       below);
     * `lifts_s8` / latterly `gains_s8`, the paired bootstrap at the GLOBAL cutoff, with the
       block that printed it. No panel drew it and the figure's own argument says not to
       read it: at one global cutoff the two scores sit at different operating points in
       every bin, which is the confound panel D exists to remove, so a published-vs-retrained
       interval there measures the confound. 500 replicates a run. Restore with
       `data.paired_deltas(threshold=D.GNOCCHI_THRESHOLD, truth_set="lax", n_bootstrap=500,
       seed=0)`;
     * `sweep_s8`, the same comparison swept across calling rates (1%, 3%, 10%), printed and
       never drawn. Its builder was deleted 2026-09-11 (see below), so restoring it means
       recovering `paired_delta_sweep` from `5865e5d` first.
   **`data.paired_delta_sweep` WAS DELETED 2026-09-11.** It had no caller and was kept
   because its argument was still good -- it swept the comparison across calling rates (1%,
   3%, 10%) and so said whether a result was a knife-edge or held along the range. Supporting
   Fig. 8 now DOES that, at 1% / 50% / 99% and in the figure itself, so the function's whole
   purpose is served by drawn panels. That is exactly the condition this entry named for
   deleting it. At `5865e5d` if ever wanted.
   **WHAT THE RERUN SETTLED (2026-09-09, commit `3a77e4c`).** The three cells that had no
   output all ran -- the C/D table (`withinbin_s8` / `gains_wb_s8`, whose matched rate moved
   from 1.002% to a flat 1% and whose `metric` moved to `lr_pos`) and the Supporting Fig. 8
   build -- and `fig5/output/supp_fig8.neutral.{pdf,png}`
   are now the FIVE-panel 20 x 5 in version. Every committed output was current as of that
   commit. No `refit.py` rerun was needed and none is (no training population changed).
   **THE 2026-09-11 RUN (`035e144`) REGENERATED EVERY PANEL**, so no committed output is
   behind `panels.py` any more -- that includes the 2026-09-10 restyles of Fig. 5F and the
   ratio panels, which had been outstanding. `supp_fig8.neutral.{pdf,png}` is now the
   NINE-panel 13 x 11.5 in version. `fig5F.neutral` printed "unchanged, left alone", which is
   `resave_ai` saying the bytes did not move: 5F's numbers are stable across the rework, as
   they must be, it using no labels.
   Geometry WAS checked offline -- matplotlib 3.11 IS in `.venv` even though the system
   python lacks it, so a panel's frame can always be rendered here on a stand-in frame;
   only its DATA needs the HPC path. 5F's label measures 300.5 px against 354.2 px of axes
   (one line, no wrap) and its legend 425.7 px against 542.5 px after both cuts. The top-1%
   ratio panel (then 8D, now 8I) was previewed on the REAL committed ratios (1.377, 1.063,
   1.085, 1.330, 1.084 with their intervals), so its shape was verified and not merely its
   geometry.
   **TWO CELLS' COMMITTED OUTPUTS ALSO SHOW ROWS THE CODE NO LONGER PRODUCES**, since the
   GC-only arm was removed on 2026-09-10 without a rerun: panels A and B's `tm_s8` table
   still prints a `GC only` row per bin, and the `budget_s8` cell still prints
   "GC content alone ... lift 2.15". Those rows are HISTORY, not current output -- nothing
   cites them, and the same run clears them. No other number in either cell moves: removing
   an arm changes which rows exist, never the published or decontaminated rows beside them,
   because each score's threshold is set from its OWN quantile.
   The layout fix rode along and IS NOW RENDERED: the 3-column gridspec with its uniform
   `wspace=0.82` became a 5-column one with SPACER COLUMNS, `width_ratios=[1, 0.72, 1, 0.30,
   1]` and `wspace=0`, which is the only way one GridSpec holds two different gaps -- gap 1
   keeps its old absolute width (3.58 in) and the panels each gain ~0.67 in. It produced a
   2000 x 500 px figure with 7 axes. If `label_panels`' offsets look wrong against the wider
   panels, C/D's `x = -0.34` is the one to try at -0.30.
   In `fig5.neutral.ai`, Supporting Fig. 8's placed PDF now goes to **13 x 11.5** -- a
   resize, not a relink, and from seven panels to nine -- and panel D of Fig. 5 is 7.6 in
   tall rather than 4.6, matching panel C. **THAT ILLUSTRATOR EDIT IS STILL TO DO**; it is
   the only part of the rebuild no run can perform.
   **The top-1% ratio panel (then D, now 8I) has these numbers, written into the prose**:
   the LR+ gains are
   **+37.7% [+1.7, +93.8], +6.3% [+0.8, +12.3], +8.5% [+2.3, +15.9], +33.0% [+8.8, +66.4]
   and +8.4% [-27.5, +58.6]** from the lowest GC bin to the highest, **four of five** clear
   of 1.0. As predicted, every one EXCEEDS its old lift counterpart (+33.3 / +4.0 / +4.2 /
   +10.8 / +2.2 per cent), which was a different statistic -- never quote those. The
   within-bin PRECISION translation, which is the smaller number and the easiest thing to
   confuse with the odds ratio, is **1.30x in the most AT-rich bin drawn falling to 1.02x in
   the most GC-rich** (from the printed per-bin lifts 1.81 -> 2.36, 1.53 -> 1.59,
   1.28 -> 1.34, 1.13 -> 1.25, 1.12 -> 1.15; the base rate cancels, so the lift ratio IS the
   precision ratio). Note it FALLS across GC while the odds ratio does not, so the caption
   sentence that once read "rising to" was corrected when it was filled. A, B and E's numbers
   were already final before the rerun, from `tm_s8`'s committed output.
   **THOSE PER-BIN PRINTS WERE ADDED 2026-09-10 AND HAVE RUN.** Cell 18 (Fig. 5F) and cell
   27 gained blocks printing each placeholder name beside its value, which is what filled the
   last of them and what caught the Fig. 5F caption error recorded under item 2 below. Read
   such values off a PRINT, never off a PDF and never by backing them out of a rounded fold
   swing.
   One cosmetic note on the notebook diff, so it is not mistaken for damage: regenerating
   through `make_fig5_nb.py` drops the per-cell `"id"` fields and the `iopub` execution
   timings, because the generator hand-builds the notebook JSON and writes
   `nbformat_minor: 5` without ids. Only `nbconvert --execute --inplace` puts them back.
   Pre-existing generator behaviour, harmless, but it makes a small source edit look like a
   thousand-line diff.
   **Supporting Figure 8 was added to the notebook on 2026-09-02** (four new cells
   spliced in before "Numbers for the caption"; every other cell kept its committed
   outputs). **It has since become TWO figures**, split 2026-09-04 along the seam between
   threshold-free and fixed-threshold measurement, because nine panels at 13.5 x 24 in was
   one argument over five rows:

     Fig. 5F            The calling rate itself, PROMOTED OUT OF THE SUPPORTING FIGURES on
                        2026-09-04 because it uses NO LABELS: the fraction of windows in
                        each GC bin clearing Gnocchi >= 4, Chen et al.'s own cutoff. Built
                        from PANEL E's table and bins (`data.calling_rate_by_gc`,
                        `panels.panel_calling_rate`), so E and F are two views of one fix
                        on one population -- E's rank returning to 0.5, F flattening --
                        rather than two measurements that happen to agree. On
                        20 bins the offline reproduction gives published 0.13% -> 46% (and
                        0% in its most AT-rich bin) against a flat ~1% for the retrained
                        score. The two are matched on OVERALL calling rate, not given a
                        common z, since retraining moves the whole distribution.
                          SINCE 2026-09-05 THE AXIS IS A PERCENTILE, NOT THE CALLING RATE
                        (`percentile_axis=True`, the default; `False` restores the old
                        panel and nothing else changes). Same quantity read from the other
                        end -- percentile = 100 x (1 - calling rate) -- but it states the
                        claim directly: the percentile is LOCAL to the GC bin, so a score
                        meeting its own promise would put a fixed z at a fixed percentile
                        everywhere and a HORIZONTAL LINE IS THE NULL. Published Gnocchi's
                        z = 4 runs from the ~99.9th percentile of AT-rich sequence to
                        roughly the 58th of GC-rich. The axis stays LOGARITHMIC in the
                        calling rate underneath, because a linear percentile axis collapses
                        the AT-rich bins (99.87, 99.9, 100, 100) onto one line and loses
                        the panel; it is INVERTED so percentiles increase upward; its ticks
                        are set explicitly to round percentiles, since the log decades label
                        only the top; a bin calling nothing is MASKED rather than drawn as a
                        spike off the edge; and it carries a mean-GC line taken from panel
                        E's own frame so both panels mark the same place.
                          THAT NULL IS NOW DRAWN, added 2026-09-05: a dashed HORIZONTAL line
                        (`calling_rate_by_gc`'s third return value, passed as
                        `matched_rate`; `matched_rate_line=False` drops it). DEFINE IT
                        EXACTLY, because the legend has to be checkable: it sits at
                        k = |{w : z_s(w) >= t_s}| / |W| over the WHOLE population W, the
                        fraction of all windows clearing the cutoff -- one number for both
                        scores, since the matching gives the retrained score the quantile of
                        its own z attaining published's k. So THE LINE AND THE CURVES ARE
                        ONE QUANTITY OVER TWO POPULATIONS: the cutoff's percentile
                        genome-wide against its percentile within each GC bin. THE PANEL'S
                        WORDING FOLLOWS THAT, changed 2026-09-05 and again 2026-09-10: the
                        y axis now STATES THE CURVES' READING ("Gnocchi percentile in GC
                        bin"), the curve entries name the cutoff bare ("Gnocchi, published
                        (cutoff = 4.00)" and "Gnocchi, decontaminated (cutoff = 3.24)" --
                        the "z" was dropped on the percentile axis, since a percentile is a
                        property of a VALUE; the calling-rate branch, percentile_axis=False,
                        keeps "z >= 4.00", which names a SET and is ungrammatical without
                        it), and the line entry names the population and NOT its value
                        ("Genome-wide percentile common to both cutoffs" -- the "(99.34th)"
                        parenthetical went on 2026-09-10, the line's height being that value
                        on a labelled axis). The series names shortened the same day, from
                        "Gnocchi, as published" and "Gnocchi, decontaminated DNM training
                        set". PANELS A AND E FOLLOWED on the same day, and so did Supporting
                        Fig. 8's ratio panels, so "Gnocchi, published" and "Gnocchi,
                        decontaminated" are
                        now the names EVERYWHERE a legend carries them; "as published"
                        survives only in prose. The legend measures 426 px against 542 px of
                        axes after both cuts, down from 509. NOTE WHAT THAT COSTS, since it was a deliberate trade:
                        from 2026-09-05 to 2026-09-10 the axis read "Cutoff's percentile in
                        the Gnocchi score distribution", NEUTRAL between the two readings
                        because it cannot say "local" while carrying a genome-wide line. The
                        new label says the curves' claim directly and therefore does NOT
                        describe the dashed line, the one element whose percentile is
                        genome-wide -- so THE LINE'S LEGEND ENTRY IS NOW LOAD-BEARING AND
                        MUST NOT BE DROPPED, and the caption still states the split
                        outright. Saying "within GC bin" on the curve entries too runs the
                        legend to 629 px against 542 px of axes, so that half is carried by
                        contrast with the word GENOME-WIDE. Measure before
                        adding words to that legend. It is the null
                        STRICTLY: bin rates average to k, and percentile = 100(1 - rate) is
                        affine, so a curve flat across GC can only be flat ON this line. It
                        belongs to BOTH curves -- both are pinned to it -- so never caption
                        it as published's own level. It is DASHED at 0.30 grey, 1.6 pt, at
                        zorder 1.5: the panel's own gridlines are dotted rules and
                        set_axisbelow(True) puts them at 0.5, so a thin dotted line at 0.45
                        grey -- what it was for its first hour -- reads as one more gridline
                        and can be painted over by them. Expect it to run UNDER the
                        retrained curve through the GC bulk, where the two agree to 3.5% in
                        rate (about 1.5 pt on this axis); that is the panel saying the
                        retrained score IS the null there, not a missing line. It is
                        computed over EVERY window,
                        including bins `min_n` drops from the drawing, so it is the
                        operating point and not a mean of the plotted points; the retrained
                        curve therefore hugs it in the GC bulk and departs in the sparse
                        tails, since what is pinned is a window-weighted mean. THE POINT OF
                        DRAWING BOTH LINES is that they do not intersect the published curve
                        at the same place: published meets the horizontal line at GC ~ 0.43
                        against a mean GC of 0.393, because a near-exponential calling rate
                        has a window-weighted mean far above its value at a typical window.
                        The two curves cross each other at essentially that same GC (0.4332
                        against 0.4305 for the published-meets-its-own-mean point, the
                        0.003 gap being the retrained curve sitting 3.5% above the matched
                        rate there) -- they would coincide exactly if that curve were flat.
                        Note for anyone recomputing it: percentile = 100 x (1 - rate) is
                        AFFINE in the rate, so a WINDOW-WEIGHTED mean percentile is exactly
                        the matched one, while an unweighted mean over drawn bins is not
                        (93.15 rather than 99.34) -- the caveat is the weighting, not the
                        transform.
                          ONE z PER SCORE, APPLIED UNCHANGED IN EVERY BIN. This is the
                        OPPOSITE convention to Supporting Fig. 8's D-I, which fix the
                        RATE per bin and let the threshold move, and the two are easy to
                        confuse. (Supporting Fig. 8's A and B share 5F's convention, one
                        fixed global cutoff per score; it is D-I that invert it.)
                        `calling_rate_by_gc` fixes both thresholds ONCE on the
                        whole population, then counts per bin; so the legend reads `z =`
                        rather than `z >=` on the percentile axis, a percentile being a
                        property of a value and not of the set above it.
     Supporting Fig. 8  NINE panels in FOUR SLOTS, two rows of two (13 x 11.5 in) -- what
                        debiasing does to DISCOVERY, which unlike Fig. 5 needs a truth set.
                        RELAID OUT 2026-09-11, when the LEFT-TAIL CONSTRUCTION WAS RETIRED
                        (read that block below before anything else here) and the figure
                        became ONE comparison at THREE matched calling rates. A slot is
                        either a stack or one full-height panel --

                          row 1   A over B          where each score sends its calls
                                  C                 the threshold-free verdict, full height
                          row 2   D over E over F   the cutoff at 99 / 50 / 1% of a bin
                                  G over H over I   the gain at 99 / 50 / 1%

                        -- so a reader meets the operating-point picture, then the verdict
                        over all thresholds, then WHERE along the ranking that verdict is
                        averaging. The two row-2 stacks are ROW-ALIGNED BY CALLING RATE (D
                        with G, E with H, F with I) and ordered LEFT TAIL -> MEDIAN -> RIGHT
                        TAIL top to bottom, which is why the rate DESCENDS: the cutoff
                        isolating the top 99% of a bin sits at z ~ -5, the top 50% IS the
                        median, the top 1% is z ~ +3 to +6. Reading down the left stack gives
                        three quantiles of one distribution (so a published cutoff climbing
                        at all three means the bias is a shift of the WHOLE score, not a
                        stretched tail, and E is a CALIBRATION CHECK needing no truth set --
                        a well-calibrated z has its median near 0 in every bin); reading down
                        the right stack sweeps C's recall axis.
                        EVERY LETTER HAS MOVED AGAIN, third time. The map:
                        old D (cutoff, top 1%) -> F; old E (ratio, top 1%) -> I; old F
                        (cutoff, bottom 1%) -> D, reread as the cutoff calling the top 99%,
                        SAME FIVE NUMBERS; old G (ratio, bottom-1% mirror) -> G but
                        RECOMPUTED, ITS NUMBERS DO NOT CARRY OVER; new E (median) and new H
                        (mid-recall ratio). Anything written before 2026-09-11 means the old
                        letters. The column gap is a SPACER COLUMN at the same 3.58 in the
                        old 3-column layout was tuned to, `width_ratios=[1.0, 0.76, 1.0]` on
                        13 in; `height_ratios=[1.0, 1.35]` favours row 2, which carries
                        three-panel stacks, and both stacks are at `hspace=0.42` because a
                        rotated label on a third-height row collides at 0.35.
                          THE LEFT TAIL HAS NO TRUTH SET, AND THAT IS WHY THE OLD F AND G
                        WENT (2026-09-11, Peter's objection; the note above
                        `data.LAX_CALL_RATES` records it). Until then the figure read the
                        BOTTOM 1% with the call at z <= t and a NON-enhancer as the hit,
                        justified by saying a low Gnocchi claims a window is unconstrained.
                        IT DOES NOT. Gnocchi is a two-sided z against a neutral expectation,
                        so UNCONSTRAINED SEQUENCE SITS AT z ~ 0 -- most of the genome -- and
                        the left tail is the OPPOSITE anomaly, MORE variation than expected,
                        whose leading explanations are hypermutability or mutation-model
                        misspecification. The repo's own numbers agree: under the neutral null
                        the 1st percentile would be z = -2.33, and published's bottom-1%
                        cutoffs run -5.67, -6.04, -5.03, -4.14, -2.98, so four of five bins
                        are far heavier than sampling noise. A non-enhancer label is evidence
                        about ENHANCER STATUS, so the mirror swapped one hypothesis for
                        another. DO NOT REINSTATE IT. The question it was built to answer --
                        where does C's wash come from -- survives, and is now asked with the
                        call at z > t and an enhancer as the hit at every rate.
                          THE SWITCH COSTS NO EVIDENCE, ONLY MAGNITUDE, and this is worth
                        knowing before anyone calls the new panel G a null. With
                        a = P(z <= t | enhancer) and b = P(z <= t | non-enhancer), the retired
                        reading was b/a and the kept one is (1 - a)/(1 - b). At a matched rate
                        inside a bin they are a MONOTONE REPARAMETRISATION of one 2x2 table,
                        so they order the scores identically in every bin AND every bootstrap
                        replicate; the sign test and the significance calls carry over, and
                        only the numbers compress, by a factor of order k. On the reconstructed
                        numbers, since confirmed by the run: 42.2% of effect in the retired
                        reading against 0.53% in the kept one, 5/5 bins agreeing. So READ G FOR SIGN AND H AND I FOR
                        MAGNITUDE -- and 50% is where most of C's area lives, so H is the
                        panel that can actually account for C's wash, which the 99% one
                        cannot (there both scores' precision is within ~1% of prevalence).
                          PANEL G IS PINNED NEAR 1.0 BY ARITHMETIC, AND THAT IS NOT A NULL.
                        LR+ = odds(p)/odds(r) at ANY calling rate, so a rate near 1 -- which
                        forces the precision p to the base rate r -- drives it to 1: at
                        k = 0.99, LR+ = 1 + (recall - k)/(1 - r) + O((1-k)^2), bounded by
                        1 + (1-k)/(1-r). Cell 31 prints that check against the exact values.
                        THE EVIDENCE IS NOT COMPRESSED WITH THE MAGNITUDE: at a matched rate
                        inside a bin, n, n_pos and n_called are fixed and shared, so the 2x2
                        table has ONE FREE COUNT and precision, lift, skill and LR+ are all
                        monotone in it -- they agree on which score is ahead in a bin and in
                        every bootstrap replicate. So read G for SIGN and H and I for
                        MAGNITUDE, and note that 50% is where most of C's area lives, so H is
                        the panel that can account for C's wash where G cannot.
                          NO LR- MATERIAL SURVIVES, 2026-09-11, and do not reintroduce it. An
                        `lr_neg` column, a `_neg_likelihood_ratio` helper and a two-readings
                        diagnostic cell existed for part of that day, to show that switching
                        from b/a to (1 - a)/(1 - b) cost no evidence. The argument is right
                        and is kept -- it is the one-free-count paragraph in
                        `data.paired_deltas` -- but it needs no second statistic, and printing
                        b/a resurrected the retired hypothesis every time the cell ran. The
                        LR- contrast had ALREADY been cut from the prose once, at `5865e5d`.
                          WHAT SURVIVES NUMERICALLY: panel I is the old E, so its
                        +37.7% [+1.7, +93.8], +6.3% [+0.8, +12.3], +8.5% [+2.3, +15.9],
                        +33.0% [+8.8, +66.4], +8.4% [-27.5, +58.6] STAND, as do A, B and C's
                        numbers and the old F's five cutoffs (now D's). WHAT DOES NOT: every
                        number attached to the old G -- the -34.3 / -22.9 / -23.3 / -42.1 /
                        -41.5 per cent reversal and published's 3.23-5.45 LR+ levels. Those
                        measured the retired hypothesis.
                          ALL NINE PANELS HAVE NOW RUN (2026-09-11, commit `035e144`), and
                        the notebook is committed with every code cell's output. THE CURVES
                        CROSS BETWEEN THE 1ST AND 50TH PERCENTILE -- not between the top and
                        the bottom, which is where the retired construction put it. Per bin,
                        decontaminated over published:

                          top 1%    +37.7  +6.3  +8.5  +33.0  +8.4    5/5 favour decon, 4 sig
                          top 50%    +0.1  -1.3  -3.1   -3.0  -3.0    1/5 favour decon, 4 sig
                          top 99%    -0.2  -0.1  -0.2   -0.4  -0.5    0/5 favour decon, 4 sig

                        So the gain is CONFINED TO THE EXTREME TOP of the ranking and is paid
                        for by a small, significant deficit through the rest of it. THE
                        "TRADE" CONCLUSION RETURNS, differently located and better for the
                        paper, since the top ~1% is the use case.
                          KEEP IT IN PROPORTION, AND THE PROSE DOES. At the median neither
                        score discriminates much -- LR+ 1.19-1.39 for both -- so the largest
                        deficit is 1.23 -> 1.19, a loss of 0.04 in LR+, against 1.95 -> 2.69
                        at the top of the same truth set. The middle of the ranking is where
                        a Gnocchi-like score is least informative and it is what debiasing
                        gives up. Both deficits GROW WITH GC (-0.1 -> -0.5 and -1.3 -> -3.0),
                        which locates whatever the retrained model discards in GC-rich
                        sequence, where the regional adjustment varies most.
                          8H CORROBORATES 8C BIN FOR BIN: C's one significant bin favours
                        published at GC 0.40-0.50, which is exactly where H's deficit is
                        largest (-3.1%). Two independent constructions landing on one bin.
                          PANEL E IS THE CHEAPEST RESULT IN THE FIGURE and the most quotable.
                        Published's median z runs -0.96, -0.32, +0.50, +1.41, +2.32, a
                        MONOTONIC swing of 3.28; the retrained score's stays within 0.54 of
                        itself (-0.22 to +0.32). So in the most GC-rich bin HALF of published
                        Gnocchi's windows already sit above z = 2.3, with Chen et al.'s cutoff
                        only 1.7 further on, while in the most AT-rich its median is BELOW
                        zero. No truth set, no cutoff anyone chose.
                          BUT READ E'S SWING, NOT ITS OFFSET FROM ZERO, and the captions say
                        so (Peter's instruction, 2026-09-11). This population RETAINS the
                        enhancer-overlapping windows it classifies -- 30.9% overall, 63.9% of
                        the top GC bin -- so it is NOT putatively neutral and a median
                        displaced from zero is not purely calibration error: genuine
                        constraint in the positive class displaces it too, and so does the
                        truth set's GC skew. What those confounders cannot produce is the
                        DIFFERENCE BETWEEN TWO CURVES SCORED ON IDENTICAL WINDOWS, +2.32
                        against +0.32. By the same token the retrained score's residual +0.3
                        is a LEVEL common to every bin, not a GC dependence, so not a
                        residual bias.
                          8G's ALGEBRA HELD: |LR+ - 1| <= 0.0152 and the first-order form
                        tracks the exact values to 3e-4, reproducing the offline
                        reconstruction to the last digit (1.0074, 1.0101, 1.0099, 1.0141,
                        1.0152 published; 1.0056-1.0097 decontaminated). So read 8G for SIGN
                        and 8H and 8I for MAGNITUDE, as the panel comments say.
                          STILL TRUE OF ANY OF THESE PANELS: within a bin the GC bias is
                        nearly a CONSTANT SHIFT, which cannot reorder a ranking, so a gap at
                        a matched rate is NOT the bias but within-bin variation in the
                        regional adjustment -- information published carries and the
                        retrained score discards along with the bias. Unsettled, and the
                        13-feature panel includes `CpG_island` and `Nucleosome`, which
                        correlate with regulatory annotation, so that component is not
                        independent of a GeneHancer truth set.
                          THE LABEL-FREE TEST FOR WHAT IS ACTUALLY IN THE LEFT TAIL is a DNM
                        one, not a truth-set one: if those windows are hypermutable their de
                        novo rate is elevated, which `gnocchi_bias/dnm_model.py`'s
                        per-stratum machinery already measures. Circular against the
                        retrained score (the DNM set is what r is fit on), so it wants the
                        held-out split already listed as optional hardening. NOT BUILT.
                                                  LR+ EVERYWHERE THE READING IS ACROSS BINS, AND THIS IS THE CHANGE
                        THAT MATTERS MOST. Lift is precision/r, r climbs 7.7x across these
                        bins, and the ceiling 1/r falls 12.0 -> 1.57 -- so a declining lift
                        curve is partly the ceiling coming down. Not a matter of degree: on
                        the SAME ROWS lift falls 1.80 -> 1.12 while ceiling-free skill RISES
                        0.073 -> 0.218, so the choice of measure decides the SIGN of the
                        trend. LR+ = P(call|Y=1)/P(call|Y=0) conditions on the true class on
                        both sides and r cancels outright. Lift survives ONLY as the
                        within-bin translation a caption quotes. Do not put it back into any
                        panel.
                          THE PER-PANEL PARAGRAPHS BELOW PREDATE 2026-09-11 AND EACH NOW
                        CARRIES ITS NEW LETTER IN ITS HEADING. Where the body text of one
                        refers to another panel by letter, apply the map in the header block
                        above. A, B and C did not move.
                          A and B: ONE PANEL PER SCORE -- A published, B retrained -- each
                        carrying RECALL AND LR+ TOGETHER on twin y axes (recall left and
                        logarithmic, LR+ right and linear), `panels.panel_recall_and_
                        enrichment`. THE COMPARISON IS INSIDE A PANEL, not between them:
                        does this score send its calls where a call is worth anything? Both
                        share both y ranges, so the shapes are comparable by eye. Read at
                        ONE FIXED GLOBAL CUTOFF PER SCORE (published at Chen et al.'s z >= 4,
                        retrained at z >= 3.140, each calling 10,051 of 1,003,036 windows).
                        A's two curves RUN IN OPPOSITE DIRECTIONS -- recall climbing
                        0.33 -> 15.07% across GC (45.7x) while LR+ falls 2.06 -> 1.25, so
                        published calls most where a call is worth least, 2,618 calls where
                        LR+ is 1.25 against 38 where it is 2.06. B's recall is FLAT,
                        2.03 -> 0.93%, and the calls move with it: 168 against 38 in the
                        AT-rich bin, 154 against 2,618 in the GC-rich. B's LR+ still declines
                        3.11 -> 1.50, and that residual is E's signal-to-noise, NO part of
                        the bias -- do not read B as "still not fixed". DO NOT impose a
                        per-bin rate here: the per-bin freedom IS the bias, and deleting it
                        deletes what A and B measure. WILSON bars, not the paired interval,
                        because the two panels sit at different operating points by
                        construction and a gap BETWEEN them is the confound D removes; their
                        claim is each curve's SHAPE. DO NOT read the two panels against each
                        other -- retrained LR+ is higher in four bins of five (3.11, 1.69,
                        1.80, 1.50 against 2.06, 1.57, 1.44, 1.25) and marginally LOWER in
                        the second (1.91 against 1.94), but that comparison is D's, cleanly.
                        Recall earns a place here and nowhere else, because Bayes gives
                        lift = recall / k and only here does k vary (45.7x).
                          F (WAS D): the per-bin threshold calling 1% of that bin, one curve per
                        score (`panels.panel_bin_thresholds`, new). The bias in the score's
                        own units, and like Fig. 5F it uses NO LABELS at all. It is 5F's
                        INVERSE, not its repetition -- 5F fixes the threshold and reads the
                        calling rate, C fixes the rate and reads the threshold -- and it
                        sits here because it is D's x-axis made visible, D's gains being
                        measured at exactly these thresholds, which is why the two share an
                        x axis. It carried a dashed z = 4 horizontal until 2026-09-08,
                        dropped because a rule crossing one row of a stacked pair reads as
                        a gridline belonging to both. NO error bars: a quantile of a
                        million windows has none worth drawing.
                          I (WAS E, AND BEFORE THAT B): the ratio panel, now the BOTTOM of the
                        right-hand stack rather than beneath the 1% threshold panel; since the
                        second 2026-09-08 revision it is a RATIO CURVE rather than two
                        levels (`panels.panel_lr_ratio`, new): the retrained score's LR+ over
                        published's, per GC bin, against a reference line at 1.0, with the
                        calling rate matched WITHIN each bin. Two level curves would invite
                        a cross-bin reading of each score against itself, which is A and B's
                        job; D asks only whether retraining helps IN a bin. IT IS THE ODDS
                        RATIO: within a bin the base rate cancels from either measure, so the
                        lift ratio IS the precision ratio, but a fold increase in precision
                        is bounded by 1/precision, a ceiling falling ~14x across these bins,
                        whereas odds(precision) = LR+ x odds(r) makes the LR+ ratio exactly
                        the odds ratio and comparable bin to bin. Hence `data.paired_deltas`'
                        default moved to metric="lr_pos". THAT FUNCTION WAS `lift_deltas`
                        UNTIL 2026-09-09, and `paired_delta_sweep` was `lift_delta_sweep`;
                        both were renamed once no caller asked for lift, `paired` naming the
                        property that does not vary and matching `pr_curve_deltas` beside
                        them. The notebook locals went too (`lifts_s8` / `lifts_wb_s8` are
                        now `gains_s8` / `gains_wb_s8`). Anything written before that date
                        means these functions under the old names. NOTE WHAT THE RETURNED
                        COLUMNS DO: `lift_*` and `lr_pos_*` are BOTH always populated with
                        the observed levels whatever `metric` says, and only `delta` and its
                        interval follow `metric`. That is a convenience and NOT a trap --
                        nothing downstream reads `lift_*` any more (8G, 8H and 8I take
                        `delta`,
                        the function's own print line
                        indexes `f"{metric}_published"`), the one reader that named it
                        directly having been retired on 2026-09-09 with `gains_s8`. KEEP
                        BOTH PAIRS ANYWAY: with the rate matched inside a bin,
                        `lift_scored / lift_published` IS the precision ratio, which is the
                        caption's analyst-legible translation, while `delta` on the lr_pos
                        default is the ODDS ratio and larger. The precision ratio is what
                        the caption quotes as 1.30x falling to 1.02x; keep both pairs, since
                        dropping `lift_*` would cost that translation.
                        NO LEGEND SINCE 2026-09-10, and the YLABEL IS NOW THE ONLY THING
                        NAMING THE CURVE: "LR+ ratio (decontaminated / published)", where it
                        read "retrained / published" before. One series, so a legend restated
                        the ylabel in smaller type; its two entries are each better placed
                        elsewhere -- direction in the label, the 95% paired interval in the
                        bars and the caption, the null in a dashed line whose height is 1.0
                        on a labelled axis. The y headroom above the highest bar went from
                        +42% to +8% with it: that band existed ONLY to hold the legend (this
                        panel has no empty corner by construction) and left the curve
                        compressed into the lower half once the legend went. `legend_loc`
                        left `panel_lr_ratio`'s signature at the same time, unused.
                        D'S NUMBERS, computed 2026-09-09, are +37.7% [+1.7, +93.8],
                        +6.3% [+0.8, +12.3], +8.5% [+2.3, +15.9], +33.0% [+8.8, +66.4] and
                        +8.4% [-27.5, +58.6] from the lowest GC bin to the highest, four of
                        five clear of 1.0. Every one EXCEEDS its old lift counterpart
                        (+33.3, +4.0, +4.2, +10.8, +2.2 per cent), exactly as predicted,
                        since the odds ratio amplifies wherever precision is high and this
                        truth set reaches 0.73 in the top bin -- those old figures are a
                        DIFFERENT STATISTIC and must never be quoted for these. Carries the
                        paired interval.
                          C: auPRC / positive-class fraction vs GC, THIRD because it is the
                        figure's CAVEAT and not its premise. Threshold-free and therefore
                        nearly blind to the bias, since a GC-dependent shift is a common
                        shift within a narrow bin and cancels from a ranking; its finding is
                        that the auPRC decline with GC SURVIVES debiasing, so that decline is
                        signal-to-noise. Its numbers are unaffected by anything above -- no
                        calling rate enters auPRC.
                          ITS LEGEND IS TWO NAMES, since 2026-09-10. The pooled values
                        (1.321, 1.341) and the clause naming the bars a 95% paired CI used
                        to travel in the labels -- two clauses per entry on a panel a third
                        of the figure's width -- and are now the CAPTION'S, where the pooled
                        numbers can be quoted; the cut took the legend from roughly the full
                        axes width to 233 px against 542. ITS y = 1 REFERENCE ALSO MOVED to
                        panel D's MATCHED_RATE_LINE_KW at zorder 1.5, from REF_LINE_KW: both
                        panels draw a null every marker is judged against, and at 0.45 grey
                        and 0.8 pt a dashed rule is barely heavier than the dotted gridlines
                        it sits among and, at set_axisbelow's zorder 0.5, can be painted
                        over by them. DELIBERATE DIVERGENCE from the rank = 0.5 and r = 1
                        references in Fig. 5A, B and E, which keep REF_LINE_KW -- there the
                        line is context for a curve read alone, here it is the frame a
                        comparison is made in.
                          THE RETIRED BLOCK THAT STOOD HERE described the old F and G (the
                        bottom-1% mirror) at length: its result, its levels, its mechanics,
                        the LR- contrast and the "do not restore the LR- passage" note. All
                        of it is superseded by the header block above and was deleted
                        2026-09-11 rather than marked, because its numbers measured a
                        hypothesis the figure no longer makes and a reader skimming would
                        have quoted them. Recover it from git history if the reasoning is
                        wanted; the note above `data.LAX_CALL_RATES` keeps the part that still matters.
                        TWO THINGS FROM IT ARE STILL LIVE AND ARE WORTH RESTATING. First,
                        the MIRROR WAS ONE KEYWORD -- `tail="lower"` flipped the call and the
                        label and nothing else -- so a tail difference could not have been an
                        artefact of two constructions; the machinery itself is deleted.
                        Second, LIFT IS
                        USELESS ON A LOW-TAIL READING (its ceiling is 1/r on the NON-enhancer
                        rate, 1.1 in the AT-rich bin), which is part of why LR+ is the measure
                        everywhere the reading is across bins.
                          THREE OPERATING POINTS NOW, AND CONFUSING THEM WITH A AND B IS THE
                        WAY TO MISREAD THIS FIGURE. A and B are ONE FIXED GLOBAL CUTOFF per
                        score, each score read at its own; D-I match the rate WITHIN each bin,
                        at 1%, 50% and 99% (`data.LAX_CALL_RATES`, new 2026-09-11 --
                        `LAX_CALL_RATE` remains the 1% one and is still what the old callers
                        pass). Since 2026-09-08 that 1% is NOT inherited from z >= 4 -- it
                        used to be 1.002%, the fraction published calls at that cutoff -- so
                        z = 4 now enters this figure only as A's actual cutoff.
                        `data.threshold_metrics` gained a `call_rate` parameter to make that
                        possible, and `panels.panel_lr_ratio` now takes the RATE (to label
                        the axis) where it took a `tail`.
                          RECALL IS DRAWN ONLY IN A AND B, AND PRECISION NOWHERE. A recall
                        panel beside D would be D rescaled by a constant -- that is why the
                        recall panel briefly numbered 8D on 2026-09-05 was cut the same
                        day. THE IDENTITY
                        CELL THAT VERIFIED recall = lift x k WAS RETIRED 2026-09-09, and the
                        reason is worth keeping: once D plotted the ODDS ratio, "a recall
                        panel would be D in other units" stopped being true, because recall
                        rescales the LIFT (= precision) ratio and not the odds ratio. The
                        caption's translation is therefore a PRECISION ratio and is SMALLER
                        than what D draws; quoting one for the other is the easiest error to
                        make with this panel. That translation is now FILLED: the
                        per-bin lifts are 1.81 -> 2.36, 1.53 -> 1.59, 1.28 -> 1.34,
                        1.13 -> 1.25 and 1.12 -> 1.15, so the precision ratio FALLS across
                        GC, 1.30x in the most AT-rich bin drawn to 1.02x in the most
                        GC-rich, while the odds ratio D draws does not -- the caption's
                        "rising to" was corrected to match when it was filled.
                        The letter C means the THRESHOLD panel, not that retired recall panel.
                          D DOES NOT USE Gnocchi >= 4, and the caption said it did until
                        2026-09-05. Each score is cut at its own 99th percentile WITHIN each
                        bin (`data._bin_thresholds`), so published's threshold moves bin to
                        bin -- panel C is that movement drawn -- and equals 4 nowhere in
                        particular, far above it in the top GC bin where an unmatched 4
                        calls 13.97%. Contrast A, B and Fig. 5F, one fixed z everywhere.
                        THERE IS STILL NO SUPPORTING FIG. 9; it existed for a few hours on
                        2026-09-04 and was merged back once its calling-rate panel moved to
                        Fig. 5F. The lower tail was briefly a standalone `lower_tail` figure
                        on 2026-09-10 before becoming this figure's F and G; nothing writes
                        `output/lower_tail.*` any more. Its orphaned `output/supp_fig9.*` were removed 2026-09-05,
                        and `fig5/README.md` -- which had announced it and denied it three
                        paragraphs apart -- was reconciled then.

   All of it lives in `fig5/data.py` and `fig5/panels.py` beside Supporting Figure 7's,
   with no module or entry point of its own. FOUR panel functions the cuts retired --
   `panel_pr_curves`, `panel_aupr_delta`, `panel_lift_vs_recall` and, since 2026-09-09,
   `panel_threshold_metric` -- moved to the GITIGNORED `fig5/panels_extra.py` so that
   reading `panels.py` means reading what is published; the first three were last tracked
   at `582c09d`, `panel_threshold_metric` at `54c7a76`. It went last because the second
   2026-09-08 revision split its job into shapes it cannot draw: A and B put TWO metrics on
   ONE score and need twin axes, D plots a paired RATIO rather than two levels. `panels.py`
   now has NO undrawn panel function, and its module docstring says so without exceptions.

   NAMES ARE ORGANIZED ON LAX vs STRINGENT, McHale et al.'s own vocabulary for their two
   truth sets. Per-truth-set constants carry the name (`LAX_GC_BINS`,
   `LAX_MIN_BIN_WINDOWS`); shared ones do not (`PR_SCORES`, `TRUTH_TARGET`). Do not put
   "enhancer" back into these names -- that is how the LAX set is defined, not what the
   section is about.

   **THE STRINGENT TRUTH SET WAS CONSIDERED AND DELIBERATELY NOT BUILT (2026-09-04). Do not
   pick it up as pending work.** `data.pr_curves` still takes `truth_set` and still accepts
   only `"lax"`; that seam stays, and the spec is still in `fig5/data.py`'s section header
   if the decision is ever revisited. Four reasons, in order of weight:

     * THE CENTRAL RESULT USES NO TRUTH SET. Fig. 5F is a property of the score and of GC
       content. Swapping GeneHancer for essential-gene enhancers cannot move it. That is
       also why it was promoted into the main figure.
     * IT IS UNDERPOWERED FOR THE COMPARISON WE MAKE. 4,933 windows against 1,003,037. The
       published-vs-retrained pooled gap on the lax set is +1.5% (auPRC/r 1.321 -> 1.341);
       the bootstrap sd on 4,933 windows is several times that, so a null would be
       uninformative. McHale et al.'s Fig. 4C effect is large because it compares TRUTH
       SETS, not scores.
     * IT WOULD NOT ESCAPE THE TRUTH SET'S GC SKEW, which is the objection the lax set
       actually raises. Stringent positives are essential-gene enhancers, GC-rich, against
       non-enhancer negatives, AT-poor -- so plausibly MORE GC-separated, not less.
     * IT IS NOT CHEAP: a third hand-supplied file, an interval-overlap join onto Chen's
       1 kb grid with a spanning rule, and new builders for a bootstrap statistic that is
       not `pr_curves`.

   A GC-MATCHED NEGATIVE SET WAS RECOMMENDED HERE UNTIL 2026-09-10 AND IS NOW REJECTED.
   DO NOT BUILD IT. The idea was to resample negatives to match the positives' GC
   distribution, making a GC-only classifier's lift 1 by construction, so that any score
   above 1 was discriminating on something other than GC. Two objections killed it, both
   Peter's:
     * IT REPEATS THE DEFECT THIS FIGURE REJECTS DOWNSAMPLING FOR. Supporting Fig. 8's
       caption declines to equalise prevalence between bins because "a balanced precision is
       a number no one will ever encounter"; a GC-matched population is the same move --
       discovery metrics computed on a genome that does not exist.
     * THE GC-ONLY CLASSIFIER IS NOT A COMPARATOR WE WANT. The claim is published against
       decontaminated. That comparison is PAIRED and computed WITHIN a GC bin, so whatever
       makes GeneHancer easy is a property of the truth set both scores face alike, and a
       score carrying no constraint information adjudicates nothing between two constraint
       scores.
   THE WHOLE GC-ONLY ARM WENT WITH THE RECOMMENDATION on 2026-09-10 -- `GC_BASELINE`,
   `include_gc_baseline` and the third row of `budget_comparison`, plus every mention in
   captions, results, README and the notebook's markdown. `data.py`'s `_score_column` keeps
   a note saying why. If it is ever wanted back it is at `5ac14fa`.

   WHAT REPLACES IT IS AN ARGUMENT, NOT A CONTROL, and it is stronger: the truth set's GC
   skew hands PUBLISHED a tailwind, since published is the GC-biased score and GeneHancer
   positives are GC-rich -- between bins, and still within them, where positives stay
   enriched at the high-GC end. Decontaminated wins in all five bins ANYWAY, so Supporting
   Fig. 8I is CONSERVATIVE. Say that rather than omitting it; a reviewer of McHale et al.
   will notice the secondary analyses rest on the set their own paper calls lax, and the
   strong form is a caption sentence carrying this plus the power argument above, not
   silence.

   ONE PREDICTION WORTH RECORDING, since it would be the cheap test if anyone does build
   the stringent set: their Fig. 4D measures the GC dependence of ranking performance,
   which Supporting Fig. 8 says survives debiasing -- so the two scores' 4D bars should be
   STATISTICALLY INDISTINGUISHABLE. A null there confirms Supporting Fig. 8 rather than
   failing to find something.

2. **`fig5/captions.txt` covers Fig. 5A-F and Supporting Fig. 8's (A)-(I), and NO
   PLACEHOLDERS REMAIN ANYWHERE** -- in it, in `results.txt`, or in the notebook's prose. The
   last four (`[8E-PUB-MEDIANS]`, `[8E-SCORED-RANGE]`, `[8G-RESULT]`, `[8H-RESULT]`) were
   filled 2026-09-11 from the run at `035e144`; values in the Supporting Fig. 8 block above.
   Fig. 5F's were filled from cell 18's own per-bin print, which THE RUN ALSO TURNED INTO
   PRINTED FACTS: `[F-HI]` = 41.667% and `[F-PCTL-HI]` = 58.333 had been INFERRED from a
   printed maximum and are now confirmed, and `[F-LO]` / `[F-PCTL-LO]` came out at 0.000% /
   100.000 because the most AT-rich bin DRAWN (GC 0.250, 700 windows) calls nothing at all.
   **THAT CAUGHT A REAL ERROR IN FIG. 5F's CAPTION**, which predated the rework: it called
   0.150% / the 99.850th percentile "the most AT-rich windows drawn", and that is the SECOND
   bin. Reworded to "the most AT-rich bin in which it calls anything", with the zero-calling
   bins named -- two exist, the most AT-rich for published and the two most GC-rich for the
   retrained score -- since quoting 0.000% as an endpoint reports a BIN'S SIZE rather than a
   score's behaviour. `[F-FLAT]` is 0.000%-0.693% over drawn bins, mean 0.498%, which is NOT
   the 0.660% matched rate: that mean is unweighted over bins and the matching is
   window-weighted. The old `[8D-*]` group and `[8F-SUMMARY]`/`[8G-RESULT]` names belonged to
   panels the rework retired; do not go looking for them.
   **The manuscript text is THREE files**, all paragraph-per-line for pasting, all on the
   narrowed run: `fig5/captions.txt` (Fig. 5 and Supporting Figs 7-8), `fig5/methods.txt`
   (the Methods subsection "How Gnocchi's regional adjustment drives its GC bias"), and
   `fig5/results.txt` -- FIVE paragraphs since 2026-09-11, up from three, when the
   Supporting Fig. 8 paragraph was split in two and a left-tail paragraph added; the longest
   fell from 4009 to 2175 characters. `METHODS.md` covers the rank statistic and points at
   them.
3. **Optional hardening**: a held-out DNM split would make panel D out-of-sample (panel E
   already is, on gnomAD counts the DNM model never sees).
4. **`DEPLETION_RANK_BED` and `NEUTRAL_WINDOWS_BED` are both set** in `fig5/config.py` and
   both have been run against their real files, on the constraint-tools HPC path -- which
   is where the figure must be rebuilt, since neither file is available offline. Outputs of
   the two window sets do not collide: refits, provenance entries and panel PDFs all carry
   `config.WINDOW_SET_SUFFIX` (`.neutral`). **Run `fig5/preflight.py` first** after any
   change to either path -- it checks both files' schemas in seconds and fails loudly on
   the quiet errors (chromosome naming, 1-based coordinates, a constant enhancer flag).
   The local `refits/` are the **wider** run's and carry the old provenance schema, so a
   narrowed-run panel cannot be rebuilt here without rerunning `refit.py`.
5. **Before quoting anything in the rebuttal**, re-read the callability caveat above:
   it brackets the over-adjustment across 1.22-1.44, so the figure must not be
   captioned with 1.22 as though it were tight -- and note that block is wider-run only.
