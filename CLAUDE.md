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

Numbers below are over the GC bins the panels actually draw (n >= 100 windows), which is
what `fig5` reports. The same statistic over all 20 bins gives 0.130 / 0.221 / 0.079
instead of 0.093 / 0.212 / 0.046; both are correct, so quote the filtered set to match the
figure.

**The causal chain, panel by panel.**

1. The GC bias is **introduced by the regional adjustment**, not inherited from the
   context-only model: mean |rank - 0.5| is 0.093 for `r == 1` against 0.212 for published
   Gnocchi.
2. That adjustment's GC dependence is **wholly non-CpG**. `r_non` runs 0.95 -> 1.79 while
   `r_CpG` stays 0.98-1.00, and the counterfactual holding non-CpG `r` at 1 is flat within
   0.6% even though CpG contexts carry 43% of the expected-count weight at GC 0.75. This
   is a decomposition identity, not a fit.
3. The training set **is not the scored population**: the fraction of training sites
   inside the analyzed windows falls 0.84 -> 0.28 across GC, and the excluded
   territory is *different*, not merely absent -- the QC-failing stratum's non-CpG DNM
   rate runs 1.55x the noncoding rate in the GC bulk and 4.06x by GC 0.61, while
   coding/noncoding stays flat at 0.90-0.99.
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
   smooth 1.57x, and the logistic regression can then track it instead of missing by 26%
   and 29% in opposite directions.
5. **Refitting `r` there removes the bias**: 0.212 -> 0.046, below the context-only
   model's own 0.093. Two controls make this the population and not something else: the
   full-population refit through the same code lands at 0.212 (so it is not the
   reimplementation), and a size-matched random subsample lands at 0.210 (so it is not
   less data).

**The adjustment is wrong, not merely present.** Measured against the adjustment the
observed DNMs support -- `DNMs / opportunities` per (context, GC bin), both sides
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
  GC/methylation; high-GC CpGs are 90-100% hypomethylated with a 2.7x lower DNM rate; the
  model over-predicts there). Step 5 refuted -- the counterfactual holding non-CpG `r` at 1
  is flat, so none of it reaches Gnocchi.
- **The dnm0 background sample is not the cause.** Building the same empirical curve four
  ways, one ingredient at a time: denominator 2.4%, aggregation 4.3%, **window population
  37.6%**. The background sample IS non-uniform in GC (2.0-fold within a context) -- it is
  just not what changes the curve.
- **Training-set *size* is not the explanation either**, though it is a real effect:
  shrinking moves Gnocchi *toward* the context-only model (1% is indistinguishable from
  it) but never past it. The population fix goes past it. `dnm_training_size/` keeps that
  contrast.
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

1. **Fig. 5 is built and verified end to end.** `fig5/README.md` has the operational
   detail; `fig5/fig5.ipynb` carries the derivation of every plotted quantity.
2. **Optional hardening**: a held-out DNM split would make panel D out-of-sample (panel E
   already is, on gnomAD counts the DNM model never sees).
3. **Still unavailable here**: `DEPLETION_RANK_BED` (panel A's third curve) and
   `NEUTRAL_WINDOWS_BED` (McHale et al.'s 693,270-window file). Both live on the
   constraint-tools HPC path and are `None` in `fig5/config.py`; the figure builds
   without them. Neither `depletion_rank.py` nor the neutral-set join has been run
   against its real file, so **run `fig5/preflight.py` first** -- it checks both files'
   schemas in seconds and fails loudly on the quiet errors (chromosome naming, 1-based
   coordinates, a constant enhancer flag). Running the figure on BOTH window sets is the
   open item; their outputs no longer collide, since refits, provenance entries and panel
   PDFs all carry `config.WINDOW_SET_SUFFIX` (`.neutral`).
4. **Before quoting anything in the rebuttal**, re-read the callability caveat above:
   it brackets the over-adjustment across 1.22-1.44, so the figure must not be
   captioned with 1.22 as though it were tight.
