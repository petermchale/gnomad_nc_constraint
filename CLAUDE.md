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

published/               Chen et al.'s data as downloaded (gitignored, ~7 GB)
refits/                  one copy of each regional-adjustment refit (gitignored, ~12 GB)
```

Each directory has its own README with operational detail. **This file holds only what
those READMEs do not**: the bucket inventory, the Hail recipe, the methods narrative for
the paper, and the settled findings below.

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
   names its three bands *QC-pass noncoding*, *QC-pass coding* and *QC-fail*, splitting
   only the first two by coding status.
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
| `fig_tables/mutation_rate_by_context_methyl.txt` | 12.5 KB | Per-`(context, ref, alt, methylation_level)` fitted mutation rate — 96 rows (32 trinucleotide contexts × 3 alt alleles, methylation level only varies for CpG-containing contexts; `run_nc_constraint_gnomad_v31_main.py` lines 86–148). Columns: `possible` = genome-wide count of sites with this context/ref/alt/methylation, after coverage (mean depth 30–32×) and black-region filtering (line 111, `possible_counts_by_context_methyl.txt`). `observed` = count of those sites with a rare (AF ≤ 0.001), PASS-filter variant in the full 76,156-genome callset (lines 100–107, `observed_counts_by_context_methyl.txt`). `proportion_observed` = `observed / possible` (line 132) — the raw empirical mutation rate proxy; it saturates below 1 because recurrent/back mutation and finite sample size mean not every possible site shows a variant even at this sample size. `mu` = an independent, pre-saturation mutation-rate estimate for the same context/ref/alt/methylation, computed from a separately downsampled (1000-genome) subset and rescaled so the genome-wide total equals a fixed constant `total_mu = 1.2e-08` (lines 43–83, `mu_by_context_methyl_downsampled_1000.txt`) — used only as the x-axis of the calibration fit below, not as the final rate. `fitted_po` = the calibrated/smoothed version of `proportion_observed`, obtained by regressing `log(1 − proportion_observed)` on `mu` (weighted least squares, weights `1/sem` of the binomial proportion) and back-transforming: `fitted_po = 1 − exp(B)·exp(A·mu)` (lines 137–141). **`fitted_po` is what the pipeline actually uses as the per-site step-1 mutation probability** — `expected = possible × fitted_po` at line 188 — so it, not `mu` or raw `proportion_observed`, is the step-1 (context-only) mutation-rate table's operative output. |
| `fig_tables/constraint_z_genome_1kb.annot.txt` | 325 MB | Real, final (step-2, r-adjusted) genome-wide 1kb table: `element_id, possible, expected, observed, oe, z, pass_qc, coding_prop` + functional annotation columns (ENCODE cCREs, FANTOM enhancers, GWAS Catalog, etc.). `expected` here is **post-r-adjustment**. |
| `logit_pickles/logit_regularized_dnm01_{context}_pbonf_pca.pkl` | ~15–20 MB each | Fitted L1-logit model, one per trinucleotide context (32 contexts). |
| `logit_pickles/logit_regularized_dnm01_{context}_pbonf_pca.pca.pkl` | ~1 KB each | Fitted PCA transform (sklearn `IncrementalPCA`) per context. |
| `logit_pickles/logit_regularized_dnm01_{context}_pbonf_pca.ft_mean_std.txt` | ~150 B each | Per-context, per-selected-feature mean/std (this mean is x̄) used to standardize features before PCA. |
| `context_prepared.ht` | ~578 GB just for `rows/parts/` (measured: 38,029 partitions, 8,771,192,175 rows total — see recipe below) | Hail native `Table`, key `(locus, alleles)`. **One row per *possible* SNV, not per polymorphic/observed site** — 3 rows per genomic position (one per alt allele), for every covered reference position genome-wide, regardless of whether gnomAD ever observed a variant there. Evidence: the row schema has no frequency/allele-count field at all (no `freq`/`AC`/`AN`); the *actual* gnomAD call set lives in a separate table, `genome_prepared.ht` (`run_nc_constraint_gnomad_v31_main.py` line 38), which does carry `.freq`/`.pass_filters`; and `context_prepared.ht` (aliased `context_ht`) is literally what gets grouped and counted to produce the `possible` denominator (line 111: `possible_ht = context_ht.group_by(context,ref,alt,methylation_level).aggregate(count())` → `possible_counts_by_context_methyl.txt`). Core columns actually used downstream: `context` (trinucleotide, e.g. `"TAA"`), `ref`, `alt`, `coverage_mean` (Float64, mean sequencing depth at that position), `methyl_level` (Int32, CpG methylation bin), `transition`/`cpg` (Boolean), `variant_type`/`variant_type_model` (String), `was_flipped` (Boolean, strand-flip flag), plus allele-splitting bookkeeping (`idx`, `a_index`, `was_split`, `old_locus`, `old_alleles`). Also carries a large unused `vep` struct (full Ensembl VEP annotation: transcript/regulatory/motif consequences, per-population MAFs, SIFT/PolyPhen, etc.) that `run_nc_constraint_gnomad_v31_main.py` never reads. Sample rows (`chr1:10002`–`10003`, not claimed to be polymorphic — just the first two reference positions): `context=TAA/AAC, ref=A, alt=C/G/T, coverage_mean=4.61/6.38`. Needs Hail to read — see the Hail-on-this-Mac recipe below. Superseded for this analysis by `expected_counts_by_context_methyl_genome_1kb.txt` below — no longer needed. |
| `expected_counts_per_context_methyl_genome_1kb.txt` | 3.3 GB (bucket root) | This *is* the exact `hl.export()` at `run_nc_constraint_gnomad_v31_main.py` lines 191–197: `expected_ht = possible_ht.group_by(key=(element_id, context)).aggregate(possible=sum, expected=sum)`, one row per `(element_id, context)` pair — multiple rows per window, one for each trinucleotide context that occurs in it (e.g. `chr1-10000-11000` has 4: `ACC, CCC, TAA, TAG`). Columns `element_id, context, possible, expected`, both **summed over every `(ref, alt, methylation_level)` combination sharing that context**: `possible` = count of possible SNV sites of this context in the window (after coverage/black-region filtering, lines 159–166); `expected` = `possible × fitted_po` per `(ref,alt,methylation_level)` (line 188, `fitted_po` from `fig_tables/mutation_rate_by_context_methyl.txt`), i.e. genome-wide expected counts from sequence context alone, `r ≡ 1`, computed *before* the regional-feature adjustment in lines 209–249. Sample: `chr1-10000-11000 / ACC → possible=3, expected=0.31501`. |
| `expected_counts_by_context_methyl_genome_1kb.txt` | 107 MB (bucket root) | **The step-1 (context-only) expected-count table, further summed down to one row per `element_id`: `element_id, possible, expected`.** Same `possible`/`expected` definitions as the row above, just summed again over all 32 contexts (so `possible` here matches the meaning of `possible` in `fig_tables/constraint_z_genome_1kb.annot.txt`, which the later `r`-adjustment never touches). Despite the name, this file is *not* produced anywhere in `run_nc_constraint_gnomad_v31_main.py` — the script only ever writes the per-`(element_id, context)` file above; this further `group_by('element_id')` sum must happen in a downstream/publication step. (Earlier text here said this was "the same situation as the missing `generic.py`/`constraint_basics.py`/`nc_constraint_utils.py`" — that was wrong: those three modules are *not* missing, they're at `misc/generic.py`, `misc/constraint_basics.py`, `misc/nc_constraint_utils.py` in the bucket, and are confirmed to be exactly what `run_nc_constraint_gnomad_v31_main.py:23–25` imports [`from generic import *`, etc.] — this local checkout just never fetched them. Checked directly: none of the three contain a `group_by('element_id')` step matching this file either, so the specific aggregation behind *this* file genuinely still isn't shown anywhere available — just don't extend that gap to the utility modules generally, see the "fitting code" section below.) Verified self-consistent by hand: summing the 4 per-context rows for `chr1-10000-11000` in the file above (`possible` 3+3+1+4=11, `expected` 0.31501+0.26256+0.074125+0.15301=0.804705) exactly matches this file's row (`11`, `0.80470500`). Trustworthy to use directly, just can't point to its exact generating code. Use this directly — no need to reconstruct step-1 from `context_prepared.ht` (Option A) or the reference FASTA (Option B). **A Hail-native counterpart, `expected_counts_by_context_methyl_genome_1kb.ht/`, also exists in the bucket** (fetch its `README.txt` and `metadata.json.gz` directly over HTTPS, same as any other bucket object) — its `table_type` schema (`Table{key:[element_id], row:Struct{element_id:String, possible:Int64, expected:Float64}}`) matches the `.txt` file column-for-column, confirming the same 3-column shape from an independent source. Its `metadata.json.gz` gives an exact row count via `sum(components.partition_counts.counts)`: 2,575,299 rows (38,029 partitions) — more precise than inferring row count from the `.txt` file's size. Written with Hail 0.2.62, created 2022/01/17. It does *not* resolve the generating-code gap above: the metadata is a standard Hail `TableSpec` (schema + partition counts only, no lineage/provenance field), so it confirms this was a real materialized Hail table upstream of the `.txt` export, but not which script produced it. **Genome-wide regeneration (not just the one hand-picked row above) confirms the r≡1 interpretation**: `preconditions/verify_expected_r1.py` downloads `expected_counts_per_context_methyl_genome_1kb.txt` (confirmed provenance, see row above) and `expected_counts_by_context_methyl_genome_1kb.txt`, sums the former over context per `element_id` in duckdb, and diffs against the latter for all 2,575,299 rows. `possible` (a plain count, independent of any model fit) matches exactly on every row. `expected` matches to within 1e-3 relative tolerance (max observed: 4.6e-5) — not bit-identical, but the residual is fully explained by GCS `customTime` metadata (each object's original creation date, preserved through a later 2023-12-15 bucket migration that overwrote every file's `timeCreated`): `mu_by_context_methyl_downsampled_1000.txt` (2022-01-16) feeds directly into `fig_tables/mutation_rate_by_context_methyl.txt` just two code-steps later (lines 134–148), yet the two are **277 days apart** (`mutation_rate_by_context_methyl.txt` is 2022-10-20) — meaning `expected_counts_by_context_methyl_genome_1kb.{txt,ht}` (clustered 2022-01-16/17/18) and `expected_counts_per_context_methyl_genome_1kb.txt` (2022-08-05, clustered with the October `mutation_rate` file) come from two separate pipeline runs, each with its own random-downsample mutation-rate refit (lines 43–83, 134–141) — plausibly explaining the tiny, symmetric-around-zero `expected` differences (mean diff ≈ 9e-6) without implicating the r≡1 assumption itself. Run `python preconditions/verify_expected_r1.py [-dest_dir published]` to reproduce (no Hail needed — both files are plain-text exports; ~8 min dominated by the 3.3 GB per-context file download). The script also saves the regenerated table itself to `{dest_dir}/expected_counts_by_context_methyl_genome_1kb.regenerated.txt` — sorted and `%.8f`-formatted to match the published file's own conventions (lexicographic `element_id` order, 8-decimal `expected`), purely so the two can be diffed/read side by side; the join-based comparison the script prints doesn't depend on this file or on row order. Local only — `published/` isn't tracked by git, so this file isn't committed. |
| `observed_counts_genome_1kb.txt` | 71 MB (bucket root) | Standalone observed-variant-count table, `element_id, variant_count`. Same numbers as the `observed` column of `fig_tables/constraint_z_genome_1kb.annot.txt` below, but much smaller if `pass_qc`/`coding_prop`/functional annotations aren't needed. |

Bucket contents are listable without `gsutil`/auth via the JSON API, e.g.:
```
curl -s "https://storage.googleapis.com/storage/v1/b/gnomad-nc-constraint-v31-paper/o?prefix=logit_pickles/&maxResults=50"
```

### Recipe: reading `context_prepared.ht` (or any `.ht`/`.mt`) with Hail on this Mac

`hail==0.2.138` and `pyspark` are already in `requirements.txt`/`.venv`. To actually use
them:

```bash
export JAVA_HOME=/opt/homebrew/opt/openjdk@11   # NOT the JDK bundled in IGV.app — plain
                                                  # `brew install openjdk@11`. Hail 0.2.138
                                                  # warns if run under Java 21 (e.g. IGV's).
export PATH="$PWD/.venv/bin:$PATH"               # puts find_spark_home.py on PATH; without
                                                  # this, hl.init(backend='local') fails with
                                                  # FileNotFoundError: find_spark_home.py
```

```python
import hail as hl
hl.init(backend='local', quiet=True)   # NOT the default 'spark' backend — that one only
                                        # has HadoopFS, which errors "No FileSystem for
                                        # scheme gs" (no Hadoop GCS connector jar here).
                                        # backend='local' uses Hail's own GoogleStorageFS,
                                        # which can read gs:// paths directly with no
                                        # gsutil/auth setup, since the bucket is public.
ht = hl.read_table('gs://gnomad-nc-constraint-v31-paper/context_prepared.ht')
ht.show(5)
```

Two gotchas actually hit when doing this (2026-07-20):

1. **`hl.init(backend='local')` itself throws `IOException: Your default credentials were
   not found`** — even for purely local paths — because Hail's `RouterFS` eagerly builds
   routes for every cloud filesystem (GCS included) at backend-construction time, which
   probes for Google Application Default Credentials whether or not you ever touch `gs://`.
   Fix: point `GOOGLE_APPLICATION_CREDENTIALS` at *any* syntactically-valid throwaway
   service-account JSON (fake key, fake project, never actually used for a real call —
   generate one with `openssl genrsa 2048` and hand-build the JSON). This is a Hail/Java
   quirk, not a real auth requirement — the bucket is public.
2. **`.show(n)`/`.take(n)` reads partitions in doubling batches (1 → 2 → 4 → 8 → ...),
   not just enough to satisfy `n` rows** — even though partition 0 alone (201,627 rows)
   already dwarfs a 5-row request. If you're mirroring a `.ht` locally instead of pointing
   at `gs://` directly (e.g. to avoid the ~578 GB full download), you need whichever
   power-of-two of partitions the doubling lands on (4 partitions/~46 MB sufficed for a
   5-row `.show()` here), not just partition 0. A local mirror needs, per partition `i`:
   `rows/parts/part-*`, `index/part-*.idx/{index,metadata.json.gz}`; plus the table-level
   `metadata.json.gz`, `globals/{metadata.json.gz,parts/part-0}`, `rows/metadata.json.gz`,
   `_SUCCESS`, `README.txt` once. All of these are plain HTTPS-fetchable (no auth) since
   the bucket is public — see the listing trick above.

## The Figure 2A-style rank statistic — canonical methods narrative

This section is the canonical, extractable methods narrative for the rebuttal/revised
paper. It was written for `compute_gc_bias_step1_vs_step2.py`, **deleted 2026-08-07**
(recoverable from git history) once fig5 panel A superseded its headline result on the
same window set with the same statistic. Every methodological choice recorded below is
still live: they are implemented in `gnocchi_bias/windows.py`, which was extracted
verbatim from that script, and fig5 inherits all of them. Read `windows.py` for the code.

TWO CAPABILITIES WENT WITH THE SCRIPT and exist nowhere else, both concerning comparison
against McHale et al.'s *existing published* figures rather than producing Fig. 5:
the 2D hexbin density heat map of (GC, rank) that reproduces Fig. 2A's visual form
(fig5 draws only the conditional-mean line), and `-bias_metric residual`, the
`expected − observed` metric Supp. Fig. 1 is defined on (fig5 uses the rank statistic
only). Recover them from git if either is wanted. Every
methodological choice below that mirrors, deviates from, or could not be replicated from
McHale, Goldberg & Quinlan 2026 ("The performance of genetic-constraint metrics varies
significantly across the human noncoding genome", `mchale_et_al_250115.pdf` + supporting
PDF, this repo) is cited by page/section, with exact quoted text where it matters.

`-bias_metric rank` (default) reproduces the statistic actually plotted in **Figure 2A**
(page 6 of `mchale_et_al_250115.pdf`; Methods, "Computation of window residuals under the
Chen model", p.15), generalized to compare step 1 vs step 2 on the same axes (the paper
only plots one model, the published Gnocchi — this script's whole point is a
step1-vs-step2 comparison, so the same rank statistic is computed for both):
1. Compute each window's own z-score from `(expected, observed)`, using the *exact*
   formula in `run_nc_constraint_gnomad_v31_main.py` lines 278–281: `oe =
   observed/expected; chisq = (observed-expected)**2/expected; z = -sqrt(chisq) if
   oe>=1 else sqrt(chisq)`; keep only `z` in `[-10, 10]` and finite (matches the official
   pipeline's own z clipping). Applied separately to `(expected_step1, observed)` and
   `(expected_step2, observed)` — step 1 gets its own from-scratch z-score, since the
   official pipeline never computes one for `r==1`.
2. Standardize each window's z to a rank in `(0, 1)` via `(rank(z) - 0.5) / n` — exactly
   the "(standardized) rank of Gnocchi" Figure 2's y-axis and caption describe ("the
   marginal distribution of Gnocchi ... is uniform with an average value of 0.5";
   Supporting Figure 1's caption: "Ranks are standardized to lie in the unit interval";
   main text Table 1's caption: "the target variables (ranks) in the fitting process are
   uniformly distributed between 0 and 1").
3. Bin windows by GC content (paper units — see below) and take the mean rank per bin —
   exactly Figure 2A's dark-grey conditional-mean-rank line, with a horizontal reference
   line at y=0.5 (not y=0, since this is a rank, not a residual) and a vertical reference
   line at the mean GC content of the analyzed window set.
4. A 2D hexbin density heat map of `(GC content, rank)` is drawn behind the line.

`-bias_metric residual` is the original metric this script started with, kept for
backward compatibility (not part of Figure 2A) — see the script's own docstring for its
definition; nothing about it was changed by the Figure-2A generalization.

**GC content units** (`-match_paper_gc_units`, on by default): this repo's own
`GC_content_1k` column (`misc/genomic_features13_genome_1kb.txt`) is a **percentage**,
0–100 (empirically confirmed: min/max/mean over a 200k-row sample were 0.9/85.2/41.0).
McHale et al.'s own GC content is a **fraction**, 0–1, computed via `bedtools nuc`
(confirmed by reading the exact scripts cited in the paper's Methods, "Assignment of
genomic feature values to genomic windows", p.14:
`github.com/quinlan-lab/constraint-tools/blob/main/experiments/germline-model/chen-et-al-2022/compute-GC-content-given-window-size-based-on-Chen-windows.sh`,
which calls `bedtools nuc -fi <genome> -bed <windows> | cut -f1-7,9`): `bedtools nuc`'s
9th output column is `pct_gc`, always a 0–1 fraction (column 8 is `pct_at`, dropped by
`cut -f1-7,9`). Figure 2A's x-axis (visually confirmed) spans roughly 0.2–0.73 —
consistent with a fraction, not 20–73. So `GC_content_1k` is divided by 100 here before
binning/plotting in rank mode.

**Heat map** (`-plot_heatmap`, on by default): a 2D hexbin density plot of
`(GC content, rank)`, one panel each for step 1 and step 2, using a log-scaled `inferno`
colormap (matching the paper's black-purple-orange-yellow palette;
`matplotlib.colors.LogNorm`, `mincnt=1` so empty cells stay white). The conditional-mean
line is drawn in light grey (`"0.9"`, close to white) rather than a plain dark grey — the
paper's "Mean observed Gnocchi" line reads as much lighter than its legend swatch
suggests once drawn over the heat map's mostly dark-purple/black cells, and a plain dark
grey line is nearly invisible against the same background. NOT reproduced: the
light-grey multivariate-linear-regression line (needs BGS/gBGC fit jointly with GC
content; only GC content is available here) and panels B/C (no BGS/gBGC data joined to
the genome-wide 1kb window table here) — out of scope per explicit request ("Fig 2A", not
2A–C).

**Axis ranges** (rank mode): y-range hardcoded to `[0, 1]` (matches Figure 2A's y-axis
exactly — ticks 0.0 to 1.0, box edges aligned with the first/last tick, no autoscale
margin; not automatic in matplotlib since the rank statistic's own natural range,
`(0.5/n, 1-0.5/n)`, is very slightly inside `[0,1]`). x-range defaults to `"0.2,0.73"` —
**read visually** from the published Figure 2A, not from any numeric value stated in the
paper's text (the paper reports no exact axis limits). Method: rasterized the PDF page at
300 DPI (`pdftoppm -png -r 300 -f 6 -l 6`), visually located the tick labels (0.2 through
0.7) and the plot box's left/right edges relative to them. A pixel-level calibration was
attempted (the plot box's horizontal extent via the longest continuous dark-pixel run in
the y=0.5 reference line, which spans the same width as the box: pixel columns ~423–1027
at 300 DPI) but tick-mark pixel positions couldn't be isolated cleanly from the label
text underneath them — so `(0.2, 0.73)` is a visual estimate, not pixel-exact or
text-sourced. Treat as approximate; refine against the actual McHale et al.
figure-generation code/data if exact bounds are needed for a citation.

**Chromosome filtering** (`-exclude_sex_chromosomes`, on by default): McHale et al.'s
Methods ("Provenance of constraint scores", p.14) state plainly: "Windows on the X and Y
chromosomes were omitted." Empirically, the genome-wide 1kb window files used here
already have chrY fully absent (0 rows) and only 2,497 chrX rows — pseudoautosomal-region
(PAR) windows, not general chrX: `run_nc_constraint_gnomad_v31_main.py`'s own upstream
filtering (`filter_to_autosomes_par`, `constraint_basics.py:224–225`,
`ht.filter(ht.locus.in_autosome_or_par())`) already restricts everything in this repo's
data to autosomes + PAR before any of these files are produced, so PAR-on-chrX is the
*only* sex-chromosome remnant possible here — consistent with, not contradicting, McHale
et al.'s statement.

**Noncoding restriction** (`-restrict_to_noncoding`, on by default — was off before this
revision): half of McHale et al.'s "neutral" window definition (Methods, "Construction of
the window sets to assess model bias...", p.14: "Noncoding windows were defined to be
Chen, Halldorsson and CDTS windows that don't significantly overlap merged exons.").
Exact threshold still unconfirmed against their Methods — default guess remains
`coding_prop == 0.0` (fully noncoding windows only); their "don't significantly overlap"
wording (mirroring the enhancer criterion below) suggests a threshold rather than a
strict zero, but no numeric value is given in the text.

**GeneHancer enhancer exclusion** — the other half of "neutral", **not fully automatable**
(`-genehancer_bed`, off by default): McHale et al.'s Methods continue: "Of the noncoding
windows, those that don't significantly overlap Genehancer enhancers (Fishilevich et al.
2017) were labeled 'neutral' ... Noncoding windows that do significantly overlap
Genehancer enhancers were labeled 'constrained'." ("significantly" is not numerically
defined in the text.) The script's `restrict_to_neutral_genehancer()` implements the
actual exclusion logic (a genomic-interval anti-join in duckdb — no bedtools dependency,
and bedtools isn't installed in this environment anyway), but it's a no-op without a
local GeneHancer BED file, because GeneHancer can't be downloaded automatically:
- Confirmed via web search (2026-07-21): "GeneHancer data must be obtained from the
  source database directly in the original format or licensed, rather from UCSC. Files
  for these tracks are not available from their download servers" — UCSC displays the
  track interactively but doesn't serve the file; it requires a GeneCards Suite/LifeMap
  Sciences license.
- McHale et al.'s own reference notebook
  (`github.com/quinlan-lab/constraint-tools/blob/main/papers/neutral_models_are_biased/8.labeled-enhancers/main.2.ipynb`,
  fetched and read directly) doesn't show a GeneHancer-download-and-intersect step
  either — it reads an already-enhancer-labeled file,
  `Supplementary_Data_2.features.constraint_scores.bed`, carrying a boolean `window
  overlaps enhancer` column, from a private HPC path (`CONSTRAINT_TOOLS_DATA`, not
  publicly accessible). Even the paper's own code doesn't publicly show the raw
  GeneHancer acquisition/intersection step.

Practically: if you (as a paper co-author) have access to that HPC path or another
licensed GeneHancer BED file, pass it via `-genehancer_bed`. Without it, "neutral" here is
only noncoding + `pass_qc` + non-sex-chromosome — NOT excluding enhancer-overlapping
(and therefore potentially actually constrained) windows. `min_frac_overlap` (bedtools
`-f` semantics) defaults to `None` (any overlap excludes the window); McHale et al.'s own
codebase uses `-f 0.5` in a *different* intersect step (assigning external
constraint-score features to truth-set windows, same notebook,
`intersect_and_aggregate()`) — a plausible hint for what "significantly" might mean here
too, but not confirmed for this specific labeling step, so not applied as a default.
**UNTESTED**: no GeneHancer file is available in this environment, so this exclusion
logic has not been run against real GeneHancer data — verify directly before relying on
it for anything reported in the rebuttal/paper.

**Window count vs. the paper** (explains the wider GC-content "fringe" visible in this
script's heat maps vs. Figure 2A): measured directly (2026-07-21, full non-downsampled
dataset, default filters — `exclude_sex_chromosomes` + `restrict_to_noncoding` +
`pass_qc`, no GeneHancer exclusion): this script's default window set has **1,843,559**
windows, vs. the paper's stated **693,270** "putatively neutral" windows (page 5) — 2.66x
more. GC content (fraction) in our set ranges 0.14–0.837 (mean 0.399) — genuinely wider
than the ~0.2–0.73 plotted range, though only 414 of 1,843,559 windows (0.02%) fall
outside `[0.2, 0.73]` — the vast majority of the extra volume is denser sampling of the
*same* GC range the paper covers, not a wider range per se; with 2.66x more windows, the
sparse GC tails naturally pick up more points, making the low-count "fringe" hexbin cells
near the plot edges more populated/visible here than in the paper's smaller set (verified
separately that matplotlib's `hexbin` `extent` correctly drops out-of-range points rather
than piling them at the boundary, so the fringe is real data, not a plotting artifact).
Likely, only partially confirmed causes of the 2.66x gap:
1. The missing GeneHancer exclusion above (confirmed missing; effect size on count not
   separately measured).
2. McHale et al.'s Methods ("Construction of the window sets...", p.14) additionally
   exclude windows overlapping "gaps in the hg38 genome assembly, Encode 'exclude
   regions' (Amemiya et al. 2019), and regions with insufficient read coverage in Gnomad
   version 3" as a named, separate filtering step — this script only applies Chen et
   al.'s own `pass_qc` (a coverage/pass-rate threshold from the annot file), not this
   additional interval-based exclusion; the two are not necessarily equivalent even
   though both are coverage-related.
3. Unconfirmed possibility that the paper's actual window source (their cited
   Supplementary Data #2 file) is a different vintage/pre-filtered export than the
   `constraint_z_genome_1kb.annot.txt` table pulled from the bucket here.


## Where to pick up

1. **Fig. 5 is built and verified end to end.** `fig5/README.md` has the operational
   detail; `fig5/fig5.ipynb` carries the derivation of every plotted quantity.
2. **Optional hardening**: a held-out DNM split would make panel D out-of-sample (panel E
   already is, on gnomAD counts the DNM model never sees).
3. **Still unavailable**: `DEPLETION_RANK_BED` (panel A's third curve, on the
   constraint-tools HPC path) and `GENEHANCER_BED` (licensed). Both are `None` in
   `fig5/config.py`; the figure builds without them, and `depletion_rank.py` has never
   been run against the real file.
4. **Before quoting anything in the rebuttal**, re-read the callability caveat above:
   it brackets the over-adjustment across 1.22-1.44, so the figure must not be
   captioned with 1.22 as though it were tight.
