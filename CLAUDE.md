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

## Repository layout (reorganized 2026-08-04)

Analysis code was split into a shared library plus one directory per figure, each with
its own notebook, so a figure and the code it imports live together:

```
gnocchi_bias/            shared library -- imported by BOTH figure directories
  windows.py             genome-wide 1kb window table: download, duckdb join, the McHale et al.
                         window filters, GC units, z-scores, standardized ranks, GC binning,
                         shared plot-style constants
  dnm_model.py           the DNM training set + per-context mutation model: loading, regime-1
                         subsampling, univariate selection, the (unpublished) multivariate
                         PCA+logit fit, genome-wide apply, training-set prediction

fig5/                    the manuscript figure -- see its own section below
  fig5.ipynb             the figure; config.py, data.py, diagnostics.py, panels.py,
                         refit.py, depletion_rank.py

dnm_training_size/       the DNM training-set-SIZE dose-response, and only that
  dnm_training_set_size.ipynb, run_dnm_training_experiment.py

preconditions/           what had to be true about Chen et al.'s pipeline before any
  verify_expected_r1.py    of this meant anything -- the r==1 table's identity, the
  verify_logit_predict_    operative r formula, the absent fitting code (verify_*), and
    behavior.py            whether the reimplementation reproduces theirs (validate.py).
  verify_missing_utils_    Kept out of the figure directories that assume them.
    files.py
  validate.py

published/               Chen et al.'s data as downloaded -- inputs, never edited
                         (gitignored, ~7 GB; was `tmp/` until 2026-08-07)
refits/                  one copy of each regional-adjustment refit (gitignored, ~12 GB)
compute_gc_bias_step1_vs_step2.py   unchanged CLI, now importing from gnocchi_bias.windows
```

(`fig3/` was deleted 2026-08-07, preserved at commit 070fee9. The layout above is
current; sections further down that describe `fig3/` outputs are the historical record of
how the result was reached.)

**Neither `gnocchi_bias` module sets a matplotlib backend** (the CLIs call
`matplotlib.use("Agg")` inside their own `main()` instead) — that is what lets the
notebooks import them and still render inline. Don't move those calls back to module
scope.

**The refactor was verified behavior-preserving, not assumed**: on the full genome-wide
path, `compute_gc_bias_step1_vs_step2.py` produces byte-identical binned output before
and after (1,843,559 windows). Validation of the reimplemented FITTING pipeline against
Chen et al.'s published outputs is a separate matter and lives in
`preconditions/`.

**Known pre-existing bug found during that verification**: `-downsample_frac` /
`-downsample_n` are **not reproducible across runs even with a fixed `-random_seed`**.
duckdb's parallel join emits rows in nondeterministic order, so `polars.sample(n, seed)`
selects a different subset each run. Confirmed by running the *unmodified pre-refactor*
script twice at the same seed and getting different per-bin counts (bin 0: n=3, then
n=11). The full, non-downsampled path is deterministic. This matters because the
`-downsample_n 5000` rough pass logged further down was subject to it — treat any
downsampled number as irreproducible.

## Confirmed finding: the paper's Methods text does not match the code

The paper (Methods, "Adjustment of the effects of regional genomic features on mutation
rates") states, verbatim:

> "the adjustment factor r is defined as the ratio of logit given x(w) to that of the
> genome-wide average x̅: r = β·x(w)/β·x̅"

i.e. a ratio of **raw linear predictors (logits)**, with no sigmoid, and with β₀
(intercept) excluded from the dot product.

The actual code, `run_nc_constraint_gnomad_v31_main.py` lines ~209–249, computes:

```python
df_adj['pred_{ctx}'] = logit.predict(sm.add_constant(df_x_pca, has_constant='add'))
ave = logit.predict(sm.add_constant(zero_row, has_constant='add'))[0]
df_adj['rr_{ctx}'] = df_adj['pred_{ctx}'] / ave
```

`logit` is a `statsmodels.discrete.discrete_model.L1BinaryResultsWrapper` (regularized
logistic regression, one per trinucleotide context). Its `.predict()` defaults to
returning **σ(linear predictor)** — a probability — not the linear predictor itself
(that requires the deprecated `linear=True` kwarg, or `which="linear"` in modern
statsmodels).

**Empirically confirmed** (not just read from code) by downloading one real fitted
model from the public bucket and testing directly:

```python
import pandas as pd, statsmodels.api as sm
logit = pd.read_pickle('AAA.pkl')   # logit_pickles/logit_regularized_dnm01_AAA_pbonf_pca.pkl
zero_row = sm.add_constant(pd.DataFrame([[0,0,0]]), has_constant='add')
logit.predict(zero_row)               # -> [0.0394]   (a probability, in (0,1))
logit.predict(zero_row, linear=True)  # -> [-3.1948]   (== logit.params[0], the intercept)
```

Note on how this verification was actually done, since `logit` here is a deserialized
object, not something read from source: the pipeline itself (line 207) does
`gsutil -m cp {input_bucket}/logit_pickles/* {output_dir}/published/` then loads via
`pickle.load(open('{output_dir}/tmp/{model}.pkl', 'rb'))` (line 233). The verification
above instead fetched the identical bucket object directly over HTTPS
(`https://storage.googleapis.com/gnomad-nc-constraint-v31-paper/logit_pickles/logit_regularized_dnm01_AAA_pbonf_pca.pkl`)
to a separate scratch path and loaded it with `pd.read_pickle`. These are equivalent:
same bucket object/bytes either way, and `pd.read_pickle` just adds compression-format
sniffing on top of the same `pickle` deserialization for an uncompressed `.pkl` — so it
reconstructs the identical `L1BinaryResultsWrapper` instance. Pickle's byte stream embeds
the object's fully-qualified class path, which is *why* `type(logit)` reliably reports
`statsmodels.discrete.discrete_model.L1BinaryResultsWrapper` even though `logit` is "just"
a deserialized blob — unpickling re-imports and re-instantiates the real class, not a
generic container. No local `{output_dir}/published/` was ever populated and the pipeline
itself was never run for this check.

This verification is captured as a standalone, reproducible script:
`preconditions/verify_logit_predict_behavior.py`. It downloads a real fitted per-context model from
the public bucket (default context `AAA`, override with `-context`), prints
`type(logit)`, and reproduces the probability-vs-linear-predictor discrepancy above:

```
python preconditions/verify_logit_predict_behavior.py [-context AAA] [-dest_dir published]
```

So the real, operative formula is:

**r(w) = σ(β₀ + β·z(w)) / σ(β₀)** — a ratio of *predicted probabilities* from the same
fitted model, where z(w) is the PCA-transformed, standardized feature vector for window
w's trinucleotide context, and the denominator is the model's probability at the
population-mean feature values (z=0).

This is a real, uncorrected discrepancy — the only published Author Correction for this
paper (Nature 626:E1, 2024, DOI 10.1038/s41586-024-07050-7) only fixes missing data
points in Supplementary Figs 6–8, and says nothing about this formula. Since the code is
presumably what actually produced the published Gnocchi scores, treat **the code's
probability-ratio formula as ground truth** for any real-data comparison — not the
paper's stated logit-ratio formula.

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

## Why `fig_tables/comparisons_*.txt` (Extended Data Fig. 6) can't answer the GC-bias question

Extended Data Fig. 6 of Chen et al. 2024 ("Comparison of constraint scores built from
different mutational models and genomic windows") plots ROC curves comparing Gnocchi
(`z`) against context-only models built at trinucleotide (`z_trimer`) and heptanucleotide
(`z_heptamer`) resolution — i.e. `z_trimer` is conceptually the same "step-1, `r ≡ 1`"
idea as `expected_counts_by_context_methyl_genome_1kb.txt`, just scored per-variant
instead of per-window. It's tempting to pull `z`/`z_trimer`/`z_heptamer` straight from
this figure's underlying data, bin by GC content, and use it as a second, independent
demonstration that the context-only model has lower local bias than Gnocchi. **This
doesn't work**, for reasons confirmed directly (not just asserted) by downloading the
real data — see `verify_comparisons_tables.py` (repo root), which fetches
`fig_tables/comparisons.tar.gz` (the source for `efig_utils.py:plt_comparison_roc_gnocchi`,
the function `generate_manuscript_efigures.py -efig 6` calls) and prints each
`comparisons_*.txt` file's real schema, row counts, and sample rows:

```
python verify_comparisons_tables.py [-dest_dir published]
```

Findings from running it:

1. **Not a genome-wide window sample — a curated variant-classification dataset.**
   Each file is one of the four positive ("functional") sets or one of six negative
   (AF-matched, downsampled TOPMed-control) sets from the ROC/AUC task. The row counts
   confirm this *is* Extended Data Fig. 6's exact data: `comparisons_gwas_catalog_repl.txt`
   has exactly 9,229 rows, `comparisons_gwas_fine-mapping_pip09_hc.txt` exactly 140, and
   `comparisons_likely_pathogenic_clinvar_hgmd.txt` has 1,273 rows but exactly 1,026
   *unique* loci (247 duplicates) — matching the paper's caption counts of "9,229 GWAS
   Catalog variants ... 140 high-confidence fine-mapped variants ... 1,026 likely
   pathogenic variants" exactly, the last one only after the dedup
   `plt_comparison_roc_gnocchi` applies (`.drop_duplicates(subset=['locus'])`, line 970).
   GWAS/fine-mapping/pathogenic variants are enriched in regulatory, promoter, and
   CpG-island-rich (i.e. high-GC) regions by construction; the "negative" TOPMed pools
   are pre-filtered by AF band and further subsampled at plot time to a fixed 10:1 ratio
   against whichever positive set they're paired with (`sampling = 10`,
   `df_0.sample(n=sampling*len(df_1))`) — confirmed by the raw pool sizes themselves
   (9,229 positive vs 129,979 candidate negative for the `topmed_maf5` pairing, 14.1x,
   pre-subsampling). None of this is a uniform sample of 1kb windows across the genome.
2. **No GC-content column, and not keyed by `element_id`.** Every file is keyed by
   `locus` (an individual variant position, e.g. `chr1:960326`), not `element_id` (a 1kb
   window) — confirmed directly from the printed column list for all 10 files. Computing
   GC content at all would first require floor-dividing each `locus` into its containing
   1kb window and joining against `misc/genomic_features13_genome_1kb.txt`'s
   `GC_content_1k` — doable, but doesn't fix problem 1.
3. **Units mismatch even setting aside ascertainment.** `z` is a signed chi-based
   statistic, `z = ±√((observed−expected)²/expected)` (`run_nc_constraint_gnomad_v31_main.py`
   lines 278–280) — not the raw `expected − observed` residual that
   `compute_gc_bias_step1_vs_step2.py` and McHale et al.'s Supp Fig. 1 are both defined
   on. Averaging `z` (or `z_trimer`) by GC bin is a different quantity from averaging
   `(expected − observed)` by GC bin, and isn't directly comparable to the existing
   step-1-vs-step-2 results even if the ascertainment problem in (1) didn't exist.

Bottom line: `comparisons_*.txt` could answer a legitimate but *different* question —
"does Gnocchi's classification advantage over trimer/heptamer hold up specifically in
high-GC vs low-GC functional variants?" — but not the reviewer's actual question about
genome-wide local bias, which the existing genome-wide analysis below already answers
correctly.

## The analysis that does work: answering the reviewer's request directly

**This whole analysis stands or falls on one interpretive claim**: that `expected` in
`expected_counts_by_context_methyl_genome_1kb.txt` is the context-only, pre-adjustment
(`r ≡ 1`) expected count — see the data-inventory row above for the full chain of
evidence (source lines, the `r`-adjustment starting only later at line ~209, and the
by-hand self-consistency check against `expected_counts_per_context_methyl_genome_1kb.txt`).
If that interpretation is wrong, step-1 local bias is not computable from this file at
all, and every step-1 number in this analysis (including the results already logged
below) is meaningless — there is no independent way to compute step-1 expected counts
without falling back to `context_prepared.ht` (Hail) or the reference FASTA, both of
which this analysis deliberately avoids. Re-check that row before trusting or extending
any step-1 result if anything here looks off.

**Goal**: compute local bias (`expected − observed`) as a function of GC content, for
the *same* real genome-wide 1kb windows, comparing:
- **Step 1** (context-only, `r ≡ 1`): expected count from sequence context alone.
- **Step 2** (real Gnocchi, `r` as actually computed by the code above): expected count
  after the regional-feature adjustment.

This is the literal reviewer request, answered directly on Chen et al.'s published output.

### Concrete steps

No Hail, no JVM, no FASTA reconstruction needed — three modest text files, all public,
cover everything:

1. **Step-1 (context-only) expected/possible per window.** Download
   `expected_counts_by_context_methyl_genome_1kb.txt` (107 MB) directly —
   `element_id, possible, expected`. This *is* the pipeline's own pre-adjustment
   (`r ≡ 1`) expected-count export; no reconstruction required.

2. **Step-2 (real Gnocchi) expected/observed/possible + QC per window.** Download
   `fig_tables/constraint_z_genome_1kb.annot.txt` (325 MB) —
   `element_id, possible, expected, observed, oe, z, pass_qc, coding_prop, ...`.
   `expected` here is post-r-adjustment.

3. **GC content per window.** Download `misc/genomic_features13_genome_1kb.txt`
   (1.44 GB), keep only `element_id` and `GC_content_1k` (drop the other 51 columns
   early — use column-filtered/chunked reads, e.g. `duckdb`, `polars`, or `pandas` with
   `usecols`/`chunksize`, given the file size).

4. **Join** all three on `element_id` (step-1 expected, step-2
   expected/observed/possible/pass_qc, and GC_content).

5. **Bin windows by GC_content** (e.g. deciles, or fixed-width bins spanning the real
   ~30–65% range) and compute mean `(expected − observed)` per bin, separately for
   step-1 and step-2 expected values. Plot both curves vs. GC bin. Sign convention
   (`expected − observed`, not the reverse) matches McHale et al.'s simulation
   (`github.com/quinlan-lab/constraint-tools`,
   `papers/neutral_models_are_biased/9.regression/fit_neutral_models.py`:
   `residuals_{model}Model = predicted_y - y`).

6. **Compare** to Supp Fig 1 of the McHale/Goldberg/Quinlan paper — does the real data
   here show the same qualitative pattern?

**No genome-wide "global" bias metric is computed.** McHale et al.'s simulation
(`github.com/quinlan-lab/constraint-tools`, `papers/neutral_models_are_biased/9.regression/`)
defines one (`compute_overall_model_bias()`, in `generate_data.py`:
`mean((predicted_y - true_rate(x))**2)`), but it requires a known ground-truth
`true_rate(x)` that only exists because their data is simulated. Real gnomAD data has no
such ground truth — only noisy observed counts — so there's no faithful real-data analog
of that particular metric. Only the GC-binned local bias (their "feature-specific bias",
the `groupby(x_bin).mean(residual)` line in `plot_residuals.py`) is ported here.

**Implemented in `compute_gc_bias_step1_vs_step2.py`** (steps 1–5 above). Run once so far
on a `-downsample_n 5000` rough pass (not the full ~3M windows, and with
`-restrict_to_noncoding` off, i.e. coding windows still included): step-1 bias stays
small/negative across the GC range, while step-2 bias is comparable to step-1 in the
bulk but flips sign and grows large in the high-GC tail (e.g. +80 at GC≈79%, n=1) —
consistent with the regional-feature adjustment overcorrecting (not just failing to
correct) in sparsely-populated tail regions. Given tail bins had n as low as 1–6 windows
in this rough pass, this needs confirming on a much larger sample (or the full genome,
no downsampling) before treating it as a robust result.

### Practical notes
- All files are public, no auth: plain `curl`/`wget` works.
- Prefer `duckdb` or `polars` (or `pandas` with `usecols=`/`chunksize=`) over naive full
  loads of the 1.44 GB / 325 MB files. In practice, this repo's scripts split the two:
  **duckdb** for the memory-management-critical part — column-pruned, multi-file
  SQL joins/group-bys where the point is to *not* materialize full-width large files
  (e.g. `compute_gc_bias_step1_vs_step2.py`'s 3-way join across the 1.44 GB features
  file, the 325 MB annot file, and the step-1 file, reading only 2–3 needed columns
  from each; `verify_expected_r1.py`'s `GROUP BY element_id` over the 3.3 GB
  per-context file); **polars** once a manageable, already-joined/aggregated
  DataFrame exists and the remaining work is chained, iterative transformation logic
  (filtering, binning, downsampling, plotting) — its expression API
  (`pl.col(...)`, `.with_columns(...)`, `.group_by(...).agg(...)`) reads more
  naturally for that than either raw SQL or pandas. `compute_gc_bias_step1_vs_step2.py`
  shows both in one script: duckdb builds the joined table, then `.pl()` (not `.df()`)
  hands it to polars for everything downstream.
- `element_id` format is `chr-start-end`, e.g. `chr1-26000-27000` — 0-based, matches
  `misc/hg38.chrom.1kb.bed` (also in the public bucket, under `misc/`).
- `context_prepared.ht`, `expected_counts_per_context_methyl_genome_1kb.txt`, and the
  reference-FASTA approach are no longer needed for this analysis — kept in the data
  inventory above only as background/cross-check options.

## The next experiment: DNM training-set size vs. Gnocchi's local bias

`chen_formula/chen_formula.tex`, section "Predictions of the hypothesis", predicts (and
the rebuttal's red-text claims to have empirically shown) that Gnocchi's GC-content bias
should shrink toward the context-only model's bias as the DNM training set shrinks
(sparse tails collapse `r_c(x)` toward 1), and should shrink again, in the tails
specifically, if the training set is densified there — concretely, by adding more
*background* (non-mutated) sites without adding more real DNMs. This section documents
what's available in the bucket for attempting the same experiment here: resize the DNM
training set, refit the regional-feature logistic regression, and observe the effect on
local (GC-binned) bias.

### The actual training data (what to subsample)

| File | Size | Contents |
|---|---|---|
| `genomic_features/DNM_decode_psychencode_site_context.mutation_rate.txt` | 24.7 MB | The **dnm1** set: 410,542 real germline de novo mutation sites (DECODE + PsychENCODE trio sequencing) — the positive/mutated class. Columns: `locus, alleles, context, ref, alt, methyl_level, sid, 3mer` (`3mer` = the context-only fitted mutation rate for that trinucleotide/methylation combo, i.e. `fitted_po` from `fig_tables/mutation_rate_by_context_methyl.txt`, pre-joined in). Sample: `chr1:137548 [G,C] CCC C G 0 CCC-0 0.26256`. |
| `genomic_features/context_prefiltered_nonmutated-dnm_sites10xdnm.mutation_rate.txt` | 190 MB | The **dnm0** ("non-mutated") background set: 4,107,802 sites — exactly 10x the dnm1 count (the "10x" in the filename and in `logit_regularized_dnm01_{context}...`), matched control sites from the same trinucleotide-context pool — the negative/unmutated class. Columns: `locus, context, methyl_level, sid, 3mer` (no `alleles`/`ref`/`alt`, since nothing mutated here). Sample: `chr1:279810 TCT 0 TCT-0 0.21248`. `analyze_individual_feature_effects.py:18` additionally drops all `chrX` sites from this set before fitting (autosomes only). |
| `genomic_features/genomic_features13_dnm1_flnk_1k-1M.txt` | 206 MB | Regional-feature values — the same 13 features × 4 window scales = 52 columns as `misc/genomic_features13_genome_1kb.txt` — for each dnm1 site, keyed by `element_id` (= the site's own locus, e.g. `chr10:100003712`, *not* a 1kb window here). ~413K rows; joined to the dnm1 site table above via `locus`↔`element_id` at `analyze_individual_feature_effects.py:15`. |
| `genomic_features/genomic_features13_dnm0_10x_flnk_1k-1M.txt` | 2.05 GB | Same 52 regional-feature columns, for each dnm0 background site — roughly 10x the row count of the dnm1 version above, consistent with the 10x site-count ratio. Joined via `locus` at `analyze_individual_feature_effects.py:20`. |

To vary training-set size: subsample rows from the dnm1 and/or dnm0 site tables (join
each to its matching `genomic_features13_dnm{0,1}_...` feature file on `locus` first),
then refit. The tex's three regimes map onto this data as: (1) shrink both dnm0+dnm1 to
remove tail-`x` coverage entirely, (2) the full dataset as published (baseline), (3) grow
*only* dnm0 (background sites) to densify tail-`x` coverage without adding real DNMs —
matching "increasing the number of background sites (only) in the DNM training set."

### The fitting code that's actually here — and the gap

`analyze_individual_feature_effects.py` (already in this repo) is the real, confirmed
source of `misc/genomic_features13_sel.txt` — its own last line says so
(`# this file corresponds to gs://gnomad-nc-constraint-v31-paper/misc/genomic_features13_sel.txt`).
It loads dnm0+dnm1 and joins in their regional features (lines 13–20), then for every
`(context, window, feature)` triple fits a **univariate** logistic regression of
mutation status (0/1) on that one z-scored feature
(`sm.Logit(...).fit_regularized()`, line 49, inside the loop at lines 31–57), and
Bonferroni-selects the significant ones (lines 61–68) — this *is* the feature-selection
step, and it's directly reproducible and directly subsample-able as-is.

**A ready-made validation target for this step**: line 29 writes its pre-Bonferroni,
per-`(context, window, feature)` coefficient table to exactly
`genomic_features/dnm01_10x_ft_logit_regularized_coef_z_3mer_context_flnk_1k-1M.txt`
(124.8 KB) — and that exact path exists in the bucket. So this file is the *actual
published output* of running this univariate fit on the full, unmodified dnm0/dnm1
training set, before any resizing. Before trusting a refit on resized (subsampled or
densified) training data, first re-run `analyze_individual_feature_effects.py` unmodified
and confirm the output matches this file — a concrete check that the fitting code
correctly reproduces the pipeline before touching training-set size at all. (The
`.selected.txt` version line 67 additionally writes is *not* separately published under
this name in the bucket — per the script's own trailing comment, that output corresponds
to `misc/genomic_features13_sel.txt` instead.)

**But this is not the final model.** The regional-adjustment factor `r(w)` that
`run_nc_constraint_gnomad_v31_main.py` actually computes (lines 209–249) comes from a
**multivariate**, **PCA-reduced** logistic regression per context — one fitted
`L1BinaryResultsWrapper` per trinucleotide context, loaded from
`logit_pickles/logit_regularized_dnm01_{context}_pbonf_pca.pkl` — fit on the *selected*
features' PCA components together, not one feature at a time like
`analyze_individual_feature_effects.py`. The code that actually *fits* that multivariate
model (the analogous `sm.Logit(...).fit_regularized()` call, but on a PCA'd,
multi-feature design matrix) is **not in this repo** — only the *apply/predict* side is
(`run_nc_constraint_gnomad_v31_main.py:231–249`, which loads an already-fitted `.pkl` and
`.pca.pkl` and computes `r(w)` from them). Checked directly: `misc/generic.py`,
`misc/constraint_basics.py`, `misc/nc_constraint_utils.py` (the three modules
`run_nc_constraint_gnomad_v31_main.py` imports, confirmed present in the bucket — see
above) contain no `PCA`, `IncrementalPCA`, or `fit_regularized` reference anywhere, so
this specific gap is real, not just an artifact of an incomplete local checkout.
Reproducing the *exact* published Gnocchi refit under a resized training set therefore
requires writing this multivariate-fit step yourself, using
`analyze_individual_feature_effects.py`'s univariate fit and
`run_nc_constraint_gnomad_v31_main.py`'s apply-side code as templates. The
feature-*selection* stage, though, is fully reproducible today as-is.

### A completely separate DNM-prediction approach, also in the bucket

Found by actually listing `misc/` in full (never done before — a 20-file directory,
cheap to check) rather than only checking directories already named by code or by the
root-level listing. None of the below is referenced by any script in this repo:

| File | Contents |
|---|---|
| `misc/RF_f18_dnm_1M.pkl` | A pickled Random Forest model — the "f18" is 18 features (17 regional features + trinucleotide context), not "feature 18". |
| `fig_tables_init/rf_f18_feature_importance.txt` | That model's feature importances, confirming 18 features: `Trinucleotide context` (importance 0.30, by far the largest), `cDNM maternal`, `Recomb male`, `Nucleosome density`, `Dist to telomere`, `Methyl oocyte`, `Methyl sperm`, `Repl BG02`, `CpG island`, `SINE`, `Dist to centromere`, `LCR`, `Methyl PGC`, `LINE`, `Recomb female`, `Methyl preimplantation`, `GC content`, `cDNM paternal`. |
| `fig_tables_init/rf_f18_predicted_dnms_1M.txt` | `element_id, observed, predicted` at 1Mb resolution (e.g. `chr1-120000000-121000000 → observed=35, predicted=47.9`) — this model's DNM-count predictions vs. real observed DNM counts, i.e. a direct regression-style alternative to the per-context-logistic-regression-plus-PCA approach documented above. |
| `misc/genomic_features17_1kb.txt`, `misc/genomic_features17_1M.txt` | The regional-feature source for the RF model above: 17 columns (`dist2telo, dist2cent, GC_content, RT_BG02, LCR, SINE, LINE, recomb_male, recomb_female, met_sperm, met_oocyte, met_preimplantation, met_pgc, Nucleosome, cDNM_maternal_05M, cDNM_paternal_05M, CpG_island`) — a **superset** of the published 13-feature panel: same 13, plus `RT_BG02` (replication timing) and three extra methylation contexts (`met_oocyte`, `met_preimplantation`, `met_pgc`) that the published pipeline's 13-feature panel never uses. |
| `misc/genomic_features13.tar.gz` | An archived form of the (published, 13-feature) `misc/genomic_features13_genome_1kb.txt` — same data, different packaging. |
| `misc/DNM_decode_psychencode.flip2hl.txt` | `locus, ref, alt` — each DNM locus listed twice, once per allele orientation (e.g. `chr10:100003712 A C` and `chr10:100003712 C A`). Looks like a strand-flip/normalization reference table for the DECODE+PsychENCODE DNM sites; exact use unconfirmed, no code in this repo references it. |

This looks like an earlier or parallel exploration (`fig_tables_init/`, not `fig_tables/`)
of predicting DNM counts directly via Random Forest regression on a broader feature
panel, distinct from — and not clearly related to — the published per-context logistic-
regression-plus-PCA `r(w)` approach. Not investigated further; flagged here so it isn't
mistaken for part of the [pipeline](okf/dnm-training-set-experiment/pipeline.md) above,
and so a future session doesn't have to re-discover it.

**On exhaustiveness**: the file lists in this document are not guaranteed complete. This
DNM-prediction material was missed in an earlier pass specifically because `misc/`
(only 20 files) was never fully listed. Directories still not fully checked for
"dnm"-adjacent content: the unexplained bucket-root `index/` (9,137 subdirs) and `rows/`
prefixes (see the Hail recipe section above — these don't obviously belong to any named
`.ht` table), and the internals of the smaller `*.ht` Hail tables (their `metadata.json`
schemas are known and don't mention DNMs, but their directory listings haven't all been
individually re-checked for stray files beyond the standard Hail structure).

### Files that look like outputs, not inputs, of a DNM-based validation

These share the `_dnm`/`_dnm_1M` naming but are **not** training data — they look like a
separate, already-computed validation of the fitted context-only model against real DNM
counts (paralleling the gnomAD-based `possible`/`expected`/`observed` triple, but for
DNMs), at both per-context and 1Mb-window resolution. No script in this repo produces or
consumes them, so this is inferred from naming/structure, not confirmed by code:

| File | Size | Contents |
|---|---|---|
| `expected_counts_by_context_methyl_dnm_1M.txt` | 114 KB | `element_id, possible, expected` at 1Mb window resolution (e.g. `chr1-0-1000000`), same structure as `expected_counts_by_context_methyl_genome_1kb.txt` but for the DNM cohort. |
| `observed_counts_dnm_1M.txt` | 74 KB | `element_id, variant_count` — observed DNM counts per 1Mb window; same `element_id`s as the row above. |
| `possible_counts_by_context_methyl_dnm.ht/`, `observed_counts_by_context_methyl_dnm.ht/`, `proportion_observed_by_context_methyl_dnm.ht/`, `proportion_observed_by_context_methyl_dnm_.ht/` | Hail tables | Presumably per-context (not 1Mb-binned) versions of the same DNM-based possible/observed/proportion-observed triple. |
| `possible_counts_by_context_methyl_dnm_1M.ht/`, `observed_counts_by_context_methyl_dnm_1M.ht/`, `expected__counts_by_context_methyl_dnm_1M.ht/` (double underscore is in the actual bucket path) | Hail tables | 1Mb-binned Hail-table versions of the two `.txt` files above. |
| `possible_counts_by_context_heptamer_methyl_dnm_1M.ht/` | Hail table | Same idea at **heptamer** (7-mer) context resolution instead of trinucleotide — presumably feeds the `z_heptamer` model referenced in Extended Data Fig. 6 (see the comparisons_*.txt section above), though again unconfirmed by any code in this repo. |

Use `list_bucket_files.py -prefix genomic_features/` or `-prefix <name>.ht/` to browse
any of these directly.

### Implementation and results (2026-07-21)

> **Pared down 2026-08-07.** `dnm_training_size/` now holds only the training-set-SIZE
> dose-response: the 1%/10%/100% bias curves and the `GC_content` selection-frequency
> table (0/4 -> 7/21 -> 23/32 contexts). Its value alongside `fig5/` is the *contrast* --
> shrinking the training set moves Gnocchi toward the context-only model but never past
> it, whereas `fig5/`'s population intervention does go past it (0.046 vs 0.093), so the
> two differ in kind. Removed as superseded: `plot_dnm_bias_comparison.py` (the notebook
> does it), `plot_reliability_gap.py` and `-mode reliability` (fig5 panel D is the same
> diagram on the populations that matter, and the gap measures a level error that cancels
> in r), the pooled GC-only diagnostic, and 2.4 GB of `rr` tables nothing read. `-mode
> validate` moved to `preconditions/`. The narrative below is the record of
> how these results were obtained and names files that no longer exist.

The plan above (regime 1 only — shrink both dnm0+dnm1 by the same random rate; regime 3,
densifying background-only, still needs Hail access to `context_prepared.ht` and wasn't
attempted) is implemented in `dnm_training_size/` (moved out of the repo root and
out of `published/` on 2026-07-21, so this experiment's code and its own outputs live together,
separate from generic scratch downloads), full methods narrative in each script's own
docstring and in `okf/dnm-training-set-experiment/log.md`:

- **`dnm_training_size/run_dnm_training_experiment.py`** — `-mode validate` runs
  `analyze_individual_feature_effects.py`'s own univariate feature-selection logic
  (bug-fixed: that script never defines `output_dir` or imports `os`/`csv`) on the full,
  unmodified training data and diffs against the published coefficient table (pipeline.md
  step 0). `-mode refit -subsample_frac F` subsamples dnm0+dnm1 at rate F, refits
  univariate selection, then fits the previously-unpublished multivariate step (the one
  real gap identified above: standardize -> `IncrementalPCA()` (all components) ->
  `sm.Logit(...).fit_regularized()`, per context, using the subsample's own mean/std, not
  the published `ft_mean_std.txt`), then applies genome-wide by reusing
  `run_nc_constraint_gnomad_v31_main.py` lines 236-270 essentially unchanged. A context
  with no Bonferroni-significant feature under the subsample (or a fit that fails to
  converge) is simply left out of the adjustment and defaults to r=1 for that context —
  exactly line 260's own published fallback, and also exactly what hypothesis claim 1
  predicts happens as training data shrinks. `-mode refit` also prints, per context,
  exactly which `(feature, window)` pairs were selected plus a feature-selection
  frequency table (see "Feature selection" below), and writes a two-panel training-set
  GC diagnostic plot (see "GC diagnostic plots" below). Downloaded bucket files are
  cached in the repo-root `published/` (`-cache_dir`, shared with
  `compute_gc_bias_step1_vs_step2.py` etc.); this experiment's own outputs go to
  `dnm_training_size/output/` (`-output_dir`).
- **`dnm_training_size/plot_dnm_bias_comparison.py`** — joins one or more
  `-mode refit` output tables against the existing step-1/published-step-2 tables and
  produces a GC-binned rank-bias comparison plot, reusing
  `compute_gc_bias_step1_vs_step2.py`'s download/filter/GC-unit/binning code via
  `import compute_gc_bias_step1_vs_step2 as base` (repo root added to `sys.path`
  explicitly, since this script now lives one directory down; only the z/rank
  computation is generalized here, from hardcoded step1/step2 to an arbitrary number of
  named curves).

**Validation, before trusting any result below**: `-mode validate` on the full training
data reproduces the published
`genomic_features/dnm01_10x_ft_logit_regularized_coef_z_3mer_context_flnk_1k-1M.txt`
closely — all 1,664 (context,window,feature) rows comparable, max |coef diff| = 2.6e-4,
100% agree to <1e-3 (not bit-identical, plausibly solver-tolerance/library-version noise).
Separately, `-mode refit -subsample_frac 1.0` (the full, unmodified training set run
through the *entire* reimplemented pipeline, including the reconstructed multivariate
step) reproduces the real published Gnocchi `expected` column
(`fig_tables/constraint_z_genome_1kb.annot.txt`) with Pearson r = 1.0 across 1,984,900
joined windows, median relative difference 4e-6 — strong evidence the reconstructed
multivariate PCA+logit step (missing-code.md's one real gap) is a faithful
reimplementation, not just the univariate selection step.

**Result**: at 1% subsample (4,105 dnm1 / 41,049 dnm0 rows), only 2/32 contexts had any
feature survive Bonferroni selection and converge to a multivariate fit — the other 30
defaulted to r=1. At 10% (41,054 / 410,488 rows), 19/32 contexts fit. Feeding the 1%,
10%, and full-data-as-sanity-check curves alongside the existing step-1/published-step-2
curves into `plot_dnm_bias_comparison.py` (same default filters as
`compute_gc_bias_step1_vs_step2.py`: noncoding, pass_qc, autosome+PAR, n=1,840,181; 20
fixed-width GC bins) gives a clean, genome-wide, dose-response confirmation of
[hypothesis.md](okf/dnm-training-set-experiment/hypothesis.md) claim 1 — output plot
`dnm_training_size/output/dnm_training_set_size_bias.pdf`:
- The full-data sanity curve overlays published step-2 almost exactly at every bin, as
  expected from the Pearson-r=1.0 check above.
- The 1% curve is nearly indistinguishable from the step-1 (context-only) curve at
  **every** GC bin across the full 0.2-0.73 range (e.g. GC bin centered at 0.71:
  step1=0.284 vs. dnm_1pct=0.283, vs. step2_published=0.867) — Gnocchi fully collapses to
  the context-only model's (much smaller) bias when the training set is this sparse.
- The 10% curve sits smoothly **between** step-1 and published step-2 at every single
  bin (same GC bin: dnm_10pct=0.808) — a partial collapse, not a step function.

This is a stronger result than chen_formula.tex's own claim (which only described the
small/large endpoints): the transition is smooth and monotonic in training-set size
across the *entire* GC range, not just in the tails, and not on a hand-picked example —
it's genome-wide, using the real fitted models.

**Feature selection, by training-set size** — a second, direct line of evidence for the
same mechanism, from `-mode refit`'s printed "feature-selection frequency" table
(`dnm_training_size/output/selected.dnm_refit_*.txt`): the number of contexts in
which `GC_content` itself clears Bonferroni significance grows with training-set size —
**0/4** contexts with any selected feature at 1% subsample, **7/21** at 10%, **23/32**
(all contexts that fit anything) at full scale. At 1%, `GC_content` never reaches
significance in *any* context — the model literally has no statistical power to detect a
GC effect to adjust for, regardless of what the true relationship is; the two most
commonly selected features instead are `cDNM_maternal_05M` and `met_sperm` (n=2 each,
out of only 4 total selected features). At full scale, `GC_content` is the 5th
most-selected feature overall (23/32 contexts), behind `recomb_male` (31), `dist2telo`
(28), `cDNM_maternal_05M` (28), and `CpG_island` (24) — `CpG_island` and `met_sperm` are
both themselves correlated with GC content (this is exactly why
`run_nc_constraint_gnomad_v31_main.py` line 217's `ft_corr_met` list excludes them,
alongside `GC_content` itself, for CpG-context models), so the *effective* GC-content
sensitivity of the full-scale model is arguably even broader than the direct
`GC_content`-selection count alone suggests.

**GC diagnostic plots** (`dnm_training_size/output/gc_diagnostic.dnm_refit_*.pdf`,
one per subsample size) — top panel: histogram of the (possibly subsampled) training
pool's own GC content (dnm1 vs dnm0); bottom panel: a POOLED (all contexts combined, GC
content only, ignoring every other feature) univariate logistic fit of mutation status on
GC content, plotted against binned empirical mutation proportions with binomial
standard-error bars. This is a deliberately simplified, context-agnostic diagnostic —
*not* the actual per-context multivariate model Gnocchi uses (that can't be reduced to a
single GC-only curve, since it's fit jointly on several PCA-whitened features and differs
by context/window). What it mainly shows is training-pool *sparsity*, not a dramatically
shifting fit: the pooled fit's coefficient is nearly flat across all three subsample
sizes (z-coef 0.180 at 1%, 0.166 at 10%, 0.167 at 100%) — a single-feature fit on tens of
thousands of points is already fairly stable even at 1% (4,105+41,049=45,154 rows). What
*does* visibly change is tail coverage and noise: at 1%, the observed GC range truncates
around ~23-73% with wide error bars at the edges (one bin has a single point sitting at
P=1.0); at 10%/100%, coverage extends further and error bars shrink, though even the
full-data plot has wide error bars below GC~25% simply because that region is
intrinsically rare in the genome. **Bottom line**: the pooled GC-only diagnostic is a
useful sparsity visualization, but the real "sensitivity to tails without fitting them
well" mechanism is best evidenced by the feature-selection-frequency numbers above and by
the actual bias comparison plot — both reflect the real per-context multivariate model,
which is what actually drives Gnocchi's adjustment; the pooled single-feature fit is not
a faithful stand-in for it.

See `okf/dnm-training-set-experiment/log.md`'s 2026-07-21 entries for exact per-bin
numbers, full per-context selected-feature listings, and run provenance (contexts fit,
timings, etc.).

**Training-set reliability diagram (`-mode reliability`, added same day)**: a more
direct alternative to the pooled GC-only diagnostic above, using the real per-context
multivariate model instead of a simplified pooled proxy. Two candidate designs were
considered and rejected first: (1) a genome-wide reliability diagram using the r(w)
numerator `pred = σ(β₀+β·z(w))` (already computed in `apply_genome_wide_context()` but
discarded) against real genome-wide `observed/possible` rates — rejected because
`pred`'s absolute level is confounded by the fixed 10:1 dnm0:dnm1 case-control training
design (case-control logistic regression gives a consistent slope but a biased
intercept), so a direct comparison to the true, much rarer genome-wide DNM rate would
mostly reflect that sampling artifact, not local GC miscalibration; (2) Platt scaling —
rejected as a fix for *local* bias since it's a single global affine recalibration and
can't repair bias that varies by GC bin without becoming bin-conditional recalibration,
which is just what adding `GC_content` as a real feature already achieves at large N.

What was implemented instead stays entirely within the training population, which
cancels the case-control intercept bias (both the prediction and the empirical rate it's
compared to come from the same case-control-sampled population): `predict_training_set()`
evaluates each context's fitted multivariate model on that context's own dnm1/dnm0
training sites (each site's own feature vector, not a window's aggregated values), and
`plot_training_reliability_diagram()` bins the pooled predictions by site-level GC
content and plots mean fitted probability against the mean empirical label rate with
binomial SE. Crucially, this mode never touches the genome-wide features/expected-count
files or the duckdb join — only the training-set tables already cached in `published/` — so it
re-ran all three subsample sizes (1%/10%/full) in well under a minute combined, and
reproduced the earlier `-mode refit` runs' feature-selection/contexts-fit counts exactly.

**Result**: at 1% subsample, mean fitted probability is essentially flat (~0.055–0.065)
across the whole GC range — the model isn't attempting any GC dependence at all,
matching `GC_content` clearing Bonferroni selection in 0/4 fitted contexts at this size.
At 10%, both curves track closely through the bulk with a small, noisy tail divergence.
At full scale, fitted and empirical probability are nearly indistinguishable through the
dense bulk (e.g. GC≈39%: pred=0.0868 vs empirical=0.0859, n≈1.04M) but the fitted curve
visibly overshoots the empirical rate in the sparse high-GC tail (GC≈74%: pred=0.272 vs
empirical=0.155, n=1,808; GC≈77%: pred=0.291 vs empirical=0.153, n=649) — a second,
more direct, in-sample confirmation of "sensitive to the tails, doesn't fit them well,"
using the real per-context multivariate model rather than the pooled GC-only proxy, and
a plausible direct explanation for why Gnocchi's r(w) adjustment runs high specifically
in high-GC windows genome-wide. Full per-bin numbers and rejected-alternatives reasoning:
`okf/dnm-training-set-experiment/log.md`, "2026-07-21 (reliability diagram)".

## `compute_gc_bias_step1_vs_step2.py` — Figure 2A-style rank-based bias analysis

The script's own docstrings are kept short and point back here; this section is the
canonical, extractable methods narrative for the rebuttal/revised paper. Every
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

## `fig3/` — the new Fig. 3 for the McHale et al. manuscript

Two stacked panels sharing a GC-content x-axis, built by `fig3/fig3.ipynb`. See
`fig3/README.md` for the operational detail; this section records the methodological
decisions so they aren't re-derived.

**Panel A** generalizes `compute_gc_bias_step1_vs_step2.py`'s rank statistic from two
curves to three: step 1 (context-only, `r≡1`), full Gnocchi (step 2), and **depletion
rank**. The claim it supports is stronger than "step 1 is less biased than Gnocchi": step
1 is biased about as little as an independently constructed metric, so the GC bias is
*introduced by* the regional-feature adjustment rather than inherited from the
sequence-context model. Measured on the full window set (1,840,310 windows after z
filtering, mean GC 0.399): mean |rank − 0.5| across bins is **0.093 for step 1 vs 0.212
for step 2** — step 1 is 2.3× closer to unbiased. The depletion-rank number slots in
alongside these once the BED is supplied.

**Depletion rank is a separate window set, not a join.** `experiment.1.ipynb` in
constraint-tools keeps Chen, Halldorsson, and CDTS windows as three separate files;
depletion rank is defined on Halldorsson windows (different size), so it is ranked within
itself and overlaid, never joined on `element_id`. Legitimate for a conditional-mean-rank
plot — the rank is uniform on (0,1) by construction for every curve — but the caption must
say so. Sign convention: LOW depletion rank means MORE constrained, the opposite of
Gnocchi's z, so `depletion_rank.py` applies the `1 - DR` complement that
`experiment.1.ipynb` uses. Getting this backwards silently flips the curve.

**The DR file is not obtainable here.** It is
`{CONSTRAINT_TOOLS_DATA}/depletion_rank_scores/41586_2022_4965_MOESM3_ESM.noncoding.enhancer.BGS.gBGC.GC_content.bed`
on the constraint-tools HPC path. `fig3/depletion_rank.py` is written defensively
(explicit column resolution, GC unit auto-detection, loud errors) and has been exercised
against synthetic input — column resolution both ways, fraction-vs-percent detection,
the complement, and the missing-column error path all verified — but **never against the
real file**. Check its printed summary the first time it runs.

**Panel B** (revised 2026-08-04, after a reviewer-style challenge: "the DNM model is
conditioned on sequence context, but that doesn't appear in Fig 3B") is now the
**multiplicative error in the adjustment factor**, stratified by CpG status:

$$\text{inflation} = r_\text{model}/r_\text{true} = \frac{\overline{\text{pred}}}{\text{empirical}}$$

(the $\sigma(\beta_0)$ denominators cancel exactly). Three findings drove the change:

1. **Every per-context model is exactly calibrated in the mean, in-sample** — max
   |mean(pred) − mean(label)| across the 32 contexts is **1.0e-7**. That is the logistic
   regression intercept score equation, not a result. So panel B contains no
   between-context information at all; all its structure is within-context and
   GC-conditional.
2. **The pooled gap is an exact n-weighted mean of within-context gaps** (composition
   scales both sides identically), so pooling is not a naive artifact — but it *hid* that
   the high-GC signal is almost entirely CpG. At GC 0.70 the CpG gap is +0.186 while the
   non-CpG gap is −0.001. CpG share rises from 0.9% of sites at GC 0.25 to 32% at GC 0.74.
3. **Mechanism**: `FT_CORR_MET` (`run_nc_constraint_gnomad_v31_main.py:217,227`) strips
   `GC_content`, `CpG_island`, `Nucleosome`, `SINE`, `met_sperm` from CpG-context models
   only — all five had cleared Bonferroni selection in all four CpG contexts and are
   discarded anyway. ACG is left fitting on just `dist2telo, recomb_male, recomb_female`.
   Gnocchi forbids GC-content adjustment in exactly the contexts that dominate high-GC
   windows.

On the ratio scale the story changes again: non-CpG is **not** clean. It under-predicts by
26% at GC 0.66 (f = 0.74) — invisible on the absolute scale because its baseline rate is
small. Both groups are badly miscalibrated above GC 0.58, in opposite directions.

**The honest caveat, and it is important.** Panel B does NOT explain panel A in the GC
bulk. Measured directly by applying a uniform inflation f to every window's expected count
and re-ranking against the unperturbed genome-wide z distribution (1,843,559 windows,
median expected 174):

| f | 0.74 | 0.90 | 0.97 | 1.00 | 1.03 | 1.10 | 1.55 |
|---|---|---|---|---|---|---|---|
| mean rank | 0.083 | 0.302 | 0.439 | 0.500 | 0.560 | 0.687 | 0.984 |

Gnocchi's rank is extremely sensitive to r — 10% inflation already moves mean rank to 0.69.
Panel A's r-adjustment contribution (step-2 rank minus step-1 rank) is ~+0.22 at GC 0.57,
requiring f ≈ 1.10–1.12. Panel B measures f ≈ 0.98–1.00 there for both groups: an order of
magnitude too small, and wrong-signed (f < 1) across GC 0.55–0.66. Only the top two GC
bins are large enough to matter.

Most likely because panel B is the wrong population for the question — it is **in-sample**,
on **case-control-sampled** training sites, at **site-level** feature vectors, whereas r(w)
is applied **out-of-sample**, genome-wide, at **window-aggregated** feature values. A
genuinely causal panel B would measure r's error genome-wide at window level;
`expected_counts_by_context_methyl_dnm_1M.txt` + `observed_counts_dnm_1M.txt` (bucket, see
data inventory) are the obvious source — 1 Mb resolution, coarser, but out-of-sample and
genome-wide. **Not attempted yet.** Until then, do not write "miscalibration translates
directly into bias" in the caption; the figure supports that only in the extreme GC tail.

The older pooled absolute-gap panel is still available as
`panels.panel_calibration_gap()`; it used symlog (not log) because the signed gap changes
sign and a log axis would silently drop the bulk-GC bins. The ratio panel uses a plain log
axis, which is available because a ratio is strictly positive.

Panel A's y-range is fixed to `[0,1]` to stay directly comparable to Fig. 2A, though the
curves only occupy ~0.26–0.88 of it; `panel_rank_bias(yrange=...)` tightens it at the cost
of that comparability.

`GENEHANCER_BED` remains unavailable (licensed), so "neutral" here is still noncoding +
`pass_qc` + autosome/PAR, exactly as in the Fig. 2A-style analysis — same caveat, same
place in the notebook's config cell.

### Methylation, and why the training-set calibration panel measures the wrong thing (2026-08-04)

Prompted by: "Chen et al also stratify by methylation, in addition to sequence context."

**Where methylation enters.** Step 1 only. `fitted_po` is keyed by
`(context, ref, alt, methylation_level)`, and for CpG contexts the C>T rate runs from
0.23 at methyl level 0 to 0.99 at level 15 — a **4.3x** range inside one trinucleotide
context, the single largest rate effect in the whole model. Step 2's `r(w)` never sees
methylation: `logit_regularized_dnm01_{context}_pbonf_pca.pkl` is one model per
trinucleotide context, pooling all 16 methylation levels (confirmed: no `methyl`
reference anywhere in `run_nc_constraint_gnomad_v31_main.py` lines 209-270).

**Methylation composition swings hard with GC.** Measured on the CpG-context training
sites (dnm0+dnm1 joined to their regional features):

| GC | 24 | 42 | 57 | 67 | 72 | 77 |
|---|---|---|---|---|---|---|
| mean methyl level | 6.12 | 6.60 | 5.12 | 2.24 | 0.35 | 0.10 |
| fraction hypomethylated (level ≤1) | 0.021 | 0.023 | 0.173 | 0.615 | 0.940 | 0.976 |
| empirical DNM rate | 0.607 | 0.530 | 0.549 | 0.470 | 0.267 | 0.204 |

High-GC CpGs are CpG islands: almost entirely unmethylated, and their mutation rate
collapses by ~2.6x. A methylation-blind per-context model cannot represent this, which is
exactly the CpG over-prediction the stratified panel B picked up (inflation 1.55 at
GC 0.70, 2.29 at GC 0.74).

**But that miscalibration does not reach Gnocchi, because r is a RATIO.**
`r(w) = σ(β₀+β·z(w))/σ(β₀)` — a level error common to numerator and denominator cancels.
Only feature-driven *variation* survives. Measured genome-wide from
`rr_by_context.dnm_refit_full.txt` joined to `GC_content_1k`:

| GC | 28 | 42 | 52 | 62 | 72 | 77 |
|---|---|---|---|---|---|---|
| mean r(w), **CpG** contexts | 0.996 | 0.996 | 1.002 | 1.004 | 0.998 | 1.005 |
| mean r(w), **non-CpG** contexts | 0.975 | 1.007 | 1.074 | 1.258 | 1.518 | 1.616 |

**CpG models are inert** — r ≈ 1.00 at every GC. `FT_CORR_MET` strips their GC-correlated
features, leaving `dist2telo, dist2cent, LINE, recomb_*`, which barely vary with GC, so
they predict ~the context average everywhere. Their large level miscalibration cancels
out entirely. **The GC bias is driven wholly by the non-CpG contexts**, which do retain
`GC_content` (selected in 23/32 contexts) and inflate r to 1.6.

**The effective per-window adjustment quantitatively predicts panel A.** With
`r_eff(w) = expected_step2/expected_step1`:

| GC | 0.268 | 0.400 | 0.469 | 0.539 | 0.573 | 0.642 |
|---|---|---|---|---|---|---|
| mean r_eff | 0.954 | 0.993 | 1.026 | 1.088 | 1.136 | 1.297 |
| rank(step2) − rank(step1) | −0.072 | −0.009 | +0.053 | +0.157 | +0.224 | +0.423 |

r_eff crosses 1.0 and the rank delta crosses 0 at the same GC (~0.40), and the deltas
match the uniform-inflation sensitivity table above to within ~0.03 at every bin.

**Ground-truth test at 1 Mb — the link that was missing.** `r` is meant to capture regional
DNM-rate variation left over after context+methylation. That residual is directly
measurable: `observed_counts_dnm_1M.txt / expected_counts_by_context_methyl_dnm_1M.txt`.
Comparing it against the model's own aggregated r (both normalized to mean 1, weighted by
step-1 expected):

| GC (1 Mb) | 33.5 | 36.8 | 39.2 | 41.7 | 44.1 | 46.6 | 49.1 | 51.4 | 54.0 |
|---|---|---|---|---|---|---|---|---|---|
| n 1 Mb windows | 275 | 564 | 640 | 504 | 341 | 204 | 115 | 48 | 20 |
| DNMs | 37.6k | 86.0k | 96.1k | 76.0k | 51.2k | 31.8k | 18.1k | 8.2k | 3.7k |
| **r model** | 0.969 | 0.977 | 0.985 | 1.005 | 1.018 | 1.033 | 1.059 | **1.117** | **1.150** |
| **r true** | 1.015 | 1.073 | 1.021 | 0.985 | 0.940 | 0.933 | 0.910 | **0.903** | **0.926** |

**They move in opposite directions.** The model's adjustment rises with GC; the real
residual DNM rate falls. At GC 51%, r is too high by a factor of 1.24. This is the direct
evidence that the adjustment is wrong, not merely that it exists — everything before this
only established that r *causes* the GC-dependence.

Limitations: 1 Mb averaging compresses GC to 33-54%, so this cannot probe the high-GC tail
where panel A's bias is largest; and it is in-sample for the DNM cohort (which makes the
failure more damning, not less — the model cannot reproduce the aggregate regional
variation of its own training data). Per-bin Poisson error on the DNM totals is 0.3-1.6%.

**Methylation is RULED OUT as a cause, not implicated.** Chen et al. model methylation
carefully and correctly, in step 1. Step 2 is methylation-blind, which is a plausible
suspect — but the CpG-context r values above are flat at 1.00 across the entire GC range,
so the methylation-blind models contribute no GC-dependent adjustment whatsoever. The
methylation/CpG-island material above explains only why the *training-set calibration*
diagnostic showed a dramatic CpG signal: that is a level error, and level errors cancel in
the ratio r. It says nothing about Gnocchi's bias. Do not put it in the causal narrative.

**Consequence for Fig. 3B.** The training-set calibration/inflation panel measures
*level* error, which cancels in r and therefore does not propagate to Gnocchi. The panel
that actually closes the causal loop is **mean r_eff vs GC, genome-wide, split by CpG
status** — same population and same unit as panel A, the literal quantity applied, and
quantitatively predictive of panel A's rank shift. Prefer it over
`panel_calibration_ratio` for the manuscript figure.

### Peter's proposed CpG mechanism — steps 1-4 confirmed, step 5 refuted (2026-08-04, end of session)

Proposed chain: (1) the CpG DNM model conditions on neither methylation nor GC content;
(2) methylated CpGs dominate the genome and are over-represented in the DNM training set
because they are more mutable; (3) so the fitted model predicts ~the methylated-CpG rate;
(4) so it overestimates at unmethylated CpGs, which are hypomethylated CpG islands in
GC-rich regions; (5) therefore r is too high in GC-rich windows, causing Gnocchi's bias.

**Steps 1-4 are confirmed by measurement.** CpG-context models are fit without
`GC_content`/`CpG_island`/`Nucleosome`/`SINE`/`met_sperm` (`FT_CORR_MET`) and without any
methylation term. CpG training sites at GC 72-77% are 94-98% hypomethylated with empirical
DNM rate 0.20-0.27 vs 0.53-0.61 in the GC bulk, and the model over-predicts there by
1.55-2.29x. All of that is real.

**Step 5 does not follow, and is contradicted directly.** Decomposing r_eff genome-wide by
CpG status, weighted by step-1 expected:

| GC | 28.5 | 42.3 | 52.1 | 62.0 | 72.1 | 77.2 |
|---|---|---|---|---|---|---|
| CpG share of expected | 0.034 | 0.088 | 0.149 | 0.246 | 0.394 | 0.453 |
| **r, CpG contexts** | 0.992 | 0.996 | 1.001 | 0.994 | 0.988 | **1.007** |
| **r, non-CpG contexts** | 0.957 | 1.005 | 1.079 | 1.316 | 1.698 | **1.921** |
| r_eff (all) | 0.958 | 1.004 | 1.067 | 1.237 | 1.418 | 1.507 |
| **counterfactual: r_eff if non-CpG r were exactly 1** | 1.000 | 1.000 | 1.000 | 0.998 | 0.995 | **1.003** |

CpG contexts do carry real weight (45% of expected at GC 77%), but their r is flat at
~1.00 across the whole GC range, and the counterfactual holding non-CpG r at 1 shows
**no GC trend whatsoever**. The entire trend is non-CpG.

**Why step 5 fails, two independent reasons:**

1. **r is a ratio; a level error cancels.** r = sigma(b0 + b.z(w)) / sigma(b0). The CpG
   model's inability to represent methylation biases its *level* — equally in numerator
   and denominator — so it divides out. Only feature-driven *variation* survives.
2. **The CpG models cannot vary with GC even in principle.** `FT_CORR_MET` strips
   `GC_content` and `CpG_island`; what remains (`dist2telo, dist2cent, LINE, recomb_*`)
   barely varies with GC. So they predict ~the context average everywhere -> r == 1.

**And r ~ 1 for CpG is arguably correct anyway**: step 1's `fitted_po` is already keyed by
methylation level, so the low rate at hypomethylated CpG-island sites is ALREADY in
expected_step1. There is nothing left for r to adjust. The methylation modelling Chen et
al. did is doing its job; the mechanism above would bite only in a variant of Gnocchi
where CpG models retained GC_content.

**Net: the CpG route is closed. The bias is entirely a non-CpG phenomenon**, and the 1 Mb
ground-truth test above (r model rising while r true falls) is the evidence that the
non-CpG adjustment is wrong rather than merely present.

## What Gnocchi applies, and whether it is right (2026-08-05)

Both panel-B candidates from the previous session are now built, as
`fig3/r_eff.py` + `fig3/empirical_r.py` + `fig3/make_r_figures.py` (one command,
`python fig3/make_r_figures.py`; ~1 min warm). They supersede the training-set
calibration panel, which measures a level error that cancels in r.

### Figure `r_eff_decomposition.pdf` — the adjustment Gnocchi applies

`r_eff(w) = E2(w)/E1(w)`, the E1-weighted mean of the pipeline's own per-context r,
decomposed exactly as `r_eff = Pi*r_CpG + (1-Pi)*r_non` with `Pi = E1_CpG/E1`.
Aggregated per GC bin as **ratios of summed expected counts**, not means of per-window
ratios, which is both the adjustment the bin actually receives and what keeps the
decomposition exact bin by bin. (Earlier ad-hoc tables in this file used unweighted
per-window means; small differences from those numbers are expected and are not errors.)

| GC | 0.236 | 0.334 | 0.400 | 0.469 | 0.539 | 0.608 | 0.677 | 0.748 |
|---|---|---|---|---|---|---|---|---|
| windows | 1,212 | 283,824 | 383,541 | 181,038 | 46,719 | 7,652 | 1,000 | 259 |
| Pi (CpG share of E1) | 0.025 | 0.046 | 0.074 | 0.112 | 0.155 | 0.219 | 0.330 | 0.426 |
| **r_non** | 0.956 | 0.968 | 0.993 | 1.029 | 1.101 | 1.264 | 1.546 | 1.785 |
| **r_CpG** | 0.991 | 0.994 | 0.996 | 0.998 | 1.004 | 0.998 | 0.989 | 0.986 |
| r_eff (all) | 0.957 | 0.969 | 0.993 | 1.025 | 1.086 | 1.205 | 1.362 | 1.444 |
| **counterfactual: non-CpG r = 1** | 1.000 | 1.000 | 1.000 | 1.000 | 1.001 | 1.000 | 0.996 | 0.994 |

The counterfactual is flat within 0.6% across the entire range while CpG contexts carry
43% of the expected-count weight at GC 0.75. This is the visual form of "the bias is
wholly non-CpG", and it is a decomposition identity, not a fit.

**Provenance, and why it is trustworthy.** The published pipeline writes its per-context
r to a local `output_dir`, not the bucket — confirmed, no such object exists under any
prefix — so this uses the reimplemented refit's `rr_by_context.dnm_refit_full.txt`. That
is *validated per figure*, not assumed: the published `r_eff` for the "all" curve is
directly computable as `expected_step2/expected_step1`, and refit vs published agree to
**max 1.0e-4 across all 20 GC bins, median 3.9e-6**. `make_r_figures.py` prints this
check on every run.

**Cost trick worth keeping**: only the four CpG contexts are joined between the 3.3 GB
per-context expected file and the 4.0 GB rr file; non-CpG comes by subtraction from
per-window totals that already exist as small files. An 85M x 85M join becomes 10M x 10M,
and the whole thing runs in ~5 s.

### Figure `r_non_vs_empirical.pdf` — and it is wrong

The empirical target is a *rate*, directly measurable per trinucleotide context c and
GC bin g:

    r_true_c(g)  proportional to  DNMs_c(g) / opportunities_c(g)

where `opportunities_c(g)` is the number of possible SNV sites of context c in the
analyzed windows of that bin (`possible` in the per-context export). An earlier
write-up put the step-1 *expected* count E1 in the denominator, which is correct but
opaque: within one non-CpG context there is a single methylation level, so
`E1 = opportunities x const_c`, and the constant cancels in the per-context
normalization. Verified — the two denominators agree to <0.1% at every bin. The code
now defaults to `opportunities` because that is what a rate needs.

This is the same construction as the 1 Mb ground-truth test, but per-context and binned on 1 kb
GC, so it reaches the high-GC tail 1 Mb averaging compresses away. DNM loci are mapped to
their containing 1 kb tile, and numerator and denominator are restricted to the *same*
analyzed windows.

**Normalization — no free constant is needed, and this was corrected once.** Since
`r_c(w) = sigma(b0 + b.z(w)) / sigma(b0)` is already the rate at w over the rate at the
average feature vector, the empirical analogue is the identical ratio: this bin's DNM
rate over the context's overall DNM rate. So **both** sides are normalized per context to
E1-weighted mean 1 over the analyzed windows, and each curve then means "1 = no
adjustment relative to this context's own average". Two reasons this shape is forced,
not chosen:

- *Per context is mandatory*, because `D_c/E1_c` is not on a common scale across
  contexts — `fitted_po` saturates by different amounts, so the "gnomAD expected" to
  "DNM expected" conversion is context-specific. Levels cannot be compared across
  contexts.
- *Both sides, identically*, because as GC rises the trinucleotide mix shifts; if only
  the empirical were normalized, un-normalized between-context level differences in the
  model would leak into the aggregate as false GC dependence.

An earlier version instead rescaled only the empirical side by
`kappa_c = Sum_g E2_c(g) / Sum_g D_c(g)`, which is algebraically this same ratio times the
model's own mean `r_c` — it worked, but made the curve's absolute level depend on the
model. The two agree to ~0.03% in the inflation ratio (the model's mean `r_c` is close to
1), so no conclusion changed; the current form is just the principled one.

| GC | 0.262 | 0.332 | 0.401 | 0.471 | 0.541 | 0.576 | 0.610 | 0.645 | 0.680 |
|---|---|---|---|---|---|---|---|---|---|
| DNMs in bin | 3,393 | 42,893 | 60,187 | 30,583 | 8,533 | 3,640 | 1,441 | 491 | 203 |
| **r_non fitted** | 0.952 | 0.969 | 0.994 | 1.030 | 1.103 | 1.166 | 1.267 | 1.408 | 1.550 |
| **r_non observed** | 1.015 | 1.012 | 0.989 | 0.989 | 1.023 | 0.995 | 1.036 | 1.122 | 1.287 |
| SE (Poisson) | 0.018 | 0.005 | 0.004 | 0.007 | 0.013 | 0.020 | 0.034 | 0.063 | 0.112 |
| **over-adjustment** | 0.94 | 0.96 | 1.01 | 1.04 | 1.08 | 1.17 | 1.22 | 1.26 | 1.21 |

The fitted adjustment climbs monotonically; the adjustment the DNMs support is flat near
1.0 until GC ~0.55 and then rises far more slowly. The ratio crosses 1 at GC ~0.40 and
reaches **1.22-1.26**, many SEs from 1. Direction and rough size agree with the
independent 1 Mb test (model r rising while true residual rate falls), now over a wider
GC range and per-context.

Three things checked so the result is not an artifact:

1. **The denominator cannot shape the curve.** Non-CpG contexts have a single methylation
   level in `mutation_rate_by_context_methyl.txt` (verified: 84 rows, all level 0, one
   `fitted_po` per `(context, ref, alt)`), so `E1_c = possible_c x const_c` exactly, and
   the constant is absorbed by the per-context rescaling. Using `possible` instead of
   `E1` is the same curve.
2. **The per-context genome table reproduces the per-window one** to 4.4e-6 in r_non, so
   both figures describe the same windows. `make_r_figures.py` asserts this.
3. **The denominator is an opportunity count, and the choice does not matter.**
   `opportunities` (= `possible`) and `E1` give the same curve to <0.1% (see above).

**The one live caveat, now quantified — CALLABILITY.** The numerator counts DNMs falling
anywhere in an analyzed window; the denominator counts only *gnomAD-callable* positions.
That fraction is **not** flat in GC — measured directly, `possible/3000` per window:

| GC bin centre | 0.30 | 0.37 | 0.44 | 0.51 | 0.57 | 0.61 | 0.68 | 0.75 |
|---|---|---|---|---|---|---|---|---|
| callable fraction | 0.905 | 0.898 | 0.879 | 0.845 | 0.790 | 0.749 | 0.737 | 0.702 |

Short-read coverage drops in GC-rich sequence, so gnomAD's depth filter removes ~20% more
territory at GC 0.61 than at GC 0.37. Whether that biases the comparison depends on the
DECODE/PsychENCODE trio call sets' own callability, which is not in this bucket:

- If DNM callability **tracks gnomAD's** (both Illumina WGS, similar GC dropout), then
  `possible` is the matched denominator and no correction applies -> over-adjustment
  **1.22** at GC 0.61.
- If the DNM call set is closer to **complete**, the denominator should be divided by the
  callable fraction -> over-adjustment **1.44** at GC 0.61, **1.51** at GC 0.645.

So the finding survives either way and the correction only strengthens it; but the
*magnitude* is uncertain across roughly [1.22, 1.44] and the caption must not quote 1.22
as if it were tight. `callable_fraction_by_bin()` computes the table and
`empirical_from_dnm_counts(..., callable_fraction=...)` applies the correction. (The
correction is applied per bin, i.e. assumed uniform across contexts within a bin;
dropout is a property of positions, so this is approximate.)

The residual, untestable part of the same caveat: if trio DNM calling is *less* sensitive
in GC-rich sequence than gnomAD is, the bias runs the other way and part of the measured
over-adjustment is technical. Both directions belong in the caption.

**The second estimator, and what actually went wrong with it** (corrected after direct
testing — an earlier version of this section blamed the wrong thing). Using the
case-control rate among dnm1/dnm0 training sites, `p_hat = k/n`, over the *whole* training
population gives the opposite answer above GC 0.54 (observed r rising to 2.02, i.e. the
model *under*-adjusting). Three candidate causes were tested by changing one thing at a
time:

| Estimator | GC 0.61 | GC 0.645 | GC 0.68 |
|---|---|---|---|
| DNMs / opportunities, analyzed windows, tile GC (primary) | 1.033 | 1.119 | 1.283 |
| DNMs / dnm0 controls, analyzed windows, tile GC | 1.059 | 1.147 | 1.363 |
| DNMs / dnm0 controls, analyzed windows, **site-flank GC** | 1.074 | 1.106 | 1.355 |
| `p_hat`, **whole genome**, site-flank GC | 1.593 | 2.021 | — |
| (fitted model) | 1.264 | 1.404 | 1.546 |

So it is **not** the denominator (a real sample of background sites reproduces E1 to
1-2%) and **not** the binning variable (site-flank vs tile GC barely matters). It is the
**population**: `combine_non_cpg` computes its E1 weights and its per-context
normalization over the analyzed window set, so rates measured over the whole genome are
being combined with weights derived from a different population. Restricting the same
sites to the analyzed windows collapses row 4 onto rows 1-3.

Two consequences. First, the primary result is *more* robust than it looked — two
independent denominators agree. Second, the earlier pooled "non-CpG under-predicts by 26%
at GC 0.66 (f = 0.74)" number is retired: it came from the same unrestricted population.

`check_dnm0_sampling()`'s finding stands as a real property of the training data — the
dnm0 pool is not GC-uniform within a context (2.0-fold for CCC; pooled, depleted 0.95 at
low GC and enriched 1.25 at high GC), and the 10:1 ratio holds only genome-wide, not per
context (0.76 for ACG to 24.8 for GAA, i.e. close to a uniform genomic sample). It is just
not the explanation for the disagreement above.

## Why the non-CpG models inflate r: the training population, not the background sample (2026-08-06)

This answers the "still open" question the previous session left, and it does so by
isolating the cause rather than arguing for it. Two scripts, both fast (they reuse
`make_r_figures.py`'s cached per-`(context, bin)` tables):

```
.venv/bin/python fig3/plot_dnm_probability.py            -> dnm_probability_non_cpg.pdf
.venv/bin/python fig3/plot_training_representativeness.py -> training_representativeness.pdf
```

### The observation that started it

`fig3/plot_dnm_probability.py` is the reliability diagram restricted to non-CpG
contexts: mean fitted P(DNM) from the per-context logistic regressions against the
empirical fraction of that GC bin's training examples that are DNMs. Through GC 0.55 the
two curves are within 2%. Above that they separate in *both* directions — the model
under-predicts by 26% at GC 0.66 (0.130 vs 0.177), then the empirical curve turns over
and falls while the fitted one keeps climbing, so the model over-predicts by 29% at
GC 0.74 (0.156 vs 0.121). The fit is smooth and monotone in GC; the training data is not.

That non-monotonicity is absent from `r_non_vs_empirical.pdf`, whose empirical curve is
flat near 1.0 until GC 0.55 and then rises smoothly. Same DNMs, different shape.

### Isolating which ingredient causes the shape difference

`fig3/training_representativeness.py` builds the same non-CpG empirical curve four ways,
changing exactly one ingredient per step, each normalized to E1-weighted mean 1 so only
shape is compared:

| GC | 0.40 | 0.47 | 0.51 | 0.54 | 0.58 | 0.61 | 0.65 | 0.68 |
|---|---|---|---|---|---|---|---|---|
| Gnocchi's fitted r | 0.994 | 1.031 | 1.060 | 1.104 | 1.166 | 1.267 | 1.408 | 1.551 |
| **A** DNMs/opportunities, analyzed windows | 0.989 | 0.989 | 0.983 | 1.023 | 0.995 | 1.036 | 1.123 | 1.288 |
| **B** DNMs/background sites, analyzed windows | 0.990 | 0.989 | 0.982 | 1.019 | 1.000 | 1.062 | 1.150 | 1.368 |
| **C** DNMs/background sites, whole genome | 0.979 | 1.015 | 1.037 | 1.138 | 1.312 | 1.702 | 2.235 | 2.188 |
| **D** pooled DNM fraction, whole genome | 0.979 | 1.037 | 1.075 | 1.188 | 1.372 | 1.754 | 2.238 | 2.250 |

Maximum disagreement up to GC 0.62, which `report_ladder()` prints on every run:

```
denominator  (A vs B): 2.4%      swapping opportunity counts for real dnm0 controls
aggregation  (C vs D): 4.3%      per-context normalization vs raw pooling
POPULATION   (B vs C): 37.6%     analyzed noncoding windows vs whole genome
```

**So the background sample is not the cause.** Replacing the genome's opportunity count
with the actual dnm0 sites — the thing the hypothesis pointed at — moves the curve by at
most 2.4%. Nor is the pooling. What moves it is *which windows* are measured, by an order
of magnitude more than either.

### Why the population matters: the training set leaves the noncoding genome at high GC

`dnm0_window_composition()` maps every non-CpG background training site to its 1 kb tile
and partitions the tiles three ways (the three exactly sum to the total, asserted in code):

| GC | 0.26 | 0.37 | 0.44 | 0.51 | 0.54 | 0.58 | 0.61 | 0.65 | 0.68 |
|---|---|---|---|---|---|---|---|---|---|
| in the analyzed noncoding genome | 0.90 | 0.83 | 0.80 | 0.73 | 0.69 | 0.58 | 0.46 | 0.35 | 0.30 |
| excluded: coding / failed QC | 0.03 | 0.05 | 0.05 | 0.10 | 0.17 | 0.28 | 0.36 | 0.39 | 0.43 |
| excluded: no gnomAD coverage | 0.07 | 0.13 | 0.15 | 0.17 | 0.14 | 0.14 | 0.18 | 0.26 | 0.27 |

By GC 0.68 fewer than a third of the training sites are in the territory Gnocchi is
scored on; 43% are coding. The models are fit on one population and applied to another,
and the two coincide in the GC bulk and come apart exactly where the bias is.

**The fitted curve sits between the two populations at every high-GC bin** (1.267 vs 1.036
analyzed / 1.702 whole-genome at GC 0.61; 1.551 vs 1.288 / 2.188 at GC 0.68). That is the
signature of a model that has partly learned the training population's steeper GC
dependence and carried it into windows where the real dependence is much flatter. Note
this is consistent with the mechanism and quantitatively suggestive, but it is not proof
of it — an equally good fit to the training population would land in the same place for
other reasons.

**What this does and does not resolve.** It closes candidate (c) from the previous
session's list — dnm0 GC-sampling non-uniformity is real (measured, up to 2.0-fold within
a context) but demonstrably not what changes the curve. It reframes candidate (a):
the relevant extrapolation is not primarily in feature *range* but in *population* —
coding and uncovered sequence at high GC. Candidate (b), collinearity with
`CpG_island`/`met_sperm`, is untouched and remains testable by refitting non-CpG contexts
without `GC_content`.

**Caveat on the reliability diagram.** `dnm_probability_non_cpg.pdf` measures a LEVEL
error in P(DNM), and levels cancel in `r = sigma(b0+b.z)/sigma(b0)`. It is a diagnostic of
the fit, not a measurement of Gnocchi's bias; the ~0.07 baseline reflects the 10:1
case-control design, not the genome-wide DNM rate. Both caveats are in the panel's
docstring. This is also why the earlier pooled "non-CpG under-predicts by 26% at GC 0.66"
number was retired from the causal narrative — it is the whole-genome population, which
the ladder above now shows is the wrong one for judging r.

## The intervention: retraining on the scored population removes the bias (2026-08-06)

The population mismatch above is not just concomitant — it is causal, and correcting it
fixes both the adjustment and the score. Two commands, ~6 min and ~2 min:

```
.venv/bin/python fig3/refit_restricted.py                     # the intervention
.venv/bin/python fig3/refit_restricted.py -control_random -tag sizematched
.venv/bin/python fig3/compare_restricted.py \
  -control "reimpl_full:dnm_training_size/output/expected_counts_by_context_methyl_genome_1kb.dnm_refit_full.txt" \
  -control "sizematched:fig3/output/expected_counts_by_context_methyl_genome_1kb.sizematched.txt"
```

**What changed, and only what changed.** `dnm_model.restrict_to_analyzed_windows()` maps
each training site to its 1 kb tile and drops sites outside the analyzed set. Retained:
**292,646 / 410,542 DNMs (71.3%)** and **3,300,888 / 4,104,878 background sites (80.4%)**.
Everything downstream is the identical pipeline (Bonferroni selection → standardize →
IncrementalPCA → L1 logit per context → genome-wide apply). 32/32 contexts fit, none
defaulting to r=1. `GC_content` clears selection in 17/32 contexts, down from 23/32.

### Result 1: the adjustment stops over-adjusting

`r_non` fitted, before and after, against what the DNMs support (`restricted_refit.pdf`
panel B):

| GC | 0.26 | 0.37 | 0.44 | 0.51 | 0.54 | 0.58 | 0.61 | 0.65 | 0.68 |
|---|---|---|---|---|---|---|---|---|---|
| fitted, published | 0.952 | 0.980 | 1.010 | 1.060 | 1.103 | 1.166 | 1.267 | 1.408 | 1.550 |
| **fitted, retrained** | 1.018 | 1.009 | 0.988 | 0.973 | 0.975 | 0.982 | 1.000 | 1.032 | 1.079 |
| observed DNMs | 1.016 | 1.005 | 0.991 | 0.982 | 1.023 | 0.995 | 1.036 | 1.122 | 1.287 |
| over-adjustment, published | 0.94 | 0.98 | 1.02 | 1.08 | 1.08 | 1.17 | **1.22** | **1.25** | 1.20 |
| **over-adjustment, retrained** | 1.00 | 1.00 | 1.00 | 0.99 | 0.95 | 0.99 | **0.97** | **0.92** | 0.84 |

The published fit climbs monotonically to 1.55; the retrained one stays within 8% of 1
across the whole range. Over-adjustment goes from 1.22–1.25 to 0.92–0.97 at GC 0.61–0.65.
Above GC 0.64 the retrained fit now *under*-adjusts (0.84 at GC 0.68), on 203 DNMs.

### Result 2: and Gnocchi's own bias drops below the context-only model's

Mean |mean rank − 0.5| across GC bins, all five curves on one window population
(1,840,165 windows after joint z filtering):

| curve | bias |
|---|---|
| context-only (`r ≡ 1`) | 0.130 |
| **Gnocchi as published** | **0.221** |
| reimplementation, full training set (control) | 0.221 |
| size-matched random subsample (control) | 0.217 |
| **Gnocchi, retrained on the scored population** | **0.079** |

Per bin, published Gnocchi's rank runs 0.29 → 0.88 across GC; the retrained one runs
0.36 → 0.54 → 0.39. It is not merely closer to 0.5 than published Gnocchi (2.8x), it is
**better than the context-only model** (0.079 vs 0.130) — the retrained adjustment
repairs the context-only model's own droop at both GC extremes, which is what a correct
`r` is supposed to do.

### The two controls, which are what make this a result rather than an anecdote

1. **Is it the reimplementation?** No. The full-scale refit through the same code lands
   at 0.221, indistinguishable from published Gnocchi's 0.221. "Before" and "after"
   differ by the intervention, not by whose code produced them.
2. **Is it just less data?** No. Drawing the *same number* of training sites
   (292,646 / 3,300,888) uniformly at random from the whole genome gives 0.217 —
   essentially unchanged from published. This is the control the result stands on: the
   improvement comes from *which* sites, not *how many*. It is also consistent with the
   earlier training-set-size experiment, where shrinking the training set moved Gnocchi
   *toward* the context-only model (never past it), whereas this goes past it.

### Size-matching: where it applies, and where it originally did not

The restricted training set is smaller than the original (292,646 vs 410,542 DNMs), so
every original-vs-scored comparison is confounded with sample size unless the
size-matched control is carried alongside. As first built (2026-08-06 morning) **only
panel A's summary was size-controlled, and even there the control was printed, not
plotted**; panel B and the P(DNM) pairs figure were not controlled at all. Both were
subsequently fixed, and the control is now IN each figure:

| figure | size-matched control |
|---|---|
| `restricted_refit.pdf` panel A | printed in the summary (0.217 vs published 0.221); deliberately not plotted, since a 4th curve indistinguishable from published adds clutter |
| `restricted_refit.pdf` panel B | **plotted** — dashed violet, lies on top of the published curve |
| `dnm_probability_pairs{,_normalized}.pdf` | **plotted** — violet pair, lies on top of the original pair |
| `training_representativeness.pdf` | N/A — no model is fit there; the ladder's rungs are rate estimates over deliberately different populations, and the 37.6% population effect dwarfs the Poisson error on any rung |

The control's verdict is the same in all three places, which is why the result holds:

- panel A: size-matched 0.217 vs published 0.221 vs restricted **0.079**
- panel B: over-adjustment size-matched 1.221 vs published 1.223 vs restricted **0.965**
  at GC 0.61 — the size-matched and published curves agree to ~0.1% at every bin
- pairs: the size-matched fitted and empirical curves overlay the original pair; the
  scored pair is clearly separated

Sizes for the record, non-CpG sites / DNMs: original 4,355,429 / 330,559 (12.2 background
per DNM); scored 3,490,733 / 241,449 (13.5); size-matched 3,471,825 / 235,307 (13.8). The
size-matched set is drawn from the whole genome with `random_state=0`, matching
`refit_restricted.py -control_random` exactly, so the same sites are used in the P(DNM)
figure and in the r figure.

### DNM rate in coding vs noncoding vs uncallable sequence — and where the GC signal really comes from (2026-08-06)

`training_representativeness.dnm_rate_by_stratum()` splits every training site by where it
sits relative to the analyzed window population. Raw and context-matched (weights = the
noncoding stratum's trinucleotide composition):

| stratum | DNMs | sites | P(DNM) raw | P(DNM) context-matched |
|---|---|---|---|---|
| coding (`coding_prop > 0`) | 23,106 | 278,304 | 0.0830 | **0.0763** |
| noncoding (the scored set) | 292,646 | 3,593,534 | 0.0814 | 0.0814 |
| no gnomAD coverage | 94,790 | 643,582 | 0.1473 | **0.1339** |

**Coding raw is HIGHER, context-matched it is 6.3% LOWER.** The raw inversion is
composition: coding is CpG-enriched (CpG share 4.5-7.9% vs 2.9% noncoding) and CpG sites
mutate ~7x faster. Splitting by CpG status:

| stratum | CpG share | P(DNM) non-CpG | P(DNM) CpG |
|---|---|---|---|
| noncoding | 0.029 | 0.0692 | 0.499 |
| coding, `coding_prop < 0.5` | 0.045 | 0.0658 | 0.433 |
| coding, `coding_prop >= 0.5` | 0.079 | 0.0730 | 0.389 |
| no coverage | 0.051 | 0.1192 | 0.670 |

So the coding deficit is mostly a CpG effect (0.50 -> 0.39, a 22% drop), which is the
same methylation story as everywhere else: coding exons — especially first exons — overlap
hypomethylated CpG islands, and unmethylated CpGs mutate ~3x less. The non-CpG coding
deficit is small (0.90-0.98x noncoding, roughly flat in GC).

**Selection cannot be the explanation, and it is the natural wrong guess.** DNMs are by
definition not yet exposed to selection, so a coding deficit in DNM rate must be
mutational, not selective — replication timing (genic regions replicate early, and early
replication carries lower mutation rate), chromatin, and the CpG-island methylation effect
above.

**The GC signal comes from uncallable sequence, not from coding.** Non-CpG P(DNM) per GC
bin, and each stratum's ratio to noncoding:

| GC | 0.26 | 0.33 | 0.40 | 0.47 | 0.51 | 0.54 | 0.58 | 0.61 | 0.645 |
|---|---|---|---|---|---|---|---|---|---|
| noncoding | .0672 | .0687 | .0685 | .0699 | .0704 | .0738 | .0736 | .0779 | .0842 |
| coding | .0494 | .0622 | .0643 | .0682 | .0659 | .0692 | .0732 | .0698 | .0803 |
| no coverage | .1322 | .1104 | .1068 | .1154 | .1294 | .1749 | .2567 | .3159 | .3178 |
| coding / noncoding | 0.74 | 0.91 | 0.94 | 0.98 | 0.94 | 0.94 | 1.00 | 0.90 | 0.95 |
| **no coverage / noncoding** | 1.97 | 1.61 | 1.56 | 1.65 | 1.84 | 2.37 | **3.49** | **4.06** | 3.78 |

The noncoding and coding curves are both nearly flat (1.2x across the whole GC range) and
nearly equal. The no-coverage curve is not: 1.56x the noncoding rate in the GC bulk rising
to **4.06x** at GC 0.61. **Essentially all of the original training set's steep GC
dependence is contributed by sequence gnomAD cannot call** — and that is also where trio
DNM calling is least reliable (low mappability, repeats, segmental duplications), so a
substantial part of the excess is plausibly false-positive DNM calls rather than real
mutation.

This is the mechanism behind every earlier result in this section: it is why the original
training set's empirical P(DNM) rises 2.4x and turns over while the scored set's rises
only 1.57x; why the fitted model learns a GC coefficient it should not apply; and why
restricting the training set removes the bias. It also explains the retention asymmetry
noted earlier (71.3% of DNMs retained vs 80.4% of background): 23.1% of DNMs but only
13.4% of background sites fall in the no-coverage stratum.

Caveat: "no coverage" is defined here as absent from `constraint_z_genome_1kb.annot.txt`,
which conflates gnomAD depth filtering with any other upstream exclusion. The split
between genuinely elevated mutation rate and DNM-calling artifact in that stratum is not
separable from these files.

### Is the empirical CpG DNM probability independent of GC? No — but r_CpG = 1 is still about right (2026-08-06)

`r_eff_decomposition.pdf` shows the CpG contexts' FITTED r is flat at ~1.00 across GC,
which is true by construction (`FT_CORR_MET` strips their GC-correlated features). It is
tempting to read that as licence to ignore CpG contexts when hunting for bias in the DNM
model. That inference does not follow, and checking it properly requires care.

**1. The raw empirical CpG DNM rate is strongly GC-dependent.** Per opportunity it falls
2.7x from GC 0.26 to 0.645 — high-GC CpGs are hypomethylated CpG islands.

**2. Step 1 absorbs most of it**, because `fitted_po` is keyed by methylation level.
Switching the denominator to the methylation-aware E1 shrinks the fall from 2.7x to 1.7x.
(This is exactly the case where the opportunities-vs-E1 denominator choice MATTERS. For a
non-CpG context they are equivalent — one methylation level, constant cancels — which is
why `empirical_r.py` defaults to opportunities there. Do not carry that default to CpG.)

**3. Almost all of the remaining 1.7x is a `fitted_po` SATURATION artifact, not a real
r != 1.** For CpG C>T, across methyl 0 -> 15 the `fitted_po` ratio is only 3.0-4.3x while
the pre-saturation `mu` ratio is 9.7-15.2x. So E1 is a compressed proxy for DNM
opportunity, compressed most where methylation is highest — and CpG methylation
composition swings hard with GC (mean level 6.5 in the bulk falling to 1.5 by GC 0.645).
`empirical_r.cpg_saturation_artifact()` quantifies the resulting spurious decline.
Dividing it out:

| GC | 0.26 | 0.33 | 0.40 | 0.47 | 0.54 | 0.58 | 0.61 | 0.645 |
|---|---|---|---|---|---|---|---|---|
| CpG DNMs in bin | 267 | 4903 | 10191 | 7355 | 2532 | 1218 | 536 | 173 |
| mean methylation level | 6.12 | 6.30 | 6.46 | 6.07 | 5.30 | 4.28 | 3.18 | 1.47 |
| measured D/E1 (rel. to GC 0.40) | 1.109 | 1.059 | 1.000 | 0.987 | 0.941 | 0.909 | 0.874 | 0.654 |
| predicted saturation artifact | 0.976 | 0.990 | 1.000 | 0.975 | 0.929 | 0.863 | 0.790 | 0.628 |
| **corrected r_true, CpG** | 1.136 | 1.070 | 1.000 | 1.012 | 1.013 | 1.053 | 1.107 | 1.041 |
| relative Poisson SE | 0.061 | 0.014 | 0.010 | 0.012 | 0.020 | 0.029 | 0.043 | 0.076 |
| fitted r_CpG | 0.994 | 0.996 | 0.999 | 1.001 | 1.007 | 1.003 | 0.996 | 0.996 |

**Corrected, the true CpG adjustment is flat within about +/-11% with no monotone trend**,
against a fitted r_CpG of 0.994-1.007. So `r_CpG = 1` is approximately RIGHT, and the
apparent decline is the artifact. Notably the correction reverses the sign of the naive
reading: uncorrected, CpG models look like they over-adjust by 1.5x at GC 0.645; corrected,
they are within a few percent.

**Why r_CpG = 1 is right is a property of the two-step design, not of good CpG
modelling.** Step 1 already conditions on methylation, so the low rate at hypomethylated
CpG-island sites is ALREADY in E1 and there is genuinely nothing left for r to adjust.
The CpG models being methylation-blind and GC-blind does not hurt, because the thing they
would need to model has been removed upstream.

**Net effect on the narrative.** The decision to focus on non-CpG contexts now has two
independent supports rather than one: CpG contexts apply no GC-dependent adjustment
(decomposition identity, `r_eff_decomposition.pdf`), AND they should not (this section).
The earlier phrasing "the bias is wholly non-CpG" was about the first only; it is now
justified as a statement about error too.

**Limits.** The artifact weights are dnm0 site counts standing in for per-(context,
methylation) `possible` counts, which no flat file in the bucket provides — the
per-context expected export is already summed over methylation, so exact weights need the
Hail table. `mu` is itself a downsampled-1000-genome estimate rescaled to a fixed total.
The top bin rests on 173 CpG DNMs and 400 background sites. Treat the correction as an
order-of-magnitude control, and the conclusion as "no evidence of a material CpG error",
not as a precise measurement of zero.

### How `dnm_probability_pairs_normalized.pdf` relates to r, and why the size-matched pair sits low

Two questions that come up on sight of that figure, both answered by measurement.

**Is the normalized figure just r marginalized?** Nearly, and the gap decomposes cleanly.
Since `p_hat_t(x) = r_t(x) * sigma(b_t0)`, a POOLED mean over contexts is
`sum_t pi_t(g) * sigma(b_t0) * E[r_t | g, t]` -- the per-context adjustments are mixed
with weights carrying each context's own baseline level, and `pi_t(g)` shifts with GC, so
the pooled curve carries composition on top of r. Normalizing each context first
(`-per_context`) removes that term. Measured, original set, fitted:

| GC | 0.36 | 0.43 | 0.47 | 0.55 | 0.58 | 0.62 | 0.66 |
|---|---|---|---|---|---|---|---|
| pooled (what `-normalize` plots) | 0.954 | 1.010 | 1.048 | 1.179 | 1.299 | 1.475 | 1.718 |
| per-context normalized | 0.977 | 1.002 | 1.020 | 1.094 | 1.174 | 1.304 | 1.483 |
| genome-level `r_non` (published fit) | 0.981 | 1.010 | 1.030 | 1.104 | 1.166 | 1.267 | 1.408 |

So the per-context version tracks `r_non` to within 1% through GC 0.35-0.47 and stays ~5%
above it at GC 0.66. Roughly three-quarters of the pooled curve's tail excess is
composition; the residual quarter is the two differences no normalization can remove:
this evaluates the model at SITE-level feature vectors (r(w) uses window-aggregated ones),
and it marginalizes the non-GC features over the TRAINING-SITE distribution (r_non
marginalizes over genomic windows weighted by E1). The second of those is the subject of
the figure, not a defect in it. `-per_context` is implemented and cached separately
(`dnm_probability_pairs_percontext_normalized.pdf`).

**Why is the size-matched pair uniformly ~11% below the original in the RAW figure?**
Pure class balance, with no GC content. The control matches the restricted set's COUNTS
(292,646 / 3,300,888), but the restriction removed 28.7% of DNMs and only 19.6% of
background -- DNMs are over-represented in the excluded coding/uncovered territory -- so
matching those counts down-samples positives harder than negatives and moves the
case:control ratio from 1:10.0 to 1:11.3. Predicted offset
`(292646/410542)/(3300888/4104878) = 0.8865`; observed overall **0.893**, and per GC bin
0.87-0.91 with no trend. The fitted curve shifts by the same factor (0.885-0.894), since
the logistic intercept absorbs class balance. **It cancels in r** (the level is
`sigma(b0)`, which is r's denominator), which is why panel B's size-matched curve sits
exactly on the published one despite the offset here. Sampling both classes at one common
rate would have removed the offset; matching absolute counts is the more faithful control
for "the same amount of data the restricted fit had", and neither choice touches r.

### Result 3: what the restriction does to P(DNM) itself

`fig3/plot_dnm_probability_pairs.py` (add `-normalize` for the shape-only version) puts
all four reliability curves on one axis, non-CpG only:
`dnm_probability_pairs{,_normalized}.pdf`.

| GC | 0.32 | 0.39 | 0.47 | 0.51 | 0.55 | 0.58 | 0.62 | 0.66 | 0.70 | 0.74 |
|---|---|---|---|---|---|---|---|---|---|---|
| fitted, original | .0707 | .0745 | .0795 | .0837 | .0895 | .0986 | .1120 | .1304 | .1452 | .1560 |
| **empirical, original** | .0722 | .0736 | .0782 | .0816 | .0910 | .1125 | .1448 | **.1769** | .1461 | .1213 |
| fitted, scored | .0688 | .0692 | .0691 | .0698 | .0711 | .0729 | .0759 | .0824 | .0894 | — |
| **empirical, scored** | .0688 | .0683 | .0700 | .0707 | .0727 | .0778 | .0818 | **.0962** | .1040 | — |

Two things happen at once:

1. **The empirical GC dependence itself shrinks and becomes monotone.** On the original
   set P(DNM) rises 2.4x and then *collapses* (0.177 at GC 0.66 -> 0.121 at 0.74). On the
   scored set it rises smoothly by 1.57x with no turnover. The wild non-monotonic tail is
   a property of the out-of-population sites, not of noncoding mutation rates.
2. **And the logistic regression can then actually fit it.** On the original set the fit
   under-predicts by 26% at GC 0.66 and over-predicts by 29% at 0.74 -- it is a smooth
   monotone surface chasing a curve that turns over. On the scored set fitted and
   empirical track within a few percent everywhere, with a mild shrinkage in the top bins.

Levels are NOT comparable between the two pairs: the class balance differs (12.2 vs 13.5
non-CpG background sites per DNM), which shifts P(DNM) for reasons unrelated to GC. Only
within-pair comparisons are exact, because there the fitted and empirical curves come
from the very same sites. Use `-normalize` to compare shape.

### On the residual under-adjustment of the retrained model

In `restricted_refit.pdf` panel B the retrained fit sits slightly *below* the observed
DNMs above GC 0.6. In SE units of the observed curve, the retrained fit's deviations are
-0.7, -2.6, +1.8, +1.1, +3.7, +0.7, +1.0, +1.4, +1.9 sigma across GC 0.33-0.68, versus
+8.3, +1.2, +6.4, +8.7, +6.1, +8.4, +6.7, +4.5, +2.4 for the published fit. So the
high-GC shortfall is **1.0-1.9 sigma, i.e. not significant**; the only >2 sigma misses
are in the bulk (GC 0.40 at -2.6, GC 0.54 at +3.7) and are 1-5% in size. The direction is
nonetheless systematic and has three plausible contributors, all pulling r toward 1: L1
regularization shrinking coefficients, a site-level fit evaluated at window-aggregated
feature values, and a smooth monotone logistic surface that cannot track a sharp tail
rise. Name the mechanism if asked; do not report the magnitude as a finding.

Note also that panel B's observed curve is the **scored subset** of DNMs (292,646 --
exactly what `restrict_to_analyzed_windows` retains), not the original 410,542, because
`load_dnm_counts_by_context_bin` applies the same window filter. That is the right target
for judging r on the scored population, and it is also why panel B is in-sample.

### Caveats to state in the paper

- **Panel B is in-sample.** The retrained model is fit on DNMs in the analyzed windows and
  panel B scores it against DNMs in those same windows, so some of that agreement is
  guaranteed. Panel A is the out-of-sample confirmation: it is measured on gnomAD
  polymorphism counts, which the DNM model never sees. A held-out split of the DNMs would
  make panel B airtight and has not been run.
- **`GENEHANCER_BED` is still unavailable**, so "the scored population" here is noncoding +
  `pass_qc` + autosome/PAR without the enhancer exclusion — the same definition used
  everywhere else in this analysis, so it is at least internally consistent.
- The 0.84 under-adjustment in the top GC bin rests on 203 DNMs; do not quote it as a
  finding.
- This is not a proposal for a corrected Gnocchi. It is a demonstration that the bias is
  attributable to the training/scoring population mismatch. A real Gnocchi 2.0 would have
  to decide what the scored population is *before* fitting, and the choice is not
  obvious (chen_formula.tex §9 argues for a different route entirely).

## `fig5/` — the manuscript figure, and the intended endpoint of all the above (2026-08-07)

**`fig3/` was deleted on 2026-08-07.** It is preserved in full at commit `070fee9`
("Preserve fig3/ in history before removing it"), which exists solely so the deletion is
reversible — several of its modules were untracked. Three capabilities were retired with
it and live only in that commit: `empirical_r.py` (fitted `r_non` against the adjustment
the observed DNMs support — over-adjustment 1.22–1.26 — plus `callable_fraction_by_bin`
and `cpg_saturation_artifact`), `training_representativeness.py`'s population ladder
(denominator 2.4% / aggregation 4.3% / **population 37.6%**), and
`compare_restricted.build_panel_b`. Everything else was carried into `fig5/`, including
`fig5/diagnostics.py`, which reproduces the two measurements panels B and C state in
prose. Sections above that describe `fig3/` outputs are kept as the historical record of
how the result was reached; the code they name is at `070fee9`.


Everything from "What Gnocchi applies" onwards was exploratory. `fig5/` is the
consolidation: five panels, one argument, each written to `fig5/output/fig5{A..E}.pdf` as
a standalone vector file for assembly in Illustrator. `fig5/fig5.ipynb` carries the LaTeX
derivation of every plotted quantity and calls the code; `fig5/README.md` is the
operational detail. **Prefer `fig5/` over `fig3/` for anything manuscript-facing.**

| Panel | Claim | Was |
|---|---|---|
| A | The bias is introduced by the regional adjustment, not inherited from the context-only model | `fig3/output/fig3.pdf` panel A |
| B | That adjustment's GC dependence is wholly non-CpG — and `r_CpG ≈ 1` is *correct*, since methylation already carries it in step 1 | `r_eff_decomposition.pdf` |
| C | The training set is not the scored population: at high GC it is mostly coding or uncovered | `training_representativeness.pdf` panel B |
| D | Restricting to the scored population flattens the empirical DNM rate, and the fit can then track it | `dnm_probability_pairs_normalized.pdf` |
| E | Refitting `r` there removes Gnocchi's bias | `restricted_refit.pdf` panel A |

Verified end to end (`nbconvert --execute`), reproducing every established number:
mean |rank − 0.5| over plotted bins is 0.093 (context-only) / 0.212 (published) / **0.046
(retrained)**, with both controls at 0.212 and 0.210; `r_non` spans 0.951–1.785 while
`r_CpG` spans 0.984–1.004 and the counterfactual is flat within 0.6%; the analyzed
fraction of background training sites falls 0.85 → 0.29 across the plotted GC range;
empirical P(DNM) spans 2.45× on the original training set vs 1.57× on the scored one.

Note the panel-A/E numbers are the **`min_n`-filtered** ones (bins with ≥100 windows,
i.e. what the panels draw). The unfiltered values quoted earlier in this document —
0.130 / 0.221 / 0.079 — are the same statistic over all 20 bins. Both are correct; quote
the filtered ones, since they match the figure.

Four things worth knowing before touching it:

- **`fig5/refit.py -population {full,scored,sizematched}`** replaces
  `fig3/refit_restricted.py`. It also writes the per-site predictions panel D needs, so
  one script produces everything. `full` is needed even though it changes nothing: the
  published pipeline never exported its per-context `r`, so panel B uses the
  reimplementation's — validated per GC bin against the published `E2/E1` (max 1.0e-4,
  median 3.9e-6, printed on every run).
- **The refit outputs live in the repo-root `refits/`** — one copy of each table (~12 GB
  total, gitignored), named `{table}.{population}.txt` with population ∈ {full, scored,
  sizematched}. `fig5/`, `fig3/` and `dnm_training_size/` all read it **directly**; there
  are no symlinks anywhere. Two writers still produce files nobody reads, and say so at
  the top of their docstrings: `fig3/refit_restricted.py` (fully superseded by
  `fig5/refit.py`) and `run_dnm_training_experiment.py -tag full` (its subsampled runs are
  still live).
- **`GENEHANCER_BED` lives in `fig5/config.py`, not in the notebook**, because it defines
  the analyzed window set and that set is used by two separate processes: `refit.py
  -population scored` decides what the model is *fit* on, the notebook decides what the
  panels are *evaluated* on. Disagreement means training on one population and scoring on
  another — the very defect the figure documents. Both read `config.py`, so they cannot
  disagree within a run; and because refits persist across edits, `refit.py` stamps the
  value into `refits/provenance.json` and `data.refit_path` refuses a refit built under a
  different setting (verified: the guard fires, and `full` is correctly exempt since it
  never builds the window table). The existing refits are stamped `null`, which is what
  `fig3/refit_restricted.py` actually used.
- **Shared with `dnm_training_size/`, so deliberately not in `fig5/`**:
  `gnocchi_bias/windows.py` and `gnocchi_bias/dnm_model.py`. Nothing in `fig5/` imports
  from `fig3/`.

Not carried over, deliberately: `panel_calibration_ratio` (measures a level error that
cancels in `r`), `panel_r_non_vs_empirical` and the 1 Mb ground-truth test (the
comparison against truth — still the right supplementary panel if a reviewer asks
whether the adjustment is *wrong* rather than merely *present*), and the four-rung
population ladder (its conclusion is now stated in panel C's markdown instead of drawn).

### Where to pick up

1. **Optional hardening of the result above**: a held-out DNM split for panel B; and
   candidate (b) from the previous session, collinearity with `CpG_island`/`met_sperm`,
   which is now much less interesting since the population intervention already explains
   the effect.
2. **Fig. 3 panel B**: use `panel_r_non_vs_empirical` (comparison against truth, full GC
   range) with `panel_r_eff_decomposition` as a supplementary panel, or stack both.
   `fig3.ipynb` still wires up the superseded `panel_calibration_ratio`. The
   `restricted_refit.pdf` pair above is a strong candidate for a *new* figure rather than
   a panel of Fig. 3.
3. Still unavailable: `DEPLETION_RANK_BED` (panel A third curve) and `GENEHANCER_BED`.
