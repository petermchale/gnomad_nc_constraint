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
`gsutil -m cp {input_bucket}/logit_pickles/* {output_dir}/tmp/` then loads via
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
generic container. No local `{output_dir}/tmp/` was ever populated and the pipeline
itself was never run for this check.

This verification is captured as a standalone, reproducible script:
`verify_logit_predict_behavior.py`. It downloads a real fitted per-context model from
the public bucket (default context `AAA`, override with `-context`), prints
`type(logit)`, and reproduces the probability-vs-linear-predictor discrepancy above:

```
python verify_logit_predict_behavior.py [-context AAA] [-dest_dir tmp]
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
| `expected_counts_by_context_methyl_genome_1kb.txt` | 107 MB (bucket root) | **The step-1 (context-only) expected-count table, further summed down to one row per `element_id`: `element_id, possible, expected`.** Same `possible`/`expected` definitions as the row above, just summed again over all 32 contexts (so `possible` here matches the meaning of `possible` in `fig_tables/constraint_z_genome_1kb.annot.txt`, which the later `r`-adjustment never touches). Despite the name, this file is *not* produced anywhere in `run_nc_constraint_gnomad_v31_main.py` — the script only ever writes the per-`(element_id, context)` file above; this further `group_by('element_id')` sum must happen in a downstream/publication step. (Earlier text here said this was "the same situation as the missing `generic.py`/`constraint_basics.py`/`nc_constraint_utils.py`" — that was wrong: those three modules are *not* missing, they're at `misc/generic.py`, `misc/constraint_basics.py`, `misc/nc_constraint_utils.py` in the bucket, and are confirmed to be exactly what `run_nc_constraint_gnomad_v31_main.py:23–25` imports [`from generic import *`, etc.] — this local checkout just never fetched them. Checked directly: none of the three contain a `group_by('element_id')` step matching this file either, so the specific aggregation behind *this* file genuinely still isn't shown anywhere available — just don't extend that gap to the utility modules generally, see the "fitting code" section below.) Verified self-consistent by hand: summing the 4 per-context rows for `chr1-10000-11000` in the file above (`possible` 3+3+1+4=11, `expected` 0.31501+0.26256+0.074125+0.15301=0.804705) exactly matches this file's row (`11`, `0.80470500`). Trustworthy to use directly, just can't point to its exact generating code. Use this directly — no need to reconstruct step-1 from `context_prepared.ht` (Option A) or the reference FASTA (Option B). **A Hail-native counterpart, `expected_counts_by_context_methyl_genome_1kb.ht/`, also exists in the bucket** (fetch its `README.txt` and `metadata.json.gz` directly over HTTPS, same as any other bucket object) — its `table_type` schema (`Table{key:[element_id], row:Struct{element_id:String, possible:Int64, expected:Float64}}`) matches the `.txt` file column-for-column, confirming the same 3-column shape from an independent source. Its `metadata.json.gz` gives an exact row count via `sum(components.partition_counts.counts)`: 2,575,299 rows (38,029 partitions) — more precise than inferring row count from the `.txt` file's size. Written with Hail 0.2.62, created 2022/01/17. It does *not* resolve the generating-code gap above: the metadata is a standard Hail `TableSpec` (schema + partition counts only, no lineage/provenance field), so it confirms this was a real materialized Hail table upstream of the `.txt` export, but not which script produced it. |
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
real data — see `verify_comparisons_tables.py`, which fetches
`fig_tables/comparisons.tar.gz` (the source for `efig_utils.py:plt_comparison_roc_gnocchi`,
the function `generate_manuscript_efigures.py -efig 6` calls) and prints each
`comparisons_*.txt` file's real schema, row counts, and sample rows:

```
python verify_comparisons_tables.py [-dest_dir tmp]
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
  loads of the 1.44 GB / 325 MB files.
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
