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
| `expected_counts_per_context_methyl_genome_1kb.txt` | 3.3 GB (bucket root) | This *is* the exact `hl.export()` at `run_nc_constraint_gnomad_v31_main.py` lines 191–197: `expected_ht = possible_ht.group_by(key=(element_id, context)).aggregate(possible=sum, expected=sum)`, one row per `(element_id, context)` pair — i.e. genome-wide expected counts from sequence context alone, `r ≡ 1`, computed *before* the regional-feature adjustment in lines 209–249. `element_id, context, possible, expected`. |
| `expected_counts_by_context_methyl_genome_1kb.txt` | 107 MB (bucket root) | **The step-1 (context-only) expected-count table, further summed down to one row per `element_id`: `element_id, possible, expected`.** Despite the name, this is *not* produced anywhere in `run_nc_constraint_gnomad_v31_main.py` — the script only ever writes the per-`(element_id, context)` file above; this further `group_by('element_id')` sum must happen in a downstream/publication step not included in this repo (same situation as the missing `generic.py`/`constraint_basics.py`/`nc_constraint_utils.py`). Verified self-consistent by hand: summing the 4 per-context rows for `chr1-10000-11000` in the file above (`possible` 3+3+1+4=11, `expected` 0.31501+0.26256+0.074125+0.15301=0.804705) exactly matches this file's row (`11`, `0.80470500`). Trustworthy to use directly, just can't point to its exact generating code. Use this directly — no need to reconstruct step-1 from `context_prepared.ht` (Option A) or the reference FASTA (Option B). |
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

## The analysis: real-data version of the reviewer's request

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

This is the literal reviewer request, done on Chen et al.'s real output instead of a
simulation — a direct empirical answer, not a toy model.

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

6. **Compare** to Supp Fig 1 of the McHale/Goldberg/Quinlan paper and to the simulation
   results in `/Users/petermchale/rebuttal-simulation/simulate_constraint_bias.py` (which
   showed: context-only has real global bias but small local/tail bias; the ad hoc
   regional correction reduces global bias but can increase local/tail bias). Does the
   real data show the same qualitative pattern?

**No genome-wide "global" bias metric is computed.** McHale et al.'s simulation defines
one (`compute_overall_model_bias()` in the same notebook directory:
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
