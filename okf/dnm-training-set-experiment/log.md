---
type: Log
title: Change log
description: Chronological record of when this bundle was created/updated and why.
tags: [gnocchi, log]
timestamp: 2026-07-20T00:00:00Z
---

# Log

- **2026-07-20** — Bundle created. Preceded by (in the same conversation):
  finding the DNM training-set files during a bucket scan prompted by
  investigating `expected_counts_by_context_methyl_dnm_1M.txt`; documenting
  them in root `CLAUDE.md`; then this planning pass answering "can we
  retrain the logistic regression on resized DNM data and recompute local
  bias?" and "can `run_nc_constraint_gnomad_v31_main.py` be repurposed?".
  Nothing in [pipeline](pipeline.md) has been implemented or run yet — this
  is a plan, not a result.

- **2026-07-20 (later same day)** — Prompted by a direct question ("how do we
  know [training-data](training-data.md)'s list is exhaustive?"), fully
  listed `misc/` for the first time (previously only spot-checked). Found:
  (a) `misc/generic.py`, `misc/constraint_basics.py`, `misc/nc_constraint_
  utils.py` exist and are exactly what `run_nc_constraint_gnomad_v31_main.py`
  imports — root `CLAUDE.md` had wrongly called these "missing"; corrected
  there, and [missing-code](missing-code.md) here updated to note they were
  checked directly and confirmed to *not* contain the multivariate PCA fit
  either, which strengthens (doesn't weaken) that specific gap claim.
  (b) A separate, unrelated Random Forest DNM-prediction approach
  (`misc/RF_f18_dnm_1M.pkl` + `fig_tables_init/rf_f18_*`), on a 17-feature
  superset panel (`misc/genomic_features17_*`) — documented in root
  `CLAUDE.md` only, not folded into this bundle's pipeline since it's a
  different modeling approach, not a resizing of the same training set.

- **2026-07-20 (later still)** — Ran `list_bucket_files.py -depth 2` across
  the whole bucket (after fixing two bugs in that script: an uncapped
  expansion that hung for minutes on `*.ht/index/` directories, and no
  retry on transient GCS 5xx errors) and grepped the saved output
  (`bucket_listing_depth2.txt`, repo root) for "dnm" to check whether
  [training-data](training-data.md)'s file list was still missing anything.
  One find: `genomic_features/dnm01_10x_ft_logit_regularized_coef_z_3mer_context_flnk_1k-1M.txt`
  is the exact published output of `analyze_individual_feature_effects.py:29`
  run on the full training data — added to [training-data](training-data.md)
  and as a new step 0 in [pipeline](pipeline.md) (validate the fitting code
  reproduces this file before touching training-set size at all). Everything
  else matching "dnm" in the listing was already documented.

- **2026-07-21** — Implemented and ran the full pipeline (regime 1 only:
  shrink both dnm0+dnm1 by the same random rate). New scripts:
  `run_dnm_training_experiment.py` (steps 0-4: validate, subsample, refit
  univariate selection, refit the missing multivariate PCA+logit step
  per [missing-code](missing-code.md), apply genome-wide) and
  `plot_dnm_bias_comparison.py` (steps 5-6: GC-binned rank-bias comparison,
  reusing `compute_gc_bias_step1_vs_step2.py`'s filters/binning/rank
  machinery, generalized to N curves).

  **Step 0 (validate)**: full, unmodified training data reproduces the
  published `dnm01_10x_ft_logit_regularized_coef_z_3mer_context_flnk_1k-1M.txt`
  closely: all 1,664 (context,window,feature) rows comparable (no fit
  succeeded/failed mismatches), max |coef diff| = 2.6e-4, all 1,664 rows
  agree to <1e-3, 360/1,664 agree to <1e-6. Close enough (not bit-identical,
  plausibly solver-tolerance/library-version noise) to trust the
  reimplementation.

  **Additional validation not in the original plan**: also ran `-mode refit
  -subsample_frac 1.0` (i.e. the full, unmodified training set fed through
  the *entire* reimplemented pipeline, including the previously-unpublished
  multivariate PCA+logit step) and compared its genome-wide `expected`
  output directly against the real published Gnocchi
  (`fig_tables/constraint_z_genome_1kb.annot.txt`): Pearson r = 1.0 across
  1,984,900 joined windows, mean |diff| = 9.3e-4, median relative diff =
  4e-6. This is strong evidence the reconstructed multivariate step
  (missing-code.md's one real gap) is a faithful reproduction, not just the
  univariate selection step.

  **Regime 1 results** (subsample both dnm0+dnm1 at a fixed rate, seed 0):
  at 1% (4,105 dnm1 / 41,049 dnm0 rows), only 2/32 contexts had any
  significant feature survive Bonferroni selection and converge to a
  multivariate fit; the other 30 defaulted to r=1 (exactly
  run_nc_constraint_gnomad_v31_main.py line 260's own fallback). At 10%
  (41,054 / 410,488 rows), 19/32 contexts fit. Feeding all three
  (1%, 10%, full-as-sanity-check) plus the existing step-1/published-step-2
  curves into `plot_dnm_bias_comparison.py` (noncoding, pass_qc,
  autosome+PAR windows, n=1,840,181, same filters as
  `compute_gc_bias_step1_vs_step2.py`'s defaults) gives a genome-wide,
  GC-binned (20 fixed-width bins) confirmation of
  [hypothesis](hypothesis.md) claim 1: at every single GC bin across the
  full 0.2-0.73 range, the 1% curve is nearly indistinguishable from the
  step-1 (context-only) curve (e.g. GC bin at 0.71: step1=0.284,
  dnm_1pct=0.283, vs. step2_published=0.867), the 10% curve sits smoothly
  between step-1 and published step-2 at every bin (same GC bin:
  dnm_10pct=0.808), and the full-data sanity curve overlays published
  step-2 almost exactly (dnm_full_sanity=0.867). This is a clean
  monotonic dose-response across 1%->10%->100%, genome-wide, not just in
  the tails and not on a hand-picked example — output plot:
  `dnm_training_set_size_bias.pdf` (repo root).

  Not attempted: regime 3 (densify background-only) — still needs Hail
  access to `context_prepared.ht` per [training-data](training-data.md)'s
  own note, out of scope for this pass.

- **2026-07-21 (later same day)** — Two additions to `run_dnm_training_experiment.py`,
  prompted by wanting to see the actual mechanism behind the bias-curve result above, not
  just its downstream effect: (1) explicit reporting of which `(feature, window)` pairs
  get Bonferroni-selected per context, plus a feature-selection frequency table across
  contexts; (2) a two-panel "training-set GC diagnostic" plot per subsample size
  (`gc_diagnostic.dnm_refit_{tag}.pdf`): top panel is the (possibly subsampled)
  training pool's own GC-content histogram (dnm1 vs dnm0); bottom panel is a pooled
  (all contexts combined, GC content only) univariate logistic fit of mutation status on
  GC content, plotted against binned empirical mutation proportions with binomial
  standard-error bars.

  Re-ran all three regime-1 runs (1%, 10%, full) with the new code (same seed, same
  subsample logic — unchanged) and confirmed byte-for-byte-equivalent core results
  before generating the new outputs: identical selected-feature counts and
  contexts-fit counts to the original run (4 rows/4 contexts/2 fit at 1%; 55 rows/21
  contexts/19 fit at 10%; 239 rows/32 contexts/32 fit at full), identical
  2,575,299-row output tables — confirming the new reporting/plotting code is purely
  additive and didn't touch the core fitting/adjustment logic.

  **Feature-selection frequency, by training-set size** (full per-context listings in
  `output/selected.dnm_refit_*.txt`): contexts with `GC_content` selected in *any*
  window, out of contexts with anything selected at all: **0/4** at 1% subsample (the
  two features that do get picked, `cDNM_maternal_05M` and `met_sperm`, are each
  selected in 2/4 contexts; `GC_content` never reaches Bonferroni significance in any
  context at this size), **7/21** at 10% (`GC_content` is the 5th most-selected feature:
  `cDNM_maternal_05M`=10, `CpG_island`=10, `met_sperm`=9, `recomb_male`=8,
  `GC_content`=7), **23/32** at full scale (`recomb_male`=31, `dist2telo`=28,
  `cDNM_maternal_05M`=28, `CpG_island`=24, `GC_content`=23). This is a second, direct
  line of evidence for the same mechanism the bias-curve plot shows indirectly: as
  training data grows, the model gains enough statistical power to detect (and adjust
  for) a GC-content effect it simply cannot see at 1%. `CpG_island` and `met_sperm` are
  themselves correlated with GC content (this is exactly why
  `run_nc_constraint_gnomad_v31_main.py` line 217's `ft_corr_met` list excludes
  `GC_content`, `SINE`, `met_sperm`, `Nucleosome`, `CpG_island` together for CpG-context
  models), so the model's effective sensitivity to GC content at full scale is arguably
  broader than the direct `GC_content`-selection count alone shows.

  **GC diagnostic plots, reviewed visually** (rendered to PNG via `pdftoppm` and viewed):
  mainly show training-pool *sparsity*, not a dramatically shifting fit — the pooled
  fit's coefficient is nearly flat across all three subsample sizes (z-coef 0.180 at 1%,
  0.166 at 10%, 0.167 at full), since a single-feature logistic fit on tens of thousands
  of points (45,154 rows even at 1%) is already fairly stable. What visibly changes is
  the *range and noise* of the training pool's GC coverage: at 1%, the histogram
  truncates around GC~23-73%, and the binned empirical-proportion points in the tails
  have very wide error bars (one bin has a single point sitting at P=1.0 with a [0.15,
  1.0]-ish error bar); at 10%/full, coverage extends further with tighter error bars,
  though even the full-data plot shows wide error bars below GC~25% simply because that
  region is intrinsically rare genome-wide. **Conclusion**: this pooled, GC-only
  diagnostic is a useful sparsity visualization, but it is *not* a faithful stand-in for
  "the fit becoming sensitive to the tails" — that mechanism is better evidenced by the
  feature-selection-frequency numbers above (which use the real per-context,
  Bonferroni-gated selection logic) and by the actual bias-comparison plot (which uses
  the real per-context multivariate PCA+logit models) than by this simplified
  single-feature pooled fit.

  **Reorganization**: moved `run_dnm_training_experiment.py` and
  `plot_dnm_bias_comparison.py`, plus every output file either had produced (previously
  scattered across `tmp/` and the repo root), into a new dedicated directory,
  `dnm_training_size/` (code) / `dnm_training_size/output/` (results) — not
  re-run, files were moved as-is. Both scripts were updated to split what had been a
  single `-dest_dir` into `-cache_dir` (downloaded bucket files — still resolves to the
  shared repo-root `tmp/` by default, so other scripts' caches, e.g.
  `compute_gc_bias_step1_vs_step2.py`'s, are reused rather than re-downloaded) and
  `-output_dir` (this experiment's own computed outputs — defaults to
  `dnm_training_size/output/`), both resolved via the script's own file location
  so they work regardless of invocation CWD. `plot_dnm_bias_comparison.py`'s
  `import compute_gc_bias_step1_vs_step2 as base` needed an explicit `sys.path` fix
  (repo root added at runtime) since that module now lives one directory up. Repo
  `.gitignore` updated: `dnm_training_size/output/rr_by_context.*` and
  `dnm_training_size/output/expected_counts_by_context_methyl_genome_1kb.dnm_refit_*.txt`
  are excluded (the former reaches ~4 GB at full scale — one row per genome-wide window
  per fitted context, kept only for diagnostic transparency, not referenced by any other
  script; the latter are ~127 MB each, the same size class as other bucket-downloaded
  per-window tables this repo already keeps out of git) — everything else in `output/`
  (coefficients, selected-feature tables, GC diagnostic PDFs, the final bias-comparison
  PDF) stays small (tens of KB to ~130 KB) and is tracked normally, matching how
  `compute_gc_bias_step1_vs_step2.py`'s own small plot outputs are already committed at
  the repo root.

- **2026-07-21 (reliability diagram)** — Prompted by a design discussion about how to
  visualize "the fit becomes sensitive to the tails but doesn't fit them well" more
  directly than the pooled GC-only diagnostic above. Considered and rejected: (1) a
  genome-wide reliability diagram using the r(w) numerator `pred` — technically doable
  (the quantity is already computed in `apply_genome_wide_context()`, just discarded) but
  requires possible-weighted aggregation (a window's `pred` stands in for `possible`-many
  sites) and, more importantly, `pred`'s absolute level is confounded by the fixed 10:1
  dnm0:dnm1 case-control training design — under case-control sampling, logistic
  regression gives consistent slope estimates but a biased intercept, so comparing raw
  `pred` against the true (much rarer) genome-wide DNM rate would mostly reflect that
  sampling-induced offset, not local GC miscalibration. (2) Platt scaling — rejected as a
  fix for local bias specifically, since it's a single global affine recalibration and
  can't repair a bias that varies by GC bin without effectively becoming bin-conditional
  recalibration, which is just what adding `GC_content` as a real feature already
  achieves at large N (the feature-selection-frequency result above).

  **What was implemented instead**: a reliability diagram evaluated entirely on the
  training set itself, sidestepping the case-control absolute-calibration problem —
  since both the model's prediction and the empirical label rate it's compared against
  come from the same fixed case-control-sampled population, the sampling-induced
  intercept bias affects both sides equally and cancels out of the comparison. New
  functions in `run_dnm_training_experiment.py`: `predict_training_set()` (evaluates
  each context's fitted multivariate model on that context's own dnm1/dnm0 training
  sites, using each site's own feature vector — not a window's aggregated values, unlike
  `apply_genome_wide_context()`) and `plot_training_reliability_diagram()` (bins the
  pooled, cross-context predictions by site-level GC content and plots mean fitted
  probability against the mean empirical label rate, with binomial SE). Exposed as a new
  `-mode reliability`, which deliberately never touches the genome-wide features/expected
  files or the duckdb join — only the (already-cached) training-set tables — so it's much
  cheaper than `-mode refit`.

  **Confirmed runnable without re-downloading or re-running anything time-consuming**:
  all four training-set input files (`DNM_decode_psychencode_site_context...txt`,
  `context_prefiltered_nonmutated-dnm_sites10xdnm...txt`, both
  `genomic_features13_dnm{0,1}_..._flnk_1k-1M.txt` files, plus
  `mutation_rate_by_context_methyl.txt` for the context list) were already present in
  the shared `tmp/` cache from earlier runs; the two genome-wide files (1.44 GB features,
  3.3 GB per-context expected counts) and the duckdb join were never touched. A smoke
  test (`-max_contexts 3`) ran in ~2s; the three real runs at 1%/10%/full training-set
  size (`-tag frac0.01_seed0`/`frac0.1_seed0`/`full`, same seed/subsample logic as the
  earlier `-mode refit` runs) completed in well under a minute combined (full scale:
  refit 4,503,034 site predictions across all 32 contexts in 30.4s). Feature-selection
  counts and per-context fit/skip counts at each size matched the earlier `-mode refit`
  runs exactly (2/32, 19/32, 32/32 contexts fit at 1%/10%/full; same GC_content
  selection counts 0/4, 7/21, 23/32) — confirming the refit logic reused here is
  identical, not just similarly-behaving.

  **Result** (`output/training_reliability.dnm_refit_{tag}.pdf`,
  `output/training_reliability_binned.dnm_refit_{tag}.txt`): at 1% subsample, mean
  fitted probability is essentially flat (~0.055–0.065) across the entire GC range —
  the model isn't attempting to fit any GC dependence at all, consistent with
  `GC_content` clearing Bonferroni significance in 0/4 fitted contexts at this size; the
  empirical rate is noisy but consistent with that flat line within error almost
  everywhere. At 10%, both curves track each other closely through the bulk of the GC
  range, with a small, noisy divergence starting in the sparse tails. At full scale, the
  two curves are nearly indistinguishable through the dense bulk (bins with 5×10^4 to
  ~10^6 sites each, e.g. GC≈39%: pred=0.0868 vs empirical=0.0859) but the fitted curve
  visibly *overshoots* the empirical rate in the sparse high-GC tail — GC≈74%:
  pred=0.272 vs empirical=0.155 (n=1,808); GC≈77%: pred=0.291 vs empirical=0.153
  (n=649) — exactly where the training pool thins out. This is a second, independent,
  more direct confirmation of the same "sensitive to the tails, doesn't fit them well"
  mechanism the pooled GC-only diagnostic could only gesture at: here it's the real
  per-context multivariate model, evaluated in-sample, and it visibly overfits/overshoots
  specifically in the low-n tail even at full published training-set size — a plausible
  direct explanation for why Gnocchi's r(w) adjustment, and hence its expected counts,
  run high in high-GC windows genome-wide (the step-2 vs step-1 divergence in
  `dnm_training_set_size_bias.pdf`).

- **2026-08-04** — Repo reorganized into `gnocchi_bias/` (shared library) + `fig3/` +
  `dnm_training_size/` (renamed from `dnm_training_experiment/`), each figure dir with its
  own notebook. Refactor verified behavior-preserving: byte-identical binned output on the
  full 1,843,559-window gc-bias path and byte-identical
  `training_reliability_binned.dnm_refit_full.txt`. Found a pre-existing bug: `-downsample_*`
  is not reproducible across runs at fixed seed (duckdb parallel-join row order feeds
  `polars.sample`); full path is deterministic.

  New Fig. 3 built (`fig3/fig3.ipynb`). Panel A = step-1 vs full Gnocchi rank vs GC
  (+ depletion rank once the Halldorsson BED is supplied; loader written and tested on
  synthetic input only). Panel B went through three designs in one session, driven by
  Peter's questions — see CLAUDE.md for the full evidence trail:
    v1 pooled absolute calibration gap -> v2 CpG-stratified ratio (r_model/r_true on
    training sites) -> both superseded, because **training-set calibration measures a LEVEL
    error, and level errors cancel in r = sigma(b0+bz)/sigma(b0)**.

  Causal chain established, with evidence per link: E2 = E1 x r_eff (identity verified to
  1.3e-15 over 2,575,299 windows; r_eff is just the E1-weighted mean of the pipeline's own
  per-context r) -> rank is highly sensitive to r (uniform f=1.10 moves mean rank 0.500 ->
  0.687) -> r_eff rises 0.954 -> 1.297 with GC -> that quantitatively reproduces the
  step2-step1 rank gap (both cross their null at GC ~0.40, deltas match within 0.03) ->
  and the adjustment is WRONG, not merely present: at 1 Mb, model r rises 0.969 -> 1.150
  across GC 33.5-54% while true residual DNM rate (observed_counts_dnm_1M /
  expected_counts_by_context_methyl_dnm_1M) FALLS 1.015 -> 0.926.

  Two mechanisms ruled out, both cleanly: (a) step-2's methylation-blindness — CpG r is
  flat at 1.00 at every GC, and step 1 already handles methylation via fitted_po; (b) the
  CpG hypomethylation/CpG-island route — real miscalibration (1.55-2.29x over-prediction at
  GC 0.70-0.74) but causally inert, and the counterfactual holding non-CpG r at 1 removes
  the entire GC trend. The bias is a wholly non-CpG phenomenon. Chen et al.'s methylation
  modelling is not implicated.

  **Next session**: explain why non-CpG models inflate r with GC (GC_content selected in
  23/32 contexts, r -> 1.9); pick panel B from the two candidates in CLAUDE.md's
  "Where to pick up tomorrow".

- **2026-08-05** — Built the two adjustment-factor figures (`fig3/make_r_figures.py`,
  backed by new `fig3/r_eff.py` and `fig3/empirical_r.py`). Full numbers in CLAUDE.md,
  "What Gnocchi applies, and whether it is right".

  `r_eff_decomposition.pdf`: r_eff = E2/E1 vs GC, decomposed exactly as
  Pi*r_CpG + (1-Pi)*r_non. Non-CpG runs 0.95 -> 1.78, CpG is flat at 0.99-1.00 at every
  GC, and the counterfactual holding non-CpG r at 1 is flat within 0.6% across the whole
  range despite CpG contexts carrying 43% of expected-count weight at GC 0.75. Uses the
  refit's per-context r (the published one was never uploaded to the bucket), validated
  against the published r_eff = expected_step2/expected_step1: max 1.0e-4 across 20 bins.

  `r_non_vs_empirical.pdf`: the fitted non-CpG r against DNMs/E1 per (context, GC bin),
  per-context rescaled so only GC shape is compared. Fitted climbs 0.95 -> 1.55; observed
  is flat near 1.0 to GC ~0.55 then rises slowly to 1.28. Ratio crosses 1 at GC ~0.40 and
  reaches 1.22-1.26, many SEs from 1 — agreeing in direction and rough size with the
  independent 1 Mb ground-truth test, but over a wider GC range and per-context.

  Checked and cleared: denominator choice cannot shape the curve (non-CpG contexts have a
  single methylation level, so E1 ∝ possible exactly per context); per-context and
  per-window tables agree to 4.4e-6; gnomAD coverage filtering cancels since numerator and
  denominator share the same windows. Live caveat: assumes trio DNM ascertainment is
  GC-uniform within the analyzed windows.

  A second, case-control estimator (p_hat over dnm1/dnm0 training sites) gives the
  opposite answer above GC 0.54 and is confounded — the dnm0 background pool is not
  sampled uniformly in GC within a context (2.0-fold for CCC). Kept in the code as a
  cross-check only; this also retires the earlier pooled f = 0.74 number.

  **Next session**: why non-CpG models inflate r with GC. Leading suspect is now the
  measured dnm0 GC-sampling non-uniformity, which biases the fitted GC coefficient
  directly.

- **2026-08-05 (correction)** — The 2026-08-05 entry above attributed the case-control
  estimator's disagreement to dnm0 GC-sampling non-uniformity. That attribution was wrong
  and is corrected in CLAUDE.md. Tested one variable at a time: swapping the E1
  denominator for real dnm0 control counts changes the observed curve by 1-2% (1.147 vs
  1.119 at GC 0.645), and switching tile GC for site-flank GC barely matters (1.106).
  The whole discrepancy (2.021 at GC 0.645) came from the *population* — the unrestricted
  training set spans the whole genome while combine_non_cpg's E1 weights and per-context
  normalization are computed over the analyzed windows. Restricting the same sites to
  those windows collapses the disagreement.

  Net effect: the primary result is more robust than reported, since two independent
  denominators now agree. Added load_control_counts_by_context_bin /
  empirical_from_control_counts as the properly matched cross-check; marked
  load_training_by_context_bin mis-specified for this use. Also recorded that the 10:1
  dnm0:dnm1 ratio holds only genome-wide, not per context (0.76 ACG to 24.8 GAA).

- **2026-08-05 (normalization corrected)** — Peter pointed out the empirical curve should
  be built the same way r itself is defined: r_c = sigma(b0+bz)/sigma(b0) is already
  "rate here / rate at the average", so the empirical analogue is just bin DNM rate over
  the context's overall DNM rate — no free constant. The previous kappa_c scheme was
  algebraically that same ratio times the model's own mean r_c, which made the level
  depend on the model.

  Now both sides are normalized per context to E1-weighted mean 1 over the analyzed
  windows. Per-context is mandatory (D/E1 is not on a common scale across contexts,
  because fitted_po saturates by different amounts), and both sides must get identical
  treatment or shifting trinucleotide composition leaks between-context level differences
  into the curve as false GC dependence.

  Numerically almost identical — inflation 1.2234 -> 1.2237 at GC 0.61, 1.2552 -> 1.2551
  at 0.645 — so no conclusion changed; the construction is just principled now, and the
  "absolute level depends on the normalization" caveat is retired. The
  normalization-free statement remains the strongest one: the over-adjustment ratio rises
  by a factor of ~1.34 across GC 0.26 -> 0.645.

- **2026-08-05 (denominator + callability)** — Peter questioned why the step-1 expected
  count E1 appeared in both numerator-normalizer and denominator of the empirical rate,
  arguing the denominator should be DNM *opportunities*. Correct, and the two are the same
  thing here: within a non-CpG context there is a single methylation level, so
  E1 = opportunities x const_c, and const_c cancels in the per-context normalization.
  Verified — the two denominators agree to <0.1% at every bin. Code now defaults to
  `opportunities` (= `possible`) because that is what a rate needs; `denominator="e1"`
  kept as a cross-check.

  That framing exposed a real and previously unquantified issue: DNMs are counted anywhere
  in an analyzed window, but `possible` counts only gnomAD-callable positions, and the
  callable fraction is NOT flat in GC — 0.905 at GC 0.30 falling to 0.749 at GC 0.61,
  since short-read coverage drops in GC-rich sequence. Whether this biases the comparison
  depends on the trio call sets' own callability, which isn't in this bucket. Both ends
  computed: over-adjustment at GC 0.61 is 1.22 if DNM callability tracks gnomAD's, 1.44 if
  the DNM set is closer to complete (1.51 at GC 0.645). The finding survives either way and
  the correction only strengthens it, but the magnitude is uncertain across ~[1.22, 1.44] —
  the caption must not quote 1.22 as tight. Added callable_fraction_by_bin() and a
  callable_fraction= argument to empirical_from_dnm_counts.
