# fig4_prevalence/ — is McHale et al.'s Fig. 4 free of the normalized-auPRC ceiling?

McHale et al.'s Fig. 4 and Supporting Fig. 5 report performance as **normalized auPRC**,
auPRC/π, where π is the fraction of windows labelled constrained. That sets a random
classifier to 1, but a classifier better than random can score at most 1/π, so normalized
auPRC still depends on π. Two normalized auPRCs are comparable only at a common π.

Both source notebooks downsample the constrained windows so that **GC bins share a common
π**, so every comparison *across GC bins* is sound: the curves within Fig. 4A, each curve in
Fig. 4B, and the two bins behind each bar in Fig. 4D. Two comparisons are *not* at a
common π, because each side is downsampled separately:

1. **Lax against stringent truth set** (Fig. 4C, Supporting Fig. 5E, F). Results reads the
   rise from lax to stringent as "a post-hoc validation of stringency". If the stringent set
   ends at a lower π, part of that rise is the higher ceiling.
2. **Metric against metric** (Fig. 4B, C). Each metric is scored on its own window set
   (Chen, Halldorsson or CDTS windows), so each has its own π. Results says the Chen model
   "has the best average performance".

These notebooks measure π beside every normalized auPRC and add **auROC**, which does not
depend on π. They only report numbers; nothing in the figures changes.

## The notebooks

Both are copies of `quinlan-lab/constraint-tools` notebooks, pinned to a commit:

| notebook | copy of | commit | produces |
|---|---|---|---|
| `main.2.prevalence.ipynb` | `papers/neutral_models_are_biased/7.CDTS/main.2.ipynb` | `6907eec7d0` | Fig. 4A, B |
| `11.compare-lax-with-stringent-truth-set.prevalence.ipynb` | `papers/neutral_models_are_biased/11.compare-lax-with-stringent-truth-set.ipynb` | `4063cdc118` | Fig. 4C, D; Supporting Fig. 5E–H |

Every edit is marked `# PREVALENCE CHECK`, and nothing else differs from the original.
The upstream outputs were cleared, since they belong to the unedited code.

**`main.2.prevalence.ipynb`**
- The `util` import and one image path were relative to the upstream directory. They now
  point at `{CONSTRAINT_TOOLS}/papers/neutral_models_are_biased`, which the notebook's
  second cell already defines.
- `plot_curves_all_bins` records π (`positive_fraction`) per GC bin and over all bins, beside
  the areas it already recorded.
- `line_plot` draws auROC as well as normalized auPRC. The ROC branch was already written;
  it was switched off.
- The wrapper returns its table, kept as `AREAS_GC`.
- Two new cells at the end print, per metric and GC bin: auPRC, normalized auPRC, auROC and
  π; then each metric's rank within each bin under normalized auPRC and under auROC.

**`11.compare-lax-with-stringent-truth-set.prevalence.ipynb`**
- All original cells are unchanged, so the published bars are regenerated too.
- A new section draws each bootstrap sample exactly as the published bars were drawn: the
  same `shuffle_bin_downsample`, the same single lax subsample per metric, and the same
  bin-size check, which drops a whole truth set if any sample fails it.
- Each sample records π (overall and per bin), the ceiling 1/π, normalized auPRC,
  auPRC skill (auPRC − π)/(1 − π), and auROC.
- It runs for GC (Fig. 4C), BGS and gBGC (Supporting Fig. 5E, F).
- It prints mean ± s.d. by metric and truth set, then stringent minus lax on each measure.
- The raw samples are written to `prevalence_check.bootstraps.tsv` beside the notebook,
  for a statistic the notebook does not print (a percentile interval, say). It is
  gitignored: every number quoted is in the notebook's committed output, and a rerun
  reproduces the conclusions, though not digit for digit, since the bootstraps are unseeded.

## Running on the HPC

Both notebooks read the data from the constraint-tools paths they always did
(`/scratch/ucgd/lustre-labs/quinlan/...`), and use the constraint-tools `.venv` kernel:

```
cd fig4_prevalence
$CONSTRAINT_TOOLS/.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 11.compare-lax-with-stringent-truth-set.prevalence.ipynb
$CONSTRAINT_TOOLS/.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 main.2.prevalence.ipynb
```

The first is quick once the lax tables are loaded, because every bootstrap sample is only
4,933 windows. The second is as slow as the original: it scores the full CDTS table.

An executed notebook can run to tens of MB, which GitHub's notebook viewer will not
render. Shrink it before committing:

```
python shrink_notebook.py 11.compare-lax-with-stringent-truth-set.prevalence.ipynb          # report only
python shrink_notebook.py 11.compare-lax-with-stringent-truth-set.prevalence.ipynb --write  # shrink in place
```

It removes stderr streams and the HTML copy of DataFrame previews (the plain-text copy
stays), and re-encodes figures with a 256-colour palette. Every printed table and every
cell source is left byte-identical. Run without `--write`, it reports which cells hold the
bytes.

The upstream bootstraps are unseeded (`random_state=None`), and so are these. The means
will therefore match the published bars to within their error bars, not digit for digit.

## Reading the result

**Lax against stringent.** Read the `auROC` block under "stringent minus lax":
- If Gnocchi and λs are higher on the stringent set by auROC, the "post-hoc validation of
  stringency" stands whatever π did.
- If the rise appears in `auPRCnorm` but not in `auROC`, it is the ceiling, and Results must
  not read it as validation.

Compare `positive_fraction` between the two truth sets first: if they are close, the question
is already settled.

**Metric against metric.** In `main.2`, compare the two rank tables:
- If the metrics rank the same way under `PRCnorm` and `ROC` in each bin (and in `all`), the
  cross-metric reading of Fig. 4B does not rest on π.
- If they reorder, the ordering in Fig. 4B is partly π.

**Sanity checks, which should pass whatever the answer.**
- `positive_fraction_bin_0` equals `positive_fraction_bin_1` (to rounding).
- In `main.2`, `positive_fraction` is constant across a metric's bins.
- The `auPRCnorm` means reproduce the published bars.

**What a synthetic run showed.** On synthetic windows where four metrics are equally good
by construction, auROC agreed across all four (0.713–0.714 over all bins). Normalized auPRC
nonetheless ranked the two scored at the lower π (0.128) at 2.20, against 1.82 for the two
at π = 0.257. That is the size of effect a π difference alone can produce, and what this
check is looking for in the real data.
