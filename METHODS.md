# Methods narrative — the Figure 2A-style rank statistic

The canonical, extractable methods narrative for the rebuttal/revised paper. Split out of
CLAUDE.md so it is read when methods are being written, rather than loaded into every
session. It was written for `compute_gc_bias_step1_vs_step2.py`, **deleted 2026-08-07**
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

**The neutral window set** — the other half of "neutral", and as of 2026-08-17 it
arrives as a **join on McHale et al.'s own window file** rather than as an exclusion
re-derived here. Their Methods say: "Of the noncoding windows, those that don't
significantly overlap Genehancer enhancers (Fishilevich et al. 2017) were labeled
'neutral' ... Noncoding windows that do significantly overlap Genehancer enhancers were
labeled 'constrained'" ("significantly" is never numerically defined). Two separate
things make that unreproducible here: GeneHancer is licensed (confirmed 2026-07-21 —
"GeneHancer data must be obtained from the source database directly ... rather from
UCSC", and UCSC does not serve the file), and their *other* interval exclusions (hg38
assembly gaps, ENCODE exclude regions, low-coverage regions) are not in the public bucket
either.

So the file is the definition -- and as of 2026-08-18 it is the WHOLE definition:
`build_window_table` applies its own noncoding / `pass_qc` / autosome-PAR filters only
when `NEUTRAL_WINDOWS_BED` is None, and skips them when the file is supplied. Filtering
first and joining second returned the intersection of two definitions that need not
agree -- `coding_prop <= 0.0` here, a strict zero, against their "doesn't significantly
overlap merged exons" with the threshold never numerically defined -- so any window in
the gap was in their set and was silently dropped here as coding. What can still remove
one of their windows is not a filter but `load_joined_table`'s three-way inner join: a
window with no row in the constraint table, the step-1 expected table or the features
table has no `expected`/`observed`/GC to be scored with.

`gnocchi_bias/windows.py`'s
`load_mchale_neutral_element_ids()` / `restrict_to_mchale_neutral_windows()` read

```
{CONSTRAINT_TOOLS_DATA}/chen-et-al-2023-published-version/41586_2023_6045_MOESM4_ESM/Supplementary_Data_2.features.constraint_scores.bed
```

— Chen et al.'s published Supplementary Data 2 re-annotated by constraint-tools with
regional features and a boolean `window overlaps enhancer` — filter it to
`window overlaps enhancer == False` (**693,270 rows**), and inner-join on `element_id`.
That is verbatim what `get_unconstrained_noncoding_chen_windows()` does in their
`papers/neutral_models_are_biased/9.regression/experiment.1.ipynb`, so the join reproduces
their Fig. 1 window set exactly rather than approximating it. Tab-separated with a header;
`chrom, start, end` are 0-based half-open, the same convention as `element_id`; the file
also carries `window overlaps merged_exon`, `B`, `GC_content_1000bp` and more.

Set it in `fig5/config.py` as `NEUTRAL_WINDOWS_BED` (path only; `None` skips the
restriction). **Both window sets are meant to be run** — 1,843,559 and 693,270 — since a
result holding on only one is a result about the window definition. Operational costs of
switching: the `scored` and `sizematched` refits must be rerun (~6 min each) and are keyed
by population alone, so one set's refits overwrite the other's; the panel-C and CpG caches
in `fig5/output/` carry a fingerprint of the GC edges and the window set, so those coexist.
Panel C gains a fourth band, `other_noncoding`, counting exactly the territory given up
in the narrowing — the band to read when asking whether the figure's conclusions survive
it. Its four strata are a subdivision of the genome's three only if their set holds no
coding window; that is not enforced, so the join prints how many kept windows have
`coding_prop > 0` (0 means the nesting holds and the `coding` band is exactly QC-pass
coding; anything else means those windows are labelled `scored`, not `coding`).

**Will the narrowing change the answer? Probably not, but it is not yet settled.**
`fig5/window_set_sensitivity.py` reruns panel A's statistic on same-sized stand-in
subsets. Over the 13 GC bins every arm draws: step1 / step2 / scored = 0.067 / 0.177 /
0.034 on the full set, 0.066 / 0.177 / 0.033 on a random 693,270, and 0.067 / 0.180 /
0.033 under a GC-tilted removal that keeps GC > 0.5 windows at 15% against 37% overall.
step2/step1 holds at 2.64-2.68x in all three, and step 2's per-bin curve is near
superimposable. What shrinks under each arm's own binning (0.212 -> 0.207 -> 0.182) is
the 100-window floor dropping high-GC bins, not the bias: step 2's mean rank in the top
drawn bin stays 0.877 / 0.876 / 0.875 as that bin moves from GC 0.75 to 0.65. **So quote
the per-bin curve, or say which bins the average covers.** What no stand-in can test is
removal correlated with *constraint* — enhancer windows are GC-rich AND variation-
depleted, so the real narrowing deletes high-z windows preferentially at high GC. A hard
cut doing exactly that does break the result (2.64x -> 0.96x), but it keeps 1.0% of
GC > 0.5 windows, deletes the (GC > 0.5, z > 2) corner outright, and selects on the
quantity being ranked, distorting all three curves together — a bound, not an estimate.

Untested against the real file (it is not available in this environment): verified instead
against a synthetic stand-in with the same column names and coordinate convention — the
join, its two diagnostics, and the guard that raises when fewer than half the file's
windows match (the signature of a `chr1`-vs-`1` mismatch, which would otherwise look like
a very strict filter). The diagnostics are the `coding_prop > 0` nesting count above and
a shortfall line; since nothing here filters any more, that line reports one thing only —
windows with no row in the joined table, almost all of them QC failures. (It used to
distinguish "filtered here" from "absent from Chen et al.'s table", via a `df_prefilter`
argument that no longer exists.) **Deleted with this change**: the `bedtools coverage`
GeneHancer exclusion (`restrict_to_neutral_genehancer`, its `min_frac_covered` cumulative-
coverage semantics, and the chromosome-naming check), recoverable at `fe51e63`. It never
ran against real GeneHancer data, and a join on their file answers the same question
without the licensed input.

**Window count vs. the paper** (explains the wider GC-content "fringe" visible in this
script's heat maps vs. Figure 2A): measured directly (2026-07-21, full non-downsampled
dataset, default filters — `exclude_sex_chromosomes` + `restrict_to_noncoding` +
`pass_qc`, no neutral-set join): this script's default window set has **1,843,559**
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
1. Enhancer-overlapping windows, which their file excludes and this repo cannot identify
   without it (effect size on the count not separately measured; supplying
   `NEUTRAL_WINDOWS_BED` now measures it directly, as the `other_noncoding` stratum).
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


