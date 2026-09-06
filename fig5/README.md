# Fig. 5 — Gnocchi's GC bias comes from its regional adjustment, fit on the wrong population

`fig5.ipynb` builds the figure and writes each panel to `output/fig5{A..F}.pdf` as a
standalone vector file for assembly in Illustrator, plus two supporting figures.

`output/supp_fig7.pdf` — **Supporting Figure 7** in the manuscript, whose four panels are
cited there as 7A-7D (A alone on the left, B-D stacked on the right: A's abscissa is
methylation level, not the GC content the other three share).

`output/supp_fig8.pdf` — **Supporting Figure 8**, **two panels**, which ask what panel E's
intervention buys or costs in *discovery*: McHale et al.'s Fig. 4A/B for published Gnocchi
against the retrained one, on their GeneHancer enhancer-overlap truth set. It is this
figure's own pipeline with one filter dropped (`keep_enhancer_windows=True`), so it is a
statement about panel E rather than about a different window set. Unlike every other panel
it does **not** build without `NEUTRAL_WINDOWS_BED` — the enhancer flag in that file *is*
the truth set, GeneHancer is licensed and not derivable from the public bucket, and
classifying against some other annotation would be a different experiment wearing this
figure's name. The cells check and skip.

**There is no Supporting Figure 9.** It existed for a few hours on 2026-09-04 and was
merged back into 8 the same day, once its calling-rate panel was promoted to Fig. 5F. Its
orphaned `output/supp_fig9.neutral.*` were removed after the fact; nothing writes them.

**Fig. 5F is where the bias appears in usable units, and Supporting Figure 8 is nearly
blind to it.** F — the fraction of windows clearing Gnocchi ≥ 4 per GC bin — uses **no
labels at all**, so it rests on neither GeneHancer nor the laxness of an enhancer proxy,
and it sits in the main figure beside the other label-free panels, built from panel E's own
table and bins (`data.calling_rate_by_gc`, `panels.panel_calling_rate`). Its y axis reads that
fraction as its complement — the percentile at which each score's cutoff falls in the bin's
own Gnocchi distribution — because that is the form the claim states itself in: a score
meeting its own promise would put a fixed z at a fixed percentile everywhere, and published
Gnocchi's z = 4 runs from the 99.9th percentile of AT-rich sequence to roughly the 58th of
GC-rich. The axis stays logarithmic in the calling rate underneath and is inverted so
percentiles increase upward; `percentile_axis=False` restores the original calling-rate
axis. Since 2026-09-05 the panel also draws that null rather than implying it: a dashed
horizontal line at the matched genome-wide calling rate (`calling_rate_by_gc`'s third
return value, passed as `matched_rate`; `matched_rate_line=False` drops it). Precisely, it sits at
`k = |{w : z_s(w) ≥ t_s}| / |W|` over the whole window population — the fraction of *all*
windows clearing the cutoff, one number for both scores since the matching gives the
retrained score the quantile of its own `z` attaining published's `k`. So the line and the
curves are one quantity over two populations: each cutoff's percentile **genome-wide**
against its percentile **within each GC bin**. The panel is worded that way — a y axis
neutral between the two readings (`Cutoff's percentile in the Gnocchi score distribution`),
curve entries naming the cutoff, and the line's entry naming the population
(`Genome-wide percentile common to both cutoffs`). Qualifying the curve entries as well
overruns the axes width at this type size, so the per-bin half rests on the contrast with
*genome-wide* and on the caption, which states it outright. It is
computed over every window including the bins the drawing floor removes — not a mean of the
plotted points — and it is the null strictly rather than loosely, since bin rates average to
`k` and percentile is affine in the rate, so a curve flat across GC can only be flat on this
line. Read with the vertical mean-GC line it makes the panel's arithmetic
visible: published meets it around GC 0.43 while the genome averages 0.393, because a
near-exponential calling rate has a window-weighted mean well above its value at a typical
window. Supporting Figure 8 is the two-panel discovery analysis that does need a truth set.

**8A — auPRC/`r` against GC for both scores**, threshold-free, `data.pr_curves` +
`data.pr_curve_deltas` through `panels.panel_aupr_by_gc`. A GC-dependent bias is very
nearly a common shift on every window in a narrow bin, positives and negatives alike, so it
cancels from a within-bin *ranking* statistic: that is why A's two curves nearly coincide,
and it means the steep decline of auPRC with GC *survives debiasing* (published
1.518 → 1.199 across the bins, retrained 1.554 → 1.298). What remains is signal-to-noise,
which is what McHale et al. conjectured in their text.

**8B — lift per GC bin with the calling rate matched *within* each bin**, which is what
separates ranking from threshold placement: `data.threshold_metrics(match_within_bin=True)`
and `data.lift_deltas(match_within_bin=True)` through
`panels.panel_threshold_metric(..., "lift", ...)`. Higher in all five bins and
significantly so in four: +33.3, +4.0, +4.2, +10.8, +2.2 per cent.

*Be precise about what B matches.* Every score calls the same fraction of every bin, and
that fraction is the **global** rate published attains at `z ≥ 4` — 1.002% of all drawn
windows. It is **not** `z ≥ 4` applied bin by bin, and **not** published's own per-bin
calling rate (which runs 0.17% → 13.97%). So B's published threshold in a bin is that bin's
98.998th percentile of `z`, equal to 4.0 only by coincidence; `data._bin_thresholds` is
where this happens. One consequence is worth stating because it decides what B can show:
with the calling rate common, `lift = recall / calling rate` makes recall a constant
rescaling of lift within a bin, and precision likewise since the base rate is shared. So a
recall panel would be B with a second y axis and the same five per-bin gains, which is why
there is no 8C: it was drawn for a few hours on 2026-09-05 on the argument that lift is the
statistician's unit and recall the analyst's, and cut because that translation is a
sentence rather than a set of axes. It now sits in the caption and in the notebook markdown
as the identity `recall = lift × k` with the five translated numbers (1.81 → 2.42%,
1.53 → 1.59, 1.29 → 1.34, 1.13 → 1.25, 1.13 → 1.15%), and `fig5.ipynb` verifies
`recall == lift * k` numerically rather than asserting it.

Both panels carry the retrained curve's **paired** bootstrap interval relative to
published. Those bars are on one curve deliberately — independent intervals would describe
the uncertainty of each *level* when the question is about the *gap*, and would be wider
than the gap's own interval, since the two scores share almost all of their sampling
variability. A bar that excludes the published marker is a real difference. The comparison
is paired (both scores on the same resampled rows, so window-sampling variability cancels),
unbalanced (the balancing is only needed for cross-bin level comparisons and costs four
fifths of the positives), and its top bin is merged to (0.55, 0.80] because their file is
nearly empty above GC 0.60.

*B is a diagnostic in a second sense.* Forcing published to call 1% of GC-rich sequence
describes a score nobody uses — Fig. 5F's whole point is that it calls 14% there. B says
what the score *contains*; F says what happens when it is *used*.

*Lift is capped at `1/r`*, a ceiling falling from 12.0 to 1.6 across these bins, so compare
the two **scores** within a bin rather than ranking bins against each other.
`data.threshold_metrics` reports ceiling-free skill and `LR+` per bin for anything
cross-bin, alongside precision and recall; none is drawn, because each tells Fig. 5F's
story in another projection. The retired panel functions — `panel_pr_curves`,
`panel_aupr_delta`, `panel_lift_vs_recall` — are in the gitignored `fig5/panels_extra.py`,
last tracked at `582c09d`.

**Fig. 5F and Supporting Fig. 8 compare the two scores at a matched calling rate**, not at
a common `z`. Retraining shifts the whole `z` distribution: at `z ≥ 4` published calls
~1.0% of windows and the retrained score ~0.13%, eight times fewer, so a common cutoff
would credit the retrained score for being strict (higher precision) and penalise it for
the same reason (lower recall). Published is held at 4 and the other takes the quantile
calling the same fraction; each legend carries its own threshold. Fig. 5F's swing is a
within-score ratio and is unaffected either way.

**"Lax" is McHale et al.'s own word and the axis these names are organized on**: a truth
set says which windows count as constrained; the lax one is GeneHancer overlap (big, but
not every enhancer window is under selection) and their **stringent** one is noncoding
windows regulating essential genes, their Fig. 4C/D. Constants belonging to a truth set
carry its name (`LAX_GC_BINS`, `LAX_MIN_BIN_WINDOWS`); those that do not, do not
(`PR_SCORES`, `TRUTH_TARGET`).

**The stringent set was considered and deliberately NOT built** (2026-09-04) — do not pick
it up as pending work. `data.pr_curves` still takes `truth_set` and still accepts only
`"lax"`; that seam stays, and `data.py`'s section header still holds the spec if the
decision is ever revisited. Four reasons, in order of weight: the central result, Fig. 5F,
uses **no truth set at all**, so swapping truth sets cannot move it; 4,933 windows against
1,003,037 is underpowered for a published-vs-retrained gap that is +1.5% pooled on the lax
set; it does not address the objection we actually have, which is that **GC content alone
beats published Gnocchi** on the lax set (lift 2.15 against 1.77), and stringent positives
are GC-rich essential-gene enhancers against AT-poor non-enhancer negatives, so plausibly
*more* GC-separated; and it is not cheap — a third hand-supplied file, an interval-overlap
join onto Chen's 1 kb grid, and new bootstrap builders that are not `pr_curves`.

**What to do instead, at the same cost: a GC-matched control.** Sample negatives from the
lax set to match the positives' GC distribution; GC-only lift is then 1 by construction and
any score's lift above 1 is discrimination that cannot be GC in disguise. Buildable from
data already in hand. And **say the limitation rather than omitting it** — a reviewer will
notice the secondary analyses rest on the set McHale et al. themselves call lax; the strong
form is a caption sentence carrying the power argument above, not silence.

Run the notebook top to bottom.

**Panel C is two rows** sharing a GC axis and built from one table: the composition of
the training sites (how much of the training set is outside the scored population --
*covariate shift*), and the LOG of each other stratum's DNM rate relative to the scored
population's (whether the part of the training set lying outside it has a *different DNM
rate* -- *concept shift*; the log because `se_log` is by construction the SE of that
quantity, so the bars drawn are a plain +/- se on a linear axis). The upper row alone shows only an absence. Both rows count **both training classes**, DNMs and
background sites: the fit minimizes its loss over the mixture, so the mixture is the
training distribution being compared against the scored one. Its bottom band is the
scored population itself, defined by MEMBERSHIP in the analyzed window table (`data.py`,
`_STRATA`) rather than by re-deriving that table's filters -- so it follows
`NEUTRAL_WINDOWS_BED` automatically, which a re-derivation did not. The bands above it
name the reason a site is outside: *QC-pass coding*, *QC-pass putatively nonneutral
noncoding* (the `other_noncoding` stratum -- empty and undrawn unless
`NEUTRAL_WINDOWS_BED` is set) and *QC-fail*. That middle band is the one to read when asking whether the figure survives on
McHale et al.'s window set: it is the territory given up in narrowing 1,843,559 windows
to their 693,270, and if its DNM rate matches the scored population's, that narrowing
costs sample size and nothing else. **Measured: it does** -- 0.94-1.03x across the
plotted range, flat, so the figure carries over between the two window sets. Its label mirrors the bottom band's own parenthetical,
*QC-pass putatively neutral noncoding*, so the two read as one partition of that category;
*putatively* is load-bearing on both sides, since being outside a set McHale et al. call
putatively neutral is not itself evidence of selection, and whether these windows differ
at all is what the lower row measures.
Only the QC-pass ones are split by coding
status, because `coding_prop` comes from the constraint table and a QC-fail window has no
row in it; measured separately, that band is 6.9% coding-overlapping against the QC-pass
windows' 7.1%.

**Supporting Figure 7** backs panel B's claim that `R_CpG ~ 1` is *correct*: the
methylation effect step 1 absorbs (3.0-4.3x within one trinucleotide, against 9.7-15.2x
pre-saturation), the CpG-island character of high-GC CpGs (2.5% hypomethylated in the
GC bulk rising to 92% in the top GC bin), and the resulting DNM-rate collapse
(0.532 -> 0.283, a 1.9x fall). Its bin floor is 100 sites, not the main figure's 500:
the two highest GC bins hold 932 and 1,434 sites and they ARE the claim, so they are drawn
with error bars rather than dropped. **A fourth panel** carries `Pi`, the CpG share of a
bin's step-1 expected counts (0.038 -> 0.264), which is why that claim matters rather
than merely holds: it is the weight in panel B's identity, so a GC trend in `R_CpG`
would have reached the applied multiplier scaled by up to 0.26 rather than erased. It is
binned over *windows* (`binned_b`, floor 100 windows) where B and C above it are binned
over *sites*, so the two do not end in the same GC bin (0.65 against 0.73); all four
panels share this figure's wider 0.2-0.8 axis. (Wider 1,843,559-window run: 90-100%
hypomethylated above GC 0.70, 0.53 -> 0.195 and a 2.7x fall, top bins of 356 and 169
sites, `Pi` 0.025 -> 0.426, and a D that ran past panel B's 0.2-0.73 range.)

```
fig5.ipynb          the figure: LaTeX derivation of each plotted quantity, then the panels
config.py           the two hand-supplied inputs, and the refit provenance stamp
data.py             one builder per plotted quantity, each cached as parquet in output/
panels.py           the panels as ax-accepting functions (no figure, no file I/O) --
                    the five, plus both supporting figures'
resave_ai.py        relink fig5.ai's panel PDFs, save it, re-export fig5.png -- via Illustrator
refit.py            the intervention and its two controls (must run before the notebook)
depletion_rank.py   loader for the Halldorsson depletion-rank window set (panel A, third curve)
preflight.py        checks the two hand-supplied files' schemas before the expensive run
RUNBOOK.md          the ordered procedure for rebuilding with both files set (HPC)
window_set_sensitivity.py  does the answer change on 693,270 windows? stand-in subsets
output/             panel PDFs, the supporting figure, and this figure's own caches
../refits/          the refit tables, shared with dnm_training_size/
```

Shared with `dnm_training_size/`, so deliberately outside this directory:
`gnocchi_bias/windows.py` (the window table, z-scores, ranks, GC binning) and
`gnocchi_bias/dnm_model.py` (the DNM training set and the per-context refit pipeline).

## After a rebuild: refresh the Illustrator assembly

`fig5.ai` and `fig5.png` are both tracked here, so the repo is the source of truth for the
assembled figure -- which means a rebuild that changes a panel leaves both stale until
Illustrator reloads the link, the document is saved, and the PNG is exported from it. That
is what `resave_ai.py` does:

```
.venv/bin/python fig5/resave_ai.py -dry_run   # what is stale, touching nothing
.venv/bin/python fig5/resave_ai.py            # relink, save, re-export, report
.venv/bin/python fig5/resave_ai.py -no_png    # ... leaving fig5.png alone
```

It drives Illustrator over `osascript ... do javascript`, since an .ai stores a path and
a cached preview per link and neither can be regenerated from outside the app. It finds
the document among the open ones before opening a copy, relinks only the links whose
content differs from what the .ai was saved against, and closes the document again only
if it opened it. macOS asks for
Automation permission the first time.

The two kinds of staleness are checked separately, because the second outlives the first:
a panel whose content the `.ai` does not hold needs relinking, an `.ai` newer than the
`.png` needs exporting. So a save you made by hand in Illustrator still gets its PNG, with nothing to
relink. The export is 300 dpi, artboard-clipped, transparent -- hardcoded in the script to
reproduce the settings the committed PNG was made with, not to redefine them. Illustrator's
scripted export writes no resolution metadata where its dialog does, so the script stamps
the `pHYs` chunk back in itself; without it the file declares no dpi and anything placing
it by physical size lays it out 4x too large.

Two things it cannot do for you. **Relinking preserves the frame, not the aspect ratio** --
panels are saved with `bbox_inches="tight"`, so one whose labels changed can come back
slightly stretched; the script prints every link it touched, so check those. And it
**saves whatever state the document is in**, since a document whose links just refreshed
is dirty in exactly the way one being edited is. `git checkout fig5/fig5.ai fig5/fig5.png`
undoes a save you did not want -- but re-save from Illustrator afterwards, or the open
document and the file on disk will disagree.

The staleness check is **content, not mtime** (since 2026-08-17). `fig5.ai.links.json`,
tracked beside the `.ai`, records what each panel PDF hashed to when the `.ai` was last
saved, and `resave_ai` compares against that — so an mtime that moved for reasons
unrelated to the artwork (`git checkout` of a panel, a stash pop, a rebase) no longer
sends it relinking, and no longer dirties `fig5.ai` on screen for nothing. Without the
manifest it falls back to mtime, which is what it did before.

Upstream of that, **panels are only written when their bytes change**. `save()` renders
to a buffer and compares (`resave_ai.save_panel`), and PDFs are written with
`CreationDate` suppressed — the one source of run-to-run nondeterminism in matplotlib's
PDF output, so an unchanged panel now renders to identical bytes. Re-running the notebook
without changing anything therefore touches no file at all: no `.ai` staleness, no
`/CreationDate`-only diffs to keep out of a commit, and no `touch -r` dance afterwards.

Between them, an asterisk on `fig5.ai`'s tab in Illustrator now means a panel genuinely
changed.

A scripted save also rewrites Illustrator's private data more compactly than an
interactive one: expect the file to roughly halve the first time. Verified lossless --
the artwork renders byte-identically, fonts stay embedded, `AIPrivateData` survives.

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
| `NEUTRAL_WINDOWS_BED` | `41586_2023_6045_MOESM4_ESM/Supplementary_Data_2.features.constraint_scores.bed` from the same path — McHale et al.'s window file | The analyzed set is this repo's 1,843,559 noncoding + `pass_qc` + autosome/PAR windows rather than their 693,270 putatively neutral ones |

**Set `NEUTRAL_WINDOWS_BED` and the whole figure recomputes on their windows** — the
file is read, filtered to `window overlaps enhancer == False`, and inner-joined on
`element_id`, which is how the enhancer exclusion (GeneHancer is licensed) and their
interval exclusions (assembly gaps, ENCODE exclude regions, low coverage) arrive without
being re-derived. Run it both ways: the two sets differ 2.66x, and a conclusion that
holds on only one of them is a conclusion about the window definition.

**The two sets' outputs land beside each other, not on top.** `config.WINDOW_SET_SUFFIX`
is `""` for the default set and `.neutral` when `NEUTRAL_WINDOWS_BED` is set, and it is
carried by everything whose *content* depends on the window set: the refit tables
(`…scored.neutral.txt`), their `provenance.json` entries, and the panel PDFs
(`fig5A.neutral.pdf`, `supp_fig7.neutral.pdf`). The default set's names are unchanged, so
nothing on disk is renamed and `fig5.ai`'s links keep resolving. The parquet caches in
`output/` need no suffix — they already carry a fingerprint of the GC edges and the window
set. What still costs time is the refits: `scored` and `sizematched` must be rerun for the
second set, ~6 min each, and `full` need not be (it never builds the window table).

**`window_set_sensitivity.py` asks in advance whether the narrowing will change the
answer**, by rerunning panel A's statistic on same-sized stand-in subsets (random, and
GC-tilted in the direction the enhancer exclusion pulls). It found the per-bin curves
near superimposable and step2/step1 at 2.64-2.68x across all arms -- so neither sample
size nor a GC-tilted removal moves the conclusion. Read its docstring for what it cannot
test (removal correlated with constraint) before treating it as settled.

`NEUTRAL_WINDOWS_BED` lives in a module, not the notebook, because it defines the analyzed
window set — which `refit.py` uses to decide what the model is **fit** on and the notebook
uses to decide what the panels are **evaluated** on. Disagreement would train on one
population and score on another, the defect this figure is about. `config.py`'s docstring
has the full argument, including how `refits/provenance.json` turns a post-refit edit into
a loud error rather than a silent mismatch.

**Run `preflight.py` before either of them is used in anger:**

```
.venv/bin/python fig5/preflight.py
```

It reads both files and checks what the loaders assume — the columns they index by name,
that the enhancer flag is Boolean and not constant, the 1 kb grid and 0-based half-open
coordinates that make `chrom-start-end` an `element_id`, the `chr` prefix, uniqueness, the
693,270 count, and for the depletion-rank file that it loads at all and that GC came out a
0-1 fraction. Seconds, no `published/` needed, non-zero exit on anything that would yield a
wrong figure rather than an error. It cannot check the depletion rank's **orientation**:
the panel ranks within that set, so any monotone transform gives the same curve and only
the direction matters — `depletion_rank.py` assumes low DR means more constrained and
takes `1 - DR`. A mirrored curve about y = 0.5 is that assumption failing.

**The neutral-windows file also carries a depletion-rank column**
(`depletion_rank_constraint_score_complement`) -- depletion rank on Chen et al.'s 1 kb
windows, already complemented. Panel A does not use it: it reads `DEPLETION_RANK_BED` and
ranks within Halldorsson's own windows, matching how McHale et al.'s notebook keeps the
two files apart, and the caption says so. Reading that column through `depletion_rank.py`
would complement it twice and mirror the curve about y = 0.5, so the loader raises on any
column whose name says `complement` unless `complement=False` is passed explicitly. Using
it properly would mean joining it onto the window table so all three curves share one
population and one set of GC bins -- a real option, not taken.

**Both files get the enhancer filter.** McHale et al.'s
`9.regression/experiment.1.ipynb` filters `window overlaps enhancer == False` on the
depletion-rank file as well as on the window file, then plots `1 - depletion_rank` (their
`depletion_rank_constraint_score_complement`). `load_depletion_rank_windows` does both,
and **raises** if the enhancer column is absent rather than quietly ranking over every
window -- which would put a population difference into the curve while the Gnocchi curves
are ranked over enhancer-excluded windows. `exclude_enhancer_windows=False` is the
deliberate opt-out.

`depletion_rank.py` was exercised against synthetic input (column resolution, GC
unit detection, the `1 - DR` complement, the enhancer filter and its absence, error
paths) before it was **run against the real file** on the HPC path: 38,632,866 windows,
30,421,618 after the enhancer filter, GC 0.095-0.784, mean |rank - 0.5| = 0.096. Check
its printed summary whenever the path changes.

## How panel A's `r = 1` curve is validated

The context-only curve carries the comparison the whole figure rests on (0.046 against
0.168 on the committed narrowed run; 0.093 against 0.212 on the wider 1,843,559-window
reproduction), and Chen et al. never published anything to check it against directly — their
pipeline computes no step-1 z. So it is validated in three separable pieces, two of them
runnable checks and one an inheritance argument:

| what | how | result |
|---|---|---|
| the **expected counts** really are pre-adjustment (`r ≡ 1`) | `preconditions/verify_expected_r1.py` regenerates the file genome-wide from `expected_counts_per_context_methyl_genome_1kb.txt`, whose provenance is confirmed — it is the literal `hl.export()` at `run_nc_constraint_gnomad_v31_main.py:191–197`, written *before* any r code runs | `possible` exact on all 2,575,299 rows; `expected` within 4.6e-5 relative, explained by two pipeline runs 277 days apart |
| the two curves describe the **same windows** | same script: `possible` in the r≡1 table against `possible` in the published constraint table, which the r-adjustment multiplies `expected` but never touches | equal on all **1,984,900** joined windows, max diff 0 — so the curves differ only in `expected` |
| the **z and rank** computed from them | cannot be checked directly, since no published step-1 z exists. Instead `gnocchi_bias/windows.py` runs the *identical* code path on the step-2 curve, which does have a published counterpart (`check_z_against_published`) | max \|z − z_published\| = **0.0** over 1,843,559 windows |

The third row is the load-bearing one and worth stating plainly to a reviewer: `z_step1`
is self-computed, and what licenses it is that the same `z_expr`, the same joint filter
and the same within-curve ranking reproduce Chen et al.'s own `z` exactly wherever a
published value exists. Both curves are also z-filtered *jointly* and ranked *after* that
filter, so neither is advantaged by its own window set.

## Things to know before quoting numbers

- **Panels A and E are the same statistic on the same windows**, so they read as
  before/after. They differ only in that E's inner join against the retrained expected
  counts drops a handful of windows, and the joint `z` filter then applies to five
  curves rather than two.
- **The depletion-rank curve is a different window set** (Halldorsson windows, a
  different window size). It is ranked within itself and overlaid, never joined on
  `element_id`. Legitimate for a conditional-mean-rank plot — the rank is uniform on
  (0,1) by construction for every curve — but the caption must say so.
- **Panel B is a decomposition identity, not a fit.** `R_eff = Pi*R_CpG + (1-Pi)*R_non`
  holds bin by bin because each bin aggregates ratios of *summed* expected counts, not
  means of per-window ratios.
- **Two claims the caption states as numbers are computed in `data.py` and printed by the
  notebook**, so nothing quoted in the text is unregenerable: the QC-fail stratum's
  non-CpG DNM rate (1.50–1.63x the scored rate through the GC bulk, **3.39x by GC 0.58**,
  while coding/noncoding stays at 0.86–1.00 and flat), and the CpG-island character of
  high-GC CpGs (92% hypomethylated in the top GC bin, DNM rate 1.9x lower than the bulk,
  against a 3.0–4.3x methylation effect that step 1 already absorbs). Wider run: 1.55x and
  4.06x by GC 0.61, coding 0.90–0.99, 90–100% hypomethylated above GC 0.70 and 2.7x lower.
  Both are also plotted —
  panel C's lower row and `output/supp_fig7.pdf`. Migrated from `fig3/` when that
  directory was retired, then from `diagnostics.py` into `data.py` once they stopped
  being prose-only.
- **That stratum is QC failure, not absent sequence** — it was called "no gnomAD coverage"
  here until it was measured. Every one of the 587,902 windows has its QC inputs on file;
  70.9% fail Chen et al.'s ≥80%-PASS rule against 3.3% failing the 25–35× coverage band.
  `preconditions/verify_qc_filter.py` also confirms the filter forwards (all 1,984,900
  scored windows satisfy all three conditions) and records both denominators, since
  **86.6% of background training sites are in QC-pass windows** — the 87.8% quoted for the
  PASS rule is *within* the 13.3% that are not. Panel C's stack counts both classes, so
  its own genome-wide average is 79.9% / 6.1% / 14.0%.
- **Panel D measures a level error**, and levels cancel in `r = sigma(b0+b.z)/sigma(b0)`.
  It diagnoses the fit; panel E is the measurement of the bias. Its y-axis is also not a
  mutation rate — the ~0.07 baseline reflects the 10:1 case-control design.
- **Panel D is in-sample**; panel E is the out-of-sample confirmation, on gnomAD
  polymorphism counts the DNM model never sees. A held-out DNM split for panel D has not
  been run.
- **`output/` holds only figures and this figure's own caches.** The refit tables live in
  the repo-root `refits/` (see above), so there is exactly one copy of each on disk.
