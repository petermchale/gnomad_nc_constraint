"""
Generate fig5/fig5.ipynb from a flat list of (kind, text) cells.

    .venv/bin/python fig5/make_fig5_nb.py && \
      .venv/bin/python -m nbconvert --to notebook --execute --inplace fig5/fig5.ipynb

THIS FILE IS THE SOURCE OF THE NOTEBOOK. Editing fig5.ipynb directly works until
someone runs this, which silently reverts the edit -- that has already happened once
(the tmp/ -> published/ rename). Change prose and code here, then regenerate and
re-execute. It lives in the repo rather than a scratch directory for exactly that
reason.
"""
import json
import os

CELLS = []


def md(text):
    CELLS.append(("markdown", text.strip("\n")))


def code(text):
    CELLS.append(("code", text.strip("\n")))


md(r"""
# Figure 5 — Gnocchi's GC bias comes from its regional adjustment, and from fitting that adjustment on the wrong population

Five panels, one argument. Each is written to `output/fig5{A..E}.pdf` as a standalone
vector file for assembly in Illustrator.

| | Claim | Quantity |
|---|---|---|
| **A** | The bias is *introduced by* the regional adjustment, not inherited from the context-only model | mean standardized rank vs GC, for $r\equiv1$ and for full Gnocchi |
| **B** | That adjustment's GC dependence is wholly non-CpG — and $R_{\mathrm{CpG}}\approx1$ is *correct*, because methylation already carries it in step 1 | $R_{\mathrm{eff}}=\sum E_2/\sum E_1$ per GC bin, decomposed by CpG status |
| **C** | The DNM training set is not the scored population: at high GC it is mostly coding, or sequence dropped for failing gnomAD's variant-call QC | composition of the training sites per GC bin |
| **D** | Restricting the training set to the scored population flattens the empirical DNM rate — and the fit then tracks it | fitted and empirical $P(\mathrm{DNM})$, non-CpG, three training populations |
| **E** | Refitting $r$ on the scored population removes the bias in Gnocchi itself | panel A's statistic, with the retrained Gnocchi added |

A and E are the same statistic on the same windows, so they are read as before/after.
B, C and D explain what happens in between.
""")

code(r"""
%matplotlib inline
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

# Arial throughout -- the manuscript's typeface. This notebook is the only thing that
# draws the panels and Supporting Figure 7, so setting it here covers both (panels.py
# deliberately configures nothing). mathtext has to be pointed at Arial separately or
# every $R_{\mathrm{eff}}$ falls back to DejaVu Sans while the plain text does not;
# fonttype 42 embeds TrueType instead of matplotlib's default Type 3, so the type stays
# live and editable in the Illustrator assembly rather than arriving as outlines.
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "pdf.fonttype": 42,
})

# Works whether the kernel's cwd is fig5/ (Jupyter's default) or the repo root.
_HERE = os.path.abspath("")
if os.path.basename(_HERE) != "fig5" and os.path.isdir(os.path.join(_HERE, "fig5")):
    _HERE = os.path.join(_HERE, "fig5")
_REPO_ROOT = os.path.dirname(_HERE)
for p in (_REPO_ROOT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import config
import data as D
import depletion_rank as DR
import panels
import resave_ai          # save() below writes panels through it; the last cell resaves

print("repo root:", _REPO_ROOT)
""")

md(r"""
## Configuration

**Two inputs are not in this repo** and must be supplied by hand. Both live in
**`fig5/config.py`**, not in the cell below, because `fig5/refit.py` reads them from
there too — see below for why that matters.

`DEPLETION_RANK_BED` — panel A's third curve. From the constraint-tools
`CONSTRAINT_TOOLS_DATA` path used by
`papers/neutral_models_are_biased/9.regression/experiment.1.ipynb`:

```
{CONSTRAINT_TOOLS_DATA}/depletion_rank_scores/41586_2022_4965_MOESM3_ESM.noncoding.enhancer.BGS.gBGC.GC_content.bed
```

Left `None`, panel A builds with two curves instead of three and prints a notice.

`NEUTRAL_WINDOWS_BED` — **which window set the whole figure is computed on.** Left
`None`, that is the 1,843,559 windows this repo builds from the public bucket: noncoding
+ `pass_qc` + autosome/PAR. Set to McHale et al.'s own file, it is their 693,270
putatively neutral windows instead — the set behind their Fig. 1:

```
{CONSTRAINT_TOOLS_DATA}/chen-et-al-2023-published-version/41586_2023_6045_MOESM4_ESM/Supplementary_Data_2.features.constraint_scores.bed
```

filtered to `window overlaps enhancer == False`, exactly as
`get_unconstrained_noncoding_chen_windows` does in their
`9.regression/experiment.1.ipynb`. A join on their file rather than a re-derivation here,
because the two things separating the sets are both unavailable: the enhancer flag comes
from GeneHancer, which is licensed, and their interval exclusions (hg38 assembly gaps,
ENCODE exclude regions, low-coverage regions) are not reproducible from the bucket.

**Run it both ways.** The 2.66x gap between the two sets is large enough that a result
holding on only one of them would be a result about the window definition. The figure is
built to be recomputed by changing this one line — see the caveats at the end for what
that costs.

**This notebook's committed run is the narrowed one.** `NEUTRAL_WINDOWS_BED` is set, so
every panel PDF and PNG in `output/` and every number quoted in the prose below comes
from McHale et al.'s 693,270 windows, GC 0.212–0.716. Three side measurements are the
exception, and each says so where it appears: the $\sqrt{E_1}$ sensitivity check in panel
A, the weighted-vs-unweighted aggregation check in panel B, and the background-class-only
variant of panel C's stack. All three were made on the 1,843,559-window reproduction, and
none of them is a result — they bound how much a modelling choice could matter, and the
narrowed set is a subset of the same windows.

**Why this one is not just a notebook constant.** It defines the analyzed window set,
which is used in two separate processes: `refit.py -population scored` uses it to decide
which training sites survive the restriction (what the model is *fit* on), and this
notebook uses it to decide which windows the panels are *evaluated* on. If those
disagree, the model is trained on one population and scored on another — precisely the
defect this figure is about. Both read it from `fig5/config.py`, so they cannot
disagree within a run; and since refits persist on disk across edits, `refit.py` stamps
the value it used into `refits/provenance.json` and `data.refit_path` refuses a refit
built under a different setting. Change `config.py` and you get a loud error naming the
refit to rerun.

**Prerequisite.** Panels B, D and E read the three refits, which are produced by

```
.venv/bin/python fig5/refit.py -population full
.venv/bin/python fig5/refit.py -population scored
.venv/bin/python fig5/refit.py -population sizematched
```

(~6 min each), into the repo-root `refits/` — one copy of each table, also read
directly by `dnm_training_size/`. `D.refit_path` raises with the exact
command if one is missing or was built under a different `NEUTRAL_WINDOWS_BED`.
""")

code(r"""
CACHE_DIR = D.CACHE_DIR                       # published/, or $GNOCCHI_PUBLISHED_DIR
OUTPUT_DIR = D.OUTPUT_DIR                     # fig5/output
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set these in fig5/config.py, NOT here -- fig5/refit.py reads them from there too.
DEPLETION_RANK_BED = config.DEPLETION_RANK_BED
NEUTRAL_WINDOWS_BED = config.NEUTRAL_WINDOWS_BED
print(f"DEPLETION_RANK_BED  = {DEPLETION_RANK_BED!r}\n"
      f"NEUTRAL_WINDOWS_BED = {NEUTRAL_WINDOWS_BED!r}")

N_BINS = D.N_BINS           # 20
XRANGE = D.XRANGE           # (0.2, 0.73), visually matched to Fig. 2A
MIN_N_WINDOWS = 100         # drop GC bins holding fewer windows than this
MIN_N_SITES = 500           # ... or fewer training sites, in panels C and D
MIN_N_CPG = 100             # ... in the supporting CpG figure, whose top bins ARE the
                            # claim (n = 932 and 1,434); error bars carry the uncertainty
FIGSIZE = (7.0, 4.6)


def save(fig, letter):
    # One vector PDF per panel for the Illustrator assembly, a PNG alongside to read --
    # via resave_ai.save_panel, which writes a file ONLY if its bytes changed. So
    # re-running this notebook without changing a panel touches nothing, and fig5.ai does
    # not go stale (or dirty on screen) over a rebuild that changed no artwork.
    # config.WINDOW_SET_SUFFIX, so the narrowed set's panels land BESIDE the default
    # set's rather than overwriting them -- and fig5.ai, which links the unsuffixed
    # names, keeps resolving.
    name = f"fig5{letter}{config.WINDOW_SET_SUFFIX}"
    written = resave_ai.save_panel(fig, os.path.join(OUTPUT_DIR, name))
    print(f"wrote {', '.join(os.path.basename(p) for p in written)}" if written
          else f"{name}: unchanged, left alone")
""")

md(r"""
## Where the model comes from, and where bias can enter

*Migrated from `chen_formula/chen_formula.tex` §§1–5 (Peter McHale), which this notebook
replaces.*

**Expected SNVs under neutrality.** In a selectively neutral window, the expected number
of polymorphic sites — a site being polymorphic if an ALT allele is seen in one or more
members of the cohort — is

$$\sum_i p_{c_i}(x_i),$$

where $i$ indexes single-nucleotide sites, $x_i$ is a vector of regional features (GC
content, replication timing, …) at site $i$, $c_i$ is its sequence and methylation
context, and $p_c(x)$ is the probability that a site of context $c$ with features $x$ is
polymorphic — which of course also depends on cohort size.

**Uniformity within a window.** Assume the feature vectors of all sites in a window $w$
are approximately equal, $x_i = x_w$. Then

$$\sum_i p_{c_i}(x_w) \;=\; \sum_c n_c^{(w)}\, p_c(x_w),$$

grouping the left-hand terms by context, with $n_c^{(w)}$ the number of sites in $w$ of
context $c$.

**The naive way to fit $p_c$.** The natural approach is a parametric model

$$p_c(x) = \sigma(\beta_c \cdot x)$$

with $\sigma$ the logistic function, trained per context $c$ on a genome-wide set of
sites that are monomorphic or polymorphic in the cohort.

**Why that fails: the training set is contaminated by selection, and the contamination
correlates with $x$.** Those training sites ought to be selectively neutral, but we do
not know how to identify neutral sites in the noncoding genome, where functional
annotation is sparse and imprecise. Worse, the sites under selection *correlate with the
features* — promoters are under selection and are GC-rich — so $p_c(x)$ comes out
inaccurate precisely in the tails of the $x$ distribution.

**Chen et al.'s two-stage fix.** De novo mutations have by definition not been exposed to
selection, so they report inherent mutability. They cannot be used directly — we want a
polymorphism probability, which scales with cohort size, not a DNM rate, which does not —
but they can be brought in as a *ratio*. Factorize:

$$p_c(x) = r_c(x)\, p_c(\bar x), \qquad r_c(x) = \frac{p_c(x)}{p_c(\bar x)},$$

with $\bar x$ the average of $x$ over the windows of interest, so $r_c$ is a
feature-dependent multiplicative adjustment to a feature-independent baseline.

*Step 1* estimates $p_c(\bar x)$ by maximum likelihood over the noncoding genome,

$$p_c(\bar x) \;\approx\; \frac{N_c^{(p)}}{N_c^{(m)} + N_c^{(p)}},$$

with $N_c^{(m)}, N_c^{(p)}$ the numbers of monomorphic and polymorphic sites of context
$c$. With that many training examples, most of them near $\bar x$, this is accurate —
subject to low levels of contamination by selection.

*Step 2* assumes the polymorphism probability is approximately proportional to the DNM
probability $\hat p_c(x)$ in a training set of DNMs and matched background sites, so that

$$r_c(x) \;\approx\; \frac{\hat p_c(x)}{\hat p_c(\bar x)},
\qquad \hat p_c(x) = \sigma(\hat\beta_c \cdot x).$$

Two consequences follow immediately, and both matter later. $\hat p_c$ is sensitive to the
class balance of the training set (the DNM-to-background ratio), but $r_c$, being a ratio
of two such probabilities, **is not** — this is why panel D's levels are not comparable
across populations while panel B's $R$ is. And although the DNM training set can be
trusted to be neutral even in the tails, it is *far* smaller than the set behind
$p_c(\bar x)$, so $r_c$ is the noisier of the two factors by construction.

**Two places the implementation departs from this derivation**, both established in
`preconditions/` and both load-bearing below:

1. $r$ is fit **per trinucleotide context only**, not per $c = (\text{trinucleotide},
   \text{methylation})$. Methylation enters in step 1 alone, through $p_c(\bar x)$. That
   is why $R_{\mathrm{CpG}} \approx 1$ in panel B is *correct* rather than a failure — the
   large methylation effect has already been applied upstream.
2. The published Methods state $r$ as a ratio of **logits**; the code computes the ratio
   of **probabilities** written above. The code is what produced the published scores
   (`preconditions/verify_logit_predict_behavior.py`).
""")

md(r"""
## Notation

Index 1 kb windows by $w$. Chen et al.'s model is a product of two factors. Write
$c=(t,m)$ for a *class* of possible SNV: trinucleotide context $t$ (32 of them) and
methylation level $m$. Let $n_c(w)$ be the number of gnomAD-callable sites of class $c$
in window $w$, and $p_c$ the fitted per-site mutation probability for that class
(`fitted_po`, keyed by $(t,\mathrm{ref},\mathrm{alt},m)$).

**Step 1 — sequence context and methylation only.**

$$E_1(w) \;=\; \sum_c n_c(w)\, p_c .$$

**Step 2 — the regional adjustment.** For each context $t$ a logistic regression is fit
on de novo mutations, then

$$r_t(w) \;=\; \frac{\sigma\!\big(\beta_{t0} + \boldsymbol\beta_t^{\!\top} \mathbf z_t(w)\big)}{\sigma(\beta_{t0})},
\qquad
E_2(w) \;=\; \sum_c n_c(w)\, p_c\, r_{t(c)}(w),$$

where $\mathbf z_t(w)$ is the standardized, PCA-transformed vector of regional features
at $w$ and $\sigma$ is the logistic function. Two consequences are used throughout:

1. $r$ is a **ratio of predicted probabilities**, so a *level* error in the fitted model
   — one common to numerator and denominator — cancels exactly and never reaches
   Gnocchi. Only feature-driven *variation* survives.
2. $r_t$ does **not** depend on methylation: there is one model per trinucleotide
   context, pooling all 16 methylation levels.

(The paper's Methods state $r=\boldsymbol\beta\cdot\mathbf x(w)\,/\,\boldsymbol\beta\cdot\bar{\mathbf x}$,
a ratio of *logits*. The code computes the ratio of probabilities above, and the code is
what produced the published scores.)

**The score.** With $O(w)$ the observed count of rare variants,

$$\chi^2_M(w)=\frac{\big(O(w)-E_M(w)\big)^2}{E_M(w)},
\qquad
z_M(w) = -\operatorname{sign}\!\big(O(w)-E_M(w)\big)\sqrt{\chi^2_M(w)}$$

for $M\in\{1,2\}$, keeping $|z|\le 10$ — so high $z$ means constrained. Standardizing to
a rank within the analyzed window set,

$$\rho_M(w)=\frac{\operatorname{rank}\big(z_M(w)\big)-\tfrac12}{n},$$

which is uniform on $(0,1)$ by construction. **Panels A and E plot
$\;\overline{\rho}_M(g)=\mathbb E\big[\rho_M(w)\,\big|\,w\in g\big]$** for GC bins $g$.
An unbiased metric sits at $0.5$ in every bin; departure from $0.5$ *is* the bias.
""")

md(r"""
## Panel A — the bias is introduced by the regional adjustment

Both curves are computed on one window population, with the two $z$ statistics filtered
jointly and ranked *after* that filter, so they describe exactly the same windows. The
only difference between them is whether $r$ was applied.

Depletion rank is an independently constructed metric on **Halldorsson's own windows**,
so it is ranked within its own set and overlaid, never joined on `element_id`. That is
legitimate here precisely because $\rho$ is uniform on $(0,1)$ for every curve: what is
compared is how each metric's uniform mass redistributes across GC. Two consequences the
caption must carry: it is binned on **its own GC edges** (its windows span a wider GC
range than the Gnocchi set), and it is drawn **without error bars**, because those
windows overlap — 30.4M of them after the enhancer filter, over a 3.1 Gb genome — so
within-bin windows are not independent and $\mathrm{std}/\sqrt{n}$ would understate the
uncertainty by about $\sqrt{\text{length}/\text{step}}$. The mean curve is unaffected,
and it is the only thing read off this curve.

Summary statistic quoted in the text: $\;\overline{|\overline{\rho}_M(g)-0.5|}\;$ across bins.

**Why this statistic is so sensitive to $r$** — worth recording, because it is what
licenses reading panels B–D as the *cause* of panel A. Suppose a window's adjustment is a
uniform inflation, $E_2 = f E_1$, and write the observed count as $O = E_1(1+\epsilon)$.
Then

$$z_2 - z_1 = \sqrt{E_1}\left[\frac{f-1-\epsilon}{\sqrt{f}} + \epsilon\right]
\;\approx\; \sqrt{E_1}\,(f-1) \qquad (f \to 1,\ \epsilon \to 0).$$

The $\sqrt{E_1}$ prefactor is the point. $E_1$ averages 170 per window (median 181), so
$\sqrt{E_1} \approx 13$, and a mere 10% inflation displaces $z$ by more than one unit —
comparable to the spread of $z$ itself. Measured directly by re-ranking the real
genome-wide $z$ distribution under a uniform $f$, the mean rank moves from 0.500 at
$f = 1$ to 0.705 at $f = 1.10$ and 0.284 at $f = 0.90$. Small multiplicative errors in $r$
are not a second-order concern; they are the whole effect. (Both of those are measured on
the 1,843,559-window reproduction, where the whole $z$ distribution is available offline.
They bound the sensitivity rather than reporting a result, and the narrowed set is a
subset of the same windows.)
""")

code(r"""
df_win = D.window_table(CACHE_DIR, neutral_windows_bed=NEUTRAL_WINDOWS_BED)
edges = D.gc_edges(df_win["GC_content"].to_numpy(), N_BINS)
print(f"analyzed window set: {df_win.height:,} windows, "
      f"GC {df_win['GC_content'].min():.3f}-{df_win['GC_content'].max():.3f}")

df_a, binned_a = D.rank_curves(df_win, min_n=MIN_N_WINDOWS)
gc_mean = float(df_a["GC_content"].mean())
binned_a
""")

code(r"""
binned_dr = None
if DEPLETION_RANK_BED:
    binned_dr = DR.bin_depletion_rank(DR.load_depletion_rank_windows(DEPLETION_RANK_BED),
                                      n_bins=N_BINS)
else:
    print("DEPLETION_RANK_BED is not set -- panel A gets two curves instead of three.")

curves_a = [
    panels.curve_from_binned(binned_a, "step1", "step1", "Gnocchi, context-only"),
    panels.curve_from_binned(binned_a, "step2", "step2", "Gnocchi, as published"),
]
if binned_dr is not None:
    # show_se=False: Halldorsson's windows OVERLAP, so std/sqrt(n) understates the
    # uncertainty and the bar cannot be defended beside two curves of disjoint 1 kb tiles.
    curves_a.append(panels.curve_from_binned(binned_dr, "dr", "dr",
                                             "Depletion rank",
                                             show_se=False))

# DRAW order and LEGEND order are different questions here, so they are stated separately.
# Draw: depletion rank first, i.e. lowest, then the context-only model, then the published
#   score on top -- it is the panel's subject and should not be crossed out where the
#   curves meet.
# Legend: the subject first, the model it is built on top of second, and depletion rank
#   LAST. It is a different metric, on a different window set, binned on its own GC edges
#   -- so it reads as the external comparison rather than as a third Gnocchi curve.
DRAW_ORDER_A = ["dr", "step1", "step2"]
LEGEND_ORDER_A = ["step2", "step1", "dr"]
curves_a.sort(key=lambda c: DRAW_ORDER_A.index(c["key"]))

fig, ax = plt.subplots(figsize=FIGSIZE)
panels.panel_rank_bias(ax, curves_a, gc_mean=gc_mean, xrange=XRANGE, min_n=MIN_N_WINDOWS,
                       legend_order=LEGEND_ORDER_A)
save(fig, "A")
""")

md(r"""
## Panel B — the adjustment Gnocchi applies, decomposed by CpG status

**Per-context expected counts.** Group step 1's sum by trinucleotide context $t$ — the
level $r$ is fit at — writing $c\!\to\!t$ for the classes $c=(t,m)$ sharing context $t$:

$$E_1^{t}(w)=\sum_{c\to t} n_c(w)\,p_c, \qquad E_1(w)=\sum_t E_1^{t}(w).$$

Because $r_t(w)$ carries no methylation index, it comes out of the inner sum, and step 2
is a *reweighting of the same per-context counts*:

$$E_2^{t}(w)=\sum_{c\to t} n_c(w)\,p_c\,r_t(w)=E_1^{t}(w)\,r_t(w),
\qquad E_2(w)=\sum_t E_1^{t}(w)\,r_t(w).$$

**What one window receives.** Dividing,

$$r_{\mathrm{eff}}(w)=\frac{E_2(w)}{E_1(w)}=\sum_t \omega_t(w)\,r_t(w),
\qquad \omega_t(w)=\frac{E_1^{t}(w)}{E_1(w)},\quad \sum_t \omega_t(w)=1,$$

an $E_1$-weighted mean of the per-context $r$. This is why $r_{\mathrm{eff}}$, not any
single $r_t$, is the quantity the score sees: two windows with identical $r_t$ still
receive different adjustments if their context composition differs — and context
composition is precisely what varies with GC.

**What a GC bin receives.** Aggregate a bin $g$ as a **ratio of summed expected counts**:

$$R_{\mathrm{eff}}(g)=\frac{\sum_{w\in g}E_2(w)}{\sum_{w\in g}E_1(w)}
=\sum_{w\in g} W(w)\,r_{\mathrm{eff}}(w),
\qquad W(w)=\frac{E_1(w)}{\sum_{w'\in g}E_1(w')},$$

again an $E_1$-weighted mean, of the window-level $r_{\mathrm{eff}}$ this time. Expected
counts add, so this is the adjustment the bin actually receives, and the weights are not a
separate choice — they fall out of dividing one sum by another. An unweighted mean of
$r_{\mathrm{eff}}(w)$ answers a different question, and would break the CpG decomposition
below, which is an identity only for ratios of sums: bin by bin it reproduces
$R_{\mathrm{eff}}$ to floating point ($\le2\times10^{-16}$), where the unweighted version
misses by up to $4.9\times10^{-3}$ because a window's CpG share and its non-CpG adjustment
are correlated within a bin. The two aggregations agree closely here in any case: Chen et
al.'s QC filter admits only windows with $\ge1{,}000$ possible variants, so $E_1$ spans
just 55.5–345.7, and switching to the unweighted mean would move $R_{\mathrm{eff}}$ by
~0.1% through the GC bulk and 0.79% at worst, in the sparsest low-GC bin. (That
comparison is on the 1,843,559-window reproduction; the narrowed set is a subset of the
same windows, so its $E_1$ span can only be tighter and the two aggregations can only
agree more closely.) **Case carries the level throughout this notebook:** lowercase
$r$ is per context or per window, capital $R$ is the bin-level aggregate, a ratio of
summed expected counts.

**Splitting at CpG.** Let $\mathcal K=\{\mathrm{ACG,CCG,GCG,TCG}\}$ and write
$E_M^{\mathcal K}=\sum_{t\in\mathcal K}E_M^{t}$, $E_M^{\neg\mathcal K}=E_M-E_M^{\mathcal K}$
for $M\in\{1,2\}$. Define each part's own adjustment, and the CpG share of the step-1
counts:

$$R_{\mathrm{CpG}}(g)=\frac{\sum_{w\in g}E_2^{\mathcal K}(w)}{\sum_{w\in g}E_1^{\mathcal K}(w)},
\qquad
R_{\mathrm{non}}(g)=\frac{\sum_{w\in g}E_2^{\neg\mathcal K}(w)}{\sum_{w\in g}E_1^{\neg\mathcal K}(w)},
\qquad
\Pi(g)=\frac{\sum_{w\in g}E_1^{\mathcal K}(w)}{\sum_{w\in g}E_1(w)} .$$

Then, splitting the numerator of $R_{\mathrm{eff}}$ and multiplying each term by 1,

$$R_{\mathrm{eff}}(g)
=\frac{\sum_g E_2^{\mathcal K}+\sum_g E_2^{\neg\mathcal K}}{\sum_g E_1}
=\frac{\sum_g E_1^{\mathcal K}}{\sum_g E_1}\cdot\frac{\sum_g E_2^{\mathcal K}}{\sum_g E_1^{\mathcal K}}
+\frac{\sum_g E_1^{\neg\mathcal K}}{\sum_g E_1}\cdot\frac{\sum_g E_2^{\neg\mathcal K}}{\sum_g E_1^{\neg\mathcal K}}
=\Pi(g)\,R_{\mathrm{CpG}}(g)+\big(1-\Pi(g)\big)\,R_{\mathrm{non}}(g),$$

using only that expected counts add and that $\sum_g E_1^{\neg\mathcal K}/\sum_g E_1=1-\Pi(g)$.
This is an **exact identity**, not a fit, so the panel reads additively.

**Reading the panel.** Its four curves are these symbols, in this order:
$R_{\mathrm{eff}}$ (solid orange diamonds, what Gnocchi applies), $R_{\mathrm{non}}$
(black dashed squares) and $R_{\mathrm{CpG}}$ (black dotted triangles) — the two terms it
decomposes into — and the counterfactual derived next, drawn in $R_{\mathrm{eff}}$'s own
colour and symbol but dashed and hollow, because it *is* $R_{\mathrm{eff}}$ with one term
switched off. The y-axis is $R(g)$ itself, a ratio of summed expected
counts, so 1 means *no adjustment* and the horizontal line at 1 is the null. Since
$R_{\mathrm{eff}}$ is a $\Pi$-weighted average of the other two, it must lie between
them: it tracks $R_{\mathrm{non}}$ closely at low GC, where $\Pi$ is small, and is pulled
down towards $R_{\mathrm{CpG}}\approx1$ as $\Pi$ grows.

**The counterfactual, and what it intervenes on.** The question is how much of
$R_{\mathrm{eff}}$'s rise the CpG contexts could account for on their own. So switch off
the *non-CpG* adjustment — set $r_t\equiv1$ for $t\notin\mathcal K$, which sends
$E_2^{\neg\mathcal K}\!\to\!E_1^{\neg\mathcal K}$ — and change nothing else: the fitted
$r_t$ for $t\in\mathcal K$ stay, the weights $\Pi(g)$ stay. The dashed orange curve is

$$R_{\mathrm{eff}}\big|_{r_t\equiv1,\;t\notin\mathcal K}(g)
=\frac{\sum_g E_2^{\mathcal K}+\sum_g E_1^{\neg\mathcal K}}{\sum_g E_1}
=\Pi(g)\,R_{\mathrm{CpG}}(g)+\big(1-\Pi(g)\big),$$

i.e. **what Gnocchi would apply if it adjusted CpG contexts alone**. It is flat within
0.4% across the whole GC range while $R_{\mathrm{eff}}$ climbs to 1.33, so *none* of the
applied trend survives the removal of the non-CpG term: the GC dependence of what Gnocchi
applies is wholly non-CpG.

Flatness here is a result, not an identity. $\Pi$ reaches **0.26** in the highest bin the
panel draws, so a GC trend in $R_{\mathrm{CpG}}$ would appear in this curve scaled by
$\Pi$, not erased — the curve is flat because $R_{\mathrm{CpG}}$ itself is (0.997–1.014),
which is the next paragraph's subject and is *correct* rather than a failure.

**Where each quantity comes from.** `data._r_eff_components` builds the four per-window
sums, one SQL query, and `data.r_eff_by_gc` does the binning and the divisions:

| symbol | column | built from |
|---|---|---|
| $E_1(w)$ | `e1` | `expected_counts_by_context_methyl_genome_1kb.txt` — published, already summed over all 32 contexts, $r\equiv1$ |
| $E_2(w)$ | `e2` | `refits/expected_counts_by_context_methyl_genome_1kb.{pop}.txt` — the same windows after the refit's $r$ |
| $E_1^{\mathcal K}(w)$ | `e1_cpg` | `expected_counts_per_context_methyl_genome_1kb.txt`, the per-$(w,t)$ export, restricted to $t\in\mathcal K$ and summed |
| $E_2^{\mathcal K}(w)$ | `e2_cpg` | the same rows times `rr` from `refits/rr_by_context.{pop}.txt`, the per-$(w,t)$ adjustment |
| $E_1^{\neg\mathcal K},E_2^{\neg\mathcal K}$ | — | subtraction, `e1 - e1_cpg` and `e2 - e2_cpg` |

The subtraction is why only the CpG slice of the two multi-GB per-context files is ever
joined: an 85M × 85M join becomes 10M × 10M. It also means the identity above mixes two
published files — totals from the summed export, CpG parts from the per-context one — so
it needs them to describe the same counts. They do:
`preconditions/verify_expected_r1.py` regenerates the first from the second genome-wide,
`possible` exactly and `expected` to $4.6\times10^{-5}$ relative.

$R_{\mathrm{eff}}$, $R_{\mathrm{CpG}}$, $R_{\mathrm{non}}$, $\Pi$ and the counterfactual
are then the columns `r_eff`, `r_cpg`, `r_non`, `pi_cpg`, `r_counterfactual`. One further
column, `r_eff_published` $=\sum_g E_2^{\mathrm{pub}}/\sum_g E_1$, replaces the refit's
numerator with Chen et al.'s own published `expected`. It needs no per-context $r$, which
is what makes it a check rather than a restatement: the two agree to $1.7\times10^{-5}$
per bin, median $2.1\times10^{-6}$, and that is what licenses using the refit's
per-context $r$ above — the published pipeline
writes its own only to a local directory, never to the bucket.

**Why $R_{\mathrm{CpG}}\approx1$ is correct, not a failure.** CpG mutability is dominated
by methylation, and $p_c$ is already keyed by methylation level — across methylation 0 to
15 the CpG C>T rate spans **3.0–4.3×** depending on context, the largest single rate
effect in the model, inside one trinucleotide. High-GC CpGs are CpG islands: **92%
hypomethylated in the top GC bin**, against 2.5% in the GC bulk, and their empirical DNM
rate falls from 0.532 in the bulk to 0.283 in that bin (**1.9×**). **Step 1 has already
applied that correction**,
via the covariation of GC content with methylation, so there is nothing left for $r$ to
adjust. (Chen et al. also strip `GC_content`, `CpG_island`, `Nucleosome`, `SINE` and
`met_sperm` from CpG-context models, so those models could not express a GC dependence
even if one were needed.) The GC trend in what Gnocchi applies is therefore entirely
non-CpG — which is where the rest of this figure looks.
""")

code(r"""
binned_b = D.r_eff_by_gc(df_win, edges, pop="full", cache_dir=CACHE_DIR)

# No mean-GC line on this panel, though A, D and E carry one -- panel_r_eff's docstring
# says why, and it is worth reading before adding one back. The mean is still printed:
# it is the number the caption would quote, and the one that says how far out in the
# tail the divergence at GC 0.6-0.7 happens.
print(f"panel B: mean GC of the {int(binned_b['n'].sum()):,} windows = "
      f"{float((binned_b['n'] * binned_b['gc_mid']).sum() / binned_b['n'].sum()):.3f}")

fig, ax = plt.subplots(figsize=FIGSIZE)
panels.panel_r_eff(ax, binned_b, min_n=MIN_N_WINDOWS, xrange=XRANGE)
save(fig, "B")

binned_b.filter(pl.col("n") >= MIN_N_WINDOWS).select(
    ["gc_mid", "n", "pi_cpg", "r_eff", "r_cpg", "r_non", "r_counterfactual"])
""")

md(r"""
## Panel C — the training set is not the scored population

$r$ is fit on de novo mutations and matched background sites drawn from the **whole
genome**, but applied to — and here evaluated on — the scored population. Panel C
measures how far apart those two drift with GC.

**Two scored populations, and the panel follows whichever is in force.**
`NEUTRAL_WINDOWS_BED` selects between the 1,843,559 windows this repo builds from the
bucket (noncoding + `pass_qc` + autosome/PAR) and McHale et al.'s own 693,270. Either
way the genome splits three ways — **QC-pass noncoding, QC-pass coding, QC-fail** — and
unset, the first of those *is* the scored population, so the stack has three bands. Set,
QC-pass noncoding subdivides into their set and the rest of it, giving a fourth:

$$N(g)=N_{\mathrm{scored}}(g)+N_{\mathrm{other}}(g)
        +N_{\mathrm{coding}}(g)+N_{\mathrm{QC}}(g).$$

Map each non-CpG training site — **both classes, DNMs and background alike** — to its
containing 1 kb tile; the stack plots those fractions per GC bin, and they sum to 1 by
construction. Nothing here re-derives a filter: a site is *scored* exactly when its
window is a row of the analyzed window table, the same table `refit.py` restricts the
training set with and panels A, B and E are evaluated on. So the bands track the
configuration on their own, and a stratum that is empty draws no band and no legend
entry. Between the two configurations QC-fail is identical — it is the windows with no
row in Chen et al.'s table — and what moves is the QC-pass noncoding territory, cut into
the scored and other bands. The coding band can shift a little too, by any window their
file lists that this repo would call coding; the join prints that count.

**The numbers below are from the narrowed run** — `NEUTRAL_WINDOWS_BED` set, which is
what this notebook's panels were built from. So the stack has four bands, its bottom one
is McHale et al.'s 693,270 windows, and the lower row's ratios divide by that band rather
than by QC-pass noncoding as a whole.

**Why both classes.** In a case-control design it is the controls that carry the
covariate distribution, which argues for stacking the background class alone — but that
is an argument about the *design*, not the *fit*. The fit minimizes its loss over the
mixture, so the mixture is the training distribution, and that is what belongs beside the
scored population. Stacking the background class alone barely moves the picture — on the
1,843,559-window reproduction, where this was measured, the QC-fail band runs 0.13 → 0.24
instead of 0.14 → 0.28 — because it outnumbers the DNMs about 12:1; what the DNM class
adds is its own steeper drift, 64% QC-fail at GC 0.64 against the background class's 26%.

**What the two rows are between them.** The upper row is *covariate shift*: $P(x)$ in the
training data is not $P(x)$ in the scored data, and the gap grows with GC. The lower row
is *concept shift*: $P(\mathrm{DNM}\mid x)$ is not the same function in the territory that
drops out. Both are driven by one latent variable — which class of window a site sits in
— and $r$ cannot see it, being fit per trinucleotide context on regional features, none of
which encodes window class. The drifting case-control ratio is not itself the problem:
under case-control sampling the fitted slopes stay consistent whatever the ratio, and
only the intercept shifts. Covariate shift alone would also be survivable, since a
correctly specified model extrapolates. It is the combination — a mis-specified smooth
fit, over a covariate range where the label-generating process is a different one — that
puts a GC slope into $r$.

**The bands above the scored one**, where the labels matter because the mechanism
differs. A *coding* window has a published Gnocchi score; it is this analysis that sets it
aside, following McHale et al.'s noncoding restriction. A *QC-pass putatively nonneutral
noncoding* window is QC-pass and noncoding but outside their set — the rest of that
category, given up in narrowing to it, for an enhancer overlap or one of their interval
exclusions. Its label is the complement of the bottom band's own, *QC-pass putatively
neutral noncoding*, so the two read as one partition of that category rather than as a
named class beside a leftover; *putatively* is load-bearing on both sides, because being
outside a set McHale et al. call putatively neutral is not itself evidence of selection,
and whether these windows differ at all is exactly what the lower row measures.
A *QC-fail* window has no score at all —
Chen et al. dropped it before scoring, keeping only windows with $\ge1{,}000$ possible
variants, $\ge80\%$ of observed variants PASS, and mean coverage 25–35×
(`run_nc_constraint_gnomad_v31_main.py:296`). Since `pass_qc` is `True` on all 1,984,900
rows of the published table, that flag is inert and a QC failure shows up only as an
absent row.

That last band is therefore **not** "no gnomAD coverage", the name it carried here until
measured: of the 587,902 absent autosomal windows, all have their QC inputs on file, and
the PASS rule dominates the failures (417,097, against 19,396 for the coverage band;
`preconditions/verify_qc_filter.py` has the full split, and confirms the filter forwards
too — all 1,984,900 scored windows satisfy all three conditions). It is also a **mixture**
of coding and noncoding, deliberately: 6.9% of these windows overlap coding exons against
7.1% of the QC-pass ones, so QC failure is near-independent of coding status and this is
not a coding band in disguise.

The QC-pass noncoding share of the training set falls from 0.82 in the GC bulk to 0.27 by
GC 0.68, and the scored band — the part of that territory McHale et al. call putatively
neutral — peaks at 0.36 near GC 0.35 and is down to 0.007 over the same distance. So the
model is fit on one population and applied to another, and the two come apart exactly
where panel A's bias is largest.

**The part of the DNM training set that lies outside the scored population has a
different DNM rate** — that is what the lower row measures, and the difference is not
mainly the coding part. Writing $\bar y_s(g)$ for the empirical
non-CpG DNM rate in stratum $s$ and bin $g$, it plots
$\log\!\big[\bar y_s(g)/\bar y_{\mathrm{scored}}(g)\big]$ for each excluded stratum,
against a linear axis, with error bars $\pm\,\mathrm{SE}$ where

$$\mathrm{SE}\Big(\log\frac{\bar y_s}{\bar y_{\mathrm{scored}}}\Big)
=\sqrt{\frac{1-\bar y_s}{k_s}+\frac{1-\bar y_{\mathrm{scored}}}{k_{\mathrm{scored}}}},$$

the delta-method binomial SE with $k$ the DNM count. That SE is symmetric in the log
ratio and in nothing else, so the log is the scale on which the bar drawn is the
interval the data support; it also puts a $2\times$ excess and a $2\times$ deficit
equally far from the reference line. Zero would mean those sites are exchangeable with
the scored ones as far as mutation rate goes. The coding stratum sits there — 0.86–1.00×,
i.e. $|\log|\lesssim0.15$, flat across the whole range — while the QC-failing stratum
runs 1.50–1.63× ($\log = 0.41$–$0.49$) through the GC bulk and **3.39×
($\log = 1.22$) by GC 0.58**. Under the narrowed population the curve to read first is
the putatively nonneutral one, because it is precisely what the narrowing gave up: it is
flat, at 0.94–1.03×, so the narrowing costs sample size and nothing else and the rest of
this figure carries over unchanged. Had it climbed with GC it would have been a third
population change, and would have belonged in the caption beside the other two.

Both rows come from one table of per-stratum counts — this row divides the DNM count by
the site count, the row above takes the site count as a share of its bin. 72,801 of the
non-CpG autosomal DNMs fall in the QC-failing stratum and 17,537 in coding windows,
against 241,487 in QC-pass noncoding ones — 91,906 of those in the scored band, 149,581
in the putatively nonneutral rest of it.

So the steep GC dependence the model learns comes from sequence gnomAD could not call
reliably, which is also where trio DNM calling is least reliable: part of that excess is
plausibly false-positive DNM calls rather than real mutation.
""")

code(r"""
# One table, two rows: `st` carries each stratum's site count (the composition above) and
# its DNM count (the rate below), so the two rows cannot describe different sites.
st = D.dnm_rate_by_stratum(edges, df_win, cache_dir=CACHE_DIR)
comp = D.training_composition(st, edges)
ratios = D.stratum_ratios(st, edges)

fig, (axC1, axC2) = plt.subplots(2, 1, figsize=(FIGSIZE[0], 7.6), sharex=True,
                                 gridspec_kw={"height_ratios": [1, 1], "hspace": 0.12})
# The bottom band's meaning follows NEUTRAL_WINDOWS_BED, but its label would not, so
# name the population in the legend rather than leaving the panel ambiguous on its own.
panels.panel_training_composition(
    axC1, comp, min_n=MIN_N_SITES, xrange=XRANGE, show_xlabel=False,
    scored_note=("QC-pass noncoding" if NEUTRAL_WINDOWS_BED is None
                 else "QC-pass putatively neutral noncoding"))
panels.panel_stratum_ratios(axC2, ratios, xrange=XRANGE)
save(fig, "C")

# Columns from D._STRATA and from whatever stratum_ratios built, rather than a hardcoded
# list: with NEUTRAL_WINDOWS_BED set there is a fourth stratum, and both tables gain
# columns.
print(comp.filter(pl.col("n_total") >= MIN_N_SITES)
          .select(["gc_mid", "n_total"] + [f"frac_{s}" for s in D._STRATA]))
print(ratios.select([c for c in ratios.columns if c != "gc_bin"]))
""")

md(r"""
## Panel D — restricting the training set flattens the DNM rate, and the fit can then track it

**What the two curves of a pair are estimating — one number, by two routes.** Fix a
training population and work over its non-CpG sites only. Write $S_g$ for the sites whose
GC content falls in bin $g$, $y_i\in\{0,1\}$ for site $i$'s DNM label, and $x_i$ for its
feature vector. The quantity the panel is about is the probability that a training site
in that bin carries a DNM,

$$P_g \;\equiv\; \Pr\!\big(y=1 \,\big|\, \mathrm{GC}\in g\big)
\;=\; \mathbb{E}\Big[\,\Pr\!\big(y=1 \mid x\big)\,\Big|\,\mathrm{GC}\in g\Big],$$

the second form by the tower property — and its inner conditional is exactly what each
trinucleotide context's logistic model estimates. The panel plots one estimate of $P_g$
from each form, both plug-in averages over the sites the bin actually holds:

$$\bar y(g)=\frac{1}{|S_g|}\sum_{i\in S_g} y_i \;\approx\; P_g \quad\text{(empirical)},
\qquad
\widehat p(g)=\frac{1}{|S_g|}\sum_{i\in S_g}\hat\pi_i \;\approx\; P_g \quad\text{(fitted)},$$

where $\hat\pi_i=\widehat{\Pr}(y=1\mid x_i)$ is the site's own fitted probability from its
own context's model, evaluated at the **site's** feature vector rather than at the
window-aggregated values the genome-wide apply uses.

The left one is the binomial MLE of $P_g$: unbiased, with standard error
$\sqrt{\bar y(1-\bar y)/|S_g|}$, which is what the error bars carry. The right one is a
Monte-Carlo average of the fitted conditional over the same sites, so it estimates the
same $P_g$ **provided the model is calibrated on that bin's covariate distribution** —
and departs from it when the model is not, which is the whole point of the panel. So
$\widehat p(g)-\bar y(g)$ is a calibration gap conditioned on GC, and each population is
a reliability diagram on itself: *fitted vs empirical within a population* is the
comparison that means something.

**The panel is drawn as two rows over one x axis** — $\bar y(g)$ for both populations on
top, $\widehat p(g)$ for both below — because there are two claims here and they are read
in different directions. The first is about the empirical curves alone (restriction
flattens $\bar y$ and removes its turnover), and it is read *within* the top row, where
those two curves are now adjacent instead of separated by two fitted ones. The second is
the calibration gap above, read *between* the rows at fixed GC; the rows therefore share
one y range, computed over all four curves, so a given vertical distance is the same
interval in either row. A population keeps one symbol and one dash pattern in both rows,
so it reads as a single object down the figure. The axis is linear: normalization puts
every curve through 1 by construction, so that 1 is not a null a departure is being
weighed against, and the ratios the panel is quoted for are read off the curves rather
than off the axis.

Two cautions on what $P_g$ is *not*. It is a probability under the **training**
distribution, not the genome's: every DNM is kept while background sites are sampled at
some rate $s$, which multiplies the odds by $1/s$ and so shifts $\operatorname{logit}P_g$
by $-\log s$ — a constant in $x$, and in particular in GC. Levels are therefore not
comparable **across** populations, whose case-control ratios differ (12.2, 12.2 and 13.4
background sites per DNM), while shapes in $g$ are; each curve is divided by its own
site-weighted mean, leaving shape only. And $g$ enters only as a conditioning event: GC
is one of the 13 candidate regional features, but $\widehat p(g)$ averages whatever the
fit selected for that context, so a pair can come apart in a bin even where GC itself was
never selected.

Two populations:

* **original** — the training set as published;
* **scored** — restricted to the analyzed windows (the intervention).

A third, **size-matched** — the same *number* of sites as *scored*, drawn at random from
the whole genome — is fit and reported, but not drawn here. It is the control that
separates "better-matched population" from "less data", and it lies on top of the
original pair; a curve indistinguishable from one already on the axes costs a series in
each row to say nothing the number cannot. It is reported under panel E, where the
same control is measured on the statistic that matters.

Two things happen at once when the training set is restricted. The **empirical** GC
dependence itself shrinks and becomes monotone — on the original set $P(\mathrm{DNM})$
rises 2.45× and then *collapses* above GC 0.66; on the scored set it rises smoothly by
1.60× with no turnover. And the logistic regression can then actually fit it: on the
original set it misses by 26% and 29% *in opposite directions*, because it is a smooth
monotone surface chasing a curve that turns over, where on the scored set it tracks to
within 6% through GC 0.58. The scored pair's final bin is the exception — 670 sites, and
the fit 28% low there — so the claim is about the GC range the restriction leaves
populated, not about the tail it thins.

This is why $R_{\mathrm{non}}$ in panel B is inflated — the model has partly learned a GC
slope that belongs to sequence outside the scored population.

**Caveat.** This panel measures a *level* error in $P(\mathrm{DNM})$, and levels cancel in
$r$. It diagnoses the fit; it is not itself a measurement of Gnocchi's bias. Panel E is.
""")

code(r"""
binned_d = D.dnm_probability(("full", "scored"), n_bins=N_BINS, min_n=MIN_N_SITES)

# ONE vertical reference for two populations, so it has to be a population the panel
# names: `full`, the original training set the other is compared against. Both
# site-weighted means are printed below -- if they ever stop agreeing to ~0.01, this
# panel needs two lines or none, because a single one would then be marking a place
# one of its curves is not centred on. In the panel's 0-1 units, hence the /100.
site_mean_gc = lambda b: float((b["n"] * b["gc_mid"]).sum() / b["n"].sum()) / 100.0
gc_mean_d = site_mean_gc(binned_d["full"])

# Two rows over one x axis, as in panel C, and the same figure height: the top holds
# the empirical curves and the bottom the fitted ones. sharex is what lets the two be
# read as one panel; the shared y range that makes the rows comparable is set inside
# panel_dnm_probability_pairs, from both rows' values, rather than here.
fig, (axD1, axD2) = plt.subplots(2, 1, figsize=(FIGSIZE[0], 7.6), sharex=True,
                                 gridspec_kw={"height_ratios": [1, 1], "hspace": 0.12})
panels.panel_dnm_probability_pairs(axD1, axD2, binned_d, min_n=MIN_N_SITES,
                                   normalize=True, gc_mean=gc_mean_d)
save(fig, "D")

print("mean GC of the training sites drawn:  "
      + ",  ".join(f"{pop} {site_mean_gc(b):.3f}" for pop, b in binned_d.items()))
for pop, b in binned_d.items():
    print(f"\n{pop}:")
    print(b[["gc_mid", "n", "n1", "mean_pred", "empirical_prop"]].round(4).to_string(index=False))
""")

md(r"""
## Panel E — refitting $r$ on the scored population removes Gnocchi's bias

Exactly one thing changes relative to the published pipeline: training sites outside the
analyzed window set are dropped. Everything downstream is identical — univariate
Bonferroni selection, standardization, `IncrementalPCA`, L1 logit per context,
genome-wide apply. So panel E is panel A's statistic with a third expected count,
$E_2^{\mathrm{scored}}$, added to the same joint $z$ filter and the same ranking.

Two controls make this a result rather than an anecdote, and both are printed below
rather than plotted (a curve indistinguishable from published Gnocchi is clutter):

* the **full-population refit** through this same code lands on published Gnocchi, so
  "before" and "after" differ by the intervention and not by whose code produced them;
* the **size-matched random refit** also lands on published Gnocchi, so the improvement
  is about *which* sites, not *how many*.

Note the retrained curve is expected to end up *better than* the context-only model, not
merely closer to it: a correct $r$ should repair the context-only model's own droop at
both GC extremes. That is the sense in which this is a positive result and not just the
removal of a defect. On this run it does: mean $|\rho-0.5|$ is 0.168 for published
Gnocchi, 0.046 for the context-only model and **0.026** for the retrained one, against
0.168 for the full-population control and 0.162 for the size-matched one. The first
three of those five are the curves the panel draws, and it carries them in its legend;
the two controls are print-only.
""")

code(r"""
extra = [("scored", D.refit_path("expected", "scored"))]
CONTROLS = [("full_refit", "full"), ("sizematched", "sizematched")]
for label, pop in CONTROLS:
    extra.append((label, D.refit_path("expected", pop)))

df_e, binned_e = D.rank_curves(df_win, extra=extra, min_n=MIN_N_WINDOWS)

# Published Gnocchi first, then the context-only model it is built on, then the retrained
# score -- the order the argument is made in, and the same first-two order as panel A.
curves_e = [
    panels.curve_from_binned(binned_e, "step2", "step2", "Gnocchi, as published"),
    panels.curve_from_binned(binned_e, "step1", "step1", "Gnocchi, context-only"),
    panels.curve_from_binned(binned_e, "scored", "scored",
                             "Gnocchi, decontaminated DNM training set"),
]

fig, ax = plt.subplots(figsize=FIGSIZE)
# Lower left: E carries three labels, each with its bias appended, and at the panel's
# type size that block runs into the published curve if the legend stays top-left as in A.
# show_bias=True: each legend entry carries its own mean |rank - 0.5|, computed over the
# same bins the panel draws. This panel IS the measurement of the bias and of its repair,
# so the three numbers belong on it. Panel A's stay in its caption -- it is the panel that
# poses the problem, and a reader meets the statistic there for the first time.
panels.panel_rank_bias(ax, curves_e, gc_mean=float(df_e["GC_content"].mean()),
                       xrange=XRANGE, min_n=MIN_N_WINDOWS, legend_loc="lower left",
                       show_bias=True)
save(fig, "E")
""")

md(r"""
## Supporting figure (**Supporting Figure 7** in the manuscript) — why $R_{\mathrm{CpG}}\approx1$ is correct

Panel B asserts that CpG contexts *should* apply no GC-dependent adjustment, because the
effect that would need adjusting has already been applied in step 1. That claim rests on
three measurements which the panel itself does not show, so they get their own figure,
`output/supp_fig7.pdf`:

**A. The size of the methylation effect step 1 absorbs.** $p_c$ is keyed by methylation
level, and for CpG C>T the fitted probability spans **3.0–4.3×** from level 0 to 15 —
within a *single* trinucleotide context, and the largest single rate effect in the model.
The dashed curves are `mu`, the same observable read in a 1,000-genome cohort and rescaled,
which spans 9.7–15.2× over the same range; it is not itself a measured rate, and the
subsection below says what it is. The gap between the curves is the saturation of $p_c$.

**B. High-GC CpGs are CpG islands.** The hypomethylated fraction (level $\leq$ 1) runs
2.5% through the GC bulk and **92% in the top GC bin**, the one bin above GC 0.70; mean
methylation falls from ~6.4 to 0.47 over the same range.

**C. So their DNM rate collapses.** Flat at 0.532 through the bulk, falling to **0.283**
in the top GC bin — a 1.9× fall, tracking B. The two highest bins hold 932 and 1,434
sites; their error bars carry that.

**D. And these contexts are not a rounding error.** $\Pi$, the CpG share of a bin's step-1
expected counts — the same weight panel B's identity
$R_{\mathrm{eff}}=\Pi R_{\mathrm{CpG}}+(1-\Pi)R_{\mathrm{non}}$ uses — rises from
**0.038 to 0.264** across the GC range. So the flatness of panel B's counterfactual is a
measurement and not an artifact of negligible weight: had $R_{\mathrm{CpG}}$ carried a GC
trend, it would have reached the applied multiplier scaled by up to 0.26. This is the
panel that says the A–C mechanism *matters* rather than merely *holds*.

Put together: a large, strongly GC-dependent CpG effect exists, step 1 already applies it,
and the contexts it applies to carry up to 26% of the expected counts. There is nothing
left for $r$ to correct, which is exactly what panel B measures. (Level, not rate: these
are case-control-sampled training sites at ~10:1, so the y-axis of C is not a genome-wide
mutation rate.)

D is binned over **Chen windows** and A–C over **training sites**, the same split the main
figure runs on, so the two do not end in the same place: D's last bin above the 100-window
floor is centred at GC 0.65 against 0.73 for B and C. All four panels share this figure's
0.2–0.8 axis. On the narrowed window set D no longer runs past panel B's own 0.2–0.73
range; that it did — reaching GC 0.75 with $\Pi$ at 0.43, so that the bins carrying the
most CpG weight were exactly the ones the main panel could not show — was a feature of the
1,843,559-window run.

### Panel A's two curves

**The model both curves live in.** Infinite sites plus a genealogy, with $u_c$ for class
$c$'s *true* per-generation mutation rate. Trace the ancestry of
the cohort's $n$ chromosomes back to their common ancestor and add up every generation of
transmission in that tree: $L_n$ generations in all, the genealogy's **total branch
length**. Each one is an independent chance for the site to mutate at rate $u_c$, so the
number of mutations the site accumulates anywhere in the cohort's ancestry is Poisson with
mean $u_c L_n$.

```
                      │      above the MRCA: NOT part of L_n - a mutation
                      │      up there is inherited by the entire cohort,
                      │      so the site looks monomorphic, never counted
                 ┌────┴────┐ <- MRCA
                 │         │
                 │         X     <- one mutation, at rate u_c per generation
                 │         │
              ┌──┴──┐   ┌──┴──┐
              │     │   │     │
             c1    c2  c3    c4     <- the cohort: n sequenced chromosomes
                       ^^    ^^
                       c3 and c4 carry the variant, c1 and c2 do not,
                       so the site is POLYMORPHIC in the cohort

  L_n = total length of every branch below the MRCA, in generations
        (here 4 tip branches + 2 internal ones)
```

Two facts turn the mutation count into the thing the data reports. First, a mutation on any
branch of that tree is inherited by *some but not all* of the cohort's chromosomes —
branches above the common ancestor are not part of $L_n$, and are exactly the ones that
would be invisible — so it shows up as a variant in the cohort. Second, *infinite sites*
says no site is ever hit twice, so no mutation reverts or lands on an allele already
present, and none of them cancel. The site is therefore polymorphic exactly when the count
is non-zero, which has probability $1-\Pr(\text{count}=0)$:

$$P_n(c) \;=\; 1-e^{-u_c L_n}.$$

A rate and a polymorphism probability are therefore two points on one curve, separated
only by $L_n$ — all of the cohort-size dependence sits there, and all of the sequence
biology in $u_c$. The single-genome case is $n=2$, where $L_2=4N_e$ and
$P_2\approx4N_e u_c$ is heterozygosity: $4\cdot2\times10^{4}\cdot1.2\times10^{-8}\approx
9.6\times10^{-4}$, the textbook human per-base value.

**Two cohorts, one parameter** (Peter McHale's reduction, and the cleanest way in). Read
the formula at the two cohort sizes the panel uses. In the 1,000-genome downsample
saturation is weak, so the exponential linearizes:

$$P_{2000}(c) \;\approx\; u_c L_{2000}.$$

Substitute $u_c = P_{2000}(c)/L_{2000}$ into $P_n(c)=1-e^{-u_c L_n}$ and the rate drops
out entirely:

$$\boxed{\;P_n(c) \;\approx\; 1-e^{-k\,P_{2000}(c)},\qquad k=\frac{L_n}{L_{2000}}\;}$$

One parameter, and it is not a mutation rate: it is the **ratio of the two cohorts'
genealogies**. Everything class-specific — context, methylation, the whole point of the
table — has cancelled, and with it any need for a notion of an absolute mutation rate:
the two cohorts calibrate each other.

**The two curves are that formula read at the two cohort sizes.** The **solid** curve is
$P_n(c)$ itself at $n=152{,}312$ chromosomes — $p_c$ = `fitted_po`, the probability a site
of that class is polymorphic in the 76,156-genome call set, and the operative output of step 1
(`expected = possible × fitted_po`). The **dashed** curve is Chen et al.'s `mu` column,
which despite its name and its units is $P_{2000}(c)$ times one global constant; the panel
divides each curve by its own methylation-0 value, so that constant cancels exactly and the
dashed curve *is* $P_{2000}$, normalized. Saturated end and unsaturated end of the same
relation, in other words — which is why the solid curve is so much the flatter of the two.
Chen et al. plot the relation itself in their **Extended Data Fig. 1a**: proportion observed
in 76,156 genomes against `mu`, "exponentially correlated" in their caption, fitted as
$y=1-\exp(-1.88\times10^{7}x-7.32\times10^{-5})$ with $R^2=0.999$ — the boxed relation, in
their units.

*"Unsaturated" is relative.* At 1,000 genomes the median row is 0.75% polymorphic, but the
top of the dashed curve — ACG C>T at methylation 15 — is **27.6%**. Read as a Poisson rate,
$-\log(1-P_{2000})$, that context's span against methylation 0 is 13.2× rather than the
11.4× the raw proportion gives, so the dashed curve understates the true spread as well and
the solid curve's compression is a lower bound.

**What the gap does and does not imply.** Saturation is not an error in Chen et al.'s
model. The expected counts built from $p_c$ are compared against observed *polymorphism*
counts, which saturate identically, so it cancels — the same cancellation that makes a
level error invisible in $r$.
""")

code(r"""
ct = D.cpg_rate_by_methyl(cache_dir=CACHE_DIR)
cpg = D.cpg_methylation_by_gc(edges, cache_dir=CACHE_DIR)

# A ON THE LEFT, B-D STACKED ON THE RIGHT. A is the one panel whose abscissa is not GC
# content -- it is methylation level -- so standing it beside the others rather than above
# them stops the figure reading as one column against one x-axis, and puts the three GC
# panels together, which is the order B -> C -> D is argued in. It also takes the figure
# from 7 x 16.5 in, a column no page holds, to a landscape page.
#
# Six rows so A can span the middle four: it then sits vertically centred against the
# three on the right at close to its own aspect, where spanning all six would stretch a
# saturation curve over the full height of the page.
fig = plt.figure(figsize=(12.5, 9.8))
gs = fig.add_gridspec(6, 2, width_ratios=[1.05, 1], wspace=0.28, hspace=0.85)
axA = fig.add_subplot(gs[1:5, 0])
axB, axC, axD = (fig.add_subplot(gs[i:i + 2, 1]) for i in (0, 2, 4))

panels.panel_cpg_methylation_effect(axA, ct)
# MIN_N_CPG, not MIN_N_SITES: the top two GC bins (n = 932 and 1,434) carry the claim,
# and their error bars show how thin they are. Quote numbers from this same subset.
#
# EVERY PANEL LABELS ITS OWN X-AXIS, hence no show_xlabel=False here. Nothing here shares
# an axis: A's abscissa is methylation level, and B-D, though stacked, are three separate
# axes with their own ticks and a gap between them rather than one sharex column, so each
# is read -- and cited -- on its own. The suppression that is right in panel C, where two
# sharex axes ARE one plot, would here leave two panels whose x quantity is named nowhere.
panels.panel_cpg_hypomethylation(axB, cpg, min_n=MIN_N_CPG)
panels.panel_cpg_dnm_rate(axC, cpg, min_n=MIN_N_CPG)
# binned_b and MIN_N_WINDOWS, not cpg/MIN_N_CPG: Pi is a per-window weight, and this is
# the same table and the same bin floor panel B decomposes, so the curve here IS the
# weight in that identity rather than a second estimate of it.
panels.panel_cpg_expected_share(axD, binned_b, min_n=MIN_N_WINDOWS)
# Two calls, two offsets: label_panels places the letter in AXES coordinates, so the -0.1
# that clears A's ylabel lands on top of the narrower right-hand panels' ylabels.
panels.label_panels((axA,), ("A",))
panels.label_panels((axB, axC, axD), ("B", "C", "D"), x=-0.26)

supp_name = f"supp_fig7{config.WINDOW_SET_SUFFIX}"
written = resave_ai.save_panel(fig, os.path.join(OUTPUT_DIR, supp_name))
print(f"wrote {', '.join(os.path.basename(p) for p in written)}" if written
      else f"{supp_name}: unchanged, left alone")
""")

code(r"""
shown = cpg.filter(pl.col("n") >= MIN_N_CPG)
bulk = shown.filter((pl.col("gc_pct") > 35) & (pl.col("gc_pct") < 50))
p_bulk = float(np.average(bulk["p"], weights=bulk["n"]))
top = shown.sort("gc_pct")[-1]
print(f"Supporting Figure 7B: hypomethylated fraction "
      f"{float(np.average(bulk['frac_hypomethylated'], weights=bulk['n'])):.3f} in the "
      f"GC bulk -> {shown.filter(pl.col('gc_pct') > 70)['frac_hypomethylated'].min():.2f}"
      f"-{shown['frac_hypomethylated'].max():.2f} above GC 0.70")
print(f"Supporting Figure 7C: DNM rate {p_bulk:.3f} in the GC bulk -> "
      f"{float(top['p'][0]):.3f} in the top GC bin "
      f"({p_bulk / float(top['p'][0]):.1f}x lower)")
pi = binned_b.filter(pl.col("n") >= MIN_N_WINDOWS)["pi_cpg"]
print(f"Supporting Figure 7D: CpG share of step-1 expected counts "
      f"{pi.min():.3f} -> {pi.max():.3f}")
print(shown.select(["gc_pct", "n", "mean_methyl", "frac_hypomethylated", "p"]))
""")

md(r"""
## Supporting figure (**Supporting Figure 8** in the manuscript) — what panel E buys, or costs, in *discovery*

Panel E says the retrained score is no longer GC-biased. It cannot say whether the biased
score was nevertheless the better **detector**: bias and signal-to-noise act on discovery
jointly (McHale et al.'s Fig. 3), and only one of the two has been changed. This figure is
that test, built as McHale et al.'s **Fig. 4A/B** — a classifier that calls a window
constrained when its Gnocchi $z$ exceeds a threshold, a **lax** truth set (does the
window overlap a GeneHancer enhancer), and performance read off the precision–recall curve
*within each GC bin* — with **two Gnocchi variants in place of their four constraint
metrics**.

***Lax* is McHale et al.'s own word, and it anticipates its opposite.** Not every
enhancer-overlapping window is under strong selection — GeneHancer covers 18.4% of the
noncoding genome while perhaps 4.51% of it is under human-specific selection — so the lax
set buys *size*, enough windows to resolve performance deep in the GC tails, at the cost
of label confidence. Their **stringent** truth set (Fig. 4C/D: noncoding windows that
regulate essential genes, against an equal number overlapping no enhancer at all) makes
the opposite trade, and is **not built here yet**. `data.pr_curves` takes a `truth_set`
argument and accepts only `"lax"` so far; the constants that belong to a truth set carry
its name (`LAX_GC_BINS`, `LAX_MIN_BIN_WINDOWS`) and the ones that do not, do not
(`PR_SCORES`, `TRUTH_TARGET`).

**It is this figure's own pipeline with one filter dropped.** The window population comes
from the same `windows.build_window_table()` call the panels above use, on the same
`NEUTRAL_WINDOWS_BED`, with `keep_enhancer_windows=True`: their file is still the
definition, the noncoding / QC / autosome filters are still skipped, and the $z$ formula,
the joint $[-10,10]$ filter and the GC units are still the ones panels A and E use. The
only change is that the `enhancer == False` step does not run, and the flag comes back as
a column instead of a filter. Panel E must *not* have those windows — a window under
selection has a low $z$ for a reason that is not bias — and this figure cannot do without
them, because they are the positive class.

So note which population is which: the `scored` refit is still **fit** on the putatively
neutral windows alone (that is the intervention), and **evaluated** here on neutral *and*
enhancer windows. Fit on the negatives, scored on both, which is what a classifier
requires. The caption should say so.

| | Shows | Quantity |
|---|---|---|
| **A** | Performance varies strongly with GC content, and does so for *both* scores | precision vs recall, one line per GC bin, over the pooled curve and the random-classifier baseline |
| **B** | What the decontamination does to discovery | auPRC normalized by the positive-class fraction, vs GC content, one curve per score |
| **C** | Whether any of B's gap is real | the **paired** gain of the retrained score over the published one, per GC bin, with a bootstrap 95% CI |
| **D** | The bias, at the threshold people actually use | fraction of windows in each GC bin with Gnocchi $\geq 4$ |
| **E** | What an analyst gets from a call | $P(\mathrm{constrained}\mid\mathrm{Gnocchi}\geq4)$ per GC bin, over the bin's base rate |
| **F** | The bias as a *shape* | lift against recall, one point per GC bin, with iso-calling-rate contours |

**A–C are nearly blind to the bias, and D–F are where it appears.** A, B and C are
*within-bin ranking* statistics, and a GC-dependent bias is very nearly a common shift
applied to every window in a narrow bin — positives and negatives alike. A common shift
cannot change a ranking, so it cancels, which is why B's two curves nearly coincide and C
finds differences of order 1%. That is not a null result about the bias; it is a statement
that auPRC within a GC bin is the wrong instrument for detecting it. Two consequences
worth stating in the caption:

* The steep decline of auPRC with GC in **B** *survives debiasing intact*. Since bias is
  the one thing that changed, what remains must be **signal-to-noise** — which is what
  McHale et al. conjectured in their text, now measured rather than assumed.
* To see the bias, hold the *threshold* fixed instead of the rank. Then the shift stops
  cancelling and decides how many windows in each GC bin are called at all.

**D–F use Gnocchi $\geq 4$, which is Chen et al.'s own cutoff** ("constrained non-coding
regions (Gnocchi ≥ 4)"), not a choice of ours, and they are computed **unbalanced** —
the base rate an analyst faces is the real one.

**The two scores are compared at a matched calling rate, not at a common number.**
Retraining moves the whole $z$ distribution, not only its GC dependence: at $z\geq4$ the
published score calls about 1.0% of windows and the retrained one about 0.13%, *eight
times fewer*, so the same numeral is a far stricter cutoff for one than the other.
Precision almost always rises as a threshold moves further into the tail, so a naive
comparison at a common $z$ would credit the retrained score for being strict and penalise
its recall for the same reason — neither of which is a statement about GC bias. So
published is held at 4 and the retrained score takes the quantile of its own $z$ that
calls the same fraction; each panel's legend carries its own threshold. **Panel D's
headline survives either way**, since a swing across GC is a ratio computed *within* one
score and a common rescaling cannot touch it — but E and F are only apples-to-apples this
way. Setting `match_call_rate=False` recovers the naive comparison, which is worth looking
at once to see the confound rather than to quote.

**Panel F is D, E and recall at once**, via an exact identity: recall $=$ calling rate
$\times$ lift. So a point's position fixes all three — recall on $x$, lift on $y$, the
calling rate their ratio — and on log–log axes that ratio becomes a difference, making
**iso-calling-rate contours parallel lines of slope 1**, drawn as the light guides. A score
whose threshold means the same thing everywhere calls the same fraction of windows in every
GC bin, so all its points lie on **one** contour; a GC-biased score's points fan **across**
them. Published spans about two orders of magnitude of contours and the retrained score
collapses onto one, and no summary statistic is doing any work in that comparison — it is
the geometry. It replaces the previous recall-versus-GC panel, whose content is now this
panel's $x$ axis.

**Its $y$ axis is already the corrected recall.** Recall does not need the base-rate
correction precision got, because its null is different: a random classifier's precision
*is* the base rate, but its recall is the *calling rate*. So recall's normaliser is $q$,
not $r$ — and $\text{recall}/q = \text{lift}$ identically, by the same cancellation. Base-rate-corrected
precision and calling-rate-corrected recall are one number. Raw recall stays on $x$
precisely because the gap between it and lift *is* the calling rate, which is the bias.

**Compare the two scores vertically, not horizontally.** The calling rates are matched
globally, not per bin, so within a bin the two scores still call very different fractions —
14% against 0.8% in the top GC bin. Horizontal distance between a square and a triangle is
therefore a calling-rate difference, not a performance difference; the legitimate
within-bin comparison is the vertical one, and `data.lift_deltas` puts an interval on it.

Neither axis is prevalence-free across bins: lift is capped at $1/r$ and recall at $q/r$ —
the same ceiling scaled by $q$ — so read each score's *shape* across bins rather than
ranking bins against each other, and use the per-bin `skill` and `LR+` for anything
cross-bin.

**Panel D needs no truth set at all.** The fraction of windows clearing a fixed cutoff is a
property of the score and of GC content; no labels enter. It is therefore the most robust
claim in this figure — it depends on neither GeneHancer nor the laxness of an
enhancer-overlap proxy — and it is the one to quote.

**Panel E will not be flattened by the correction, and should not be expected to be.**
Precision at a fixed threshold rises with GC for *any* score, because the base rate itself
climbs about 7.7$\times$ across these bins; that is why the panel draws the base rate as
its reference, and why the gap between a curve and the dashed line beneath it — not the
curve's height — is what carries information. That gap is the **lift**,
$P(Y{=}1\mid\text{called})/P(Y{=}1)$ — standard in classification evaluation, and called
*fold-enrichment* in genomics — and by Bayes it is also recall/calling-rate, which is the
identity tying D, E and F together.

**At a fixed calling budget, precision and recall are the same question.** The number of
calls is fixed by construction and the number of positives is a property of the truth set,
so precision $=TP/N_{\mathrm{called}}$ and recall $=TP/P$ are both monotone in $TP$; the
scores differ only in how they *spend* the budget. And at a fixed budget $TP$ is maximised
by ranking on $P(Y{=}1\mid\text{window})$, so adding any function of a covariate raises
$TP$ **iff** that covariate carries information about the label beyond what the score
already has. Here the base rate climbs ~7.7$\times$ with GC, so GC does — meaning
published Gnocchi's bias is partly acting as a *GC detector*, and removing it must cost
unconditional $TP$.

**That is why `data.budget_comparison` ranks on GC content alone as a fourth arm.** If GC
by itself approaches — or beats — published Gnocchi at the same budget, then a genome-wide
precision-recall comparison on an enhancer-overlap truth set is substantially a
GC-content contest and cannot be the criterion by which a bias correction is judged. Read
that table beside D–F and never instead of them: what the correction buys is *conditional*
(within a GC stratum, and threshold portability across strata), and a genome-wide average
marginalises over exactly the variable being fixed. The per-bin `GC only` rows in the
threshold table are the same control ranked **within** each bin, which is the yardstick a
score's within-bin lift has to beat to be measuring constraint rather than residual
within-bin GC.

**Lift is comparable between the two scores within a bin, but not between bins.**
Precision cannot exceed 1, so lift cannot exceed $1/r$ — a ceiling that falls from 12.0 in
(0.20, 0.30] to 1.57 in (0.55, 0.80]. A lift declining across GC is therefore partly the
ceiling coming down, not only the score getting worse; a cross-bin statement needs the
positive likelihood ratio $P(\text{call}\mid Y{=}1)/P(\text{call}\mid Y{=}0)$, or the skill
score $(\text{precision}-r)/(1-r)$ which maps random to 0 and perfect to 1. Both are
reported per bin as `LR+` and `skill`, so a cross-bin sentence can be written without the
ceiling caveat; neither is plotted. Within a bin both scores face the same $r$ and the same
ceiling, which is where `data.lift_deltas` compares them — with a **paired** bootstrap, for the reason panel C uses one. There the
base rate cancels outright, so the lift ratio *is* the precision ratio.

**Its intervals are wide wherever the calls are few**, because a precision at a fixed
threshold rests only on the *called* windows — a few dozen in the GC-poorest bin, against
the tens of thousands behind panel C. Hence `data.lift_delta_sweep`, which repeats the
comparison across calling rates. The intervals close as $\sqrt{\text{calls}}$ while lift is
nearly flat in the calling rate: measured on the stand-in, 1.67 → 1.56 in (0.20, 0.30]
over 1% → 10%, against a CI half-width falling 17.2% → 4.2% — a four-fold tightening for a
seven percent erosion. At 1% no bin's interval excluded zero; at 10% three did. A single
anchored point can therefore be a knife-edge, and the sweep is what shows whether it is.

It also makes a continuum in this figure explicit. As the calling rate rises the statistic
integrates over more of the ranking and converges toward what auPRC already measures —
panel C, whose intervals are ~0.2% wide. **Panel C is the powerful-but-uninterpretable end
of that axis and $z\geq4$ the interpretable-but-noisy end**, and neither is the right one
on its own: "the top 10% of the genome by Gnocchi" is not "Gnocchi $\geq$ 4", and only the
latter is the cutoff Chen et al. use and therefore the score as people apply it.

**Panel C is where the claim is decided, and panel B cannot do its job.** B draws two
curves that cross and wobble, without uncertainty, so a reader cannot separate a real gap
from a thin bin — and the largest gap sits in the thinnest bin, which is exactly the
pattern a chance finding makes. C reduces the comparison to one number per bin with an
interval on it.

Three things make that interval narrow enough to be worth drawing, and all three are
choices rather than luck:

* **It is paired.** The two scores are columns of one table — identical windows, identical
  positives, identical bins. Almost all of the sampling variability in auPRC is variability
  in *which windows the truth set happens to contain*, and that is common to both scores,
  so it cancels in the difference. Each bootstrap replicate resamples a bin's rows once and
  scores *both* models on that same resample. Independent error bars on B's two curves
  would describe the uncertainty of each **level** when the question is about the **gap**,
  and would make the result look far weaker than it is.
* **It is unbalanced.** The class balancing exists to make bins comparable in level — it is
  what makes A's single dashed baseline valid. A within-bin, between-score comparison needs
  none of it, and $r$ cancels from the relative gain outright, while the balancing discards
  about four fifths of the positives and bites hardest at high GC where positives are
  densest and bins are thinnest. Keeping them is free power exactly where the question is.
* **Its top bin is wider**, (0.55, 0.80] against B's (0.55, 0.60]. Their file is nearly
  empty above GC 0.60 — after balancing the three bins there hold 1,086, 65 and 2 windows —
  so lowering the floor alone buys one marginal bin and two useless ones. One merged tail
  bin is the honest use of it. **The caption must say so**, or C's last point looks like it
  disagrees with B's about where the measurement is.

Both scores live on the same rows, so the per-GC-bin positive-class downsampling happens
**once**, on the labelled table, rather than once per score as in the reference notebook
([`7.CDTS/main.2.ipynb`](https://github.com/quinlan-lab/constraint-tools/blob/main/papers/neutral_models_are_biased/7.CDTS/main.2.ipynb),
which compares metrics carried on four *different* window files and so has no choice). A
gap between the two curves in **B** is therefore the score and nothing else.

**Unlike every other panel, this one does not build without `NEUTRAL_WINDOWS_BED`.** The
truth set *is* the GeneHancer flag in that file, and GeneHancer is licensed and not
redistributable, so there is no version of it derivable from the public bucket. The cells
below check and skip rather than classify against some other annotation, which would be a
different experiment wearing this figure's name.

This is the one place in the notebook that builds a **second** window table, on a
different population from panel E's — but that is cheap, not the bottleneck it sounds
like: the duckdb scan behind it is column-pruned, and the whole figure (window table,
refit join, $z$, balancing, both panels' PR curves) times at **under four seconds** on a
warm cache. Adding it costs the rebuild essentially nothing.
""")

code(r"""
# One call: labelled table -> GC bins -> class balancing -> precision-recall curves.
# pr_curves() reads config.NEUTRAL_WINDOWS_BED itself, for both the population and the
# lax truth set's label -- one join supplies both, which is what keeps this figure's
# windows identical to panel E's apart from the enhancer-overlapping half.
curves_s8 = D.pr_curves(truth_set="lax", seed=0) if NEUTRAL_WINDOWS_BED else None
if curves_s8 is None:
    print("NEUTRAL_WINDOWS_BED is not set -- Supporting Figure 8 skipped.\n"
          "The lax truth set is the GeneHancer enhancer flag in that file, which is "
          "licensed and\nnot derivable from the public bucket, so there is nothing to "
          "classify against.")
""")

code(r"""
# Panel C: the paired bootstrap. Its own cell because it is the one slow step in this
# figure -- ~500 resamples x one precision-recall pass per drawn bin, twice -- and because
# re-running it should not mean rebuilding A and B. A couple of minutes.
deltas_s8 = D.pr_curve_deltas(truth_set="lax", seed=0, n_bootstrap=500) \
    if NEUTRAL_WINDOWS_BED else None
""")

code(r"""
# Panels D-F: everything at a FIXED threshold. Cheap -- no bootstrap, Wilson intervals in
# closed form, since these are plain proportions and the panels draw levels rather than a
# paired difference.
tm_s8 = D.threshold_metrics(threshold=D.GNOCCHI_THRESHOLD, truth_set="lax",
                            match_call_rate=True) \
    if NEUTRAL_WINDOWS_BED else None
""")

code(r"""
# The paired bootstrap on panel E's comparison, so it has the standing panel C's does.
# Same matched thresholds (both go through data._threshold_setup), so the interval is for
# the statistic the panel plots. Its intervals are WIDE wherever the calls are few: a
# precision at a fixed threshold rests only on the called windows, a few hundred in the
# lowest GC bin against the tens of thousands behind panel C.
lifts_s8 = D.lift_deltas(threshold=D.GNOCCHI_THRESHOLD, truth_set="lax",
                         n_bootstrap=500, seed=0) if NEUTRAL_WINDOWS_BED else None
""")

code(r"""
# The genome-wide table, with GC content alone as a baseline. Not a panel -- four numbers
# that settle what the UNCONDITIONAL precision-recall of this truth set is measuring. It
# prints as it computes.
budget_s8 = D.budget_comparison(threshold=D.GNOCCHI_THRESHOLD, truth_set="lax") \
    if NEUTRAL_WINDOWS_BED else None
""")

code(r"""
# The same paired comparison across CALLING RATES. The threshold statistics are
# noise-limited by the number of CALLED windows -- a few dozen in the GC-poorest bin at
# z >= 4 -- and loosening the cutoff closes the intervals as sqrt(calls) while costing
# almost nothing in effect size, because lift is nearly flat in the calling rate. This says
# whether a result holds along the range or only at one point. ~10 s per rate. Reported
# ALONGSIDE the anchored z >= 4 result, never instead of it.
sweep_s8 = D.lift_delta_sweep(call_rates=(0.01, 0.03, 0.10), truth_set="lax",
                              n_bootstrap=500, seed=0) if NEUTRAL_WINDOWS_BED else None
""")

code(r"""
# Guarded, so a run without NEUTRAL_WINDOWS_BED skips this figure rather than dying.
if curves_s8 is not None:
    # A IS TWO AXES, B IS ONE. Panel A draws one score per axes -- three GC-binned PR curves
    # on top of each other in a single frame would be unreadable -- and the two share a y
    # axis, which is what makes them comparable: both run to 3x the positive fraction, and
    # after the class balancing that fraction is the same number in both.
    # TWO ROWS, NOT ONE. A needs two axes of its own, and B and C are a pair -- C is the
    # inference on the gap B draws -- so they belong side by side beneath it, on the same
    # GC axis and at the same width. Four panels in a row would make each too narrow for a
    # two-line y label and a legend carrying names and numbers.
    # SIX COLUMNS so the last row can hold THREE panels against the upper rows' two. A
    # needs two axes of its own; B and C are a pair (C is the inference on the gap B
    # draws); D, E and F are a triple read together at one threshold. A 2-2-3 figure is
    # what that argument looks like, and a uniform grid cannot express it.
    fig = plt.figure(figsize=(13.5, 15.0))
    gs = fig.add_gridspec(3, 6, wspace=1.5, hspace=0.36)
    axA1 = fig.add_subplot(gs[0, 0:3])
    axA2 = fig.add_subplot(gs[0, 3:6], sharey=axA1)
    axB = fig.add_subplot(gs[1, 0:3])
    axC = fig.add_subplot(gs[1, 3:6])
    axD = fig.add_subplot(gs[2, 0:2])
    axE = fig.add_subplot(gs[2, 2:4])
    axF = fig.add_subplot(gs[2, 4:6])

    for ax, key in zip((axA1, axA2), D.PR_SCORES):
        panels.panel_pr_curves(ax, curves_s8, key, show_ylabel=ax is axA1)
    panels.panel_aupr_by_gc(axB, curves_s8)
    if deltas_s8 is not None:
        panels.panel_aupr_delta(axC, deltas_s8)
    if tm_s8 is not None:
        # D is log-y: published Gnocchi's calling rate spans nearly two orders of
        # magnitude across GC, and on a linear axis every bin but the last would sit on the
        # floor. E stays linear -- it is a probability read against a base rate drawn beside
        # it, and a log axis would distort that comparison. F is log-log because that is
        # what makes its iso-calling-rate contours straight.
        panels.panel_threshold_metric(axD, tm_s8, "call_rate", D.GNOCCHI_THRESHOLD)
        panels.panel_threshold_metric(axE, tm_s8, "precision", D.GNOCCHI_THRESHOLD)
        panels.panel_lift_vs_recall(axF, tm_s8, D.GNOCCHI_THRESHOLD)

    # Each letter clears its own panel's ylabel: A's is one line and its panel is wide,
    # B/C's are two lines, D-F are narrower still so their letters sit further out.
    panels.label_panels((axA1,), ("A",))
    for ax, lab in ((axB, "B"), (axC, "C")):
        panels.label_panels((ax,), (lab,), x=-0.16)
    for ax, lab in ((axD, "D"), (axE, "E"), (axF, "F")):
        panels.label_panels((ax,), (lab,), x=-0.34)

    s8_name = f"supp_fig8{config.WINDOW_SET_SUFFIX}"
    written = resave_ai.save_panel(fig, os.path.join(OUTPUT_DIR, s8_name))
    print(f"wrote {', '.join(os.path.basename(p) for p in written)}" if written
          else f"{s8_name}: unchanged, left alone")
""")

code(r"""
# Guarded for the same reason as the cell above.
if curves_s8 is not None:
    # Numbers for Supporting Figure 8's caption. Computed over the bins the panels actually
    # draw -- pr_curves() has already dropped those below the window floor -- so a caption
    # cannot quote a bin the reader cannot see.
    print("truth set: `window overlaps enhancer` (GeneHancer) from "
          f"{os.path.basename(NEUTRAL_WINDOWS_BED)}")
    print(f"positive fraction after balancing: {curves_s8['published']['r']:.4f}")
    for key, c in curves_s8.items():
        print(f"\n{c['display']}: pooled auPRC/r = {c['all']['aupr_norm']:.3f}")
        for e in c["bins"]:
            print(f"  GC ({e['lo']:.2f}, {e['hi']:.2f}]  n = {e['n']:>9,}  "
                  f"auPRC/r = {e['aupr_norm']:.3f}")

    pub, dec = curves_s8["published"]["bins"], curves_s8["scored"]["bins"]
    print("\npanel B, retrained - published, per GC bin (no uncertainty -- see panel C):")
    for a, b in zip(pub, dec):
        print(f"  GC ({a['lo']:.2f}, {a['hi']:.2f}]  {b['aupr_norm'] - a['aupr_norm']:+.3f}")

if deltas_s8 is not None:
    print("\npanel C, paired gain of the retrained score, unbalanced, 95% bootstrap CI.")
    print("A bin whose CI excludes 0 is a real difference; P is the bootstrap fraction")
    print("above 0, so it reads as a one-sided posterior-style probability, not a p-value.")
    for r in deltas_s8.iter_rows(named=True):
        star = "  *" if (r["ci_lo"] > 0 or r["ci_hi"] < 0) else "   "
        print(f"  GC ({r['lo']:.2f}, {r['hi']:.2f}]  n = {r['n']:>9,}  "
              f"pos = {r['n_pos']:>8,}  r = {r['r']:.3f}  "
              f"gain = {100 * r['delta']:+6.2f}%  "
              f"[{100 * r['ci_lo']:+6.2f}, {100 * r['ci_hi']:+6.2f}]  "
              f"P(>0) = {r['p_gt0']:.3f}{star}")

if tm_s8 is not None:
    print(f"\npanels D-F. Published held at Gnocchi >= {D.GNOCCHI_THRESHOLD:g} (Chen")
    print("et al.'s own cutoff); the retrained score takes the threshold calling the SAME")
    print("fraction of windows, so precision and recall compare like with like. Unbalanced.")
    print("call_rate uses NO labels, so it is the most robust number here.")
    for key in D.PR_SCORES:
        rows = tm_s8.filter(pl.col("score") == key).sort("mid")
        cr = rows["call_rate"].to_numpy()
        print(f"\n  Gnocchi, {rows['short'][0]} (z >= {rows['threshold_used'][0]:.3f}):  "
              f"calling rate spans {100 * cr.min():.2f}% - {100 * cr.max():.2f}%  "
              f"({cr.max() / cr.min():.2f}x across GC)")
        for r in rows.iter_rows(named=True):
            print(f"    GC ({r['lo']:.2f}, {r['hi']:.2f}]  called {r['n_called']:>7,} "
                  f"({100 * r['call_rate']:6.2f}%)  precision {r['precision']:.3f} "
                  f"[{r['precision_lo']:.3f}, {r['precision_hi']:.3f}]  "
                  f"base rate {r['r']:.3f}  lift {r['lift']:.2f}  "
                  f"recall {100 * r['recall']:6.2f}%")

if lifts_s8 is not None:
    print("\npanel E's comparison, paired bootstrap on the LIFT RATIO (= the precision")
    print("ratio: the base rate cancels, since both scores see the same rows). A bin whose")
    print("CI excludes 0 is a real difference. `ceiling` is 1/base-rate, the largest lift")
    print("attainable in that bin -- lift is comparable BETWEEN scores here, not between")
    print("bins, because the ceiling moves with the base rate.")
    for r in lifts_s8.iter_rows(named=True):
        star = "  *" if (r["ci_lo"] > 0 or r["ci_hi"] < 0) else "   "
        print(f"  GC ({r['lo']:.2f}, {r['hi']:.2f}]  "
              f"calls {r['n_called_published']:>6,} / {r['n_called_scored']:>6,}  "
              f"lift {r['lift_published']:.2f} -> {r['lift_scored']:.2f} "
              f"(ceiling {r['ceiling']:.1f})  gain {100 * r['delta']:+6.1f}% "
              f"[{100 * r['ci_lo']:+6.1f}, {100 * r['ci_hi']:+6.1f}]  "
              f"P(>0) = {r['p_gt0']:.3f}{star}")
""")


md(r"""
## Numbers for the caption

Everything quoted in the caption should come from here, not from memory.
""")

code(r"""
# Every number below is computed over the bins the panels actually plot -- the same
# min_n floor and, for panel C, the same x range. Otherwise a caption can quote a bin
# the reader cannot see.
print(f"panel A: {df_a.height:,} windows after joint z filtering, mean GC {gc_mean:.3f}")
for label, name in [("step1", "context-only (r = 1)"), ("step2", "Gnocchi as published")]:
    print(f"  mean |rank - 0.5|  {name:<34} {D.rank_bias(binned_a, label, MIN_N_WINDOWS):.3f}")
if binned_dr is not None:
    print(f"  mean |rank - 0.5|  {'depletion rank':<34} "
          f"{D.rank_bias(binned_dr, 'dr', MIN_N_WINDOWS):.3f}")

b = binned_b.filter(pl.col("n") >= MIN_N_WINDOWS)
print(f"\npanel B: r_non spans {b['r_non'].min():.3f}-{b['r_non'].max():.3f}; "
      f"r_CpG spans {b['r_cpg'].min():.3f}-{b['r_cpg'].max():.3f}; "
      f"counterfactual spans {b['r_counterfactual'].min():.3f}-"
      f"{b['r_counterfactual'].max():.3f}")
print(f"  CpG share of expected counts rises {b['pi_cpg'].min():.3f} -> {b['pi_cpg'].max():.3f}")

# Restricted to the plotted x range as well: the bins below it are almost entirely
# QC-failing sequence, so an unrestricted min() would report an analyzed fraction of 0.00
# from a bin the panel does not draw.
c = comp.filter((pl.col("n_total") >= MIN_N_SITES)
                & pl.col("gc_mid").is_between(*XRANGE)).sort("gc_mid")
print(f"\npanel C: {int(c['n_total'].sum()):,} non-CpG training sites in the plotted "
      f"range; scored-population fraction {c['frac_scored'][0]:.2f} at GC "
      f"{c['gc_mid'][0]:.2f} -> {c['frac_scored'][-1]:.2f} at GC {c['gc_mid'][-1]:.2f}")

print("\npanel D: empirical P(DNM), max / min ratio within each population")
for pop, tab in binned_d.items():
    e = tab["empirical_prop"]
    print(f"  {pop:<12} {e.max() / e.min():.2f}x   "
          f"(fitted {tab['mean_pred'].max() / tab['mean_pred'].min():.2f}x)")

print(f"\npanel E: {df_e.height:,} windows after joint z filtering")
for label, name in [("step1", "context-only (r = 1)"),
                    ("step2", "Gnocchi as published"),
                    ("scored", "Gnocchi, retrained on the filtered training set"),
                    ("full_refit", "  control: refit on the full training set"),
                    ("sizematched", "  control: size-matched random subsample")]:
    print(f"  mean |rank - 0.5|  {name:<44} {D.rank_bias(binned_e, label, MIN_N_WINDOWS):.3f}")
""")

md(r"""
## Caveats that belong in the caption

* **Panel D is in-sample.** The retrained model is fit on training sites in the analyzed
  windows and scored against the labels of those same sites. Panel E is the
  out-of-sample confirmation: it is measured on gnomAD polymorphism counts, which the
  DNM model never sees. A held-out DNM split would make panel D airtight; it has not
  been run.
* **The depletion-rank curve comes from a different window set** (Halldorsson windows, a
  different window size) than the two Gnocchi curves. They are not joined; each is
  ranked within itself, and each is binned on its own GC edges. Halldorsson's windows
  also **overlap**, so no error bar is drawn for that curve: the within-bin windows are
  not independent, which the shared `std / sqrt(n)` assumes. It reads last in the
  legend for the same reason — it is the external comparison, not a third Gnocchi curve.
* **This run used McHale et al.'s 693,270 windows**, and the caption says so.
  `NEUTRAL_WINDOWS_BED` unset would mean the 1,843,559 noncoding + `pass_qc` +
  autosome/PAR windows this repo builds from the bucket; set, as here, it means their
  693,270, taken from their file with none of those three filters applied on top —
  theirs is the definition, so it enters whole (`windows.build_window_table`). Either
  way it is one definition applied consistently: every panel, the population the
  retrained model is fit on, and panel C's bottom band, which is membership in that
  population rather than a re-derivation of its filters. The gap between the two is
  2.66x and is not only enhancer-overlapping windows — their assembly-gap,
  ENCODE-exclude and low-coverage exclusions are in it too — so a result that holds on
  only one of them is a result about the window definition.

  Switching recomputes everything: A, B and E directly, D through the `scored` refit
  (`sizematched` is no longer drawn there, but panel E still reports it), and C through
  both its bottom band and the shared GC bin edges,
  which span the window set's own GC range. Two operational costs. The refits must be
  rerun (`config.check()` refuses one stamped with a different value, naming the
  command), and since they are keyed by population alone, one window set's refits
  overwrite the other's — switching back means rerunning again, ~6 min each. The panel-C
  and CpG caches in `output/` are keyed by a fingerprint of the edges and the window
  set, so those two do coexist.
""")

md(r"""
## Refresh the Illustrator assembly

The manuscript figure is assembled in `fig5.ai` from the panel PDFs above, as **links**,
and `fig5.png` is exported from that assembly — the readable copy, since an .ai renders as
nothing on GitHub. Any panel this run rewrote therefore leaves both stale, and the repo --
which tracks them and is the source of truth for both -- would carry an assembly that
disagrees with its own panels until Illustrator reloads the links, the document is saved,
and the PNG is exported again.

The cell below does all three, so a notebook run ends with everything in step. It relinks
only the panels whose content the .ai does not already hold — judged by hash against
`fig5.ai.links.json`, not by mtime — and prints each one: relinking preserves a placed
item's frame rather than its aspect ratio, so a panel whose bounding box moved is worth a
look. A run that changed no artwork has nothing to relink, because `save()` above left
those files untouched. Nothing here is required to build the figure -- without Illustrator, or without the
.ai, it prints a notice and moves on -- and the neutral window set's assembly (`fig5.neutral.ai`)
does not exist until someone builds it, which is a notice too. `fig5/resave_ai.py` is the
same code as a command; `-suffix .neutral` picks the other assembly.
""")

code(r"""
_ = resave_ai.refresh(quiet_if_absent=True,        # the return code is for the shell, not here
                      suffix=config.WINDOW_SET_SUFFIX)
""")


nb = {
    "cells": [
        {"cell_type": k, "metadata": {},
         **({"source": v.splitlines(keepends=True), "outputs": [], "execution_count": None}
            if k == "code" else {"source": v.splitlines(keepends=True)})}
        for k, v in CELLS
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

# Beside this file. (Was two lines: a wrong os.path.join -- it landed in the repo root --
# immediately overwritten by a hardcoded absolute path from one machine.)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig5.ipynb")
with open(out, "w") as fh:
    # ensure_ascii=False to match what nbconvert --inplace writes back after executing.
    # Without it every em dash in the prose lands as \u2014 here and as itself there, so
    # a regenerate-then-execute shows several hundred lines of diff that are not changes.
    json.dump(nb, fh, indent=1, ensure_ascii=False)
print("wrote", out, f"({len(CELLS)} cells)")
