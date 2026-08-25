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
                            # claim (n = 356 and 169); error bars carry the uncertainty
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
windows overlap — roughly 38.6M of them over a 3.1 Gb genome — so within-bin windows are
not independent and $\mathrm{std}/\sqrt{n}$ would understate the uncertainty by about
$\sqrt{\text{length}/\text{step}}$. The mean curve is unaffected, and it is the only
thing read off this curve.

Summary statistic quoted in the text: $\;\overline{|\overline{\rho}_M(g)-0.5|}\;$ across bins.

**Why this statistic is so sensitive to $r$** — worth recording, because it is what
licenses reading panels B–D as the *cause* of panel A. Suppose a window's adjustment is a
uniform inflation, $E_2 = f E_1$, and write the observed count as $O = E_1(1+\epsilon)$.
Then

$$z_2 - z_1 = \sqrt{E_1}\left[\frac{f-1-\epsilon}{\sqrt{f}} + \epsilon\right]
\;\approx\; \sqrt{E_1}\,(f-1) \qquad (f \to 1,\ \epsilon \to 0).$$

The $\sqrt{E_1}$ prefactor is the point. A typical analyzed window has $E_1 \approx 174$,
so $\sqrt{E_1} \approx 13$, and a mere 10% inflation displaces $z$ by more than one unit —
comparable to the spread of $z$ itself. Measured directly by re-ranking the real
genome-wide $z$ distribution under a uniform $f$, the mean rank moves from 0.500 at
$f = 1$ to 0.687 at $f = 1.10$ and 0.302 at $f = 0.90$. Small multiplicative errors in $r$
are not a second-order concern; they are the whole effect.
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
    panels.curve_from_binned(binned_a, "step1", "step1", r"Context-only model ($r \equiv 1$)"),
    panels.curve_from_binned(binned_a, "step2", "step2", "Gnocchi as published"),
]
if binned_dr is not None:
    # show_se=False: Halldorsson's windows OVERLAP, so std/sqrt(n) understates the
    # uncertainty and the bar cannot be defended beside two curves of disjoint 1 kb tiles.
    curves_a.append(panels.curve_from_binned(binned_dr, "dr", "dr",
                                             "Depletion rank (Halldorsson windows)",
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
just 55.5–345.7 across the analyzed set, and switching to the unweighted mean would move
$R_{\mathrm{eff}}$ by ~0.1% through the GC bulk and 0.79% at worst, in the sparsest
low-GC bin. **Case carries the level throughout this notebook:** lowercase
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
$R_{\mathrm{eff}}$ (violet, what Gnocchi applies), $R_{\mathrm{non}}$ (orange) and
$R_{\mathrm{CpG}}$ (green) — the two terms it decomposes into — and the dashed grey
counterfactual derived next. The y-axis is $R(g)$ itself, a ratio of summed expected
counts, so 1 means *no adjustment* and the horizontal line at 1 is the null. Since
$R_{\mathrm{eff}}$ is a $\Pi$-weighted average of the other two, it must lie between
them: it tracks $R_{\mathrm{non}}$ closely at low GC, where $\Pi$ is small, and is pulled
down towards $R_{\mathrm{CpG}}\approx1$ as $\Pi$ grows.

**The counterfactual, and what it intervenes on.** The question is how much of
$R_{\mathrm{eff}}$'s rise the CpG contexts could account for on their own. So switch off
the *non-CpG* adjustment — set $r_t\equiv1$ for $t\notin\mathcal K$, which sends
$E_2^{\neg\mathcal K}\!\to\!E_1^{\neg\mathcal K}$ — and change nothing else: the fitted
$r_t$ for $t\in\mathcal K$ stay, the weights $\Pi(g)$ stay. The dashed grey curve is

$$R_{\mathrm{eff}}\big|_{r_t\equiv1,\;t\notin\mathcal K}(g)
=\frac{\sum_g E_2^{\mathcal K}+\sum_g E_1^{\neg\mathcal K}}{\sum_g E_1}
=\Pi(g)\,R_{\mathrm{CpG}}(g)+\big(1-\Pi(g)\big),$$

i.e. **what Gnocchi would apply if it adjusted CpG contexts alone**. It is flat within
0.6% across the whole GC range while $R_{\mathrm{eff}}$ climbs to 1.44, so *none* of the
applied trend survives the removal of the non-CpG term: the GC dependence of what Gnocchi
applies is wholly non-CpG.

Flatness here is a result, not an identity. $\Pi$ reaches **0.43** at high GC, so a GC
trend in $R_{\mathrm{CpG}}$ would appear in this curve scaled by $\Pi$, not erased — the
curve is flat because $R_{\mathrm{CpG}}$ itself is (0.98–1.00), which is the next
paragraph's subject and is *correct* rather than a failure.

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
is what makes it a check rather than a restatement: the two agree to $10^{-4}$ per bin,
and that is what licenses using the refit's per-context $r$ above — the published pipeline
writes its own only to a local directory, never to the bucket.

**Why $R_{\mathrm{CpG}}\approx1$ is correct, not a failure.** CpG mutability is dominated
by methylation, and $p_c$ is already keyed by methylation level — across methylation 0 to
15 the CpG C>T rate spans **3.0–4.3×** depending on context, the largest single rate
effect in the model, inside one trinucleotide. High-GC CpGs are CpG islands: **90–100%
hypomethylated above GC 0.70**, against ~2% in the GC bulk, and their empirical DNM rate
falls from 0.53 in the bulk to 0.19 in the top GC bin (**2.8×**). **Step 1 has already
applied that correction**,
via the covariation of GC content with methylation, so there is nothing left for $r$ to
adjust. (Chen et al. also strip `GC_content`, `CpG_island`, `Nucleosome`, `SINE` and
`met_sperm` from CpG-context models, so those models could not express a GC dependence
even if one were needed.) The GC trend in what Gnocchi applies is therefore entirely
non-CpG — which is where the rest of this figure looks.
""")

code(r"""
binned_b = D.r_eff_by_gc(df_win, edges, pop="full", cache_dir=CACHE_DIR)

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

**The numbers below are from the unset run** — the one this notebook's panels were built
from. Under the narrowed one, read them from the table printed under the code cell,
since the scored band is then a subset and the ratios divide by it.

**Why both classes.** In a case-control design it is the controls that carry the
covariate distribution, which argues for stacking the background class alone — but that
is an argument about the *design*, not the *fit*. The fit minimizes its loss over the
mixture, so the mixture is the training distribution, and that is what belongs beside the
scored population. Stacking the background class alone barely moves the picture (the
QC-fail band runs 0.13 → 0.24 instead of 0.14 → 0.28) because it outnumbers the DNMs
about 12:1; what the DNM class adds is its own steeper drift, 64% QC-fail at GC 0.64
against the background class's 26%.

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

The QC-pass noncoding share of the training set falls from ~0.82 in the GC bulk to under
0.28 by GC 0.68. So the model is fit on one population and applied to another, and the
two come apart exactly where panel A's bias is largest.

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
the scored ones as far as mutation rate goes. The coding stratum sits there — within
~10%, i.e. $|\log|\lesssim0.1$, flat across the whole range — while the QC-failing
stratum runs 1.55× ($\log = 0.44$) in the GC bulk and **4.06× ($\log = 1.40$) by
GC 0.61**. Under the narrowed population, read the putatively nonneutral curve
first: flat at 0 across the range, the narrowing costs sample size and nothing else and the
rest of this figure carries over; climbing with GC, it is a third population change and
belongs in the caption beside the other two.

Both rows come from one table of per-stratum counts — this row divides the DNM count by
the site count, the row above takes the site count as a share of its bin. 72,801 of the
non-CpG autosomal DNMs fall in the QC-failing stratum, against 17,545 coding and 241,479
in QC-pass noncoding windows.

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
$\widehat p(g)-\bar y(g)$ is a calibration gap conditioned on GC, and each pair is a
reliability diagram on its own population: *fitted vs empirical within a pair* is the
comparison that means something.

Two cautions on what $P_g$ is *not*. It is a probability under the **training**
distribution, not the genome's: every DNM is kept while background sites are sampled at
some rate $s$, which multiplies the odds by $1/s$ and so shifts $\operatorname{logit}P_g$
by $-\log s$ — a constant in $x$, and in particular in GC. Levels are therefore not
comparable **across** populations, whose case-control ratios differ (12.2 vs 13.5
background sites per DNM), while shapes in $g$ are; each curve is divided by its own
site-weighted mean, leaving shape only. And $g$ enters only as a conditioning event: GC
is one of the 13 candidate regional features, but $\widehat p(g)$ averages whatever the
fit selected for that context, so a pair can come apart in a bin even where GC itself was
never selected.

Three populations:

* **original** — the training set as published;
* **scored** — restricted to the analyzed windows (the intervention);
* **size-matched** — the same *number* of sites as *scored*, drawn at random from the
  whole genome. This is the control that separates "better-matched population" from
  "less data", and it lies on top of the original pair.

Two things happen at once when the training set is restricted. The **empirical** GC
dependence itself shrinks and becomes monotone — on the original set $P(\mathrm{DNM})$
rises 2.4× and then *collapses* above GC 0.66; on the scored set it rises smoothly by
1.57× with no turnover. And the logistic regression can then actually fit it: on the
original set it misses by 26% and 29% *in opposite directions*, because it is a smooth
monotone surface chasing a curve that turns over.

This is why $R_{\mathrm{non}}$ in panel B is inflated — the model has partly learned a GC
slope that belongs to sequence outside the scored population.

**Caveat.** This panel measures a *level* error in $P(\mathrm{DNM})$, and levels cancel in
$r$. It diagnoses the fit; it is not itself a measurement of Gnocchi's bias. Panel E is.
""")

code(r"""
binned_d = D.dnm_probability(("full", "scored", "sizematched"), n_bins=N_BINS,
                             min_n=MIN_N_SITES)

fig, ax = plt.subplots(figsize=FIGSIZE)
panels.panel_dnm_probability_pairs(ax, binned_d, min_n=MIN_N_SITES, normalize=True)
save(fig, "D")

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
removal of a defect.
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
    panels.curve_from_binned(binned_e, "step2", "step2", "Gnocchi as published"),
    panels.curve_from_binned(binned_e, "step1", "step1", r"Context-only model ($r \equiv 1$)"),
    panels.curve_from_binned(binned_e, "scored", "scored",
                             "Gnocchi, $r$ retrained on the\n"
                             "training set restricted to scored population"),
]

fig, ax = plt.subplots(figsize=FIGSIZE)
# Lower left: E's third label wraps to two lines, and at the panel's type size that
# second line runs through the published curve if the legend stays top-left as in A.
panels.panel_rank_bias(ax, curves_e, gc_mean=float(df_e["GC_content"].mean()),
                       xrange=XRANGE, min_n=MIN_N_WINDOWS, legend_loc="lower left")
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
~2.5% through the GC bulk and **90–100% above GC 0.70**; mean methylation falls from ~6.5
to near zero over the same range.

**C. So their DNM rate collapses.** Flat at ~0.53 through the bulk, falling to **0.195**
in the top GC bin — a 2.7× fall, tracking B. The top two bins hold 356 and 169 sites;
their error bars carry that.

**D. And these contexts are not a rounding error.** $\Pi$, the CpG share of a bin's step-1
expected counts — the same weight panel B's identity
$R_{\mathrm{eff}}=\Pi R_{\mathrm{CpG}}+(1-\Pi)R_{\mathrm{non}}$ uses — rises from
**0.025 to 0.426** across the GC range. So the flatness of panel B's counterfactual is a
measurement and not an artifact of negligible weight: had $R_{\mathrm{CpG}}$ carried a GC
trend, it would have reached the applied multiplier scaled by up to 0.43. This is the
panel that says the A–C mechanism *matters* rather than merely *holds*.

Put together: a large, strongly GC-dependent CpG effect exists, step 1 already applies it,
and the contexts it applies to carry up to 43% of the expected counts. There is nothing
left for $r$ to correct, which is exactly what panel B measures. (Level, not rate: these
are case-control-sampled training sites at ~10:1, so the y-axis of C is not a genome-wide
mutation rate.)

D is binned over **Chen windows** and A–C over **training sites**, the same split the main
figure runs on, so the two do not end in the same place: D's last bin above the 100-window
floor is centred at GC 0.75 against 0.78 for B and C. Note also that D is drawn on this
figure's wider 0.2–0.8 axis, so it shows the two bins (0.71 and 0.75, where $\Pi$ reaches
0.38 and 0.43) that panel B's own 0.2–0.73 range cuts off — the bins where the weight is
largest are exactly the ones the main panel cannot show.

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

fig, axes = plt.subplots(4, 1, figsize=(7.0, 16.5),
                         gridspec_kw={"height_ratios": [1, 1, 1, 1], "hspace": 0.32})
panels.panel_cpg_methylation_effect(axes[0], ct)
# MIN_N_CPG, not MIN_N_SITES: the top two GC bins (n = 356 and 169) carry the claim,
# and their error bars show how thin they are. Quote numbers from this same subset.
panels.panel_cpg_hypomethylation(axes[1], cpg, min_n=MIN_N_CPG, show_xlabel=False)
panels.panel_cpg_dnm_rate(axes[2], cpg, min_n=MIN_N_CPG, show_xlabel=False)
# binned_b and MIN_N_WINDOWS, not cpg/MIN_N_CPG: Pi is a per-window weight, and this is
# the same table and the same bin floor panel B decomposes, so the curve here IS the
# weight in that identity rather than a second estimate of it.
panels.panel_cpg_expected_share(axes[3], binned_b, min_n=MIN_N_WINDOWS)
panels.label_panels(axes, ("A", "B", "C", "D"))

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
* **Which window set this run used is printed by the config cell**, and the caption must
  say which. `NEUTRAL_WINDOWS_BED` unset means the 1,843,559 noncoding + `pass_qc` +
  autosome/PAR windows this repo builds from the bucket; set, it means McHale et al.'s
  693,270, taken from their file with none of those three filters applied on top —
  theirs is the definition, so it enters whole (`windows.build_window_table`). Either
  way it is one definition applied consistently: every panel, the population the
  retrained model is fit on, and panel C's bottom band, which is membership in that
  population rather than a re-derivation of its filters. The gap between the two is
  2.66x and is not only enhancer-overlapping windows — their assembly-gap,
  ENCODE-exclude and low-coverage exclusions are in it too — so a result that holds on
  only one of them is a result about the window definition.

  Switching recomputes everything: A, B and E directly, D through the `scored` and
  `sizematched` refits, and C through both its bottom band and the shared GC bin edges,
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
    json.dump(nb, fh, indent=1)
print("wrote", out, f"({len(CELLS)} cells)")
