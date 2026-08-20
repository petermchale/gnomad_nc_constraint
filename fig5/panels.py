"""
The five panels of Fig. 5, as ax-accepting functions. Nothing here creates a figure,
sets a matplotlib backend, or writes a file -- fig5.ipynb owns all of that.

Every panel's x-axis is the GC content of a 1 kb window, on a 0-1 fraction scale, but
over two populations: A/B/E bin Chen windows by their own GC, C/D bin DNM training
SITES by the GC_content_1k regional feature at that site. Same quantity, two
populations; the caption should say so.

COLOUR IS CONSISTENT ACROSS THE FIGURE, which is the point of keeping the panels in
one module: blue = the context-only model or an empirical measurement, orange = the
published pipeline, violet = the scored-population intervention, aqua = an
independent metric (depletion rank), grey = a control or a hypothetical.
"""
import matplotlib.ticker as mticker
import numpy as np

# Categorical slots of the validated default palette (dataviz skill,
# references/palette.md). The aqua's contrast against the surface is 2.74:1, a WARN
# that obligates relief -- discharged by the legend plus a distinct marker per
# series, so identity is never carried by colour alone.
SERIES_COLORS = {"step1": "#2a78d6", "step2": "#eb6834",
                 "dr": "#1baf7a", "scored": "#4a3aa7", "control": "0.55"}
SERIES_MARKERS = {"step1": "o", "step2": "s", "dr": "^", "scored": "D", "control": "v"}

GRID_KW = {"color": "0.85", "linewidth": 0.6}
REF_LINE_KW = {"color": "0.45", "linewidth": 0.8, "linestyle": "--"}
# Sized for the figure as it appears in the manuscript, where each panel PDF is placed
# at roughly half a page width -- at the previous 12/11/10 the tick labels were the first
# thing to become unreadable there. Every panel reads these three, so the figure cannot
# drift into mixed type sizes; a panel that needs a smaller legend takes it relative
# (LEGEND_FONTSIZE - 1), never as its own literal.
AXIS_LABEL_FONTSIZE = 15
TICK_LABEL_FONTSIZE = 13
LEGEND_FONTSIZE = 12


def _finish(ax, ylabel, xrange, show_xlabel, legend_loc="upper left",
            legend_fontsize=LEGEND_FONTSIZE, grid_axis="both", handles=None) -> None:
    """
    The frame every panel shares: range, labels, grid, despined box, legend.

    `handles` overrides the legend's order, which otherwise follows the order things were
    drawn in. Draw order is not free -- a stack has to be drawn bottom-up, and a curve
    drawn later sits on top -- so a panel whose legend should read in the order the
    reader sees things on the page passes its handles here instead of reordering its
    drawing.
    """
    ax.set_xlim(xrange)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
    if show_xlabel:
        ax.set_xlabel("GC content", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(True, axis=grid_axis, **GRID_KW)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if handles is not None:
        ax.legend(handles, [h.get_label() for h in handles],
                  fontsize=legend_fontsize, frameon=False, loc=legend_loc)
    else:
        ax.legend(fontsize=legend_fontsize, frameon=False, loc=legend_loc)


def _log_ratio_axis(ax, values, ticks=(0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5),
                    pad_lo: float = 0.96, pad_hi: float = 1.05) -> None:
    """
    r is a ratio, so a log axis is both available (strictly positive) and correct: a
    25% over- and a 25% under-adjustment should read as equal departures from r = 1,
    which a linear axis would not show.
    """
    ax.set_yscale("log")
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    ax.set_yticks([t for t in ticks if lo * pad_lo <= t <= hi * pad_hi])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_ylim(lo * pad_lo, hi * pad_hi)


def curve_from_binned(binned, label: str, key: str, display: str) -> dict:
    """One column of a binned_rank_curves() table as the dict panel_rank_bias wants."""
    return {"key": key, "display": display,
            "gc": binned["gc_mid"].to_numpy(), "mean": binned[f"mean_{label}"].to_numpy(),
            "se": binned[f"se_{label}"].to_numpy(), "n": binned["n"].to_numpy()}


def panel_rank_bias(ax, curves: list[dict], gc_mean: float | None = None,
                    xrange=(0.2, 0.73), yrange=(0.0, 1.0), min_n: int = 100,
                    show_xlabel: bool = True, legend_loc: str = "upper left") -> None:
    """
    Panels A and E. Mean standardized rank of each constraint metric per GC bin.

    y = 0.5 is where an unbiased metric sits: the rank is uniform on (0,1) by
    construction, so a metric with no GC-dependent bias has conditional mean rank 0.5
    in every bin. Error bars are the within-bin standard error of the mean rank, so
    they narrow where a bin holds many windows and widen in the sparse GC tails.

    min_n drops bins thinner than that many windows, where the mean is noise.
    yrange defaults to the full (0,1) of Fig. 2A, so the panels are read on that scale.
    """
    for c in curves:
        keep = c["n"] >= min_n if min_n else np.ones_like(c["n"], dtype=bool)
        ax.errorbar(c["gc"][keep], c["mean"][keep], yerr=c["se"][keep],
                    marker=SERIES_MARKERS[c["key"]], color=SERIES_COLORS[c["key"]],
                    markersize=5, linewidth=2, capsize=3, elinewidth=1, label=c["display"])
    ax.axhline(0.5, **REF_LINE_KW)
    if gc_mean is not None:
        ax.axvline(gc_mean, color="0.45", linewidth=0.8)
    ax.set_ylim(*yrange)
    # legend_loc is a parameter because A and E carry different numbers of curves in the
    # same frame: A's two fit above the rising published curve, E's three do not -- its
    # third label wraps to a second line that runs straight through that curve. The rank
    # axis is fixed to (0,1) while no curve goes below ~0.27, so the bottom of the panel
    # is empty by construction and is where a legend goes when the top is full.
    # Panels A and E share this label, as they share the statistic and the axes -- two
    # names for one quantity across two panels of one figure would read as two quantities.
    _finish(ax, "Constraint metric (rank)", xrange, show_xlabel, legend_loc=legend_loc)


def panel_r_eff(ax, binned, min_n: int = 100, xrange=(0.2, 0.73),
                show_xlabel: bool = True) -> None:
    """
    Panel B. The adjustment Gnocchi actually applies to a GC bin, R_eff(g) = sum E2 /
    sum E1, split into its CpG and non-CpG parts. `binned` is data.r_eff_by_gc() output.

    Every label is a symbol from the notebook's panel B derivation, and the legend is
    ordered as that derivation reads: the applied quantity first, then the two parts it
    decomposes into, then the hypothetical.

        R_eff = Pi*R_CpG + (1-Pi)*R_non          exact, bin by bin

    THE COUNTERFACTUAL IS AN INTERVENTION ON THE NON-CpG TERM, not on the CpG one: it
    sets r_t = 1 for non-CpG contexts and leaves the fitted CpG r_t and the weights Pi
    untouched, giving Pi*R_CpG + (1-Pi) -- what Gnocchi would apply if it adjusted CpG
    contexts alone. Drawn as a dashed grey hypothetical rather than a fourth measured
    series. Its flatness is the claim, and it is not automatic: Pi reaches 0.43 at high
    GC, so a GC trend in R_CpG would show up here scaled by Pi rather than erased.
    """
    df = binned.to_pandas() if hasattr(binned, "to_pandas") else binned
    df = df[df["n"] >= min_n].sort_values("gc_mid") if min_n else df.sort_values("gc_mid")

    for col, key, label in [
            ("r_eff", "scored", r"$R_{\mathrm{eff}}$ — all contexts, what Gnocchi applies"),
            ("r_non", "step2", r"$R_{\mathrm{non}}$ — non-CpG contexts"),
            ("r_cpg", "dr", r"$R_{\mathrm{CpG}}$ — CpG contexts")]:
        ax.plot(df["gc_mid"], df[col], marker=SERIES_MARKERS[key],
                color=SERIES_COLORS[key], markersize=5, linewidth=2, label=label)
    ax.plot(df["gc_mid"], df["r_counterfactual"], linestyle="--", linewidth=1.8,
            color="0.45",
            label=r"$\Pi R_{\mathrm{CpG}} + (1-\Pi)$ — if only CpG were adjusted")

    ax.axhline(1.0, **REF_LINE_KW)
    _log_ratio_axis(ax, df[["r_non", "r_eff", "r_cpg"]].to_numpy())
    _finish(ax, "Regional adjustment", xrange, show_xlabel)


# Panel C's two rows share one colour per stratum, defined once here so the band in the
# upper row and the line in the lower row cannot drift apart.
#
# `scored` is deliberately neutral grey for two reasons: it is the reference the lower
# row divides by, and it is the one stratum with no line there, so a saturated hue would
# promise a curve that does not exist. That also puts the colour on the bands whose
# growth is the point.
#
# NOT violet for `failed_qc`, though it reads well here: violet already carries the
# scored-population intervention through panels B, D and E, and a fourth meaning would
# undo that thread. Aqua is the slot the bottom band vacated, so it collides only with
# panel A's depletion-rank curve -- a different panel with its own legend, and not part
# of a thread that runs across several. `other_noncoding` takes the palette's blue on the
# same reasoning: blue means the context-only model in panels A and E, but nothing in
# panel C does, and blue is the last categorical slot that is not load-bearing across
# panels.
#
# No "Excluded:" prefix, though both lower strata are indeed outside the scored
# population. The word needs an antecedent the legend does not supply, and it papers over
# a difference: a coding window HAS a published Gnocchi score and is dropped by this
# analysis's noncoding restriction, while a QC-failing window was dropped by Chen et al.
# before scoring and has none. Naming each stratum by the two properties that define it --
# did the window pass QC, does it overlap coding exons -- says more in fewer words.
#
# The labels also stop `failed_qc` from reading as "the noncoding remainder". It is
# a MIXTURE: 40,509 of the 587,902 QC-fail autosomal windows overlap coding exons (6.9%),
# against 7.1% among the QC-pass ones, so QC failure is close to independent of coding
# status. Site-weighted, which is what the band counts, 5.8% of its sites sit in
# coding-overlapping windows. Only the QC-pass strata are separated by coding status;
# `failed_qc` deliberately is not, because `coding_prop` comes from the constraint table
# and these windows have no row in it (measured here from Chen et al.'s own upstream
# input, misc/genome_1kb_coding_exons.txt, in preconditions/verify_qc_filter.py).
#
# `failed_qc`, not `no_coverage`: a window absent from the published constraint table
# failed one of the paper's three QC conditions (>= 80% of observed variants PASS, mean
# coverage 25-35x, >= 1000 possible variants), and it is overwhelmingly the first of
# those, not missing coverage -- 417,097 of the 587,902 absent autosomal windows fail the
# PASS rule against 19,396 failing coverage. preconditions/verify_qc_filter.py measures it.
#
# The bottom band is named for what it IS -- the population Gnocchi is scored on and the
# retrained model is fit on -- rather than for the filters that define it, because those
# filters change: with NEUTRAL_WINDOWS_BED supplied it is McHale et al.'s 693,270
# neutral windows, and a legend reading "QC-pass noncoding windows" would then be
# quietly wrong. The bands above
# it keep naming their filter, since each one IS a reason for exclusion.
STRATUM_COLORS = {"scored": "0.78", "coding": "#eb6834",
                  "other_noncoding": "#2a78d6", "failed_qc": "#1baf7a"}
STRATUM_LABELS = {
    "scored": "In the scored population",
    "coding": "In QC-pass coding windows",
    "other_noncoding": "In other QC-pass noncoding windows",
    "failed_qc": "In QC-fail windows",
}
# Bottom to top, matching data._STRATA: the scored population, then each kind of territory
# outside it. `other_noncoding` is empty unless config.NEUTRAL_WINDOWS_BED is set, and an
# empty stratum is dropped at draw time rather than conditioned on here -- the panel
# functions take a table, not a configuration.
COMPOSITION_STYLE = [
    (f"frac_{key}", STRATUM_COLORS[key], STRATUM_LABELS[key])
    for key in ("scored", "coding", "other_noncoding", "failed_qc")
]


def panel_training_composition(ax, comp, min_n: int = 500, xrange=(0.2, 0.73),
                               show_xlabel: bool = True,
                               scored_note: str | None = None) -> None:
    """
    Panel C, upper row. Where the non-CpG training sites actually sit, as a stacked
    composition per GC bin. `comp` is data.training_composition() output; the fractions
    partition each bin by construction, so the stack fills to 1.

    `scored_note` names, parenthetically in the legend, WHICH population the bottom band
    is -- "QC-pass noncoding" or "McHale et al.'s neutral set". The band is defined by
    membership in the analyzed window table, so its meaning changes with
    config.NEUTRAL_WINDOWS_BED while its label would not; a reader of the panel alone
    cannot tell the two apart, and the composition means something different in each. The
    caller supplies it because this module takes a table, never a configuration.

    Both training classes are counted, DNMs and background sites alike: the fit minimizes
    its loss over the mixture, so the mixture is the training distribution this row is
    comparing against the scored one.

    The training and scored populations coincide in the GC bulk and come apart in the
    GC-rich tail, so a model fit on the former and applied to the latter extrapolates
    exactly where panel A's bias is largest.
    """
    df = comp.to_pandas() if hasattr(comp, "to_pandas") else comp
    if min_n:
        df = df[df["n_total"] >= min_n]
    # Clip to the plotted range rather than letting stackplot interpolate in from an
    # out-of-range bin: the lowest-GC bins are almost entirely uncovered sequence,
    # which would draw a steep edge artifact at x = xrange[0] where there is no bin.
    df = df[(df["gc_mid"] >= xrange[0]) & (df["gc_mid"] <= xrange[1])].sort_values("gc_mid")

    # A stratum with no sites in any plotted bin gets no band and no legend entry --
    # `other_noncoding` whenever config.NEUTRAL_WINDOWS_BED is unset. Drawn, it would be an invisible
    # zero-height band with a legend swatch promising territory that does not exist.
    style = [(c, col, lab) for c, col, lab in COMPOSITION_STYLE
             if c in df.columns and float(df[c].max()) > 0]
    if scored_note:
        style = [(c, col, f"{lab} ({scored_note})" if c == "frac_scored" else lab)
                 for c, col, lab in style]

    # alpha=1: the lower row draws these same colours as solid lines, and any
    # transparency here would make the band read as a slightly different hue.
    bands = ax.stackplot(df["gc_mid"], *[df[c] for c, _, _ in style],
                         colors=[c for _, c, _ in style],
                         labels=[lab for _, _, lab in style], alpha=1.0)
    ax.set_ylim(0, 1)
    # A stack is drawn bottom-up, so the legend reads bottom-up unless reversed -- and a
    # reader matching swatch to band scans the plot top-down. Reversed, the two orders
    # agree and the legend can be read straight down the stack.
    _finish(ax, "Fraction of non-CpG\ntraining sites", xrange, show_xlabel,
            legend_loc="lower left", grid_axis="y", handles=bands[::-1])


# Exactly the colours and labels of the composition stack above, taken from the same
# dicts rather than repeated, so the band and the line for a stratum always match.
STRATUM_STYLE = {
    key: {"color": STRATUM_COLORS[key], "marker": marker, "label": STRATUM_LABELS[key]}
    for key, marker in (("coding", "s"), ("other_noncoding", "o"), ("failed_qc", "v"))
}


def panel_stratum_ratios(ax, ratios, xrange=(0.2, 0.73), show_xlabel: bool = True) -> None:
    """
    Panel C, lower row. Each excluded stratum's non-CpG P(DNM) relative to the scored
    population's, per GC bin. `ratios` is data.stratum_ratios() output.

    THE CLAIM, and why the two rows belong together. The row above shows that the
    training set leaves the scored population at high GC; on its own that is only an
    absence. This row shows that the part of the training set lying outside the scored
    population has a DIFFERENT DNM RATE: the QC-pass coding stratum tracks the scored
    population within ~10% and flat across the whole GC range, while the QC-fail stratum
    runs 1.55x in the GC bulk and 4.06x by GC 0.61.

    The QC-fail stratum mixes coding and noncoding windows (6.9% of it overlaps coding
    exons, against 7.1% of the QC-pass windows), so its excess is not the coding effect
    arriving by another route -- and the coding curve beside it is what shows that
    directly. So the steep GC
    dependence the model learns comes from sequence gnomAD could not call reliably --
    which is also where trio DNM calling is least reliable, making part of that excess
    plausibly false-positive calls rather than real mutation.

    A log ratio of 0 is the reference: the excluded sites would then be exchangeable with
    the scored ones as far as mutation rate is concerned.

    WHY THE LOG, ON A LINEAR AXIS. `se_log` is by construction the standard error of
    log(ratio) -- the delta-method binomial SE, symmetric in that quantity and in no
    other -- so plotting log(ratio) against a linear axis is the one form in which the
    error bar drawn is the interval the data support, `estimate +/- se`, with no
    transformation between the two. It also puts a 2x excess and a 2x deficit at equal
    distances from the reference line, which a linear ratio axis would not. Natural log,
    matching se_log: 0.44 is the 1.55x of the GC bulk, 1.40 the 4.06x at GC 0.61.
    """
    df = ratios.to_pandas() if hasattr(ratios, "to_pandas") else ratios
    df = df.sort_values("gc_mid")
    # Whichever strata the table carries: data.stratum_ratios omits one with no bins
    # left, so an unsupplied neutral-window file costs a curve rather than a KeyError.
    drawn = [s for s in STRATUM_STYLE if f"{s}_ratio" in df.columns]
    handles = {}
    for stratum in drawn:
        style = STRATUM_STYLE[stratum]
        log_r, se = np.log(df[f"{stratum}_ratio"]), df[f"{stratum}_se_log"]
        handles[stratum] = ax.errorbar(
            df["gc_mid"], log_r, yerr=se,
            marker=style["marker"], color=style["color"], markersize=5,
            linewidth=2, capsize=3, elinewidth=1, label=style["label"])
    ax.axhline(0.0, **REF_LINE_KW)
    # Highest curve first, matching the row above, where the legend reads top-down. Which
    # stratum is highest is a result here rather than a layout fact, so it is read off the
    # data instead of hardcoded -- an other-noncoding curve near 0 would land between the
    # QC-fail and coding ones, and the legend follows it there.
    order = sorted(drawn, key=lambda s: float(np.log(df[f"{s}_ratio"]).mean()),
                   reverse=True)
    # "log fold change", not "log P(DNM) relative to ...": the reader is being asked to
    # read 0.44 as 1.55x, so the label has to say that what is logged is the ratio and
    # not the rate. Plain text at the panel's own type size, not a mathtext \\dfrac,
    # which renders smaller and in a different font from every other label here.
    _finish(ax, "log fold change of DNM rate\nrelative to scored population"
                "\n(non-CpG sites)", xrange,
            show_xlabel, handles=[handles[s] for s in order])


PAIR_STYLE = {
    "full": {"key": "step2", "label": "original training set"},
    "scored": {"key": "scored", "label": "training set restricted to scored population"},
    # Same NUMBER of sites as `scored`, drawn from the same population as `full`. It
    # should track `full`, and that it does is what rules out sample size.
    "sizematched": {"key": "control", "label": "size-matched random control"},
}


# --------------------------------------------------------- supporting CpG figure
# One hue per CpG trinucleotide context. These are a fourth axis of the figure (not a
# population, model or stratum), so they get their own qualitative set rather than
# reusing the main palette's role-coded slots.
CPG_COLORS = {"ACG": "#2a78d6", "CCG": "#eb6834", "GCG": "#1baf7a", "TCG": "#4a3aa7"}


def panel_cpg_methylation_effect(ax, ct, show_mu: bool = True) -> None:
    """
    Supporting Figure 7 (the manuscript's label), row A. The CpG C>T rate against methylation level, per context,
    each curve divided by its own value at level 0. `ct` is
    data.cpg_rate_by_methyl() output.

    PLOTTED AS A FOLD-CHANGE, not an absolute rate, for two reasons: the claim being
    visualized is a span ("3.0-4.3x across methylation 0 to 15"), which is then read
    directly off the y-axis; and it puts fitted_po and mu on one honest axis. Rescaling
    the absolute curves to share an axis would push mu above 1, which is impossible for
    the probability the axis would then be labelled with.

    Solid: `fitted_po`, the probability step 1 actually applies -- a 3.0-4.3x range
    within a SINGLE trinucleotide context, the largest single rate effect in the model,
    and the reason step 1 already absorbs the CpG-island signal that R_CpG would
    otherwise have to correct.

    Dashed: `mu`, the independent pre-saturation estimate, spanning 9.7-15.2x over the
    same range. The gap between solid and dashed is the saturation of fitted_po, which
    is a polymorphism probability and so compresses hardest where the true rate is
    highest. It is why a naive DNM-count-over-E1 ratio understates the CpG rate at high
    methylation.
    """
    df = ct.to_pandas() if hasattr(ct, "to_pandas") else ct
    for context, sub in df.groupby("context"):
        sub = sub.sort_values("methylation_level")
        color = CPG_COLORS.get(context, "0.4")
        ax.plot(sub["methylation_level"], sub["fitted_po"] / sub["fitted_po"].iloc[0],
                marker="o", color=color, markersize=4, linewidth=2, label=context)
        if show_mu:
            ax.plot(sub["methylation_level"], sub["mu"] / sub["mu"].iloc[0],
                    linestyle="--", linewidth=1.5, color=color, alpha=0.5)

    # Spans are measured from the data, never hardcoded, so the annotation cannot drift
    # from the curves it describes.
    span = lambda col: (df.groupby("context")[col].max()  # noqa: E731
                        / df.groupby("context")[col].min())
    po, mu = span("fitted_po"), span("mu")
    ax.axhline(1.0, **REF_LINE_KW)
    _log_ratio_axis(ax, np.array([1.0, float(mu.max())]),
                    ticks=(1, 1.5, 2, 3, 4, 6, 8, 10, 15, 20))
    ax.set_xlabel("Methylation level", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("CpG C>T rate, relative to\nmethylation level 0",
                  fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(f"fitted (solid) spans {po.min():.1f}-{po.max():.1f}$\\times$;  "
                 f"pre-saturation (dashed), {mu.min():.1f}-{mu.max():.1f}$\\times$",
                 fontsize=LEGEND_FONTSIZE, color="0.3")
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(True, **GRID_KW)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=LEGEND_FONTSIZE, frameon=False, loc="lower right", ncol=2)


def panel_cpg_hypomethylation(ax, cpg, min_n: int = 100, xrange=(0.2, 0.8),
                              show_xlabel: bool = True) -> None:
    """
    Supporting Figure 7, row B. The fraction of CpG training sites that are
    hypomethylated (level <= 1) against GC content, with mean methylation level on a
    right-hand axis. `cpg` is data.cpg_methylation_by_gc() output.

    High-GC CpGs are CpG islands: the hypomethylated fraction runs ~2.5% through the GC
    bulk and rises to 90-100% above GC 0.70, with mean methylation falling from ~6.5 to
    near 0 over the same range.
    """
    df = cpg.to_pandas() if hasattr(cpg, "to_pandas") else cpg
    df = df[df["n"] >= min_n].sort_values("gc_pct") if min_n else df.sort_values("gc_pct")
    gc = df["gc_pct"] / 100.0

    ax.plot(gc, df["frac_hypomethylated"], marker="o", color=SERIES_COLORS["step1"],
            markersize=5, linewidth=2, label="Hypomethylated fraction (level $\\leq$ 1)")
    ax.set_ylim(0, 1.02)
    ax.yaxis.label.set_color(SERIES_COLORS["step1"])
    ax.tick_params(axis="y", colors=SERIES_COLORS["step1"])

    rax = ax.twinx()
    rax.plot(gc, df["mean_methyl"], marker="^", color="0.45", markersize=5,
             linewidth=2, linestyle="--", label="Mean methylation level")
    rax.set_ylabel("Mean methylation level", fontsize=AXIS_LABEL_FONTSIZE, color="0.35")
    rax.tick_params(axis="y", labelsize=TICK_LABEL_FONTSIZE, colors="0.35")
    rax.spines["top"].set_visible(False)

    handles = ax.get_lines() + rax.get_lines()
    _finish(ax, "Fraction of CpG sites\nthat are hypomethylated", xrange, show_xlabel)
    # Not "upper left": the mean-methylation curve is flat and high across the whole
    # left half, so a legend there sits on top of it. Mid-left is the empty region.
    ax.legend(handles, [h.get_label() for h in handles], fontsize=LEGEND_FONTSIZE,
              frameon=False, loc="center left")


def panel_cpg_dnm_rate(ax, cpg, min_n: int = 100, xrange=(0.2, 0.8),
                       show_xlabel: bool = True) -> None:
    """
    Supporting Figure 7, row C. The empirical DNM rate over CpG training sites against GC
    content, with binomial error bars.

    It is flat at ~0.53 through the GC bulk and collapses to ~0.195 in the top GC bin --
    a 2.7x fall, tracking the hypomethylation above. This is the effect step 1 models
    (via fitted_po's methylation key) and step 2 does not need to: it is why
    R_CpG ~ 1 in panel B is correct rather than a failure.

    The level is not a genome-wide mutation rate -- these are case-control-sampled
    training sites at ~10:1, and CpG contexts are the mutable ones.
    """
    df = cpg.to_pandas() if hasattr(cpg, "to_pandas") else cpg
    df = df[df["n"] >= min_n].sort_values("gc_pct") if min_n else df.sort_values("gc_pct")
    se = np.sqrt(df["p"] * (1 - df["p"]) / df["n"])

    ax.errorbar(df["gc_pct"] / 100.0, df["p"], yerr=se, marker="o",
                color=SERIES_COLORS["dr"], markersize=5, linewidth=2, capsize=3,
                elinewidth=1, label="Empirical P(DNM), CpG contexts")
    _finish(ax, "P(DNM) in the training set\n(CpG contexts)", xrange, show_xlabel,
            legend_loc="lower left")


def panel_cpg_expected_share(ax, binned, min_n: int = 100, xrange=(0.2, 0.8),
                             show_xlabel: bool = True) -> None:
    """
    Supporting Figure 7, row D. Pi(g), the CpG contexts' share of a GC bin's step-1
    expected counts. `binned` is data.r_eff_by_gc() output -- the same table panel B
    decomposes, so this curve is literally the weight in R_eff = Pi*R_CpG + (1-Pi)*R_non.

    It is what makes R_CpG ~ 1 a finding rather than a triviality: Pi runs 0.025 in the
    lowest GC bin to 0.426 in the highest, so by the top of the GC range nearly half the
    expected counts sit in contexts the regional adjustment leaves alone. A GC trend in
    R_CpG would therefore reach the applied multiplier scaled by up to 0.43, not erased
    -- which is why the counterfactual in panel B is flat by measurement and not by
    construction.

    Drawn against the same GC axis as the rows above, so it ends earlier: Pi is binned
    over Chen windows, whose analyzed set thins out above GC 0.73, while the CpG training
    sites of rows B and C reach 0.8.
    """
    df = binned.to_pandas() if hasattr(binned, "to_pandas") else binned
    df = df[df["n"] >= min_n].sort_values("gc_mid") if min_n else df.sort_values("gc_mid")

    ax.plot(df["gc_mid"], df["pi_cpg"], marker=SERIES_MARKERS["scored"],
            color=SERIES_COLORS["scored"], markersize=5, linewidth=2,
            label=r"$\Pi$ — CpG share of step-1 expected counts")
    ax.set_ylim(0, float(df["pi_cpg"].max()) * 1.15)
    _finish(ax, "CpG share of step-1\nexpected counts, $\\Pi$", xrange, show_xlabel)


def label_panels(axes, labels=("A", "B", "C"), x: float = -0.1, y: float = 1.02) -> None:
    """Bold panel letters in axes coordinates, for figures saved as a single file."""
    for ax, letter in zip(axes, labels, strict=True):
        ax.text(x, y, letter, transform=ax.transAxes, fontsize=17,
                fontweight="bold", va="bottom", ha="right")


def panel_dnm_probability_pairs(ax, binned: dict, min_n: int = 500, normalize: bool = True,
                                xrange=(0.2, 0.76), show_xlabel: bool = True) -> None:
    """
    Panel D. Fitted and empirical P(DNM) vs GC, non-CpG contexts, for each training
    population. `binned` maps population name -> data.dnm_probability() table.

    Colour encodes the population; line style encodes fitted (solid, filled) vs
    empirical (dashed, open, binomial error bars). So each pair reads as one
    reliability diagram and the pairs are separable.

    LEVELS ARE NOT COMPARABLE ACROSS POPULATIONS. The class balance differs (12.2 vs
    13.5 background sites per DNM), which shifts P(DNM) by that factor for reasons
    unrelated to GC. normalize=True divides each curve by its own site-weighted mean,
    which removes that offset and compares SHAPE, and is what the figure needs. Within
    a pair the comparison is exact either way -- fitted and empirical come from the
    very same sites.
    """
    for name, df in binned.items():
        style = PAIR_STYLE[name]
        color = SERIES_COLORS[style["key"]]
        d = df[df["n"] >= min_n] if min_n else df
        gc = d["gc_mid"] / 100.0
        d = d[(gc >= xrange[0]) & (gc <= xrange[1])].sort_values("gc_mid")
        gc = d["gc_mid"] / 100.0

        pred, emp, se = d["mean_pred"], d["empirical_prop"], d["se"]
        if normalize:
            pred = pred / np.average(pred, weights=d["n"])
            wemp = np.average(emp, weights=d["n"])
            emp, se = emp / wemp, se / wemp

        ax.plot(gc, pred, marker=SERIES_MARKERS[style["key"]], color=color,
                markersize=5, linewidth=2, label=f"fitted, {style['label']}")
        ax.errorbar(gc, emp, yerr=se, marker="o", color=color, markersize=5,
                    linewidth=2, linestyle="--", markerfacecolor="white",
                    capsize=3, elinewidth=1, label=f"empirical, {style['label']}")

    if normalize:
        ax.axhline(1.0, **REF_LINE_KW)
    _finish(ax, "P(DNM) relative to\nits own mean\n(non-CpG contexts)" if normalize
            else "P(DNM) in the\ntraining set\n(non-CpG contexts)",
            xrange, show_xlabel, legend_fontsize=LEGEND_FONTSIZE - 1)
