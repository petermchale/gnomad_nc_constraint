"""
The five panels of Fig. 5, as ax-accepting functions. Nothing here creates a figure,
sets a matplotlib backend, or writes a file -- fig5.ipynb owns all of that.

Every panel's x-axis is the GC content of a 1 kb window, on a 0-1 fraction scale, but
over two populations: A/B/E bin Chen windows by their own GC, C/D bin DNM training
SITES by the GC_content_1k regional feature at that site. Same quantity, two
populations; the caption should say so.

IDENTITY IS CARRIED BY SYMBOL, NOT COLOUR, in the line panels. A, D and E are drawn
entirely in near-black (MONO): a curve is named by its marker shape, its fill, its
linestyle and whether it carries error bars, so every one of them survives greyscale
printing, a photocopy and colour-blind readers without the legend having to be read in
colour. Two exemptions, both deliberate:

  * PANEL C, whose stacked bands cannot be told apart by symbol at all -- there is no
    marker on a filled area -- so hue is the only channel available. It uses hue to
    carry the finding rather than to enumerate categories: green for the two QC-pass
    strata that are merely excluded, red for the QC-fail one that is genuinely
    different. See STRATUM_COLORS.
  * PANEL B's applied pair, R_eff and its counterfactual, which share one colour and one
    marker and differ only in linestyle. They are the same quantity under two worlds and
    should read as a pair; the contrast between them IS that panel. See panel_r_eff.
  * SUPPORTING FIGURE 8's precision-recall panel, whose curves are not categories at all
    but one ORDERED variable, the GC bin. Its claim is that performance departs from the
    pooled curve in OPPOSITE directions at the two ends of the GC axis, so a diverging
    blue-to-red ramp is the reading of the panel rather than a way of enumerating lines --
    and it is what McHale et al.'s published Fig. 4A does. See panel_pr_curves.

The Supporting Figure follows the same rule. Its single-series rows are monochrome --
one curve and no legend leaves a hue naming nothing -- and colour survives only in its
hypomethylation row, where two curves share an x axis and have separate y axes, and the
hue is what says which curve reads against which scale.
"""
import matplotlib
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
import numpy as np

# Categorical slots of the validated default palette (dataviz skill,
# references/palette.md). SERIES_COLORS is down to one reader, the Supporting Figure's
# hypomethylation row, where the two hues tie each curve to its own y axis; everything
# else is monochrome (see the module docstring). SERIES_MARKERS still keys panels A and
# E, where the marker is the whole of a curve's identity. Both keep the same key per
# series so the two figures agree on which quantity is which.
SERIES_COLORS = {"step1": "#2a78d6", "step2": "#eb6834",
                 "dr": "#1baf7a", "scored": "#4a3aa7"}
SERIES_MARKERS = {"step1": "o", "step2": "s", "dr": "v", "scored": "^"}

# Near-black rather than pure black: at linewidth 2 over a 0.85 grid, "0.15" keeps the
# curves dominant without the hard edge of #000, and it is the ink every monochrome
# series in A, B, D and E is drawn in.
MONO = "0.15"
# The one hue that survives in the line panels, and it is a PAIR's colour rather than a
# series' -- panel B's R_eff and the counterfactual that removes its non-CpG term.
#
# WARM AND LIGHT, not the palette's violet. The pair has to separate from two things at
# once: the black R_non and R_CpG curves, and the grey dashed R = 1 reference that the
# counterfactual lies along for the whole panel. Violet (#4a3aa7) has the hue for the
# first but not the luminance -- at 0.073 relative luminance against MONO's 0.015 it
# still reads as a dark line among dark lines, and a dashed dark line sitting on a dashed
# grey one is exactly the confusion the counterfactual needed rescuing from. This orange
# sits at 0.278, so the pair is lighter than every black curve and darker and far more
# saturated than the reference, and it holds 3.2:1 against the surface as a 2 px line.
APPLIED_COLOR = "#eb6834"
# Which of panels A and E's series are drawn with a WHITE marker face. Shape alone
# separates three black curves in the clear, but they cross near GC 0.40 with markers
# 5 pt across, and two filled black shapes there merge into one blob. Alternating fill
# splits the pair that actually overlaps: in both panels the context-only model runs
# along the two curves nearest it through the whole GC bulk, and it is the one drawn
# open. Fill is also the cue that survives being printed small, where one filled shape
# versus another is the first distinction to go.
#
# A AND E SHARE TWO GLYPHS AND MUST NOT SHARE A THIRD. They are read as a before/after
# pair on one statistic, so the square and the circle carrying the same meaning in both
# is the point; a third glyph common to both would read as a third shared series when it
# is nothing of the kind -- depletion rank is an external metric on its own windows in A,
# the retrained score is the intervention in E. Hence `dr` takes the DOWN triangle and
# `scored` the up one: distinguishable side by side on the page, and neither panel has
# to be read against the other to know which is which.
MONO_OPEN = ("step1",)

GRID_KW = {"color": "0.85", "linewidth": 0.6}
REF_LINE_KW = {"color": "0.45", "linewidth": 0.8, "linestyle": "--"}
# The vertical reference a GC panel can carry: where the population it draws actually
# sits on the axis. Every panel here divides at high GC, and the line says how far out
# in the tail that happens -- solid to distinguish it from the dashed horizontal
# references (rank 0.5, r = 1), thin and grey because it is a reference, not a series.
GC_MEAN_LINE_KW = {"color": "0.45", "linewidth": 0.8}
# Sized for the figure as it appears in the manuscript, where each panel PDF is placed
# at roughly half a page width -- at the previous 12/11/10 the tick labels were the first
# thing to become unreadable there. Every panel reads these three, so the figure cannot
# drift into mixed type sizes; a panel that needs a smaller legend takes it relative
# (LEGEND_FONTSIZE - 1), never as its own literal.
AXIS_LABEL_FONTSIZE = 15
TICK_LABEL_FONTSIZE = 13
LEGEND_FONTSIZE = 12


def _finish(ax, ylabel, xrange, show_xlabel, legend_loc="upper left",
            legend_fontsize=LEGEND_FONTSIZE, grid_axis="both", handles=None,
            legend_frame: bool = False, legend_bbox=None, legend_ncol: int = 1,
            legend: bool = True, legend_handlelength: float | None = None) -> None:
    """
    The frame every panel shares: range, labels, grid, despined box, legend.

    `handles` overrides the legend's order, which otherwise follows the order things were
    drawn in. Draw order is not free -- a stack has to be drawn bottom-up, and a curve
    drawn later sits on top -- so a panel whose legend should read in the order the
    reader sees things on the page passes its handles here instead of reordering its
    drawing.

    `legend=False` draws no legend at all. For a panel with ONE series whose y-axis
    label already names it, a legend restates the ylabel in smaller type and takes a
    corner of the artwork to do it -- the Supporting Figure's rows B, C and D. It is the
    ylabel that has to carry the identity then, so a panel that switches the legend off
    must say in its ylabel everything the legend entry said.

    `legend_bbox` (+ `legend_ncol`) puts the legend OUTSIDE the axes, as bbox_to_anchor in
    axes coordinates. For a stacked composition there is no empty region inside the axes
    to place it in -- the bands fill [0, 1] by construction -- so the only placement that
    occludes nothing is off the artwork entirely. Panels are saved with
    bbox_inches="tight", so an outside legend is included in the PDF rather than clipped.

    `legend_handlelength` lengthens the handle column. Matplotlib's default 2.0 is too
    short to fit one period of panel D's (4, 1.6) dash pattern, so a dashed series draws
    as a solid line in the legend and a dashed one on the axes -- worse than no linestyle
    cue at all. Any panel whose curves are named by their dashes has to pass this.
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
    # frameon=False everywhere except where the legend sits ON the artwork rather than on
    # white: panel C's stack is a solid field of colour, and a swatch the same colour as
    # the band under it is invisible. An opaque white box is what makes the grey scored-
    # population swatch readable, and the box costs nothing there because the legend
    # covers a uniform region of that band, not a boundary.
    frame_kw = ({"frameon": True, "facecolor": "white", "edgecolor": "0.7",
                 "framealpha": 1.0} if legend_frame else {"frameon": False})
    if legend_handlelength is not None:
        frame_kw["handlelength"] = legend_handlelength
    if legend_bbox is not None:
        frame_kw["bbox_to_anchor"] = legend_bbox
        frame_kw["ncol"] = legend_ncol
    if not legend:
        return
    if handles is not None:
        ax.legend(handles, [h.get_label() for h in handles],
                  fontsize=legend_fontsize, loc=legend_loc, **frame_kw)
    else:
        ax.legend(fontsize=legend_fontsize, loc=legend_loc, **frame_kw)


def _gc_mean_line(ax, gc_mean: float | None) -> None:
    """
    Vertical line at `gc_mean`, in the panel's own x units (GC as a 0-1 fraction).

    EACH PANEL MARKS THE MEAN OF THE POPULATION IT DRAWS, which is not one number
    across the figure: A and E bin the windows left after their joint z filter, D bins
    training SITES. On McHale et al.'s window set those means span 0.390-0.402, so the
    lines land within half a bin of each other -- close enough to read across panels,
    and not so close that quoting one number for all of them would be true. The caller
    passes the mean of the very frame it plots, so the line cannot drift from the curves;
    None draws nothing.

    PANEL B DELIBERATELY DOES NOT CALL THIS. The reason is in panel_r_eff, and it is
    about that panel rather than about this line, so read it before adding one there.
    """
    if gc_mean is not None:
        ax.axvline(gc_mean, **GC_MEAN_LINE_KW)


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


def curve_from_binned(binned, label: str, key: str, display: str,
                      show_se: bool = True) -> dict:
    """
    One column of a binned_rank_curves() table as the dict panel_rank_bias wants.

    `show_se=False` drops the error bars, and the ONE case for it is a curve whose
    windows are not independent. bin_by_gc computes se = std / sqrt(n), which assumes
    they are; Halldorsson's depletion-rank windows overlap (~38.6M of them over a 3.1 Gb
    genome), so neighbouring windows share most of their sequence, the effective sample
    size per GC bin is far below n, and the bar is understated by roughly
    sqrt(window_length / step). The mean curve is unaffected -- it is unbiased either
    way, and it is the only thing panel A reads off this curve -- but a bar we cannot
    defend, drawn beside two curves whose 1 kb windows genuinely are disjoint, invites a
    precision comparison that is not available. Better to draw none.
    """
    return {"key": key, "display": display,
            "gc": binned["gc_mid"].to_numpy(), "mean": binned[f"mean_{label}"].to_numpy(),
            "se": binned[f"se_{label}"].to_numpy() if show_se else None,
            "n": binned["n"].to_numpy()}


def panel_rank_bias(ax, curves: list[dict], gc_mean: float | None = None,
                    xrange=(0.2, 0.73), yrange=(0.0, 1.0), min_n: int = 100,
                    show_xlabel: bool = True, legend_loc: str = "upper left",
                    legend_order: list[str] | None = None,
                    show_bias: bool = False) -> None:
    """
    Panels A and E. Mean standardized rank of each constraint metric per GC bin.

    y = 0.5 is where an unbiased metric sits: the rank is uniform on (0,1) by
    construction, so a metric with no GC-dependent bias has conditional mean rank 0.5
    in every bin. Error bars are the within-bin standard error of the mean rank, so
    they narrow where a bin holds many windows and widen in the sparse GC tails.

    min_n drops bins thinner than that many windows, where the mean is noise.
    yrange defaults to the full (0,1) of Fig. 2A, so the panels are read on that scale.

    `show_bias` appends "(bias=x.xxx)" to each curve's legend label, which is the panel's
    own summary of itself: the departure from 0.5 IS the bias, and the number saying how
    large it is belongs beside the curve rather than only in the caption. It is computed
    here, from the very bins the panel draws (the `keep` mask below), so it is identical
    by construction to data.rank_bias(binned, label, min_n) and cannot drift from the
    plotted curve when min_n changes. Panel E carries it; panel A does not, and quotes
    its numbers in the caption instead.

    The word "bias" is doing definitional work it cannot do alone -- the caption has to
    say that it means the mean of |mean rank - 0.5| over the GC bins drawn. A legend
    title spelling that out was tried here and taken out: it is a formula in 11 pt type
    sitting on the artwork, where a caption sentence costs nothing and can also say which
    bins the average covers, which matters (see the min_n note above).

    `legend_order` is a list of curve keys, and it decouples the legend from draw order.
    They are not the same question: draw order decides which curve is on top where curves
    cross, and the panel's subject should not be crossed out; legend order decides how the
    panel reads, and a curve that is a different metric on a different window set belongs
    at the end of the list rather than wherever its z-order put it. Keys absent from
    `curves` are skipped, so a caller can name a curve that a run did not build.
    """
    drawn = {}
    for c in curves:
        keep = c["n"] >= min_n if min_n else np.ones_like(c["n"], dtype=bool)
        # se is None for a curve whose windows are not independent, so no bar can be
        # computed honestly -- see curve_from_binned. errorbar(yerr=None) draws the line
        # and markers exactly as for the others, which is what keeps the curves comparable.
        label = c["display"]
        if show_bias:
            label += f"  (bias={np.abs(c['mean'][keep] - 0.5).mean():.3f})"
        drawn[c["key"]] = ax.errorbar(
                    c["gc"][keep], c["mean"][keep],
                    yerr=c["se"][keep] if c["se"] is not None else None,
                    marker=SERIES_MARKERS[c["key"]], color=MONO,
                    markerfacecolor="white" if c["key"] in MONO_OPEN else MONO,
                    markeredgewidth=1.2,
                    markersize=5, linewidth=2, capsize=3, elinewidth=1, label=label)
    ax.axhline(0.5, **REF_LINE_KW)
    _gc_mean_line(ax, gc_mean)
    ax.set_ylim(*yrange)
    # legend_loc is a parameter because A and E carry different numbers of curves in the
    # same frame: A's two fit above the rising published curve, E's three do not -- its
    # third label wraps to a second line that runs straight through that curve. The rank
    # axis is fixed to (0,1) while no curve goes below ~0.27, so the bottom of the panel
    # is empty by construction and is where a legend goes when the top is full.
    # Panels A and E share this label, as they share the statistic and the axes -- two
    # names for one quantity across two panels of one figure would read as two quantities.
    handles = ([drawn[k] for k in legend_order if k in drawn]
               if legend_order is not None else None)
    _finish(ax, "Constraint metric (rank)", xrange, show_xlabel, legend_loc=legend_loc,
            handles=handles)


def panel_r_eff(ax, binned, min_n: int = 100, xrange=(0.2, 0.73),
                show_xlabel: bool = True) -> None:
    """
    Panel B. The adjustment Gnocchi actually applies to a GC bin, R_eff(g) = sum E2 /
    sum E1, split into its CpG and non-CpG parts. `binned` is data.r_eff_by_gc() output.

    Every label is a symbol from the notebook's panel B derivation, and the legend is
    ordered as that derivation reads: the applied quantity first, then the two parts it
    decomposes into, then the hypothetical.

        R_eff = Pi*R_CpG + (1-Pi)*R_non          exact, bin by bin

    THE TWO R_eff ENTRIES CARRY THEIR FORMULAE RATHER THAN A GLOSS. The panel's whole
    content is one identity and one intervention on it, and the legend is where a reader
    can be shown both at once: the applied curve is written out in full, and the
    counterfactual is written as the same expression with R_non replaced by 1, under the
    same name. Side by side, the single substitution IS the argument, which no amount of
    "what Gnocchi applies" / "if only CpG were adjusted" says as directly. Those glosses
    are in the caption, where a sentence is the right form for them. R_non and R_CpG keep
    theirs, since a formula cannot say which contexts a term is summed over.

    THE COUNTERFACTUAL IS AN INTERVENTION ON THE NON-CpG TERM, not on the CpG one: it
    sets r_t = 1 for non-CpG contexts and leaves the fitted CpG r_t and the weights Pi
    untouched, giving Pi*R_CpG + (1-Pi) -- what Gnocchi would apply if it adjusted CpG
    contexts alone. Its flatness is the claim, and it is not automatic: Pi reaches 0.43 at
    high GC, so a GC trend in R_CpG would show up here scaled by Pi rather than erased.

    IT IS DRAWN AS R_eff's TWIN, and this is the one place in the line panels where
    colour is used. It was a dashed grey line with no marker, which put it in the same
    ink as the R = 1 reference it sits on top of, underneath R_CpG, in the region of the
    panel where three curves already converge -- invisible exactly where its flatness is
    supposed to be read. It now takes R_eff's colour and R_eff's diamond and differs from
    it only in being dashed and hollow, because that is what it IS: the same applied
    quantity with the non-CpG term switched off. The gap between the two curves is then
    the whole result of the panel, drawn as one pair pulling apart rather than as a
    measured series and an unrelated grey hypothetical. R_non and R_CpG stay monochrome
    -- they are the decomposition, not the claim -- and are separated from each other by
    marker and by dash pattern, so the panel still reads in greyscale.

    NO MEAN-GC LINE HERE, though A, D and E carry one (_gc_mean_line) and uniformity
    would argue for it. It was drawn here for one commit and taken out: measured off the
    render, R_eff and R_non cross 1 at GC 0.413 against a mean at 0.393, 0.8 of a bin
    apart, and a vertical line that close to where three curves meet the horizontal
    R = 1 reference reads as the explanation for the crossing. It is not one. r = 1
    where the fitted linear predictor is zero -- at each CONTEXT's own training mean in
    its own standardized space -- and the published ft_mean_std files put the
    GC_content_1k mean anywhere from 37.5% (TAT) to 44.0% (CCC) across the 23 contexts
    that select GC, over up to 12 features of which GC is one, with coefficients of both
    signs. R_eff crosses 1 where an E1-weighted average of all that balances, which is
    nobody's pivot and is not tied to the window population's mean. (The four CpG
    contexts carry no GC, SINE, met_sperm, CpG_island or Nucleosome term at all --
    FT_CORR_MET strips them -- which is why R_CpG is flat.) The mean is also the wrong
    centre for this panel twice over: it is n-weighted over windows where R_eff is
    E1-weighted. If the point to make is that the high-GC divergence happens in a sparse
    tail -- true, and this is the one panel with no error bars to say so -- a rug or
    marginal of window GC along the bottom axis says it without nominating an x value
    for the reader to pair with the crossing. `binned` already carries n per bin.
    """
    df = binned.to_pandas() if hasattr(binned, "to_pandas") else binned
    df = df[df["n"] >= min_n].sort_values("gc_mid") if min_n else df.sort_values("gc_mid")

    # The two decomposition terms, in the figure's monochrome: marker AND linestyle, so
    # neither depends on the other surviving a greyscale conversion.
    parts = {}
    for col, marker, dash, label in [
            ("r_non", "s", (4, 1.6), r"$R_{\mathrm{non}}$ — non-CpG contexts"),
            ("r_cpg", "^", (1, 1.6), r"$R_{\mathrm{CpG}}$ — CpG contexts")]:
        parts[col], = ax.plot(df["gc_mid"], df[col], marker=marker, color=MONO,
                              markersize=5, linewidth=2, dashes=dash, label=label)
    # The applied pair: same colour, same marker, differing only in linestyle. Drawn
    # after the two parts so neither crosses out the quantity the panel is about.
    parts["r_eff"], = ax.plot(
        df["gc_mid"], df["r_eff"], marker="D", color=APPLIED_COLOR, markersize=5,
        linewidth=2,
        label=r"$R_{\mathrm{eff}} = \Pi R_{\mathrm{CpG}} + (1-\Pi)R_{\mathrm{non}}$")
    parts["r_counterfactual"], = ax.plot(
        df["gc_mid"], df["r_counterfactual"], marker="D", color=APPLIED_COLOR,
        markersize=4, markerfacecolor="white", markeredgewidth=1.2,
        linestyle="--", linewidth=1.8,
        label=r"$R_{\mathrm{eff}}$ (counterfactual)$ = \Pi R_{\mathrm{CpG}} + (1-\Pi)$")

    ax.axhline(1.0, **REF_LINE_KW)
    _log_ratio_axis(ax, df[["r_non", "r_eff", "r_cpg"]].to_numpy())
    # Legend order is the derivation's, not the draw order's: applied quantity, then the
    # two parts it decomposes into, then the hypothetical.
    _finish(ax, "Regional adjustment", xrange, show_xlabel,
            handles=[parts[k] for k in
                     ("r_eff", "r_non", "r_cpg", "r_counterfactual")])


# Panel C's two rows share one colour per stratum, defined once here so the band in the
# upper row and the line in the lower row cannot drift apart.
#
# `scored` is deliberately neutral grey for two reasons: it is the reference the lower
# row divides by, and it is the one stratum with no line there, so a saturated hue would
# promise a curve that does not exist. That also puts the colour on the bands whose
# growth is the point.
#
# HUE CARRIES THE FINDING HERE, rather than merely enumerating four categories. This is
# the one panel that cannot encode identity in symbols -- there is no marker on a filled
# area -- so its colours are spent on the result the lower row measures: the two QC-PASS
# strata outside the scored population are two shades of GREEN, and the QC-FAIL stratum
# is RED. Green says "the same territory, merely excluded", which is what the lower row
# finds (coding 0.86-1.00x, other-noncoding 0.94-1.03x, both flat across GC); red says
# "different", which is the QC-fail stratum alone (1.50-1.63x through the bulk, 3.39x by
# GC 0.58). A reader gets the panel's conclusion from the palette before reading a
# number, and the two greens being shades of one hue says the two strata are one kind of
# thing -- which is precisely the claim that the narrowing costs sample size and nothing
# else.
#
# Dark green below light green in the stack, so the pair reads as one ramp against the
# grey band under it rather than as two unrelated bands; red on top, where it is the
# thing that grows.
#
# This freed the panel from a constraint it used to be under -- avoiding hues that meant
# something in A, B, D or E. Those panels are monochrome now (see the module docstring),
# so no assignment here can collide with them.
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
# The lighter green is bounded from below by the lower row rather than by the stack: as
# a band it could be much paler and still read, but the same value is a 2 px line on
# white there, and at #79c9a4 that line was the palest thing in the figure. #3aa77c
# clears 3:1 against the surface; the darker green drops to #0b5238 to keep the two
# shades apart once the light one has been pulled down.
STRATUM_COLORS = {"scored": "0.78", "coding": "#0b5238",
                  "other_noncoding": "#3aa77c", "failed_qc": "#c9384a"}

# `other_noncoding` is named as the complement of the bottom band's own parenthetical
# ("QC-pass putatively neutral noncoding", supplied by the caller as `scored_note`), so
# the two read as one partition of QC-pass noncoding territory rather than as a named
# category beside a leftover. "Putatively" carries the same weight on both sides and is
# load-bearing on this one: these windows are outside a set McHale et al. call putatively
# neutral, which is not itself evidence of selection.
#
# Labels wrap over two lines where they are long enough to widen the legend past the
# axes. The break is explicit rather than left to the legend, which does not wrap: an
# unbroken 34-character entry sets the column width, and with two columns above the axes
# that pushed the legend wider than the panel it labels. Written as one string per label
# with an embedded newline, so the wrap travels with the text -- both rows of panel C
# read these same labels, and a label edited in one place cannot re-wrap in only one row.
STRATUM_LABELS = {
    "scored": "In the Gnocchi-scored test population",
    "coding": "In QC-pass coding windows",
    "other_noncoding": "In QC-pass putatively nonneutral\nnoncoding windows",
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
        # On its own line: with the note appended inline the bottom band's entry ran to
        # 61 characters, half again the width of the panel.
        style = [(c, col, f"{lab}\n({scored_note})" if c == "frac_scored" else lab)
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
    # ABOVE the axes, not in "lower left". The stack fills [0, 1] by construction, so an
    # inside legend always covers artwork -- tolerable when the bottom-left band was one
    # solid field (the default window set, where `scored` is 84% of the low-GC bins), but
    # not on McHale et al.'s neutral set, where four bands share that corner and the
    # legend hides the composition it is labelling. Outside, no frame is needed either.
    _finish(ax, "Fraction of non-CpG\ntraining sites", xrange, show_xlabel,
            legend_loc="lower center", legend_bbox=(0.5, 1.01), legend_ncol=2,
            grid_axis="y", handles=bands[::-1])


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
    # "Log fold change of", not "Log P(DNM) relative to ...": the reader is being asked
    # to read 0.44 as 1.55x, so the label has to say that what is logged is the ratio
    # and not the rate itself. Plain text at the panel's own type size, not a mathtext
    # \\dfrac, which renders smaller and in a different font from every other label here.
    _finish(ax, "Log fold change of empirical P(DNM)\nrelative to test population"
                "\n(non-CpG sites)", xrange,
            show_xlabel, handles=[handles[s] for s in order])


# Panel D's two populations. A population's marker AND dash pattern are its identity in
# BOTH of the panel's rows -- fitted and empirical are told apart by which row they are
# in, not by anything on the curve -- so the pair still reads as one object, now
# vertically; see panel_dnm_probability_pairs.
#
# The size-matched random control (same NUMBER of sites as `scored`, drawn from the same
# population as `full`) used to be a third pair here. It is not plotted any more: it lies
# on top of the original pair, which is the whole of what it has to say, and saying it
# cost two of the panel's six curves. It survives as a refit and is reported numerically
# under panel E, where the same control lands at 0.162 against published Gnocchi's 0.168.
PAIR_STYLE = {
    "full": {"marker": "s", "dashes": None,
             "label": "Original training set"},
    "scored": {"marker": "D", "dashes": (4, 1.6),
               "label": "Decontaminated training set"},
}


# --------------------------------------------------------- supporting CpG figure
# One hue per CpG trinucleotide context. These are a fourth axis of the figure (not a
# population, model or stratum), so they get their own qualitative set rather than
# reusing the main palette's role-coded slots.
CPG_COLORS = {"ACG": "#2a78d6", "CCG": "#eb6834", "GCG": "#1baf7a", "TCG": "#4a3aa7"}


def panel_cpg_methylation_effect(ax, ct, show_mu: bool = True) -> None:
    """
    Supporting Figure 7 (the manuscript's label), panel A. The CpG C>T rate against methylation level, per context,
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
    Supporting Figure 7, panel B. The fraction of CpG training sites that are
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

    # No legend: the two curves are named by the two y-axis labels, each in its own
    # curve's colour (blue left, grey right), so a legend would say it a second time in
    # smaller type -- and it had to sit mid-left, the one region the flat, high
    # mean-methylation curve leaves free. The threshold that DEFINES hypomethylated was
    # the one thing only the legend carried, so it moves into the ylabel.
    _finish(ax, "Fraction of CpG sites that are\nhypomethylated (level $\\leq$ 1)",
            xrange, show_xlabel, legend=False)


def panel_cpg_dnm_rate(ax, cpg, min_n: int = 100, xrange=(0.2, 0.8),
                       show_xlabel: bool = True) -> None:
    """
    Supporting Figure 7, panel C. The empirical DNM rate over CpG training sites against GC
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

    # MONO, like every other single-series panel here: there is one curve and no legend,
    # so a hue would be naming nothing. Colour is spent in this figure only on row B,
    # where it ties each curve to its own y axis.
    ax.errorbar(df["gc_pct"] / 100.0, df["p"], yerr=se, marker="o",
                color=MONO, markersize=5, linewidth=2, capsize=3,
                elinewidth=1, label="Empirical P(DNM), CpG contexts")
    # One series, and the ylabel already names it -- see _finish's `legend`.
    _finish(ax, "P(DNM) in the training set\n(CpG contexts)", xrange, show_xlabel,
            legend=False)


def panel_cpg_expected_share(ax, binned, min_n: int = 100, xrange=(0.2, 0.8),
                             show_xlabel: bool = True) -> None:
    """
    Supporting Figure 7, panel D. Pi(g), the CpG contexts' share of a GC bin's step-1
    expected counts. `binned` is data.r_eff_by_gc() output -- the same table panel B
    decomposes, so this curve is literally the weight in R_eff = Pi*R_CpG + (1-Pi)*R_non.

    It is what makes R_CpG ~ 1 a finding rather than a triviality: Pi runs 0.025 in the
    lowest GC bin to 0.426 in the highest, so by the top of the GC range nearly half the
    expected counts sit in contexts the regional adjustment leaves alone. A GC trend in
    R_CpG would therefore reach the applied multiplier scaled by up to 0.43, not erased
    -- which is why the counterfactual in panel B is flat by measurement and not by
    construction.

    Drawn against the same GC axis as B and C above it, so it ends earlier: Pi is binned
    over Chen windows, whose analyzed set thins out above GC 0.73, while the CpG training
    sites of those two reach 0.8.
    """
    df = binned.to_pandas() if hasattr(binned, "to_pandas") else binned
    df = df[df["n"] >= min_n].sort_values("gc_mid") if min_n else df.sort_values("gc_mid")

    ax.plot(df["gc_mid"], df["pi_cpg"], marker=SERIES_MARKERS["scored"],
            color=MONO, markersize=5, linewidth=2,
            label=r"$\Pi$ — CpG share of step-1 expected counts")
    ax.set_ylim(0, float(df["pi_cpg"].max()) * 1.15)
    # One series, and the ylabel already names it -- see _finish's `legend`.
    _finish(ax, "CpG share of step-1\nexpected counts, $\\Pi$", xrange, show_xlabel,
            legend=False)


def label_panels(axes, labels=("A", "B", "C"), x: float = -0.1, y: float = 1.02) -> None:
    """Bold panel letters in axes coordinates, for figures saved as a single file."""
    for ax, letter in zip(axes, labels, strict=True):
        ax.text(x, y, letter, transform=ax.transAxes, fontsize=17,
                fontweight="bold", va="bottom", ha="right")


def panel_dnm_probability_pairs(ax_empirical, ax_fitted, binned: dict, min_n: int = 500,
                                normalize: bool = True, xrange=(0.2, 0.76),
                                show_xlabel: bool = True,
                                gc_mean: float | None = None) -> None:
    """
    Panel D, as two stacked rows over one x axis: the top row carries only the EMPIRICAL
    P(DNM) curves, the bottom only the FITTED ones, one curve per training population in
    each. `binned` maps population name -> data.dnm_probability() table.

    WHAT MOVES TO THE ROW, AND WHAT STAYS ON THE CURVE. Fitted-versus-empirical is now
    said by which row a curve is in, so the only thing left for a curve to say within its
    row is which population it belongs to -- `full` solid squares, `scored` dashed
    diamonds, the same symbol and dash in both rows, so a population reads as one object
    down the figure. The old within-pair cues are kept anyway rather than freed up: the
    empirical row keeps its error bars and hollow markers, the fitted row its filled
    markers and no bars, because that distinction is true and not merely a convention --
    the binomial standard error belongs to the measurement, and a prediction has no such
    bar to draw. A row lifted out of the figure still says which quantity it holds
    without its ylabel being read.

    WHAT THE SPLIT COSTS, AND WHAT PAYS FOR IT. The panel's second claim -- that the fit
    MISSES the empirical curve by 26% and 29% in opposite directions on the original
    training set, and tracks it to within 6% on the scored one -- is a comparison within
    a population, whose two curves now sit on different axes. So the rows are given ONE
    shared y range, computed over every curve in both, on top of the shared x: a given
    vertical distance is then the same interval in either row and the gap can be read
    across the break. What the split buys is the first claim, which was the crowded one -- four
    curves on one axis, and the two that carry it (empirical against empirical: 2.45x and
    turning over, against 1.60x and monotone) were the two the pair styling deliberately
    kept apart.

    LEVELS ARE NOT COMPARABLE ACROSS POPULATIONS. The class balance differs (12.2 vs
    13.5 background sites per DNM), which shifts P(DNM) by that factor for reasons
    unrelated to GC. normalize=True divides each curve by its own site-weighted mean,
    which removes that offset and compares SHAPE, and is what the figure needs. Within
    a population the comparison is exact either way -- fitted and empirical come from the
    very same sites.

    `show_xlabel` applies to the bottom row, the only one with visible tick labels under
    the sharex the caller sets up. `gc_mean` is in the panel's 0-1 x units, i.e. the
    tables' `gc_mid` / 100, and is drawn on both rows so neither can be read without it.
    """
    frames, drawn_values = {}, []
    for name, df in binned.items():
        d = df[df["n"] >= min_n] if min_n else df
        gc = d["gc_mid"] / 100.0
        d = d[(gc >= xrange[0]) & (gc <= xrange[1])].sort_values("gc_mid")

        pred, emp, se = d["mean_pred"], d["empirical_prop"], d["se"]
        if normalize:
            pred = pred / np.average(pred, weights=d["n"])
            wemp = np.average(emp, weights=d["n"])
            emp, se = emp / wemp, se / wemp
        frames[name] = (d["gc_mid"] / 100.0, pred, emp, se)
        # Error bars included, so the shared limits below cannot clip a cap -- the lowest
        # of them, at 0.84, sat outside a pad computed from the markers alone.
        #
        # The bar stays a symmetric +-se on the DATA, and is not rebuilt for the log
        # axis. p +- se is the interval the binomial supports; the axis only maps its
        # endpoints, so the arms come out unequal, by 11.5% on the widest bar (scored,
        # GC 0.621: 85.0 px up against 94.8 down at 300 dpi) and under 2% through the
        # GC bulk. That unevenness is the transform being honest -- forcing the caps
        # equal would draw an interval the data do not support. Panel C's lower row
        # uses the delta-method SE of the log instead, and is right to: it plots
        # log(ratio) AS the quantity, on a linear axis, so the SE of the log is the
        # only bar that means anything there. Here the quantity is the ratio itself.
        drawn_values.append(np.concatenate([pred, emp - se, emp + se]))

    handles = {id(ax_empirical): [], id(ax_fitted): []}
    for name, (gc, pred, emp, se) in frames.items():
        style = PAIR_STYLE[name]
        dash_kw = {} if style["dashes"] is None else {"dashes": style["dashes"]}
        empirical = ax_empirical.errorbar(
            gc, emp, yerr=se, marker=style["marker"], color=MONO, markersize=5,
            linewidth=2, markerfacecolor="white", markeredgewidth=1.2,
            capsize=3, elinewidth=1, label=style["label"], **dash_kw)
        fitted, = ax_fitted.plot(gc, pred, marker=style["marker"], color=MONO,
                                 markersize=5, linewidth=2, label=style["label"],
                                 **dash_kw)
        handles[id(ax_empirical)].append(empirical)
        handles[id(ax_fitted)].append(fitted)

    values = np.concatenate(drawn_values)
    # Broken to keep the LONGEST LINE short rather than to keep the line count low: a
    # rotated ylabel's height on the page is its longest line, and each row is now barely
    # over 3 in tall, where the single-axis panel had 4.6. "Empirical P(DNM) relative to
    # its" on one line overran the row and was clipped at the figure edge.
    quantity = ("relative to its\nGC-averaged value" if normalize
                else "in the training set")
    # BOTH ROWS TAKE THE SAME LIMITS, from every curve in both, so the two axes come out
    # identical. That is what makes the rows comparable at all: the fitted curves span
    # less than the empirical ones, and an axis fitted to each row separately would
    # magnify the fitted row until a fit that misses by 26% looked like one that tracks.
    #
    # LINEAR, INCLUDING WHEN NORMALIZED. The normalized quantity is a positive ratio with
    # its reference at 1, so a log axis is available and panel B's argument for one --
    # that a 25% excess and a 25% deficit should read as equal departures -- transfers on
    # its face. It is not worth taking here. Panel B's r = 1 is a substantive null, where
    # this 1 is put there by the normalization itself: every curve is divided by its own
    # GC-averaged value, so every curve crosses 1 by construction and there is no
    # hypothesis a departure from it is being weighed against. What the panel is read for
    # is the SHAPE of each curve and the GAP between the rows, and the numbers it is
    # quoted for (2.45x rising then collapsing, 1.60x monotone, a fit 26% high and 29%
    # low) come off the curves' own values rather than off the axis. The log's other
    # claim, that it opens up the low-GC end where the curves sit inside 0.89-1.01, is
    # worth even less: they are pinned there BY CONSTRUCTION, by a mean the GC bulk
    # dominates, so magnifying that band magnifies the pinning and not a measurement.
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    for ax, kind, bottom in ((ax_empirical, "Empirical", False),
                             (ax_fitted, "Fitted", True)):
        if normalize:
            ax.axhline(1.0, **REF_LINE_KW)
        ax.set_ylim(lo - 0.04 * (hi - lo), hi + 0.04 * (hi - lo))
        # One line for both populations, which is honest here only because their site-
        # weighted mean GCs agree to ~0.01 (see _gc_mean_line): it marks where the
        # training set sits, and every curve's divergence from the others is out in the
        # tail beyond it. The caller says which population's mean it is.
        _gc_mean_line(ax, gc_mean)
        # "GC-averaged value", not "its own mean": the divisor is that curve's mean
        # ACROSS GC bins, site-weighted, so the label has to say which average was taken
        # out. The row's own word -- Empirical or Fitted -- leads the label, since with
        # the pair split that is the one thing the symbols no longer carry.
        _finish(ax, f"{kind} P(DNM)\n{quantity}\n(non-CpG sites)", xrange,
                show_xlabel and bottom, handles=handles[id(ax)],
                legend_handlelength=3.2)


# ------------------------------------------------------- Supporting Figure 8

# GC-poor to GC-rich, and the reference notebook's own choice
# (papers/neutral_models_are_biased/7.CDTS/main.2.ipynb). A DIVERGING ramp, for the reason
# in the module docstring's third exemption: the panel's content is a departure from the
# pooled curve in two directions, which a sequential ramp cannot show.
GC_CMAP = "coolwarm"

# Panel E draws published Gnocchi as a filled square and the score retrained on the scored
# population as a filled up triangle. Supporting Figure 8 compares the same two quantities
# and takes the same two glyphs, so a reader moving between the figures does not have to
# relearn which curve is which.
SCORE_MARKERS = {"published": SERIES_MARKERS["step2"],
                 "scored": SERIES_MARKERS["scored"]}


def panel_pr_curves(ax, curves: dict, key: str, ylim_scale: float = 3.0,
                    legend: bool = True, show_ylabel: bool = True) -> None:
    """
    Supporting Figure 8, panel A, for ONE score -- `curves` is data.pr_curves()
    output. Precision against recall, one line per GC bin, over the pooled curve and the
    random-classifier baseline.

    ONE SCORE PER AXES. Two scores' worth of GC-binned PR curves in a single frame is six
    lines in one colour ramp with no way to say which belongs to which; the figure draws
    this panel twice instead, side by side, and shares the y axis between them.

    y IS PRECISION AND ITS SCALE IS SET BY THE BASELINE, not by the data: the axis runs to
    `ylim_scale` times the positive fraction, so "3" here means "three times better than
    guessing" whatever the truth set's prevalence, and the dashed baseline sits exactly
    one third of the way up. That is what lets the two axes be read against each other.

    The pooled black curve is not a summary of the coloured ones -- it is the performance
    a user of the score actually gets when they do not condition on GC content. The
    panel's point is the spread of the coloured curves AROUND it.
    """
    c = curves[key]
    entries, r = c["bins"], c["r"]
    cmap = matplotlib.colormaps[GC_CMAP]

    for i, e in enumerate(entries):
        ax.plot(e["recall"], e["precision"], color=cmap(i / max(len(entries) - 1, 1)),
                linewidth=2, label=f"GC in ({e['lo']:.2f}, {e['hi']:.2f}]")
    ax.plot(c["all"]["recall"], c["all"]["precision"], color="black", linewidth=3,
            label="all GC content")
    ax.axhline(r, color="black", linestyle="--", linewidth=2, label="random classifier")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, ylim_scale * r)
    ax.set_xlabel("Recall", fontsize=AXIS_LABEL_FONTSIZE)
    if show_ylabel:
        ax.set_ylabel("Precision", fontsize=AXIS_LABEL_FONTSIZE)
    # The SHORT name, not the display one: the two axes sit either side of panel B's
    # letter, and "Gnocchi (decontaminated training set)" is wider than its own axes and
    # runs into it. It is also the word panel B's legend uses, so the figure names each
    # score once and identically.
    ax.set_title(f"Gnocchi, {c['short']}", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(True, **GRID_KW)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    # Not _finish: that helper labels x as "GC content", which is every other panel's
    # abscissa and not this one's -- here x is recall and GC is the colour ramp.
    if legend:
        ax.legend(fontsize=LEGEND_FONTSIZE - 2, loc="upper right", frameon=False)


def panel_aupr_by_gc(ax, curves: dict, xrange=(0.2, 0.8), show_xlabel: bool = True,
                     legend_loc: str = "lower right", deltas=None) -> None:
    """
    Supporting Figure 8, panel B. Normalized auPRC against GC content, one curve per
    score.

    y = 1 is the random classifier, by construction of the normalization, and it is the
    only horizontal line on the panel that means anything: a curve at 1.4 finds enhancers
    40% more precisely than guessing, averaged over the recall axis of panel A.

    THIS PANEL IS THE COMPARISON THE FIGURE EXISTS FOR. Panel E of Fig. 5 establishes that
    retraining the regional adjustment on the scored population removes the score's GC
    bias. What it cannot say is whether the biased score was nevertheless the better
    detector -- bias and signal-to-noise act on discovery jointly -- so the two curves
    here are the direct test: the gap between them at a given GC is what the retraining
    buys or costs at that GC, on identical windows and an identical set of positives.

    `deltas` PUTS THE PAIRED INTERVAL ON THE RETRAINED CURVE, and it is drawn on ONE curve
    rather than both for a reason that matters. Independent bars on the two curves would be
    the wrong object twice over: they would describe the uncertainty of each LEVEL when the
    question is about the GAP, and they would be far WIDER than the gap's own interval,
    because the two scores are columns of one table and almost all of the sampling
    variability -- which windows the truth set happens to contain -- is common to both and
    cancels in the difference. Marginal bars would therefore hide the one real difference
    here (-1.40% at GC 0.40-0.50) while still leaving the eye-catching top-bin gap
    ambiguous. So the bar drawn on each retrained point is its 95% paired-bootstrap
    interval RELATIVE TO PUBLISHED, mapped back into this panel's units: the published
    value in that bin times (1 + the interval on the relative gain). A bar that excludes
    the published marker is a real difference; one that spans it is not.

    `deltas` must be data.pr_curve_deltas() run on THIS panel's bins and balancing --
    gc_bins=LAX_GC_BINS, min_n=LAX_MIN_BIN_WINDOWS, balance=True -- or the interval belongs
    to a different statistic than the markers. Bins are matched on their midpoint.

    The legend sits BOTTOM RIGHT rather than in the usual top corner: both curves fall
    monotonically from the left edge, so the top left is where the panel's content is and
    the bottom right is empty by construction. The pooled value travels in the legend
    label because it is the number a reader wants beside the curve -- the performance
    someone gets from the score without conditioning on GC at all -- and it has no place
    on the axes, which are conditional on GC everywhere.
    """
    # The relative gain is the same for auPRC and for auPRC/r, since within a bin both
    # scores are divided by the same base rate -- so the interval maps into this panel's
    # units by scaling the PUBLISHED level, with no renormalisation.
    ci = {}
    if deltas is not None:
        pub = {round(e["mid"], 6): e["aupr_norm"] for e in curves["published"]["bins"]}
        for row in deltas.iter_rows(named=True):
            base = pub.get(round(row["mid"], 6))
            if base is not None:
                ci[round(row["mid"], 6)] = (base * (1 + row["ci_lo"]),
                                            base * (1 + row["ci_hi"]))

    for key, c in curves.items():
        if not c["bins"]:
            continue
        x = np.array([e["mid"] for e in c["bins"]])
        y = np.array([e["aupr_norm"] for e in c["bins"]])
        yerr = None
        if key == "scored" and ci:
            lo = np.array([y[i] - ci.get(round(v, 6), (y[i], y[i]))[0]
                           for i, v in enumerate(x)])
            hi = np.array([ci.get(round(v, 6), (y[i], y[i]))[1] - y[i]
                           for i, v in enumerate(x)])
            yerr = np.vstack([np.maximum(lo, 0), np.maximum(hi, 0)])
        label = f"Gnocchi, {c['short']} (pooled {c['all']['aupr_norm']:.3f})"
        if yerr is not None:
            label += ", 95% CI vs published"
        ax.errorbar(x, y, yerr=yerr,
                    marker=SCORE_MARKERS[key], color=MONO,
                    markerfacecolor="white" if key == "published" else MONO,
                    markeredgewidth=1.2, markersize=6, linewidth=2,
                    capsize=3, elinewidth=1.2, label=label)
    ax.axhline(1.0, **REF_LINE_KW)

    # Headroom BELOW the y = 1 reference, which autoscaling does not leave: every curve
    # point sits above 1, so matplotlib puts the bottom spine within a hair of the dashed
    # line and the legend then prints across it. The reference is what the whole y axis is
    # read against and must not be crossed out by a label.
    ys = [e["aupr_norm"] for cv in curves.values() for e in cv["bins"]]
    lo, hi = min(min(ys), 1.0), max(ys)
    ax.set_ylim(lo - 0.22 * (hi - lo), hi + 0.04 * (hi - lo))

    # A size down: each label carries a name AND a number, and this panel is a third of
    # the figure's width, where the default size overflows the axes to the left and lands
    # the legend on the y tick labels.
    _finish(ax, "auPRC (normalized by\npositive-class fraction)", xrange, show_xlabel,
            legend_loc=legend_loc, legend_fontsize=LEGEND_FONTSIZE - 2)


def panel_aupr_delta(ax, deltas, xrange=(0.2, 0.8), show_xlabel: bool = True,
                     ylabel: str | None = None) -> None:
    """
    Supporting Figure 8, panel C. The PAIRED gain of the retrained score over the published
    one, per GC bin, with a bootstrap confidence interval. `deltas` is
    data.pr_curve_deltas() output.

    WHY THIS PANEL EXISTS WHEN PANEL B IS RIGHT THERE. Panel B invites the eye to compare
    two curves that cross, wobble, and are drawn without uncertainty; the reader cannot
    tell a real gap from a thin bin. This panel is that same comparison as ONE quantity
    with an interval on it. The interval is what the panel is for, so the marker is
    secondary and the bars are drawn heavy enough to read at figure scale.

    y = 0 IS THE CLAIM'S NULL and is the only reference on the panel: an interval clear of
    it in a bin says the two scores genuinely differ there. Bars are 95% percentile
    intervals from the paired bootstrap -- see data.pr_curve_deltas for why pairing is
    what makes them narrow, and why they are NOT the same thing as error bars on panel B's
    two curves, which would describe the uncertainty of each level rather than of the gap.

    THE TOP BIN IS WIDER THAN PANEL B's (0.55-0.80 against 0.55-0.60), because McHale et
    al.'s window file is nearly empty above 0.60 and this panel would otherwise say nothing
    about the tail -- which is where panel E's bias reduction is largest and so where the
    figure's question actually lives. Its marker sits at the merged bin's centre, well to
    the right of panel B's last point; the caption has to say so, or the two panels look
    like they disagree about where the last measurement is.
    """
    x = deltas["mid"].to_numpy()
    y = 100.0 * deltas["delta"].to_numpy()
    lo = y - 100.0 * deltas["ci_lo"].to_numpy()
    hi = 100.0 * deltas["ci_hi"].to_numpy() - y

    ax.axhline(0.0, **REF_LINE_KW)
    ax.errorbar(x, y, yerr=np.vstack([lo, hi]),
                marker=SCORE_MARKERS["scored"], color=MONO, markerfacecolor=MONO,
                markeredgewidth=1.2, markersize=6, linewidth=2, capsize=4, elinewidth=1.4)
    # NO LEGEND: one series, and the ylabel already names it -- a legend here would
    # restate the ylabel in smaller type across the top of the very bin the panel is about
    # (the tail one, whose interval is the tallest thing on the axes). What the legend
    # would have said that the ylabel does not -- that the bars are 95% percentile
    # intervals from a paired bootstrap -- belongs in the caption, which can also say how
    # many replicates.
    # `ylabel` is how this panel serves two statistics. The drawing is identical -- a
    # paired gain per GC bin with a bootstrap interval and a zero reference -- and only the
    # quantity differs: panel C's auPRC gain integrated over all thresholds, and panel I's
    # lift gain at a matched per-bin operating point. Those two answer different questions
    # and, on the real run, give different answers, which is the point of drawing both.
    _finish(ax, ylabel or "auPRC gain of the retrained\nscore (%)", xrange, show_xlabel,
            legend=False)


# The two scores' curves in panels D and E, which unlike panel C draw LEVELS rather than a
# difference and so need to be told apart from each other. Same square/triangle and same
# open/filled convention as panel B, so a reader learns the pair once for the whole figure.
def _threshold_series(tm, key: str):
    rows = tm.filter(tm["score"] == key).sort("mid")
    return rows


def panel_threshold_metric(ax, tm, metric: str = "precision", threshold: float = 4.0,
                           xrange=(0.2, 0.8), show_xlabel: bool = True,
                           show_prevalence: bool = True, logy: bool | None = None,
                           legend_loc: str | None = None, deltas=None) -> None:
    """
    Supporting Figure 8, panels D-F. One of three quantities at a FIXED Gnocchi threshold,
    per GC bin, one curve per score, with Wilson intervals. `tm` is
    data.threshold_metrics() output.

      metric="call_rate"  the fraction of windows in the bin that clear the threshold.
                          THE PANEL TO QUOTE: it uses no labels at all, so it rests on
                          neither GeneHancer nor the laxness of an enhancer proxy -- it is
                          a property of the score and of GC content. Drawn on a log y axis,
                          because published Gnocchi's spans nearly two orders of magnitude
                          across the GC range and a linear axis would show only the top bin.
      metric="precision"  P(constrained | called) -- the analyst's number.
      metric="recall"     P(called | constrained) -- what fraction is caught.
      metric="lift"       precision / base rate, the ceiling-limited quality measure.
      metric="skill"      (precision - r)/(1 - r), its ceiling-free companion.

    `deltas` REPLACES THE MARGINAL INTERVALS WITH THE PAIRED ONE, drawn on the retrained
    curve alone. The Wilson bars this function draws by default are correct for what the
    panel otherwise shows -- two LEVELS, each with its own binomial error -- but they are
    the wrong object as soon as the question is whether the two curves DIFFER, and they are
    far wider than the difference's own interval, since the two scores are columns of one
    table and the variability in which windows the bin happens to contain cancels between
    them. So when `deltas` (a data.lift_deltas() table on the same bins) is supplied, the
    published curve loses its bars and each retrained marker gains its 95% paired interval
    RELATIVE TO PUBLISHED, mapped into the panel's units by scaling the published level.
    A bar that excludes the published marker is a real difference; one that spans it is
    not. Same device, same reasoning, as panel_aupr_by_gc's.

    WHY THIS PANEL IS NOT A RESTATEMENT OF B. B and C hold the threshold free and ask how
    well each score RANKS windows within a GC bin, which a GC-dependent shift barely
    affects. Here the threshold is fixed at the value Chen et al. themselves use to call a
    window constrained, so the shift decides how many windows in each bin are called at
    all -- and the analyst's question, "my window scores above 4, how likely is it
    constrained", is precisely this panel's y-axis.

    THE DASHED CURVE IS THE BIN'S BASE RATE, not a random-classifier line borrowed from
    panel A, and drawing it is what stops the panel being misread. Enhancer prevalence
    climbs about 7.7x across these bins, so precision at a fixed threshold rises with GC
    for ANY score, bias or no bias; what matters is the gap between a curve and the dashed
    line beneath it. A reader who takes a rising precision curve as evidence of good
    performance at high GC has read the base rate, not the score. (On the recall panel
    there is no such reference -- recall conditions on the positives, so the base rate has
    already been divided out -- and show_prevalence is ignored.)
    """
    if metric not in ("call_rate", "precision", "recall", "lift", "skill"):
        raise ValueError(f"metric {metric!r} is not drawable here")
    logy = (metric == "call_rate") if logy is None else logy
    # PRECISION IS THE ODD ONE OUT and needs both a different corner and a shorter
    # legend. Its two curves run diagonally from bottom left to top right with a base-rate
    # line beneath them, so no corner is free: upper left sits on the curves, and lower
    # right -- the emptiest region -- is only wide enough if the entries are short. So the
    # thresholds are dropped from ITS legend and carried by D and F, which are log panels
    # whose published curve leaves the whole top-left empty. The caption says the two
    # scores are matched on calling rate; the panel beside it shows the numbers.
    # Precision rises left to right, so its only free corner is lower right; lift and
    # skill fall left to right, so theirs is lower left; call_rate and recall are read
    # against a flat corrected curve low on the axes, leaving the top left empty.
    legend_loc = ({"precision": "lower right", "lift": "lower left",
                   "skill": "lower left"}.get(metric, "upper left")
                  if legend_loc is None else legend_loc)
    with_threshold = metric != "precision"

    ci = {}
    if deltas is not None:
        base = {round(r["mid"], 6): r[metric]
                for r in _threshold_series(tm, "published").iter_rows(named=True)}
        for row in deltas.iter_rows(named=True):
            b = base.get(round(row["mid"], 6))
            if b is not None:
                ci[round(row["mid"], 6)] = (b * (1 + row["ci_lo"]), b * (1 + row["ci_hi"]))

    for key in ("published", "scored"):
        rows = _threshold_series(tm, key)
        if not rows.height:
            continue
        x = rows["mid"].to_numpy()
        y = rows[metric].to_numpy()
        if deltas is not None:
            if key == "published":
                lo = hi = np.zeros_like(y)          # the reference carries no bars
            else:
                lo = np.array([max(y[i] - ci.get(round(v, 6), (y[i], y[i]))[0], 0.0)
                               for i, v in enumerate(x)])
                hi = np.array([max(ci.get(round(v, 6), (y[i], y[i]))[1] - y[i], 0.0)
                               for i, v in enumerate(x)])
        else:
            lo = y - rows[f"{metric}_lo"].to_numpy()
            hi = rows[f"{metric}_hi"].to_numpy() - y
        # EACH SCORE'S OWN THRESHOLD GOES IN ITS LEGEND ENTRY, because they are not the
        # same number: data.threshold_metrics matches the two on CALLING RATE rather than
        # on z, so that precision and recall compare like with like (retraining moves the
        # whole z distribution, so a common cutoff is ~8x stricter for the retrained score
        # -- see that function). A reader who is not told will assume a common cutoff.
        # ONE threshold or MANY. Under global matching a score has a single cutoff and the
        # legend states it. Under match_within_bin every bin has its own, and printing the
        # first bin's would be a quiet lie -- so the suffix is dropped entirely rather than
        # repeated identically on both entries, which only doubles the legend's width in a
        # panel whose one free corner is already tight. The caption carries it.
        ts = rows["threshold_used"].unique().to_list()
        label = f"Gnocchi, {rows['short'][0]}"
        if deltas is not None and key == "scored":
            label += ", 95% CI vs published"
        if with_threshold:
            label += (f"  ($z \\geq {float(ts[0]):.2f}$)" if len(ts) == 1
                      else "")
        ax.errorbar(x, y, yerr=np.vstack([lo, hi]),
                    marker=SCORE_MARKERS[key], color=MONO,
                    markerfacecolor="white" if key == "published" else MONO,
                    markeredgewidth=1.2, markersize=6, linewidth=2, capsize=3,
                    elinewidth=1.2, label=label)

    if metric == "precision" and show_prevalence:
        base = _threshold_series(tm, "published")
        ax.plot(base["mid"].to_numpy(), base["r"].to_numpy(),
                linestyle="--", color="0.45", linewidth=1.4,
                label="base rate in the bin")

    if logy:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda v, _: f"{100 * v:g}%" if v > 0 else ""))
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        # Headroom for the legend. The legend sits upper LEFT and the published curve's
        # top point is upper RIGHT, but the box spans most of the panel width, so without
        # this the entry runs into that marker. A factor on a log axis, not a margin.
        vals = [v for v in tm["call_rate" if metric == "call_rate" else metric].to_list()
                if v and v > 0]
        if vals:
            ax.set_ylim(min(vals) / 1.6, max(vals) * 4.0)
    # The threshold is NOT in the y label: the two curves use different ones (matched on
    # calling rate), so a single number in the label would be wrong for one of them. Each
    # legend entry carries its own instead.
    ylabel = {
        "call_rate": "Windows called",
        "precision": "P(constrained | called)",
        "recall": "Fraction of constrained\nwindows called",
        "lift": "Lift (precision / base rate)",
        "skill": "Skill  (precision $-$ $r$) / (1 $-$ $r$)",
    }[metric]
    _finish(ax, ylabel, xrange, show_xlabel, legend_loc=legend_loc,
            legend_fontsize=LEGEND_FONTSIZE - 2)


def _log_ticks(lo: float, hi: float) -> list:
    """
    Tick values for a log axis spanning well under a decade, where the default locator puts
    one label on the whole axis. A 1-1.5-2-3-5-7 sequence per decade, kept to the range.
    """
    # Density chosen from the span: the dense sequence is right for a fraction of a
    # decade and unreadable over two, where the labels collide into a smear.
    span = hi / lo
    mults = (1, 3) if span > 50 else (1, 2, 5) if span > 8 else (1, 1.5, 2, 3, 5, 7)
    out = []
    k = int(np.floor(np.log10(lo)))
    while 10.0 ** k <= hi * 10:
        for m in mults:
            v = m * 10.0 ** k
            if lo <= v <= hi:
                out.append(v)
        k += 1
    return out


def panel_lift_vs_recall(ax, tm, threshold: float = 4.0, guides=(0.001, 0.01, 0.1),
                         show_xlabel: bool = True, legend_loc: str = "lower left",
                         y: str = "lift") -> None:
    """
    Supporting Figure 8, the operating-point panel: lift against recall, one point per GC
    bin, at a fixed threshold. `tm` is data.threshold_metrics() output.

    IT IS PANELS D, E AND F AT ONCE, because of an exact identity:

        recall = calling rate x lift

    (recall = TP/P, calling rate = N_called/N, lift = (TP/N_called)/(P/N); multiply the
    last two and the call counts cancel). So a point's position fixes all three: recall on
    x, lift on y, and the calling rate is the ratio. ON LOG-LOG AXES that ratio becomes a
    difference, so ISO-CALLING-RATE CONTOURS ARE PARALLEL LINES OF SLOPE 1 -- drawn here as
    the light guides, the same device as iso-F1 curves on a precision-recall plot.

    AND THAT IS WHY THE PANEL SHOWS THE BIAS AS A SHAPE. A score whose threshold means the
    same thing everywhere calls the same fraction of windows in every GC bin, so all of its
    points lie on ONE contour and the panel shows a single line segment sliding along it.
    A GC-biased score's calling rate moves with GC, so its points fan ACROSS contours. The
    published score spans about two orders of magnitude of them; the retrained score
    collapses onto one. No summary statistic is doing any work in that comparison -- it is
    the geometry.

    THE Y AXIS IS ALREADY THE CORRECTED RECALL, which is worth saying because the obvious
    question about this panel is whether recall needs the base-rate correction precision
    got. It does not, and the reason is that its null is different: a random classifier's
    precision IS the base rate, but its recall is the CALLING RATE, whatever the base rate
    is. So recall's normaliser is q, not r -- and recall/q = lift, identically, by the
    same cancellation as above. Base-rate-corrected precision and calling-rate-corrected
    recall are one number. Raw recall stays on x precisely because the gap between it and
    lift is the calling rate, which is the bias; correcting x would plot lift against lift.

    COMPARE THE TWO SCORES VERTICALLY, NOT HORIZONTALLY. The calling rates are matched
    GLOBALLY, not per bin, so within a bin the two scores still call very different
    fractions -- 14% against 0.8% in the top GC bin on the real run. Horizontal distance
    between a square and a triangle in one bin is therefore a CALLING-RATE difference and
    not a performance difference: published's 15.1% recall against the retrained score's
    0.9% there is almost entirely that, both sitting at lift ~1.1. The legitimate within-bin
    comparison is the vertical one, and data.lift_deltas is what puts an interval on it.

    NEITHER AXIS IS PREVALENCE-FREE ACROSS BINS. Lift is capped at 1/r, and recall at a
    calling rate q is capped at q/r -- 12% where r = 0.083 but 1.6% where r = 0.639, the
    same ceiling scaled by q. So read each score's SHAPE across bins, and do not rank bins
    against each other; the per-bin `skill` and `LR+` in data.threshold_metrics are the
    ceiling-free quantities for that.

    y="skill" DRAWS THE CEILING-FREE COMPANION, (precision - r)/(1 - r), against the same
    recall axis. What is gained is that the y axis can be read ACROSS bins: lift is capped
    at 1/r, which falls from 12.0 to 1.6 over these bins and compresses the high-GC
    differences into a few percent, whereas skill maps random to 0 and perfect to 1
    everywhere. What is LOST is the geometry: skill = [r/(1-r)](lift - 1), so its relation
    to recall runs through the bin's own base rate and the iso-calling-rate contours are no
    longer one universal family -- each bin would need its own. They are therefore not
    drawn in this mode, and the only reference is skill = 0, the random classifier. Use
    y="lift" to read the calling rate off the geometry and y="skill" to compare bins.

    GC IS THE TRACED PARAMETER AND TAKES PANEL A's COLOUR RAMP, blue for GC-poor through
    red for GC-rich. It is an ordered variable, which is the same exemption to this
    module's monochrome rule that panel A gets, and using one ramp for one variable across
    the figure is what lets a reader carry the mapping between them. Score identity stays
    with the marker: open square published, filled triangle retrained, as everywhere else.
    """
    # Only the two scores this panel draws set its limits. threshold_metrics also carries
    # the GC-content baseline rows, whose within-bin lift dips below 1 and would stretch
    # the axes around a series that is never plotted here.
    drawn_scores = ("published", "scored")
    tm = tm.filter(tm["score"].is_in(drawn_scores))

    cmap = matplotlib.colormaps[GC_CMAP]
    bins = sorted(set(tm["mid"].to_list()))
    colour = {m: cmap(i / max(len(bins) - 1, 1)) for i, m in enumerate(bins)}

    if y not in ("lift", "skill"):
        raise ValueError(f"y must be 'lift' or 'skill', not {y!r}")
    logy = y == "lift"

    xs = [v for v in tm["recall"].to_list() if v and v > 0]
    ys = [v for v in tm[y].to_list() if v is not None and np.isfinite(v)]
    lo_x, hi_x = min(xs) / 2.2, max(xs) * 2.2
    lo_y, hi_y = ((min(ys) / 1.3, max(ys) * 1.5) if logy else
                  (min(0.0, min(ys)) - 0.06 * max(ys), max(ys) * 1.28))

    # LIMITS BEFORE GUIDES. The contours run over orders of magnitude; left to autoscale
    # they set the y range and squash every marker into a band. Fixing the limits to the
    # DATA and letting the guides clip is what keeps this a plot of the points.
    ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlim(lo_x, hi_x)
    ax.set_ylim(lo_y, hi_y)

    if not logy:
        ax.axhline(0.0, **REF_LINE_KW)      # skill = 0 is the random classifier
    for c in (guides if logy else ()):
        ax.plot([lo_x, hi_x], [lo_x / c, hi_x / c], color="0.82", linewidth=0.9, zorder=0)
        # Label where the contour leaves the top of the axes, or the right edge if it
        # never gets there. Horizontal: the axes are ~2 decades wide and ~1 tall, so a
        # slope-1 line is not at 45 degrees on the page and a rotated label would lie.
        x_top = hi_y * c
        if lo_x <= x_top <= hi_x:
            ax.annotate(f"{100 * c:g}% called", xy=(x_top, hi_y), xytext=(2, -2),
                        textcoords="offset points", fontsize=LEGEND_FONTSIZE - 3,
                        color="0.5", ha="left", va="top")
        elif lo_y <= hi_x / c <= hi_y:
            ax.annotate(f"{100 * c:g}% called", xy=(hi_x, hi_x / c), xytext=(-2, 2),
                        textcoords="offset points", fontsize=LEGEND_FONTSIZE - 3,
                        color="0.5", ha="right", va="bottom")

    handles = []
    for key in drawn_scores:
        rows = tm.filter(tm["score"] == key).sort("mid")
        if not rows.height:
            continue
        xv, yv = rows["recall"].to_numpy(), rows[y].to_numpy()
        # The connecting line orders the points by GC; it is not a series in its own right
        # and must not compete with the coloured markers.
        ax.plot(xv, yv, color=MONO, linewidth=1.4, zorder=2)
        if f"{y}_lo" in rows.columns:
            ax.errorbar(xv, yv, yerr=np.vstack([yv - rows[f"{y}_lo"].to_numpy(),
                                                rows[f"{y}_hi"].to_numpy() - yv]),
                        fmt="none", ecolor=MONO, elinewidth=1.0, capsize=3, zorder=1)
        for xi, yi, mid in zip(xv, yv, rows["mid"].to_list()):
            ax.plot([xi], [yi], marker=SCORE_MARKERS[key], markersize=8,
                    markerfacecolor="white" if key == "published" else colour[mid],
                    markeredgecolor=colour[mid] if key == "published" else MONO,
                    markeredgewidth=1.6, zorder=3, linestyle="none")
        # A proxy carrying the MARKER, since the two curves share one line style and a
        # plain line handle would name them identically.
        handles.append(mlines.Line2D([], [], color=MONO, linewidth=1.4,
                                     marker=SCORE_MARKERS[key], markersize=8,
                                     markerfacecolor="white" if key == "published" else MONO,
                                     markeredgecolor=MONO, markeredgewidth=1.6,
                                     # SHORT names here, unlike D-F. This panel's free
                                     # space is one corner and the full names do not fit
                                     # in it; the y label already says the subject is
                                     # Gnocchi, and D carries the full names beside it.
                                     label=rows["short"][0]))

    # EXPLICIT TICKS. Lift spans well under a decade here, so the default log locator
    # labels a single value and the axis reads as unscaled. The contours are the reason to
    # keep a log axis at all -- they are straight lines only in log-log -- so the ticks
    # have to be supplied rather than the scale changed.
    ax.set_xticks(_log_ticks(lo_x, hi_x))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{100 * v:g}%" if v > 0 else ""))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    if logy:
        ax.set_yticks(_log_ticks(lo_y, hi_y))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_locator(mticker.NullLocator())
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    if show_xlabel:
        ax.set_xlabel("Recall", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Lift (precision / base rate)" if logy else
                  "Skill  (precision $-$ $r$) / (1 $-$ $r$)",
                  fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(True, **GRID_KW)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(handles=handles, fontsize=LEGEND_FONTSIZE - 2, loc=legend_loc, frameon=False)
