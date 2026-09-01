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

The Supporting Figure follows the same rule. Its single-series rows are monochrome --
one curve and no legend leaves a hue naming nothing -- and colour survives only in its
hypomethylation row, where two curves share an x axis and have separate y axes, and the
hue is what says which curve reads against which scale.
"""
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
