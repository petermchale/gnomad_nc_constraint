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

GRID_KW = dict(color="0.85", linewidth=0.6)
REF_LINE_KW = dict(color="0.45", linewidth=0.8, linestyle="--")
AXIS_LABEL_FONTSIZE = 12
TICK_LABEL_FONTSIZE = 11
LEGEND_FONTSIZE = 10


def _finish(ax, ylabel, xrange, show_xlabel, legend_loc="upper left",
            legend_fontsize=LEGEND_FONTSIZE, grid_axis="both") -> None:
    """The frame every panel shares: range, labels, grid, despined box, legend."""
    ax.set_xlim(xrange)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
    if show_xlabel:
        ax.set_xlabel("GC content", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(True, axis=grid_axis, **GRID_KW)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
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
                    show_xlabel: bool = True) -> None:
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
    _finish(ax, "Mean standardized rank\nof constraint metric", xrange, show_xlabel)


def panel_r_eff(ax, binned, min_n: int = 100, xrange=(0.2, 0.73),
                show_xlabel: bool = True) -> None:
    """
    Panel B. The adjustment Gnocchi actually applies, r_eff = E2/E1, split into its
    CpG and non-CpG parts. `binned` is data.r_eff_by_gc() output.

    The three curves are an exact decomposition, r_eff = Pi*r_CpG + (1-Pi)*r_non, so
    the panel reads additively. The counterfactual holds the non-CpG term at 1 and
    changes nothing else; it is drawn as a dashed grey hypothetical rather than a
    fourth measured series. Its flatness is the claim.
    """
    df = binned.to_pandas() if hasattr(binned, "to_pandas") else binned
    df = df[df["n"] >= min_n].sort_values("gc_mid") if min_n else df.sort_values("gc_mid")

    for col, key, label in [("r_non", "step2", "Non-CpG contexts"),
                            ("r_eff", "scored", r"All contexts (applied $r_{\mathrm{eff}}$)"),
                            ("r_cpg", "dr", "CpG contexts")]:
        ax.plot(df["gc_mid"], df[col], marker=SERIES_MARKERS[key],
                color=SERIES_COLORS[key], markersize=5, linewidth=2, label=label)
    ax.plot(df["gc_mid"], df["r_counterfactual"], linestyle="--", linewidth=1.8,
            color="0.45", label=r"Counterfactual: non-CpG $r \equiv 1$")

    ax.axhline(1.0, **REF_LINE_KW)
    _log_ratio_axis(ax, df[["r_non", "r_eff", "r_cpg"]].to_numpy())
    _finish(ax, "Adjustment factor applied\nto expected counts, $r$", xrange, show_xlabel)


COMPOSITION_STYLE = [
    ("frac_analyzed", "#1baf7a", "In the analyzed noncoding genome"),
    ("frac_coding", "#eb6834", "Excluded: coding / failed QC"),
    ("frac_noannot", "0.72", "Excluded: no gnomAD coverage"),
]


def panel_training_composition(ax, comp, min_n: int = 500, xrange=(0.2, 0.73),
                               show_xlabel: bool = True) -> None:
    """
    Panel C. Where the non-CpG background training sites actually sit, as a stacked
    composition per GC bin. `comp` is data.dnm0_composition() output; the three
    fractions partition each bin exactly (asserted there), so the stack fills to 1.

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

    ax.stackplot(df["gc_mid"], *[df[c] for c, _, _ in COMPOSITION_STYLE],
                 colors=[c for _, c, _ in COMPOSITION_STYLE],
                 labels=[lab for _, _, lab in COMPOSITION_STYLE], alpha=0.9)
    ax.set_ylim(0, 1)
    _finish(ax, "Fraction of background\ntraining sites", xrange, show_xlabel,
            legend_loc="lower left", grid_axis="y")


PAIR_STYLE = {
    "full": {"key": "step2", "label": "original training set"},
    "scored": {"key": "scored", "label": "scored-population training set"},
    # Same NUMBER of sites as `scored`, drawn from the same population as `full`. It
    # should track `full`, and that it does is what rules out sample size.
    "sizematched": {"key": "control", "label": "size-matched random control"},
}


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
    _finish(ax, "P(DNM) relative to its own mean\n(non-CpG contexts)" if normalize
            else "P(DNM) in the training set\n(non-CpG contexts)",
            xrange, show_xlabel, legend_fontsize=LEGEND_FONTSIZE - 1)
