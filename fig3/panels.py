"""
The two panels of Fig. 3, as ax-accepting functions.

Panel A  GC-binned mean standardized rank of three constraint metrics on the
         same axes: step-1 (context-only, r=1) Gnocchi, full (r-adjusted)
         Gnocchi, and depletion rank. The point: step 1 is not merely LESS
         GC-biased than full Gnocchi, it is biased about as little as depletion
         rank -- so the bias is introduced by the regional-feature adjustment,
         not inherited from the sequence-context model.

Panel B  The multiplicative error in the adjustment factor the pipeline
         actually applies, r_model/r_true, stratified by CpG status
         (panel_calibration_ratio). Replaced the earlier pooled, absolute-gap
         version (panel_calibration_gap, still here) for two reasons:

         1. The pooled curve hid that the high-GC signal is almost entirely a
            CpG-context effect. CpG contexts are the only ones denied
            GC_content by FT_CORR_MET, and they rise from 0.9% of sites at
            GC 0.25 to 32% at GC 0.74.
         2. The absolute gap is not comparable across contexts whose baseline
            rates differ ~6x, and is not the quantity that perturbs expected
            counts. The ratio is both.

         READ THE CAVEAT in this module's panel_calibration_ratio docstring
         before drawing a causal arrow from panel B to panel A: in the GC bulk
         the measured errors are far too small to account for panel A's bias.

Panels share the x-axis, which only works because both are GC content of a 1kb
window on a 0-1 fraction scale: panel A bins Chen windows by GC_content_1k/100,
and panel B bins training SITES by the GC_content_1k regional feature at that
site (also the GC content of the surrounding 1kb, also /100). Same quantity,
different population.

Every function takes an `ax` and draws into it. Nothing here creates a figure,
sets a matplotlib backend, or writes a file -- the notebook owns all of that.
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import polars as pl

# Categorical slots 1-3 of the validated default palette (dataviz skill,
# references/palette.md), checked with scripts/validate_palette.js --mode light:
# all of lightness band, chroma floor, CVD separation (worst adjacent deutan
# dE 9.2) and normal-vision floor (27.6) PASS. The aqua's contrast vs surface
# is 2.74:1, a WARN that obligates relief -- discharged here by the legend plus
# distinct markers per series, so identity is never carried by color alone.
SERIES_COLORS = {
    "step1": "#2a78d6",   # blue
    "step2": "#eb6834",   # orange
    "dr": "#1baf7a",      # aqua
    "gap": "#4a3aa7",     # violet (slot 7) -- panel B, a different quantity
}
SERIES_MARKERS = {"step1": "o", "step2": "s", "dr": "^", "gap": "D"}

GRID_KW = dict(color="0.85", linewidth=0.6)
REF_LINE_KW = dict(color="0.45", linewidth=0.8, linestyle="--")

AXIS_LABEL_FONTSIZE = 12
TICK_LABEL_FONTSIZE = 11
LEGEND_FONTSIZE = 10
PANEL_LABEL_FONTSIZE = 14


def curve_from_binned(binned: pl.DataFrame, label: str, key: str, display: str) -> dict:
    """
    Turn one bin_by_gc() output column into the plain dict panel_rank_bias
    consumes. `key` selects the palette/marker slot ("step1"/"step2"/"dr");
    `label` is the mean_/se_ column suffix; `display` is the legend text.
    """
    return {
        "key": key,
        "display": display,
        "gc": binned["gc_mid"].to_numpy(),
        "mean": binned[f"mean_{label}"].to_numpy(),
        "se": binned[f"se_{label}"].to_numpy(),
        "n": binned["n"].to_numpy(),
    }


def panel_rank_bias(ax, curves: list[dict], gc_mean: float | None = None,
                     xrange: tuple[float, float] = (0.2, 0.73),
                     yrange: tuple[float, float] = (0.0, 1.0),
                     min_n: int = 0, show_xlabel: bool = True) -> None:
    """
    Panel A. Each curve is a dict from curve_from_binned(). Error bars are the
    within-bin standard error of the mean rank, se = std/sqrt(n) -- so they
    narrow exactly where a GC bin holds many windows and widen in the sparse
    GC tails, which is the "error bars reflecting the number of windows in each
    bin" the figure needs.

    min_n drops bins thinner than that many windows before plotting. The
    extreme GC bins can hold a handful of windows, where the mean rank is
    essentially noise; dropping them keeps the eye on the region that carries
    the claim. 0 (default) keeps every bin.

    The y=0.5 reference line is where an unbiased metric sits: the rank is
    uniform on (0,1) by construction, so a model with no GC-dependent bias has
    conditional mean rank 0.5 in every bin.

    yrange defaults to the full (0,1) that Fig. 2A uses, so the two figures are
    read on the same scale. The curves only occupy roughly 0.26-0.88 of it,
    which leaves visible dead space when this panel is stacked above panel B;
    tighten it (e.g. (0.2, 0.95)) if the stacked figure needs the room, at the
    cost of no longer being directly comparable to Fig. 2A by eye.
    """
    for c in curves:
        keep = c["n"] >= min_n if min_n else np.ones_like(c["n"], dtype=bool)
        ax.errorbar(
            c["gc"][keep], c["mean"][keep], yerr=c["se"][keep],
            marker=SERIES_MARKERS[c["key"]], color=SERIES_COLORS[c["key"]],
            markersize=5, linewidth=2, capsize=3, elinewidth=1,
            label=c["display"],
        )

    ax.axhline(0.5, **REF_LINE_KW)
    if gc_mean is not None:
        ax.axvline(gc_mean, color="0.45", linewidth=0.8)

    ax.set_xlim(xrange)
    ax.set_ylim(*yrange)
    ax.set_ylabel("Mean standardized rank\nof constraint metric", fontsize=AXIS_LABEL_FONTSIZE)
    if show_xlabel:
        ax.set_xlabel("GC content", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(True, **GRID_KW)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=LEGEND_FONTSIZE, frameon=False, loc="upper left")


def panel_calibration_gap(ax, binned_reliability, min_n: int = 0,
                           yscale: str = "symlog", linthresh: float = 0.005,
                           gc_as_fraction: bool = True,
                           xrange: tuple[float, float] = (0.2, 0.73),
                           show_xlabel: bool = True) -> None:
    """
    Panel B: the DNM model's calibration gap, mean fitted P(DNM) minus the
    empirical DNM rate, per GC bin, with the empirical rate's binomial SE as
    the error bar. That SE is the dominant sampling noise in the comparison --
    mean_pred is a deterministic function of each site's already-fixed feature
    vector, not itself resampled.

    `binned_reliability` is bin_training_reliability()'s output (pandas), i.e.
    the same table -mode reliability writes as
    training_reliability_binned.dnm_refit_*.txt. Its gc_mid is on this repo's
    native 0-100 percent scale, so gc_as_fraction divides by 100 to put panel B
    on panel A's axis. Leave it True whenever the panels share an x-axis.

    yscale: "symlog" by default, NOT "log". The gap changes sign across GC
    (negative through the bulk, strongly positive in the high-GC tail), and a
    plain log axis cannot render negative values at all -- it would silently
    drop exactly the bulk-GC bins that establish the model is well calibrated
    where the data is dense. symlog keeps both signs, compresses the large
    tail, and stays linear within +/-linthresh so the near-zero bulk does not
    blow up into visual noise. Pass yscale="linear" for a plain axis.
    """
    df = binned_reliability
    if min_n:
        n_before = len(df)
        df = df[df["n"] >= min_n].reset_index(drop=True)
        print(f"panel B: dropped {n_before - len(df)} bin(s) with n < {min_n:,}")
    df = df.sort_values("gc_mid")

    gc = df["gc_mid"] / 100.0 if gc_as_fraction else df["gc_mid"]
    gap = df["mean_pred"] - df["empirical_prop"]

    ax.errorbar(gc, gap, yerr=df["se"], marker=SERIES_MARKERS["gap"],
                 color=SERIES_COLORS["gap"], markersize=5, linewidth=2,
                 capsize=3, elinewidth=1,
                 label="DNM model: fitted $-$ empirical P(DNM)")

    ax.axhline(0, **REF_LINE_KW)
    if yscale == "symlog":
        ax.set_yscale("symlog", linthresh=linthresh)
    elif yscale != "linear":
        ax.set_yscale(yscale)

    ax.set_xlim(xrange)
    ax.set_ylabel("Calibration gap\n(predicted $-$ empirical)", fontsize=AXIS_LABEL_FONTSIZE)
    if show_xlabel:
        ax.set_xlabel("GC content", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(True, **GRID_KW)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=LEGEND_FONTSIZE, frameon=False, loc="upper left")


GROUP_STYLE = {
    # CpG contexts are the ones whose models are denied GC_content by FT_CORR_MET,
    # and the ones that concentrate at high GC -- they get the emphatic hue.
    "CpG": {"color": "#eb6834", "marker": "s", "label": "CpG contexts (ACG/CCG/GCG/TCG)"},
    "non-CpG": {"color": "#2a78d6", "marker": "o", "label": "non-CpG contexts (28)"},
    "all": {"color": "#4a3aa7", "marker": "D", "label": "all contexts pooled"},
}


def panel_calibration_ratio(ax, binned, min_n: int = 500,
                             gc_as_fraction: bool = True,
                             xrange: tuple[float, float] = (0.2, 0.73),
                             show_xlabel: bool = True, logy: bool = True) -> None:
    """
    Panel B, ratio form: the multiplicative error in the adjustment factor the
    pipeline actually applies,

        inflation = mean fitted P(DNM) / empirical P(DNM) = r_model / r_true,

    stratified by CpG status. See bin_training_calibration()'s docstring for
    why sigma(b0) cancels out of that ratio and why this is the right scale
    (the absolute gap is not comparable across contexts whose baseline rates
    differ by 6x).

    Reading: inflation > 1 means Gnocchi's expected count is inflated there,
    which depresses observed/expected and pushes its z (and rank) UP; < 1 does
    the opposite. The reference line is 1.0 -- no error.

    A log y-axis is the natural choice here and is available (unlike for the
    signed absolute gap, which needed symlog): the ratio is strictly positive,
    and a log axis makes a 2x over-prediction and a 2x under-prediction equal
    distances from the line, which is what "multiplicative error" means.

    Error bars are the delta-method SE of log(inflation), i.e.
    SE(empirical)/empirical, treating mean_pred as fixed.

    CAVEAT -- DO NOT DRAW A CAUSAL ARROW FROM THIS PANEL TO PANEL A IN THE GC
    BULK. Measured directly (2026-08-04) by applying a uniform inflation f to
    every window's expected count and re-ranking against the unperturbed
    genome-wide z distribution (1,843,559 windows, median expected 174):

        f     1.00   1.03   1.10   1.55   0.90   0.74
        rank  0.500  0.560  0.687  0.984  0.302  0.083

    So Gnocchi's rank is very sensitive to r: 10% inflation already moves the
    mean rank to 0.69. Panel A's r-adjustment contribution (step-2 rank minus
    step-1 rank) is ~+0.22 at GC 0.57, which needs f ~ 1.10-1.12. But this
    panel measures f ~ 0.98-1.00 there for BOTH groups -- an order of
    magnitude too small, and at GC 0.55-0.66 pointing the wrong way (f < 1).
    Only the top two GC bins (CpG f = 1.55, 2.29) are large enough to matter.

    The likely reason is that this panel is the wrong population for the
    question: it is measured IN-SAMPLE, on case-control-sampled training
    sites, at SITE-level feature vectors, whereas r(w) is applied
    OUT-OF-SAMPLE, genome-wide, at WINDOW-aggregated feature values. A
    genuinely causal panel B would measure r's error genome-wide at window
    level -- for which expected_counts_by_context_methyl_dnm_1M.txt and
    observed_counts_dnm_1M.txt in the bucket are the obvious (1 Mb, coarser,
    but out-of-sample and genome-wide) source. Not attempted yet.
    """
    df = binned.dropna(subset=["inflation"])
    if min_n:
        df = df[df["n"] >= min_n]

    for group, sub in df.groupby("group"):
        sub = sub.sort_values("gc_mid")
        style = GROUP_STYLE.get(group, GROUP_STYLE["all"])
        gc = sub["gc_mid"] / 100.0 if gc_as_fraction else sub["gc_mid"]
        # Symmetric in log space -> asymmetric in linear space, which is correct for a ratio.
        lo = sub["inflation"] * (1 - sub["se_log_inflation"])
        hi = sub["inflation"] * (1 + sub["se_log_inflation"])
        ax.errorbar(gc, sub["inflation"],
                     yerr=[sub["inflation"] - lo, hi - sub["inflation"]],
                     marker=style["marker"], color=style["color"], markersize=5,
                     linewidth=2, capsize=3, elinewidth=1, label=style["label"])

    ax.axhline(1.0, **REF_LINE_KW)
    if logy:
        ax.set_yscale("log")
        ax.set_yticks([0.5, 0.7, 1.0, 1.5, 2.0, 3.0])
        ax.set_yticklabels(["0.5", "0.7", "1.0", "1.5", "2.0", "3.0"])
        # A log axis also emits minor-tick labels ("6 x 10^-1" etc.), which collide
        # with the major labels set above; silence them.
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        lo_, hi_ = df["inflation"].min(), df["inflation"].max()
        ax.set_ylim(min(0.62, lo_ * 0.92), max(1.6, hi_ * 1.12))

    ax.set_xlim(xrange)
    ax.set_ylabel("Error in the adjustment factor\n" r"$r_{\mathrm{model}}\,/\,r_{\mathrm{true}}$",
                   fontsize=AXIS_LABEL_FONTSIZE)
    if show_xlabel:
        ax.set_xlabel("GC content", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(True, **GRID_KW)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=LEGEND_FONTSIZE, frameon=False, loc="upper left")


def _log_ratio_axis(ax, values, ticks=(0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5),
                     pad_lo: float = 0.96, pad_hi: float = 1.05) -> None:
    """
    Shared y-axis treatment for the adjustment-factor panels. r is a ratio, so a
    log axis is both available (strictly positive) and correct: a 25% over- and a
    25% under-adjustment should read as equal departures from the r = 1 line, which
    a linear axis would not show.
    """
    ax.set_yscale("log")
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    keep = [t for t in ticks if lo * pad_lo <= t <= hi * pad_hi]
    ax.set_yticks(keep)
    ax.set_yticklabels([f"{t:g}" for t in keep])
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_ylim(lo * pad_lo, hi * pad_hi)


def panel_r_eff_decomposition(ax, binned, min_n: int = 100,
                               xrange: tuple[float, float] = (0.2, 0.73),
                               show_xlabel: bool = True,
                               show_counterfactual: bool = True) -> None:
    """
    The effective per-window adjustment r_eff = E2/E1 versus GC content, split into
    its CpG and non-CpG parts.

    `binned` is r_eff.bin_r_eff()'s output (polars): gc_mid, n, r_eff, r_cpg,
    r_non, r_counterfactual.

    The three curves are an exact decomposition, r_eff = Pi*r_CpG + (1-Pi)*r_non,
    so the figure can be read additively. The counterfactual holds the non-CpG term
    at 1 and leaves everything else untouched; it is drawn as a dashed grey
    hypothetical rather than a fourth colored series, because it is not a
    measurement. Its flatness is the claim: the entire GC trend in what Gnocchi
    actually applies comes from the non-CpG contexts.
    """
    df = binned.to_pandas() if hasattr(binned, "to_pandas") else binned
    if min_n:
        df = df[df["n"] >= min_n]
    df = df.sort_values("gc_mid")

    series = [("r_non", "step2", "Non-CpG contexts"),
              ("r_eff", "gap", "All contexts (applied $r_{\\mathrm{eff}}$)"),
              ("r_cpg", "dr", "CpG contexts")]
    for col, key, label in series:
        ax.plot(df["gc_mid"], df[col], marker=SERIES_MARKERS[key],
                 color=SERIES_COLORS[key], markersize=5, linewidth=2, label=label)

    if show_counterfactual:
        ax.plot(df["gc_mid"], df["r_counterfactual"], linestyle="--", linewidth=1.8,
                 color="0.45", label="Counterfactual: non-CpG $r \\equiv 1$")

    ax.axhline(1.0, **REF_LINE_KW)
    _log_ratio_axis(ax, df[["r_non", "r_eff", "r_cpg"]].to_numpy())
    ax.set_xlim(xrange)
    ax.set_ylabel("Adjustment factor applied\nto expected counts, $r$",
                   fontsize=AXIS_LABEL_FONTSIZE)
    if show_xlabel:
        ax.set_xlabel("GC content", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(True, **GRID_KW)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=LEGEND_FONTSIZE, frameon=False, loc="upper left")


def panel_r_non_vs_empirical(ax, binned, min_dnm: int = 200,
                              xrange: tuple[float, float] = (0.2, 0.73),
                              show_xlabel: bool = True,
                              show_ratio_on: object = None) -> None:
    """
    The non-CpG adjustment Gnocchi applies, against the adjustment the observed de
    novo mutations actually support.

    `binned` is empirical_r.combine_non_cpg() + attach_gc_mid() output (polars):
    gc_mid, dnm_total, r_non_model, r_non_empirical, se_r_non_empirical, inflation.

    Only the empirical curve carries error bars -- it is a Poisson count over a
    fixed denominator, whereas the model curve is a deterministic function of
    already-fixed feature values. min_dnm drops GC bins holding fewer than that
    many observed DNMs, where the empirical estimate is noise.

    If show_ratio_on is a second axes, the model/empirical ratio is drawn there on
    a shared x-axis -- the quantity that states the size of the error directly.
    """
    df = binned.to_pandas() if hasattr(binned, "to_pandas") else binned
    if min_dnm and "dnm_total" in df.columns:
        n_before = len(df)
        df = df[df["dnm_total"].fillna(0) >= min_dnm]
        print(f"r_non panel: dropped {n_before - len(df)} GC bin(s) with < {min_dnm} DNMs")
    df = df.sort_values("gc_mid")

    ax.plot(df["gc_mid"], df["r_non_model"], marker=SERIES_MARKERS["step2"],
             color=SERIES_COLORS["step2"], markersize=5, linewidth=2,
             label="Gnocchi's fitted $r$ (non-CpG)")
    ax.errorbar(df["gc_mid"], df["r_non_empirical"], yerr=df["se_r_non_empirical"],
                 marker=SERIES_MARKERS["step1"], color=SERIES_COLORS["step1"],
                 markersize=5, linewidth=2, capsize=3, elinewidth=1,
                 label="Observed de novo mutations")

    ax.axhline(1.0, **REF_LINE_KW)
    _log_ratio_axis(ax, df[["r_non_model", "r_non_empirical"]].to_numpy())
    ax.set_xlim(xrange)
    ax.set_ylabel("Non-CpG adjustment factor, $r$", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(True, **GRID_KW)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=LEGEND_FONTSIZE, frameon=False, loc="upper left")

    if show_ratio_on is not None:
        rax = show_ratio_on
        rel = df["se_r_non_empirical"] / df["r_non_empirical"]
        rax.errorbar(df["gc_mid"], df["inflation"],
                      yerr=df["inflation"] * rel,
                      marker=SERIES_MARKERS["gap"], color=SERIES_COLORS["gap"],
                      markersize=5, linewidth=2, capsize=3, elinewidth=1,
                      label="Over-adjustment, fitted $/$ observed")
        rax.axhline(1.0, **REF_LINE_KW)
        _log_ratio_axis(rax, df["inflation"].to_numpy(),
                         ticks=(0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5))
        rax.set_xlim(xrange)
        rax.set_ylabel("Over-adjustment\n(fitted $/$ observed)", fontsize=AXIS_LABEL_FONTSIZE)
        rax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
        rax.grid(True, **GRID_KW)
        rax.set_axisbelow(True)
        for side in ("top", "right"):
            rax.spines[side].set_visible(False)
        rax.legend(fontsize=LEGEND_FONTSIZE, frameon=False, loc="upper left")
        ax = rax

    if show_xlabel:
        ax.set_xlabel("GC content", fontsize=AXIS_LABEL_FONTSIZE)


def label_panels(axes, labels=("A", "B"), x: float = -0.09, y: float = 1.02) -> None:
    """Bold A/B panel letters in axes coordinates, journal-style."""
    for ax, letter in zip(axes, labels):
        ax.text(x, y, letter, transform=ax.transAxes, fontsize=PANEL_LABEL_FONTSIZE,
                 fontweight="bold", va="bottom", ha="right")
