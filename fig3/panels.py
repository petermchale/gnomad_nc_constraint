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


def panel_dnm_probability(ax, binned, group: str = "non-CpG", min_n: int = 500,
                           gc_as_fraction: bool = True,
                           xrange: tuple[float, float] = (0.2, 0.73),
                           show_xlabel: bool = True, logy: bool = False) -> None:
    """
    The DNM model's reliability diagram on its own training set: mean fitted
    P(DNM) from the per-context logistic regressions against the empirical
    fraction of training examples in that GC bin that are DNMs.

    `binned` is dnm_model.bin_training_calibration(..., stratify_cpg=True)
    output (pandas); `group` selects which stratum to draw ("non-CpG" by
    default -- the contexts that retain GC_content and that r_eff.py shows
    carry the entire GC trend in the applied adjustment).

    Both curves are on the same case-control-sampled population (dnm0:dnm1 ~
    10:1 genome-wide), so their absolute level is NOT the genome-wide DNM
    rate. That is fine here because the comparison is within one population:
    the intercept bias the sampling induces is common to both curves and
    cancels in their difference. Do not read the y-axis as a mutation rate.

    Only the empirical curve carries error bars (binomial). mean_pred is a
    deterministic function of already-fixed feature vectors, not resampled.

    min_n drops GC bins holding fewer than that many training sites. The
    extreme bins hold as few as 1 site, where the empirical fraction is 0 or 1
    and carries no information.

    READ THIS BEFORE USING IT AS EVIDENCE OF GNOCCHI'S BIAS. This panel
    measures a LEVEL error in P(DNM). Gnocchi applies the RATIO
    r = sigma(b0 + b.z)/sigma(b0), in which a level error common to numerator
    and denominator cancels exactly and never reaches the constraint score --
    see CLAUDE.md, "Methylation, and why the training-set calibration panel
    measures the wrong thing". It is also measured over the WHOLE training
    population, whereas the E1 weights and per-context normalization used by
    panel_r_non_vs_empirical are computed over the analyzed window set; the two
    populations give materially different answers in the high-GC tail. This is
    a diagnostic of the fit, not a measurement of the bias.
    """
    df = binned[binned["group"] == group] if "group" in binned.columns else binned
    if min_n:
        n_before = len(df)
        df = df[df["n"] >= min_n]
        print(f"DNM probability panel: dropped {n_before - len(df)} GC bin(s) with n < {min_n:,}")
    df = df.sort_values("gc_mid")
    gc = df["gc_mid"] / 100.0 if gc_as_fraction else df["gc_mid"]

    ax.plot(gc, df["mean_pred"], marker=SERIES_MARKERS["step2"],
            color=SERIES_COLORS["step2"], markersize=5, linewidth=2,
            label="Fitted: logistic-regression P(DNM)")
    ax.errorbar(gc, df["empirical_prop"], yerr=df["se"],
                marker=SERIES_MARKERS["step1"], color=SERIES_COLORS["step1"],
                markersize=5, linewidth=2, capsize=3, elinewidth=1,
                label="Empirical: fraction of examples that are DNMs")

    if logy:
        ax.set_yscale("log")
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlim(xrange)
    ax.set_ylabel("P(DNM) in the training set\n(non-CpG contexts)"
                  if group == "non-CpG" else "P(DNM) in the training set",
                  fontsize=AXIS_LABEL_FONTSIZE)
    if show_xlabel:
        ax.set_xlabel("GC content", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(True, **GRID_KW)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=LEGEND_FONTSIZE, frameon=False, loc="upper left")


# Panel-A rungs of the representativeness figure. The two "analyzed windows" rungs
# share a hue and the two "whole genome" rungs share another, so the eye groups them
# by POPULATION -- which is the variable that turns out to matter -- rather than by
# denominator or aggregation, which do not.
LADDER_STYLE = {
    "model":    {"color": "#eb6834", "marker": "s"},   # orange, as in the other panels
    "analyzed": {"color": "#2a78d6", "marker": "o"},   # blue
    "genome":   {"color": "#4a3aa7", "marker": "D"},   # violet
}


def panel_population_ladder(ax, ladder, series, xrange: tuple[float, float] = (0.2, 0.73),
                            show_xlabel: bool = True) -> None:
    """
    The empirical non-CpG DNM probability built four ways, changing one ingredient
    at a time: denominator, window population, aggregation.

    `ladder` is training_representativeness.build_ladder() output (polars); `series`
    is its LADDER_SERIES list of (column, style key, dashed, label).

    Every curve is normalized to E1-weighted mean 1, so the y-axis is shape only.
    A log axis is used because these are ratios -- a 25% over- and under-statement
    should read as equal departures from 1.

    The figure's claim is carried by which curves superimpose: the two blue rungs
    differ only in denominator and the two violet rungs differ only in aggregation,
    while blue and violet differ only in which windows are measured.
    """
    df = ladder.to_pandas() if hasattr(ladder, "to_pandas") else ladder
    df = df.sort_values("gc_mid")

    for col, key, dashed, label in series:
        style = LADDER_STYLE[key]
        ax.plot(df["gc_mid"], df[col],
                marker=style["marker"], color=style["color"],
                markersize=5, linewidth=2,
                linestyle="--" if dashed else "-",
                markerfacecolor="white" if dashed else style["color"],
                label=label)

    ax.axhline(1.0, **REF_LINE_KW)
    _log_ratio_axis(ax, df[[c for c, *_ in series]].to_numpy(),
                    ticks=(0.8, 0.9, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0))
    ax.set_xlim(xrange)
    ax.set_ylabel("Empirical DNM probability\n(non-CpG, relative to its own mean)",
                  fontsize=AXIS_LABEL_FONTSIZE)
    if show_xlabel:
        ax.set_xlabel("GC content", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(True, **GRID_KW)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=LEGEND_FONTSIZE - 1, frameon=False, loc="upper left")


COMPOSITION_STYLE = [
    ("frac_analyzed", "#1baf7a", "In the analyzed noncoding genome"),
    ("frac_coding",   "#eb6834", "Excluded: coding / failed QC"),
    ("frac_noannot",  "0.72",    "Excluded: no gnomAD coverage"),
]


def panel_training_composition(ax, comp, min_n: int = 500,
                               xrange: tuple[float, float] = (0.2, 0.73),
                               show_xlabel: bool = True) -> None:
    """
    Where the DNM training set's background sites actually sit, as a stacked
    composition per GC bin: inside the analyzed noncoding genome, or excluded from it
    for being coding / failing QC, or absent from the constraint table altogether.

    `comp` is training_representativeness.dnm0_window_composition() output (polars).
    The three fractions partition each bin exactly, so the stack fills to 1.

    This is the explanation for panel A: the training population and the scored
    population coincide in the GC bulk and come apart in the GC-rich tail, so a model
    fit on the former and applied to the latter is extrapolating there.
    """
    df = comp.to_pandas() if hasattr(comp, "to_pandas") else comp
    if min_n:
        df = df[df["n_total"] >= min_n]
    # Clip to the plotted range rather than letting stackplot interpolate in from an
    # out-of-range bin: the lowest-GC bins are almost entirely uncovered sequence,
    # which would draw a steep edge artifact at x = xrange[0] where there is no bin.
    df = df[(df["gc_mid"] >= xrange[0]) & (df["gc_mid"] <= xrange[1])]
    df = df.sort_values("gc_mid")

    ax.stackplot(df["gc_mid"], *[df[c] for c, _, _ in COMPOSITION_STYLE],
                 colors=[c for _, c, _ in COMPOSITION_STYLE],
                 labels=[lab for _, _, lab in COMPOSITION_STYLE], alpha=0.9)

    ax.set_xlim(xrange)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of background\ntraining sites", fontsize=AXIS_LABEL_FONTSIZE)
    if show_xlabel:
        ax.set_xlabel("GC content", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(True, axis="y", **GRID_KW)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=LEGEND_FONTSIZE, frameon=False, loc="lower left")


def panel_r_fitted_vs_observed(ax, binned, fits, xrange: tuple[float, float] = (0.2, 0.73),
                               show_xlabel: bool = True) -> None:
    """
    Several fitted non-CpG adjustments against the adjustment the observed de novo
    mutations support.

    `binned` is compare_restricted.build_panel_b() output (polars); `fits` is a list of
    (label, style key, dashed, legend text), one per fitted curve, where label matches
    the r_non_model_{label} column.

    The observed curve is identical for every fit by construction -- same DNMs, same
    opportunities, same windows -- which the caller asserts rather than assumes, so the
    only thing that moves between fitted curves is the model. Including a SIZE-MATCHED
    fit here is what separates "trained on a better-matched population" from "trained
    on less data"; without it the comparison is confounded.

    Only the observed curve carries error bars; the fitted curves are deterministic
    functions of already-fixed feature values.
    """
    df = binned.to_pandas() if hasattr(binned, "to_pandas") else binned
    df = df.sort_values("gc_mid")

    for label, key, dashed, text in fits:
        ax.plot(df["gc_mid"], df[f"r_non_model_{label}"], marker=SERIES_MARKERS[key],
                color=SERIES_COLORS[key], markersize=5, linewidth=2,
                linestyle="--" if dashed else "-",
                markerfacecolor="white" if dashed else SERIES_COLORS[key], label=text)
    ax.errorbar(df["gc_mid"], df["r_non_empirical"], yerr=df["se_r_non_empirical"],
                marker=SERIES_MARKERS["step1"], color=SERIES_COLORS["step1"],
                markersize=5, linewidth=2, capsize=3, elinewidth=1,
                label="Observed de novo mutations")

    ax.axhline(1.0, **REF_LINE_KW)
    _log_ratio_axis(ax, df[[f"r_non_model_{lab}" for lab, *_ in fits]
                           + ["r_non_empirical"]].to_numpy())
    ax.set_xlim(xrange)
    ax.set_ylabel("Non-CpG adjustment factor, $r$", fontsize=AXIS_LABEL_FONTSIZE)
    if show_xlabel:
        ax.set_xlabel("GC content", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(True, **GRID_KW)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=LEGEND_FONTSIZE - 1, frameon=False, loc="upper left")


PAIR_STYLE = {
    "original":    {"color": "#eb6834", "fitted_marker": "s", "label": "original training set"},
    "scored":      {"color": "#1baf7a", "fitted_marker": "^", "label": "scored-population training set"},
    # The size-matched random control: same number of sites as "scored", drawn from the
    # same population as "original". It should track "original", and that it does is
    # what rules out sample size as the explanation.
    "sizematched": {"color": "#4a3aa7", "fitted_marker": "D", "label": "size-matched random control"},
}


def panel_dnm_probability_pairs(ax, binned: dict, min_n: int = 500,
                                gc_as_fraction: bool = True, normalize: bool = False,
                                xrange: tuple[float, float] = (0.2, 0.76),
                                show_xlabel: bool = True) -> None:
    """
    Fitted and empirical P(DNM) vs GC for two training populations at once, non-CpG.

    `binned` maps population name -> a table with n, gc_mid, mean_pred,
    empirical_prop, se (plot_dnm_probability_pairs.non_cpg_binned output). Color
    encodes the population, line style encodes fitted (solid, filled) vs empirical
    (dashed, open, with binomial error bars) -- so each pair reads as one reliability
    diagram and the two pairs are visually separable.

    LEVELS ARE NOT COMPARABLE ACROSS POPULATIONS unless normalize=True: the two sets
    have different case-control ratios (10.0 vs 11.3 background per DNM), which shifts
    P(DNM) by that factor for reasons unrelated to GC. Within a pair the comparison is
    exact, because the fitted and empirical curves come from the very same sites.
    """
    for name, df in binned.items():
        style = PAIR_STYLE[name]
        d = df[df["n"] >= min_n] if min_n else df
        d = d[(d["gc_mid"] / 100.0 >= xrange[0]) & (d["gc_mid"] / 100.0 <= xrange[1])]
        d = d.sort_values("gc_mid")
        gc = d["gc_mid"] / 100.0 if gc_as_fraction else d["gc_mid"]

        pred, emp, se = d["mean_pred"], d["empirical_prop"], d["se"]
        if normalize:
            wpred = np.average(pred, weights=d["n"])
            wemp = np.average(emp, weights=d["n"])
            pred, emp, se = pred / wpred, emp / wemp, se / wemp

        ax.plot(gc, pred, marker=style["fitted_marker"], color=style["color"],
                markersize=5, linewidth=2, label=f"fitted, {style['label']}")
        ax.errorbar(gc, emp, yerr=se, marker="o", color=style["color"],
                    markersize=5, linewidth=2, linestyle="--", markerfacecolor="white",
                    capsize=3, elinewidth=1, label=f"empirical, {style['label']}")

    ax.set_xlim(xrange)
    ax.set_ylabel("P(DNM) relative to its own mean\n(non-CpG contexts)" if normalize
                  else "P(DNM) in the training set\n(non-CpG contexts)",
                  fontsize=AXIS_LABEL_FONTSIZE)
    if normalize:
        ax.axhline(1.0, **REF_LINE_KW)
    if show_xlabel:
        ax.set_xlabel("GC content", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(True, **GRID_KW)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=LEGEND_FONTSIZE - 1, frameon=False, loc="upper left")


def label_panels(axes, labels=("A", "B"), x: float = -0.09, y: float = 1.02) -> None:
    """Bold A/B panel letters in axes coordinates, journal-style."""
    for ax, letter in zip(axes, labels):
        ax.text(x, y, letter, transform=ax.transAxes, fontsize=PANEL_LABEL_FONTSIZE,
                 fontweight="bold", va="bottom", ha="right")
