"""
Compute local bias as a function of GC content, for the *same* real
genome-wide 1kb windows, comparing:
  - step 1 (context-only, r == 1): expected count from sequence context alone.
  - step 2 (full Gnocchi, r as actually computed by the Chen et al. code):
    expected count after the regional-genomic-feature adjustment.

This is the real-data counterpart of the reviewer's request for a mechanistic
dissection of GC-content bias (see CLAUDE.md, "The analysis: real-data version
of the reviewer's request", and the companion simulation at
/Users/petermchale/rebuttal-simulation/simulate_constraint_bias.py).

FULL METHODS NARRATIVE, WITH CITATIONS, FOR THE REBUTTAL/REVISED PAPER: see
CLAUDE.md, "compute_gc_bias_step1_vs_step2.py -- Figure 2A-style rank-based
bias analysis". This docstring and the ones below only summarize mechanics;
every quoted-Methods-text citation, empirical measurement, and caveat lives
in that section so it isn't duplicated in two places.

Two bias metrics are supported, selected with -bias_metric (default: rank):

`-bias_metric rank` (default): reproduces the statistic plotted in Figure 2A
  of McHale et al. 2026 (mchale_et_al_250115.pdf), generalized to compare
  step 1 vs step 2 on the same axes. Per window: compute a z-score from
  (expected, observed) using the exact formula in
  run_nc_constraint_gnomad_v31_main.py lines 278-281 (see add_z_columns()),
  standardize to a rank in (0,1) via (rank(z)-0.5)/n (see add_rank_columns()),
  then bin by GC content and take the mean rank per bin -- Figure 2A's
  conditional-mean-rank line, plotted with a hexbin density heat map behind
  it (see plot_bias_rank()).

`-bias_metric residual`: the original metric this script started with, kept
  for backward compatibility (not part of Figure 2A). Per-window
  bias = expected - observed (McHale et al.'s residual sign convention),
  averaged per GC bin -- the "feature-specific bias" of Supp Fig 1 of
  McHale/Goldberg/Quinlan and simulate_constraint_bias.py's
  plot_residuals.py. No heat map, no [0,1] y-range, no paper x-range.

Pipeline:
  1. Download plain-text files from the public bucket
     gs://gnomad-nc-constraint-v31-paper (no auth needed):
       - expected_counts_by_context_methyl_genome_1kb.txt   step-1 expected+possible per window (r==1)
       - misc/genomic_features13_genome_1kb.txt              GC_content_1k per window, among 51 other cols
       - fig_tables/constraint_z_genome_1kb.annot.txt         step-2 expected+observed+pass_qc+coding_prop+z
         per window (this file already carries `observed` and the published
         `z`, so the separate observed_counts_genome_1kb.txt file isn't
         needed once annot is loaded)
  2. Join all three on `element_id` using duckdb (column-pruned reads --
     avoid loading the full 1.44 GB / 325 MB files into memory).
  3. Optionally exclude chrX/chrY windows (on by default), restrict to
     noncoding windows (on by default), restrict to pass_qc windows (on by
     default), and restrict to GeneHancer-non-overlapping windows (off by
     default -- needs a local file, see above).
  4. Optionally downsample uniformly at random, for fast/rough iteration.
  5. Depending on -bias_metric:
       rank (default): per-window z (see above) for step 1 and step 2, self-
             checked against the published z, then standardized to a rank in
             (0,1).
       residual: per-window expected-observed for step 1 and step 2.
  6. Bin windows by GC content (converted to paper units in rank mode); per
     bin, average the per-window metric (not the difference of per-bin
     averages) for step 1 and step 2.
  7. Plot both curves vs GC-content bin (apples-to-apples comparison), with
     standard-error bars (rank mode: plus the heat map, [0,1] y-range, paper
     x-range) -- PDF output only, no CSV is written (the binned summary table
     is still printed to stdout).
"""

import argparse
import os

import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

# The download/join/filter/z/rank/bin machinery, and the shared plot-style
# constants, now live in gnocchi_bias.windows so that the Fig. 3 notebook
# (fig5/) and the training-set-size experiment (dnm_training_size/) can import
# the same code instead of copying it. This script is unchanged as a CLI: it
# still owns the argument parsing and the plotting below.
from gnocchi_bias.windows import (  # noqa: F401  (re-exported for existing importers)
    BUCKET_URL, REMOTE_FILES,
    download, load_joined_table,
    exclude_sex_chromosomes, add_gc_content_fraction,
    restrict_to_noncoding, restrict_to_neutral_genehancer,
    maybe_downsample, add_bias_columns, add_z_columns, bin_by_gc,
    HEATMAP_LINE_COLOR, RANK_YLABEL, AXIS_LABEL_FONTSIZE,
    TICK_LABEL_FONTSIZE, TITLE_FONTSIZE, LEGEND_FONTSIZE,
)
from gnocchi_bias.windows import add_rank_columns as _add_rank_columns


def add_rank_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Two-curve wrapper: adds rank_step1/rank_step2 (see gnocchi_bias.windows)."""
    return _add_rank_columns(df, ["step1", "step2"])


def plot_bias_residual(binned: pl.DataFrame, output_path: str) -> None:
    """
    -bias_metric residual plot: mean_bias_step1 and mean_bias_step2 vs
    gc_mid (this repo's native 0-100 GC_content_1k units), with SE error
    bars, horizontal reference line at y=0. Not part of Figure 2A -- see
    plot_bias_rank() for that.
    """
    gc = binned["gc_mid"].to_numpy()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(gc, binned["mean_bias_step1"].to_numpy(), yerr=binned["se_bias_step1"].to_numpy(),
                marker="o", capsize=3, label="context-only (step 1, r=1)")
    ax.errorbar(gc, binned["mean_bias_step2"].to_numpy(), yerr=binned["se_bias_step2"].to_numpy(),
                marker="s", capsize=3, label="full Gnocchi (step 2, r-adjusted)")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("GC content (1kb window, %)")
    ax.set_ylabel("Expected - observed")
    ax.set_title("Local mutational bias vs GC content")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_rank_heatmap_panel(ax, gc: np.ndarray, rank: np.ndarray, binned_gc: np.ndarray,
                              binned_mean: np.ndarray, gc_mean: float, title: str,
                              xrange: tuple[float, float], gridsize: int) -> None:
    """
    One Figure-2A-style panel: hexbin density heat map (log-scaled 'inferno')
    with the conditional-mean-rank line, 0.5 horizontal line, and gc_mean
    vertical line on top. y fixed to [0,1]; x fixed to `xrange`. See
    CLAUDE.md, "Heat map" / "Axis ranges", for the paper citations behind
    each choice.
    """
    hb = ax.hexbin(gc, rank, gridsize=gridsize, cmap="inferno",
                    norm=matplotlib.colors.LogNorm(vmin=1), mincnt=1,
                    extent=(xrange[0], xrange[1], 0, 1))
    cbar = plt.colorbar(hb, ax=ax)
    cbar.set_label("Number of windows", fontsize=AXIS_LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)

    ax.plot(binned_gc, binned_mean, color=HEATMAP_LINE_COLOR, linewidth=2, marker="o", markersize=3)
    ax.axhline(0.5, color="black", linewidth=0.8)
    ax.axvline(gc_mean, color="black", linewidth=0.8)

    ax.set_xlim(xrange)
    ax.set_ylim(0, 1)
    ax.set_xlabel("GC content", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(RANK_YLABEL, fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)


def _plot_rank_overlay_panel(ax, binned: pl.DataFrame, gc_mean: float, xrange: tuple[float, float]) -> None:
    """
    Third panel: step-1 and step-2 conditional-mean-rank lines overlaid on
    shared axes, with SE error bars, no heat map -- for a direct read of how
    much the two models' bias differs at a glance (the two heat map panels
    make this comparison hard to eyeball since they're on separate axes).
    Same [0,1] y-range and paper-matched x-range as the heat map panels.
    """
    gc = binned["gc_mid"].to_numpy()
    ax.errorbar(gc, binned["mean_rank_step1"].to_numpy(), yerr=binned["se_rank_step1"].to_numpy(),
                marker="o", capsize=3, label="context-only (step 1, r=1)")
    ax.errorbar(gc, binned["mean_rank_step2"].to_numpy(), yerr=binned["se_rank_step2"].to_numpy(),
                marker="s", capsize=3, label="full Gnocchi (step 2, r-adjusted)")
    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--")
    ax.axvline(gc_mean, color="black", linewidth=0.8)
    ax.set_xlim(xrange)
    ax.set_ylim(0, 1)
    ax.set_xlabel("GC content", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(RANK_YLABEL, fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.set_title("Step 1 vs step 2 (overlay, no heat map)", fontsize=TITLE_FONTSIZE)
    ax.legend(fontsize=LEGEND_FONTSIZE)


def plot_bias_rank(df: pl.DataFrame, binned: pl.DataFrame, output_path: str,
                    gc_mean: float, xrange: tuple[float, float], gridsize: int,
                    plot_heatmap: bool) -> None:
    """
    -bias_metric rank plot, reproducing Figure 2A's style (page 6,
    mchale_et_al_250115.pdf). If plot_heatmap (default True): three panels --
    step-1 heat map, step-2 heat map (see _plot_rank_heatmap_panel(); the
    paper only plots one model, here both are shown for direct comparison),
    and a third panel overlaying both models' conditional-mean-rank lines
    with no heat map, for an easy side-by-side read of the two (see
    _plot_rank_overlay_panel()). If plot_heatmap is False: just the third
    (overlay, no heat map) panel, as a standalone plot. All panels share the
    fixed [0,1] y-range and the paper-matched x-range.
    """
    if not plot_heatmap:
        fig, ax = plt.subplots(figsize=(7, 5))
        _plot_rank_overlay_panel(ax, binned, gc_mean, xrange)
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
        return

    gc = df["GC_content"].to_numpy()
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    _plot_rank_heatmap_panel(
        axes[0], gc, df["rank_step1"].to_numpy(),
        binned["gc_mid"].to_numpy(), binned["mean_rank_step1"].to_numpy(),
        gc_mean, "Step 1: context-only (r=1)", xrange, gridsize)
    _plot_rank_heatmap_panel(
        axes[1], gc, df["rank_step2"].to_numpy(),
        binned["gc_mid"].to_numpy(), binned["mean_rank_step2"].to_numpy(),
        gc_mean, "Step 2: full Gnocchi (r-adjusted)", xrange, gridsize)
    _plot_rank_overlay_panel(axes[2], binned, gc_mean, xrange)
    fig.suptitle("Local bias vs GC content (Figure 2A style of McHale et al. 2026)", fontsize=TITLE_FONTSIZE)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main():
    # Set the non-interactive backend here rather than at import time: this
    # module's plotters are imported by fig5/'s notebook, which needs its own
    # interactive backend to render inline.
    matplotlib.use("Agg")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-dest_dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "published"),
                         help="local directory to download bucket files into")
    parser.add_argument("-bias_metric", choices=["rank", "residual"], default="rank",
                         help="'rank' (default): standardized rank of each window's own z-score per "
                              "GC bin, matching Figure 2A of McHale et al. 2026 (mchale_et_al_250115.pdf) "
                              "-- see module docstring. 'residual': mean(expected-observed) per GC bin -- "
                              "the original metric, matches already-logged CLAUDE.md results, not part "
                              "of Figure 2A.")
    parser.add_argument("-n_bins", type=int, default=20)
    parser.add_argument("-bin_method", choices=["fixed", "quantile"], default="fixed")
    parser.add_argument("-apply_qc_filter", action="store_true", default=True,
                         help="restrict to pass_qc windows (matches published figures)")
    parser.add_argument("-no_qc_filter", dest="apply_qc_filter", action="store_false")
    parser.add_argument("-restrict_to_noncoding", action="store_true", default=True,
                         help="restrict to noncoding windows (on by default -- half of McHale et al.'s "
                              "'neutral' window definition, see module docstring)")
    parser.add_argument("-include_coding", dest="restrict_to_noncoding", action="store_false")
    parser.add_argument("-exclude_sex_chromosomes", action="store_true", default=True,
                         help="drop chrX/chrY windows, matching McHale et al.'s Methods (on by default)")
    parser.add_argument("-include_sex_chromosomes", dest="exclude_sex_chromosomes", action="store_false")
    parser.add_argument("-genehancer_bed", default=None,
                         help="local GeneHancer BED file, to complete McHale et al.'s 'neutral' window "
                              "definition by excluding enhancer-overlapping windows -- see "
                              "restrict_to_neutral_genehancer()'s docstring. Off (None) by default: "
                              "GeneHancer cannot be downloaded automatically (see module docstring).")
    parser.add_argument("-genehancer_min_frac_overlap", type=float, default=None,
                         help="if given with -genehancer_bed, only exclude a window when the overlapping "
                              "GeneHancer interval covers at least this fraction of it (bedtools -f "
                              "semantics); default None = any overlap excludes the window")
    parser.add_argument("-match_paper_gc_units", action="store_true", default=True,
                         help="rank mode only: convert GC_content_1k (0-100%%) to a 0-1 fraction, "
                              "matching McHale et al.'s bedtools-nuc-derived units (on by default -- "
                              "see module docstring's 'GC content units' section)")
    parser.add_argument("-no_match_paper_gc_units", dest="match_paper_gc_units", action="store_false")
    parser.add_argument("-xrange", default="0.2,0.73",
                         help="rank mode only: x-axis range, visually matched to Figure 2A -- see module "
                              "docstring's 'Axis ranges' section for the important caveat that this is a "
                              "visual estimate, not a value stated in the paper's text")
    parser.add_argument("-plot_heatmap", action="store_true", default=True,
                         help="rank mode only: draw the Figure-2A-style hexbin density heat map behind "
                              "the conditional-mean-rank line (on by default)")
    parser.add_argument("-no_plot_heatmap", dest="plot_heatmap", action="store_false")
    parser.add_argument("-hexbin_gridsize", type=int, default=50,
                         help="rank mode heat map only: matplotlib hexbin gridsize")
    parser.add_argument("-downsample_frac", type=float, default=None,
                         help="randomly (uniformly) keep this fraction of windows, for a fast/rough result")
    parser.add_argument("-downsample_n", type=int, default=None,
                         help="randomly (uniformly) keep this many windows, for a fast/rough result")
    parser.add_argument("-random_seed", type=int, default=0)
    parser.add_argument("-output_plot", default=None,
                         help="default: gc_bias_step1_vs_step2.rank.pdf (rank) or "
                              "gc_bias_step1_vs_step2.pdf (residual)")
    args = parser.parse_args()

    if args.output_plot is None:
        args.output_plot = ("gc_bias_step1_vs_step2.rank.pdf" if args.bias_metric == "rank"
                             else "gc_bias_step1_vs_step2.pdf")
    xrange = tuple(float(v) for v in args.xrange.split(","))

    os.makedirs(args.dest_dir, exist_ok=True)

    local_paths = {k: download(v, args.dest_dir) for k, v in REMOTE_FILES.items()}

    df = load_joined_table(local_paths)
    if args.exclude_sex_chromosomes:
        df = exclude_sex_chromosomes(df)
    if args.restrict_to_noncoding:
        df = restrict_to_noncoding(df)
    if args.apply_qc_filter:
        df = df.filter(pl.col("pass_qc"))
    df = restrict_to_neutral_genehancer(df, args.genehancer_bed, args.genehancer_min_frac_overlap)
    df = maybe_downsample(df, args.downsample_frac, args.downsample_n, args.random_seed)

    if args.bias_metric == "rank":
        if args.match_paper_gc_units:
            df = add_gc_content_fraction(df)
            gc_col = "GC_content"
        else:
            df = df.with_columns(pl.col("GC_content_1k").alias("GC_content"))
            gc_col = "GC_content"
        df = add_z_columns(df)
        df = add_rank_columns(df)
        gc_mean = df[gc_col].mean()
        binned = bin_by_gc(df, gc_col, args.n_bins, args.bin_method,
                            value_cols={"rank_step1": "rank_step1", "rank_step2": "rank_step2"})
        plot_bias_rank(df, binned, args.output_plot, gc_mean, xrange, args.hexbin_gridsize, args.plot_heatmap) # type: ignore
    else:
        df = add_bias_columns(df)
        binned = bin_by_gc(df, "GC_content_1k", args.n_bins, args.bin_method,
                            value_cols={"bias_step1": "bias_step1", "bias_step2": "bias_step2"})
        plot_bias_residual(binned, args.output_plot)

    print(binned)


if __name__ == "__main__":
    main()
