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
import subprocess

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

BUCKET_URL = "https://storage.googleapis.com/gnomad-nc-constraint-v31-paper"

REMOTE_FILES = {
    "step1_expected": "expected_counts_by_context_methyl_genome_1kb.txt",  # element_id, possible, expected  (step-1, r==1)
    "features": "misc/genomic_features13_genome_1kb.txt",                 # element_id, GC_content_1k, + 51 other cols
    "annot": "fig_tables/constraint_z_genome_1kb.annot.txt",              # element_id, possible, expected (step-2, r-adjusted), observed, oe, z, pass_qc, coding_prop, ...
}


def download(relpath: str, dest_dir: str) -> str:
    """
    curl `relpath` from BUCKET_URL into dest_dir if not already present locally.
    Return the local path. Streams straight to disk (curl, not a buffered
    Python download) given the file sizes involved (up to 1.44 GB), downloads
    to a .part sidecar first and renames on success so a half-finished
    download is never mistaken for a complete one on a later run.
    """
    local_path = os.path.join(dest_dir, os.path.basename(relpath))
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return local_path

    url = f"{BUCKET_URL}/{relpath}"
    tmp_path = local_path + ".part"
    subprocess.run(["curl", "-fL", "-o", tmp_path, url], check=True)
    os.rename(tmp_path, local_path)
    return local_path


def load_joined_table(local_paths: dict) -> pl.DataFrame:
    """
    Use duckdb to build the analysis table without loading full files into
    memory at once: column-pruned scans of the 1.44 GB features file and the
    325 MB annot file, inner-joined with the (already small) step-1 expected
    file on element_id, pulled out as polars via `.pl()` (not `.df()`).

    Return a polars DataFrame with columns:
      element_id, possible_step1, expected_step1, possible_step2,
      expected_step2, observed, pass_qc, coding_prop, GC_content_1k,
      z_published (the official, already-computed Gnocchi z, used only as a
      sanity check against this script's own from-scratch z_step2 -- see
      add_z_columns())
    """
    query = f"""
        SELECT
            s1.element_id      AS element_id,
            s1.possible         AS possible_step1,
            s1.expected         AS expected_step1,
            an.possible         AS possible_step2,
            an.expected         AS expected_step2,
            an.observed         AS observed,
            an.pass_qc          AS pass_qc,
            an.coding_prop      AS coding_prop,
            an.z                AS z_published,
            ft.GC_content_1k    AS GC_content_1k
        FROM read_csv_auto('{local_paths["step1_expected"]}', header=True) s1
        INNER JOIN (
            SELECT element_id, possible, expected, observed, pass_qc, coding_prop, z
            FROM read_csv_auto('{local_paths["annot"]}', header=True)
        ) an USING (element_id)
        INNER JOIN (
            SELECT element_id, GC_content_1k
            FROM read_csv_auto('{local_paths["features"]}', header=True)
        ) ft USING (element_id)
    """
    con = duckdb.connect()
    return con.execute(query).pl()


def exclude_sex_chromosomes(df: pl.DataFrame) -> pl.DataFrame:
    """
    Drop chrX/chrY windows (in practice, 2,497 chrX pseudoautosomal-region
    rows; chrY is already absent). See CLAUDE.md, "Chromosome filtering", for
    the Methods citation and why PAR-on-chrX is the only remnant possible.
    """
    chrom = df["element_id"].str.extract(r"^(chr[^-]+)-")
    return df.filter(~chrom.is_in(["chrX", "chrY"]))


def add_gc_content_fraction(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add GC_content = GC_content_1k / 100 (this repo's 0-100 percentage ->
    McHale et al.'s 0-1 fraction). See CLAUDE.md, "GC content units", for the
    citation trail (bedtools nuc's pct_gc column).
    """
    return df.with_columns((pl.col("GC_content_1k") / 100.0).alias("GC_content"))


def restrict_to_noncoding(df: pl.DataFrame, coding_prop_threshold: float = 0.0) -> pl.DataFrame:
    """
    Filter to noncoding 1kb windows (coding_prop <= threshold) -- half of
    McHale et al.'s "neutral" window definition. See
    restrict_to_neutral_genehancer() for the other half, and CLAUDE.md,
    "Noncoding restriction", for the Methods citation and the still-unconfirmed
    exact threshold.
    """
    return df.filter(pl.col("coding_prop") <= coding_prop_threshold)


def restrict_to_neutral_genehancer(
    df: pl.DataFrame,
    genehancer_bed_path: str | None,
    min_frac_overlap: float | None = None,
) -> pl.DataFrame:
    """
    Exclude windows overlapping a GeneHancer enhancer -- the other half of
    McHale et al.'s "neutral" definition. No-op (with a printed warning)
    unless genehancer_bed_path is given: GeneHancer isn't freely
    downloadable, so this can't run end-to-end automatically. See CLAUDE.md,
    "GeneHancer enhancer exclusion", for the full citation trail and why.

    genehancer_bed_path: a standard BED file (tab-separated, no header,
    chrom/start/end in the first three columns; extra columns ignored).
    min_frac_overlap: bedtools -f semantics; None (default) excludes on any
    overlap. UNTESTED -- no GeneHancer file is available in this environment;
    verify directly before relying on it for the rebuttal/paper.
    """
    if genehancer_bed_path is None:
        print(
            "WARNING: -genehancer_bed not given -- 'neutral' here is only "
            "noncoding + pass_qc (+ non-sex-chromosome), NOT excluding "
            "GeneHancer-enhancer-overlapping windows. See CLAUDE.md, "
            "'GeneHancer enhancer exclusion'."
        )
        return df

    windows = df.with_columns([
        pl.col("element_id").str.extract(r"^(chr[^-]+)-").alias("_chrom"),
        pl.col("element_id").str.extract(r"^chr[^-]+-(\d+)-").cast(pl.Int64).alias("_start"),
        pl.col("element_id").str.extract(r"^chr[^-]+-\d+-(\d+)$").cast(pl.Int64).alias("_end"),
    ])

    con = duckdb.connect()
    con.register("windows", windows.to_pandas())

    overlap_condition = "w._chrom = g.column0 AND w._start < g.column2 AND w._end > g.column1"
    if min_frac_overlap is not None:
        overlap_condition += f"""
            AND (LEAST(w._end, g.column2) - GREATEST(w._start, g.column1))::DOUBLE
                / (w._end - w._start) >= {min_frac_overlap}
        """

    query = f"""
        SELECT w.* EXCLUDE (_chrom, _start, _end)
        FROM windows w
        WHERE NOT EXISTS (
            SELECT 1
            FROM read_csv_auto('{genehancer_bed_path}', header=False) g
            WHERE {overlap_condition}
        )
    """
    return con.execute(query).pl()


def maybe_downsample(df: pl.DataFrame, frac: float | None, n: int | None, seed: int) -> pl.DataFrame:
    """
    Escape hatch for compute: if `frac` or `n` is given, randomly (uniformly)
    subsample `df` before binning; otherwise return df unchanged. At most one
    of frac/n may be set.
    """
    if frac is not None and n is not None:
        raise ValueError("specify at most one of -downsample_frac / -downsample_n")
    if frac is not None:
        return df.sample(fraction=frac, seed=seed)
    if n is not None:
        return df.sample(n=n, seed=seed)
    return df


def add_bias_columns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add per-window bias_step1 = expected_step1 - observed and
    bias_step2 = expected_step2 - observed, matching McHale et al.'s residual
    sign convention (predicted - y, not y - predicted). Used by bin_by_gc()
    for -bias_metric residual.
    """
    return df.with_columns([
        (pl.col("expected_step1") - pl.col("observed")).alias("bias_step1"),
        (pl.col("expected_step2") - pl.col("observed")).alias("bias_step2"),
    ])


def add_z_columns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute step-1 and step-2 z-scores from (expected, observed), using the
    *exact* formula in run_nc_constraint_gnomad_v31_main.py lines 278-280:
        oe = observed / expected
        chisq = (observed - expected)**2 / expected
        z = -sqrt(chisq) if oe >= 1 else sqrt(chisq)
    Adds z_step1 (from expected_step1, i.e. r==1) and z_step2 (from
    expected_step2, i.e. the real, r-adjusted Gnocchi expected count -- the
    official pipeline never computes a step-1-only z, so z_step1 is entirely
    self-computed here).

    Sanity check: prints the max |z_step2 - z_published| across all windows,
    where z_published is the official z column already in
    fig_tables/constraint_z_genome_1kb.annot.txt -- if this formula is right,
    the two should match almost exactly (up to floating-point/export-rounding
    noise), since both start from the same (expected_step2, observed) pair.

    Also replicates run_nc_constraint_gnomad_v31_main.py line 281's filtering
    (`df_z[df_z['z'].between(-10,10)].dropna()`): drops any window where
    EITHER z_step1 or z_step2 is outside [-10, 10] or non-finite, so step 1
    and step 2 are compared on an identical window population (apples-to-
    apples), not two differently-filtered sets.
    """
    def _z(expected_col: str, observed_col: str) -> pl.Expr:
        oe = pl.col(observed_col) / pl.col(expected_col)
        chisq = (pl.col(observed_col) - pl.col(expected_col)) ** 2 / pl.col(expected_col)
        return pl.when(oe >= 1).then(-chisq.sqrt()).otherwise(chisq.sqrt())

    df = df.with_columns([
        _z("expected_step1", "observed").alias("z_step1"),
        _z("expected_step2", "observed").alias("z_step2"),
    ])

    max_diff = (df["z_step2"] - df["z_published"]).abs().max()
    print(f"sanity check: self-computed z_step2 vs published z, "
          f"max |diff| across {df.height:,} windows = {max_diff}")

    df = df.filter(
        pl.col("z_step1").is_between(-10, 10) & pl.col("z_step1").is_finite()
        & pl.col("z_step2").is_between(-10, 10) & pl.col("z_step2").is_finite()
    )
    return df


def add_rank_columns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Standardized rank of z_step1 and z_step2, each in (0, 1) with mean
    exactly 0.5: rank = (rank(z) - 0.5) / n -- matches Figure 2's y-axis
    definition (see CLAUDE.md for the exact quoted caption text). Requires
    z_step1/z_step2 (see add_z_columns()).
    """
    n = df.height
    return df.with_columns([
        ((pl.col("z_step1").rank() - 0.5) / n).alias("rank_step1"),
        ((pl.col("z_step2").rank() - 0.5) / n).alias("rank_step2"),
    ])


def bin_by_gc(df: pl.DataFrame, gc_col: str, n_bins: int, bin_method: str, value_cols: dict[str, str]) -> pl.DataFrame:
    """
    Assign each window to a GC-content bin (fixed-width edges spanning the
    observed range of `gc_col`, or equal-count quantile edges), via
    numpy.digitize -- avoids depending on a specific polars cut()/qcut() API
    version.

    gc_col is the column to bin on: "GC_content" (0-1 fraction, paper units)
    in rank mode, or "GC_content_1k" (0-100 percentage, this repo's native
    units) in residual mode -- see add_gc_content_fraction().

    value_cols maps an output suffix to the per-window column to average
    within each bin -- e.g. {"bias_step1": "bias_step1", "bias_step2":
    "bias_step2"} for -bias_metric residual (see add_bias_columns()), or
    {"rank_step1": "rank_step1", "rank_step2": "rank_step2"} for
    -bias_metric rank (see add_rank_columns()). This lets one binning
    implementation serve both metrics without duplicating the digitize/
    group_by logic.

    Per bin, computes:
      - n      = window count
      - gc_mid = mean gc_col value in the bin (x-axis value for plotting)
      - for each (suffix, col) in value_cols:
          mean_{suffix} = mean(col) in the bin
          se_{suffix}   = std(col) / sqrt(n) in the bin
    (per-window value averaged, not a difference of per-bin averages, so the
    residual metric matches the "local bias" definition used in Supp Fig 1 /
    the simulation script; the rank metric matches Figure 2A's conditional-
    mean-rank line)

    Return the binned summary DataFrame (polars), one row per GC bin, sorted
    by gc_mid.
    """
    gc = df[gc_col].to_numpy()
    if bin_method == "quantile":
        edges = np.quantile(gc, np.linspace(0, 1, n_bins + 1))
    else:
        edges = np.linspace(gc.min(), gc.max(), n_bins + 1)
    edges = np.unique(edges)
    edges[-1] += 1e-9  # make the max value fall inside the last bin

    bin_idx = np.digitize(gc, edges[1:-1], right=False)
    df = df.with_columns(pl.Series("gc_bin", bin_idx))

    aggs = [pl.len().alias("n"), pl.col(gc_col).mean().alias("gc_mid")]
    for suffix, col in value_cols.items():
        aggs.append(pl.col(col).mean().alias(f"mean_{suffix}"))
        aggs.append((pl.col(col).std() / pl.len().sqrt()).alias(f"se_{suffix}"))

    binned = df.group_by("gc_bin").agg(aggs).sort("gc_mid")
    return binned


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


HEATMAP_LINE_COLOR = "0.9"  # light grey/near-white -- closer than plain dark grey to how
                             # the paper's line actually reads over its mostly dark
                             # hexbin cells; see CLAUDE.md, "Heat map".

RANK_YLABEL = "Standardized rank of constraint metric"
AXIS_LABEL_FONTSIZE = 13
TICK_LABEL_FONTSIZE = 14
TITLE_FONTSIZE = 16
LEGEND_FONTSIZE = 13


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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-dest_dir", default="tmp",
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
