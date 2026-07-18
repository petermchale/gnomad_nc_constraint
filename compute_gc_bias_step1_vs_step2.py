"""
Compute local bias (expected - observed) as a function of GC content, for the
*same* real genome-wide 1kb windows, comparing:
  - step 1 (context-only, r == 1): expected count from sequence context alone.
  - step 2 (full Gnocchi, r as actually computed by the Chen et al. code):
    expected count after the regional-genomic-feature adjustment.

This is the real-data counterpart of the reviewer's request for a mechanistic
dissection of GC-content bias (see CLAUDE.md, "The analysis: real-data version
of the reviewer's request", and the companion simulation at
/Users/petermchale/rebuttal-simulation/simulate_constraint_bias.py).

NOTE (2026-07-15): restrict_to_noncoding() is implemented but OFF by default
(-restrict_to_noncoding not passed) -- per request, the first pass runs on all
1kb windows, coding included. Flip it on later once the exact noncoding
definition/threshold is confirmed against the McHale/Goldberg/Quinlan Methods.

NOTE (2026-07-15): bias sign convention matches McHale et al.'s simulation
(github.com/quinlan-lab/constraint-tools, papers/neutral_models_are_biased/
9.regression/fit_neutral_models.py): `residuals_{model}Model = predicted_y - y`,
i.e. bias = expected - observed (not observed - expected). No genome-wide
"global" bias is computed here -- McHale's compute_overall_model_bias() relies
on comparing the prediction to a known ground-truth true_rate(x), which only
exists in their simulation; real gnomAD data has no such ground truth, only
noisy observed counts, so there's no faithful real-data analog of that
particular metric. Only the GC-binned local bias (their "feature-specific
bias", the groupby(x_bin).mean(residual) line in plot_residuals.py) is ported.

Pipeline:
  1. Download plain-text files from the public bucket
     gs://gnomad-nc-constraint-v31-paper (no auth needed):
       - expected_counts_by_context_methyl_genome_1kb.txt   step-1 expected+possible per window (r==1)
       - misc/genomic_features13_genome_1kb.txt              GC_content_1k per window, among 51 other cols
       - fig_tables/constraint_z_genome_1kb.annot.txt         step-2 expected+observed+pass_qc+coding_prop
         per window (this file already carries `observed`, so the separate
         observed_counts_genome_1kb.txt file isn't needed once annot is loaded)
  2. Join all three on `element_id` using duckdb (column-pruned reads --
     avoid loading the full 1.44 GB / 325 MB files into memory).
  3. Optionally restrict to noncoding windows (off by default -- see note above),
     and optionally to pass_qc windows.
  4. Optionally downsample uniformly at random, for fast/rough iteration.
  5. Bin windows by GC_content_1k.
  6. Per bin, compute mean(expected - observed) for step 1 and step 2 -- i.e.
     average the per-window bias, not the difference of the per-bin averages
     -- to match how local bias is defined in Supp Fig 1 of
     McHale/Goldberg/Quinlan and in simulate_constraint_bias.py.
  7. Plot both curves vs GC-content bin (apples-to-apples comparison), with
     standard-error bars, and write out the binned summary table as CSV.
"""
import argparse
import os
import subprocess

import duckdb
import matplotlib
matplotlib.use("Agg")
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
      expected_step2, observed, pass_qc, coding_prop, GC_content_1k
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
            ft.GC_content_1k    AS GC_content_1k
        FROM read_csv_auto('{local_paths["step1_expected"]}', header=True) s1
        INNER JOIN (
            SELECT element_id, possible, expected, observed, pass_qc, coding_prop
            FROM read_csv_auto('{local_paths["annot"]}', header=True)
        ) an USING (element_id)
        INNER JOIN (
            SELECT element_id, GC_content_1k
            FROM read_csv_auto('{local_paths["features"]}', header=True)
        ) ft USING (element_id)
    """
    con = duckdb.connect()
    return con.execute(query).pl()


def restrict_to_noncoding(df: pl.DataFrame, coding_prop_threshold: float = 0.0) -> pl.DataFrame:
    """
    Filter to noncoding 1kb windows only, matching the McHale/Goldberg/Quinlan
    analysis (this repo's whole reason for existing -- see CLAUDE.md).

    TODO before turning this on: confirm the exact definition/threshold against
    the McHale/Goldberg/Quinlan Methods and/or
    /Users/petermchale/rebuttal-simulation/simulate_constraint_bias.py --
    default guess here is `coding_prop == 0.0` (fully noncoding windows only),
    but they may have used a small nonzero cutoff instead. Currently unused by
    default (see module docstring) -- only called if -restrict_to_noncoding
    is passed on the command line.
    """
    return df.filter(pl.col("coding_prop") <= coding_prop_threshold)


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
    sign convention (predicted - y, not y - predicted). Used by bin_by_gc().
    """
    return df.with_columns([
        (pl.col("expected_step1") - pl.col("observed")).alias("bias_step1"),
        (pl.col("expected_step2") - pl.col("observed")).alias("bias_step2"),
    ])


def bin_by_gc(df: pl.DataFrame, n_bins: int, bin_method: str) -> pl.DataFrame:
    """
    Assign each window to a GC-content bin (fixed-width edges spanning the
    observed GC_content_1k range, or equal-count quantile edges), via
    numpy.digitize -- avoids depending on a specific polars cut()/qcut() API
    version. Requires bias_step1/bias_step2 columns (see add_bias_columns()).

    Per bin, compute apples-to-apples step-1 vs step-2 local bias:
      - n               = window count
      - gc_mid          = mean GC_content_1k in the bin (x-axis value for plotting)
      - mean_bias_step1 = mean(expected_step1 - observed)
      - se_bias_step1   = std(expected_step1 - observed) / sqrt(n)
      - mean_bias_step2 = mean(expected_step2 - observed)
      - se_bias_step2   = std(expected_step2 - observed) / sqrt(n)
    (per-window difference averaged, not mean(expected) - mean(observed), so
    the metric matches the "local bias" definition used in Supp Fig 1 / the
    simulation script)

    Return the binned summary DataFrame (polars), one row per GC bin, sorted
    by gc_mid.
    """
    gc = df["GC_content_1k"].to_numpy()
    if bin_method == "quantile":
        edges = np.quantile(gc, np.linspace(0, 1, n_bins + 1))
    else:
        edges = np.linspace(gc.min(), gc.max(), n_bins + 1)
    edges = np.unique(edges)
    edges[-1] += 1e-9  # make the max value fall inside the last bin

    bin_idx = np.digitize(gc, edges[1:-1], right=False)
    df = df.with_columns(pl.Series("gc_bin", bin_idx))

    binned = (
        df.group_by("gc_bin")
        .agg([
            pl.len().alias("n"),
            pl.col("GC_content_1k").mean().alias("gc_mid"),
            pl.col("bias_step1").mean().alias("mean_bias_step1"),
            (pl.col("bias_step1").std() / pl.len().sqrt()).alias("se_bias_step1"),
            pl.col("bias_step2").mean().alias("mean_bias_step2"),
            (pl.col("bias_step2").std() / pl.len().sqrt()).alias("se_bias_step2"),
        ])
        .sort("gc_mid")
    )
    return binned


def plot_bias(binned: pl.DataFrame, output_path: str) -> None:
    """
    Plot mean_bias_step1 and mean_bias_step2 vs gc_mid (with se error bars) on
    the same axes, for a direct apples-to-apples comparison of local bias
    between the context-only and full Gnocchi models.
    """
    gc = binned["gc_mid"].to_numpy()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(gc, binned["mean_bias_step1"].to_numpy(), yerr=binned["se_bias_step1"].to_numpy(),
                marker="o", capsize=3, label="context-only (step 1, r=1)")
    ax.errorbar(gc, binned["mean_bias_step2"].to_numpy(), yerr=binned["se_bias_step2"].to_numpy(),
                marker="s", capsize=3, label="full Gnocchi (step 2, r-adjusted)")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("GC content (1kb window)")
    ax.set_ylabel("Expected - observed")
    ax.set_title("Local mutational bias vs GC content")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-dest_dir", default="tmp",
                         help="local directory to download bucket files into")
    parser.add_argument("-n_bins", type=int, default=20)
    parser.add_argument("-bin_method", choices=["fixed", "quantile"], default="fixed")
    parser.add_argument("-apply_qc_filter", action="store_true", default=True,
                         help="restrict to pass_qc windows (matches published figures)")
    parser.add_argument("-no_qc_filter", dest="apply_qc_filter", action="store_false")
    parser.add_argument("-restrict_to_noncoding", action="store_true", default=False,
                         help="restrict to noncoding windows (off by default for now -- see module docstring)")
    parser.add_argument("-downsample_frac", type=float, default=None,
                         help="randomly (uniformly) keep this fraction of windows, for a fast/rough result")
    parser.add_argument("-downsample_n", type=int, default=None,
                         help="randomly (uniformly) keep this many windows, for a fast/rough result")
    parser.add_argument("-random_seed", type=int, default=0)
    parser.add_argument("-output_csv", default="gc_bias_step1_vs_step2.csv")
    parser.add_argument("-output_plot", default="gc_bias_step1_vs_step2.pdf")
    args = parser.parse_args()

    os.makedirs(args.dest_dir, exist_ok=True)

    local_paths = {k: download(v, args.dest_dir) for k, v in REMOTE_FILES.items()}

    df = load_joined_table(local_paths)
    if args.restrict_to_noncoding:
        df = restrict_to_noncoding(df)
    if args.apply_qc_filter:
        df = df.filter(pl.col("pass_qc"))
    df = maybe_downsample(df, args.downsample_frac, args.downsample_n, args.random_seed)
    df = add_bias_columns(df)

    binned = bin_by_gc(df, args.n_bins, args.bin_method)

    binned.write_csv(args.output_csv)
    plot_bias(binned, args.output_plot)

    print(binned)


if __name__ == "__main__":
    main()
