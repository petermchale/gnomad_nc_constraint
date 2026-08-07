"""
Compare local (GC-binned) bias curves across the DNM training-set-size
experiment: step-1 (context-only, r==1), published step-2 (full Gnocchi,
the real published training set), and one or more resized-training-set
refits produced by run_dnm_training_experiment.py -mode refit.

See okf/dnm-training-set-experiment/pipeline.md steps 5-6, and CLAUDE.md's
"compute_gc_bias_step1_vs_step2.py" section for the shared Figure-2A-style
rank-statistic methodology this script reuses via
`import compute_gc_bias_step1_vs_step2 as base` (download/join logic,
window filters, GC-unit conversion, and GC-binning are NOT reimplemented
here -- see that module for the citation trail behind each of those
choices). Only the multi-curve (N > 2) generalization of the z/rank
computation is new here.

Usage, e.g. after two refit runs at 1%% and 10%% (regime 1):
  python plot_dnm_bias_comparison.py \\
      -refit_table output/expected_counts_by_context_methyl_genome_1kb.dnm_refit_frac0.01_seed0.txt:dnm_1pct \\
      -refit_table output/expected_counts_by_context_methyl_genome_1kb.dnm_refit_frac0.1_seed0.txt:dnm_10pct

Each -refit_table is "path:label" (repeatable); label is used in the plot
legend and as the column-name suffix.

Lives in dnm_training_experiment/ (moved here from the repo root 2026-07-21,
alongside run_dnm_training_experiment.py). Downloaded bucket files (step-1/
published-step-2 tables) are cached in the repo-root tmp/ (-cache_dir, shared
with compute_gc_bias_step1_vs_step2.py -- avoids re-downloading multi-GB
files); the output plot defaults to dnm_training_experiment/output/ next to
this script.
"""
import argparse
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import polars as pl

# The shared window-table/binning code lives in gnocchi_bias.windows at the
# repo root, one level up from this script's own directory -- add the repo root
# to sys.path so the import below works regardless of invocation CWD.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from gnocchi_bias import windows as base

DEFAULT_CACHE_DIR = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", "tmp"))
DEFAULT_OUTPUT_PLOT = os.path.join(_SCRIPT_DIR, "output", "dnm_training_set_size_bias.pdf")


def load_base_table(dest_dir: str) -> pl.DataFrame:
    local_paths = {k: base.download(v, dest_dir) for k, v in base.REMOTE_FILES.items()}
    return base.load_joined_table(local_paths)


def load_refit_table(path: str) -> pl.DataFrame:
    return pl.read_csv(path, separator='\t')


def add_z_column(df: pl.DataFrame, label: str, expected_col: str, observed_col: str = "observed") -> pl.DataFrame:
    """
    Same z-score formula as base.add_z_columns() (run_nc_constraint_gnomad_v31_main.py
    lines 278-280), generalized to an arbitrary number of named curves
    instead of hardcoded step1/step2.
    """
    oe = pl.col(observed_col) / pl.col(expected_col)
    chisq = (pl.col(observed_col) - pl.col(expected_col)) ** 2 / pl.col(expected_col)
    z = pl.when(oe >= 1).then(-chisq.sqrt()).otherwise(chisq.sqrt())
    return df.with_columns(z.alias(f"z_{label}"))


def main():
    matplotlib.use("Agg")  # set here, not at import time, so notebooks can import this module
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-cache_dir", default=DEFAULT_CACHE_DIR,
                         help="local dir for downloaded bucket files, shared with other scripts in this repo "
                              f"(default: repo-root tmp/, resolved as {DEFAULT_CACHE_DIR})")
    parser.add_argument("-refit_table", action="append", default=[], required=True,
                         help="path:label, repeatable -- one or more refit output tables from "
                              "run_dnm_training_experiment.py -mode refit")
    parser.add_argument("-n_bins", type=int, default=20)
    parser.add_argument("-bin_method", choices=["fixed", "quantile"], default="fixed")
    parser.add_argument("-apply_qc_filter", action="store_true", default=True,
                         help="restrict to pass_qc windows (matches published figures)")
    parser.add_argument("-no_qc_filter", dest="apply_qc_filter", action="store_false")
    parser.add_argument("-restrict_to_noncoding", action="store_true", default=True)
    parser.add_argument("-include_coding", dest="restrict_to_noncoding", action="store_false")
    parser.add_argument("-exclude_sex_chromosomes", action="store_true", default=True)
    parser.add_argument("-include_sex_chromosomes", dest="exclude_sex_chromosomes", action="store_false")
    parser.add_argument("-downsample_frac", type=float, default=None)
    parser.add_argument("-downsample_n", type=int, default=None)
    parser.add_argument("-random_seed", type=int, default=0)
    parser.add_argument("-xrange", default="0.2,0.73",
                         help="see compute_gc_bias_step1_vs_step2.py module docstring, 'Axis ranges' -- "
                              "a visual estimate from Figure 2A, not a value stated in the paper's text")
    parser.add_argument("-output_plot", default=DEFAULT_OUTPUT_PLOT)
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_plot)), exist_ok=True)
    xrange = tuple(float(v) for v in args.xrange.split(","))

    df = load_base_table(args.cache_dir)
    if args.exclude_sex_chromosomes:
        df = base.exclude_sex_chromosomes(df)
    if args.restrict_to_noncoding:
        df = base.restrict_to_noncoding(df)
    if args.apply_qc_filter:
        df = df.filter(pl.col("pass_qc"))

    curves = [("step1_context_only", "expected_step1"), ("step2_published", "expected_step2")]
    for spec in args.refit_table:
        path, label = spec.split(":", 1)
        refit = load_refit_table(path).rename({"possible": f"possible_{label}", "expected": f"expected_{label}"})
        n_before = df.height
        df = df.join(refit, on="element_id", how="inner")
        print(f"joined refit table '{label}' ({path}): {n_before:,} -> {df.height:,} windows")
        curves.append((label, f"expected_{label}"))

    df = base.maybe_downsample(df, args.downsample_frac, args.downsample_n, args.random_seed)
    df = base.add_gc_content_fraction(df)

    for label, expected_col in curves:
        df = add_z_column(df, label, expected_col)

    # Keep windows finite/in-range on EVERY curve, so all curves are compared
    # on an identical window population (apples-to-apples), same principle
    # as base.add_z_columns().
    for label, _ in curves:
        df = df.filter(pl.col(f"z_{label}").is_between(-10, 10) & pl.col(f"z_{label}").is_finite())

    n = df.height
    for label, _ in curves:
        df = df.with_columns(((pl.col(f"z_{label}").rank() - 0.5) / n).alias(f"rank_{label}"))

    value_cols = {label: f"rank_{label}" for label, _ in curves}
    binned = base.bin_by_gc(df, "GC_content", args.n_bins, args.bin_method, value_cols)

    gc_mean = df["GC_content"].mean()
    gc = binned["gc_mid"].to_numpy()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    for i, (label, _) in enumerate(curves):
        ax.errorbar(gc, binned[f"mean_{label}"].to_numpy(), yerr=binned[f"se_{label}"].to_numpy(),
                     marker=markers[i % len(markers)], capsize=3, label=label)
    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--")
    ax.axvline(gc_mean, color="black", linewidth=0.8) # type: ignore
    ax.set_xlim(xrange) # type: ignore
    ax.set_ylim(0, 1)
    ax.set_xlabel("GC content", fontsize=base.AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(base.RANK_YLABEL, fontsize=base.AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=base.TICK_LABEL_FONTSIZE)
    ax.set_title("Local bias vs GC content: effect of DNM training-set size", fontsize=base.TITLE_FONTSIZE)
    ax.legend(fontsize=base.LEGEND_FONTSIZE)
    fig.tight_layout()
    fig.savefig(args.output_plot)
    plt.close(fig)

    print(f"\nn_windows analyzed = {n:,}, mean GC content = {gc_mean:.4f}\n")
    print(binned)
    print(f"\nwrote {args.output_plot}")


if __name__ == "__main__":
    main()
