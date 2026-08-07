"""
Plot the training-set reliability-diagram calibration gap (mean_pred -
empirical_prop) as a function of GC content, with the empirical rate's
binomial SE as error bars.

Motivation: training_reliability.dnm_refit_*.pdf overlays mean_pred and
empirical_prop as two separate lines -- small, systematic gaps between them
(e.g. a few tenths of a percentage point, but backed by n in the hundreds of
thousands) are easy to miss by eye on a 0-0.4 probability axis. Plotting the
gap itself, with error bars, makes it visually clear whether a "well
calibrated"-looking region is actually flat at zero or just small relative
to the plot's y-range.

The error bar is the empirical rate's own binomial SE (already computed by
run_dnm_training_experiment.py -mode reliability, sqrt(p(1-p)/n)) -- the
dominant source of sampling noise in the comparison, since mean_pred is a
deterministic function of each site's already-fixed feature vector, not
itself resampled at plot time.

Usage:
  python plot_reliability_gap.py -binned_table output/training_reliability_binned.dnm_refit_full.txt
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-binned_table", required=True,
                         help="training_reliability_binned.dnm_refit_*.txt from -mode reliability")
    parser.add_argument("-output_plot", default=None)
    parser.add_argument("-gc_as_percent", dest="gc_as_fraction", action="store_false", default=True,
                         help="plot GC on its native 0-100 scale instead of converting to a 0-1 fraction")
    parser.add_argument("-min_n", type=int, default=0,
                         help="drop bins with fewer than this many sites (tiny-n bins have huge SE and can "
                              "dominate the y-axis, obscuring the well-populated region)")
    args = parser.parse_args()

    df = pd.read_csv(args.binned_table, sep="\t").sort_values("gc_mid").reset_index(drop=True)
    if args.min_n:
        n_before = len(df)
        df = df[df["n"] >= args.min_n].reset_index(drop=True)
        print(f"dropped {n_before - len(df)} bin(s) with n < {args.min_n}")
    gc = df["gc_mid"] / 100.0 if args.gc_as_fraction else df["gc_mid"]
    gap = df["mean_pred"] - df["empirical_prop"]

    output_plot = args.output_plot or args.binned_table.replace(
        "training_reliability_binned", "training_reliability_gap").replace(".txt", ".pdf")
    os.makedirs(os.path.dirname(os.path.abspath(output_plot)), exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.errorbar(gc, gap, yerr=df["se"], fmt="o-", color="crimson", capsize=3, markersize=4, linewidth=1,
                label="mean_pred - empirical_prop (+/- 1 SE)")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    for x, y, n in zip(gc, gap, df["n"]):
        ax.annotate(f"n={n:,}", (x, y), textcoords="offset points", xytext=(0, 9), fontsize=6, ha="center")
    xlabel = "GC content (fraction)" if args.gc_as_fraction else "GC content (%)"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("calibration gap (predicted - empirical probability)")
    ax.set_title(f"Training-set reliability gap vs GC content\n({os.path.basename(args.binned_table)})")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_plot)
    plt.close(fig)

    out = df.assign(gc=gc, gap=gap, gap_se_ratio=gap / df["se"])
    print(out[["gc", "n", "mean_pred", "empirical_prop", "gap", "se", "gap_se_ratio"]].to_string(index=False))
    print(f"\nwrote {output_plot}")


if __name__ == "__main__":
    main()
