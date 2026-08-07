"""
Reliability diagram for the DNM model, non-CpG contexts only.

Plots, per GC bin over the DNM training set:

  fitted     mean P(DNM) predicted by the per-context multivariate logistic
             regressions, evaluated on each training site's OWN feature vector
  empirical  the fraction of that bin's training examples (dnm1 + dnm0) that
             are DNMs

    python fig3/plot_dnm_probability.py [-group non-CpG] [-min_n 500]

Reads the per-site prediction table -mode reliability already wrote
(training_reliability_predictions.dnm_refit_full.txt); nothing is refit.

WHAT THIS IS AND IS NOT. It is an in-sample check of how well the DNM model
fits the GC dependence of its own training data. It is NOT a measurement of
Gnocchi's bias, for two reasons documented at length in CLAUDE.md:

  1. It is a LEVEL quantity. Gnocchi applies the ratio
     r = sigma(b0 + b.z(w)) / sigma(b0), and a level error common to numerator
     and denominator cancels exactly -- so a gap here need not reach the score.
  2. It is measured over the WHOLE training population. The empirical
     adjustment in fig3/empirical_r.py is measured over the analyzed window set
     (noncoding, pass_qc, autosome+PAR), and the two disagree materially above
     GC 0.55. Neither is wrong; they answer different questions.

For the quantity that does propagate to Gnocchi, use make_r_figures.py.
"""

import argparse
import os
import sys

import matplotlib
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gnocchi_bias.dnm_model import bin_training_calibration  # noqa: E402
import panels  # noqa: E402

DEFAULT_PREDICTIONS = "refits/training_reliability_predictions.full.txt"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-predictions", default=DEFAULT_PREDICTIONS,
                    help="per-site table from run_dnm_training_experiment.py -mode reliability")
    ap.add_argument("-output_dir", default="fig3/output")
    ap.add_argument("-group", default="non-CpG", choices=["non-CpG", "CpG"])
    ap.add_argument("-n_bins", type=int, default=20)
    ap.add_argument("-min_n", type=int, default=500,
                    help="drop GC bins holding fewer than this many training sites")
    ap.add_argument("-logy", action="store_true", help="log y-axis")
    ap.add_argument("-xmax", type=float, default=0.76,
                    help="right x-limit. Wider than panel A's 0.73 by default so the "
                         "top surviving GC bin is not clipped mid-curve; set 0.73 when "
                         "stacking this against panel A")
    args = ap.parse_args()

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(args.output_dir, exist_ok=True)
    df_pred = pd.read_csv(args.predictions, sep="\t")
    print(f"{len(df_pred):,} training sites, {df_pred['context'].nunique()} contexts fit")

    binned = bin_training_calibration(df_pred, n_bins=args.n_bins, stratify_cpg=True)
    sub = binned[binned["group"] == args.group]
    print(f"{args.group}: {int(sub['n'].sum()):,} sites, {int(sub['n1'].sum()):,} DNMs")

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    panels.panel_dnm_probability(ax, binned, group=args.group, min_n=args.min_n,
                                 logy=args.logy, xrange=(0.2, args.xmax))
    slug = args.group.replace("-", "_").lower()
    stem = os.path.join(args.output_dir, f"dnm_probability_{slug}")
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    fig.savefig(stem + ".png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {stem}.pdf")

    out = sub[sub["n"] >= args.min_n].copy()
    out["gc"] = out["gc_mid"] / 100.0
    out.to_csv(stem + ".txt", sep="\t", index=False, float_format="%.6g")
    pd.set_option("display.width", 200)
    print(out[["gc", "n", "n1", "mean_pred", "empirical_prop", "se",
               "inflation"]].to_string(index=False))


if __name__ == "__main__":
    main()
