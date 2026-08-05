"""
Build the two adjustment-factor figures.

  r_eff_decomposition.pdf  -- what Gnocchi applies: r_eff vs GC, split CpG/non-CpG,
                              with the counterfactual that holds non-CpG r at 1.
                              Shows the GC trend is wholly non-CpG.
  r_non_vs_empirical.pdf   -- whether that adjustment is right: the fitted non-CpG r
                              against the r the observed de novo mutations support,
                              plus their ratio.

Both read the same window population and the same GC bin edges as fig3.ipynb's
panel A, so the three figures are directly comparable. Intermediate per-(context,
bin) tables are cached as parquet in OUTPUT_DIR and reused on later runs; delete
them to recompute.

    python fig3/make_r_figures.py [-cache_dir tmp] [-output_dir fig3/output] [-force]

See CLAUDE.md ("Peter's proposed CpG mechanism", "Methylation, and why the
training-set calibration panel measures the wrong thing") for the evidence trail
these two figures make visual, and r_eff.py / empirical_r.py's docstrings for the
construction and its assumptions.
"""

import argparse
import os
import sys

import matplotlib
import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gnocchi_bias import windows as W  # noqa: E402
import empirical_r as E  # noqa: E402
import panels  # noqa: E402
import r_eff as R  # noqa: E402

N_BINS = 20
XRANGE = (0.2, 0.73)


def _cached(path: str, force: bool, build):
    if os.path.exists(path) and not force:
        print(f"reusing {path}")
        return pl.read_parquet(path)
    df = build()
    df.write_parquet(path)
    print(f"wrote {path}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-cache_dir", default="tmp", help="downloaded bucket files")
    ap.add_argument("-output_dir", default="fig3/output", help="figures and intermediates")
    ap.add_argument("-min_n_windows", type=int, default=100,
                    help="drop GC bins with fewer windows (figure 1)")
    ap.add_argument("-min_dnm", type=int, default=200,
                    help="drop GC bins with fewer observed DNMs (figure 2)")
    ap.add_argument("-force", action="store_true", help="recompute cached tables")
    args = ap.parse_args()

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(args.output_dir, exist_ok=True)
    out = lambda name: os.path.join(args.output_dir, name)  # noqa: E731

    # ---------------------------------------------------------- window table
    df_win = W.build_window_table(args.cache_dir)
    edges = R.gc_edges(df_win["GC_content"].to_numpy(), N_BINS)
    np.save(out("edges.npy"), edges)
    print(f"window set: {df_win.height:,} windows, "
          f"GC {df_win['GC_content'].min():.3f}-{df_win['GC_content'].max():.3f}")

    # ------------------------------------------------- figure 1: what is applied
    comp = _cached(out("r_eff_components.parquet"), args.force,
                   lambda: R.load_r_eff_components())
    df = R.attach_components(df_win, comp)
    binned = R.bin_r_eff(df, edges)
    R.report_refit_validation(binned)
    binned.write_parquet(out("r_eff_binned.parquet"))

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    panels.panel_r_eff_decomposition(ax, binned, min_n=args.min_n_windows, xrange=XRANGE)
    fig.savefig(out("r_eff_decomposition.pdf"), bbox_inches="tight")
    fig.savefig(out("r_eff_decomposition.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out('r_eff_decomposition.pdf')}")

    # ------------------------------------------------ figure 2: is it correct
    genome = _cached(out("genome_by_context_bin.parquet"), args.force,
                     lambda: E.load_genome_by_context_bin(edges))
    counts = _cached(out("dnm_counts_by_context_bin.parquet"), args.force,
                     lambda: E.load_dnm_counts_by_context_bin(edges))

    # The per-(context, bin) genome table must reproduce figure 1's non-CpG curve;
    # if it does not, the two figures are not describing the same windows.
    chk = (genome.filter(~pl.col("context").is_in(R.CPG_CONTEXTS))
                 .group_by("gc_bin")
                 .agg([pl.col("e1").sum().alias("e1"), pl.col("e2").sum().alias("e2")])
                 .with_columns((pl.col("e2") / pl.col("e1")).alias("r_non_check"))
                 .sort("gc_bin"))
    ref = binned.sort("gc_mid").with_columns(pl.Series("gc_bin", np.arange(binned.height)))
    worst = float((ref.join(chk, on="gc_bin")["r_non_check"]
                   - ref.join(chk, on="gc_bin")["r_non"]).abs().max())
    print(f"cross-check: max |r_non(per-context) - r_non(per-window)| = {worst:.2e}")
    assert worst < 1e-4, "per-context and per-window non-CpG curves disagree"

    emp = E.empirical_from_dnm_counts(genome, counts)
    res = E.attach_gc_mid(E.combine_non_cpg(genome, emp, min_n_eff=0,
                                            min_weight_covered=0.0), edges)
    res = res.join(counts.group_by("gc_bin").agg(pl.col("n_dnm").sum().alias("dnm_total")),
                   on="gc_bin", how="left")
    res.write_parquet(out("empirical_r_non.parquet"))

    fig, (axA, axB) = plt.subplots(2, 1, figsize=(7.0, 7.4), sharex=True,
                                   gridspec_kw={"height_ratios": [1.4, 1], "hspace": 0.12})
    panels.panel_r_non_vs_empirical(axA, res, min_dnm=args.min_dnm, xrange=XRANGE,
                                    show_ratio_on=axB)
    panels.label_panels((axA, axB), labels=("A", "B"))
    fig.savefig(out("r_non_vs_empirical.pdf"), bbox_inches="tight")
    fig.savefig(out("r_non_vs_empirical.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out('r_non_vs_empirical.pdf')}")

    # ------------------------------------------------------- numbers to quote
    pl.Config.set_tbl_rows(30)
    shown = res.filter(pl.col("dnm_total").fill_null(0) >= args.min_dnm)
    print("\nfigure 1 (applied adjustment):")
    print(binned.filter(pl.col("n") >= args.min_n_windows)
                .select(["gc_mid", "n", "pi_cpg", "r_eff", "r_cpg", "r_non",
                         "r_counterfactual"]))
    print("\nfigure 2 (fitted vs observed, non-CpG):")
    print(shown.select(["gc_mid", "dnm_total", "r_non_model", "r_non_empirical",
                        "se_r_non_empirical", "inflation"]))


if __name__ == "__main__":
    main()
