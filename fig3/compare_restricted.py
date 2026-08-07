"""
Does matching the training population to the scored population fix Gnocchi's GC bias?

    .venv/bin/python fig3/compare_restricted.py [-cache_dir tmp] [-output_dir fig3/output]

Run fig3/refit_restricted.py first; this reads its two outputs.

Builds one two-panel figure, restricted_refit.pdf:

  A  Gnocchi's own GC bias -- mean standardized rank vs GC content -- for the
     context-only model (r == 1), published Gnocchi, and Gnocchi rebuilt with the
     retrained adjustment. This is the Fig. 2A statistic, on the same window set, so
     it is directly comparable to Fig. 3A.

  B  The adjustment itself: the fitted non-CpG r before and after retraining, against
     the adjustment the observed de novo mutations support. Same construction as
     r_non_vs_empirical.pdf, with a second fitted curve added.

Panel B says whether the model was fixed; panel A says whether fixing it fixed the
score. Reporting both is the point -- a change in B that does not show up in A would
mean r's error was never what drove the bias.

Every curve in each panel is computed on one identical window population with one
identical set of GC bin edges, so no curve is advantaged by its filtering.
"""

import argparse
import os
import sys

import matplotlib
import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import empirical_r as E  # noqa: E402
import panels  # noqa: E402
import r_eff as R  # noqa: E402
from gnocchi_bias import windows as W  # noqa: E402

N_BINS = 20
XRANGE = (0.2, 0.73)

RESTRICTED_EXPECTED = "refits/expected_counts_by_context_methyl_genome_1kb.scored.txt"
RESTRICTED_RR = "refits/rr_by_context.scored.txt"


def load_restricted_expected(path: str) -> pl.DataFrame:
    """The retrained genome-wide expected counts, as element_id + expected_restricted."""
    df = pl.read_csv(path, separator="\t")
    return df.select(["element_id",
                      pl.col("expected").alias("expected_restricted")])


def rank_bias_summary(binned: pl.DataFrame, label: str) -> float:
    """
    Mean |mean rank - 0.5| across GC bins: one number for "how GC-biased is this
    metric", on the same scale used for the step-1 vs step-2 comparison in CLAUDE.md
    (0.093 vs 0.212). Bins are unweighted, matching that earlier figure.
    """
    return float((binned[f"mean_{label}"] - 0.5).abs().mean())


def build_panel_a(cache_dir: str, restricted_path: str, n_bins: int,
                  extra: list[tuple[str, str]] | None = None):
    """
    The rank statistic for the context-only model, published Gnocchi, and the
    retrained Gnocchi, plus any `extra` (label, path) refit tables.

    `extra` exists for the controls, which belong in the printed summary rather than
    the figure: the reimplementation's own full-scale refit (which should land on
    published Gnocchi, proving "before" and "after" differ by the intervention and
    not by whose code produced them), and the size-matched random refit.
    """
    df = W.build_window_table(cache_dir)
    df = df.join(load_restricted_expected(restricted_path), on="element_id", how="inner")
    labels = [("step1", "expected_step1"), ("step2", "expected_step2"),
              ("restricted", "expected_restricted")]
    for label, path in (extra or []):
        col = f"expected_{label}"
        df = df.join(load_restricted_expected(path).rename({"expected_restricted": col}),
                     on="element_id", how="inner")
        labels.append((label, col))
    print(f"panel A window set: {df.height:,} windows with every expected count present")

    df, binned = W.binned_rank_curves(df, curves=labels, n_bins=n_bins)
    print(f"  after joint z filtering: {df.height:,} windows")
    for label, _ in labels:
        print(f"  mean |rank - 0.5|  {label:<12} = {rank_bias_summary(binned, label):.3f}")
    return df, binned


def build_panel_b(edges: np.ndarray, output_dir: str, fits: dict, min_dnm: int):
    """
    The fitted-vs-observed non-CpG adjustment, for an arbitrary set of fits.

    `fits` maps a short label to the per-(window, context) rr table for that fit, or to
    None for the published pipeline's own r (recovered as expected_step2/expected_step1
    and needing no rr table). Every fitted curve and the observed curve are aggregated
    with the SAME E1 weights over the SAME analyzed windows and normalized per context
    the same way -- see empirical_r.combine_non_cpg -- so the only thing that differs
    between fitted curves is the model.

    Columns out: gc_bin, gc_mid, dnm_total, r_non_empirical, se_r_non_empirical, and
    r_non_model_{label} / inflation_{label} per fit.
    """
    def cached(name, build):
        path = os.path.join(output_dir, name)
        if os.path.exists(path):
            print(f"reusing {path}")
            return pl.read_parquet(path)
        df = build()
        df.write_parquet(path)
        return df

    counts = cached("dnm_counts_by_context_bin.parquet",
                    lambda: E.load_dnm_counts_by_context_bin(edges))

    out, emp = None, None
    for label, rr_path in fits.items():
        suffix = "" if rr_path is None else f".{label}"
        genome = cached(f"genome_by_context_bin{suffix}.parquet",
                        lambda p=rr_path: E.load_genome_by_context_bin(edges)
                        if p is None else E.load_genome_by_context_bin(edges,
                                                                       rr_by_context=p))
        if emp is None:
            # Built once, from the first genome table; e1 and opportunities are
            # identical across fits (they are step-1 quantities), so this is the same
            # observed curve for every fit -- asserted below rather than assumed.
            emp = E.empirical_from_dnm_counts(genome, counts)
        binned = E.combine_non_cpg(genome, emp, min_n_eff=0, min_weight_covered=0.0)
        cols = binned.select([
            "gc_bin",
            pl.col("r_non_model").alias(f"r_non_model_{label}"),
            pl.col("inflation").alias(f"inflation_{label}"),
            pl.col("r_non_empirical"), pl.col("se_r_non_empirical"),
        ])
        if out is None:
            out = cols
        else:
            worst = float((out.join(cols, on="gc_bin", suffix="_chk")["r_non_empirical"]
                           - out.join(cols, on="gc_bin",
                                      suffix="_chk")["r_non_empirical_chk"]).abs().max())
            assert worst < 1e-9, f"observed curve differs between fits by {worst:.2e}"
            out = out.join(cols.drop(["r_non_empirical", "se_r_non_empirical"]),
                           on="gc_bin")

    dnm_total = counts.group_by("gc_bin").agg(pl.col("n_dnm").sum().alias("dnm_total"))
    out = (out.join(dnm_total, on="gc_bin", how="left")
              .filter(pl.col("dnm_total").fill_null(0) >= min_dnm))
    return E.attach_gc_mid(out, edges).sort("gc_mid")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-cache_dir", default="tmp")
    ap.add_argument("-output_dir", default="fig3/output")
    ap.add_argument("-restricted_expected", default=RESTRICTED_EXPECTED)
    ap.add_argument("-restricted_rr", default=RESTRICTED_RR)
    ap.add_argument("-sizematched_rr",
                    default="refits/rr_by_context.sizematched.txt",
                    help="rr table from the size-matched random control. Plotted in "
                         "panel B so that panel is size-controlled too, not just "
                         "panel A's printed summary. Pass '' to omit.")
    ap.add_argument("-min_dnm", type=int, default=200)
    ap.add_argument("-control", action="append", default=[], metavar="LABEL:PATH",
                    help="extra refit table to include in the printed summary only "
                         "(repeatable). Use for the reimplementation's full-scale refit "
                         "and the size-matched random control.")
    args = ap.parse_args()

    for path in (args.restricted_expected, args.restricted_rr):
        if not os.path.exists(path):
            raise SystemExit(f"missing {path} -- run fig3/refit_restricted.py first")

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(args.output_dir, exist_ok=True)
    out = lambda name: os.path.join(args.output_dir, name)  # noqa: E731

    extra = []
    for spec in args.control:
        label, _, path = spec.partition(":")
        if not os.path.exists(path):
            raise SystemExit(f"missing control table {path}")
        extra.append((label, path))

    df, binned_a = build_panel_a(args.cache_dir, args.restricted_expected, N_BINS,
                                 extra=extra)
    binned_a.write_parquet(out("restricted_binned_ranks.parquet"))
    gc_mean = float(df["GC_content"].mean())

    edges = np.load(out("edges.npy")) if os.path.exists(out("edges.npy")) else \
        R.gc_edges(df["GC_content"].to_numpy(), N_BINS)
    fits = {"published": None, "restricted": args.restricted_rr}
    if args.sizematched_rr and os.path.exists(args.sizematched_rr):
        fits["sizematched"] = args.sizematched_rr
    binned_b = build_panel_b(edges, args.output_dir, fits, args.min_dnm)
    binned_b.write_parquet(out("restricted_r_non.parquet"))

    fig, (axA, axB) = plt.subplots(2, 1, figsize=(7.4, 8.4), sharex=True,
                                   gridspec_kw={"height_ratios": [1.15, 1], "hspace": 0.12})

    curves = [
        panels.curve_from_binned(binned_a, "step1", "step1",
                                 "Context-only model ($r \\equiv 1$)"),
        panels.curve_from_binned(binned_a, "step2", "step2",
                                 "Gnocchi as published"),
        panels.curve_from_binned(binned_a, "restricted", "dr",
                                 "Gnocchi, adjustment retrained on the scored population"),
    ]
    panels.panel_rank_bias(axA, curves, gc_mean=gc_mean, xrange=XRANGE,
                           show_xlabel=False)
    panel_b_fits = [("published", "step2", False, "Fitted $r$, as published"),
                    ("restricted", "dr", False,
                     "Fitted $r$, retrained on the scored population")]
    if "sizematched" in fits:
        panel_b_fits.append(("sizematched", "gap", True,
                             "Fitted $r$, size-matched random control"))
    panels.panel_r_fitted_vs_observed(axB, binned_b, panel_b_fits, xrange=XRANGE)
    panels.label_panels((axA, axB), labels=("A", "B"))

    fig.savefig(out("restricted_refit.pdf"), bbox_inches="tight")
    fig.savefig(out("restricted_refit.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out('restricted_refit.pdf')}")

    pl.Config.set_tbl_rows(30)
    pl.Config.set_tbl_width_chars(200)
    print("\npanel A (mean standardized rank by GC bin):")
    print(binned_a.select(["gc_mid", "n", "mean_step1", "mean_step2", "mean_restricted"]))
    print("\npanel B (non-CpG adjustment, each fit vs observed):")
    cols = (["gc_mid", "dnm_total"]
            + [f"r_non_model_{k}" for k in fits] + ["r_non_empirical"]
            + [f"inflation_{k}" for k in fits])
    print(binned_b.select(cols))


if __name__ == "__main__":
    main()
