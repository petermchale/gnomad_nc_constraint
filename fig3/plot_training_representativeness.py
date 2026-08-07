"""
Does the DNM training set represent the noncoding genome Gnocchi is scored on?

    python fig3/plot_training_representativeness.py [-cache_dir tmp] [-force]

Panel A  The empirical non-CpG DNM probability vs GC, built four ways, changing one
         ingredient at a time (denominator, window population, aggregation). The two
         "analyzed windows" rungs superimpose and the two "whole genome" rungs
         superimpose; the two groups come apart above GC 0.5. So the shape difference
         between r_non_vs_empirical.pdf and dnm_probability_non_cpg.pdf is not caused
         by the choice of background sites.

Panel B  Why: the fraction of the training set's background sites that lie in the
         analyzed noncoding genome at all, falling from ~0.83 in the GC bulk to under
         0.30 by GC 0.68 as GC-rich sequence turns coding or loses gnomAD coverage.

Reuses the cached per-(context, bin) tables make_r_figures.py writes; the two extra
tables this needs are cached alongside them. See training_representativeness.py for
the construction and what it does and does not establish.
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
import training_representativeness as T  # noqa: E402
from gnocchi_bias import windows as W  # noqa: E402

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
    ap.add_argument("-cache_dir", default="tmp")
    ap.add_argument("-output_dir", default="fig3/output")
    ap.add_argument("-min_dnm", type=int, default=200,
                    help="drop GC bins with fewer observed DNMs in the analyzed windows")
    ap.add_argument("-force", action="store_true")
    args = ap.parse_args()

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(args.output_dir, exist_ok=True)
    out = lambda name: os.path.join(args.output_dir, name)  # noqa: E731

    edges_path = out("edges.npy")
    if os.path.exists(edges_path) and not args.force:
        edges = np.load(edges_path)
        print(f"reusing {edges_path}")
    else:
        edges = R.gc_edges(W.build_window_table(args.cache_dir)["GC_content"].to_numpy(),
                           N_BINS)
        np.save(edges_path, edges)

    genome = _cached(out("genome_by_context_bin.parquet"), args.force,
                     lambda: E.load_genome_by_context_bin(edges))
    counts = _cached(out("dnm_counts_by_context_bin.parquet"), args.force,
                     lambda: E.load_dnm_counts_by_context_bin(edges))
    controls = _cached(out("control_counts_by_context_bin.parquet"), args.force,
                       lambda: E.load_control_counts_by_context_bin(edges))
    training = _cached(out("training_by_context_bin.parquet"), args.force,
                       lambda: E.load_training_by_context_bin(edges))
    comp = _cached(out("dnm0_window_composition.parquet"), args.force,
                   lambda: T.dnm0_window_composition(edges))

    ladder = T.build_ladder(genome, counts, controls, training, edges,
                            min_dnm=args.min_dnm)
    ladder.write_parquet(out("population_ladder.parquet"))
    print("\none-ingredient-at-a-time disagreement (GC <= 0.62):")
    T.report_ladder(ladder)

    fig, (axA, axB) = plt.subplots(2, 1, figsize=(7.4, 8.0), sharex=True,
                                   gridspec_kw={"height_ratios": [1.5, 1], "hspace": 0.12})
    panels.panel_population_ladder(axA, ladder, T.LADDER_SERIES, xrange=XRANGE,
                                   show_xlabel=False)
    panels.panel_training_composition(axB, comp, xrange=XRANGE)
    panels.label_panels((axA, axB), labels=("A", "B"))
    fig.savefig(out("training_representativeness.pdf"), bbox_inches="tight")
    fig.savefig(out("training_representativeness.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out('training_representativeness.pdf')}")

    pl.Config.set_tbl_rows(30)
    pl.Config.set_tbl_width_chars(200)
    print("\npanel A (each curve normalized to weighted mean 1):")
    print(ladder.select(["gc_mid", "dnm_total"] + [c for c, *_ in T.LADDER_SERIES]))
    print("\npanel B (composition of the non-CpG background training sites):")
    print(comp.filter(pl.col("n_total") >= 500)
              .select(["gc_mid", "n_total", "frac_analyzed", "frac_coding",
                       "frac_noannot"]))


if __name__ == "__main__":
    main()
