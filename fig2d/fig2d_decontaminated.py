"""
McHale et al. Fig. 2D, redrawn with the decontaminated Gnocchi beside the published one.

THE ORIGINAL is `plot_gnocchi_distribution_vs_standard_normal` in
quinlan-lab/constraint-tools, papers/neutral_models_are_biased/
10.residuals-are-over-dispersed.ipynb (cell 6), and this script follows it step for step:

  * the windows are McHale et al.'s window file (config.NEUTRAL_WINDOWS_BED) filtered to
    `window overlaps enhancer == False` -- all 693,270 rows, read directly, with NO
    [-10, 10] z filter and NO join to the features table;
  * GC is the file's own `GC_content_1000bp` (bedtools nuc), its mean and sd taken over
    all of those rows, and the slice is STRICT: mean - 0.1 sd < GC < mean + 0.1 sd;
  * the published score is the file's own `gnocchi` column;
  * 250 edges over [-10, 10], the standard normal evaluated at bin centres and scaled by
    n * bin width, log counts from 1 to 2e4.

Their notebook printed 58,286 windows in that slice, and this script checks it reproduces
that number before drawing anything else.

THE ONE ADDITION is the decontaminated score: the `scored.neutral` refit's expected count
against Chen et al.'s observed count (from their constraint table), through the pipeline's
own z formula (windows.z_expr). Both scores are drawn on the SAME windows, so a window with
no refit row is dropped from both, and the shortfall is printed.

WHICH OBSERVED COUNT. Not the window file's `N_observed`, which differs from Chen et al.'s
`observed` in 691,490 of the 693,270 windows. The constraint table's `observed` is the one
that reproduces the file's `gnocchi` through z_expr (max |diff| 1.8e-15), so it is the count
the published score was computed from, and the decontaminated score must use the same one.

Needs the constraint-tools HPC path (the window file) and the `scored.neutral` refit, so it
runs there:

    python fig2d/fig2d_decontaminated.py
"""
import argparse
import json
import os
import sys

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy import stats

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
# fig5.config is imported, not copied: NEUTRAL_WINDOWS_BED there is what the `scored` refit
# was FIT on, and this script SCORES that refit. A second copy of the path is how the two
# would drift apart. Nothing else here depends on fig5.
from fig5 import config   # noqa: E402
from gnocchi_bias import windows as W   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HERE, "output")
REFITS_DIR = os.path.join(REPO_ROOT, "refits")   # written by fig5/refit.py
REFIT_EXPECTED = "expected_counts_by_context_methyl_genome_1kb.scored.neutral.txt"

ORIGINAL_SLICE_N = 58_286   # "actual number of intervals", printed by their notebook
STD_SCALE = 0.1
EDGES = np.linspace(-10, 10, 250)

SCORES = {  # label -> (legend name, histogram style)
    "published": ("Gnocchi, published", {"alpha": 0.2, "color": "black"}),
    "scored": ("Gnocchi, decontaminated",
               {"histtype": "step", "color": "0.3", "linewidth": 1.8}),
}


def refit_expected(neutral_windows_bed: str) -> pl.DataFrame:
    """The `scored.neutral` refit's expected counts, checked against its provenance stamp:
    a refit fit on one window set must not be scored on another."""
    path = os.path.join(REFITS_DIR, REFIT_EXPECTED)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path}\nRun: python fig5/refit.py -population scored")
    with open(os.path.join(REFITS_DIR, "provenance.json")) as fh:
        built_with = json.load(fh).get("scored.neutral", {}).get("neutral_windows_bed")
    if built_with != neutral_windows_bed:
        raise RuntimeError(f"{path} was built with NEUTRAL_WINDOWS_BED={built_with!r}, "
                           f"not {neutral_windows_bed!r}")
    print(f"decontaminated expected counts: {path}")
    return (pl.read_csv(path, separator="\t")
              .select(["element_id", pl.col("expected").alias("expected_scored")]))


def their_windows(bed: str, cache_dir: str) -> tuple[pl.DataFrame, pl.Series]:
    """
    Their notebook's population, and the GC values its mean and sd are taken over.

    Returns (element_id, GC, z_published, observed, expected_step2) for every window that
    also has a row in Chen et al.'s constraint table -- which supplies `observed` for the
    decontaminated z -- and, separately, GC over ALL their rows, so the slice's bounds do
    not depend on that join.
    """
    df = (pl.read_csv(bed, separator="\t", infer_schema_length=10_000,
                      schema_overrides={"chrom": pl.String})
            .rename(lambda c: c.strip().replace(" ", "_"))
            .filter(~pl.col(W.MCHALE_ENHANCER_COLUMN).cast(pl.Boolean))
            .select((pl.col("chrom") + "-" + pl.col("start").cast(pl.String) + "-"
                     + pl.col("end").cast(pl.String)).alias("element_id"),
                    pl.col("GC_content_1000bp").alias("GC"),
                    pl.col("gnocchi").alias("z_published")))
    print(f"their windows: {df.height:,} with `window overlaps enhancer` False "
          f"(expected {W.MCHALE_NEUTRAL_WINDOW_COUNT:,})")
    gc_population = df["GC"]

    annot = W.download(W.REMOTE_FILES["annot"], cache_dir)
    chen = duckdb.connect().execute(
        f"SELECT element_id, observed, expected AS expected_step2 "
        f"FROM read_csv_auto('{annot}', header=True)").pl()
    joined = df.join(chen, on="element_id", how="inner")
    print(f"  {joined.height:,} of them in Chen et al.'s constraint table "
          f"({df.height - joined.height:,} not)")
    return joined, gc_population


def summarize(z: np.ndarray) -> dict:
    return {
        "n": int(z.size), "mean": float(z.mean()), "median": float(np.median(z)),
        "sd": float(z.std(ddof=1)), "var": float(z.var(ddof=1)),
        # Robust spread: a normal has IQR / 1.349 = sd, so this is sd without the tails.
        "sd_iqr": float(np.subtract(*np.percentile(z, [75, 25])) / 1.349),
        "frac_abs_gt2": float(np.mean(np.abs(z) > 2)),
        "frac_abs_gt2_normal": float(2 * stats.norm.sf(2)),
        "n_outside_bins": int(np.sum((z < EDGES[0]) | (z > EDGES[-1]))),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("-neutral_windows_bed", default=config.NEUTRAL_WINDOWS_BED,
                    help="override the window file (testing only)")
    ap.add_argument("-cache_dir", default=W.CACHE_DIR)
    args = ap.parse_args()
    bed = args.neutral_windows_bed
    if not bed:
        raise ValueError("Fig. 2D is drawn on McHale et al.'s window file: set "
                         "NEUTRAL_WINDOWS_BED in fig5/config.py.")

    df, gc_population = their_windows(bed, args.cache_dir)

    mu, sd = float(gc_population.mean()), float(gc_population.std())
    lo, hi = mu - STD_SCALE * sd, mu + STD_SCALE * sd
    n_slice_all = int(((gc_population > lo) & (gc_population < hi)).sum())
    print(f"GC slice: mean {mu:.4f}, sd {sd:.4f} over {gc_population.len():,} windows; "
          f"{lo:.4f} < GC < {hi:.4f} -> {n_slice_all:,} windows")
    verdict = "REPRODUCED" if n_slice_all == ORIGINAL_SLICE_N else "MISMATCH"
    print(f"  {verdict}: their notebook's slice held {ORIGINAL_SLICE_N:,}")

    sl = df.filter((pl.col("GC") > lo) & (pl.col("GC") < hi))
    # The published z rebuilt from the constraint table's (expected, observed): this is what
    # licenses computing the decontaminated z from that same observed count.
    z_check = float(sl.select((W.z_expr("expected_step2") - pl.col("z_published")).abs()
                              .max()).item())
    print(f"  z_expr(published expected, observed) vs published z: max |diff| = {z_check:.2e}")
    refit = refit_expected(bed)
    sl = (sl.join(refit, on="element_id", how="inner")
            .with_columns(W.z_expr("expected_scored").alias("z_scored"))
            .filter(pl.col("z_published").is_finite() & pl.col("z_scored").is_finite()))
    print(f"  paired: {sl.height:,} windows carry both scores "
          f"({n_slice_all - sl.height:,} of the slice dropped from both)")

    rows = []
    for label in SCORES:
        s = summarize(sl[f"z_{label}"].to_numpy())
        rows.append({"score": label, **s})
        print(f"  {label:<10} n={s['n']:,}  mean={s['mean']:+.3f}  median={s['median']:+.3f}  "
              f"sd={s['sd']:.3f}  var={s['var']:.3f}  sd(IQR)={s['sd_iqr']:.3f}  "
              f"P(|z|>2)={s['frac_abs_gt2']:.3f} (normal {s['frac_abs_gt2_normal']:.3f})  "
              f"outside [-10, 10]: {s['n_outside_bins']:,}")
    z_pub, z_dec = sl["z_published"].to_numpy(), sl["z_scored"].to_numpy()
    print(f"  corr(published, decontaminated) = {np.corrcoef(z_pub, z_dec)[0, 1]:.4f}; "
          f"mean(decontaminated - published) = {np.mean(z_dec - z_pub):+.3f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stem = os.path.join(OUTPUT_DIR, "fig2d_decontaminated.neutral")
    pl.DataFrame(rows).write_csv(f"{stem}.tsv", separator="\t")

    centres, width = (EDGES[1:] + EDGES[:-1]) / 2, EDGES[1] - EDGES[0]
    with plt.rc_context({"font.size": 27, "font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "DejaVu Sans"]}):
        fig, ax = plt.subplots(figsize=(10, 8))
        for label, (name, style) in SCORES.items():
            ax.hist(sl[f"z_{label}"].to_numpy(), bins=EDGES, label=name, **style)
        ax.plot(centres, stats.norm.pdf(centres) * sl.height * width, color="black", lw=2,
                label="Expected (standard normal)")
        ax.set_xlabel("Gnocchi")
        ax.set_ylabel("Number of intervals")
        ax.set_yscale("log")
        ax.set_ylim(1, 2e4)
        ax.legend(frameon=False, loc="upper left", fontsize=22)
        for ext in ("pdf", "png"):
            fig.savefig(f"{stem}.{ext}", bbox_inches="tight")
        plt.close(fig)
    print(f"wrote {stem}.{{pdf,png,tsv}}")


if __name__ == "__main__":
    main()
