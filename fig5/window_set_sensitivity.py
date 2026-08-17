"""
Does Fig. 5 survive the narrowing to McHale et al.'s 693,270 putatively neutral windows?

THE QUESTION. The figure is computed on 1,843,559 windows (noncoding + pass_qc +
autosome/PAR). McHale et al.'s Fig. 1 uses 693,270 -- 2.66x fewer, after excluding
GeneHancer-enhancer-overlapping windows and hg38 assembly gaps / ENCODE exclude regions /
low-coverage regions. Their file settles it directly (set NEUTRAL_WINDOWS_BED in
config.py and rerun), but it is not in this environment, so this script brackets the
answer with same-sized subsets that stand in for what the narrowing could do:

  random       the same windows, fewer of them            -- isolates sample size
  gc_tilted    keep probability falling with GC           -- the direction the enhancer
                                                             exclusion pulls, since
                                                             enhancers are GC-rich
  adversarial  keep the 693,270 LEAST "enhancer-like"     -- a hard cut on GC and
               windows, scoring on GC and published z        published z jointly

Ranks are computed WITHIN a window set, so every statistic is recomputed per arm. The
headline table bins all arms on ONE set of edges (the full set's) and averages over bins
holding >= 100 windows in every arm, so the comparison is bin-for-bin rather than
confounded by each arm choosing its own binning.

WHAT IT FOUND (2026-08-17, seed 0), over the 13 bins every arm draws, GC 0.23-0.65:

  set          step1   step2  scored   step2/step1
  full         0.067   0.177   0.034        2.64x
  random       0.066   0.177   0.033        2.68x
  gc_tilted    0.067   0.180   0.033        2.68x

Sample size is not the issue and neither is a GC-tilted removal, even one keeping GC > 0.5
windows at 15% against 37% overall. The per-bin curves are near superimposable -- step 2's
mean rank runs 0.29 -> 0.85 across those bins in the full set and 0.32 -> 0.86 under the
tilt, the offset being rank renormalization inside a lower-GC set -- and every qualitative
claim holds: step 2's bias is ~2.6x step 1's, and the refit lands below both.

Computed with each arm's OWN binning, as the figure does, the summary shrinks: 0.212
(full) -> 0.207 (random) -> 0.182 (gc_tilted) for step 2. That is the 100-window floor,
not a smaller bias: the top drawn bin moves from GC 0.75 to 0.65 while step 2's mean rank
in it stays 0.877 / 0.876 / 0.875. Quote the per-bin curve, or say which bins the average
covers.

The adversarial arm (`-adversarial`) DOES break it -- step1 0.225, step2 0.217, ratio
0.96x, with step 2's per-bin curve inverting to 0.52 -> 0.02 -- and it is here so nobody
has to rediscover it. But read what it does first: it keeps 1.0% of GC > 0.5 windows and
none of the (GC > 0.5, z > 2) corner, and it selects on z, which is the quantity being
ranked. All three curves distort together, which is the signature of a truncation artifact
rather than a property of the window set. It bounds the claim; it does not estimate it.
The real check is the file.

STILL OPEN, and this is the honest gap: none of these arms can reproduce removing windows
because they are CONSTRAINED. Enhancer windows are both GC-rich and depleted of variation,
so the real narrowing deletes high-z windows preferentially at high GC -- partially,
within GC bins, not by truncation. Panel C's `non_neutral` band measures the empirical
version of this once the file is supplied: if the removed territory carries the scored
population's own DNM rate, the narrowing costs sample size and nothing else.

Run:  .venv/bin/python fig5/window_set_sensitivity.py [-target 693270] [-seed 0]
"""
import argparse
import os
import sys

import numpy as np
import polars as pl

# Repo root then fig5/, exactly as refit.py does it: `gnocchi_bias` is a package at the
# root, `data`/`config` are siblings here.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D  # noqa: E402

CURVES = ("step1", "step2", "scored")


def arms(full: pl.DataFrame, target: int, seed: int) -> dict:
    """The three stand-in subsets, each `target` windows, plus the full set."""
    rng = np.random.default_rng(seed)
    gc = full["GC_content"].to_numpy()
    n = full.height

    random_set = full[np.sort(rng.choice(n, target, replace=False))]

    # exp(-6*(GC - min)) keep-probability, rescaled to the target size: monotone in GC,
    # and at 6 it keeps GC > 0.5 windows at roughly half the overall rate -- a real tilt,
    # harsher than the enhancer exclusion is likely to be.
    p = np.exp(-6.0 * (gc - gc.min()))
    p = p / p.max()
    tilted = full.filter(pl.Series(rng.random(n) < p * (target / p.sum())))

    # "enhancer-like" = GC-rich and constrained. Selecting on z is circular for a
    # z-derived statistic; see the module docstring before reading this arm as evidence.
    z = np.clip(full["z_published"].to_numpy(), -10, 10)
    score = ((gc - gc.mean()) / gc.std() + (z - z.mean()) / z.std()) / 2
    adversarial = full[np.sort(np.argsort(score)[:target])]

    return {"full": full, "random": random_set,
            "gc_tilted": tilted, "adversarial": adversarial}


def binned_ranks(df: pl.DataFrame, edges: np.ndarray) -> pl.DataFrame:
    """Mean rank per GC bin for each curve, on caller-supplied (shared) edges."""
    d, _ = D.rank_curves(df, extra=[("scored", D.refit_path("expected", "scored"))],
                         min_n=100)
    d = d.with_columns(pl.Series("gc_bin", D.assign_bin(d["GC_content"].to_numpy(), edges)))
    return (d.group_by("gc_bin")
             .agg([pl.len().alias("n")]
                  + [pl.col(f"rank_{c}").mean().alias(c) for c in CURVES])
             .sort("gc_bin"))


def report(sets: dict, edges: np.ndarray, mids: np.ndarray, header: str) -> None:
    """One comparison table over the GC bins every set in `sets` draws."""
    tabs = {name: binned_ranks(df, edges) for name, df in sets.items()}
    common = set.intersection(*[set(t.filter(pl.col("n") >= 100)["gc_bin"])
                                for t in tabs.values()])
    print(f"\n--- {header} ---")
    print(f"{len(common)} bins hold >= 100 windows in every arm, "
          f"GC {mids[min(common)]:.2f}-{mids[max(common)]:.2f}\n")
    print(f"{'set':<12} {'step1':>7} {'step2':>7} {'scored':>7}  {'step2/step1':>11}")
    for name, t in tabs.items():
        t = t.filter(pl.col("gc_bin").is_in(list(common)))
        v = {c: float((t[c] - 0.5).abs().mean()) for c in CURVES}
        print(f"{name:<12} {v['step1']:>7.3f} {v['step2']:>7.3f} {v['scored']:>7.3f}  "
              f"{v['step2'] / v['step1']:>10.2f}x")

    print("\nper-bin mean rank of published Gnocchi (step2), shared bins:")
    print(f"{'GC':<12} " + "  ".join(f"{mids[b]:.2f}" for b in sorted(common)))
    for name, t in tabs.items():
        t = t.filter(pl.col("gc_bin").is_in(list(common))).sort("gc_bin")
        print(f"{name:<12} " + "  ".join(f"{v:.2f}" for v in t["step2"]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-target", type=int, default=693_270,
                    help="subset size (default: McHale et al.'s window count)")
    ap.add_argument("-seed", type=int, default=0)
    ap.add_argument("-cache_dir", default=D.CACHE_DIR)
    ap.add_argument("-adversarial", action="store_true",
                    help="also run the hard cut that selects on z. Off by default: it is "
                         "circular for a z-derived statistic, and it truncates the GC "
                         "axis, which would drag the shared-bin set down for every arm.")
    args = ap.parse_args()

    full = D.window_table(args.cache_dir)
    gc = full["GC_content"].to_numpy()
    print(f"full: {full.height:,} windows, GC {gc.min():.3f}-{gc.max():.3f}, "
          f"mean {gc.mean():.3f}\n")

    sets = arms(full, args.target, args.seed)
    adversarial = sets.pop("adversarial")

    hi = gc > 0.5
    for name, df in list(sets.items()) + ([("adversarial", adversarial)]
                                          if args.adversarial else []):
        ids = set(df["element_id"].to_list())
        kept = np.fromiter((e in ids for e in full["element_id"].to_list()),
                           bool, full.height)
        g = df["GC_content"].to_numpy()
        print(f"{name:<12} {df.height:>9,} windows   GC mean {g.mean():.3f}   "
              f"GC>0.5 kept {kept[hi].mean():>6.1%}")

    # ONE set of edges for every arm, so the arms are compared bin for bin.
    edges = D.gc_edges(gc, D.N_BINS)
    mids = 0.5 * (edges[:-1] + edges[1:])
    report(sets, edges, mids, header="stand-ins for the narrowing")

    if args.adversarial:
        report({"full": full, "adversarial": adversarial}, edges, mids,
               header="the adversarial arm, on its own shared bins")
        print("  It selects on z, the quantity being ranked, and deletes the GC-rich "
              "corner outright.\n  All three curves distort together and the per-bin "
              "curve inverts -- a truncation\n  artifact, a bound on the claim rather "
              "than an estimate of it. See the docstring.")

    print("\n`scored` is the retrained model EVALUATED on each arm; it was fit on the "
          "full window population.\nOn the real neutral set, refit.py must be rerun -- "
          "the size-matched control adapts with it.")


if __name__ == "__main__":
    main()
