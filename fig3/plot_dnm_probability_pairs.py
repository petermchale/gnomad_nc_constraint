"""
P(DNM) vs GC -- fitted and empirical, on the original and the scored training set.

    .venv/bin/python fig3/plot_dnm_probability_pairs.py

Four curves, non-CpG contexts only:

  1  fitted,    original set   per-context logistic regressions fit on the full
                               training set, predicted on their own training sites
  2  empirical, original set   fraction of those sites that are DNMs
  3  fitted,    scored set     the same, for models fit on training sites restricted
                               to the analyzed window population
  4  empirical, scored set     fraction of THOSE sites that are DNMs

Each pair is a reliability diagram on its own population, so 1-vs-2 and 3-vs-4 are the
comparisons that mean something. 2-vs-4 shows how much the empirical GC dependence
itself changes when the out-of-population sites are dropped.

LEVELS ARE NOT COMPARABLE ACROSS THE TWO PAIRS. The class balance differs -- the full
set is 10.0 background per DNM, the restricted set 11.3 -- so the restricted pair sits
lower by roughly that ratio for reasons that have nothing to do with GC. Pass
-normalize to divide each curve by its own site-weighted mean, which removes the offset
and compares shape only; the raw version is the default because P(DNM) is what was
asked for.

Reuses the full-set predictions -mode reliability already wrote, and the restricted
set's already-computed feature selection, so only the restricted predictions are
recomputed (~4 min, dominated by loading the 2 GB background feature table).
"""

import argparse
import os
import sys

import matplotlib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gnocchi_bias import dnm_model as M  # noqa: E402
from gnocchi_bias import windows as W  # noqa: E402
import panels  # noqa: E402

FULL_PREDICTIONS = "refits/training_reliability_predictions.full.txt"
SUBSETS = {
    # name -> (selected-features table, per-site prediction cache)
    "scored": ("refits/selected.scored.txt",
               "refits/training_reliability_predictions.scored.txt"),
    "sizematched": ("refits/selected.sizematched.txt",
                    "refits/training_reliability_predictions.sizematched.txt"),
}


def subset_predictions(cache_dir: str, wanted: dict, seed: int = 0) -> dict:
    """
    Per-site fitted probability for each requested training subset, cached to disk.

    `wanted` maps population name -> (selected-features path, output path). Names are
    "scored" (restricted to the analyzed windows) and "sizematched" (the same NUMBER of
    sites drawn uniformly at random from the whole genome, seed `seed` -- exactly what
    refit_restricted.py -control_random fits). The size-matched population is what makes
    the original-vs-scored comparison interpretable: without it, the two pairs differ in
    both population AND size, and nothing distinguishes the two explanations.

    The selected-features tables come from refit_restricted.py's own runs, so the models
    predicted here are the models whose r produced restricted_refit.pdf.

    Loads the (large) training data at most once, and only if something is uncached.
    """
    todo = {k: v for k, v in wanted.items() if not os.path.exists(v[1])}
    out = {}
    for name, (_, path) in wanted.items():
        if name not in todo:
            print(f"reusing {path}")
            out[name] = pd.read_csv(path, sep="\t")
    if not todo:
        return out

    df_win = W.build_window_table(cache_dir)
    contexts = M.load_contexts(cache_dir)
    df_dnm1, df_dnm0 = M.load_training_data(cache_dir)
    n1, n0 = M.count_in_analyzed_windows(df_dnm1, df_dnm0, df_win["element_id"].to_list())

    for name, (selected_path, path) in todo.items():
        if name == "scored":
            d1, d0 = M.restrict_to_analyzed_windows(
                df_dnm1, df_dnm0, df_win["element_id"].to_list())
        elif name == "sizematched":
            print(f"size-matched control: {n1:,} / {n0:,} sites drawn at random")
            d1 = df_dnm1.sample(n=n1, random_state=seed)
            d0 = df_dnm0.sample(n=n0, random_state=seed)
        else:
            raise ValueError(f"unknown population {name}")
        df_sel = pd.read_csv(selected_path, sep="\t")
        df_pred = M.predict_training_set(d1, d0, contexts, df_sel)
        df_pred.to_csv(path, sep="\t", index=False)
        print(f"wrote {path}  ({len(df_pred):,} sites)")
        out[name] = df_pred
    return out


def non_cpg_binned(df_pred: pd.DataFrame, n_bins: int, edges: np.ndarray,
                   per_context: bool = False) -> pd.DataFrame:
    """
    Non-CpG rows of the reliability table, binned on SHARED edges so the populations
    land in the same bins (they have different GC ranges, so letting each pick its own
    linspace would misalign them).

    per_context=False (default) pools all non-CpG contexts, which is what P(DNM) means.
    Note what that implies about the relationship to r: since p_hat_t(x) =
    r_t(x)*sigma(b_t0), a pooled mean is sum_t pi_t(g)*sigma(b_t0)*E[r_t | g,t] -- the
    per-context adjustments are mixed with weights carrying each context's own BASELINE
    level, and pi_t(g) shifts with GC, so the pooled curve contains composition-driven
    GC dependence on top of r.

    per_context=True normalizes each context to its own site-weighted mean first and
    then recombines with the same weights, removing that composition term. The result
    tracks empirical_r's genome-level r_non to within 1% through the GC bulk, and
    remains ~5% above it at GC 0.66 -- the residual being the two differences no
    normalization can remove: this evaluates the model at SITE-level feature vectors
    (r(w) uses window-aggregated ones), and it marginalizes the non-GC features over
    the TRAINING-SITE distribution (r_non marginalizes over genomic windows weighted by
    E1). That second difference is the subject of this figure, not a defect in it.
    """
    df = df_pred[~df_pred["context"].isin(M.CPG_CONTEXTS)].copy()
    df["bin"] = np.clip(np.digitize(df["gc"], edges[1:-1]), 0, len(edges) - 2)
    out = df.groupby("bin").agg(
        n=("label", "size"), n1=("label", "sum"),
        gc_mid=("gc", "mean"), mean_pred=("pred", "mean"),
    ).reset_index()
    out["empirical_prop"] = out["n1"] / out["n"]
    out["se"] = np.sqrt(out["empirical_prop"] * (1 - out["empirical_prop"]) / out["n"])

    if per_context:
        g = df.groupby(["context", "bin"]).agg(
            n=("pred", "size"), pred=("pred", "mean"), emp=("label", "mean")).reset_index()
        for col in ("pred", "emp"):
            mean = (g.groupby("context")
                     .apply(lambda d, c=col: np.average(d[c], weights=d["n"]),
                            include_groups=False).rename(col + "_mean"))
            g = g.merge(mean, on="context")
            g[col + "_r"] = g[col] / g[col + "_mean"]
        rec = (g.groupby("bin")
                .apply(lambda d: pd.Series({
                    "mean_pred": np.average(d["pred_r"], weights=d["n"]),
                    "empirical_prop": np.average(d["emp_r"], weights=d["n"])}),
                       include_groups=False)
                .reset_index())
        # Keep the binomial SE on the same relative scale as the rescaled rate.
        scale = rec["empirical_prop"] / out["empirical_prop"]
        out = out.drop(columns=["mean_pred", "empirical_prop"]).merge(rec, on="bin")
        out["se"] = out["se"] * scale.values
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-cache_dir", default="tmp")
    ap.add_argument("-output_dir", default="fig3/output")
    ap.add_argument("-full_predictions", default=FULL_PREDICTIONS)
    ap.add_argument("-no_sizematched", action="store_true",
                    help="omit the size-matched random control (it is what separates "
                         "population from sample size; only drop it for a quick look)")
    ap.add_argument("-seed", type=int, default=0)
    ap.add_argument("-n_bins", type=int, default=20)
    ap.add_argument("-min_n", type=int, default=500)
    ap.add_argument("-normalize", action="store_true",
                    help="divide each curve by its own site-weighted mean, removing the "
                         "class-balance offset between populations (see the module "
                         "docstring: it is ~0.89 for the size-matched control, and is "
                         "pure class balance, flat in GC)")
    ap.add_argument("-per_context", action="store_true",
                    help="normalize each trinucleotide context to its own mean BEFORE "
                         "pooling, removing the composition term and making the curve "
                         "directly comparable to empirical_r's genome-level r_non. See "
                         "non_cpg_binned's docstring.")
    args = ap.parse_args()

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(args.output_dir, exist_ok=True)
    wanted = dict(SUBSETS)
    if args.no_sizematched:
        wanted.pop("sizematched")
    full = pd.read_csv(args.full_predictions, sep="\t")
    preds = {"original": full}
    preds.update(subset_predictions(args.cache_dir, wanted, seed=args.seed))

    for name, df in preds.items():
        nz = df[~df["context"].isin(M.CPG_CONTEXTS)]
        print(f"{name:<9} non-CpG: {len(nz):,} sites, {int(nz['label'].sum()):,} DNMs, "
              f"{len(nz) / max(int(nz['label'].sum()), 1) - 1:.1f} background per DNM")

    # Shared bin edges spanning every population, so they land in the same bins.
    gc_all = np.concatenate([d["gc"].to_numpy() for d in preds.values()])
    edges = np.linspace(gc_all.min(), gc_all.max(), args.n_bins + 1)
    edges[-1] += 1e-9

    binned = {k: non_cpg_binned(v, args.n_bins, edges, per_context=args.per_context)
              for k, v in preds.items()}

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    panels.panel_dnm_probability_pairs(ax, binned, min_n=args.min_n,
                                       normalize=args.normalize)
    stem = os.path.join(args.output_dir, "dnm_probability_pairs"
                        + ("_percontext" if args.per_context else "")
                        + ("_normalized" if args.normalize else ""))
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    fig.savefig(stem + ".png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {stem}.pdf")

    rows = []
    for name, df in binned.items():
        d = df[df["n"] >= args.min_n].copy()
        d["population"] = name
        d["gc"] = d["gc_mid"] / 100.0
        rows.append(d)
    tab = pd.concat(rows)
    tab.to_csv(stem + ".txt", sep="\t", index=False, float_format="%.6g")
    pd.set_option("display.width", 220)
    print(tab[["population", "gc", "n", "n1", "mean_pred", "empirical_prop", "se"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
