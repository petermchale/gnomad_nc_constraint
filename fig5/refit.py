"""
Refit Gnocchi's regional adjustment on one training population and apply it genome-wide.

    .venv/bin/python fig5/refit.py -population {full,scored,sizematched}

  full          the training set as Chen et al. use it. Reproduces the published
                Gnocchi expected counts at Pearson r = 1.0 (median relative difference
                4e-6), so it is both the source of the per-context r panel B needs --
                the published pipeline writes its own to a local dir, never to the
                bucket -- and the control for "is the effect just the reimplementation?"
  scored        THE INTERVENTION: training sites restricted to the analyzed window
                set, i.e. the population r is actually applied to.
  sizematched   the control for "is it just less data?": the same NUMBER of sites as
                `scored`, drawn uniformly at random from the whole genome.

Only which training rows are used differs. Everything downstream is the identical
pipeline: univariate Bonferroni selection -> standardize -> IncrementalPCA -> L1 logit
per context -> genome-wide apply. So any change downstream is attributable to the
training population (or, for `sizematched`, to its size).

Writes into the repo-root refits/ -- ONE copy of each table, also read directly by
dnm_training_size/. ~6 min and ~4 GB per population, dominated by the genome-wide apply and the
per-context rr table (the whole directory is gitignored):

    expected_counts_by_context_methyl_genome_1kb.{pop}.txt   panels B, E
    rr_by_context.{pop}.txt                                  panel B
    training_reliability_predictions.{pop}.txt               panel D
    selected.{pop}.txt, coef_univariate.{pop}.txt            the fit itself
"""
import argparse
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402
from gnocchi_bias import dnm_model as M  # noqa: E402
from gnocchi_bias import windows as W  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-population", required=True,
                    choices=["full", "scored", "sizematched"])
    ap.add_argument("-cache_dir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "published"))
    ap.add_argument("-output_dir", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "refits"))
    ap.add_argument("-seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    t_start = time.time()

    contexts = M.load_contexts(args.cache_dir)
    df_dnm1, df_dnm0 = M.load_training_data(args.cache_dir)
    print(f"full training set: {len(df_dnm1):,} DNMs / {len(df_dnm0):,} background")

    if args.population != "full":
        # The analyzed window set, defined exactly as everywhere else in fig5/.
        # GENEHANCER_BED comes from config.py, which the notebook reads too, so the
        # population fit on here is the population the panels are scored on. The value
        # is stamped below and re-checked at read time.
        element_ids = W.build_window_table(
            args.cache_dir, genehancer_bed=config.GENEHANCER_BED)["element_id"].to_list()
        n1, n0 = M.count_in_analyzed_windows(df_dnm1, df_dnm0, element_ids)
        if args.population == "scored":
            df_dnm1, df_dnm0 = M.restrict_to_analyzed_windows(df_dnm1, df_dnm0, element_ids)
        else:
            print(f"CONTROL: sampling {n1:,} / {n0:,} sites uniformly at random from "
                  f"the whole genome (same size as `scored`, same population as `full`)")
            df_dnm1 = df_dnm1.sample(n=n1, random_state=args.seed)
            df_dnm0 = df_dnm0.sample(n=n0, random_state=args.seed)
        if df_dnm1.empty or df_dnm0.empty:
            raise SystemExit("training population is empty; check the element_id mapping")
        print(f"training on {len(df_dnm1):,} DNMs / {len(df_dnm0):,} background")

    df_ft_genome = pd.read_csv(W.download(M.GENOME_FEATURES_FILE, args.cache_dir), sep="\t")
    percontext = W.download(M.GENOME_EXPECTED_PERCONTEXT_FILE, args.cache_dir)
    df_out = M.refit_and_apply(df_dnm1, df_dnm0, contexts, df_ft_genome, percontext,
                               output_dir=args.output_dir, tag=args.population)
    # refit_and_apply hardcodes a "dnm_refit_" prefix on the three tables it writes
    # itself; drop it so everything in refits/ is uniformly {name}.{population}.txt.
    for name in ("coef_univariate", "selected", "rr_by_context"):
        src = os.path.join(args.output_dir, f"{name}.dnm_refit_{args.population}.txt")
        if os.path.exists(src):
            os.replace(src, os.path.join(args.output_dir, f"{name}.{args.population}.txt"))

    out = os.path.join(
        args.output_dir,
        f"expected_counts_by_context_methyl_genome_1kb.{args.population}.txt")
    df_out.to_csv(out, sep="\t", index=False)
    print(f"wrote {out}  ({len(df_out):,} windows)")

    # Panel D: each site's own fitted probability, from its own context's model,
    # evaluated on the SITE's feature vector -- not the window-aggregated values
    # apply_genome_wide_context uses. Refits the per-context models (seconds); the
    # genome-wide apply above is what costs time.
    df_sel = pd.read_csv(
        os.path.join(args.output_dir, f"selected.{args.population}.txt"), sep="\t")
    df_pred = M.predict_training_set(df_dnm1, df_dnm0, contexts, df_sel)
    pred_out = os.path.join(
        args.output_dir, f"training_reliability_predictions.{args.population}.txt")
    df_pred.to_csv(pred_out, sep="\t", index=False)
    print(f"wrote {pred_out}  ({len(df_pred):,} sites)")

    config.record(args.output_dir, args.population, config.GENEHANCER_BED)
    print(f"stamped GENEHANCER_BED={config.GENEHANCER_BED!r} for {args.population!r}")
    print(f"total {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
