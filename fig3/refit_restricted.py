"""
SUPERSEDED by fig5/refit.py, which does the same thing for all three populations
(full / scored / sizematched), also writes the per-site predictions, writes into the
shared repo-root refits/ under uniform {table}.{population}.txt names, and stamps the
GENEHANCER_BED it used so a stale refit cannot be read silently.

Every consumer -- fig3/compare_restricted.py, fig3/plot_dnm_probability*.py,
fig3/r_eff.py, and all of fig5/ -- now reads from refits/. THIS SCRIPT STILL WRITES TO
fig3/output/ UNDER ITS OLD NAMES, so anything it produces will be ignored. Run
`fig5/refit.py -population scored` instead.

Refit Gnocchi's regional-adjustment model on a training set restricted to the
population it is actually applied to, then apply it genome-wide.

    .venv/bin/python fig3/refit_restricted.py [-cache_dir tmp] [-output_dir fig3/output]

THE INTERVENTION. Chen et al. fit their per-context DNM models on training sites drawn
from the whole genome, but r(w) is applied to -- and Gnocchi is scored on -- noncoding,
pass_qc, autosome/PAR windows. fig3/training_representativeness.py measures how far
apart those two populations drift: the fraction of background training sites inside the
analyzed set falls from 0.83 at GC 0.37 to 0.30 by GC 0.68, because GC-rich sequence is
disproportionately coding or lacks gnomAD coverage. It also shows that this population
mismatch -- not the choice of background sites, and not the aggregation -- is what makes
the empirical DNM probability curve change shape (37.6% vs 2.4% and 4.3%).

This script changes exactly one thing: it drops training sites outside the analyzed
window set, then runs the identical pipeline (univariate Bonferroni selection ->
standardize -> IncrementalPCA -> L1 logit, per context -> genome-wide apply). Nothing
else differs from the full-scale refit that reproduces published Gnocchi at Pearson
r = 1.0, so any change downstream is attributable to the restriction.

OUTPUTS, named `restricted` to sit alongside the existing `dnm_refit_full` tables:

    expected_counts_by_context_methyl_genome_1kb.restricted.txt   element_id/possible/expected
    rr_by_context.restricted.txt                                  per-(window, context) r
    selected.restricted.txt, coef_univariate.restricted.txt       the fit itself

Then compare_restricted.py builds the two figures from them.

COST. ~10 min and ~30 GB of writes, dominated by the genome-wide apply and the 4 GB
per-context rr table. Both output tables are gitignored (see .gitignore).
"""

import argparse
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnocchi_bias import dnm_model as M  # noqa: E402
from gnocchi_bias import windows as W  # noqa: E402
from gnocchi_bias.windows import download  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-cache_dir", default="tmp")
    ap.add_argument("-output_dir", default="fig3/output")
    ap.add_argument("-tag", default="restricted")
    ap.add_argument("-control_random", action="store_true",
                    help="THE CONTROL: instead of restricting to the analyzed windows, "
                         "draw the same NUMBER of training sites uniformly at random "
                         "from the whole genome. Isolates population from sample size -- "
                         "if this reproduces the restricted refit's improvement, the "
                         "result is about having less data, not better-matched data.")
    ap.add_argument("-seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    t_start = time.time()

    # The analyzed population, defined exactly as everywhere else in fig3/.
    df_win = W.build_window_table(args.cache_dir)
    print(f"analyzed window set: {df_win.height:,} windows")

    contexts = M.load_contexts(args.cache_dir)
    print("loading training data...")
    df_dnm1, df_dnm0 = M.load_training_data(args.cache_dir)
    print(f"  full training set: {len(df_dnm1):,} DNMs / {len(df_dnm0):,} background")

    n1_kept, n0_kept = M.count_in_analyzed_windows(
        df_dnm1, df_dnm0, df_win["element_id"].to_list())
    if args.control_random:
        print(f"CONTROL: sampling {n1_kept:,} / {n0_kept:,} sites uniformly at random "
              f"from the whole genome (same size as the restricted set, same "
              f"population as the published fit)")
        df_dnm1 = df_dnm1.sample(n=n1_kept, random_state=args.seed)
        df_dnm0 = df_dnm0.sample(n=n0_kept, random_state=args.seed)
    else:
        df_dnm1, df_dnm0 = M.restrict_to_analyzed_windows(
            df_dnm1, df_dnm0, df_win["element_id"].to_list())
        if df_dnm1.empty or df_dnm0.empty:
            raise SystemExit("restriction removed an entire class; check element_id mapping")

    df_ft_genome = pd.read_csv(download(M.GENOME_FEATURES_FILE, args.cache_dir), sep="\t")
    print(f"genome-wide features: {len(df_ft_genome):,} windows")

    percontext = download(M.GENOME_EXPECTED_PERCONTEXT_FILE, args.cache_dir)
    df_out = M.refit_and_apply(df_dnm1, df_dnm0, contexts, df_ft_genome,
                               percontext, output_dir=args.output_dir, tag=args.tag)

    out_path = os.path.join(
        args.output_dir,
        f"expected_counts_by_context_methyl_genome_1kb.{args.tag}.txt")
    df_out.to_csv(out_path, sep="\t", index=False)
    print(f"wrote {out_path}  ({len(df_out):,} windows)")
    print(f"total {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
