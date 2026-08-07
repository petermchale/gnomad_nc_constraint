"""
Refit Gnocchi's regional adjustment on a randomly SHRUNK DNM training set, and apply it
genome-wide.

    .venv/bin/python dnm_training_size/run_dnm_training_experiment.py -subsample_frac 0.01
    .venv/bin/python dnm_training_size/run_dnm_training_experiment.py -subsample_frac 0.1

"Regime 1": subsample dnm1 (mutated) and dnm0 (background) independently at the same
rate, so the class balance and the sampling scheme are unchanged and only the AMOUNT of
data differs. Everything downstream is the identical pipeline -- univariate Bonferroni
selection -> standardize -> IncrementalPCA -> L1 logit per context -> genome-wide apply.

WHAT THIS TESTS, and how it differs from fig5's size-matched control.
chen_formula.tex ("Predictions of the hypothesis") predicts that as the training set
shrinks, sparse feature tails collapse r_c(x) toward 1, so Gnocchi's GC bias should
collapse toward the context-only model's. It does, smoothly and monotonically across the
whole GC range -- but it never goes PAST the context-only model. fig5's intervention,
which changes the training POPULATION at fixed size, does go past it (0.046 vs the
context-only 0.093). Having both is what shows the population fix differs in kind from
merely having less data or more shrinkage; fig5's size-matched control tests a single
size and cannot show the dose-response.

A context with no Bonferroni-significant feature under the subsample, or whose fit fails
to converge, is left out of the adjustment and defaults to r = 1 -- which is
run_nc_constraint_gnomad_v31_main.py line 260's own fallback, and mechanistically what
the prediction above says should happen as data shrinks.

Outputs, per size, into output/ (the expected-count table is what the notebook plots;
the selected-feature table carries the second line of evidence, the GC_content selection
frequency):

    expected_counts_by_context_methyl_genome_1kb.dnm_refit_{tag}.txt
    selected.dnm_refit_{tag}.txt, coef_univariate.dnm_refit_{tag}.txt
    rr_by_context.dnm_refit_{tag}.txt        (large, and nothing here reads it)

The FULL-scale refit is not run from here -- `fig5/refit.py -population full` produces it
into the shared repo-root refits/, which is where the notebook reads it from.
Validation of the reimplementation itself lives in validate_reimplementation/.
"""

import argparse
import os
import sys
import time

import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# The training data, the fitting core, and the genome-wide apply step live in
# gnocchi_bias.dnm_model, shared with fig5/. This script is only the size sweep.
from gnocchi_bias.dnm_model import (  # noqa: E402
    GENOME_EXPECTED_PERCONTEXT_FILE, GENOME_FEATURES_FILE, load_contexts,
    load_training_data, refit_and_apply, subsample_regime1,
)
from gnocchi_bias.windows import download  # noqa: E402

DEFAULT_CACHE_DIR = os.path.join(_REPO_ROOT, "tmp")
DEFAULT_OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "output")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-subsample_frac", type=float, required=True,
                    help="regime-1 random subsample rate, applied independently to both "
                         "dnm0 and dnm1 (e.g. 0.01 for 1%%)")
    ap.add_argument("-cache_dir", default=DEFAULT_CACHE_DIR,
                    help="downloaded bucket files, shared with the rest of the repo")
    ap.add_argument("-output_dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("-random_seed", type=int, default=0)
    ap.add_argument("-tag", default=None,
                    help="label for output filenames; default derived from "
                         "-subsample_frac/-random_seed")
    ap.add_argument("-max_contexts", type=int, default=None,
                    help="DEBUG ONLY: restrict to the first N contexts (alphabetical), "
                         "for a fast smoke test -- not a real result at any N < 32")
    args = ap.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    tag = args.tag or f"frac{args.subsample_frac}_seed{args.random_seed}"

    contexts = load_contexts(args.cache_dir)
    if args.max_contexts is not None:
        contexts = contexts[:args.max_contexts]
    df_dnm1, df_dnm0 = load_training_data(args.cache_dir)
    print(f"{len(contexts)} contexts, dnm1={len(df_dnm1):,} rows, dnm0={len(df_dnm0):,} rows")

    df_dnm1_sub, df_dnm0_sub = subsample_regime1(
        df_dnm1, df_dnm0, args.subsample_frac, args.random_seed)
    print(f"\nsubsampled (regime 1, frac={args.subsample_frac}, seed={args.random_seed}): "
          f"dnm1 {len(df_dnm1):,}->{len(df_dnm1_sub):,}, "
          f"dnm0 {len(df_dnm0):,}->{len(df_dnm0_sub):,}")

    print("\nloading genome-wide features table (large; cached after first run)...")
    t0 = time.time()
    df_ft_genome = pd.read_csv(download(GENOME_FEATURES_FILE, args.cache_dir), sep="\t")
    print(f"  loaded {len(df_ft_genome):,} windows in {time.time() - t0:.1f}s")

    t0 = time.time()
    df_out = refit_and_apply(df_dnm1_sub, df_dnm0_sub, contexts, df_ft_genome,
                             download(GENOME_EXPECTED_PERCONTEXT_FILE, args.cache_dir),
                             args.output_dir, tag)
    print(f"\nrefit+apply total: {time.time() - t0:.1f}s")

    out = os.path.join(
        args.output_dir, f"expected_counts_by_context_methyl_genome_1kb.dnm_refit_{tag}.txt")
    df_out.to_csv(out, sep="\t", index=False)
    print(f"wrote {out} ({len(df_out):,} rows)")


if __name__ == "__main__":
    main()
