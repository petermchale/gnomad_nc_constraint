"""
Does this repo's reimplementation of Chen et al.'s fitting pipeline reproduce theirs?

    .venv/bin/python validate_reimplementation/validate.py [-check {both,coefficients,expected}]

Everything downstream -- fig5/ and dnm_training_size/ alike -- rests on the claim that
`gnocchi_bias/dnm_model.py` reproduces the published pipeline, because the multivariate
PCA+logit step that produces r(w) has NO published source anywhere (the bucket ships the
fitted .pkl models and the apply-side code, never the fitting code). This directory holds
the two checks that claim rests on, kept separate from the experiments that assume it.

  coefficients  The UNIVARIATE feature-selection stage, against Chen et al.'s own fitted
                coefficient table. This is the only check anywhere in the repo against a
                published FITTED-PARAMETER file rather than against a downstream output,
                so it is the one that says the fitting code itself agrees rather than
                that the numbers happen to come out the same. ~10 min: it refits all
                (context, window, feature) triples on the full training set.

  expected      The END-TO-END check: the full-population refit's genome-wide expected
                counts against the published Gnocchi `expected` column. Covers the
                multivariate step, which has no published parameters to diff against, so
                it can only be validated through its output. Seconds; needs
                `fig5/refit.py -population full` to have been run.

The two are complementary and neither substitutes for the other: `coefficients` checks a
stage whose output the figures never use directly, `expected` checks the composite of a
stage that is validated and a stage that cannot be.

fig5 carries two further downstream checks, which is why they are not repeated here:
per-GC-bin r_eff refit-vs-published (max 1.0e-4, printed on every run), and the
full-population refit landing on published Gnocchi in panel E (mean |rank-0.5| 0.212 vs
0.212).
"""
import argparse
import os
import sys
import time

import duckdb
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnocchi_bias import windows as W  # noqa: E402
from gnocchi_bias.dnm_model import (  # noqa: E402
    PUBLISHED_COEF_FILE, fit_univariate, load_contexts, load_training_data,
    validate_against_published,
)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
FULL_REFIT_EXPECTED = os.path.join(
    REPO_ROOT, "refits", "expected_counts_by_context_methyl_genome_1kb.full.txt")


def check_coefficients(cache_dir: str, output_dir: str) -> None:
    """Refit univariate selection on the full training set and diff against published."""
    contexts = load_contexts(cache_dir)
    df_dnm1, df_dnm0 = load_training_data(cache_dir)
    print(f"{len(contexts)} contexts, dnm1={len(df_dnm1):,} rows, dnm0={len(df_dnm0):,} rows")

    print("\nfitting univariate feature selection on the FULL, unmodified training set...")
    t0 = time.time()
    df_coef = fit_univariate(df_dnm1, df_dnm0, contexts)
    print(f"  done in {time.time() - t0:.1f}s")

    out = os.path.join(output_dir, "coef_univariate.txt")
    df_coef.to_csv(out, sep="\t", index=False)
    print(f"wrote {out}")
    validate_against_published(df_coef, W.download(PUBLISHED_COEF_FILE, cache_dir))


def check_expected(cache_dir: str, refit_expected: str, memory_limit: str = "8GB") -> None:
    """
    Per-window Pearson r and median relative difference between the full-population
    refit's expected counts and the published ones.

    Joined on element_id, so it compares the same windows; no filtering is applied,
    because the question is whether the pipeline reproduces the published table
    everywhere, not whether it does so on the analyzed subset.
    """
    if not os.path.exists(refit_expected):
        raise SystemExit(f"missing {refit_expected}\n"
                         "Run:  .venv/bin/python fig5/refit.py -population full")
    annot = W.download(W.REMOTE_FILES["annot"], cache_dir)
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{memory_limit}'")
    row = con.execute(f"""
        WITH j AS (
            SELECT p.expected AS pub, r.expected AS refit
            FROM (SELECT element_id, expected FROM read_csv_auto('{annot}', delim='\t',
                  header=True)) p
            INNER JOIN read_csv_auto('{refit_expected}', delim='\t', header=True) r
              ON p.element_id = r.element_id
            WHERE p.expected > 0)
        SELECT COUNT(*), corr(pub, refit),
               median(abs(refit - pub) / pub), max(abs(refit - pub) / pub)
        FROM j
    """).fetchone()
    n, r, med, mx = row
    print(f"\nend-to-end expected counts, refit vs published:")
    print(f"  {n:,} windows joined")
    print(f"  Pearson r                     = {r:.6f}")
    print(f"  median relative difference    = {med:.2e}")
    print(f"  max relative difference       = {mx:.2e}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-check", choices=["both", "coefficients", "expected"], default="both")
    ap.add_argument("-cache_dir", default=os.path.join(REPO_ROOT, "tmp"))
    ap.add_argument("-output_dir", default=os.path.join(HERE, "output"))
    ap.add_argument("-refit_expected", default=FULL_REFIT_EXPECTED)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.check in ("both", "expected"):
        check_expected(args.cache_dir, args.refit_expected)
    if args.check in ("both", "coefficients"):
        check_coefficients(args.cache_dir, args.output_dir)


if __name__ == "__main__":
    main()
