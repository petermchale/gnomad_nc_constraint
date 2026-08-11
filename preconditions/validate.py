"""
Does this repo's reimplementation of Chen et al.'s fitting pipeline reproduce theirs?

    .venv/bin/python preconditions/validate.py [-check {both,coefficients,expected}]

fig5/ and dnm_training_size/ both rest on gnocchi_bias/dnm_model.py reproducing the
published pipeline, because the multivariate PCA+logit step that produces r(w) has no
published source (the bucket ships fitted .pkl models and the apply side, never the
fitting code). This script holds the two checks that claim rests on:

  coefficients  The UNIVARIATE stage against Chen et al.'s own fitted coefficient
                table -- the repo's only check against published FITTED PARAMETERS
                rather than a downstream output, so the one that says the fitting code
                agrees rather than that the numbers come out the same. ~2 min with the
                2.5 GB of training tables cached; the fitting itself is 77s.

  expected      END-TO-END: the full-population refit's genome-wide expected counts
                against the published `expected` column. The only way to validate the
                multivariate step, which has no published parameters to diff against.
                Seconds; needs `fig5/refit.py -population full` first.

Neither substitutes for the other: `coefficients` checks a stage the figures never use
directly, `expected` checks a validated stage composed with one that cannot be.

fig5 carries two further downstream checks, so they are not repeated here: per-GC-bin
r_eff refit-vs-published (max 1.0e-4, printed on every run) and the full-population
refit landing on published Gnocchi in panel E (mean |rank-0.5| 0.212 vs 0.212).
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
    PUBLISHED_COEF_FILE,
    PUBLISHED_SEL_FILE,
    bonferroni_select,
    fit_univariate,
    load_contexts,
    load_training_data,
)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
FULL_REFIT_EXPECTED = os.path.join(
    REPO_ROOT, "refits", "expected_counts_by_context_methyl_genome_1kb.full.txt")


def validate_against_published(df_coef: pd.DataFrame, published_path: str,
                                published_sel_path: str | None = None):
    """
    Is fit_univariate's output the same as Chen et al.'s? Two comparisons, printed.

    This is the repo's ONLY check against a published FITTED-PARAMETER file rather than
    against a downstream output, which is what lets it say the fitting code agrees --
    not merely that the numbers come out the same after several more stages. (The
    multivariate step has no such check available here: the bucket ships fitted .pkl
    models per context, but nothing in this repo currently diffs against them.)

    1. COEFFICIENTS, reported scale-free as |coef_new - coef_pub| / se_pub. Read that
       one, not the absolute difference: the coefficients are order 1e-2 (median |coef|
       0.027), so an absolute tolerance means nothing without that context, and relative
       error is worse still -- it reaches 172% on coefficients of order 1e-5 that are
       statistically indistinguishable from zero. Dividing by the standard error the
       original fit already reported asks the right question: does the refit land inside
       the uncertainty Chen et al. themselves had? Measured: max 0.0212, median 0.0008
       over all 1,664 (context, window, feature) rows.
    2. SELECTION, if `published_sel_path` is given: our Bonferroni-surviving rows against
       their own selected-feature file. Stronger than any coefficient tolerance, because
       it is a verdict rather than a distance -- and because the selected set, not the
       coefficients, is what each context's multivariate model is then fit on. Measured:
       239 rows each, none on either side alone. Without the path it falls back to
       applying bonferroni_select to their coefficients, which only tests our rule.

    The merge is an OUTER join, so a row whose fit failed on one side alone shows up as a
    NaN mismatch and is listed rather than silently dropped; rows that failed on both are
    counted and excluded from the distance statistics.

    Args:
        df_coef: fit_univariate's output -- context, window, feature, coef, se, pval.
        published_path: Chen et al.'s fitted coefficient table (PUBLISHED_COEF_FILE).
        published_sel_path: their selected-feature file (PUBLISHED_SEL_FILE). Optional
            only so the coefficient half can run alone; pass it when you have it.

    Returns the outer-joined frame, for inspecting individual rows. Prints rather than
    raises: preconditions/validate.py is read by a human, and the interesting outcome is
    a distribution, not a boolean.
    """
    df_pub = pd.read_csv(published_path, sep='\t')
    merged = df_coef.merge(df_pub, on=['context', 'window', 'feature'], suffixes=('_new', '_pub'), how='outer')
    n_total = len(merged)
    n_both_nan = (merged['coef_new'].isna() & merged['coef_pub'].isna()).sum()
    n_nan_mismatch = (merged['coef_new'].isna() != merged['coef_pub'].isna()).sum()

    ok = merged.dropna(subset=['coef_new', 'coef_pub'])
    for col in ['coef', 'se', 'pval']:
        ok = ok.assign(**{f'{col}_diff': (ok[f'{col}_new'] - ok[f'{col}_pub']).abs()})
    # The headline statistic. An absolute |coef diff| cannot be read without knowing the
    # coefficients' scale, and it is NOT order 1: median |coef| is 0.027. Relative error
    # is no better -- it blows up to 172% on coefficients of order 1e-5 that are
    # indistinguishable from zero anyway. Dividing by the published standard error is
    # scale-free and is the natural yardstick for a fitted coefficient: it asks whether
    # the refit lands inside the uncertainty the original fit already had.
    ok = ok.assign(coef_diff_se=ok['coef_diff'] / ok['se_pub'])

    print(f"\nrows: {n_total}  both-NaN (fit failed in both): {n_both_nan}  "
          f"NaN-mismatch (fit failed in only one): {n_nan_mismatch}  comparable: {len(ok)}")
    if len(ok):
        print(f"max |coef diff| / se_pub = {ok['coef_diff_se'].max():.4f}   "
              f"(median {ok['coef_diff_se'].median():.4f})  <-- scale-free; 1.0 would be "
              f"a full standard error")
        print(f"max |coef diff| = {ok['coef_diff'].max():.3e}   "
              f"max |se diff|   = {ok['se_diff'].max():.3e}   "
              f"max |pval diff| = {ok['pval_diff'].max():.3e}")
        print(f"coef matches to <1e-6: {(ok['coef_diff'] < 1e-6).sum()}/{len(ok)}   "
              f"<1e-3: {(ok['coef_diff'] < 1e-3).sum()}/{len(ok)}   "
              f"(absolute, against median |coef| = {ok['coef_pub'].abs().median():.4f})")
    # What propagates downstream is not the coefficients themselves but which rows clear
    # Bonferroni: the selected set is what each context's multivariate model is fit on.
    # Comparing it sidesteps the scale question entirely -- a verdict either flips or it
    # does not -- and against published_sel_path it compares our selection with Chen et
    # al.'s OWN selection output, not merely with our logic re-applied to their coefs.
    key = ['context', 'feature', 'window']
    sel_new = bonferroni_select(df_coef)[key]
    sel_ref = (pd.read_csv(published_sel_path, sep='\t')[key] if published_sel_path
               else bonferroni_select(df_pub)[key])
    against = "published selected file" if published_sel_path else "their coefs, our rule"
    agree = sel_new.merge(sel_ref, how='outer', indicator=True)['_merge'].value_counts()
    print(f"selected features vs {against}: ours {len(sel_new)}, theirs {len(sel_ref)}, "
          f"in both {agree.get('both', 0)}, ours only {agree.get('left_only', 0)}, "
          f"theirs only {agree.get('right_only', 0)}")

    if n_nan_mismatch:
        print("rows where fit succeeded in exactly one of new/published:")
        print(merged[merged['coef_new'].isna() != merged['coef_pub'].isna()]
              [['context', 'window', 'feature', 'coef_new', 'coef_pub']].to_string(index=False))
    return merged


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
    validate_against_published(df_coef,
                               W.download(PUBLISHED_COEF_FILE, cache_dir),
                               W.download(PUBLISHED_SEL_FILE, cache_dir))


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
    print("\nend-to-end expected counts, refit vs published:")
    print(f"  {n:,} windows joined")
    print(f"  Pearson r                     = {r:.6f}")
    print(f"  median relative difference    = {med:.2e}")
    print(f"  max relative difference       = {mx:.2e}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-check", choices=["both", "coefficients", "expected"], default="both")
    ap.add_argument("-cache_dir", default=os.path.join(REPO_ROOT, "published"))
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
