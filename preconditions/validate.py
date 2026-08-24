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

The two checks fig5 prints on every run are not additional evidence: both are `expected`
seen downstream. Per-GC-bin r_eff refit-vs-published (max 1.0e-4) is `expected`'s own
ratio aggregated over the analyzed windows -- both sides divide by the same step-1 table,
so r_eff/r_eff_published reduces to refit/published expected -- and the full-population
refit landing on published Gnocchi in panel E (mean |rank-0.5| 0.212 vs 0.212) is that
same ratio after z-scoring and ranking. `expected` is the stronger form, per window and
unaggregated over the whole genome, which is why it is the one carrying a claim here.
They stay in fig5 because they are free on a run that builds the figure anyway, and
because what they add is local to it: r_eff per bin is what licenses panel B's use of the
refit's per-context `rr` in place of the published one, which was never published.

Outcome of the last run: preconditions/output/STATUS.md (transcripts in the .log files
beside it). The thresholds the PASS/FAIL verdicts use are TOLERANCES below, each with
the reason it was set where it is -- they are choices, so read the measured numbers in
the log rather than the verdict alone.
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
from preconditions.report import Report  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
FULL_REFIT_EXPECTED = os.path.join(
    REPO_ROOT, "refits", "expected_counts_by_context_methyl_genome_1kb.full.txt")

# Thresholds for the PASS/FAIL verdicts, set an order of magnitude above what a healthy
# run measures so ordinary solver/library noise cannot trip them while a real
# reimplementation error still would. Each is a CHOICE; the measured value is printed
# and logged next to it.
MAX_COEF_DIFF_SE = 0.25   # measured 0.0212. A quarter of the published fit's own SE:
                          # far inside its uncertainty, far outside rounding.
MIN_PEARSON_R = 0.9999    # measured 1.000000 over 1,984,900 windows.
MAX_MEDIAN_REL_DIFF = 1e-4  # measured 3.8e-6.


def validate_against_published(rep: Report, df_coef: pd.DataFrame, published_path: str,
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
        rep: the Report to record claims on, so output/STATUS.md carries the verdict.
        df_coef: fit_univariate's output -- context, window, feature, coef, se, pval.
        published_path: Chen et al.'s fitted coefficient table (PUBLISHED_COEF_FILE).
        published_sel_path: their selected-feature file (PUBLISHED_SEL_FILE). Optional
            only so the coefficient half can run alone; pass it when you have it.

    Returns the outer-joined frame, for inspecting individual rows. The interesting
    outcome is a distribution, not a boolean, so the distribution is what gets printed;
    the claims exist so a reader who has not run this can see it passed, and each carries
    its measured number for exactly that reason.
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

    n_both = int(agree.get('both', 0))
    rep.claim(n_nan_mismatch == 0,
              f"every one of the {n_total:,} (context, window, feature) rows fits in both "
              f"implementations or neither -- {n_nan_mismatch} one-sided failures")
    rep.claim(len(ok) and ok['coef_diff_se'].max() < MAX_COEF_DIFF_SE,
              f"every coefficient lands within {ok['coef_diff_se'].max():.4f} of the "
              f"published fit's OWN standard error (median {ok['coef_diff_se'].median():.4f}, "
              f"threshold {MAX_COEF_DIFF_SE}) over {len(ok):,} rows")
    rep.claim(n_both == len(sel_new) == len(sel_ref) and len(sel_new) > 0,
              f"our feature selection reproduces theirs EXACTLY against {against}: "
              f"{len(sel_new)} rows each, {n_both} in both, none on either side alone -- "
              f"and the selected set is what each context's multivariate model is fit on")
    return merged


def check_coefficients(rep: Report, cache_dir: str, output_dir: str) -> None:
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
    validate_against_published(rep, df_coef,
                               W.download(PUBLISHED_COEF_FILE, cache_dir),
                               W.download(PUBLISHED_SEL_FILE, cache_dir))


def check_expected(rep: Report, cache_dir: str, refit_expected: str,
                   memory_limit: str = "8GB") -> None:
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

    # The max relative difference is deliberately NOT a claim: it is dominated by windows
    # with a tiny `expected`, where a rounding-level absolute difference is a large
    # relative one. The median is the honest summary of a per-window agreement.
    rep.claim(r >= MIN_PEARSON_R,
              f"the full-population refit reproduces the published genome-wide expected "
              f"counts at Pearson r = {r:.6f} over {n:,} windows (threshold "
              f"{MIN_PEARSON_R}) -- the only available check on the multivariate step, "
              f"which has no published parameters to diff against")
    rep.claim(med < MAX_MEDIAN_REL_DIFF,
              f"median per-window relative difference is {med:.1e} (threshold "
              f"{MAX_MEDIAN_REL_DIFF:.0e}); max is {mx:.1e}, dominated by windows whose "
              f"`expected` is near zero")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-check", choices=["both", "coefficients", "expected"], default="both")
    ap.add_argument("-cache_dir", default=W.CACHE_DIR)
    ap.add_argument("-output_dir", default=os.path.join(HERE, "output"))
    ap.add_argument("-refit_expected", default=FULL_REFIT_EXPECTED)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    # One Report per check, not one for the run: the two are independently runnable
    # (seconds vs minutes) and STATUS.md should never imply that `-check expected` alone
    # vouched for the coefficients.
    if args.check in ("both", "expected"):
        with Report("validate.expected") as rep:
            check_expected(rep, args.cache_dir, args.refit_expected)
    if args.check in ("both", "coefficients"):
        with Report("validate.coefficients") as rep:
            check_coefficients(rep, args.cache_dir, args.output_dir)


if __name__ == "__main__":
    main()
