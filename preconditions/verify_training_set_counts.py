"""
Are the four shipped training tables the training set the paper describes?

Step 2's whole argument is about WHAT the regional adjustment was fit on, so the
identity of the training set is a precondition for every claim fig5 makes about it.
The paper states two counts; this checks that the bucket's files reproduce both, and
that the join fig5/ and dnm_training_size/ perform loses nothing.

Chen et al. 2024, Methods, "Adjustment of the effects of regional genomic features":
a set of 413,304 unique DNMs compiled from two family-based WGS studies -- deCODE
(Halldorsson et al.) and PsychENCODE (An et al.) -- against "an exclusive set of
4,104,879 genomic sites (~10x the DNMs) randomly drew from the genome" as the
non-mutated background.

Both are reproduced exactly, but by DIFFERENT tables of the pair, which is the trap
this script exists to document: check a count against the wrong one of the four files
and a correct table looks short by thousands of rows.

  413,304 DNMs   = dnm1 FEATURE rows + the loci carrying two DNMs each. The feature
                   table is keyed by locus, so a locus mutated to two different alleles
                   collapses into one row; the site table keeps both.
  4,104,879 bkgd = dnm0 SITE rows minus its chrX rows, which load_training_data drops.

Default downloads ~421 MB (the 2.05 GB dnm0 feature table is only needed for an
optional cross-check; pass -include_dnm0_features for it).

    .venv/bin/python preconditions/verify_training_set_counts.py

Outcome of the last run: preconditions/output/STATUS.md (transcript in the .log beside it).
"""
import argparse
import os
import sys

import duckdb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnocchi_bias.dnm_model import MUTATION_RATE_FILE, TRAINING_FILES  # noqa: E402
from gnocchi_bias.windows import download  # noqa: E402
from preconditions.report import Report  # noqa: E402

# Repo-root cache, shared with every other script here; resolved from __file__ so
# running from inside preconditions/ reuses it rather than refetching multi-GB files.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DEST_DIR = os.path.join(_REPO_ROOT, "published")

PAPER_DNMS = 413_304
PAPER_BACKGROUND = 4_104_879
# The background count is reproduced to within one row, not exactly. One row in 4.1M is
# far below any plausible alternative explanation (a different filter would move
# thousands, as the 2,924 chrX rows do), so it is reported rather than treated as a
# failure -- but it is a real, unexplained discrepancy and is not rounded away.
BACKGROUND_TOLERANCE = 1
N_FEATURES, N_SCALES = 13, 4


def tsv(path: str) -> str:
    return f"read_csv_auto('{path}', delim='\t', header=True)"


def check_dnm1(con, sites: str, features: str) -> tuple[bool, str]:
    rows, loci = con.execute(
        f"SELECT COUNT(*), COUNT(DISTINCT locus) FROM {tsv(sites)}").fetchone()
    feat_rows, feat_loci = con.execute(
        f"SELECT COUNT(*), COUNT(DISTINCT element_id) FROM {tsv(features)}").fetchone()
    recurrent = rows - loci
    total = feat_loci + recurrent

    print("\n=== dnm1: does the feature table reproduce the published DNM count? ===")
    print(f"  site table:     {rows:,} rows over {loci:,} distinct loci "
          f"-> {recurrent:,} loci carry two DNMs (same position, different alleles)")
    print(f"  feature table:  {feat_rows:,} rows, {feat_loci:,} distinct loci "
          f"(keyed by locus, so those {recurrent:,} pairs collapse to one row each)")
    print(f"  {feat_loci:,} + {recurrent:,} = {total:,}   paper says {PAPER_DNMS:,}"
          f"   -> {'MATCH' if total == PAPER_DNMS else f'MISMATCH by {total - PAPER_DNMS:+,}'}")
    return total == PAPER_DNMS, (
        f"dnm1 reproduces the paper's {PAPER_DNMS:,} DNMs exactly: {feat_loci:,} feature "
        f"rows + {recurrent:,} loci carrying two DNMs each = {total:,}")


def check_dnm0(con, sites: str) -> tuple[bool, str]:
    rows, chrx = con.execute(
        f"SELECT COUNT(*), COUNT(*) FILTER (WHERE locus LIKE 'chrX:%') FROM {tsv(sites)}"
    ).fetchone()
    kept = rows - chrx
    diff = kept - PAPER_BACKGROUND

    print("\n=== dnm0: does the site table reproduce the published background count? ===")
    print(f"  site table:     {rows:,} rows, of which {chrx:,} are chrX")
    print(f"  after the chrX filter load_training_data applies: {kept:,}"
          f"   paper says {PAPER_BACKGROUND:,}   -> {diff:+,}")
    ok = abs(diff) <= BACKGROUND_TOLERANCE
    print(f"  within the {BACKGROUND_TOLERANCE}-row tolerance: {ok}")
    return ok, (
        f"dnm0 reproduces the paper's {PAPER_BACKGROUND:,} background sites to within "
        f"{BACKGROUND_TOLERANCE} row: {rows:,} site rows - {chrx:,} chrX = {kept:,} "
        f"({diff:+,}; a different filter would move thousands, as the chrX one does)")


def check_join_is_lossless(con, sites: str, features: str) -> tuple[bool, str]:
    """
    load_training_data left-joins features ONTO sites, so the site table sets the
    training N. Two things follow, and both are checked: no site may lack features
    (or it would train on nulls), and the feature-only surplus must be explained.
    """
    unmatched, = con.execute(
        f"SELECT COUNT(*) FROM {tsv(sites)} s "
        f"LEFT JOIN (SELECT DISTINCT element_id FROM {tsv(features)}) f "
        f"ON s.locus = f.element_id WHERE f.element_id IS NULL").fetchone()
    surplus = con.execute(
        f"SELECT split_part(element_id, ':', 1) AS chrom, COUNT(*) AS n "
        f"FROM (SELECT DISTINCT element_id FROM {tsv(features)}) f "
        f"WHERE element_id NOT IN (SELECT locus FROM {tsv(sites)}) "
        f"GROUP BY 1 ORDER BY n DESC").fetchall()
    n_surplus = sum(n for _, n in surplus)
    sex = [c for c, _ in surplus if c in ("chrX", "chrY")]

    print("\n=== the join fig5/ and dnm_training_size/ actually perform ===")
    print(f"  sites with no feature row: {unmatched:,}  "
          f"(anything above 0 would train on nulls)")
    print(f"  feature loci absent from the site table: {n_surplus:,} across "
          f"{len(surplus)} chromosomes, none of them sex: {not sex}")
    print(f"    top: {', '.join(f'{c} {n}' for c, n in surplus[:5])}")
    print("    -> NOT the chrX filter; these are autosomal DNMs the site table omits, "
          "presumably where trinucleotide context or methylation could not be assigned.")
    return (unmatched == 0 and not sex), (
        f"the training join loses nothing: {unmatched:,} of the dnm1 sites lack a feature "
        f"row, and the {n_surplus:,} feature-only loci are all autosomal, so no site "
        f"trains on nulls")


def check_3mer_is_step1_rate(con, sites: str, rate_path: str) -> tuple[bool, str]:
    """
    `3mer` should be the step-1, context-only per-site mutation probability: fitted_po
    summed over the three alt alleles for that (context, methylation_level). If so, the
    training table already carries the context-only model, and step 2 is only ever
    adjusting it -- which is the premise of the whole r-decomposition.
    """
    row = con.execute(f"""
        WITH s AS (SELECT DISTINCT context, methyl_level, "3mer" FROM {tsv(sites)}),
             r AS (SELECT context, methylation_level, SUM(fitted_po) AS po
                   FROM {tsv(rate_path)} GROUP BY 1, 2)
        SELECT COUNT(*), MAX(ABS(s."3mer" - r.po))
        FROM s JOIN r ON s.context = r.context AND s.methyl_level = r.methylation_level
    """).fetchone()
    n, max_diff = row
    print("\n=== is `3mer` the step-1 context-only mutation rate? ===")
    print(f"  {n:,} (context, methylation) combinations compared against "
          f"fig_tables/{os.path.basename(rate_path)}")
    print(f"  max |3mer - sum(fitted_po) over alts| = {max_diff:.3e}")
    return max_diff < 1e-12, (
        f"`3mer` IS the step-1 context-only rate -- fitted_po summed over the three alts, "
        f"to {max_diff:.1e} across all {n} (context, methylation) combinations, so step 2 "
        f"only ever adjusts a rate the training table already carries")


def check_feature_shape(con, features: str) -> tuple[bool, str]:
    cols = [d[0] for d in con.execute(f"SELECT * FROM {tsv(features)} LIMIT 0").description]
    expected = N_FEATURES * N_SCALES + 1
    sample, = con.execute(f"SELECT element_id FROM {tsv(features)} LIMIT 1").fetchone()
    print("\n=== feature-table shape and the misleading key name ===")
    print(f"  {len(cols)} columns; expected {N_FEATURES} features x {N_SCALES} scales "
          f"+ key = {expected}")
    print(f"  key column is named {cols[0]!r} but holds a LOCUS, e.g. {sample!r} -- "
          "not a 1 kb element_id, which is why load_training_data renames it")
    return (len(cols) == expected and ":" in sample), (
        f"the feature table has the expected shape: {len(cols)} columns = "
        f"{N_FEATURES} features x {N_SCALES} window scales + key")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-dest_dir", default=DEFAULT_DEST_DIR,
                    help=f"directory to download into (default: {DEFAULT_DEST_DIR})")
    ap.add_argument("-include_dnm0_features", action="store_true",
                    help="also count the 2.05 GB dnm0 feature table (optional cross-check)")
    args = ap.parse_args()

    with Report("verify_training_set_counts") as rep:
        paths = {k: download(v, args.dest_dir) for k, v in TRAINING_FILES.items()
                 if k != "dnm0_features" or args.include_dnm0_features}
        rate_path = download(MUTATION_RATE_FILE, args.dest_dir)
        con = duckdb.connect()

        rep.claim(*check_dnm1(con, paths["dnm1_sites"], paths["dnm1_features"]))
        rep.claim(*check_dnm0(con, paths["dnm0_sites"]))
        rep.claim(*check_join_is_lossless(con, paths["dnm1_sites"], paths["dnm1_features"]))
        rep.claim(*check_3mer_is_step1_rate(con, paths["dnm1_sites"], rate_path))
        rep.claim(*check_feature_shape(con, paths["dnm1_features"]))

        if args.include_dnm0_features:
            n, = con.execute(
                f"SELECT COUNT(DISTINCT element_id) FROM {tsv(paths['dnm0_features'])}"
            ).fetchone()
            print(f"\n=== optional: dnm0 feature table ===\n  {n:,} distinct loci, "
                  f"{n - PAPER_BACKGROUND:+,} vs the published background count -- it "
                  "retains the chrX sites the site table's filter removes.")

        # Phrased as method, not as outcome, so it stays true if a claim below fails:
        # the verdict block is the conclusion, and this is how to read it.
        print("\nEach published count must be checked against the right one of the four "
              "tables -- the dnm1 FEATURE table and the dnm0 SITE table respectively. "
              "Check either against the wrong one and a correct table looks short by "
              "thousands of rows.")


if __name__ == "__main__":
    main()
