"""
Re-derive Chen et al.'s window filter from its own inputs, in both directions: every
window they scored satisfies it, and every window they dropped fails it.

They keep a window only if (Methods; run_nc_constraint_gnomad_v31_main.py:296)

    pass_qc = (>= 80% of observed variants PASS) & (mean coverage in 25-35x)
              & (>= 1000 possible variants)

and line 302 drops the `pass` and `coverage` columns before export, so the published
table cannot say which condition a window failed -- or show that a window it kept met
them. Both inputs are in the bucket as misc/genome_1kb_gnomad_v31_{pass,coverage}.txt,
and `possible` is in the step-1 expected-count table, so the filter can simply be
evaluated again on every window and the result compared with membership.

WHICH FILE IS THE SCORED SET. Not
`expected_counts_by_context_methyl_genome_1kb.txt` -- that is the step-1 universe,
2,575,299 windows, and it is exactly the one that still contains the QC failures. The
scored set is `fig_tables/constraint_z_genome_1kb.annot.txt`, whose 1,984,900 rows are
what carries a z. This check works on the step-1 file as the universe and the constraint
table as the label, which is what makes the confusion matrix meaningful.

Four things it establishes, all load-bearing for fig5 panel C:

  1. `pass_qc` is TRUE on every row of the published table. The column is a constant, so
     any code that filters on it is filtering on nothing, and a QC failure never appears
     as a row with pass_qc = False -- it appears as no row at all. That is why panel C's
     middle stratum ("coding") contains no QC failures despite once naming them.

  2. FORWARD: every scored window really does satisfy all three conditions when they are
     re-evaluated from the raw inputs. The flag is not merely self-consistent, it is
     what the paper says it is, and there are no windows admitted in spite of it.

  3. REVERSE: the absent windows are not uncovered. Every one has its QC inputs on file,
     and the dominant failure is the PASS-fraction rule, not the coverage band. "Failed
     variant-call QC" is the accurate name for that stratum; "no coverage" is not.

  4. The residual: a small number of absent windows pass all three conditions and are
     unexplained, so the filter is not a complete account of who is in the table.

Outcome of the last run: preconditions/output/STATUS.md (transcript in the .log beside it).
"""
import argparse
import os
import sys

import duckdb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnocchi_bias import dnm_model as M  # noqa: E402
from gnocchi_bias.windows import REMOTE_FILES, download  # noqa: E402
from preconditions.report import Report  # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DEST_DIR = os.path.join(_REPO_ROOT, "published")

PASS_FILE = "misc/genome_1kb_gnomad_v31_pass.txt"          # element_id, fraction PASS
COVERAGE_FILE = "misc/genome_1kb_gnomad_v31_coverage.txt"  # element_id, mean coverage

# The filter, transcribed from run_nc_constraint_gnomad_v31_main.py:296. Comparisons are
# inclusive on both sides exactly as pandas' `>=` and `.between(25, 35)` are; 1,723 scored
# windows sit at pass_frac = 0.8 precisely, so `>` instead of `>=` would be a visible
# error rather than a matter of taste.
QC_PREDICATE = "pass_frac >= 0.8 AND cov BETWEEN 25 AND 35 AND possible >= 1000"

# The paper's own counts, quoted so a mismatch is visible rather than inferred:
# "This resulted in 1,984,900 autosomal windows (77.5% of initial) ..., of which 141,341
# overlapped with coding regions and 1,843,559 were exclusively non-coding."
PAPER_WINDOWS, PAPER_CODING, PAPER_NONCODING = 1_984_900, 141_341, 1_843_559

# Site -> containing 1 kb tile. Duplicated from fig5/data.py rather than imported: that
# module pulls in matplotlib-adjacent siblings and a config file, and preconditions/ must
# stay runnable on its own.
ELEMENT_ID_FROM_LOCUS = (
    "split_part(locus,':',1) || '-' || "
    "CAST(((CAST(split_part(locus,':',2) AS BIGINT)-1)//1000)*1000 AS VARCHAR) || '-' || "
    "CAST(((CAST(split_part(locus,':',2) AS BIGINT)-1)//1000)*1000+1000 AS VARCHAR)")


def one(con: duckdb.DuckDBPyConnection, query: str) -> tuple:
    row = con.execute(query).fetchone()
    assert row is not None, "aggregate query returned no row"
    return row


def universe_sql(annot: str, step1: str, passf: str, covf: str) -> str:
    """
    One row per autosomal 1 kb window, carrying whether it was scored and whether the
    published filter says it should have been.

    `possible` comes from the step-1 table rather than the constraint table because it
    must exist for the dropped windows too, which have no constraint row at all. The two
    agree exactly on the windows they share -- verify_expected_r1 checks that on all
    1,984,900 of them -- so nothing depends on which is used for the scored ones.
    """
    return f"""
        WITH an AS (SELECT element_id
                    FROM read_csv_auto('{annot}', delim='\t', header=True)),
        s1 AS (SELECT element_id, possible
               FROM read_csv_auto('{step1}', delim='\t', header=True)),
        pa AS (SELECT column0 AS element_id, column1 AS pass_frac
               FROM read_csv_auto('{passf}', delim='\t', header=False)),
        co AS (SELECT column0 AS element_id, column1 AS cov
               FROM read_csv_auto('{covf}', delim='\t', header=False))
        SELECT s1.element_id AS element_id, s1.possible AS possible,
               pa.pass_frac AS pass_frac, co.cov AS cov,
               (an.element_id IS NOT NULL) AS scored,
               ({QC_PREDICATE}) AS qc_ok
        FROM s1 LEFT JOIN an ON s1.element_id = an.element_id
                LEFT JOIN pa ON s1.element_id = pa.element_id
                LEFT JOIN co ON s1.element_id = co.element_id
        WHERE s1.element_id NOT LIKE 'chrX-%' AND s1.element_id NOT LIKE 'chrY-%'
    """


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-dest_dir", default=DEFAULT_DEST_DIR)
    ap.add_argument("-memory_limit", default="10GB")
    args = ap.parse_args()

    with Report("verify_qc_filter") as rep:
        annot = download(REMOTE_FILES["annot"], args.dest_dir)
        step1 = download(REMOTE_FILES["step1_expected"], args.dest_dir)
        passf = download(PASS_FILE, args.dest_dir)
        covf = download(COVERAGE_FILE, args.dest_dir)
        dnm0 = download(M.TRAINING_FILES["dnm0_sites"], args.dest_dir)

        con = duckdb.connect()
        con.execute(f"SET memory_limit='{args.memory_limit}'")
        universe = universe_sql(annot, step1, passf, covf)

        # ---------------------------------------------------------- the flag is constant
        n_rows, n_false, n_null, n_coding = one(con, f"""
            SELECT COUNT(*), SUM(CASE WHEN NOT pass_qc THEN 1 ELSE 0 END),
                   SUM(CASE WHEN pass_qc IS NULL THEN 1 ELSE 0 END),
                   SUM(CASE WHEN coding_prop > 0 THEN 1 ELSE 0 END)
            FROM read_csv_auto('{annot}', delim='\t', header=True)
        """)
        print(f"\npublished constraint table: {n_rows:,} rows, "
              f"pass_qc false on {int(n_false):,}, null on {int(n_null):,}")
        rep.claim(int(n_false) == 0 and int(n_null) == 0 and n_rows == PAPER_WINDOWS,
                  f"pass_qc is True on all {n_rows:,} rows of the published constraint "
                  f"table -- the column is a constant, so a QC failure appears as an "
                  f"ABSENT row, never as pass_qc = False")
        print(f"  coding-overlapping {int(n_coding):,}, "
              f"exclusively noncoding {n_rows - int(n_coding):,}")
        rep.claim(int(n_coding) == PAPER_CODING and n_rows - int(n_coding) == PAPER_NONCODING,
                  f"the window set is the paper's: {n_rows:,} autosomal windows, "
                  f"{int(n_coding):,} coding-overlapping and "
                  f"{n_rows - int(n_coding):,} exclusively noncoding "
                  f"(Methods: {PAPER_WINDOWS:,} / {PAPER_CODING:,} / {PAPER_NONCODING:,})")

        # ------------------------------- the filter re-evaluated, against membership
        n_auto, n_scored, no_inputs, tp, fp, tn, fn = (int(v) for v in one(con, f"""
            SELECT COUNT(*),
                   SUM(CASE WHEN scored THEN 1 ELSE 0 END),
                   SUM(CASE WHEN pass_frac IS NULL OR cov IS NULL
                             OR possible IS NULL THEN 1 ELSE 0 END),
                   SUM(CASE WHEN scored AND qc_ok THEN 1 ELSE 0 END),
                   SUM(CASE WHEN scored AND NOT qc_ok THEN 1 ELSE 0 END),
                   SUM(CASE WHEN NOT scored AND NOT qc_ok THEN 1 ELSE 0 END),
                   SUM(CASE WHEN NOT scored AND qc_ok THEN 1 ELSE 0 END)
            FROM ({universe})
        """))
        print(f"\nautosomal windows in the step-1 table: {n_auto:,}\n"
              f"  scored (in the constraint table): {n_scored:,}\n"
              f"  absent from it:                   {n_auto - n_scored:,}")
        rep.claim(n_scored == PAPER_WINDOWS,
                  f"the two tables partition cleanly: {n_auto:,} autosomal step-1 windows "
                  f"= {PAPER_WINDOWS:,} scored + {n_auto - n_scored:,} absent")
        rep.claim(no_inputs == 0,
                  f"every one of the {n_auto:,} windows has all three QC inputs on file "
                  f"(0 missing a pass fraction, a coverage or a possible count), so the "
                  f"filter can be re-evaluated everywhere rather than assumed")
        print(f"\nfilter re-evaluated from the raw inputs, against membership:\n"
              f"  scored and satisfies it:     {tp:,}\n"
              f"  scored and VIOLATES it:      {fp:,}\n"
              f"  absent and fails it:         {tn:,}\n"
              f"  absent but satisfies it:     {fn:,}   [unexplained]")
        rep.claim(fp == 0,
                  f"FORWARD DIRECTION: all {tp:,} scored windows satisfy the published "
                  f"filter when it is re-evaluated from the raw pass/coverage/possible "
                  f"inputs -- {fp:,} violate it, so nothing was admitted in spite of the "
                  f"rule the Methods state")

        # The margins say the same thing continuously rather than as a boolean, and pin
        # down that the comparisons are inclusive.
        min_pass, min_cov, max_cov, min_poss, at_edge = one(con, f"""
            SELECT MIN(pass_frac), MIN(cov), MAX(cov), MIN(possible),
                   SUM(CASE WHEN pass_frac = 0.8 THEN 1 ELSE 0 END)
            FROM ({universe}) WHERE scored
        """)
        print(f"\nmargins over the scored windows:\n"
              f"  pass fraction  min {min_pass:.4f}   (threshold 0.8; "
              f"{int(at_edge):,} sit exactly on it)\n"
              f"  mean coverage  {min_cov:.3f} - {max_cov:.3f}   (band 25-35)\n"
              f"  possible       min {min_poss:,}   (threshold 1,000)")
        rep.claim(min_pass >= 0.8 and 25 <= min_cov and max_cov <= 35 and min_poss >= 1000,
                  f"the margins agree: over the scored set the pass fraction bottoms out "
                  f"at {min_pass:.4f}, coverage spans {min_cov:.3f}-{max_cov:.3f} and "
                  f"`possible` bottoms out at {min_poss:,} -- and the {int(at_edge):,} "
                  f"windows at exactly 0.8 confirm the comparison is >=, not >")

        # ------------------------------------------------- what absence means, by reason
        f_pass, f_cov, f_poss = (int(v) for v in one(con, f"""
            SELECT SUM(CASE WHEN pass_frac < 0.8 THEN 1 ELSE 0 END),
                   SUM(CASE WHEN cov < 25 OR cov > 35 THEN 1 ELSE 0 END),
                   SUM(CASE WHEN possible < 1000 THEN 1 ELSE 0 END)
            FROM ({universe}) WHERE NOT scored
        """))
        n_miss = n_auto - n_scored
        print(f"\nreasons the {n_miss:,} absent windows fail (categories overlap):\n"
              f"  fail >= 80% PASS:        {f_pass:,} ({f_pass / n_miss:.1%})\n"
              f"  fail >= 1000 possible:   {f_poss:,} ({f_poss / n_miss:.1%})\n"
              f"  fail 25-35x coverage:    {f_cov:,} ({f_cov / n_miss:.1%})")
        rep.claim((n_miss - fn) / n_miss >= 0.95,
                  f"REVERSE DIRECTION: {(n_miss - fn) / n_miss:.1%} of absent windows fail "
                  f"at least one of the paper's three conditions (threshold 95%); the "
                  f"residual {fn:,} pass all three and are unexplained, so the filter is "
                  f"not a complete account of membership")
        rep.claim(f_pass > 10 * f_cov,
                  f"absence is the PASS-fraction rule, not missing coverage: {f_pass:,} "
                  f"windows fail >= 80% PASS against {f_cov:,} failing the 25-35x band "
                  f"-- so 'no gnomAD coverage' is the wrong name for this stratum")

        # ------------------------------- weighted by the training sites panel C counts
        sites = f"""
            SELECT u.element_id, u.possible, u.pass_frac, u.cov
            FROM (SELECT context, {ELEMENT_ID_FROM_LOCUS} AS element_id
                  FROM read_csv_auto('{dnm0}', delim='\t', header=True)
                  WHERE locus NOT LIKE 'chrX:%'
                    AND context NOT IN ({', '.join(repr(c) for c in M.CPG_CONTEXTS)})) d0
            JOIN ({universe}) u ON d0.element_id = u.element_id
            WHERE NOT u.scored
        """
        s_n, s_pass, s_cov, s_poss = (int(v) for v in one(con, f"""
            SELECT COUNT(*),
                   SUM(CASE WHEN pass_frac < 0.8 THEN 1 ELSE 0 END),
                   SUM(CASE WHEN cov < 25 OR cov > 35 THEN 1 ELSE 0 END),
                   SUM(CASE WHEN possible < 1000 THEN 1 ELSE 0 END)
            FROM ({sites})
        """))
        print(f"\nnon-CpG background training sites in absent windows: {s_n:,}\n"
              f"  fail >= 80% PASS:        {s_pass:,} ({s_pass / s_n:.1%})\n"
              f"  fail >= 1000 possible:   {s_poss:,} ({s_poss / s_n:.1%})\n"
              f"  fail 25-35x coverage:    {s_cov:,} ({s_cov / s_n:.1%})")
        rep.claim(s_pass / s_n > 0.5,
                  f"the same holds site-weighted, which is what panel C's third band "
                  f"counts: of its {s_n:,} sites, {s_pass / s_n:.1%} are in windows "
                  f"failing the PASS rule against {s_cov / s_n:.1%} failing coverage")


if __name__ == "__main__":
    main()
