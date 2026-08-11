"""
Verify what the published constraint table's window set IS, and what being absent from
it means -- because fig5 panel C's third stratum is defined by that absence, and it was
labelled "no gnomAD coverage" on the strength of the name alone.

Chen et al. keep a window only if (Methods; run_nc_constraint_gnomad_v31_main.py:296)

    pass_qc = (>= 80% of observed variants PASS) & (mean coverage in 25-35x)
              & (>= 1000 possible variants)

and line 302 drops the `pass` and `coverage` columns before export, so the published
table cannot say WHICH condition a window failed. Both inputs are in the bucket, though,
as misc/genome_1kb_gnomad_v31_{pass,coverage}.txt, and `possible` is in the step-1
expected-count table -- so the filter can simply be re-evaluated on the windows that are
missing, which is what this does.

Two things it establishes, both load-bearing for panel C:

  1. `pass_qc` is TRUE on every row of the published table. The column is a constant, so
     any code that filters on it is filtering on nothing, and a QC failure never appears
     as a row with pass_qc = False -- it appears as no row at all. That is why panel C's
     middle stratum ("coding") contains no QC failures despite naming them.

  2. The absent windows are not uncovered. Every one has its QC inputs on file, and the
     dominant failure is the PASS-fraction rule, not the coverage band. "Failed
     variant-call QC" is the accurate name for that stratum; "no coverage" is not.

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

        # ------------------------------------------------- what absence from it means
        missing = f"""
            WITH an AS (SELECT element_id
                        FROM read_csv_auto('{annot}', delim='\t', header=True)),
            s1 AS (SELECT element_id, possible
                   FROM read_csv_auto('{step1}', delim='\t', header=True)),
            pa AS (SELECT column0 AS element_id, column1 AS pass_frac
                   FROM read_csv_auto('{passf}', delim='\t', header=False)),
            co AS (SELECT column0 AS element_id, column1 AS cov
                   FROM read_csv_auto('{covf}', delim='\t', header=False))
            SELECT s1.element_id, s1.possible, pa.pass_frac, co.cov
            FROM s1 LEFT JOIN an ON s1.element_id = an.element_id
                    LEFT JOIN pa ON s1.element_id = pa.element_id
                    LEFT JOIN co ON s1.element_id = co.element_id
            WHERE an.element_id IS NULL
              AND s1.element_id NOT LIKE 'chrX-%' AND s1.element_id NOT LIKE 'chrY-%'
        """
        n_auto, = one(con, f"""
            SELECT COUNT(*) FROM read_csv_auto('{step1}', delim='\t', header=True)
            WHERE element_id NOT LIKE 'chrX-%' AND element_id NOT LIKE 'chrY-%'
        """)
        n_miss, no_inputs, f_pass, f_cov, f_poss, ok = one(con, f"""
            SELECT COUNT(*),
                   SUM(CASE WHEN pass_frac IS NULL OR cov IS NULL THEN 1 ELSE 0 END),
                   SUM(CASE WHEN pass_frac < 0.8 THEN 1 ELSE 0 END),
                   SUM(CASE WHEN cov < 25 OR cov > 35 THEN 1 ELSE 0 END),
                   SUM(CASE WHEN possible < 1000 THEN 1 ELSE 0 END),
                   SUM(CASE WHEN pass_frac >= 0.8 AND cov BETWEEN 25 AND 35
                             AND possible >= 1000 THEN 1 ELSE 0 END)
            FROM ({missing})
        """)
        n_miss, no_inputs, f_pass, f_cov, f_poss, ok = (int(v) for v in
                                                        (n_miss, no_inputs, f_pass,
                                                         f_cov, f_poss, ok))
        print(f"\nautosomal windows in the step-1 table: {n_auto:,}\n"
              f"  in the constraint table:  {n_auto - n_miss:,}\n"
              f"  absent from it:           {n_miss:,}")
        rep.claim(n_auto - n_miss == PAPER_WINDOWS,
                  f"the two tables partition cleanly: {n_auto:,} autosomal step-1 windows "
                  f"= {PAPER_WINDOWS:,} scored + {n_miss:,} absent")
        rep.claim(no_inputs == 0,
                  f"all {n_miss:,} absent windows have their QC inputs on file "
                  f"(0 missing a pass fraction or a coverage), so the reason for each "
                  f"absence is adjudicable rather than assumed")
        print(f"  fail >= 80% PASS:        {f_pass:,} ({f_pass / n_miss:.1%})\n"
              f"  fail >= 1000 possible:   {f_poss:,} ({f_poss / n_miss:.1%})\n"
              f"  fail 25-35x coverage:    {f_cov:,} ({f_cov / n_miss:.1%})\n"
              f"  pass all three anyway:   {ok:,} ({ok / n_miss:.1%})   [unexplained]")
        rep.claim((n_miss - ok) / n_miss >= 0.95,
                  f"{(n_miss - ok) / n_miss:.1%} of absent windows fail at least one of "
                  f"the paper's three conditions (threshold 95%); the residual {ok:,} "
                  f"pass all three and are unexplained")
        rep.claim(f_pass > 10 * f_cov,
                  f"absence is the PASS-fraction rule, not missing coverage: {f_pass:,} "
                  f"windows fail >= 80% PASS against {f_cov:,} failing the 25-35x band "
                  f"-- so 'no gnomAD coverage' is the wrong name for this stratum")

        # ------------------------------- weighted by the training sites panel C counts
        sites = f"""
            WITH an AS (SELECT element_id
                        FROM read_csv_auto('{annot}', delim='\t', header=True)),
            s1 AS (SELECT element_id, possible
                   FROM read_csv_auto('{step1}', delim='\t', header=True)),
            pa AS (SELECT column0 AS element_id, column1 AS pass_frac
                   FROM read_csv_auto('{passf}', delim='\t', header=False)),
            co AS (SELECT column0 AS element_id, column1 AS cov
                   FROM read_csv_auto('{covf}', delim='\t', header=False)),
            d0 AS (SELECT context, {ELEMENT_ID_FROM_LOCUS} AS element_id
                   FROM read_csv_auto('{dnm0}', delim='\t', header=True)
                   WHERE locus NOT LIKE 'chrX:%')
            SELECT d0.element_id, s1.possible, pa.pass_frac, co.cov
            FROM d0 LEFT JOIN an ON d0.element_id = an.element_id
                    LEFT JOIN s1 ON d0.element_id = s1.element_id
                    LEFT JOIN pa ON d0.element_id = pa.element_id
                    LEFT JOIN co ON d0.element_id = co.element_id
            WHERE an.element_id IS NULL
              AND d0.context NOT IN ({', '.join(repr(c) for c in M.CPG_CONTEXTS)})
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
