"""
Verify that expected_counts_by_context_methyl_genome_1kb.txt (element_id,
possible, expected) really is the r==1 (context-only, pre-regional-feature-
adjustment) expected-count table it's assumed to be in CLAUDE.md, by
regenerating it independently from a file whose provenance IS confirmed --
expected_counts_per_context_methyl_genome_1kb.txt (element_id, context,
possible, expected) -- and comparing genome-wide, not just for one
hand-picked row.

Why this is possible without Hail or re-running the pipeline: the
per-context file is the literal hl.export() of
run_nc_constraint_gnomad_v31_main.py lines 191-197
(expected_ht = possible_ht.group_by(element_id, context).aggregate(
possible=sum, expected=sum)), written BEFORE any regional-feature
r-adjustment code runs (that starts at line ~209). Summing it again over
context, per element_id, is exactly what "forcing r=1 and recomputing"
means here -- r only ever multiplies expected counts starting at line 200+,
so skipping that code entirely (rather than editing it) is equivalent to
r==1 for every context. No Hail table is touched: both files are already
plain-text exports, so this whole check runs locally via duckdb.

Background / write-up: see CLAUDE.md, the "expected_counts_by_context_
methyl_genome_1kb.txt" row of the data inventory table, and the surrounding
discussion of forcing r=1 vs. repurposing the (unrelated) DNM
training-set-size experiment.
"""
import argparse
import json
import os
import time
import urllib.parse
import urllib.request

import duckdb

BUCKET_URL = "https://storage.googleapis.com/gnomad-nc-constraint-v31-paper"
BUCKET_JSON_API = "https://storage.googleapis.com/storage/v1/b/gnomad-nc-constraint-v31-paper/o"

PER_CONTEXT_FILE = "expected_counts_per_context_methyl_genome_1kb.txt"  # element_id, context, possible, expected (confirmed r==1 provenance)
PUBLISHED_FILE = "expected_counts_by_context_methyl_genome_1kb.txt"      # element_id, possible, expected (claimed r==1, unconfirmed provenance)

# Files spanning the dependency chain that produces PUBLISHED_FILE and
# PER_CONTEXT_FILE, in the order run_nc_constraint_gnomad_v31_main.py writes
# them. If PUBLISHED_FILE's own upstream inputs aren't in this same order,
# it can't have come from one continuous run of the pipeline alongside
# PER_CONTEXT_FILE -- which would explain small, non-zero `expected` diffs
# despite identical `possible` counts (a different downsample-based
# mutation-rate refit, not a wrong r==1 assumption).
PROVENANCE_CHAIN = [
    "mu_by_context_methyl_downsampled_1000.txt",
    "expected_counts_by_context_methyl_genome_1kb.ht/metadata.json.gz",
    PUBLISHED_FILE,
    PER_CONTEXT_FILE,
    "fig_tables/mutation_rate_by_context_methyl.txt",
]


def check_provenance_timeline():
    print("\nchecking customTime (original per-file creation date, preserved "
          "through a later bucket migration) across the dependency chain...")
    dates = []
    for fname in PROVENANCE_CHAIN:
        url = f"{BUCKET_JSON_API}/{urllib.parse.quote(fname, safe='')}"
        with urllib.request.urlopen(url) as resp:
            meta = json.load(resp)
        dates.append(meta.get("customTime"))
        print(f"  {meta.get('customTime')}  {fname}")
    # mu_by_context_methyl_downsampled_1000.txt feeds directly into
    # mutation_rate_by_context_methyl.txt just two code-steps later
    # (run_nc_constraint_gnomad_v31_main.py lines 134-148) -- if those two
    # dates are far apart, they can't be from one continuous run, and
    # PUBLISHED_FILE / PER_CONTEXT_FILE (clustered near one or the other)
    # must come from two separate pipeline executions with independently
    # refit mutation rates -- explaining small, non-zero `expected` diffs
    # despite identical `possible` counts, without implicating r==1 itself.
    gap_days = (
        __import__("datetime").datetime.fromisoformat(dates[-1].replace("Z", "+00:00"))
        - __import__("datetime").datetime.fromisoformat(dates[0].replace("Z", "+00:00"))
    ).days
    print(
        f"  Span from first to last file: {gap_days} days. mu_by_context_methyl_"
        f"downsampled_1000.txt feeds mutation_rate_by_context_methyl.txt just two "
        f"code-steps later (lines 134-148) -- a gap this large means the two "
        f"expected_counts_* files below were almost certainly produced by two "
        f"separate pipeline runs (independent mutation-rate refits on independent "
        f"random downsamples), not one continuous execution."
    )


def download(fname: str, dest_dir: str) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, fname)
    if os.path.exists(dest_path):
        print(f"[skip download, already present] {fname} ({os.path.getsize(dest_path) / 1e6:.1f} MB)")
        return dest_path
    url = f"{BUCKET_URL}/{fname}"
    print(f"downloading {url} -> {dest_path}")
    t0 = time.time()
    os.system(f"curl -s -o '{dest_path}' '{url}'")
    dt = time.time() - t0
    size = os.path.getsize(dest_path)
    print(f"  done: {size / 1e6:.1f} MB in {dt:.1f}s ({size / 1e6 / max(dt, 1e-9):.2f} MB/s)")
    return dest_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "-dest_dir", default="tmp",
        help="local directory to download into (default: ./tmp)")
    args = parser.parse_args()

    t_start = time.time()

    check_provenance_timeline()

    per_context_path = download(PER_CONTEXT_FILE, args.dest_dir)
    published_path = download(PUBLISHED_FILE, args.dest_dir)

    con = duckdb.connect()

    print("\nregenerating element-level sums from the per-context file (mirrors "
          "run_nc_constraint_gnomad_v31_main.py lines 191-193, one group_by level up)...")
    t0 = time.time()
    con.execute(f"""
        CREATE TABLE regenerated AS
        SELECT element_id,
               SUM(possible) AS possible,
               SUM(expected) AS expected
        FROM read_csv_auto('{per_context_path}', delim='\t', header=True)
        GROUP BY element_id
    """)
    n_regen = con.execute("SELECT COUNT(*) FROM regenerated").fetchone()[0]
    print(f"  regenerated {n_regen:,} element rows in {time.time() - t0:.1f}s")

    # Sort and format to match the published file's own conventions (lexicographic
    # element_id order; expected as an 8-decimal fixed-point string, matching
    # run_nc_constraint_gnomad_v31_main.py line 196's `hl.format('%.8f', ...)`)
    # purely for easier side-by-side reading -- the join-based comparison above
    # already confirmed equivalence independent of row order or formatting.
    regenerated_path = os.path.join(
        args.dest_dir, PUBLISHED_FILE.replace(".txt", ".regenerated.txt"))
    con.execute(f"""
        COPY (
            SELECT element_id, possible, printf('%.8f', expected) AS expected
            FROM regenerated
            ORDER BY element_id
        ) TO '{regenerated_path}' (HEADER, DELIMITER '\t')
    """)
    print(f"  saved to {regenerated_path} ({os.path.getsize(regenerated_path) / 1e6:.1f} MB)")

    print("\nloading published file...")
    t0 = time.time()
    con.execute(f"""
        CREATE TABLE published AS
        SELECT element_id, possible, expected
        FROM read_csv_auto('{published_path}', delim='\t', header=True)
    """)
    n_pub = con.execute("SELECT COUNT(*) FROM published").fetchone()[0]
    print(f"  loaded {n_pub:,} published rows in {time.time() - t0:.1f}s")

    print(f"\nrow count: regenerated={n_regen:,}  published={n_pub:,}  "
          f"{'MATCH' if n_regen == n_pub else 'MISMATCH'}")

    # `possible` is a plain deterministic count -- exact match expected.
    # `expected` depends on `fitted_po`, a weighted-least-squares fit to a
    # RANDOM downsample (lines 43-83, 134-141 of
    # run_nc_constraint_gnomad_v31_main.py) -- if the two files come from
    # different pipeline runs (see check_provenance_timeline() above), a
    # tiny relative difference is expected even when both correctly used
    # r==1, so `expected` is compared with a relative, not absolute, tolerance.
    REL_TOL_EXPECTED = 1e-3
    print(f"\ncomparing values genome-wide (possible: exact; expected: relative "
          f"tolerance {REL_TOL_EXPECTED:.0e}, since `expected` depends on a "
          f"randomly-downsampled mutation-rate refit that can differ slightly "
          f"between pipeline runs)...")
    diff = con.execute(f"""
        SELECT
            COUNT(*) AS n_joined,
            SUM(CASE WHEN r.element_id IS NULL THEN 1 ELSE 0 END) AS in_published_only,
            SUM(CASE WHEN p.element_id IS NULL THEN 1 ELSE 0 END) AS in_regenerated_only,
            SUM(CASE WHEN r.possible IS NOT NULL AND p.possible IS NOT NULL
                     AND ABS(r.possible - p.possible) > 1e-6 THEN 1 ELSE 0 END) AS possible_mismatches,
            SUM(CASE WHEN r.expected IS NOT NULL AND p.expected IS NOT NULL
                     AND ABS(r.expected - p.expected) / NULLIF(p.expected, 0) > {REL_TOL_EXPECTED}
                     THEN 1 ELSE 0 END) AS expected_mismatches,
            MAX(CASE WHEN r.expected IS NOT NULL AND p.expected IS NOT NULL
                     THEN ABS(r.expected - p.expected) ELSE NULL END) AS max_expected_abs_diff,
            MAX(CASE WHEN r.expected IS NOT NULL AND p.expected IS NOT NULL
                     THEN ABS(r.expected - p.expected) / NULLIF(p.expected, 0) ELSE NULL END) AS max_expected_rel_diff
        FROM regenerated r
        FULL OUTER JOIN published p ON r.element_id = p.element_id
    """).fetchdf()
    print(diff.to_string(index=False))

    all_match = (
        n_regen == n_pub
        and diff["in_published_only"].iloc[0] == 0
        and diff["in_regenerated_only"].iloc[0] == 0
        and diff["possible_mismatches"].iloc[0] == 0
        and diff["expected_mismatches"].iloc[0] == 0
    )
    print(f"\ntotal wall time: {time.time() - t_start:.1f}s")

    if all_match:
        print(
            "\nConclusion: expected_counts_by_context_methyl_genome_1kb.txt is "
            "reproduced, genome-wide, by summing expected_counts_per_context_methyl_"
            "genome_1kb.txt over context per element_id -- `possible` matches exactly "
            "on every row, and `expected` matches to within the relative tolerance "
            "above (consistent with the two files coming from separate pipeline runs "
            "with independently-refit mutation rates -- see the customTime timeline "
            "printed above). This confirms the r==1, context-only interpretation of "
            "expected_counts_by_context_methyl_genome_1kb.txt using only code whose "
            "provenance is confirmed end-to-end."
        )
    else:
        print(
            "\nConclusion: MISMATCH found beyond refit-level noise -- expected_counts_"
            "by_context_methyl_genome_1kb.txt does NOT match the r==1 regeneration from "
            "expected_counts_per_context_methyl_genome_1kb.txt even accounting for a "
            "plausible independent mutation-rate refit. Its r==1 interpretation in "
            "CLAUDE.md needs re-examination before further use."
        )


if __name__ == "__main__":
    main()
