"""
Genome-wide decomposition of Gnocchi's regional adjustment factor by CpG status.

The quantity plotted here is the one Gnocchi literally applies to every window:

    r_eff(w) = E2(w) / E1(w) = sum_c E1_c(w) r_c(w) / sum_c E1_c(w)

i.e. the step-1-expected-weighted mean of the pipeline's own per-context r. E1
is the context+methylation-only expected count (r == 1); E2 is the r-adjusted
expected count that produces the published Gnocchi z. See CLAUDE.md, "Methylation,
and why the training-set calibration panel measures the wrong thing", for why this
is the right panel-B quantity: it is the same population, the same unit, and the
same x-axis as panel A, and it quantitatively predicts panel A's rank shift.

Splitting the sum at CpG vs non-CpG contexts gives an exact identity,

    r_eff = Pi * r_CpG + (1 - Pi) * r_non,     Pi = E1_CpG / E1

which is what shows that the GC trend is wholly non-CpG: r_CpG is flat at ~1.00 at
every GC, so holding the non-CpG term at 1 (the counterfactual curve) removes the
entire trend even though CpG contexts carry up to 45% of the expected-count weight.

HOW THE PER-CONTEXT r IS OBTAINED. The published pipeline writes its per-context r
table to a local output_dir, not to the public bucket (confirmed: no such object
exists under any bucket prefix), so this module uses the reimplemented full-scale
refit's rr_by_context.dnm_refit_full.txt. That refit reproduces the published
expected column at Pearson r = 1.0, median relative difference 4e-6 -- and the
reproduction is re-checked here per GC bin, since the published r_eff for the "all"
curve is directly computable as expected_step2 / expected_step1 without any refit.
Always read that printed check before trusting the CpG/non-CpG split.

COST. Only the four CpG contexts are joined between the 3.3 GB per-context expected
file and the 4.0 GB rr file; the non-CpG side is obtained by subtraction from
per-window totals that already exist as small files. That turns an 85M x 85M row
join into a ~10M x 10M one.
"""

import os

import duckdb
import numpy as np
import polars as pl

from gnocchi_bias.dnm_model import CPG_CONTEXTS

# Per-window totals that make the subtraction trick work: E1 from the published
# step-1 export, E2 from the refit's own genome-wide apply (same per-context
# expected file, so the two are exactly consistent).
DEFAULT_STEP1_EXPECTED = "tmp/expected_counts_by_context_methyl_genome_1kb.txt"
DEFAULT_PERCONTEXT_EXPECTED = "tmp/expected_counts_per_context_methyl_genome_1kb.txt"
DEFAULT_REFIT_EXPECTED = "refits/expected_counts_by_context_methyl_genome_1kb.full.txt"
DEFAULT_RR_BY_CONTEXT = "refits/rr_by_context.full.txt"


# ------------------------------------------------------------- shared GC bins

def gc_edges(gc: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Fixed-width bin edges spanning the observed GC range, matching
    windows.bin_by_gc's own "fixed" branch exactly (linspace over min..max, with
    the top edge nudged so the maximum value lands inside the last bin).

    Returned explicitly, rather than recomputed inside each consumer, because
    three different populations get binned on this axis -- genome-wide windows,
    per-(context, bin) genome-wide expected counts aggregated in duckdb, and DNM
    training sites -- and they are only comparable if the edges are identical.
    """
    edges = np.linspace(float(np.min(gc)), float(np.max(gc)), n_bins + 1)
    edges = np.unique(edges).astype(float)
    edges[-1] += 1e-9
    return edges


def assign_bin(gc: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin index in [0, len(edges)-2] for each GC value, clipped at both ends."""
    idx = np.digitize(np.asarray(gc, dtype=float), edges[1:-1], right=False)
    return np.clip(idx, 0, len(edges) - 2)


def sql_bin_expr(gc_expr: str, edges: np.ndarray) -> str:
    """
    duckdb equivalent of assign_bin, for grouping inside a query rather than
    materializing 79M rows. Edges are uniform by construction (gc_edges), so the
    bin is a clipped floor-divide -- no CASE ladder needed.
    """
    lo, hi, n = float(edges[0]), float(edges[-1]), len(edges) - 1
    width = (hi - lo) / n
    return (f"LEAST(GREATEST(CAST(FLOOR(({gc_expr} - {lo!r}) / {width!r}) AS INTEGER), 0), {n - 1})")


# ------------------------------------------------- per-window r decomposition

def load_r_eff_components(percontext_expected: str = DEFAULT_PERCONTEXT_EXPECTED,
                          rr_by_context: str = DEFAULT_RR_BY_CONTEXT,
                          step1_expected: str = DEFAULT_STEP1_EXPECTED,
                          refit_expected: str = DEFAULT_REFIT_EXPECTED,
                          cache_path: str | None = None,
                          memory_limit: str = "8GB") -> pl.DataFrame:
    """
    Per-window expected-count components, one row per element_id:

        e1, e2          totals over all 32 contexts (step 1 / refit step 2)
        e1_cpg, e2_cpg  the same sums restricted to ACG/CCG/GCG/TCG

    Non-CpG components are then just e1 - e1_cpg and e2 - e2_cpg, which is why
    only the CpG slice of the two multi-GB files ever has to be joined.

    Windows with no CpG-context row at all (possible at 1kb resolution) get
    e1_cpg = e2_cpg = 0, so their r_CpG is undefined and r_non == r_eff; bin_r_eff
    handles that by aggregating sums rather than averaging per-window ratios.
    """
    if cache_path and os.path.exists(cache_path):
        print(f"r_eff components: reusing cache {cache_path}")
        return pl.read_parquet(cache_path)

    ctx_list = ", ".join(f"'{c}'" for c in CPG_CONTEXTS)
    query = f"""
        WITH cpg AS (
            SELECT e.element_id AS element_id,
                   SUM(e.expected) AS e1_cpg,
                   SUM(e.expected * COALESCE(r.rr, 1.0)) AS e2_cpg
            FROM (SELECT element_id, context, expected
                  FROM read_csv_auto('{percontext_expected}', delim='\t', header=True)
                  WHERE context IN ({ctx_list})) e
            LEFT JOIN (SELECT element_id, context, rr
                       FROM read_csv_auto('{rr_by_context}', delim='\t', header=True)
                       WHERE context IN ({ctx_list})) r
              ON e.element_id = r.element_id AND e.context = r.context
            GROUP BY e.element_id
        )
        SELECT t1.element_id            AS element_id,
               t1.expected              AS e1,
               t2.expected              AS e2,
               COALESCE(cpg.e1_cpg, 0.0) AS e1_cpg,
               COALESCE(cpg.e2_cpg, 0.0) AS e2_cpg
        FROM read_csv_auto('{step1_expected}', delim='\t', header=True) t1
        INNER JOIN read_csv_auto('{refit_expected}', delim='\t', header=True) t2
          ON t1.element_id = t2.element_id
        LEFT JOIN cpg ON t1.element_id = cpg.element_id
    """
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{memory_limit}'")
    df = con.execute(query).pl()
    print(f"r_eff components: {df.height:,} windows "
          f"({(df['e1_cpg'] > 0).sum():,} with at least one CpG-context site)")

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.write_parquet(cache_path)
        print(f"r_eff components: cached to {cache_path}")
    return df


def attach_components(df_win: pl.DataFrame, comp: pl.DataFrame) -> pl.DataFrame:
    """
    Inner-join the components onto an already-filtered window table (from
    windows.build_window_table, which carries GC_content plus the published
    expected_step1/expected_step2), and add the per-window ratios.

    The published r_eff column is what validates the refit: e1 is the same
    published step-1 total in both, so r_eff_published vs r_eff is a direct
    published-vs-refit comparison of the adjustment itself.
    """
    df = df_win.join(comp, on="element_id", how="inner")
    return df.with_columns([
        (pl.col("e1") - pl.col("e1_cpg")).alias("e1_non"),
        (pl.col("e2") - pl.col("e2_cpg")).alias("e2_non"),
        (pl.col("e2") / pl.col("e1")).alias("r_eff"),
        (pl.col("expected_step2") / pl.col("expected_step1")).alias("r_eff_published"),
    ])


def bin_r_eff(df: pl.DataFrame, edges: np.ndarray, gc_col: str = "GC_content") -> pl.DataFrame:
    """
    Aggregate to GC bins as ratios of summed expected counts, not as means of
    per-window ratios.

    WHY WEIGHTED. Expected counts add: the total step-2 expectation in a GC bin is
    sum(E2) and the step-1 expectation is sum(E1), so sum(E2)/sum(E1) is the
    adjustment the bin actually receives. It also keeps the decomposition exact --
    R_eff = Pi*R_CpG + (1-Pi)*R_non holds bin by bin, which an average of
    per-window ratios would only satisfy approximately. (CLAUDE.md's earlier
    ad-hoc table used unweighted per-window means, so small differences from the
    numbers quoted there are expected.)

    Columns out: gc_bin, n, gc_mid, r_eff, r_cpg, r_non, pi_cpg, r_counterfactual,
    r_eff_published, se_r_eff.
    """
    idx = assign_bin(df[gc_col].to_numpy(), edges)
    df = df.with_columns(pl.Series("gc_bin", idx))

    binned = df.group_by("gc_bin").agg([
        pl.len().alias("n"),
        pl.col(gc_col).mean().alias("gc_mid"),
        pl.col("e1").sum().alias("s_e1"),
        pl.col("e2").sum().alias("s_e2"),
        pl.col("e1_cpg").sum().alias("s_e1_cpg"),
        pl.col("e2_cpg").sum().alias("s_e2_cpg"),
        pl.col("e1_non").sum().alias("s_e1_non"),
        pl.col("e2_non").sum().alias("s_e2_non"),
        pl.col("expected_step1").sum().alias("s_pub1"),
        pl.col("expected_step2").sum().alias("s_pub2"),
        (pl.col("r_eff").std() / pl.len().sqrt()).alias("se_r_eff"),
    ]).sort("gc_mid")

    return binned.with_columns([
        (pl.col("s_e2") / pl.col("s_e1")).alias("r_eff"),
        (pl.col("s_e2_cpg") / pl.col("s_e1_cpg")).alias("r_cpg"),
        (pl.col("s_e2_non") / pl.col("s_e1_non")).alias("r_non"),
        (pl.col("s_e1_cpg") / pl.col("s_e1")).alias("pi_cpg"),
        ((pl.col("s_e2_cpg") + pl.col("s_e1_non")) / pl.col("s_e1")).alias("r_counterfactual"),
        (pl.col("s_pub2") / pl.col("s_pub1")).alias("r_eff_published"),
    ]).drop(["s_e1", "s_e2", "s_e1_cpg", "s_e2_cpg", "s_e1_non", "s_e2_non", "s_pub1", "s_pub2"])


def report_refit_validation(binned: pl.DataFrame) -> float:
    """
    Print the per-GC-bin published-vs-refit agreement in r_eff and return the max
    absolute difference. This is the check that licenses using the refit's
    per-context r for the CpG/non-CpG split, since the published pipeline never
    exported its own per-context r.
    """
    diff = (binned["r_eff"] - binned["r_eff_published"]).abs()
    worst = float(diff.max())
    print(f"refit validation: max |r_eff(refit) - r_eff(published)| across "
          f"{binned.height} GC bins = {worst:.2e} "
          f"(median {float(diff.median()):.2e})")
    return worst
