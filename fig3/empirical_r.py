"""
Is the non-CpG regional adjustment r_non(w) actually RIGHT?

r_eff.py establishes that Gnocchi's GC bias is carried entirely by the non-CpG
contexts' adjustment, which climbs from 0.95 to ~2.1 across the GC range. That is a
statement about the model, not about truth. This module builds the empirical
counterpart of the same quantity from the de novo mutations the model was fit on,
so the two can be plotted on the same axes.

WHAT r IS SUPPOSED TO BE. r_c(w) is the factor by which the real mutation rate at
window w departs from the context+methylation-only prediction. So the empirical
target is a rate, directly measurable per trinucleotide context c and GC bin g:

    r_true_c(g)  proportional to  DNMs_c(g) / opportunities_c(g)

where opportunities_c(g) is the number of possible SNV sites of context c in the
analyzed windows of that bin (`possible` in the per-context export). This is the
construction of the 1 Mb ground-truth test in CLAUDE.md
(observed_counts_dnm_1M / expected_counts_by_context_methyl_dnm_1M), but computed per
context and binned on 1 kb GC, so it reaches the high-GC tail that 1 Mb averaging
compresses away. Numerator and denominator are restricted to the same analyzed window
set and DNM sites are assigned to windows by position, so the denominator counts the
territory the numerator was counted over.

(Using the step-1 expected count E1_c(g) in place of opportunities gives the same
curve to <0.1%: within one non-CpG context there is a single methylation level, so
E1 = opportunities x a per-context constant, and that constant cancels in the
per-context normalization below. Both are implemented; opportunities is the default
because it is what a rate actually needs.)

NORMALIZATION. r_c(w) = sigma(b0+b.z(w))/sigma(b0) is a ratio of the rate at w to the
rate at the average feature vector, so the empirical analogue is the same ratio: the
DNM rate in this GC bin over the DNM rate for that context overall. No free constant
is needed. Both sides are normalized per context to E1-weighted mean 1 over the
analyzed windows -- see combine_non_cpg for why per context is mandatory (D/E1 is not
on a common scale across contexts) and why the model must be normalized identically
(otherwise the shifting trinucleotide composition of high-GC windows leaks
between-context level differences into the curve as false GC dependence). Only GC
SHAPE is compared, which is the part that survives into Gnocchi's expected counts:
CLAUDE.md's methylation section shows a level error common to numerator and
denominator cancels in r and never reaches Gnocchi.

THE ASSUMPTION. Trio DNM ascertainment must not vary systematically with GC. If the
DECODE/PsychENCODE call sets are less sensitive in GC-rich sequence, the empirical
curve falls with GC for technical reasons. This is the same caveat the 1 Mb test
carries, it is not testable from the files in this bucket, and it must be stated
wherever this comparison is reported.

A SECOND, INDEPENDENT ESTIMATOR is provided for cross-checking: DNMs per dnm0
background control site, which replaces the model-derived denominator E1_c(g) with a
real sample of background sites. On matched population and binning it agrees with
the E1 denominator to 1-2% through GC 0.65, so the result does not rest on E1.

What DOES matter is that rates be measured on the same window population the E1
weights and the normalization come from -- see load_training_by_context_bin, which
is the unrestricted version and is mis-specified for exactly that reason.
"""

import duckdb
import numpy as np
import polars as pl

from gnocchi_bias.dnm_model import CPG_CONTEXTS
from r_eff import (DEFAULT_PERCONTEXT_EXPECTED, DEFAULT_RR_BY_CONTEXT, sql_bin_expr)

FEATURES_GENOME = "tmp/genomic_features13_genome_1kb.txt"
ANNOT_GENOME = "tmp/constraint_z_genome_1kb.annot.txt"

TRAIN_DNM1_SITES = "tmp/DNM_decode_psychencode_site_context.mutation_rate.txt"
TRAIN_DNM0_SITES = "tmp/context_prefiltered_nonmutated-dnm_sites10xdnm.mutation_rate.txt"
TRAIN_DNM1_FEATURES = "tmp/genomic_features13_dnm1_flnk_1k-1M.txt"
TRAIN_DNM0_FEATURES = "tmp/genomic_features13_dnm0_10x_flnk_1k-1M.txt"

# 1-based locus "chr1:137548" -> the 0-based 1kb tile id "chr1-137000-138000" that
# windows.py, the features file and the expected-count exports are all keyed by.
# The -1 is the 1-based to 0-based conversion; it only changes the assignment of
# sites at position = 1 (mod 1000), i.e. ~0.1% of DNMs.
ELEMENT_ID_FROM_LOCUS = (
    "split_part(locus,':',1) || '-' || "
    "CAST(((CAST(split_part(locus,':',2) AS BIGINT)-1)//1000)*1000 AS VARCHAR) || '-' || "
    "CAST(((CAST(split_part(locus,':',2) AS BIGINT)-1)//1000)*1000+1000 AS VARCHAR)")


def _connect(memory_limit: str) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute("SET enable_progress_bar=false")
    return con


def _analyzed_windows_cte(edges: np.ndarray, features: str, annot: str,
                          coding_prop_threshold: float) -> str:
    """
    SQL for the analyzed window set with its GC bin. Filters match
    windows.build_window_table's defaults exactly (pass_qc, coding_prop <= 0,
    autosome+PAR), so everything downstream describes the same windows panel A does.
    """
    return f"""
        SELECT ft.element_id AS element_id, {sql_bin_expr("ft.GC_content_1k / 100.0", edges)} AS gc_bin
        FROM (SELECT element_id, GC_content_1k
              FROM read_csv_auto('{features}', delim='\t', header=True)) ft
        INNER JOIN (SELECT element_id FROM read_csv_auto('{annot}', delim='\t', header=True)
                    WHERE pass_qc AND coding_prop <= {coding_prop_threshold}) an
          ON ft.element_id = an.element_id
        WHERE ft.element_id NOT LIKE 'chrX-%' AND ft.element_id NOT LIKE 'chrY-%'
    """


# ------------------------------------------------- genome side: E1_c(g), r_c(g)

def load_genome_by_context_bin(edges: np.ndarray,
                               percontext_expected: str = DEFAULT_PERCONTEXT_EXPECTED,
                               rr_by_context: str = DEFAULT_RR_BY_CONTEXT,
                               features: str = FEATURES_GENOME,
                               annot: str = ANNOT_GENOME,
                               coding_prop_threshold: float = 0.0,
                               memory_limit: str = "12GB") -> pl.DataFrame:
    """
    Per (context, GC bin) over the analyzed window set: summed step-1 expected count
    e1, summed r-adjusted expected count e2, and the model's adjustment
    r_model = e2/e1.

    Aggregating r_model over non-CpG contexts reproduces r_eff.py's per-window r_non
    curve bin for bin; the driver asserts that rather than assuming it.
    """
    query = f"""
        WITH win AS ({_analyzed_windows_cte(edges, features, annot, coding_prop_threshold)})
        SELECT e.context AS context,
               win.gc_bin AS gc_bin,
               SUM(e.possible) AS opportunities,
               SUM(e.expected) AS e1,
               SUM(e.expected * COALESCE(r.rr, 1.0)) AS e2
        FROM read_csv_auto('{percontext_expected}', delim='\t', header=True) e
        INNER JOIN win ON e.element_id = win.element_id
        LEFT JOIN read_csv_auto('{rr_by_context}', delim='\t', header=True) r
          ON e.element_id = r.element_id AND e.context = r.context
        GROUP BY e.context, win.gc_bin
    """
    df = _connect(memory_limit).execute(query).pl()
    df = df.with_columns((pl.col("e2") / pl.col("e1")).alias("r_model"))
    print(f"genome by (context, bin): {df.height} rows, "
          f"{df['context'].n_unique()} contexts, {df['gc_bin'].n_unique()} bins")
    return df


# --------------------------------- empirical estimator 1: DNM counts over E1

def load_dnm_counts_by_context_bin(edges: np.ndarray,
                                   dnm_sites: str = TRAIN_DNM1_SITES,
                                   features: str = FEATURES_GENOME,
                                   annot: str = ANNOT_GENOME,
                                   coding_prop_threshold: float = 0.0,
                                   memory_limit: str = "8GB") -> pl.DataFrame:
    """
    Observed de novo mutations per (context, GC bin), assigned to windows by
    position and restricted to the analyzed window set -- the numerator of the
    empirical adjustment, matched to load_genome_by_context_bin's E1 denominator.

    Binning by the containing tile's GC (not by the DNM site's own flanking-window
    GC feature) is what makes numerator and denominator describe the same territory.
    """
    query = f"""
        WITH win AS ({_analyzed_windows_cte(edges, features, annot, coding_prop_threshold)}),
        dnm AS (
            SELECT context, {ELEMENT_ID_FROM_LOCUS} AS element_id
            FROM read_csv_auto('{dnm_sites}', delim='\t', header=True)
        )
        SELECT dnm.context AS context, win.gc_bin AS gc_bin, COUNT(*) AS n_dnm
        FROM dnm INNER JOIN win ON dnm.element_id = win.element_id
        GROUP BY dnm.context, win.gc_bin
    """
    df = _connect(memory_limit).execute(query).pl()
    print(f"DNM counts by (context, bin): {df.height} rows, "
          f"{int(df['n_dnm'].sum()):,} DNMs inside the analyzed window set")
    return df


def empirical_from_dnm_counts(genome: pl.DataFrame, counts: pl.DataFrame,
                              denominator: str = "opportunities",
                              callable_fraction: pl.DataFrame | None = None) -> pl.DataFrame:
    """
    Per (context, GC bin) empirical DNM rate, r_raw = n_dnm / opportunities, with
    Poisson error on the DNM count. Unnormalized -- combine_non_cpg divides each
    context by its own mean. n_eff is the DNM count, which limits precision.

    `opportunities` is `possible`: the count of possible SNV sites of that context in
    those windows, which is the natural denominator for a rate. `denominator="e1"`
    uses the step-1 expected count instead and gives the SAME curve to <0.1%, because
    for a non-CpG context E1 = opportunities x (a per-context constant) -- there is
    only one methylation level, so fitted_po does not vary within the context -- and
    that constant cancels in the per-context normalization. Kept as a cross-check.

    callable_fraction: optional per-gc_bin table (columns gc_bin, callable_fraction)
    that divides the denominator, converting "gnomAD-callable opportunities" into
    "all positions". See callable_fraction_by_bin() for why this matters and for the
    resulting sensitivity range -- it is the largest known uncertainty in this
    comparison, not a rounding detail.
    """
    col = {"opportunities": "opportunities", "e1": "e1"}[denominator]
    df = (genome.select(["context", "gc_bin", col])
                .join(counts, on=["context", "gc_bin"], how="left")
                .with_columns(pl.col("n_dnm").fill_null(0).cast(pl.Float64))
                .with_columns(pl.col(col).cast(pl.Float64).alias("denom")))
    if callable_fraction is not None:
        df = (df.join(callable_fraction, on="gc_bin", how="left")
                .with_columns((pl.col("denom") / pl.col("callable_fraction")).alias("denom")))
    return df.with_columns([
        (pl.col("n_dnm") / pl.col("denom")).alias("r_raw"),
        (pl.col("n_dnm").sqrt() / pl.col("denom")).alias("se_raw"),
        pl.col("n_dnm").alias("n_eff"),
    ]).select(["context", "gc_bin", "r_raw", "se_raw", "n_eff"])


def callable_fraction_by_bin(edges: np.ndarray,
                             step1_expected: str = "tmp/expected_counts_by_context_methyl_genome_1kb.txt",
                             features: str = FEATURES_GENOME,
                             annot: str = ANNOT_GENOME,
                             coding_prop_threshold: float = 0.0,
                             memory_limit: str = "8GB") -> pl.DataFrame:
    """
    The fraction of each 1 kb window's 3,000 possible SNVs that survive gnomAD's
    coverage and black-region filtering, averaged per GC bin.

    WHY THIS MATTERS. The numerator counts DNMs falling anywhere in an analyzed
    window; the denominator counts only gnomAD-callable positions. If the callable
    fraction varied with GC, the two would be mismatched -- and it does, strongly:
    0.905 at GC 0.30 falling to 0.70-0.75 above GC 0.60, because short-read coverage
    drops in GC-rich sequence.

    Whether that is a bias depends on the DECODE/PsychENCODE trio call sets'
    callability, which is not in this bucket. If it tracks gnomAD's (both are Illumina
    WGS, so the GC dropout is similar in kind), `possible` is the matched denominator
    and no correction applies. If the DNM call set is closer to complete, dividing by
    this fraction is the right correction. Both were computed; the over-adjustment at
    GC 0.61 is 1.22 uncorrected and 1.44 fully corrected, so the finding survives
    either way and the correction only strengthens it.

    Caveat on the correction itself: it is applied per bin, i.e. assumed uniform
    across contexts within a bin. Coverage dropout is a property of positions, and
    contexts occupy different positions, so this is approximate.
    """
    query = f"""
        WITH win AS ({_analyzed_windows_cte(edges, features, annot, coding_prop_threshold)})
        SELECT win.gc_bin AS gc_bin, COUNT(*) AS n_windows,
               AVG(s.possible) / 3000.0 AS callable_fraction
        FROM read_csv_auto('{step1_expected}', delim='\t', header=True) s
        INNER JOIN win ON s.element_id = win.element_id
        GROUP BY win.gc_bin
    """
    df = _connect(memory_limit).execute(query).pl().sort("gc_bin")
    print(f"callable fraction: {df['callable_fraction'].min():.3f}-"
          f"{df['callable_fraction'].max():.3f} across GC bins")
    return df


# ------------------------- empirical estimator 2 (cross-check): case-control

def load_control_counts_by_context_bin(edges: np.ndarray,
                                       control_sites: str = TRAIN_DNM0_SITES,
                                       features: str = FEATURES_GENOME,
                                       annot: str = ANNOT_GENOME,
                                       coding_prop_threshold: float = 0.0,
                                       memory_limit: str = "8GB") -> pl.DataFrame:
    """
    dnm0 background sites per (context, GC bin), mapped to windows by position and
    restricted to the analyzed window set -- i.e. the same treatment
    load_dnm_counts_by_context_bin gives the positives.

    This makes a genuine cross-check possible: pairing these counts with the DNM
    counts gives an estimator that differs from the E1-denominator one ONLY in the
    denominator (a sample of real background sites instead of the model's
    possible-site expectation). On matched population and binning the two agree to
    1-2% through GC 0.65, so the result does not depend on that choice.
    """
    query = f"""
        WITH win AS ({_analyzed_windows_cte(edges, features, annot, coding_prop_threshold)}),
        d0 AS (
            SELECT context, {ELEMENT_ID_FROM_LOCUS} AS element_id
            FROM read_csv_auto('{control_sites}', delim='\t', header=True)
            WHERE locus NOT LIKE 'chrX:%'
        )
        SELECT d0.context AS context, win.gc_bin AS gc_bin, COUNT(*) AS n_control
        FROM d0 INNER JOIN win ON d0.element_id = win.element_id
        GROUP BY d0.context, win.gc_bin
    """
    df = _connect(memory_limit).execute(query).pl()
    print(f"control counts by (context, bin): {df.height} rows, "
          f"{int(df['n_control'].sum()):,} background sites in the analyzed window set")
    return df


def empirical_from_control_counts(counts: pl.DataFrame,
                                  controls: pl.DataFrame) -> pl.DataFrame:
    """
    Per (context, GC bin) empirical adjustment as DNMs per background control site,
    in the same (r_raw, se_raw, n_eff) shape empirical_from_dnm_counts returns.
    Poisson error on the DNM count dominates -- there are ~11x more controls.
    """
    df = (controls.join(counts, on=["context", "gc_bin"], how="left")
                  .with_columns(pl.col("n_dnm").fill_null(0).cast(pl.Float64))
                  .filter(pl.col("n_control") > 0))
    return df.with_columns([
        (pl.col("n_dnm") / pl.col("n_control")).alias("r_raw"),
        (pl.col("n_dnm").sqrt() / pl.col("n_control")).alias("se_raw"),
        pl.col("n_dnm").alias("n_eff"),
    ]).select(["context", "gc_bin", "r_raw", "se_raw", "n_eff"])


def load_training_by_context_bin(edges: np.ndarray,
                                 drop_chrx_from_dnm1: bool = True,
                                 memory_limit: str = "8GB") -> pl.DataFrame:
    """
    Per (context, GC bin) counts over the dnm1/dnm0 training population as fitted:
    n_sites, k_dnm, p_hat = k/n, binned by each site's own GC_content_1k feature.

    MIS-SPECIFIED AS AN EMPIRICAL REFERENCE -- kept because it is the population the
    models were actually fit on, and because check_dnm0_sampling() needs it.

    The problem is not the dnm0 denominator and not the site-flank binning; both were
    tested and neither matters (see load_control_counts_by_context_bin). The problem
    is that this spans the WHOLE genome, while the E1 weights and the per-context
    normalization in combine_non_cpg are computed over the analyzed window set
    (pass_qc, noncoding, autosome+PAR). Feeding rates measured on one population into
    weights derived from another is inconsistent, and it inflates the empirical curve
    dramatically in the high-GC tail: 2.02 at GC 0.645 versus 1.11-1.15 once the same
    sites are restricted to the analyzed windows. Use
    load_control_counts_by_context_bin instead.

    drop_chrx_from_dnm1 removes chrX from the positives as well as the negatives. The
    published fitting code (analyze_individual_feature_effects.py:18, mirrored by
    dnm_model.load_training_data) drops chrX from dnm0 ONLY, which inflates the
    apparent DNM rate on chrX; fitted models inherit that asymmetry, an empirical
    reference should not.
    """
    bin_expr = sql_bin_expr("gc / 100.0", edges)
    chrx_dnm1 = "AND s.locus NOT LIKE 'chrX:%'" if drop_chrx_from_dnm1 else ""
    query = f"""
        WITH sites AS (
            SELECT s.context AS context, f.GC_content_1k AS gc, 1 AS label
            FROM (SELECT locus, context
                  FROM read_csv_auto('{TRAIN_DNM1_SITES}', delim='\t', header=True)) s
            INNER JOIN (SELECT element_id, GC_content_1k
                        FROM read_csv_auto('{TRAIN_DNM1_FEATURES}', delim='\t', header=True)) f
              ON s.locus = f.element_id
            WHERE f.GC_content_1k IS NOT NULL {chrx_dnm1}
            UNION ALL
            SELECT s.context AS context, f.GC_content_1k AS gc, 0 AS label
            FROM (SELECT locus, context
                  FROM read_csv_auto('{TRAIN_DNM0_SITES}', delim='\t', header=True)) s
            INNER JOIN (SELECT element_id, GC_content_1k
                        FROM read_csv_auto('{TRAIN_DNM0_FEATURES}', delim='\t', header=True)) f
              ON s.locus = f.element_id
            WHERE f.GC_content_1k IS NOT NULL AND s.locus NOT LIKE 'chrX:%'
        )
        SELECT context, {bin_expr} AS gc_bin, COUNT(*) AS n_sites,
               CAST(SUM(label) AS DOUBLE) AS k_dnm, AVG(gc) / 100.0 AS gc_mid_sites
        FROM sites GROUP BY 1, 2
    """
    df = _connect(memory_limit).execute(query).pl()
    df = df.with_columns((pl.col("k_dnm") / pl.col("n_sites")).alias("p_hat"))
    print(f"training by (context, bin): {df.height} rows, "
          f"{int(df['n_sites'].sum()):,} sites, {int(df['k_dnm'].sum()):,} DNMs")
    return df


def empirical_from_case_control(training: pl.DataFrame) -> pl.DataFrame:
    """Per (context, GC bin) case-control rate with binomial error, in the same
    (r_raw, se_raw, n_eff) shape empirical_from_dnm_counts returns."""
    return training.with_columns([
        pl.col("p_hat").alias("r_raw"),
        (pl.col("p_hat") * (1 - pl.col("p_hat")) / pl.col("n_sites")).sqrt().alias("se_raw"),
        pl.col("k_dnm").alias("n_eff"),
    ]).select(["context", "gc_bin", "r_raw", "se_raw", "n_eff"])


# ---------------------------------------------------------- the comparison

def combine_non_cpg(genome: pl.DataFrame, empirical: pl.DataFrame,
                    min_n_eff: float = 20, min_weight_covered: float = 0.90) -> pl.DataFrame:
    """
    Aggregate model and empirical adjustments over non-CpG contexts with identical
    E1 weights, after normalizing BOTH per context to E1-weighted mean 1 over the
    analyzed windows.

    WHY BOTH, AND WHY PER CONTEXT. r_c(w) = sigma(b0 + b.z(w)) / sigma(b0) is already
    a ratio of the rate at w to the rate at the average feature vector, so the
    empirical analogue is the same ratio -- the DNM rate in this GC bin over the DNM
    rate for that context overall. That needs no free constant. Per context is
    mandatory rather than optional, because D_c/E1_c is not on a common scale across
    contexts: fitted_po saturates by different amounts in different contexts, so the
    conversion from "gnomAD expected" to "DNM expected" is context-specific. Having
    normalized the empirical side per context, the model side must be normalized the
    same way or the aggregate comparison is no longer composition-matched -- as GC
    rises the trinucleotide mix shifts, and un-normalized between-context level
    differences would leak into the curve as if they were GC dependence.

    The result is that both curves mean the same thing -- 1 is "no adjustment
    relative to this context's own average" -- and every difference between them is
    GC shape, which is the only part that survives into Gnocchi's expected counts.

    A (context, bin) cell is used only if it carries at least min_n_eff observed
    DNMs, and a GC bin is reported only if the usable cells still carry
    min_weight_covered of that bin's non-CpG step-1 expected weight -- so a bin is
    never built from an unrepresentative minority of its contexts.

    Columns out: gc_bin, e1_non, n_contexts, weight_covered, r_non_model,
    r_non_empirical, se_r_non_empirical, inflation (= model / empirical).
    """
    non = (genome.filter(~pl.col("context").is_in(CPG_CONTEXTS))
                 .join(empirical, on=["context", "gc_bin"], how="left")
                 .with_columns((pl.col("n_eff").fill_null(0) >= min_n_eff).alias("usable")))

    # Each context's own E1-weighted mean, over the same cells on both sides.
    norm = (non.filter(pl.col("usable"))
               .group_by("context")
               .agg([(pl.col("e1") * pl.col("r_model")).sum().alias("num_model"),
                     (pl.col("e1") * pl.col("r_raw")).sum().alias("num_emp"),
                     pl.col("e1").sum().alias("den")])
               .with_columns([(pl.col("num_model") / pl.col("den")).alias("mean_model"),
                              (pl.col("num_emp") / pl.col("den")).alias("mean_emp")])
               .select(["context", "mean_model", "mean_emp"]))

    non = non.join(norm, on="context", how="left").with_columns([
        (pl.col("r_model") / pl.col("mean_model")).alias("r_model_norm"),
        (pl.col("r_raw") / pl.col("mean_emp")).alias("r_empirical"),
        (pl.col("se_raw") / pl.col("mean_emp")).alias("se_empirical"),
    ])

    total_w = non.group_by("gc_bin").agg(pl.col("e1").sum().alias("e1_all"))
    used = non.filter(pl.col("usable") & pl.col("r_empirical").is_not_null())

    binned = used.group_by("gc_bin").agg([
        pl.col("e1").sum().alias("e1_non"),
        (pl.col("e1") * pl.col("r_model_norm")).sum().alias("num_model"),
        (pl.col("e1") * pl.col("r_empirical")).sum().alias("num_emp"),
        ((pl.col("e1") * pl.col("se_empirical")).pow(2)).sum().alias("var_num"),
        pl.len().alias("n_contexts"),
    ]).join(total_w, on="gc_bin", how="left")

    binned = binned.with_columns([
        (pl.col("num_model") / pl.col("e1_non")).alias("r_non_model"),
        (pl.col("num_emp") / pl.col("e1_non")).alias("r_non_empirical"),
        (pl.col("var_num").sqrt() / pl.col("e1_non")).alias("se_r_non_empirical"),
        (pl.col("e1_non") / pl.col("e1_all")).alias("weight_covered"),
    ]).with_columns(
        (pl.col("r_non_model") / pl.col("r_non_empirical")).alias("inflation")
    ).filter(pl.col("weight_covered") >= min_weight_covered).sort("gc_bin")

    dropped = total_w.height - binned.height
    if dropped:
        print(f"combine_non_cpg: dropped {dropped} GC bin(s) below "
              f"{min_weight_covered:.0%} non-CpG weight coverage")
    return binned.drop(["num_model", "num_emp", "var_num", "e1_all"])


CPG_METHYL_WEIGHTS_SQL = """
    WITH ft AS (SELECT element_id, GC_content_1k
                FROM read_csv_auto('{features}', delim='\t', header=True)),
    an AS (SELECT element_id FROM read_csv_auto('{annot}', delim='\t', header=True)
           WHERE pass_qc AND coding_prop <= {coding}),
    d0 AS (SELECT context, methyl_level, {eid} AS element_id
           FROM read_csv_auto('{sites}', delim='\t', header=True)
           WHERE locus NOT LIKE 'chrX:%')
    SELECT d0.context AS context, d0.methyl_level AS methyl_level,
           {bin_expr} AS gc_bin, COUNT(*) AS n
    FROM d0 JOIN ft ON d0.element_id = ft.element_id
            JOIN an ON ft.element_id = an.element_id
    WHERE d0.context IN ({ctx})
      AND ft.element_id NOT LIKE 'chrX-%' AND ft.element_id NOT LIKE 'chrY-%'
    GROUP BY 1, 2, 3
"""


def cpg_saturation_artifact(edges: np.ndarray,
                            mutation_rate: str = "tmp/mutation_rate_by_context_methyl.txt",
                            control_sites: str = TRAIN_DNM0_SITES,
                            features: str = FEATURES_GENOME,
                            annot: str = ANNOT_GENOME,
                            coding_prop_threshold: float = 0.0,
                            ref_bin: int | None = None,
                            memory_limit: str = "8GB") -> pl.DataFrame:
    """
    How much of the CpG contexts' apparent GC-dependent adjustment is an artifact of
    fitted_po saturation rather than a real departure from r = 1.

    WHY THIS IS NEEDED, and why it does NOT arise for non-CpG contexts. The empirical
    adjustment is D_c(g) / E1_c(g), and E1 is built from `fitted_po`, the gnomAD
    polymorphism probability. For a NON-CpG context there is a single methylation level,
    so fitted_po is one constant that cancels in the per-context normalization. For a CpG
    context E1 mixes 16 methylation levels whose fitted_po values are saturated to wildly
    different degrees: across methyl 0 -> 15 the C>T `fitted_po` ratio is only 3.0-4.3x
    while the pre-saturation rate `mu` ratio is 9.7-15.2x. So E1 is a COMPRESSED proxy
    for DNM opportunity, compressed most where methylation is highest.

    CpG methylation composition swings hard with GC (mean level ~6.4 in the bulk falling
    to ~1.5 by GC 0.645, since high-GC CpGs are hypomethylated CpG islands), so the
    compression varies with GC and manufactures a spurious decline in D/E1.

    This function quantifies that. Per GC bin it computes

        artifact(g) = weighted_mean_m(mu) / weighted_mean_m(fitted_po)

    over CpG contexts, with weights = the number of dnm0 background sites at each
    (context, methylation level) inside the analyzed windows. Dividing a measured
    D/E1 curve by this leaves the part attributable to a genuine r != 1.

    LIMITS. The weights are dnm0 site counts standing in for per-(context, methylation)
    `possible` counts, which no flat file in the bucket provides -- the per-context
    expected export is already summed over methylation, so the exact weights would need
    the Hail table. And `mu` is itself a downsampled-1000-genome estimate rescaled to a
    fixed total, the best available pre-saturation proxy rather than ground truth. Treat
    the correction as an order-of-magnitude control, not a precise deconfounding.

    ref_bin normalizes the output to 1 at that bin (default: the most populated one).
    """
    query = CPG_METHYL_WEIGHTS_SQL.format(
        features=features, annot=annot, coding=coding_prop_threshold,
        sites=control_sites, eid=ELEMENT_ID_FROM_LOCUS,
        bin_expr=sql_bin_expr("ft.GC_content_1k / 100.0", edges),
        ctx=", ".join(f"'{c}'" for c in CPG_CONTEXTS))
    comp = _connect(memory_limit).execute(query).pl()

    rate = pl.read_csv(mutation_rate, separator="\t")
    agg = (rate.filter(pl.col("context").is_in(CPG_CONTEXTS))
               .group_by(["context", "methylation_level"])
               .agg([pl.col("mu").sum(), pl.col("fitted_po").sum()])
               .rename({"methylation_level": "methyl_level"}))

    m = comp.join(agg, on=["context", "methyl_level"], how="inner")
    out = (m.group_by("gc_bin").agg([
        pl.col("n").sum().alias("n"),
        ((pl.col("methyl_level") * pl.col("n")).sum() / pl.col("n").sum()).alias("mean_methyl"),
        ((pl.col("mu") * pl.col("n")).sum() / pl.col("n").sum()).alias("mu_bar"),
        ((pl.col("fitted_po") * pl.col("n")).sum() / pl.col("n").sum()).alias("po_bar"),
    ]).with_columns((pl.col("mu_bar") / pl.col("po_bar")).alias("artifact")).sort("gc_bin"))

    if ref_bin is None:
        ref_bin = int(out.filter(pl.col("n") == out["n"].max())["gc_bin"][0])
    ref = float(out.filter(pl.col("gc_bin") == ref_bin)["artifact"][0])
    out = out.with_columns((pl.col("artifact") / ref).alias("artifact_rel"))
    print(f"CpG saturation artifact: mean methyl {out['mean_methyl'].max():.1f} -> "
          f"{out['mean_methyl'].min():.1f}, artifact_rel "
          f"{out['artifact_rel'].max():.3f} -> {out['artifact_rel'].min():.3f} "
          f"(normalized at bin {ref_bin})")
    return out.select(["gc_bin", "n", "mean_methyl", "artifact", "artifact_rel"])


def attach_gc_mid(binned: pl.DataFrame, edges: np.ndarray) -> pl.DataFrame:
    """Bin centres from the shared edges, for plotting on the panel-A axis."""
    centres = 0.5 * (edges[:-1] + edges[1:])
    return binned.with_columns(
        pl.col("gc_bin").map_elements(lambda i: float(centres[i]), return_dtype=pl.Float64)
        .alias("gc_mid"))


# ------------------------------------------------------------- the assumption

def check_dnm0_sampling(genome: pl.DataFrame, training: pl.DataFrame,
                        top_n: int = 6) -> pl.DataFrame:
    """
    How closely the dnm0 background pool tracks the genome. For each context, compare
    the GC distribution of its dnm0 sites against the genome-wide
    step-1-expected-weighted GC distribution for the same context -- the population
    dnm0 is meant to represent.

    The pool is NOT uniform: up to 2.0-fold across GC within a context (CCC), and
    pooled over non-CpG contexts it is depleted at low GC (0.95) and enriched at high
    GC (1.25). Note also that the 10:1 dnm0:dnm1 ratio holds only genome-wide, not per
    context -- it ranges from 0.76 (ACG) to 24.8 (GAA) -- so dnm0 is close to a uniform
    genomic sample rather than a per-context-matched one.

    This is a real property of the training data and worth knowing, but it does NOT
    explain the disagreement between the two empirical estimators: pairing DNM counts
    with these same controls on the analyzed window set reproduces the E1-denominator
    curve to 1-2%. Population mismatch, not dnm0 composition, was the cause.

    Returns the per-(context, bin) shares and prints the worst contexts by the
    spread of that ratio across well-populated bins.
    """
    non = genome.filter(~pl.col("context").is_in(CPG_CONTEXTS))
    g_share = (non.with_columns(
                    (pl.col("e1") / pl.col("e1").sum().over("context")).alias("genome_share"))
                  .select(["context", "gc_bin", "genome_share", "e1"]))

    t = training.with_columns((pl.col("n_sites") - pl.col("k_dnm")).alias("n_dnm0"))
    t_share = (t.with_columns(
                    (pl.col("n_dnm0") / pl.col("n_dnm0").sum().over("context")).alias("dnm0_share"))
                 .select(["context", "gc_bin", "dnm0_share", "n_dnm0"]))

    cmp = (g_share.join(t_share, on=["context", "gc_bin"], how="left")
                  .with_columns(pl.col("dnm0_share").fill_null(0.0))
                  .with_columns((pl.col("dnm0_share") / pl.col("genome_share")).alias("ratio")))

    well = cmp.filter((pl.col("genome_share") >= 0.01) & (pl.col("n_dnm0").fill_null(0) >= 200))
    spread = (well.group_by("context")
                  .agg([pl.col("ratio").min().alias("min_ratio"),
                        pl.col("ratio").max().alias("max_ratio"),
                        pl.len().alias("bins")])
                  .with_columns((pl.col("max_ratio") / pl.col("min_ratio")).alias("fold"))
                  .sort("fold", descending=True))
    print("dnm0 sampling uniformity (dnm0 GC share / genome E1 GC share, "
          "well-populated bins only):")
    print(spread.head(top_n))

    pooled = (well.group_by("gc_bin")
                  .agg([pl.col("n_dnm0").sum().alias("n_dnm0"), pl.col("e1").sum().alias("e1")])
                  .sort("gc_bin"))
    pooled = pooled.with_columns(
        ((pl.col("n_dnm0") / pl.col("n_dnm0").sum()) / (pl.col("e1") / pl.col("e1").sum()))
        .alias("pooled_ratio"))
    print("pooled non-CpG dnm0/genome GC share ratio by bin:")
    print(pooled.select(["gc_bin", "pooled_ratio"]))
    return cmp
