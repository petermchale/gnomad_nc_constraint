"""
Is the DNM training set representative of the noncoding genome Gnocchi is scored on?

Two observations motivate this module. The empirical non-CpG DNM probability rises
smoothly and modestly with GC when measured over the analyzed window set
(r_non_vs_empirical.pdf), but rises steeply and non-monotonically when measured over
the DNM training population (dnm_probability_non_cpg.pdf). Since the training curve's
denominator is a sample of background sites rather than the genome's own opportunity
count, the natural suspicion is that the background sample is what changes the shape.

This module tests that by building the SAME curve four ways, changing one thing at a
time, so the responsible ingredient is identified rather than guessed:

  A  DNMs / opportunities, analyzed windows, per-context normalized   (fig. 3's curve)
  B  DNMs / dnm0 background sites, analyzed windows, per-context normalized
  C  DNMs / dnm0 background sites, WHOLE GENOME, per-context normalized
  D  pooled DNM fraction, whole genome, NO per-context normalization  (the reliability
     diagram's curve)

A -> B changes only the denominator. B -> C changes only the window population.
C -> D changes only the aggregation. Measured, A and B agree to 0.4% through GC 0.61
and C and D agree to ~2%, while B and C differ by 63% at GC 0.61 -- so neither the
choice of background sites nor the pooling is responsible. The window population is.

WHY THE POPULATION MATTERS is then measured directly by
dnm0_window_composition(): the fraction of the training set's background sites that
lie in the analyzed noncoding genome at all. It falls from 0.83 in the GC bulk to
0.29 by GC 0.68, because GC-rich sequence is disproportionately coding or lacks
gnomAD coverage. The models are fit on that population and applied to the analyzed
one.

So the user-facing conclusion is a refinement, not a confirmation: the training set
IS unrepresentative of the noncoding genome, increasingly so with GC, but the
unrepresentativeness is in WHICH REGIONS it covers, not in how background sites were
drawn within a region.
"""

import numpy as np
import polars as pl

import empirical_r as E
from gnocchi_bias.dnm_model import CPG_CONTEXTS
from r_eff import sql_bin_expr

# The four rungs, in the order they are drawn. Each entry is
# (column, style key, dashed, legend label).
LADDER_SERIES = [
    ("r_non_model", "model", False, "Gnocchi's fitted $r$ (non-CpG)"),
    ("A", "analyzed", False, "DNMs / opportunities, analyzed windows"),
    ("B", "analyzed", True, "DNMs / background sites, analyzed windows"),
    ("C", "genome", False, "DNMs / background sites, whole genome"),
    ("D", "genome", True, "Pooled DNM fraction, whole genome"),
]


def _rung(genome: pl.DataFrame, empirical: pl.DataFrame, name: str) -> pl.DataFrame:
    """One rung: aggregate an empirical estimate over non-CpG contexts exactly the
    way fig. 3 does, so the rungs differ only in the ingredient being varied."""
    b = E.combine_non_cpg(genome, empirical, min_n_eff=0, min_weight_covered=0.0)
    return (b.select(["gc_bin", "e1_non", "r_non_model", "r_non_empirical"])
             .rename({"r_non_empirical": name}))


def empirical_from_training_counts(training: pl.DataFrame) -> pl.DataFrame:
    """
    DNMs per background site over the WHOLE-GENOME training population, in the
    (r_raw, se_raw, n_eff) shape combine_non_cpg consumes. Deliberately the
    unrestricted population -- that is the variable this module isolates.
    """
    return (training.with_columns((pl.col("n_sites") - pl.col("k_dnm")).alias("n0"))
                    .filter(pl.col("n0") > 0)
                    .with_columns([(pl.col("k_dnm") / pl.col("n0")).alias("r_raw"),
                                   (pl.col("k_dnm").sqrt() / pl.col("n0")).alias("se_raw"),
                                   pl.col("k_dnm").alias("n_eff")])
                    .select(["context", "gc_bin", "r_raw", "se_raw", "n_eff"]))


def build_ladder(genome: pl.DataFrame, counts: pl.DataFrame, controls: pl.DataFrame,
                 training: pl.DataFrame, edges: np.ndarray,
                 min_dnm: int = 200) -> pl.DataFrame:
    """
    All four rungs plus the fitted model curve, on one set of GC bins.

    Every curve is rescaled to E1-weighted mean 1 over the retained bins, using the
    same weights, so only SHAPE is compared -- the rungs have different natural
    levels (a rate per opportunity, a rate per sampled control, a raw fraction) and
    those levels are not meaningful against each other.

    min_dnm drops GC bins holding fewer than that many observed DNMs inside the
    analyzed window set, where every rung is noise.
    """
    A = _rung(genome, E.empirical_from_dnm_counts(genome, counts,
                                                  denominator="opportunities"), "A")
    B = _rung(genome, E.empirical_from_control_counts(counts, controls), "B")
    C = _rung(genome, empirical_from_training_counts(training), "C")

    # D: pooled over non-CpG with no per-context normalization -- the reliability
    # diagram's own construction, carried onto this axis.
    D = (training.filter(~pl.col("context").is_in(CPG_CONTEXTS))
                 .group_by("gc_bin")
                 .agg([pl.col("n_sites").sum().alias("n_sites"),
                       pl.col("k_dnm").sum().alias("k_dnm")])
                 .with_columns((pl.col("k_dnm") / pl.col("n_sites")).alias("D")))
    dnm_total = counts.group_by("gc_bin").agg(pl.col("n_dnm").sum().alias("dnm_total"))

    out = (A.join(B.drop(["e1_non", "r_non_model"]), on="gc_bin")
            .join(C.drop(["e1_non", "r_non_model"]), on="gc_bin")
            .join(D.select(["gc_bin", "D"]), on="gc_bin")
            .join(dnm_total, on="gc_bin")
            .filter(pl.col("dnm_total") >= min_dnm)
            .sort("gc_bin"))

    w = out["e1_non"].to_numpy()
    for col, *_ in LADDER_SERIES:
        v = out[col].to_numpy()
        out = out.with_columns(pl.Series(col, v / np.average(v, weights=w)))

    centres = 0.5 * (edges[:-1] + edges[1:])
    return out.with_columns(
        pl.Series("gc_mid", [float(centres[i]) for i in out["gc_bin"]]))


def report_ladder(ladder: pl.DataFrame) -> None:
    """Print the pairwise disagreements the figure's claim rests on."""
    def worst(a, b, upto_gc=0.62):
        sub = ladder.filter(pl.col("gc_mid") <= upto_gc)
        return float((sub[a] / sub[b] - 1).abs().max())
    print(f"  denominator  (A vs B): max |ratio-1| = {worst('A','B'):.1%}")
    print(f"  aggregation  (C vs D): max |ratio-1| = {worst('C','D'):.1%}")
    print(f"  POPULATION   (B vs C): max |ratio-1| = {worst('B','C'):.1%}")


def dnm_rate_by_stratum(edges: np.ndarray | None = None,
                        dnm_sites: str = E.TRAIN_DNM1_SITES,
                        control_sites: str = E.TRAIN_DNM0_SITES,
                        features: str = E.FEATURES_GENOME,
                        annot: str = E.ANNOT_GENOME,
                        non_cpg_only: bool = False,
                        memory_limit: str = "10GB") -> pl.DataFrame:
    """
    Empirical P(DNM) in the training set, split by where the site sits relative to the
    analyzed window population: noncoding (the scored set), coding, or absent from the
    constraint table altogether (no gnomAD coverage).

    Returns counts per (stratum, context) if `edges` is None, or per (stratum, gc_bin)
    if edges are given -- the caller does the aggregation, since the interesting
    comparisons need different weightings (raw, context-matched, CpG-split).

    WHY THIS IS THE ANSWER TO "why does restricting the training set help". The
    noncoding and coding strata have nearly flat, nearly equal non-CpG DNM rates across
    GC. The no-coverage stratum does not: its rate runs 1.55x the noncoding rate in the
    GC bulk and 4.06x by GC 0.61. So essentially all of the original training set's steep
    GC dependence is contributed by sequence gnomAD cannot call -- which is also where
    trio DNM calling is least reliable.

    chrX and chrY are dropped from BOTH classes here. The published fitting code drops
    chrX from dnm0 only, which inflates the apparent rate on chrX; an empirical
    reference must not inherit that asymmetry.
    """
    bin_sql = (f"{sql_bin_expr('ft.GC_content_1k / 100.0', edges)} AS gc_bin"
               if edges is not None else "s.context AS context")
    join_ft = ("JOIN (SELECT element_id, GC_content_1k FROM "
               f"read_csv_auto('{features}', delim='\t', header=True)) ft "
               "ON s.element_id = ft.element_id" if edges is not None else "")
    cpg_filter = ("WHERE s.context NOT IN ('ACG','CCG','GCG','TCG')"
                  if non_cpg_only else "")
    query = f"""
        WITH an AS (SELECT element_id, pass_qc, coding_prop
                    FROM read_csv_auto('{annot}', delim='\t', header=True)),
        s AS (
          SELECT context, {E.ELEMENT_ID_FROM_LOCUS} AS element_id, 1 AS label
          FROM read_csv_auto('{dnm_sites}', delim='\t', header=True)
          WHERE locus NOT LIKE 'chrX:%' AND locus NOT LIKE 'chrY:%'
          UNION ALL
          SELECT context, {E.ELEMENT_ID_FROM_LOCUS} AS element_id, 0 AS label
          FROM read_csv_auto('{control_sites}', delim='\t', header=True)
          WHERE locus NOT LIKE 'chrX:%' AND locus NOT LIKE 'chrY:%')
        SELECT CASE WHEN an.element_id IS NULL THEN 'no_coverage'
                    WHEN an.coding_prop <= 0.0 THEN 'noncoding'
                    ELSE 'coding' END AS stratum,
               {bin_sql},
               CAST(SUM(s.label) AS BIGINT) AS k, COUNT(*) AS n
        FROM s {join_ft} LEFT JOIN an ON s.element_id = an.element_id
        {cpg_filter}
        GROUP BY 1, 2
    """
    df = E._connect(memory_limit).execute(query).pl()
    return df.with_columns((pl.col("k") / pl.col("n")).alias("p"))


def dnm0_window_composition(edges: np.ndarray,
                            control_sites: str = E.TRAIN_DNM0_SITES,
                            features: str = E.FEATURES_GENOME,
                            annot: str = E.ANNOT_GENOME,
                            coding_prop_threshold: float = 0.0,
                            memory_limit: str = "8GB") -> pl.DataFrame:
    """
    Where each GC bin's non-CpG background training sites actually sit, relative to
    the analyzed noncoding genome. Sites are mapped to their containing 1 kb tile and
    split three ways:

        n_analyzed  tile passes pass_qc, coding_prop <= threshold, autosome/PAR
        n_coding    tile is annotated but exceeds the coding_prop threshold
        n_noannot   tile has no row in the constraint table at all (no gnomAD
                    coverage / excluded upstream)

    These partition the total exactly, which the caller asserts. The point of the
    table is the trend in n_analyzed/n_total: the training data at high GC largely
    does not live in the territory Gnocchi is evaluated on.
    """
    b = sql_bin_expr("ft.GC_content_1k / 100.0", edges)
    ctx = ", ".join(f"'{c}'" for c in CPG_CONTEXTS)
    query = f"""
        WITH ft AS (SELECT element_id, GC_content_1k
                    FROM read_csv_auto('{features}', delim='\t', header=True)),
        an AS (SELECT element_id, pass_qc, coding_prop
               FROM read_csv_auto('{annot}', delim='\t', header=True)),
        d0 AS (SELECT context, {E.ELEMENT_ID_FROM_LOCUS} AS element_id
               FROM read_csv_auto('{control_sites}', delim='\t', header=True)
               WHERE locus NOT LIKE 'chrX:%')
        SELECT {b} AS gc_bin,
               COUNT(*) AS n_total,
               CAST(SUM(CASE WHEN an.pass_qc AND an.coding_prop <= {coding_prop_threshold}
                              AND ft.element_id NOT LIKE 'chrX-%'
                              AND ft.element_id NOT LIKE 'chrY-%'
                         THEN 1 ELSE 0 END) AS BIGINT) AS n_analyzed,
               CAST(SUM(CASE WHEN an.element_id IS NOT NULL
                              AND NOT (an.pass_qc AND an.coding_prop <= {coding_prop_threshold})
                         THEN 1 ELSE 0 END) AS BIGINT) AS n_coding,
               CAST(SUM(CASE WHEN an.element_id IS NULL THEN 1 ELSE 0 END) AS BIGINT) AS n_noannot
        FROM d0
        JOIN ft ON d0.element_id = ft.element_id
        LEFT JOIN an ON d0.element_id = an.element_id
        WHERE d0.context NOT IN ({ctx})
        GROUP BY 1
    """
    df = E._connect(memory_limit).execute(query).pl().sort("gc_bin")
    resid = (df["n_total"] - df["n_analyzed"] - df["n_coding"] - df["n_noannot"])
    assert int(resid.abs().sum()) == 0, "composition categories do not partition the total"

    centres = 0.5 * (edges[:-1] + edges[1:])
    df = df.with_columns([
        pl.Series("gc_mid", [float(centres[i]) for i in df["gc_bin"]]),
        (pl.col("n_analyzed") / pl.col("n_total")).alias("frac_analyzed"),
        (pl.col("n_coding") / pl.col("n_total")).alias("frac_coding"),
        (pl.col("n_noannot") / pl.col("n_total")).alias("frac_noannot"),
    ])
    print(f"dnm0 composition: {int(df['n_total'].sum()):,} non-CpG background sites; "
          f"analyzed fraction {df['frac_analyzed'].min():.2f}-{df['frac_analyzed'].max():.2f} "
          f"across GC bins")
    return df
