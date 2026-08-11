"""
The measurements behind claims panels B and C state in prose but do not plot.

Migrated out of fig3/ (training_representativeness.py, empirical_r.py) when that
directory was retired at c913d87 -- readable in full at 070fee9 -- so that no number
asserted in the figure's text is unregenerable.
Three questions, three functions:

  dnm_rate_by_stratum   Panel C: is the training territory excluded from the scored
                        population actually different, or just absent? -> the sequence
                        dropped for failing gnomAD's variant-call QC carries essentially
                        all of the training set's GC dependence (4.1x the noncoding rate
                        by GC 0.61).

  cpg_methylation_by_gc Panel B: are high-GC CpGs really hypomethylated, and does their
                        DNM rate really collapse? -> 90-100% hypomethylated above GC 0.70
                        (the three bins with n >= 100 are 89.6 / 96.9 / 100%), with the
                        rate falling 0.535 in the GC bulk to 0.195 in the top bin, 2.7x.

  cpg_rate_by_methyl    Panel B: how large is the methylation effect step 1 already
                        absorbs? -> a 4.3-fold range in fitted_po within one
                        trinucleotide context, the largest single rate effect in the
                        model.

None of these is plotted. They exist so the caption can be checked.
"""
import numpy as np
import polars as pl

# First-party. `data` is a sibling module, not a third-party package -- keep it grouped
# with gnocchi_bias, and do not let an isort autofix hoist it above.
import data as D
from gnocchi_bias import dnm_model as M
from gnocchi_bias import windows as W

# The three strata a training site can fall into, relative to the scored population:
# its window is noncoding and in the published constraint table, coding and in it, or not
# in it at all. The third is `failed_qc` and not `no_coverage`: every window absent from
# that table has QC inputs on file and fails one of the paper's three conditions (>= 80%
# of observed variants PASS, mean coverage 25-35x, >= 1000 possible variants), the first
# of those dominating. preconditions/verify_qc_filter.py measures the split.
#
# An earlier comment here claimed this `noncoding` was deliberately broader than panel
# C's `analyzed`, since that one also requires pass_qc. It is not: the published table
# carries pass_qc = True on all 1,984,900 of its rows, so the flag is inert and the two
# definitions pick out the same windows. Keeping both spellings anyway -- each says what
# its own panel means, and a future unfiltered vintage of the table would separate them.
_STRATUM = """CASE WHEN an.element_id IS NULL THEN 'failed_qc'
                   WHEN an.coding_prop <= 0.0 THEN 'noncoding'
                   ELSE 'coding' END"""

# chrX/chrY dropped from BOTH classes. The published fitting code drops chrX from the
# background class only, which inflates the apparent rate there; an empirical reference
# must not inherit that asymmetry.
_TRAINING_SITES = """
    SELECT context, methyl_level, {eid} AS element_id, 1 AS label
    FROM read_csv_auto('{dnm1}', delim='\t', header=True)
    WHERE locus NOT LIKE 'chrX:%' AND locus NOT LIKE 'chrY:%'
    UNION ALL
    SELECT context, methyl_level, {eid} AS element_id, 0 AS label
    FROM read_csv_auto('{dnm0}', delim='\t', header=True)
    WHERE locus NOT LIKE 'chrX:%' AND locus NOT LIKE 'chrY:%'
"""


def _training_sql(cache_dir: str) -> str:
    return _TRAINING_SITES.format(
        eid=D.ELEMENT_ID_FROM_LOCUS,
        dnm1=W.download(M.TRAINING_FILES["dnm1_sites"], cache_dir),
        dnm0=W.download(M.TRAINING_FILES["dnm0_sites"], cache_dir))


def _binned_training_query(cache_dir: str, edges: np.ndarray, where: str,
                           dims: list[tuple[str, str]] = (), aggs: str = "") -> str:
    """
    Training sites joined to their 1 kb tile's GC and constraint annotation, aggregated
    per GC bin (plus any extra grouping `dims`, each an (expression, alias) pair).
    `aggs` adds further aggregate select items.

    GROUP BY repeats the expressions rather than using positional indices: the two
    callers have different select-list shapes, and positional GROUP BY silently grouped
    by the wrong column when this was first written.
    """
    gc_bin = D.sql_bin_expr("ft.GC_content_1k / 100.0", edges)
    dim_select = "".join(f"{expr} AS {alias}, " for expr, alias in dims)
    group_by = ", ".join([expr for expr, _ in dims] + [gc_bin])
    return f"""
        WITH an AS (SELECT element_id, pass_qc, coding_prop
                    FROM read_csv_auto('{W.download(W.REMOTE_FILES["annot"], cache_dir)}',
                                       delim='\t', header=True)),
        ft AS (SELECT element_id, GC_content_1k
               FROM read_csv_auto('{W.download(W.REMOTE_FILES["features"], cache_dir)}',
                                  delim='\t', header=True)),
        s AS ({_training_sql(cache_dir)})
        SELECT {dim_select}{gc_bin} AS gc_bin,
               CAST(SUM(s.label) AS BIGINT) AS k, COUNT(*) AS n,
               AVG(ft.GC_content_1k) AS gc_pct{"," if aggs else ""} {aggs}
        FROM s JOIN ft ON s.element_id = ft.element_id
               LEFT JOIN an ON s.element_id = an.element_id
        WHERE {where}
        GROUP BY {group_by}
    """


def dnm_rate_by_stratum(edges: np.ndarray, cache_dir: str = D.CACHE_DIR,
                        force: bool = False, memory_limit: str = "10GB") -> pl.DataFrame:
    """
    Empirical P(DNM) over non-CpG training sites, per GC bin, split by where the site
    sits relative to the scored population: noncoding, coding, or absent from the
    constraint table because its window failed gnomAD variant-call QC.

    Both classes are labelled, DNMs included -- that is what makes this a rate rather
    than a composition. 72,801 of the non-CpG autosomal DNMs sit in the QC-failing
    stratum, against 17,545 coding and 241,479 noncoding.

    THE POINT. The noncoding and coding curves are both nearly flat in GC and nearly
    equal, so the coding exclusion is not what makes the training set's GC dependence
    steep. The QC-failing curve is not flat: it runs ~1.6x the noncoding rate in the GC
    bulk and ~4.1x by GC 0.61. Essentially all of the original training set's GC
    dependence is contributed by sequence gnomAD could not call reliably -- which is also
    where trio DNM calling is least reliable, so part of the excess is plausibly
    false-positive DNM calls rather than real mutation.

    Columns: stratum, gc_bin, gc_pct, k (DNMs), n (sites), p = k/n.
    """
    def build():
        q = _binned_training_query(
            cache_dir, edges, dims=[(_STRATUM, "stratum")],
            where=f"s.context NOT IN ({', '.join(repr(c) for c in M.CPG_CONTEXTS)})")
        return D.duck(memory_limit).execute(q).pl()

    df = D.cached(f"dnm_rate_by_stratum.{len(edges) - 1}bins.parquet", build, force)
    return df.with_columns((pl.col("k") / pl.col("n")).alias("p")).sort(["stratum", "gc_bin"])


def stratum_ratios(st: pl.DataFrame, edges: np.ndarray, min_n: int = 2000) -> pl.DataFrame:
    """
    dnm_rate_by_stratum() reshaped to the two ratios panel C plots: each excluded
    stratum's non-CpG DNM rate over the noncoding stratum's, per GC bin.

    Ratios rather than raw rates because the question is comparative -- is the excluded
    territory DIFFERENT from the territory Gnocchi is scored on -- and because the
    noncoding rate itself drifts mildly with GC, which a ratio divides out.

    Error bars are the delta-method SE of log(ratio), SE = sqrt((1-p_a)/k_a +
    (1-p_b)/k_b), i.e. binomial noise in both strata. min_n drops bins where either
    stratum holds fewer than that many sites.

    Columns: gc_bin, gc_mid, and {coding,failed_qc}_{ratio,se_log}.
    """
    keep = st.filter(pl.col("n") >= min_n)
    base = keep.filter(pl.col("stratum") == "noncoding").select(
        ["gc_bin", pl.col("p").alias("p_nc"), pl.col("k").alias("k_nc")])
    out = base
    for stratum in ("coding", "failed_qc"):
        s = keep.filter(pl.col("stratum") == stratum).select(
            ["gc_bin", pl.col("p").alias("p_s"), pl.col("k").alias("k_s")])
        out = out.join(s, on="gc_bin", how="inner").with_columns([
            (pl.col("p_s") / pl.col("p_nc")).alias(f"{stratum}_ratio"),
            (((1 - pl.col("p_s")) / pl.col("k_s")
              + (1 - pl.col("p_nc")) / pl.col("k_nc")).sqrt()).alias(f"{stratum}_se_log"),
        ]).drop(["p_s", "k_s"])
    out = out.drop(["p_nc", "k_nc"]).sort("gc_bin")
    return out.with_columns(D.bin_centres(edges, out["gc_bin"]))


def cpg_methylation_by_gc(edges: np.ndarray, cache_dir: str = D.CACHE_DIR,
                          force: bool = False, memory_limit: str = "10GB") -> pl.DataFrame:
    """
    CpG-context training sites per GC bin: mean methylation level, the fraction that are
    hypomethylated (level <= 1), and the empirical DNM rate.

    THE POINT. High-GC CpGs are CpG islands -- 94-98% hypomethylated above GC 0.7,
    against ~2% in the GC bulk -- and their DNM rate collapses by ~2.6x. That is a large,
    strongly GC-dependent effect, and it is exactly the effect step 1 already models,
    since fitted_po is keyed by methylation level. Which is why r_CpG ~ 1 in panel B is
    the CORRECT behaviour and not a failure: there is nothing left for the regional
    adjustment to correct.

    Measured over the whole training population, not restricted to the analyzed windows:
    the claim is about CpG biology, not about the scored population.

    Columns: gc_bin, gc_pct, n, k, p (DNM rate), mean_methyl, frac_hypomethylated.
    """
    def build():
        q = _binned_training_query(
            cache_dir, edges,
            aggs=("AVG(CAST(s.methyl_level AS DOUBLE)) AS mean_methyl, "
                  "AVG(CASE WHEN s.methyl_level <= 1 THEN 1.0 ELSE 0.0 END) "
                  "AS frac_hypomethylated"),
            where=f"s.context IN ({', '.join(repr(c) for c in M.CPG_CONTEXTS)})")
        return D.duck(memory_limit).execute(q).pl()

    df = D.cached(f"cpg_methylation_by_gc.{len(edges) - 1}bins.parquet", build, force)
    return df.with_columns((pl.col("k") / pl.col("n")).alias("p")).sort("gc_bin")


def cpg_rate_by_methyl(cache_dir: str = D.CACHE_DIR) -> pl.DataFrame:
    """
    The CpG C>T mutation rate by methylation level, straight from the published
    per-(context, ref, alt, methylation) table -- the size of the effect step 1 absorbs.

    Two columns matter. `fitted_po` is what the pipeline actually uses as the per-site
    step-1 probability; across methylation 0 -> 15 it spans ~4.3x within a single
    trinucleotide context, the largest single rate effect in the model. `mu` is the
    independent pre-saturation estimate, and it spans ~10-15x over the same range --
    the gap between the two IS the saturation of fitted_po, which is why a naive
    D/E1 ratio understates the CpG rate at high methylation. (That is a control on the
    CpG story, not part of the figure's argument; panel B's claim rests on r being a
    ratio in which such level effects cancel.)
    """
    rate = pl.read_csv(W.download(M.MUTATION_RATE_FILE, cache_dir), separator="\t")
    ct = (rate.filter(pl.col("context").is_in(M.CPG_CONTEXTS)
                      & (pl.col("ref") == "C") & (pl.col("alt") == "T"))
              .select(["context", "methylation_level", "mu", "fitted_po"])
              .sort(["context", "methylation_level"]))
    lo = ct.filter(pl.col("methylation_level") == ct["methylation_level"].min())
    hi = ct.filter(pl.col("methylation_level") == ct["methylation_level"].max())
    span = (lo.join(hi, on="context", suffix="_hi")
              .with_columns([(pl.col("fitted_po_hi") / pl.col("fitted_po")).alias("po_ratio"),
                             (pl.col("mu_hi") / pl.col("mu")).alias("mu_ratio")]))
    print(f"CpG C>T, methylation {ct['methylation_level'].min()} -> "
          f"{ct['methylation_level'].max()}:  fitted_po spans "
          f"{span['po_ratio'].min():.1f}-{span['po_ratio'].max():.1f}x, "
          f"mu spans {span['mu_ratio'].min():.1f}-{span['mu_ratio'].max():.1f}x")
    return ct
