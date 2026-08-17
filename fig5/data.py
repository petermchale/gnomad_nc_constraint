"""
Data for the five panels of Fig. 5 and for Supporting Figure 7. Every builder caches its
result as parquet in fig5/output/, so the notebook is instant after the first pass.

One builder per plotted quantity, grouped by the panel that draws it. Panel C and
Supporting Figure 7 were a separate module, diagnostics.py, back when they measured
claims the text asserted in prose and nothing plotted -- panel C gained a lower row and
the supporting figure was built, so they are ordinary panel data now and live here with
the rest. Read that history at ea1805c if a comment seems to assume it.

Inputs, in three groups:

  * the public gnomAD-NC-constraint bucket, downloaded on demand into published/ by
    gnocchi_bias.windows.download;
  * the three refits fig5/refit.py produces (full / scored / sizematched), which live in
    the repo-root refits/ -- ONE copy, also read directly by dnm_training_size/;
  * two files that are NOT in this repo, read from fig5/config.py (NOT set in the
    notebook -- refit.py reads the same module, and the two must agree) -- a
    depletion-rank BED (panel A's third curve) and McHale et al.'s neutral-window file
    (which narrows the analyzed set to their 693,270 windows). Both default to None;
    the figure builds without them.
"""
import hashlib
import os

import duckdb
import numpy as np
import pandas as pd
import polars as pl

# First-party. `config` is a sibling module, not a third-party package -- keep it
# grouped with gnocchi_bias, and do not let an isort autofix hoist it above.
import config
from gnocchi_bias import dnm_model as M
from gnocchi_bias import windows as W

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
OUTPUT_DIR = os.path.join(HERE, "output")     # figures and this figure's own caches
CACHE_DIR = os.path.join(REPO_ROOT, "published")    # downloaded bucket files
REFITS_DIR = os.path.join(REPO_ROOT, "refits")  # the shared refit outputs

N_BINS = 20
XRANGE = (0.2, 0.73)   # visually matched to McHale et al. Fig. 2A

# Written by fig5/refit.py into REFITS_DIR; `pop` is full / scored / sizematched.
REFIT_FILES = {
    "expected": "expected_counts_by_context_methyl_genome_1kb.{pop}.txt",
    "rr": "rr_by_context.{pop}.txt",
    "predictions": "training_reliability_predictions.{pop}.txt",
    "selected": "selected.{pop}.txt",
}

# Site -> containing 1 kb tile, in SQL. element_id is 0-based chr-start-end.
ELEMENT_ID_FROM_LOCUS = (
    "split_part(locus,':',1) || '-' || "
    "CAST(((CAST(split_part(locus,':',2) AS BIGINT)-1)//1000)*1000 AS VARCHAR) || '-' || "
    "CAST(((CAST(split_part(locus,':',2) AS BIGINT)-1)//1000)*1000+1000 AS VARCHAR)")


def refit_path(kind: str, pop: str, refits_dir: str = REFITS_DIR) -> str:
    """
    A refit table, verified to have been built under the CURRENT
    config.NEUTRAL_WINDOWS_BED.
    That check is the reason to route every read through here: `scored` is fit on the
    analyzed window set, so a refit built under a different setting than the panels are
    evaluated under is trained on one population and scored on another.
    """
    path = os.path.join(refits_dir, REFIT_FILES[kind].format(pop=pop))
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path}\nRun:  .venv/bin/python fig5/refit.py -population {pop}")
    config.check(refits_dir, pop)
    return path


def cached(name: str, build, force: bool = False) -> pl.DataFrame:
    path = os.path.join(OUTPUT_DIR, name)
    if os.path.exists(path) and not force:
        print(f"reusing {path}")
        return pl.read_parquet(path)
    df = build()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.write_parquet(path)
    print(f"wrote {path}")
    return df


def duck(memory_limit: str = "8GB") -> duckdb.DuckDBPyConnection:
    """A memory-capped duckdb connection, shared by every query builder below."""
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{memory_limit}'")
    return con


# --------------------------------------------------------------- shared GC bins

def gc_edges(gc: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """
    Fixed-width edges spanning the observed GC range, matching windows.bin_by_gc's
    "fixed" branch exactly. Returned explicitly rather than recomputed per consumer
    because three populations get binned on this axis -- genome-wide windows,
    per-(context, bin) expected counts aggregated in duckdb, and DNM training sites --
    and they are only comparable if the edges are identical.
    """
    edges = np.unique(np.linspace(float(np.min(gc)), float(np.max(gc)), n_bins + 1))
    edges[-1] += 1e-9
    return edges.astype(float)


def assign_bin(gc: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.digitize(np.asarray(gc, float), edges[1:-1]), 0, len(edges) - 2)


def sql_bin_expr(gc_expr: str, edges: np.ndarray) -> str:
    """duckdb equivalent of assign_bin, for grouping inside a query rather than
    materializing tens of millions of rows. Edges are uniform, so it is a clipped
    floor-divide -- no CASE ladder needed."""
    lo, hi, n = float(edges[0]), float(edges[-1]), len(edges) - 1
    width = (hi - lo) / n
    return f"LEAST(GREATEST(CAST(FLOOR(({gc_expr} - {lo!r}) / {width!r}) AS INTEGER), 0), {n - 1})"


def bin_centres(edges: np.ndarray, gc_bin) -> pl.Series:
    centres = 0.5 * (edges[:-1] + edges[1:])
    return pl.Series("gc_mid", [float(centres[i]) for i in gc_bin])


# ------------------------------------------------------------ panels A and E

def window_table(cache_dir: str = CACHE_DIR,
                 neutral_windows_bed: str | None = config.NEUTRAL_WINDOWS_BED
                 ) -> pl.DataFrame:
    """
    The analyzed window population: noncoding, pass_qc, autosome/PAR, with GC content
    as a 0-1 fraction -- and, if config.NEUTRAL_WINDOWS_BED is set, narrowed by a join
    to McHale et al.'s own 693,270 putatively neutral windows. This is both the test set
    Gnocchi is scored on and (in panels C-E) the population the retrained adjustment is
    fit on.

    The default comes from fig5/config.py, which fig5/refit.py reads too -- so the
    population fit on and the population scored on cannot drift apart. Do not pass this
    explicitly unless you also rerun the refits with the same value.
    """
    return W.build_window_table(cache_dir, neutral_windows_bed=neutral_windows_bed)


def rank_bias(binned: pl.DataFrame, label: str, min_n: int = 0) -> float:
    """
    Mean |mean rank - 0.5| over GC bins holding at least min_n windows: one number for
    "how GC-biased is this metric". Bins are unweighted. Pass the SAME min_n the panel
    plots with, or this summarizes bins the reader cannot see.
    """
    b = binned.filter(pl.col("n") >= min_n) if min_n else binned
    return float((b[f"mean_{label}"] - 0.5).abs().mean())


def rank_curves(df_win: pl.DataFrame, extra: list[tuple[str, str]] = (),
                min_n: int = 100):
    """
    The Fig. 2A rank statistic for the context-only model (r == 1), published Gnocchi,
    and any `extra` (label, expected-table path) refits, on ONE window population with
    ONE set of GC bins -- all curves are z-filtered jointly and ranked after that
    filter, so no curve is advantaged by its own filtering.
    """
    curves = [("step1", "expected_step1"), ("step2", "expected_step2")]
    for label, path in extra:
        col = f"expected_{label}"
        df_win = df_win.join(
            pl.read_csv(path, separator="\t").select(
                ["element_id", pl.col("expected").alias(col)]),
            on="element_id", how="inner")
        curves.append((label, col))

    df, binned = W.binned_rank_curves(df_win, curves=curves, n_bins=N_BINS)
    print(f"  {df.height:,} windows after joint z filtering")
    for label, _ in curves:
        print(f"  mean |rank - 0.5|  {label:<12} = {rank_bias(binned, label, min_n):.3f}"
              f"  (over bins with n >= {min_n:,})")
    return df, binned


# ------------------------------------------------------------------- panel B

def _r_eff_components(pop: str, cache_dir: str, refits_dir: str,
                      memory_limit: str) -> pl.DataFrame:
    """
    Per-window expected-count components, in the notation of fig5.ipynb's panel B cell:

        e1      E1(w)            sum over all 32 contexts, published step-1 table
        e2      E2(w)            the same sum after the refit's r, i.e. sum_t E1^t r_t
        e1_cpg  E1^K(w)          sum_t E1^t(w) over the four CpG contexts K
        e2_cpg  E2^K(w)          sum_t E1^t(w) r_t(w) over those same four

    Non-CpG is then a subtraction the caller does (e1_non = e1 - e1_cpg), which is why
    only the CpG slice of the two multi-GB per-context files is ever joined -- an
    85M x 85M join becomes 10M x 10M.

    That subtraction mixes two published files -- the totals come from the summed export,
    the CpG parts from the per-context one -- so it needs them to describe the same
    counts. They do: preconditions/verify_expected_r1.py regenerates the first from the
    second genome-wide, `possible` exactly and `expected` to 4.6e-5 relative.

    The published pipeline writes its per-context r to a local dir, not the bucket, so
    the refit's rr table stands in. That substitution is validated per GC bin in
    r_eff_by_gc, since the published r_eff is separately computable as E2/E1.
    """
    ctx = ", ".join(f"'{c}'" for c in M.CPG_CONTEXTS)
    percontext = W.download(M.GENOME_EXPECTED_PERCONTEXT_FILE, cache_dir)
    step1 = W.download(W.REMOTE_FILES["step1_expected"], cache_dir)
    query = f"""
        -- The CpG half of the split, built per (element_id, context) and summed back to
        -- one row per window: E1^K(w) = sum_t E1^t(w) and E2^K(w) = sum_t E1^t(w) r_t(w),
        -- both over t in K = the four CpG contexts.
        WITH cpg AS (
            SELECT e.element_id AS element_id,
                   SUM(e.expected) AS e1_cpg,
                   -- rr is missing for a context with no fitted model (no feature cleared
                   -- Bonferroni, or the fit did not converge). Those get r = 1, which is
                   -- what refit_and_apply's genome-wide apply does with them.
                   SUM(e.expected * COALESCE(r.rr, 1.0)) AS e2_cpg
            -- E1^t(w): the per-context step-1 export, one row per (window, context).
            FROM (SELECT element_id, context, expected
                  FROM read_csv_auto('{percontext}', delim='\t', header=True)
                  WHERE context IN ({ctx})) e
            -- r_t(w): per (window, context), from the refit. Chen et al. publish fitted
            -- .pkl models but never this table, which is why a refit supplies it.
            LEFT JOIN (SELECT element_id, context, rr
                       FROM read_csv_auto('{refit_path("rr", pop, refits_dir)}',
                                          delim='\t', header=True)
                       WHERE context IN ({ctx})) r
              ON e.element_id = r.element_id AND e.context = r.context
            GROUP BY e.element_id)
        -- The totals, already summed over all 32 contexts by whoever wrote each file, so
        -- no per-context arithmetic is repeated here for the 28 non-CpG ones.
        SELECT t1.element_id AS element_id, t1.expected AS e1, t2.expected AS e2,
               -- A window with no CpG-context row has E1^K = E2^K = 0, not NULL: it is
               -- entirely non-CpG, and must still contribute its e1/e2 to the bin.
               COALESCE(cpg.e1_cpg, 0.0) AS e1_cpg, COALESCE(cpg.e2_cpg, 0.0) AS e2_cpg
        -- E1(w): published, r == 1 (verify_expected_r1 is what establishes that).
        FROM read_csv_auto('{step1}', delim='\t', header=True) t1
        -- E2(w): the same windows after the refit's r. INNER, so a window missing from
        -- either side is dropped rather than silently scored against a partial numerator.
        INNER JOIN read_csv_auto('{refit_path("expected", pop, refits_dir)}',
                                 delim='\t', header=True) t2
          ON t1.element_id = t2.element_id
        LEFT JOIN cpg ON t1.element_id = cpg.element_id
    """
    return duck(memory_limit).execute(query).pl()


def r_eff_by_gc(df_win: pl.DataFrame, edges: np.ndarray, pop: str = "full",
                refits_dir: str = REFITS_DIR, cache_dir: str = CACHE_DIR,
                force: bool = False, memory_limit: str = "8GB") -> pl.DataFrame:
    """
    Panel B's table: r_eff = E2/E1 per GC bin, decomposed by CpG status.

    Aggregated as RATIOS OF SUMMED expected counts, not means of per-window ratios.
    Expected counts add, so sum(E2)/sum(E1) is the adjustment the bin actually
    receives -- and it keeps the decomposition r_eff = Pi*r_CpG + (1-Pi)*r_non exact
    bin by bin, which an average of per-window ratios would only satisfy approximately.

    Prints the published-vs-refit agreement in r_eff. Read it before trusting the
    CpG/non-CpG split: it is what licenses using the refit's per-context r at all.
    """
    comp = cached(f"r_eff_components.{pop}.parquet",
                  lambda: _r_eff_components(pop, cache_dir, refits_dir, memory_limit),
                  force)
    df = df_win.join(comp, on="element_id", how="inner")
    df = df.with_columns([
        (pl.col("e1") - pl.col("e1_cpg")).alias("e1_non"),
        (pl.col("e2") - pl.col("e2_cpg")).alias("e2_non"),
        pl.Series("gc_bin", assign_bin(df["GC_content"].to_numpy(), edges)),
    ])
    s = df.group_by("gc_bin").agg(
        [pl.len().alias("n"), pl.col("GC_content").mean().alias("gc_mid")]
        + [pl.col(c).sum().alias(c) for c in
           ("e1", "e2", "e1_cpg", "e2_cpg", "e1_non", "e2_non",
            "expected_step1", "expected_step2")]).sort("gc_mid")

    binned = s.with_columns([
        (pl.col("e2") / pl.col("e1")).alias("r_eff"),
        (pl.col("e2_cpg") / pl.col("e1_cpg")).alias("r_cpg"),
        (pl.col("e2_non") / pl.col("e1_non")).alias("r_non"),
        (pl.col("e1_cpg") / pl.col("e1")).alias("pi_cpg"),
        ((pl.col("e2_cpg") + pl.col("e1_non")) / pl.col("e1")).alias("r_counterfactual"),
        (pl.col("expected_step2") / pl.col("expected_step1")).alias("r_eff_published"),
    ])
    diff = (binned["r_eff"] - binned["r_eff_published"]).abs()
    print(f"refit validation: max |r_eff(refit) - r_eff(published)| over "
          f"{binned.height} GC bins = {float(diff.max()):.2e} "
          f"(median {float(diff.median()):.2e})")
    return binned


# --------------------------------------------- panel C, and Supporting Figure 7
#
# Both of panel C's rows are views of ONE table, dnm_rate_by_stratum() below: the upper
# row is its per-stratum site counts (a composition -- how much of the training set sits
# outside the scored population) and the lower row is its per-stratum DNM rates (whether
# the territory outside is also DIFFERENT). One query, so the two rows cannot end up
# describing different sites.
#
# Both rows count BOTH training classes, DNMs and background alike. The upper row counted
# background sites only until 2026-08-14, on the reasoning that the background class is
# what carries the covariate distribution in a case-control design; that is true of the
# design but not of the fit, which minimizes its loss over the mixture. The mixture is
# therefore the training distribution, and it is the training distribution that the panel
# is comparing against the scored one.

# The strata a training site can fall into, relative to the scored population.
#
# THE FIRST ONE IS DEFINED BY MEMBERSHIP, not by re-deriving the window filters here.
# `scored` means the site's 1 kb window is a row of the analyzed window table --
# windows.build_window_table, which carries the coding restriction, the QC filter, the
# autosome/PAR restriction and, once config.NEUTRAL_WINDOWS_BED is set, the join down to
# McHale et al.'s 693,270 neutral windows. That table is also what
# dnm_model.restrict_to_analyzed_windows filters the training set with, so `scored` here
# is exactly "survives the panel D/E intervention".
# Re-deriving those filters in SQL is how this panel silently stops describing the
# population the retrained model is fit and scored on: until 2026-08-17 the first stratum
# tested `an.coding_prop <= 0.0` and so would have counted windows outside the neutral
# set as inside the scored population the moment that file was supplied.
#
# The other three name the REASON a site is outside it, in stacking order:
#   coding      scored by Chen et al., but the window overlaps coding exons
#   non_neutral noncoding, QC-pass, and in the constraint table, but NOT in McHale et
#               al.'s neutral set -- the territory dropped in going from 1,843,559
#               windows to their 693,270 (enhancer-overlapping windows, plus their
#               assembly-gap / ENCODE-exclude / low-coverage exclusions; the file does
#               not say which, and this band does not need to). Necessarily empty while
#               config.NEUTRAL_WINDOWS_BED is None, and an empty stratum draws no band
#               and no legend entry (panels.py). It is the band to read when asking
#               whether the figure's conclusions survive on their window set: if the
#               removed territory has the scored population's own DNM rate, restricting
#               to it costs sample size and nothing else.
#   failed_qc   no row in the constraint table at all, so it has no coding_prop to test.
#               `failed_qc` and not `no_coverage`: every absent window has its QC inputs
#               on file and fails one of the paper's three conditions (>= 80% of observed
#               variants PASS, mean coverage 25-35x, >= 1000 possible variants), the
#               first dominating. preconditions/verify_qc_filter.py measures the split.
#
# The order of the CASE arms is load-bearing: membership is tested first, so a window in
# the analyzed table can never be relabelled by one of the reason arms below it.
_STRATA = ("scored", "coding", "non_neutral", "failed_qc")


def _stratum_expr() -> str:
    """
    Panel C's CASE expression. `sw` is the analyzed window table registered as a duckdb
    relation by dnm_rate_by_stratum; `an` is the published constraint table.

    The coding arm reads its threshold from windows.NONCODING_MAX_CODING_PROP -- the same
    constant restrict_to_noncoding filters on -- rather than repeating a literal, so
    "overlaps coding exons" means one thing in this figure.
    """
    return f"""CASE WHEN sw.element_id IS NOT NULL THEN 'scored'
                    WHEN an.element_id IS NULL THEN 'failed_qc'
                    WHEN an.coding_prop > {W.NONCODING_MAX_CODING_PROP!r} THEN 'coding'
                    ELSE 'non_neutral' END"""

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
        eid=ELEMENT_ID_FROM_LOCUS,
        dnm1=W.download(M.TRAINING_FILES["dnm1_sites"], cache_dir),
        dnm0=W.download(M.TRAINING_FILES["dnm0_sites"], cache_dir))


def _binned_training_query(cache_dir: str, edges: np.ndarray, where: str,
                           dims: list[tuple[str, str]] = (), aggs: str = "",
                           extra_joins: str = "") -> str:
    """
    Training sites joined to their 1 kb tile's GC and constraint annotation, aggregated
    per GC bin (plus any extra grouping `dims`, each an (expression, alias) pair).
    `aggs` adds further aggregate select items. `extra_joins` appends further join
    clauses, for a relation the caller registered on its own connection -- panel C
    joins the analyzed window table that way, rather than re-deriving it from `an`.

    GROUP BY repeats the expressions rather than using positional indices: the two
    callers have different select-list shapes, and positional GROUP BY silently grouped
    by the wrong column when this was first written.
    """
    gc_bin = sql_bin_expr("ft.GC_content_1k / 100.0", edges)
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
               {extra_joins}
        WHERE {where}
        GROUP BY {group_by}
    """


def _fingerprint(edges: np.ndarray, df_win: pl.DataFrame | None = None) -> str:
    """
    Six hex characters standing for "these GC edges over this window population", for a
    cache key. `df_win` is omitted by builders that bin the whole training population
    rather than a window set -- they still depend on the edges, which move with it.

    Both inputs move when config.NEUTRAL_WINDOWS_BED changes -- the window set directly,
    the edges because gc_edges spans its GC min and max -- and neither is visible in the
    old `{n}bins` key, so a cached table built under one setting would be silently
    reused under the other. It also keeps the two window sets' tables side by side in
    fig5/output/ instead of one overwriting the other. Order-independent (xor over per-id hashes), and a polars
    version bump can only cost a rebuild, never a wrong answer.
    """
    h = hashlib.blake2s(np.asarray(edges, float).tobytes(), digest_size=3)
    if df_win is not None:
        ids = df_win["element_id"]
        h.update(f"{ids.len()}:"
                 f"{int(np.bitwise_xor.reduce(ids.hash(seed=0).to_numpy()))}".encode())
    return h.hexdigest()


def dnm_rate_by_stratum(edges: np.ndarray, df_win: pl.DataFrame,
                        cache_dir: str = CACHE_DIR, force: bool = False,
                        memory_limit: str = "10GB") -> pl.DataFrame:
    """
    Empirical P(DNM) over non-CpG training sites, per GC bin, split by where the site
    sits relative to the scored population: inside it, or outside it because the window
    is coding, outside McHale et al.'s neutral set, or absent from the constraint table
    for failing gnomAD variant-call QC. See _STRATA above for what defines each.

    `df_win` is the analyzed window table (data.window_table). It is required, and it is
    the SAME frame the panels are evaluated on -- passing a different one would label
    sites against a population no panel uses.

    Both classes are labelled, DNMs included -- that is what makes this a rate rather
    than a composition. 72,801 of the non-CpG autosomal DNMs sit in the QC-failing
    stratum, against 17,545 coding and 241,479 in the scored population.

    THE POINT. The scored and coding curves are both nearly flat in GC and nearly
    equal, so the coding exclusion is not what makes the training set's GC dependence
    steep. The QC-failing curve is not flat: it runs ~1.6x the scored rate in the GC
    bulk and ~4.1x by GC 0.61. Essentially all of the original training set's GC
    dependence is contributed by sequence gnomAD could not call reliably -- which is also
    where trio DNM calling is least reliable, so part of the excess is plausibly
    false-positive DNM calls rather than real mutation.

    Columns: stratum, gc_bin, gc_pct, k (DNMs), n (sites), p = k/n.
    """
    def build():
        con = duck(memory_limit)
        # Registered rather than written out: duckdb reads the polars frame in place, so
        # the analyzed window set enters the query as itself, not as a re-derivation.
        con.register("scored_windows", df_win.select("element_id"))
        q = _binned_training_query(
            cache_dir, edges, dims=[(_stratum_expr(), "stratum")],
            extra_joins="LEFT JOIN scored_windows sw ON s.element_id = sw.element_id",
            where=f"s.context NOT IN ({', '.join(repr(c) for c in M.CPG_CONTEXTS)})")
        return con.execute(q).pl()

    df = cached(f"dnm_rate_by_stratum.{len(edges) - 1}bins."
                f"{_fingerprint(edges, df_win)}.parquet", build, force)
    return df.with_columns((pl.col("k") / pl.col("n")).alias("p")).sort(["stratum", "gc_bin"])


def training_composition(st: pl.DataFrame, edges: np.ndarray) -> pl.DataFrame:
    """
    Panel C's upper row: each GC bin's non-CpG training sites split by stratum, as counts
    and as fractions of the bin. `st` is dnm_rate_by_stratum() output, whose `n` is
    exactly this count -- so the composition is a reshape of the table the lower row takes
    its rates from, not a second query that could drift from it.

    Both training classes are counted, DNMs and background alike. Restricted to the
    background class (n - k) this reproduces the standalone dnm0-only query it replaced
    exactly, bin by bin and stratum by stratum, on all 20 bins; what the mixture adds is
    the DNM class's own, steeper drift out of the scored population.

    The strata partition the bin by construction -- _stratum_expr() is a CASE expression,
    so a site lands in exactly one -- which is why nothing is asserted here.

    Columns: gc_bin, gc_mid, n_total, n_{stratum}, frac_{stratum}, for every stratum in
    _STRATA including any that is empty genome-wide (`non_neutral`, while
    NEUTRAL_WINDOWS_BED is None) -- the shape does not depend on the configuration, and panels.py drops a band
    that is zero everywhere rather than drawing an invisible one with a legend entry.
    """
    # A GC bin can be missing a stratum entirely (the lowest two hold no coding sites),
    # which pivots to null rather than to an absent row. A stratum missing from EVERY bin
    # has no column at all, hence the explicit zero fill below.
    wide = (st.pivot(values="n", index="gc_bin", on="stratum", aggregate_function="first")
              .fill_null(0).sort("gc_bin"))
    absent = [s for s in _STRATA if s not in wide.columns]
    if absent:
        wide = wide.with_columns([pl.lit(0, dtype=pl.Int64).alias(s) for s in absent])
    df = wide.with_columns([
        bin_centres(edges, wide["gc_bin"]),
        pl.sum_horizontal([pl.col(s) for s in _STRATA]).alias("n_total"),
    ])
    df = df.with_columns(
        [(pl.col(s) / pl.col("n_total")).alias(f"frac_{s}") for s in _STRATA]
    ).rename({s: f"n_{s}" for s in _STRATA})
    # Deliberately no fraction range printed here: the lowest-GC bins hold a handful of
    # sites that are essentially all QC-failing, so an unrestricted min() reads 0.00 and
    # would get copied into a caption. The notebook reports the range over plotted bins.
    print(f"training composition: {int(df['n_total'].sum()):,} non-CpG training sites "
          f"over {df.height} GC bins")
    return df


def stratum_ratios(st: pl.DataFrame, edges: np.ndarray, min_n: int = 2000) -> pl.DataFrame:
    """
    dnm_rate_by_stratum() reshaped to the ratios panel C plots: each excluded stratum's
    non-CpG DNM rate over the scored population's, per GC bin.

    Ratios rather than raw rates because the question is comparative -- is the excluded
    territory DIFFERENT from the territory Gnocchi is scored on -- and because the
    scored population's own rate drifts mildly with GC, which a ratio divides out.

    Error bars are the delta-method SE of log(ratio), SE = sqrt((1-p_a)/k_a +
    (1-p_b)/k_b), i.e. binomial noise in both strata. min_n drops bins where either
    stratum holds fewer than that many sites.

    Columns: gc_bin, gc_mid, and {stratum}_{ratio,se_log} for each excluded stratum that
    has any bin left after min_n -- so an empty `non_neutral` stratum contributes no
    columns rather than a column of nulls, and panels.py plots whichever it finds.
    """
    keep = st.filter(pl.col("n") >= min_n)
    base = keep.filter(pl.col("stratum") == "scored").select(
        ["gc_bin", pl.col("p").alias("p_nc"), pl.col("k").alias("k_nc")])
    out = base
    for stratum in [s for s in _STRATA if s != "scored"]:
        s = keep.filter(pl.col("stratum") == stratum).select(
            ["gc_bin", pl.col("p").alias("p_s"), pl.col("k").alias("k_s")])
        if s.height == 0:
            continue
        out = out.join(s, on="gc_bin", how="inner").with_columns([
            (pl.col("p_s") / pl.col("p_nc")).alias(f"{stratum}_ratio"),
            (((1 - pl.col("p_s")) / pl.col("k_s")
              + (1 - pl.col("p_nc")) / pl.col("k_nc")).sqrt()).alias(f"{stratum}_se_log"),
        ]).drop(["p_s", "k_s"])
    out = out.drop(["p_nc", "k_nc"]).sort("gc_bin")
    return out.with_columns(bin_centres(edges, out["gc_bin"]))


# ------------------------------------------------------------------- panel D

def dnm_probability(populations=("full", "scored", "sizematched"), n_bins: int = N_BINS,
                    refits_dir: str = REFITS_DIR, min_n: int = 500) -> dict:
    """
    Panel D's tables: per-GC-bin fitted and empirical P(DNM) over the non-CpG training
    sites of each population, from the per-site predictions fig5/refit.py wrote.

    Binned on SHARED edges across populations -- they have different GC ranges, so
    letting each pick its own linspace would misalign them. GC here is in this repo's
    native 0-100 percent units (the GC_content_1k regional feature at the site);
    panel_dnm_probability_pairs divides by 100.

    The empirical column is the fraction of that bin's training examples that are
    DNMs, so its level reflects the case-control design (dnm0:dnm1 ~ 10:1), not the
    genome-wide DNM rate. Only shape is interpretable across populations.
    """
    preds = {}
    for pop in populations:
        df = pd.read_csv(refit_path("predictions", pop, refits_dir), sep="\t")
        df = df[~df["context"].isin(M.CPG_CONTEXTS)]
        n1 = int(df["label"].sum())
        print(f"{pop:<12} non-CpG: {len(df):,} sites, {n1:,} DNMs, "
              f"{len(df) / max(n1, 1) - 1:.1f} background per DNM")
        preds[pop] = df

    gc_all = np.concatenate([d["gc"].to_numpy() for d in preds.values()])
    edges = np.linspace(gc_all.min(), gc_all.max(), n_bins + 1)
    edges[-1] += 1e-9

    out = {}
    for pop, df in preds.items():
        df = df.assign(bin=assign_bin(df["gc"].to_numpy(), edges))
        b = df.groupby("bin").agg(n=("label", "size"), n1=("label", "sum"),
                                  gc_mid=("gc", "mean"),
                                  mean_pred=("pred", "mean")).reset_index()
        b["empirical_prop"] = b["n1"] / b["n"]
        b["se"] = np.sqrt(b["empirical_prop"] * (1 - b["empirical_prop"]) / b["n"])
        out[pop] = b[b["n"] >= min_n] if min_n else b
    return out


# ------------------------------------------------------ Supporting Figure 7
#
# Why r_CpG ~ 1 is CORRECT and not a failure: the effect that would need adjusting is
# already applied in step 1. These two builders measure its size and its GC dependence;
# the fourth row of that figure, Pi, comes from r_eff_by_gc above rather than here.

def cpg_methylation_by_gc(edges: np.ndarray, cache_dir: str = CACHE_DIR,
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
    the claim is about CpG biology, not about the scored population. The GC bin EDGES
    still come from the analyzed windows, though, which is why the cache key
    fingerprints them.

    Columns: gc_bin, gc_pct, n, k, p (DNM rate), mean_methyl, frac_hypomethylated.
    """
    def build():
        q = _binned_training_query(
            cache_dir, edges,
            aggs=("AVG(CAST(s.methyl_level AS DOUBLE)) AS mean_methyl, "
                  "AVG(CASE WHEN s.methyl_level <= 1 THEN 1.0 ELSE 0.0 END) "
                  "AS frac_hypomethylated"),
            where=f"s.context IN ({', '.join(repr(c) for c in M.CPG_CONTEXTS)})")
        return duck(memory_limit).execute(q).pl()

    df = cached(f"cpg_methylation_by_gc.{len(edges) - 1}bins."
                f"{_fingerprint(edges)}.parquet", build, force)
    return df.with_columns((pl.col("k") / pl.col("n")).alias("p")).sort("gc_bin")


def cpg_rate_by_methyl(cache_dir: str = CACHE_DIR) -> pl.DataFrame:
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
