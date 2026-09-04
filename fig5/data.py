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
from sklearn.metrics import auc, precision_recall_curve

# First-party. `config` is a sibling module, not a third-party package -- keep it
# grouped with gnocchi_bias, and do not let an isort autofix hoist it above.
import config
from gnocchi_bias import dnm_model as M
from gnocchi_bias import windows as W

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
OUTPUT_DIR = os.path.join(HERE, "output")     # figures and this figure's own caches
CACHE_DIR = W.CACHE_DIR      # downloaded bucket files; $GNOCCHI_PUBLISHED_DIR moves them
REFITS_DIR = os.path.join(REPO_ROOT, "refits")  # the shared refit outputs

N_BINS = 20
XRANGE = (0.2, 0.73)   # read off McHale et al. Fig. 2A by eye, at 300 DPI; approximate,
                       # not a value their text states -- METHODS.md, "Axis ranges"

# Written by fig5/refit.py into REFITS_DIR; `pop` is full / scored / sizematched, carrying
# config.WINDOW_SET_SUFFIX so both window sets' refits coexist (config.tagged()).
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
    path = os.path.join(refits_dir, REFIT_FILES[kind].format(pop=config.tagged(pop)))
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
    # config.tagged, not a bare `pop`, and today that changes NOTHING: panel B is the only
    # caller, it passes pop="full", and `full` is not WINDOW_DEPENDENT -- its refit never
    # builds the window table, so its tables are identical under either window set and one
    # cache correctly serves both. The tag matters only if this is ever called with
    # "scored" or "sizematched", whose content DOES move with config.NEUTRAL_WINDOWS_BED.
    # It would move silently: cached() short-circuits before the builder runs, so
    # refit_path -- and therefore config.check -- is never reached on a cache hit. Naming
    # the cache the way the refit it is built from is named closes that off in advance.
    comp = cached(f"r_eff_components.{config.tagged(pop)}.parquet",
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
# THE FIRST STRATUM IS ASSIGNED BY A LOOKUP, NOT BY A TEST. A site is `scored` iff its
# 1 kb window has a row in the analyzed window table -- the CASE arm below is a join on
# element_id (`sw.element_id IS NOT NULL`), and nothing in this file re-states the
# conditions that put the window in that table.
#
# Those conditions live in one place, windows.build_window_table, and they are not fixed:
# with config.NEUTRAL_WINDOWS_BED unset the table is the coding restriction + QC filter +
# autosome/PAR restriction; with it set the table is McHale et al.'s own 693,270-window
# file, with none of those applied on top. The same table is what
# dnm_model.restrict_to_analyzed_windows filters the training set with, so joining
# against it makes `scored` here mean exactly "survives the panel D/E intervention" under
# whichever of the two definitions is in force -- automatically, with no second copy of
# the definition to keep in step.
#
# The remaining three strata are for sites OUTSIDE the scored population, and each
# names the reason it is outside. In stacking order, bottom to top:
#   coding          scored by Chen et al., but the window overlaps coding exons -- and,
#                   once NEUTRAL_WINDOWS_BED is set, is outside their set too, since a
#                   coding window their file lists is tested by the `scored` arm first
#                   and kept.
#   other_noncoding QC-pass and noncoding and in the constraint table, but NOT in McHale
#                   et al.'s window set -- the rest of the QC-pass noncoding territory,
#                   the part given up in going from 1,843,559 windows to their 693,270
#                   (enhancer-overlapping windows, plus their assembly-gap /
#                   ENCODE-exclude / low-coverage exclusions; the file does not say
#                   which, and this band does not need to).
#                   NAMED FOR WHERE IT SITS, NOT FOR WHAT IT IS. It was `non_neutral`
#                   until 2026-08-18, which asserted more than the data does: these
#                   windows are outside a set McHale et al. call putatively neutral,
#                   which is not evidence that they are under selection. Whether they
#                   differ from the scored population at all is the open question this
#                   band exists to answer -- if the given-up territory has the scored
#                   population's own DNM rate, restricting to their set costs sample
#                   size and nothing else -- so the name must not presume the answer.
#                   Necessarily empty while config.NEUTRAL_WINDOWS_BED is None, and an
#                   empty stratum draws no band and no legend entry (panels.py). Note
#                   the asymmetry with `coding` above: this arm is reached only by
#                   windows their file does not list at all.
#   failed_qc       no row in the constraint table at all, so no coding_prop to test.
#                   `failed_qc` and not `no_coverage`: every absent window has its QC
#                   inputs on file and fails one of the paper's three conditions (>= 80%
#                   of observed variants PASS, mean coverage 25-35x, >= 1000 possible
#                   variants), the first dominating. preconditions/verify_qc_filter.py
#                   measures the split.
#
# This tuple's order is the DRAWING order -- bottom to top in panel C's stacked bars,
# matching panels.COMPOSITION_STYLE. It is not the order the CASE arms are tested in;
# see _stratum_expr for that.
_STRATA = ("scored", "coding", "other_noncoding", "failed_qc")


def _stratum_expr() -> str:
    """
    Panel C's CASE expression. `sw` is the analyzed window table registered as a duckdb
    relation by dnm_rate_by_stratum; `an` is the published constraint table.

    The coding arm reads its threshold from windows.NONCODING_MAX_CODING_PROP -- the same
    constant restrict_to_noncoding filters on -- rather than repeating a literal, so
    "overlaps coding exons" means one thing in this figure.

    The genome's 1 kb windows partition three ways (top row). `sw` is drawn over that
    partition twice, once per setting of config.NEUTRAL_WINDOWS_BED, with the stratum
    each region gets underneath it:

    |<------------------ QC-pass: a row in `an` ------------------->||<- QC-fail ->|
    +-----------------------------------------+---------------------+--------------+
    |                noncoding                |        coding       |no row in `an`|
    +-----------------------------------------+---------------------+--------------+

    |<------------ sw, BED unset ------------>|
    |            scored (1,843,559)           |        coding       |  failed_qc   |

    |<--- sw, BED set ---->|                  |<- * ->|
    |   scored (693,270)   | other_noncoding  | scored|    coding   |  failed_qc   |

    Unset, `sw` IS the noncoding cell -- build_window_table produces it by filtering on
    coding_prop and QC, so the two coincide and every arm below `scored` is reached only
    by windows outside it. Set, `sw` is McHale et al.'s file taken whole, and it respects
    neither internal boundary: it covers part of the noncoding cell (the rest becomes
    `other_noncoding`, empty and undrawn in the unset case) and MAY reach into the coding
    cell (*), since nothing filters their file on coding_prop.

    THAT OVERLAP IS WHY THE `scored` ARM MUST COME FIRST. A window in (*) satisfies both
    `sw.element_id IS NOT NULL` and `an.coding_prop > threshold`; panels D/E fit and score
    on it, so `scored` is its true label, and testing membership before coding is what
    makes this CASE agree with them. windows.restrict_to_mchale_neutral_windows prints
    how many such windows there are -- 0 means (*) is empty and the two drawings differ
    only inside the noncoding cell. (693,270 is their file's own row count, before the
    three-way join drops windows with no constraint/expected/features row.)
    """
    return f"""CASE WHEN sw.element_id IS NOT NULL THEN 'scored'
                    WHEN an.element_id IS NULL THEN 'failed_qc'
                    WHEN an.coding_prop > {W.NONCODING_MAX_CODING_PROP!r} THEN 'coding'
                    ELSE 'other_noncoding' END"""

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
                           extra_group_by: list[tuple[str, str]] = (), aggs: str = "",
                           extra_joins: str = "") -> str:
    """
    Training sites joined to their 1 kb tile's GC and constraint annotation, aggregated
    per GC bin. `extra_group_by` adds further grouping keys alongside the GC bin, each an
    (expression, alias) pair that goes into BOTH the select list and the GROUP BY -- panel
    C passes one, the stratum CASE expression. `aggs` adds further aggregate select items.
    `extra_joins` appends further join clauses, for a relation the caller registered on
    its own connection -- panel C joins the analyzed window table that way, rather than
    re-deriving it from `an`.
    """
    gc_bin = sql_bin_expr("ft.GC_content_1k / 100.0", edges)
    key_select = "".join(f"{expr} AS {alias}, " for expr, alias in extra_group_by)
    # GROUP BY names the select aliases -- duckdb resolves them -- so each grouping
    # expression is written once. Safe only while no alias collides with a column of
    # `s`, `ft`, `an` or a registered relation: a collision binds to the column instead.
    group_by = ", ".join([alias for _, alias in extra_group_by] + ["gc_bin"])
    return f"""
        WITH an AS (SELECT element_id, pass_qc, coding_prop
                    FROM read_csv_auto('{W.download(W.REMOTE_FILES["annot"], cache_dir)}',
                                       delim='\t', header=True)),
        ft AS (SELECT element_id, GC_content_1k
               FROM read_csv_auto('{W.download(W.REMOTE_FILES["features"], cache_dir)}',
                                  delim='\t', header=True)),
        s AS ({_training_sql(cache_dir)})
        SELECT {key_select}{gc_bin} AS gc_bin,
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
            cache_dir, edges, extra_group_by=[(_stratum_expr(), "stratum")],
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
    _STRATA including any that is empty genome-wide (`other_noncoding`, while
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

    `{stratum}_se_log` is the delta-method SE of log(ratio), SE = sqrt((1-p_a)/k_a +
    (1-p_b)/k_b), i.e. binomial noise in both strata (Var(log p_hat) = Var(p_hat)/p^2 =
    (1-p)/(n p) = (1-p)/k, with k the DNM count). It is the SE of the LOG ratio and of
    nothing else, which is why panel C plots log(ratio) on a linear axis and draws the
    bars as a plain +/- se there. min_n drops bins where either stratum holds fewer than
    that many sites.

    Columns: gc_bin, gc_mid, and {stratum}_{ratio,se_log} for each excluded stratum that
    has any bin left after min_n -- so an empty `other_noncoding` stratum contributes no
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


# --------------------------------------------------------- Supporting Figure 8

# WHAT THIS SECTION IS FOR. Panel E says the retrained score is no longer GC-biased. It
# cannot say whether the biased score was nevertheless the better DETECTOR: bias and
# signal-to-noise act on discovery jointly (McHale et al.'s Fig. 3), and only one of the
# two has been changed. Supporting Figure 8 is that test, built as their Fig. 4A/B -- a
# classifier that calls a window constrained when its Gnocchi z exceeds a threshold, and
# performance read off the precision-recall curve within each GC bin -- with TWO GNOCCHI
# VARIANTS in place of their four constraint metrics.
#
# LAX AND STRINGENT, WHICH IS MCHALE ET AL.'S OWN VOCABULARY AND THE AXIS THESE NAMES ARE
# ORGANIZED ON. A truth set says which windows are "constrained", and their paper uses two:
#
#   LAX        a window is constrained if it overlaps a GeneHancer enhancer. Large enough
#              to resolve performance deep in the GC tails, and lax because not every
#              enhancer-overlapping window is under strong selection -- GeneHancer covers
#              18.4% of the noncoding genome while perhaps 4.51% is under human-specific
#              selection. Their Fig. 4A/B. THIS IS WHAT IS IMPLEMENTED HERE.
#   STRINGENT  noncoding windows conjectured to be under strong negative selection because
#              they regulate essential genes, plus an equal number overlapping no enhancer
#              at all. Small, so noisy in the tails, but a post-hoc validation of the lax
#              set: overall performance rises on moving to it. Their Fig. 4C/D, from
#              papers/neutral_models_are_biased/11.compare-lax-with-stringent-truth-set.ipynb.
#              NOT BUILT YET -- and what that will take is written down below, because two
#              things about it are not guessable from this section.
#
# WHAT BUILDING THE STRINGENT SET WILL ACTUALLY TAKE (read from that notebook, 2026-09-03).
#
#   1. A THIRD HAND-SUPPLIED FILE, not a relabelling of the lax windows:
#      {CONSTRAINT_TOOLS_DATA}/stringent_truth_set/truth-set.gnocchi.lambda_s.depletion_rank.CDTS.bed
#      4,933 rows, columns `chromosome, start, end, gnocchi, truly constrained, B,
#      B_M1star.EUR, GC_content_1000bp, lambda_s,
#      depletion_rank_constraint_score_complement,
#      percentile_rank_of_observed_minus_expected_complement`. Note the target is
#      `truly constrained`, and the coordinate column is `chromosome` where the lax file
#      says `chrom` -- so it needs its own config entry and its own preflight check.
#
#   2. ITS INTERVALS ARE NOT ON CHEN'S 1 kb GRID, and this is the load-bearing problem for
#      US specifically. The positives are enhancer intervals of arbitrary length and offset
#      (chr1-2128961-2129161 is 200 bp; chr1-6240740-6241540 is 800 bp); only the negatives
#      are 1 kb tiles. Every other truth set here is joined by building an element_id from
#      chrom-start-end, and that cannot work: the retrained score exists per Chen 1 kb
#      window, so mapping a stringent interval onto it needs an INTERVAL OVERLAP plus a
#      rule for an interval spanning two tiles. Their file's own `gnocchi` column was
#      presumably carried over by exactly such an intersection (cell 9 says so explicitly
#      for lambda_s), so the first job is to establish what rule they used and reuse it --
#      not to invent one, or the retrained and published scores would be mapped
#      differently and the comparison would be meaningless.
#
#   3. FIG. 4C/D ARE A DIFFERENT STATISTIC FROM THIS SECTION'S, so they are new builders
#      rather than another value threaded through pr_curves():
#        - bootstrap: resample the truth set WITH replacement (sample(frac=1, replace=True)),
#          cut into exactly TWO feature bins, then the same class balancing as here;
#        - per replicate compute auPRCnorm = auc/r pooled, and
#          delta = (auc[low bin] - auc[high bin]) / auc[pooled];
#        - 1,000 replicates; report mean and sd of each. 4C is the auPRCnorm bar chart,
#          4D the delta one.
#        - bins are pairs, not a sweep: GC (0.20, 0.375) and (0.40, 0.70); BGS (0.5, 0.76)
#          and (0.9, 1.0); gBGC (-0.3, 0.2) and (0.4, 1.2).
#        - their bin floor there is 500, not LAX_MIN_BIN_WINDOWS, and a bin under it raises,
#          which skips that truth set for that feature entirely.
#        - the lax set is first .sample(n=len(stringent)) so the two are size-matched; that
#          is what makes 4C a comparison of truth sets rather than of sample sizes.
#
# So the constants that are properties of a truth set carry its name (LAX_GC_BINS,
# LAX_MIN_BIN_WINDOWS) and the ones that are not do not (PR_SCORES, TRUTH_TARGET);
# pr_curves() takes `truth_set` and accepts only "lax" so far. Adding the stringent set
# should mean adding names beside these, never renaming them. Do not reach for "enhancer"
# in a name here: enhancer overlap is how the LAX set is defined, not what this section is.
#
# Reference implementation for the lax set, followed for the bins, the balancing, the
# bin-size floor and the trapezoidal auc(recall, precision):
# constraint-tools papers/neutral_models_are_biased/7.CDTS/main.2.ipynb.
#
# THE LAX TRUTH SET IS GENEHANCER AND THERE IS NO SUBSTITUTE FOR IT. `window overlaps
# enhancer` in config.NEUTRAL_WINDOWS_BED -- licensed, not redistributable, not derivable
# from the public bucket, and nothing in Chen et al.'s annotation table is the same truth
# set under another name. So UNLIKE every other quantity here, this one does not build
# without that file: pr_curves raises, and the notebook checks and skips.

# The tail bins of LAX_GC_BINS merged into one, for the paired-difference panel. McHale et
# al.'s window file is nearly empty above GC 0.60 -- after class balancing the three bins
# there hold 1,086, 65 and 2 windows -- so drawing them separately says nothing and drawing
# none of them throws the tail away. One (0.55, 0.80] bin is the honest use of it, and the
# tail is where the whole question lives, since that is where panel E's bias reduction is
# largest. The lower bins are LAX_GC_BINS's, unchanged, so the two panels' x axes line up
# everywhere below 0.55.
DELTA_GC_BINS = [(0.20, 0.30), (0.30, 0.40), (0.40, 0.50), (0.50, 0.55), (0.55, 0.80)]

# A bin thinner than this is dropped from the difference panel. FAR below
# LAX_MIN_BIN_WINDOWS, and legitimately so: that floor guards a per-bin precision-recall
# CURVE, which is a staircase when windows are few, whereas this panel draws one number per
# bin with a bootstrap interval that widens honestly as the bin thins. The interval is the
# guard, so the floor only has to exclude bins where the bootstrap itself is degenerate.
DELTA_MIN_BIN_WINDOWS = 500

# The reference notebook's GC bins for the lax set, verbatim -- wide at the ends where
# windows are scarce, narrow through the bulk where the performance trend actually turns.
# NOT the N_BINS fixed-width edges the other panels share: after the class balancing below,
# 20 equal bins would leave almost all of them under the window floor, and a
# precision-recall curve needs far more windows per bin than a conditional mean rank does.
# Reused rather than re-derived so the panel's x axis is comparable to McHale et al.'s
# Fig. 4B. LAX_, because the stringent set is orders of magnitude smaller and will need its
# own, coarser edges rather than these.
LAX_GC_BINS = [(0.20, 0.30), (0.30, 0.40), (0.40, 0.50), (0.50, 0.55),
               (0.55, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 0.80)]

# A GC bin thinner than this is dropped before any curve is drawn: the notebook's
# threshold for the lax set, and it is doing real work at both ends of the axis, where a
# precision-recall curve built on a few hundred windows is mostly staircase. LAX_ for the
# same reason as the bins -- a truth set of a few thousand windows cannot clear 4,000 in
# any bin, so the stringent set will need its own floor, not a re-tuning of this one.
LAX_MIN_BIN_WINDOWS = 4_000

# The two curves. NOT truth-set specific -- the same two scores are evaluated against
# whichever truth set is in play, which is the whole point of having more than one.
# key -> (expected-count column, panel title, legend word). Panel A gives
# each score a whole axes and could afford the full name, but its two axes sit either side
# of panel B's letter and the long one runs into it; the short word is also what panel B's
# legend has room for, so the figure names each score once and identically. Both strings
# travel in the curve dicts, which is what keeps panels.py from having to import this
# module.
PR_SCORES = {
    "published": (W.PUBLISHED_EXPECTED_COL, "Gnocchi (published)", "published"),
    "scored": ("expected_scored", "Gnocchi (decontaminated training set)",
               "decontaminated"),
}

# The label column, whichever truth set produced it -- for the lax set it is
# windows.MCHALE_ENHANCER_FLAG renamed. Every truth set writes this one column, so the
# binning, the balancing and the curves never have to know which one they are working on,
# and adding the stringent set adds a builder rather than a code path.
TRUTH_TARGET = "constrained"


def _lax_labelled_windows(cache_dir: str, neutral_windows_bed: str | None,
                          refit_expected: str | None) -> pl.DataFrame:
    """
    The LAX truth set: one row per evaluated window -- element_id, GC_content (0-1),
    `constrained` (does it overlap a GeneHancer enhancer), and one z column per entry of
    PR_SCORES.

    The stringent set gets its own builder beside this one, returning the same columns, so
    everything downstream of here is shared.

    IT IS window_table()'s PIPELINE WITH ONE FILTER DROPPED. The population comes from the
    same windows.build_window_table() call, on the same config.NEUTRAL_WINDOWS_BED, with
    `keep_enhancer_windows=True` -- so their file is still the definition, this repo's
    noncoding/QC/autosome filters are still skipped, and the GC units and the join are
    still theirs. The only difference is that the `enhancer == False` step does not run
    and the flag comes back as a column instead. Panel E must NOT have those windows (a
    window under selection has a low z for a reason that is not bias); this figure cannot
    do without them (they are the positive class). Nothing else about the two populations
    differs, which is what makes the figure a statement ABOUT panel E rather than about a
    different window set.

    Note which population is which: the `scored` refit is still FIT on the putatively
    neutral windows alone -- that is the intervention -- and is EVALUATED here on neutral
    AND enhancer windows. Fit on the negatives, scored on both, which is what a classifier
    requires, and what the caption should say.

    Both z columns are filtered JOINTLY to the pipeline's [-10, 10] (windows.
    filter_z_in_range), so the two scores describe one identical set of windows and
    neither is advantaged by its own filtering.
    """
    if not neutral_windows_bed:
        raise ValueError(
            "Supporting Figure 8 needs McHale et al.'s window file: it carries the "
            "GeneHancer enhancer flag, which IS the truth set, and there is no substitute "
            "for it in the public bucket. Set NEUTRAL_WINDOWS_BED in fig5/config.py -- it "
            "is on the constraint-tools HPC path, which is where this figure is built.")

    df = W.build_window_table(cache_dir, neutral_windows_bed=neutral_windows_bed,
                              keep_enhancer_windows=True)
    df = df.rename({W.MCHALE_ENHANCER_FLAG: TRUTH_TARGET})

    expected_path = refit_expected or refit_path("expected", "scored")
    print(f"decontaminated expected counts: {expected_path}")
    df = df.join(
        pl.read_csv(expected_path, separator="\t")
          .select(["element_id", pl.col("expected").alias("expected_scored")]),
        on="element_id", how="inner")

    for label, (col, _, _) in PR_SCORES.items():
        df = W.add_z_column(df, label, col)
        if col == W.PUBLISHED_EXPECTED_COL:
            W.check_z_against_published(df, label)
    df = W.filter_z_in_range(df, list(PR_SCORES))

    n_pos = int(df[TRUTH_TARGET].sum())
    print(f"evaluated: {df.height:,} windows, {n_pos:,} positive "
          f"({100 * n_pos / df.height:.1f}%)")
    return df


def _assign_gc_bins(df: pl.DataFrame, gc_bins: list) -> pl.DataFrame:
    """Add `gc_bin` (index into gc_bins) and drop windows outside every bin. Intervals are
    left-open and right-closed, matching the pandas.cut default the reference notebook
    relies on."""
    gc = df["GC_content"].to_numpy()
    idx = np.full(gc.shape, -1, dtype=int)
    for i, (lo, hi) in enumerate(gc_bins):
        idx[(gc > lo) & (gc <= hi)] = i
    return df.with_columns(pl.Series("gc_bin", idx)).filter(pl.col("gc_bin") >= 0)


def _balance_positive_fraction(df: pl.DataFrame, seed: int) -> pl.DataFrame:
    """
    Downsample the positive class within each GC bin so that every bin carries the SAME
    positive fraction -- the reference notebook's `downsample`, reproduced.

    WHY IT IS NECESSARY, AND IT IS NOT A NICETY. Precision is anchored to the positive
    fraction -- a random classifier's precision IS that fraction -- and enhancer density
    rises steeply with GC content. Over the bins this figure draws, the raw positive
    fraction runs from about 0.06 in (0.20, 0.30] to about 0.45 in (0.55, 0.60], a
    SEVENFOLD span (measured on a stand-in truth set of similar overall prevalence; the
    real GeneHancer flag will differ in level, not in the fact of the gradient). Plot raw
    precision against that and the GC-rich bins win by construction: the panel would be
    measuring enhancer density and reporting it as performance, which is the opposite of
    the figure's claim. Each bin is thinned to the smallest positive:negative ratio
    present, negatives untouched, which pegs every bin to the same fraction and is what
    makes the ONE dashed random-classifier line in panel A valid for every curve on it.

    WHAT IT COSTS, AND WHAT IT DOES NOT. It discards roughly four fifths of the positives,
    because the peg is set by the GC-poorest bin. It does NOT cost any drawn GC bin: the
    two bins the LAX_MIN_BIN_WINDOWS floor drops are already below that floor before
    any downsampling, so the balancing changes which windows are in a bin, never which
    bins survive. Worth re-checking on the real truth set, since a different enhancer
    annotation moves the peg.

    ONCE, ON THE LABELLED TABLE, not once per score -- the two scores are columns of the
    same rows. (The reference notebook balances per metric because its four metrics are
    carried on four different window files and it has no choice.) One balancing means the
    two curves in panel B are computed on an identical set of windows and an identical set
    of positives, so a difference between them is the score and nothing else.

    PANEL B'S /r IS NOT A SECOND GUARD ON THIS, once the balancing has run: r is then the
    same number in every bin, so dividing by it is a constant rescale that cannot change
    the panel's shape. Its job is to put the random classifier at exactly 1.0 so the axis
    reads "times better than guessing". The notebook computes it per bin and so does
    pr_curves, which is what would keep the bins comparable if the balancing were
    ever skipped -- but only partly, since auPRC/r is not prevalence-invariant for a real
    classifier the way it is for a random one. The downsampling is doing the work.
    """
    rng = np.random.default_rng(seed)
    bins = sorted(df["gc_bin"].unique().to_list())
    ratios = []
    for b in bins:
        sub = df.filter(pl.col("gc_bin") == b)
        n_pos = int(sub[TRUTH_TARGET].sum())
        n_neg = sub.height - n_pos
        ratios.append(n_pos / n_neg if n_neg else np.inf)
    target = float(min(ratios))

    kept = []
    for b in bins:
        sub = df.filter(pl.col("gc_bin") == b)
        neg = sub.filter(~pl.col(TRUTH_TARGET))
        pos = sub.filter(pl.col(TRUTH_TARGET))
        n_keep = int(target * neg.height)
        if n_keep < pos.height:
            pos = pos[np.sort(rng.choice(pos.height, size=n_keep, replace=False))]
        kept.append(pl.concat([pos, neg]))
    out = pl.concat(kept)
    print(f"class balancing: positive fraction pegged to {target / (1 + target):.4f} in "
          f"every GC bin; {df.height:,} -> {out.height:,} windows")
    return out


def _positive_fraction(df: pl.DataFrame) -> float:
    """The random classifier's precision on `df` -- panel A's dashed line, and panel B's
    normalizer. It is `r` in the baseline-classifier theory of McHale et al.'s Methods."""
    return float(df[TRUTH_TARGET].mean())  # type: ignore[arg-type]


def pr_curves(truth_set: str = "lax", cache_dir: str = CACHE_DIR,
              neutral_windows_bed: str | None = config.NEUTRAL_WINDOWS_BED,
              refit_expected: str | None = None, seed: int = 0,
              gc_bins: list | None = None, min_n: int | None = None) -> dict:
    """
    Everything Supporting Figure 8's two panels draw, against one truth set: labelled
    table -> GC bins -> class balancing -> precision-recall curves, in one call.

    `truth_set` is "lax" (GeneHancer enhancer overlap; McHale et al.'s Fig. 4A/B) and so
    far only that. "stringent" -- their essential-gene set -- is the planned second value,
    needing its own labelled-window builder beside _lax_labelled_windows, its own GC bins
    and its own bin floor; `gc_bins` and `min_n` default to the named truth set's own
    (LAX_*).

    BUT DO NOT EXPECT THE STRINGENT SET TO ARRIVE ONLY THROUGH THIS FUNCTION. Their
    Fig. 4C/D are a bootstrap statistic over two feature bins, not a per-bin PR sweep, and
    at 4,933 windows a curve per bin would be hopeless anyway -- see item 3 of the section
    header. This function is where the stringent set gets LABELLED and SCORED; the 4C/D
    statistic is separate builders that will want the labelled table, not these curves.

    ONE BUILDER, like every other entry point in this module, and the steps between are
    private because none of them is separately quotable. It is also the expensive one --
    it builds a SECOND window table, on a population panel E does not have -- so it should
    run exactly once per truth set per notebook execution.

    Returns, per score key:
        display, short   the two names for the curve (panel title, legend word)
        bins             list of {lo, hi, mid, n, recall, precision, aupr, aupr_norm}
        all              the same, pooled across GC bins -- panel A's black curve
        r                the positive fraction, identical across bins after balancing

    auPRC is sklearn's trapezoidal auc() over (recall, precision) -- the reference
    notebook's `auc(recall, precision)`, not `average_precision_score`. They differ
    slightly, and matching the notebook is what makes these numbers comparable with McHale
    et al.'s Fig. 4B rather than merely similar to it.

    `aupr_norm` divides by the bin's own positive fraction, so 1.0 is the random classifier
    and the axis reads "times better than guessing". After the balancing that divisor is
    the same in every bin, so it is a constant rescale rather than a second prevalence
    correction -- see _balance_positive_fraction, which is where the correction happens.
    """
    if truth_set != "lax":
        raise ValueError(
            f"truth_set={truth_set!r}: only 'lax' (GeneHancer enhancer overlap) is built. "
            "The stringent set is McHale et al.'s Fig. 4C/D and needs its own labelled-"
            "window builder here.")
    gc_bins = LAX_GC_BINS if gc_bins is None else gc_bins
    min_n = LAX_MIN_BIN_WINDOWS if min_n is None else min_n
    df = _lax_labelled_windows(cache_dir, neutral_windows_bed, refit_expected)
    df = _assign_gc_bins(df, gc_bins)
    df = _balance_positive_fraction(df, seed)
    print(f"mean GC of the evaluated windows: {df['GC_content'].mean():.3f}")

    out = {}
    for key, (_, display, short) in PR_SCORES.items():
        z = f"z_{key}"
        entries = []
        for b, (lo, hi) in enumerate(gc_bins):
            sub = df.filter(pl.col("gc_bin") == b)
            if sub.height < min_n:
                print(f"  {display}: GC ({lo}, {hi}] dropped, n = {sub.height:,} "
                      f"< {min_n:,}")
                continue
            precision, recall, _ = precision_recall_curve(
                sub[TRUTH_TARGET].to_numpy(), sub[z].to_numpy())
            r = _positive_fraction(sub)
            entries.append({
                "lo": lo, "hi": hi, "mid": 0.5 * (lo + hi), "n": sub.height,
                "recall": recall, "precision": precision,
                "aupr": float(auc(recall, precision)),
                "aupr_norm": float(auc(recall, precision)) / r,
            })
        precision, recall, _ = precision_recall_curve(
            df[TRUTH_TARGET].to_numpy(), df[z].to_numpy())
        r_all = _positive_fraction(df)
        out[key] = {
            "display": display, "short": short, "bins": entries, "r": r_all,
            "all": {"recall": recall, "precision": precision,
                    "aupr": float(auc(recall, precision)),
                    "aupr_norm": float(auc(recall, precision)) / r_all},
        }
        print(f"  {display}: auPRC/r = {out[key]['all']['aupr_norm']:.3f} pooled; "
              + ", ".join(f"({e['lo']:.2f},{e['hi']:.2f}]={e['aupr_norm']:.3f}"
                          for e in entries))
    return out


def _aupr(y: np.ndarray, score: np.ndarray) -> float:
    """Trapezoidal auc() over the precision-recall curve -- the same estimator
    enhancer_pr_curves uses, so a delta computed here is a delta between the numbers that
    panel plots."""
    precision, recall, _ = precision_recall_curve(y, score)
    return float(auc(recall, precision))


def pr_curve_deltas(truth_set: str = "lax", cache_dir: str = CACHE_DIR,
                    neutral_windows_bed: str | None = config.NEUTRAL_WINDOWS_BED,
                    refit_expected: str | None = None, seed: int = 0,
                    gc_bins: list | None = None, min_n: int = DELTA_MIN_BIN_WINDOWS,
                    n_bootstrap: int = 500, balance: bool = False) -> pl.DataFrame:
    """
    The PAIRED difference in performance between the two scores, per GC bin, with a
    bootstrap confidence interval. This is the panel that decides whether the retrained
    score is actually better anywhere, or whether the per-bin wobble in the auPRC curves is
    noise.

    THE STATISTIC IS A RELATIVE GAIN:

        delta(g) = [auPRC_scored(g) - auPRC_published(g)] / auPRC_published(g)

    and the normalizer r CANCELS from it, since both curves in panel B are the same auPRCs
    divided by the same per-bin r. So this is the same comparison panel B invites the eye to
    make, with the prevalence normalization taken out rather than applied twice, and it is
    dimensionless -- "the retrained score finds x% more of the enhancers, at equal recall".

    WHY PAIRED, AND WHY IT MATTERS SO MUCH HERE. The two scores are columns of ONE table:
    identical windows, identical positives, identical GC bins (_lax_labelled_windows joins
    both expected-count tables onto one window set and z-filters them jointly). Almost all
    of the sampling variability in auPRC is variability in WHICH WINDOWS the truth set
    happens to contain, and that is common to both scores, so it cancels in the difference.
    Independent error bars on panel B's two curves would therefore be a much weaker -- and
    misleading -- statement than this: they would show the uncertainty of each level, when
    the question is about the gap. Each bootstrap replicate here resamples the bin's rows
    once and scores BOTH models on that same resample, which is what preserves the pairing.

    RESAMPLING IS STRATIFIED BY BIN, i.e. within each GC bin separately. The bins are fixed
    strata defined by a covariate, not a random draw, so the inference wanted is conditional
    on them: "given these windows at this GC, how sure are we of the gap?"

    `balance=False` BY DEFAULT, WHICH IS THE OPPOSITE OF panel B, and deliberately.
    _balance_positive_fraction exists to make bins comparable in LEVEL -- it is what makes
    panel A's single dashed baseline valid for every curve on it. A within-bin,
    between-score comparison needs none of that: both scores see the same rows and the same
    prevalence, and r cancels from the statistic anyway. Meanwhile the balancing discards
    about four fifths of the positives, and it bites hardest exactly at high GC where
    positives are densest and the bins are already thin. Keeping them is a large gain in
    power precisely where the question is. Pass balance=True to compute the difference on
    panel B's own rows instead, which is the check that the two panels agree.

    n_bootstrap=500 gives a percentile interval whose own Monte-Carlo error is small
    against the widths involved; the cost is roughly n_bootstrap x one pass of
    precision_recall_curve over every drawn bin, twice.

    Returns one row per drawn bin: lo, hi, mid, n, n_pos, r, aupr_published, aupr_scored,
    delta (the point estimate, from the observed data -- not the bootstrap mean), ci_lo,
    ci_hi (2.5 and 97.5 percentiles), and p_gt0, the fraction of replicates with a positive
    gap.
    """
    if truth_set != "lax":
        raise ValueError(f"truth_set={truth_set!r}: only 'lax' is built.")
    gc_bins = DELTA_GC_BINS if gc_bins is None else gc_bins

    df = _lax_labelled_windows(cache_dir, neutral_windows_bed, refit_expected)
    df = _assign_gc_bins(df, gc_bins)
    if balance:
        df = _balance_positive_fraction(df, seed)

    rng = np.random.default_rng(seed)
    rows = []
    for b, (lo, hi) in enumerate(gc_bins):
        sub = df.filter(pl.col("gc_bin") == b)
        y = sub[TRUTH_TARGET].to_numpy()
        if sub.height < min_n or y.sum() == 0 or y.sum() == sub.height:
            print(f"  GC ({lo}, {hi}] dropped, n = {sub.height:,} "
                  f"(floor {min_n:,}, positives {int(y.sum()):,})")
            continue
        s_pub = sub["z_published"].to_numpy()
        s_sco = sub["z_scored"].to_numpy()
        a_pub, a_sco = _aupr(y, s_pub), _aupr(y, s_sco)

        boot = np.empty(n_bootstrap)
        for k in range(n_bootstrap):
            # ONE index draw, used for BOTH scores -- this line is the pairing.
            idx = rng.integers(0, sub.height, sub.height)
            yb = y[idx]
            if yb.sum() == 0 or yb.sum() == yb.size:
                boot[k] = np.nan
                continue
            boot[k] = _aupr(yb, s_sco[idx]) / _aupr(yb, s_pub[idx]) - 1.0
        boot = boot[~np.isnan(boot)]

        rows.append({
            "lo": lo, "hi": hi, "mid": 0.5 * (lo + hi), "n": sub.height,
            "n_pos": int(y.sum()), "r": float(y.mean()),
            "aupr_published": a_pub, "aupr_scored": a_sco,
            "delta": a_sco / a_pub - 1.0,
            "ci_lo": float(np.percentile(boot, 2.5)),
            "ci_hi": float(np.percentile(boot, 97.5)),
            "p_gt0": float((boot > 0).mean()),
        })
        r = rows[-1]
        print(f"  GC ({lo:.2f}, {hi:.2f}]  n = {r['n']:>9,}  pos = {r['n_pos']:>8,}  "
              f"delta = {100 * r['delta']:+6.2f}%  "
              f"[{100 * r['ci_lo']:+6.2f}, {100 * r['ci_hi']:+6.2f}]  "
              f"P(delta > 0) = {r['p_gt0']:.3f}")
    return pl.DataFrame(rows)


# ------------------------------------- Supporting Figure 8, panels D and E

# Chen et al.'s OWN cutoff for calling a window constrained, not a choice of ours: the
# paper says "constrained non-coding regions (Gnocchi >= 4)" and counts "19,471 constrained
# windows (Gnocchi >= 4)". Using it is what makes panels D and E a statement about the
# score as people actually apply it.
GNOCCHI_THRESHOLD = 4.0


def _wilson(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """
    Wilson score interval for a binomial proportion.

    NOT a bootstrap, and not for want of one. Precision and recall at a fixed threshold are
    plain proportions, so their sampling error has a closed form; and unlike panel C this
    panel plots two LEVELS rather than their difference, so there is no pairing to exploit
    -- an interval on each curve is exactly the right object. Wilson rather than the normal
    approximation because the counts get small in the thin GC bins and at a threshold only
    ~1% of windows clear, which is where Wald intervals run outside [0, 1].
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def threshold_metrics(threshold: float = GNOCCHI_THRESHOLD, truth_set: str = "lax",
                      cache_dir: str = CACHE_DIR,
                      neutral_windows_bed: str | None = config.NEUTRAL_WINDOWS_BED,
                      refit_expected: str | None = None, gc_bins: list | None = None,
                      min_n: int = DELTA_MIN_BIN_WINDOWS) -> pl.DataFrame:
    """
    Precision, recall and calling rate at a FIXED Gnocchi threshold, per GC bin, for both
    scores. Panels D and E, and the numbers behind them.

    WHY A FIXED THRESHOLD CHANGES WHAT IS MEASURED, and why this is the panel where the
    bias finally shows. Panels A-C are within-bin RANKING statistics: a bias that is a
    function of GC is very nearly a common shift applied to every window in a narrow bin,
    positives and negatives alike, and a common shift cannot change a within-bin ranking.
    So A-C are almost blind to the very thing Fig. 5 is about, which is why their two
    scores nearly coincide. Fix the threshold instead and the shift stops cancelling: it
    decides how many windows in each GC bin are CALLED at all.

    THE ANALYST'S QUESTION IS PRECISION AT A FIXED THRESHOLD. Someone handed a window with
    Gnocchi >= 4 wants P(constrained | called), and if that probability depends on the
    window's GC content then the threshold means different things in different parts of the
    genome and the score cannot be used as a single genome-wide cutoff. That is a stronger
    practical claim than anything in panels A-C, and it is the one a bias correction should
    be able to deliver.

    THREE QUANTITIES, BECAUSE PRECISION ALONE CONFOUNDS TWO EFFECTS THE FIGURE MUST KEEP
    APART. Per bin g and score:

        call_rate(g)  = P(z >= t | g)                 -- pure exposure to the bias,
                                                        and it needs NO truth set
        precision(g)  = P(constrained | z >= t, g)    -- the analyst's number
        recall(g)     = P(z >= t | constrained, g)    -- what fraction is caught
        lift(g)       = precision(g) / r(g)           -- precision over the bin's base rate

    call_rate is where the bias lives undiluted: with no GC bias the fraction of windows
    clearing a fixed cutoff should not track GC. It is also the ONE quantity here that does
    not involve the labels at all -- it is a property of the score and of GC content -- so
    it is the most robust claim this figure can make, resting on neither GeneHancer nor the
    laxness of an enhancer-overlap proxy. Quote it accordingly. precision does NOT have to be flat even
    for a perfect score -- the base rate r(g) climbs about 7.7x across these bins in the lax
    truth set, and a bin with more enhancers yields higher precision at any threshold -- so
    the panel draws r(g) as its reference and `lift` is the base-rate-free version. Expect
    the correction to flatten call_rate strongly, precision and recall less so, and do not
    read a residual slope in precision as a residual bias: declining signal-to-noise with GC
    survives debiasing (that is panels B/C's finding) and shows up here too.

    UNBALANCED, unlike panels A and B. The base rate an analyst faces is the real one; class
    balancing would answer a question nobody has. The prevalence reference line is what
    keeps the bins comparable instead.

    Intervals are Wilson, per curve -- see _wilson for why not a bootstrap here.

    Returns one row per (GC bin, score): lo, hi, mid, score, n, n_pos, r, n_called,
    call_rate, precision, precision_lo, precision_hi, recall, recall_lo, recall_hi, lift.
    """
    if truth_set != "lax":
        raise ValueError(f"truth_set={truth_set!r}: only 'lax' is built.")
    gc_bins = DELTA_GC_BINS if gc_bins is None else gc_bins

    df = _lax_labelled_windows(cache_dir, neutral_windows_bed, refit_expected)
    df = _assign_gc_bins(df, gc_bins)

    rows = []
    for b, (lo, hi) in enumerate(gc_bins):
        sub = df.filter(pl.col("gc_bin") == b)
        if sub.height < min_n:
            print(f"  GC ({lo}, {hi}] dropped, n = {sub.height:,} < {min_n:,}")
            continue
        y = sub[TRUTH_TARGET].to_numpy()
        n_pos = int(y.sum())
        for key, (_, display, short) in PR_SCORES.items():
            called = sub[f"z_{key}"].to_numpy() >= threshold
            n_called = int(called.sum())
            tp = int((called & y).sum())
            prec_lo, prec_hi = _wilson(tp, n_called)
            rec_lo, rec_hi = _wilson(tp, n_pos)
            call_lo, call_hi = _wilson(n_called, sub.height)
            r = n_pos / sub.height
            rows.append({
                "lo": lo, "hi": hi, "mid": 0.5 * (lo + hi), "score": key,
                "display": display, "short": short,
                "n": sub.height, "n_pos": n_pos, "r": r,
                "n_called": n_called, "call_rate": n_called / sub.height,
                "call_rate_lo": call_lo, "call_rate_hi": call_hi,
                "precision": tp / n_called if n_called else float("nan"),
                "precision_lo": prec_lo, "precision_hi": prec_hi,
                "recall": tp / n_pos if n_pos else float("nan"),
                "recall_lo": rec_lo, "recall_hi": rec_hi,
                "lift": (tp / n_called) / r if n_called and r else float("nan"),
            })
            e = rows[-1]
            print(f"  GC ({lo:.2f}, {hi:.2f}]  {short:<15} called {n_called:>7,} "
                  f"({100 * e['call_rate']:5.2f}% of windows)  "
                  f"precision {e['precision']:.3f} [{prec_lo:.3f}, {prec_hi:.3f}]  "
                  f"base rate {r:.3f}  lift {e['lift']:.2f}  recall {e['recall']:.4f}")
    return pl.DataFrame(rows)
