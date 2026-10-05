"""
Data for the six panels of Fig. 5 and for Supporting Figures 7 and 8. Every builder caches its
result as parquet in fig5/output/, so the notebook is instant after the first pass.

One builder per plotted quantity, grouped by the panel that draws it. (Panel C and
Supporting Figure 7 were once a separate diagnostics.py; history at ea1805c.)

Inputs:
  * the public gnomAD-NC-constraint bucket, downloaded on demand into published/;
  * the three refits fig5/refit.py writes (full / scored / sizematched) to the repo-root
    refits/, one copy, also read by dnm_training_size/;
  * two files NOT in this repo, read from fig5/config.py (not the notebook -- refit.py
    reads the same module, and the two must agree): a depletion-rank BED (panel A's third
    curve) and McHale et al.'s window file (their 693,270 putatively neutral windows, and
    the enhancer flag that is Supporting Fig. 8's truth set). Both are set, to HPC paths.
    Unset, Fig. 5 builds on the wider window set and Supporting Fig. 8 not at all.
"""
import contextlib
import hashlib
import io
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


@contextlib.contextmanager
def quiet():
    """
    Swallow a builder's stdout, for the notebook's REPEAT calls: Supporting Fig. 8's
    builders each reprint the same preamble, so the first call of each kind runs loud and
    the repeats run in here. Exceptions still propagate; the captured text is yielded.

    A context manager, not a `verbose` parameter, because much of that preamble is printed
    by gnocchi_bias/windows.py, which preconditions/ and dnm_training_size/ import too --
    a flag would change three other entry points to tidy one notebook.
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield buf

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
    config.NEUTRAL_WINDOWS_BED. Route every read through here: `scored` is fit on the
    analyzed window set, so a stale refit is trained on one population and scored on
    another.
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


def _wilson(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """
    Wilson score interval for a binomial proportion -- for a panel plotting LEVELS (e.g.
    Fig. 5F's calling rates), where each proportion has a closed-form error. A panel
    plotting a DIFFERENCE between scores takes the paired bootstrap instead. Wilson, not
    Wald, because counts get small in thin GC bins, where Wald runs outside [0, 1].
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


# --------------------------------------------------------------- shared GC bins

def gc_edges(gc: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """
    Fixed-width edges spanning the observed GC range, matching windows.bin_by_gc's
    "fixed" branch. Passed explicitly to every consumer -- windows, duckdb-aggregated
    expected counts, DNM training sites -- since they compare only on identical edges.
    """
    edges = np.unique(np.linspace(float(np.min(gc)), float(np.max(gc)), n_bins + 1))
    edges[-1] += 1e-9
    return edges.astype(float)


def assign_bin(gc: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.digitize(np.asarray(gc, float), edges[1:-1]), 0, len(edges) - 2)


def sql_bin_expr(gc_expr: str, edges: np.ndarray) -> str:
    """duckdb equivalent of assign_bin, to group inside a query rather than materialize
    tens of millions of rows. Edges are uniform, so a clipped floor-divide suffices."""
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
    The analyzed window population: noncoding, pass_qc, autosome/PAR, GC as a 0-1
    fraction -- or, if config.NEUTRAL_WINDOWS_BED is set, McHale et al.'s 693,270
    putatively neutral windows. Both the set Gnocchi is scored on and (panels C-E) the
    one the retrained adjustment is fit on. The default is read from fig5/config.py, as
    refit.py reads it; pass another value only if you rerun the refits with it.
    """
    return W.build_window_table(cache_dir, neutral_windows_bed=neutral_windows_bed)


def rank_bias(binned: pl.DataFrame, label: str, min_n: int = 0) -> float:
    """
    Mean |mean rank - 0.5| over GC bins with >= min_n windows, unweighted: one number for
    "how GC-biased is this metric". Pass the panel's own min_n, or this summarizes bins
    the reader cannot see.
    """
    b = binned.filter(pl.col("n") >= min_n) if min_n else binned
    return float((b[f"mean_{label}"] - 0.5).abs().mean())


def rank_curves(df_win: pl.DataFrame, extra: list[tuple[str, str]] = (),
                min_n: int = 100):
    """
    The Fig. 2A rank statistic for the context-only model (r == 1), published Gnocchi and
    any `extra` (label, expected-table path) refits, on one window population and one set
    of GC bins, z-filtered jointly so no curve is advantaged by its own filtering.
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

    The caller gets non-CpG by subtraction (e1_non = e1 - e1_cpg), so only the CpG slice
    of the multi-GB per-context files is joined (10M x 10M rows, not 85M x 85M). That
    mixes two published files, totals from the summed export and CpG parts from the
    per-context one; preconditions/verify_expected_r1.py shows they agree (`possible`
    exactly, `expected` to 4.6e-5 relative).

    Chen et al. never published per-context r, so the refit's rr table stands in;
    r_eff_by_gc validates that per GC bin against the published E2/E1.
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

    RATIOS OF SUMMED expected counts, not means of per-window ratios: sum(E2)/sum(E1) is
    the adjustment the bin actually receives, and it keeps r_eff = Pi*r_CpG + (1-Pi)*r_non
    exact bin by bin. Prints the published-vs-refit agreement in r_eff, which is what
    licenses using the refit's per-context r for the CpG/non-CpG split.
    """
    # config.tagged(pop) is a no-op for the only caller (pop="full", not WINDOW_DEPENDENT).
    # It guards a future "scored"/"sizematched" caller: cached() returns before
    # refit_path's config.check runs, so an untagged cache would be reused silently
    # across window sets.
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
# Both of panel C's rows are views of ONE query, dnm_rate_by_stratum(): per-stratum site
# counts above (how much of the training set sits outside the scored population),
# per-stratum DNM rates below (whether that territory is also DIFFERENT). Both count DNMs
# and background alike, since the fit's loss is over that mixture; the background class
# alone would describe the case-control design, not the fit.

# The strata a training site can fall into, in DRAWING order (bottom to top, matching
# panels.COMPOSITION_STYLE; _stratum_expr tests them in a different order):
#   scored          its 1 kb window is in the analyzed window table -- a join, not a
#                   re-statement of that table's filters, so `scored` means "survives the
#                   panel D/E intervention" under whichever definition
#                   windows.build_window_table is applying (the same table
#                   dnm_model.restrict_to_analyzed_windows filters the training set with).
#   coding          in the constraint table, but overlapping coding exons.
#   other_noncoding QC-pass noncoding but outside McHale et al.'s neutral set: windows their
#                   file flags as enhancer-overlapping, plus ones it does not list (their
#                   assembly-gap / ENCODE-exclude / low-coverage exclusions). Named for
#                   where it sits, not what it is: being outside a putatively neutral set
#                   is not evidence of selection, and whether this territory differs at all
#                   is what the band measures. Empty, and undrawn, while
#                   config.NEUTRAL_WINDOWS_BED is None.
#   failed_qc       no row in the constraint table. Not `no_coverage`: every such window
#                   has its QC inputs on file and fails one of the paper's three
#                   conditions, mostly the >= 80% PASS rule (preconditions/
#                   verify_qc_filter.py).
_STRATA = ("scored", "coding", "other_noncoding", "failed_qc")


def _stratum_expr() -> str:
    """
    Panel C's CASE expression. `sw` is the analyzed window table, registered by
    dnm_rate_by_stratum; `an` is the published constraint table. The coding threshold is
    windows.NONCODING_MAX_CODING_PROP, the one restrict_to_noncoding uses.

    The genome's 1 kb windows partition three ways (top row); below it, the stratum each
    region gets under each setting of config.NEUTRAL_WINDOWS_BED:

    |<------------------ QC-pass: a row in `an` ------------------->||<- QC-fail ->|
    +-----------------------------------------+---------------------+--------------+
    |                noncoding                |        coding       |no row in `an`|
    +-----------------------------------------+---------------------+--------------+

    |<------------ sw, BED unset ------------>|
    |            scored (1,843,559)           |        coding       |  failed_qc   |

    |<--- sw, BED set ---->|                  |<- * ->|
    |   scored (693,270)   | other_noncoding  | scored|    coding   |  failed_qc   |

    Unset, `sw` IS the noncoding cell. Set, it is McHale et al.'s file filtered to
    enhancer == False and nothing else, so it covers part of the noncoding cell and MAY
    reach into the coding one (*), since nothing filters their file on coding_prop.

    HENCE `scored` IS TESTED FIRST: panels D/E fit and score on a window in (*), so that is
    its true label. windows.restrict_to_mchale_neutral_windows prints how many there are.
    (693,270 is their enhancer == False row count, before the join drops windows lacking
    a constraint/expected/features row.)
    """
    return f"""CASE WHEN sw.element_id IS NOT NULL THEN 'scored'
                    WHEN an.element_id IS NULL THEN 'failed_qc'
                    WHEN an.coding_prop > {W.NONCODING_MAX_CODING_PROP!r} THEN 'coding'
                    ELSE 'other_noncoding' END"""

# chrX/chrY dropped from BOTH classes. The published fitting code drops chrX from the
# background class only, which inflates the apparent rate there.
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
    per GC bin. `extra_group_by`: further (expression, alias) grouping keys, e.g. panel C's
    stratum CASE. `aggs`: further aggregate select items. `extra_joins`: join clauses for a
    relation the caller registered on its own connection, e.g. the analyzed window table.
    """
    gc_bin = sql_bin_expr("ft.GC_content_1k / 100.0", edges)
    key_select = "".join(f"{expr} AS {alias}, " for expr, alias in extra_group_by)
    # GROUP BY names the select aliases, which duckdb resolves -- safe only while no alias
    # collides with a column of `s`, `ft`, `an` or a registered relation.
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
    Six hex characters for "these GC edges over this window population", for a cache key.
    Both move with config.NEUTRAL_WINDOWS_BED (gc_edges spans the window set's GC range),
    so without this a table cached under one setting would be silently reused under the
    other. `df_win` is omitted by builders that bin the whole training population.
    Order-independent; a polars version bump can only cost a rebuild.
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
    Empirical P(DNM) over non-CpG training sites, per GC bin and _STRATA stratum. `df_win`
    must be the analyzed window table the panels use (data.window_table), or sites are
    labelled against a population no panel shows. On the wider window set, 72,801 of the
    non-CpG autosomal DNMs are QC-failing, 17,545 coding and 241,479 scored.

    THE POINT. The scored and coding curves are nearly flat and nearly equal, so the
    coding exclusion is not what makes the training set's GC dependence steep. The
    QC-failing curve is not flat: 1.50-1.63x the scored rate in the GC bulk and 3.39x by
    GC 0.58 on the committed (narrowed) run (wider: 1.55x, 4.06x by GC 0.61). Essentially
    all of the training set's GC dependence comes from sequence gnomAD could not call
    reliably -- also where trio DNM calling is least reliable, so part of the excess is
    plausibly false-positive DNM calls.

    Columns: stratum, gc_bin, gc_pct, k (DNMs), n (sites), p = k/n.
    """
    def build():
        con = duck(memory_limit)
        # Registered, so the analyzed window set enters the query as itself.
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
    Panel C's upper row: each GC bin's non-CpG training sites by stratum, as counts and
    fractions -- a reshape of dnm_rate_by_stratum()'s `n`, so it cannot drift from the
    lower row. Its background class alone (n - k) reproduces the dnm0-only query it
    replaced exactly; the mixture adds the DNM class's steeper drift out of the scored
    population. The CASE expression puts each site in exactly one stratum, so nothing is
    asserted.

    Columns: gc_bin, gc_mid, n_total, n_{stratum}, frac_{stratum}, for every stratum in
    _STRATA, empty ones included; panels.py drops a band that is zero everywhere.
    """
    # A stratum missing from a bin pivots to null; one missing from EVERY bin has no
    # column at all, hence the zero fill.
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
    # No fraction range printed: the near-empty lowest-GC bins would make min() read 0.00.
    # The notebook reports it over plotted bins.
    print(f"training composition: {int(df['n_total'].sum()):,} non-CpG training sites "
          f"over {df.height} GC bins")
    return df


def stratum_ratios(st: pl.DataFrame, edges: np.ndarray, min_n: int = 2000) -> pl.DataFrame:
    """
    Panel C's lower row: each excluded stratum's non-CpG DNM rate over the scored
    population's, per GC bin. A ratio, because the question is whether the excluded
    territory DIFFERS, and it divides out the scored rate's own mild GC drift.

    `{stratum}_se_log` is the delta-method SE of log(ratio), sqrt((1-p_a)/k_a +
    (1-p_b)/k_b) with k the DNM count -- the SE of the LOG ratio only, which is why panel
    C plots log(ratio) on a linear axis with plain +/- se bars. min_n drops bins where
    either stratum has fewer sites.

    Columns: gc_bin, gc_mid, and {stratum}_{ratio,se_log} for each excluded stratum with
    any bin left after min_n; panels.py plots whichever it finds.
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
    Panel D's tables: per-GC-bin fitted and empirical P(DNM) over each population's
    non-CpG training sites, from refit.py's per-site predictions. Binned on edges SHARED
    across populations, whose GC ranges differ. GC is in 0-100 percent units here
    (GC_content_1k); panel_dnm_probability_pairs divides by 100.

    The empirical level reflects the case-control design (dnm0:dnm1 ~ 10:1), not the
    genome-wide DNM rate; only shape compares across populations.
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
# Why r_CpG ~ 1 is CORRECT: the effect that would need adjusting is already applied in
# step 1. These two builders measure its size and GC dependence; the figure's fourth row,
# Pi, comes from r_eff_by_gc.

def cpg_methylation_by_gc(edges: np.ndarray, cache_dir: str = CACHE_DIR,
                          force: bool = False, memory_limit: str = "10GB") -> pl.DataFrame:
    """
    CpG-context training sites per GC bin: mean methylation level, the fraction that are
    hypomethylated (level <= 1), and the empirical DNM rate.

    THE POINT. High-GC CpGs are CpG islands -- 92% hypomethylated in the top GC bin on
    the committed (narrowed) run, 90-100% above GC 0.70 on the wider one, against ~2% in
    the GC bulk -- and their DNM rate collapses 1.9x (wider: 2.7x). Step 1 already models
    exactly that, fitted_po being keyed by methylation, so r_CpG ~ 1 in panel B is correct:
    nothing is left for the regional adjustment to correct.

    Over the whole training population, since the claim is about CpG biology; only the
    bin EDGES come from the analyzed windows, hence the fingerprinted cache key.

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

    `fitted_po`, the per-site step-1 probability the pipeline uses, spans ~4.3x across
    methylation 0 -> 15 within one context, the largest single rate effect in the model.
    `mu`, the pre-saturation estimate, spans ~10-15x; the gap IS fitted_po's saturation,
    which is why a naive D/E1 ratio understates the CpG rate at high methylation. A
    control on the CpG story only: panel B rests on r being a ratio, where level effects
    cancel.
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


# ------------------------------------------------------------------- panel F

def calling_rate_by_gc(df: pl.DataFrame, threshold: float = 4.0,
                       labels=(("step2", "published"), ("scored", "decontaminated")),
                       reference: str = "step2", n_bins: int = N_BINS,
                       min_n: int = 100) -> tuple[pl.DataFrame, dict, float]:
    """
    Panel F: the fraction of windows in each GC bin whose z clears a fixed threshold.

    Panels A and E's bias in the units the score is used in: at Chen et al.'s own cutoff,
    what fraction of each part of the genome is called constrained? It uses NO LABELS, so
    it sits with Fig. 5's label-free panels. Computed on panel E's own table and bins, so
    E (rank returning to 0.5) and F (calling rate flattening) are one fix on one window
    set.

    THE SCORES ARE MATCHED ON OVERALL CALLING RATE, not a common number: retraining moves
    the whole z distribution. `reference` is held at `threshold`; every other score gets
    the quantile of its own z calling the same fraction of the whole population. The swing
    ACROSS GC is within one score, so that choice does not touch it.

    Returns (binned, thresholds, target): per GC bin, gc_mid, n and rate/lo/hi per label
    (Wilson); each score's threshold; and the matched calling rate. `target` IS A PANEL
    ELEMENT -- the horizontal line both curves are pinned to (panels.panel_calling_rate's
    `matched_rate`). It is over every window in `df`, INCLUDING bins `min_n` drops, so it
    is the genome-wide operating point, not a mean over plotted points.
    """
    gc = df["GC_content"].to_numpy()
    edges = gc_edges(gc, n_bins)
    idx = assign_bin(gc, edges)

    ref = df[f"z_{reference}"].to_numpy()
    target = float((ref >= threshold).mean())
    thresholds = {}
    for key, _ in labels:
        thresholds[key] = (threshold if key == reference else
                           float(np.quantile(df[f"z_{key}"].to_numpy(), 1.0 - target)))
    print(f"  matched calling rate {100 * target:.3f}%, set by {reference} at "
          f"z >= {threshold:g}")
    for key, disp in labels:
        print(f"    {disp:<38} z >= {thresholds[key]:.3f}")

    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        n = int(m.sum())
        if n < min_n:
            continue
        row = {"gc_mid": float(0.5 * (edges[b] + edges[b + 1])), "n": n}
        for key, _ in labels:
            k = int((df[f"z_{key}"].to_numpy()[m] >= thresholds[key]).sum())
            lo, hi = _wilson(k, n)
            row[f"rate_{key}"], row[f"lo_{key}"], row[f"hi_{key}"] = k / n, lo, hi
        rows.append(row)
    out = pl.DataFrame(rows)
    for key, disp in labels:
        r = out[f"rate_{key}"].to_numpy()
        nz = r[r > 0]
        # Swing quoted over bins where the score calls anything: published calls nothing
        # in the most AT-rich bins, which would make the raw ratio infinite.
        span = (f"{r.max() / nz.min():.0f}x over bins it calls in"
                if len(nz) < len(r) else f"{r.max() / r.min():.0f}x across GC")
        zeros = len(r) - len(nz)
        print(f"  {disp:<38} calling rate {100 * r.min():.3f}% - {100 * r.max():.3f}%  "
              f"({span}" + (f", and 0% in {zeros} bin(s))" if zeros else ")"))
    return out, thresholds, target


# --------------------------------------------------------- Supporting Figure 8

# WHAT THIS SECTION IS FOR. Panel E says the retrained score is no longer GC-biased, not
# whether the biased score was nevertheless the better DETECTOR: bias and signal-to-noise
# act on discovery jointly (McHale et al.'s Fig. 3). Supporting Figure 8 is that test,
# built as their Fig. 4A/B -- call a window constrained when its z exceeds a threshold,
# read performance within each GC bin -- with two Gnocchi variants as the classifiers.
#
# Names are organized on LAX vs STRINGENT, McHale et al.'s two truth sets:
#   LAX        constrained = overlaps a GeneHancer enhancer. Large enough to resolve the GC
#              tails; lax because GeneHancer covers 18.4% of the noncoding genome while
#              perhaps 4.51% is under human-specific selection. Their Fig. 4A/B. BUILT HERE.
#   STRINGENT  essential-gene enhancers, plus an equal number of no-enhancer windows. Their
#              Fig. 4C/D (papers/neutral_models_are_biased/
#              11.compare-lax-with-stringent-truth-set.ipynb). DELIBERATELY NOT BUILT
#              (2026-09-04): Fig. 5F needs no truth set, 4,933 windows cannot resolve a
#              ~1.5% pooled gap, and its positives are plausibly MORE GC-skewed. If ever
#              revisited, three things are not guessable (read from that notebook):
#     1. It is a third hand-supplied file, stringent_truth_set/
#        truth-set.gnocchi.lambda_s.depletion_rank.CDTS.bed under CONSTRAINT_TOOLS_DATA:
#        4,933 rows, target column `truly constrained`, coordinates in `chromosome` (not
#        `chrom`) -- its own config entry and preflight check.
#     2. Its positives are NOT on Chen's 1 kb grid (e.g. chr1-2128961-2129161, 200 bp), so
#        mapping them to the retrained score needs an interval overlap plus a spanning
#        rule -- and it must be THEIR rule (their `gnocchi` column was carried over by one;
#        cell 9 shows it for lambda_s), or the two scores are mapped differently.
#     3. 4C/D are a different statistic, so new builders, not a pr_curves() value: 1,000
#        bootstrap replicates (resample with replacement), exactly TWO feature bins (GC
#        (0.20, 0.375) / (0.40, 0.70); BGS (0.5, 0.76) / (0.9, 1.0); gBGC (-0.3, 0.2) /
#        (0.4, 1.2)), the same balancing, then auPRCnorm = auc/r pooled (4C) and
#        delta = (auc[low] - auc[high]) / auc[pooled] (4D), mean and sd of each. Bin floor
#        500, a thinner bin skipping that feature; the lax set first .sample(n=len(stringent)).
#
# Truth-set-specific constants carry the set's name (LAX_GC_BINS); shared ones do not
# (PR_SCORES, TRUTH_TARGET). Not "enhancer": that is how LAX is defined, not what this is.
#
# Reference implementation for the lax set (bins, balancing, bin floor, trapezoidal
# auc(recall, precision)): constraint-tools papers/neutral_models_are_biased/7.CDTS/main.2.ipynb.
#
# THE LAX TRUTH SET IS GENEHANCER AND HAS NO SUBSTITUTE: `window overlaps enhancer` in
# config.NEUTRAL_WINDOWS_BED is licensed and not derivable from the public bucket. So unlike
# everything else here this does not build without that file: pr_curves raises, and the
# notebook checks and skips.

# The GC bins of Supporting Fig. 8's fixed-threshold panels (A, B, D-I) and of
# budget_comparison: LAX_GC_BINS with the tail merged. McHale et al.'s window file is nearly
# empty above GC 0.60 -- after class balancing the three bins there hold 1,086, 65 and 2
# windows -- so one (0.55, 0.80] bin is the honest use of the tail, which is where panel E's
# bias reduction is largest. Below 0.55 the bins are LAX_GC_BINS's, so these panels' x axes
# line up with panel C's.
THRESHOLD_GC_BINS = [(0.20, 0.30), (0.30, 0.40), (0.40, 0.50), (0.50, 0.55), (0.55, 0.80)]

# Their bin floor. Far below LAX_MIN_BIN_WINDOWS, which guards a precision-recall CURVE (a
# staircase on few windows); these panels draw one number per bin with an honest interval.
THRESHOLD_MIN_BIN_WINDOWS = 500

# The reference notebook's lax-set GC bins, verbatim, so panel C compares with McHale et
# al.'s Fig. 4B: wide at the scarce ends, narrow through the bulk. Not N_BINS fixed-width
# edges -- after balancing, 20 equal bins would mostly fall under the window floor.
LAX_GC_BINS = [(0.20, 0.30), (0.30, 0.40), (0.40, 0.50), (0.50, 0.55),
               (0.55, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 0.80)]

# The reference notebook's bin floor for the lax set: a precision-recall curve on a few
# hundred windows is mostly staircase.
LAX_MIN_BIN_WINDOWS = 4_000

# The two scores: key -> (expected-count column, long name, legend word). Both names
# travel in the returned dicts, so panels.py need not import this module.
PR_SCORES = {
    "published": (W.PUBLISHED_EXPECTED_COL, "Gnocchi (published)", "published"),
    "scored": ("expected_scored", "Gnocchi (decontaminated training set)",
               "decontaminated"),
}

# The label column, whichever truth set produced it (for lax, windows.MCHALE_ENHANCER_FLAG
# renamed), so nothing downstream needs to know which one it is working on.
TRUTH_TARGET = "constrained"


def _lax_labelled_windows(cache_dir: str, neutral_windows_bed: str | None,
                          refit_expected: str | None) -> pl.DataFrame:
    """
    The LAX truth set: one row per evaluated window -- element_id, GC_content (0-1),
    `constrained` (overlaps a GeneHancer enhancer), and a z column per PR_SCORES entry.

    window_table()'s PIPELINE WITH ONE FILTER DROPPED (`keep_enhancer_windows=True`): the
    enhancer == False step does not run and the flag comes back as a column. Panel E must
    not have those windows (selection lowers z for a reason that is not bias); this figure
    needs them as its positive class. Nothing else differs, so the figure is a statement
    ABOUT panel E. The `scored` refit is still FIT on the neutral windows alone and
    EVALUATED here on both. Both z columns are filtered JOINTLY to [-10, 10].
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
    """Add `gc_bin` (index into gc_bins), dropping windows outside every bin. Intervals
    are (lo, hi], the pandas.cut default the reference notebook relies on."""
    gc = df["GC_content"].to_numpy()
    idx = np.full(gc.shape, -1, dtype=int)
    for i, (lo, hi) in enumerate(gc_bins):
        idx[(gc > lo) & (gc <= hi)] = i
    return df.with_columns(pl.Series("gc_bin", idx)).filter(pl.col("gc_bin") >= 0)


def _balance_positive_fraction(df: pl.DataFrame, seed: int) -> pl.DataFrame:
    """
    Downsample positives within each GC bin to the smallest positive:negative ratio
    present, so every bin carries the SAME positive fraction -- the reference notebook's
    `downsample`. NECESSARY: a random classifier's precision IS the positive fraction, and
    that rises ~7.7x with GC (from 0.083 in (0.20, 0.30]), so raw precision would report
    enhancer density as performance. It is what makes one random-classifier line valid for
    every bin. Panel C's /r is then a constant rescale (putting random at 1.0), not a
    second correction -- and could not be one, auPRC/r not being prevalence-invariant for
    a real classifier.

    COST: four fifths of the positives (246,930 of 309,908 on the committed run), the peg
    being set by the GC-poorest bin; after it the three bins above GC 0.60 hold 1,086, 65
    and 2 windows, under LAX_MIN_BIN_WINDOWS. Done ONCE on the labelled table, not per
    score, so both of 8C's curves see identical windows and positives.
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
    """The random classifier's precision on `df` -- `r` in McHale et al.'s Methods, and
    the normalizer of Supporting Fig. 8C's y axis."""
    return float(df[TRUTH_TARGET].mean())  # type: ignore[arg-type]


def pr_curves(truth_set: str = "lax", cache_dir: str = CACHE_DIR,
              neutral_windows_bed: str | None = config.NEUTRAL_WINDOWS_BED,
              refit_expected: str | None = None, seed: int = 0,
              gc_bins: list | None = None, min_n: int | None = None) -> dict:
    """
    Everything Supporting Fig. 8C draws: labelled table -> GC bins -> class balancing ->
    precision-recall curves. `truth_set` accepts only "lax" (see the section header);
    `gc_bins` and `min_n` default to LAX_*. Expensive -- it builds a second window table --
    so run it once per notebook execution.

    Returns, per score key:
        display, short   the two names for the curve (panel title, legend word)
        bins             list of {lo, hi, mid, n, recall, precision, aupr, aupr_norm}
        all              the same, pooled across GC bins -- the "all GC content" curve
        r                the positive fraction, identical across bins after balancing

    auPRC is the trapezoidal auc(recall, precision), as in the reference notebook, not
    average_precision_score, so the numbers compare with McHale et al.'s Fig. 4B.
    `aupr_norm` divides by the bin's positive fraction, putting random at 1.0.
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
    """Trapezoidal auc() over the precision-recall curve, the estimator pr_curves uses."""
    precision, recall, _ = precision_recall_curve(y, score)
    return float(auc(recall, precision))


def pr_curve_deltas(truth_set: str = "lax", cache_dir: str = CACHE_DIR,
                    neutral_windows_bed: str | None = config.NEUTRAL_WINDOWS_BED,
                    refit_expected: str | None = None, seed: int = 0,
                    gc_bins: list | None = None, min_n: int = LAX_MIN_BIN_WINDOWS,
                    n_bootstrap: int = 500, balance: bool = False) -> pl.DataFrame:
    """
    The PAIRED difference in auPRC between the two scores, per GC bin, with a bootstrap
    interval: Supporting Fig. 8C's error bars. Defaults to C's bins and floor (LAX_*); the
    notebook passes them, and balance=True, explicitly.

        delta(g) = auPRC_scored(g) / auPRC_published(g) - 1

    r cancels, both of C's curves being divided by the same per-bin r, so this is the
    comparison C invites the eye to make.

    PAIRED because the two scores are columns of ONE table: almost all of auPRC's sampling
    variability is in WHICH windows the truth set contains, which is common to both and
    cancels in the difference -- independent bars on each level would understate the
    evidence about the gap. Each replicate resamples a bin's rows once and scores BOTH
    models on it. Stratified by bin, since the bins are fixed strata of a covariate.

    balance=False by default: within a bin r cancels anyway, and balancing discards four
    fifths of the positives, most at high GC. But pass True when the result is drawn as
    C's error bars -- an interval on different rows belongs to a different statistic.

    Returns one row per drawn bin: lo, hi, mid, n, n_pos, r, aupr_published, aupr_scored,
    delta (observed, not the bootstrap mean), ci_lo, ci_hi (2.5/97.5 percentiles) and
    p_gt0, the fraction of replicates with a positive gap.
    """
    if truth_set != "lax":
        raise ValueError(f"truth_set={truth_set!r}: only 'lax' is built.")
    gc_bins = LAX_GC_BINS if gc_bins is None else gc_bins

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


# ----------------- Supporting Figure 8A/B and D-I: fixed thresholds, matched calling rates

# Chen et al.'s OWN cutoff ("constrained non-coding regions (Gnocchi >= 4)"), so Fig. 5F
# and Supporting Fig. 8A/B describe the score as people actually apply it.
GNOCCHI_THRESHOLD = 4.0

# Supporting Fig. 8F and I's operating point: the top 1% of each GC bin by each score. F
# draws the thresholds, I the gain measured at them -- one construction seen twice. The
# other LAX_CALL_RATES pair D with G and E with H the same way.
#
# NOT A AND B's OPERATING POINT. A and B apply ONE GLOBAL CUTOFF per score and let each
# bin's calling fraction fall where it may; that freedom IS the bias, so a per-bin rate
# there would delete what they measure. D-I impose the rate to remove it, isolating
# ranking from threshold placement. A round 1%, not published's 1.002% at z >= 4, because
# a per-bin cutoff is a construction nobody applies.
LAX_CALL_RATE = 0.01

# The three per-bin calling rates Supporting Fig. 8 reads one comparison at, each matched
# between the scores within the bin. In all three A CALL IS z > t AND A HIT IS AN
# ENHANCER -- one hypothesis, three points along the recall axis:
#
#     0.01   the top 1% of a bin           low recall    the most constrained sequence
#     0.50   the upper half of a bin       mid recall    where most of C's area lives
#     0.99   all but the bottom 1%         high recall   precision pinned near prevalence
#
# A DECOMPOSITION OF 8C: auPRC integrates over the whole recall axis, so a wash there can
# hide a gain at one end and a loss at the other. The 99% point replaced a bottom-1%
# construction (call z <= t, hit a NON-enhancer) -- see the left-tail note below.
LAX_CALL_RATES = (LAX_CALL_RATE, 0.50, 0.99)

# The column each score is ranked by, as _lax_labelled_windows built it.
#
# NO GC-ONLY ARM, DELIBERATELY. It would measure how much of this truth set is a GC
# contest -- true of GeneHancer, but the claim is published against decontaminated,
# paired within a GC bin, and a score with no constraint information adjudicates nothing
# between two constraint scores. The truth set's GC skew gives PUBLISHED a tailwind, so
# Supporting Fig. 8I is conservative without a third classifier.
def _score_column(key: str) -> str:
    return f"z_{key}"


def _threshold_setup(threshold: float, cache_dir: str, neutral_windows_bed: str | None,
                     refit_expected: str | None, gc_bins: list, min_n: int,
                     match_call_rate: bool, reference_score: str,
                     call_rate: float | None = None):
    """
    The labelled table, the drawn bins and each score's threshold -- shared by
    threshold_metrics and paired_deltas so a panel and its interval cannot disagree about
    which windows are called.
    """
    if reference_score not in PR_SCORES:
        raise ValueError(f"reference_score={reference_score!r} is not one of {list(PR_SCORES)}")

    df = _lax_labelled_windows(cache_dir, neutral_windows_bed, refit_expected)
    df = _assign_gc_bins(df, gc_bins)

    # Drawn bins decided before matching, so the matched fraction is the one drawn.
    drawn = []
    for b, (lo, hi) in enumerate(gc_bins):
        n = df.filter(pl.col("gc_bin") == b).height
        if n < min_n:
            print(f"  GC ({lo}, {hi}] dropped, n = {n:,} < {min_n:,}")
        else:
            drawn.append(b)
    df = df.filter(pl.col("gc_bin").is_in(drawn))

    keys = list(PR_SCORES)
    # `call_rate` sets the REFERENCE score's threshold by quantile instead of taking the
    # absolute z: "the top q" rather than "Gnocchi >= 4". That is how Supporting Fig. 8's
    # D-I are built (LAX_CALL_RATES); A and B keep the absolute z, Chen et al.'s own cutoff.
    if call_rate is not None:
        threshold = float(np.quantile(df[f"z_{reference_score}"].to_numpy(), 1.0 - call_rate))
        print(f"  calling rate {100 * call_rate:.2f}% sets {reference_score} at "
              f"z >= {threshold:.3f}")
    thresholds = {k: threshold for k in keys}
    target = float("nan")
    if match_call_rate:
        ref = df[f"z_{reference_score}"].to_numpy()
        target = float((ref >= threshold).mean())
        for key in keys:
            if key == reference_score:
                continue
            thresholds[key] = float(
                np.quantile(df[_score_column(key)].to_numpy(), 1.0 - target))
        print(f"  matched calling rate: {100 * target:.3f}% of the {df.height:,} windows "
              f"drawn, set by {reference_score} at z >= {threshold:g}")
        for key, t in thresholds.items():
            print(f"    {PR_SCORES[key][2]:<15} z >= {t:.3f}")
    return df, drawn, thresholds, target


def _odds_ratio(p: float, r: float) -> float:
    """
    LR+ as an odds ratio: odds that a called window is positive over the odds in the bin
    at large (equal to TPR/FPR -- see `lr_pos` in threshold_metrics). The base rate
    cancels, so LR+ compares ACROSS bins whose prevalence spans 7.7x; lift (/r) and skill
    (/(1 - r)) disagree about the SIGN of the trend over that span, so neither is a
    prevalence correction.
    """
    if not 0 < p < 1 or not 0 < r < 1:
        return float("inf") if p >= 1 else float("nan")
    return (p / (1 - p)) / (r / (1 - r))


# THE LEFT-TAIL NOTE: why the left tail is not read as a discovery problem (the notebook
# and fig5/README.md point here; the code that did so was deleted 2026-09-11).
#
# Supporting Fig. 8 once called z <= t and counted a NON-enhancer as the hit, on the
# grounds that a low Gnocchi claims a window is unconstrained. IT DOES NOT: Gnocchi is a
# two-sided z, so UNCONSTRAINED SEQUENCE SITS AT z ~ 0 and the left tail is the OPPOSITE
# anomaly, MORE variation than expected -- hypermutability or model misspecification,
# not absent selection. Under the null the 1st percentile is z = -2.33; published's
# bottom-1% cutoffs run -5.67, -6.04, -5.03, -4.14, -2.98. There is no truth set for that
# tail, so DO NOT REBUILD IT; a DNM-rate test would need no labels.
#
# THE SWITCH COST NO EVIDENCE, ONLY MAGNITUDE. With a = P(z <= t | enhancer) and
# b = P(z <= t | non-enhancer), the retired reading was b/a and the kept one (call_rate =
# 0.99) is (1 - a)/(1 - b): at a matched rate in a bin, monotone in the same single free
# count (paired_deltas), so they order the scores identically in every bin and replicate.
# Only the numbers compress: 42.2% of effect against 0.53%, 5/5 bins agreeing.


def _bin_thresholds(sub: pl.DataFrame, keys, target: float) -> dict:
    """
    Per-BIN thresholds: each score's own quantile at 1 - target within this bin, so every
    score calls the same fraction OF THIS BIN.

    The global matching in _threshold_setup equalises each score's calling fraction over
    the WHOLE population, not per bin -- the per-bin difference IS the bias (Fig. 5F) -- so
    without this a per-bin comparison is made at two DIFFERENT operating points: in the
    top GC bin published calls 13.97% and the retrained score 0.82%.

    It is how 8C (order-1% auPRC differences, one significant bin NEGATIVE) and the large
    top-1% gains were reconciled: both are right, the scores' curves crossing along the
    recall axis. Note it describes a score nobody uses -- published at z >= 4 calls 14% of
    GC-rich sequence. Supporting Fig. 8I says what the score CONTAINS; Fig. 5F what
    happens when it is USED.
    """
    q = 1.0 - target
    return {k: float(np.quantile(sub[_score_column(k)].to_numpy(), q)) for k in keys}


def threshold_metrics(threshold: float = GNOCCHI_THRESHOLD, truth_set: str = "lax",
                      cache_dir: str = CACHE_DIR,
                      neutral_windows_bed: str | None = config.NEUTRAL_WINDOWS_BED,
                      refit_expected: str | None = None, gc_bins: list | None = None,
                      min_n: int = THRESHOLD_MIN_BIN_WINDOWS,
                      match_call_rate: bool = True,
                      reference_score: str = "published",
                      match_within_bin: bool = False,
                      call_rate: float | None = None,
                      ) -> pl.DataFrame:
    """
    Precision, recall and calling rate at a threshold, per GC bin, for both scores. Feeds
    Supporting Fig. 8's A, B and D-I, and the numbers behind them.

    WHY A THRESHOLD CHANGES WHAT IS MEASURED. A GC-dependent bias is nearly a common shift
    within a narrow bin, so it cannot change a within-bin RANKING (why 8C's curves nearly
    coincide). At a fixed threshold it stops cancelling: it decides how many windows per
    bin are CALLED. That is also the analyst's question -- given Gnocchi >= 4, what is
    P(constrained | called), and does it depend on GC?

    Per bin g and score:

        call_rate(g)  = P(z >= t | g)                 -- exposure to the bias; no labels
        precision(g)  = P(constrained | z >= t, g)    -- the analyst's number
        recall(g)     = P(z >= t | constrained, g)
        lift(g)       = precision(g) / r(g)           -- within-bin only: ceiling 1/r
        lr_pos(g)     = P(call | Y=1) / P(call | Y=0) -- base rate cancels: compares
                                                        BETWEEN bins
        skill(g)      = (precision - r) / (1 - r)

    DRAWN: recall and lr_pos (8A, 8B), the per-bin thresholds (8D-F). The rest are printed
    only; precision and lift are what the caption quotes. Precision need NOT be flat even
    for a perfect score, r(g) climbing ~7.7x across these bins, and a residual slope after
    debiasing is signal-to-noise (8C's finding), not residual bias.

    MATCHED CALLING RATE, NOT A COMMON NUMBER. Retraining moves the whole z distribution:
    at Gnocchi >= 4 published calls ~1.00% and the retrained score ~0.13%. So
    `reference_score` is held at `threshold` and every other score gets the quantile of its
    own z calling the SAME fraction of the DRAWN bins (`threshold_used`).
    match_call_rate=False gives the naive common-threshold comparison, for understanding
    the confound only. `call_rate` replaces `threshold` as the anchor -- the reference
    cutoff becomes its own quantile at 1 - call_rate -- which is how D-I are built; pass one
    of LAX_CALL_RATES. match_within_bin matches per bin instead (_bin_thresholds).

    UNBALANCED, unlike 8C: the base rate an analyst faces is the real one. Wilson
    intervals, per curve. A CALL IS ALWAYS z >= t AND A HIT ALWAYS AN ENHANCER (see the
    left-tail note).

    Returns one row per (GC bin, score): lo, hi, mid, score, threshold_used, n, n_pos,
    r, n_called, call_rate, precision, recall, lift, skill, lr_pos, the last six each with
    _lo/_hi bounds.
    """
    if truth_set != "lax":
        raise ValueError(f"truth_set={truth_set!r}: only 'lax' is built.")
    gc_bins = THRESHOLD_GC_BINS if gc_bins is None else gc_bins

    df, drawn, thresholds, target = _threshold_setup(
        threshold, cache_dir, neutral_windows_bed, refit_expected, gc_bins, min_n,
        match_call_rate, reference_score, call_rate=call_rate)

    if match_within_bin:
        print(f"  MATCHING WITHIN EACH BIN at {100 * target:.3f}%: every score calls that "
              "fraction of every bin, so per-bin comparisons are at one operating point "
              "-- see _bin_thresholds.")

    rows = []
    for b in drawn:
        lo, hi = gc_bins[b]
        sub = df.filter(pl.col("gc_bin") == b)
        if match_within_bin:
            thresholds = _bin_thresholds(sub, list(thresholds), target)
        y = sub[TRUTH_TARGET].to_numpy()
        n_pos = int(y.sum())
        for key in thresholds:
            display, short = PR_SCORES[key][1], PR_SCORES[key][2]
            called = sub[_score_column(key)].to_numpy() >= thresholds[key]
            n_called = int(called.sum())
            tp = int((called & y).sum())
            prec_lo, prec_hi = _wilson(tp, n_called)
            rec_lo, rec_hi = _wilson(tp, n_pos)
            call_lo, call_hi = _wilson(n_called, sub.height)
            r = n_pos / sub.height
            rows.append({
                "lo": lo, "hi": hi, "mid": 0.5 * (lo + hi), "score": key,
                "display": display, "short": short,
                "threshold_used": thresholds[key],
                "n": sub.height, "n_pos": n_pos, "r": r,
                "n_called": n_called, "call_rate": n_called / sub.height,
                "call_rate_lo": call_lo, "call_rate_hi": call_hi,
                "precision": tp / n_called if n_called else float("nan"),
                "precision_lo": prec_lo, "precision_hi": prec_hi,
                "recall": tp / n_pos if n_pos else float("nan"),
                "recall_lo": rec_lo, "recall_hi": rec_hi,
                "lift": (tp / n_called) / r if n_called and r else float("nan"),
                # CEILING-FREE COMPANIONS TO LIFT, for cross-bin statements: lift <= 1/r,
                # a ceiling falling 12.0 -> 1.57 over these bins. skill normalises by the
                # headroom (0 = random, 1 = perfect); lr_pos, below, cancels r outright.
                "skill": ((tp / n_called) - r) / (1 - r)
                         if n_called and r < 1 else float("nan"),
                # Within a bin r is a constant, so skill, lift and LR+ are each monotone in
                # precision and its Wilson bounds map through exactly -- no second bootstrap.
                "skill_lo": (prec_lo - r) / (1 - r) if n_called and r < 1 else float("nan"),
                "skill_hi": (prec_hi - r) / (1 - r) if n_called and r < 1 else float("nan"),
                "lift_lo": (prec_lo / r) if n_called and r else float("nan"),
                "lift_hi": (prec_hi / r) if n_called and r else float("nan"),
                "lr_pos": ((tp / n_pos) / ((n_called - tp) / (sub.height - n_pos)))
                          if n_pos and (n_called - tp) and sub.height > n_pos
                          else float("nan"),
                # LR+ IS AN ODDS RATIO: with p = tp/n_called and r = n_pos/n,
                #     LR+ = [p / (1 - p)] / [r / (1 - r)],
                # increasing in p, hence the bounds below (e.g. p = 0.718, r = 0.639 -> 1.44).
                #
                # AT A CALLING RATE k NEAR 1 IT IS PINNED NEAR 1 BY ARITHMETIC -- read
                # Supporting Fig. 8G with this in mind. p is forced to r, and
                #     LR+ = 1 + (recall - k)/(1 - r) + O((1 - k)^2),
                # bounded to first order by 1 + (1 - k)/(1 - r), so a ratio of two LR+ values
                # there sits within ~1% of 1.0 however sharp the scores. Only the magnitude
                # is compressed, not the significance (the one-free-count argument in
                # paired_deltas).
                "lr_pos_lo": _odds_ratio(prec_lo, r) if n_called and 0 < r < 1
                             else float("nan"),
                "lr_pos_hi": _odds_ratio(prec_hi, r) if n_called and 0 < r < 1
                             else float("nan"),
            })
            e = rows[-1]
            print(f"  GC ({lo:.2f}, {hi:.2f}]  {short:<15} called {n_called:>7,} "
                  f"({100 * e['call_rate']:5.2f}%)  precision {e['precision']:.3f}  "
                  f"base rate {r:.3f}  lift {e['lift']:.2f}  skill {e['skill']:+.3f}  "
                  f"LR+ {e['lr_pos']:.2f}  recall {100 * e['recall']:.2f}%")
    return pl.DataFrame(rows)


def paired_deltas(threshold: float = GNOCCHI_THRESHOLD, truth_set: str = "lax",
                  cache_dir: str = CACHE_DIR,
                  neutral_windows_bed: str | None = config.NEUTRAL_WINDOWS_BED,
                  refit_expected: str | None = None, gc_bins: list | None = None,
                  min_n: int = THRESHOLD_MIN_BIN_WINDOWS, match_call_rate: bool = True,
                  reference_score: str = "published", n_bootstrap: int = 500,
                  seed: int = 0, call_rate: float | None = None,
                  match_within_bin: bool = False,
                  metric: str = "lr_pos") -> pl.DataFrame:
    """
    The PAIRED difference between the two scores in one effect measure, per GC bin, with a
    bootstrap interval. Supporting Fig. 8G, 8H and 8I plot the ratio and this interval.

    `metric` DEFAULTS TO "lr_pos" BECAUSE THE PANELS ARE CURVES ACROSS BINS. Lift
    (= precision / r) is the right control WITHIN a bin but has a ceiling 1/r that moves
    with the base rate -- 12.0 at r = 0.083, 1.57 at r = 0.639 -- so a lift falling 2.64 to
    1.14 across GC (the retrained score at panel B's cutoff) is partly that ceiling coming
    down. LR+ = P(call|Y=1)/P(call|Y=0) has no such ceiling. "lift" keeps the older measure.

    Either way r cancels from the RATIO, in every replicate too, both scores seeing the
    same rows:

        lift_scored / lift_published = precision_scored / precision_published
        LR+_scored / LR+_published   = odds(precision_scored) / odds(precision_published)

    Reported as ratio minus one, like pr_curve_deltas. The two disagree on magnitude, never
    sign: the odds ratio amplifies where precision is high (0.73 in the top bin), so every
    LR+ gain exceeds its lift counterpart -- a property of the measure, not evidence.

    ONE FREE COUNT. At a matched rate inside a bin, n, n_pos and n_called are fixed and
    shared, so the 2x2 table has a single free count and precision, lift, skill and LR+
    are all increasing in it: they agree on which score is ahead in every bin and replicate,
    and differ only in dynamic range and cross-bin comparability. The retired left-tail
    reading was another monotone function of the same count (the left-tail note).

    PAIRED as in pr_curve_deltas: one index vector per replicate, both scores scored on it.
    THRESHOLDS ARE HELD FIXED across replicates, being quantiles of ~10^6 windows.

    Returns one row per drawn bin: lo, hi, mid, n, n_pos, r, ceiling (1/r),
    threshold_published, threshold_scored, n_called_*, both measures' levels whatever
    `metric` is (lift_*, lr_pos_*), `metric`, delta, n_boot (surviving replicates), ci_lo,
    ci_hi, p_gt0. `delta` and its interval always belong to `metric`; the level columns are
    for inspection. Keep both pairs: with the rate matched in a bin, lift_scored /
    lift_published IS the precision ratio the caption quotes (1.30x in the most AT-rich
    bin), smaller than the odds ratio `delta` reports.
    """
    if truth_set != "lax":
        raise ValueError(f"truth_set={truth_set!r}: only 'lax' is built.")
    gc_bins = THRESHOLD_GC_BINS if gc_bins is None else gc_bins
    df, drawn, thresholds, target = _threshold_setup(
        threshold, cache_dir, neutral_windows_bed, refit_expected, gc_bins, min_n,
        match_call_rate, reference_score, call_rate=call_rate)
    if match_within_bin:
        print(f"  MATCHING WITHIN EACH BIN at {100 * target:.3f}% "
              "-- see _bin_thresholds")

    other = [k for k in PR_SCORES if k != reference_score]
    if len(other) != 1:
        raise ValueError("paired_deltas compares exactly two scores")
    if metric not in ("lift", "lr_pos"):
        raise ValueError(f"metric={metric!r}: 'lift' or 'lr_pos'")
    other = other[0]

    rng = np.random.default_rng(seed)
    rows = []
    for b in drawn:
        lo, hi = gc_bins[b]
        sub = df.filter(pl.col("gc_bin") == b)
        y = sub[TRUTH_TARGET].to_numpy()
        zr = sub[f"z_{reference_score}"].to_numpy()
        zo = sub[f"z_{other}"].to_numpy()
        r = float(y.mean())

        def _prec(mask, yy):
            k = int(mask.sum())
            return (yy[mask].sum() / k) if k else np.nan

        # The reported ratio, as a function of the two precisions; both forms cancel r.
        def _ratio(p_lo, p_hi):
            if metric == "lift":
                return p_hi / p_lo
            if not 0 < p_lo < 1 or not 0 < p_hi < 1:
                return np.nan
            return (p_hi / (1 - p_hi)) / (p_lo / (1 - p_lo))

        thr = _bin_thresholds(sub, [reference_score, other], target) \
            if match_within_bin else thresholds
        cr = zr >= thr[reference_score]
        co = zo >= thr[other]
        p_ref, p_oth = _prec(cr, y), _prec(co, y)

        boot = np.empty(n_bootstrap)
        for k in range(n_bootstrap):
            idx = rng.integers(0, sub.height, sub.height)   # one draw, both scores
            yb = y[idx]
            a, c = _prec(cr[idx], yb), _prec(co[idx], yb)
            boot[k] = (_ratio(a, c) - 1.0) if (a and np.isfinite(a) and np.isfinite(c)) \
                else np.nan
        boot = boot[np.isfinite(boot)]
        # A SATURATED PRECISION (0 or 1) LEAVES NO ODDS RATIO: _ratio returns nan, and if
        # enough replicates saturate there is nothing left to take a percentile of. Unlikely
        # with an enhancer as the hit, but report it as a missing interval rather than
        # raising; n_boot says how much of the resampling survived.
        ci_lo, ci_hi, p_gt0 = (float(np.percentile(boot, 2.5)),
                               float(np.percentile(boot, 97.5)),
                               float((boot > 0).mean())) if boot.size else \
                              (float("nan"), float("nan"), float("nan"))

        rows.append({
            "lo": lo, "hi": hi, "mid": 0.5 * (lo + hi), "n": sub.height,
            "call_rate": call_rate if call_rate is not None else float("nan"),
            "n_pos": int(y.sum()), "r": r, "ceiling": 1.0 / r if r else np.nan,
            "threshold_published": thr[reference_score],
            "threshold_scored": thr[other],
            "n_called_published": int(cr.sum()), "n_called_scored": int(co.sum()),
            "lift_published": p_ref / r, "lift_scored": p_oth / r,
            "lr_pos_published": _odds_ratio(p_ref, r),
            "lr_pos_scored": _odds_ratio(p_oth, r),
            "metric": metric,
            "delta": _ratio(p_ref, p_oth) - 1.0,
            "n_boot": int(boot.size),
            "ci_lo": ci_lo, "ci_hi": ci_hi, "p_gt0": p_gt0,
        })
        e = rows[-1]
        star = " *" if (e["ci_lo"] > 0 or e["ci_hi"] < 0) else "  "
        if boot.size < n_bootstrap:
            star += f" [{n_bootstrap - boot.size} replicate(s) saturated]"
        name = "LR+" if metric == "lr_pos" else "lift"
        print(f"  GC ({lo:.2f}, {hi:.2f}]  {name} {e[f'{metric}_published']:.2f} -> "
              f"{e[f'{metric}_scored']:.2f}  (lift ceiling {e['ceiling']:.1f})  "
              f"gain {100 * e['delta']:+6.1f}%  "
              f"[{100 * e['ci_lo']:+6.1f}, {100 * e['ci_hi']:+6.1f}]  "
              f"P(>0) = {e['p_gt0']:.3f}{star}")
    return pl.DataFrame(rows)


def budget_comparison(threshold: float = GNOCCHI_THRESHOLD, truth_set: str = "lax",
                      cache_dir: str = CACHE_DIR,
                      neutral_windows_bed: str | None = config.NEUTRAL_WINDOWS_BED,
                      refit_expected: str | None = None, gc_bins: list | None = None,
                      min_n: int = THRESHOLD_MIN_BIN_WINDOWS,
                      reference_score: str = "published") -> pl.DataFrame:
    """
    The GENOME-WIDE comparison at a fixed calling budget. Not a panel: a table of three
    arms (published, scored, random) that settles what the UNCONDITIONAL precision-recall
    of this truth set measures, and why published wins it.

    At a fixed budget precision = TP/N_called and recall = TP/P are both monotone in TP, so
    there is one quantity. TP is maximised by ranking on P(Y=1 | window), so any covariate
    correlated with the label raises it -- and here the base rate climbs ~7.7x with GC,
    while published is precisely the score that calls GC-rich sequence constrained.
    PUBLISHED WINS THIS TABLE BECAUSE ITS BIAS ACTS AS AN ENHANCER DETECTOR: 5,485
    positives against 4,396 at a common budget. A real cost, but not evidence against the
    correction -- judging a debiasing by a statistic the bias inflates is circular.

    The same skew makes the WITHIN-bin comparison conservative: positives stay enriched at
    the high-GC end of every bin, so published keeps a tailwind there, and the
    decontaminated score wins every bin at the top 1% anyway. Read this table beside
    Fig. 5F, never instead of it.

    Returns one row per arm: n_called, tp, precision, recall, lift. `random` is the
    analytic expectation, budget x base rate.
    """
    if truth_set != "lax":
        raise ValueError(f"truth_set={truth_set!r}: only 'lax' is built.")
    gc_bins = THRESHOLD_GC_BINS if gc_bins is None else gc_bins
    df, drawn, thresholds, _ = _threshold_setup(
        threshold, cache_dir, neutral_windows_bed, refit_expected, gc_bins, min_n,
        match_call_rate=True, reference_score=reference_score)

    y = df[TRUTH_TARGET].to_numpy()
    n, n_pos = df.height, int(y.sum())
    r = n_pos / n
    budget = int((df[_score_column(reference_score)].to_numpy()
                  >= thresholds[reference_score]).sum())
    print(f"  budget: {budget:,} calls ({100 * budget / n:.2f}% of {n:,} windows); "
          f"{n_pos:,} positives ({100 * r:.1f}%)")

    rows = []
    for key in list(thresholds) + ["random"]:
        if key == "random":
            tp, n_called = budget * r, budget
        else:
            called = df[_score_column(key)].to_numpy() >= thresholds[key]
            n_called, tp = int(called.sum()), float(y[called].sum())
        rows.append({
            "score": key,
            "display": PR_SCORES[key][1] if key in PR_SCORES else "random",
            "n_called": n_called, "tp": tp,
            "precision": tp / n_called if n_called else float("nan"),
            "recall": tp / n_pos if n_pos else float("nan"),
            "lift": (tp / n_called) / r if n_called and r else float("nan"),
        })
        e = rows[-1]
        print(f"    {e['display']:<32} calls {e['n_called']:>7,}  TP {e['tp']:>8,.0f}  "
              f"precision {e['precision']:.3f}  recall {100 * e['recall']:5.2f}%  "
              f"lift {e['lift']:.2f}")
    return pl.DataFrame(rows)
