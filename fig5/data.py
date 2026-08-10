"""
Data for the five panels of Fig. 5. Every builder caches its result as parquet in
fig5/output/, so the notebook is instant after the first pass.

Inputs, in three groups:

  * the public gnomAD-NC-constraint bucket, downloaded on demand into published/ by
    gnocchi_bias.windows.download;
  * the three refits fig5/refit.py produces (full / scored / sizematched), which live in
    the repo-root refits/ -- ONE copy, also read directly by dnm_training_size/;
  * two files that are NOT in this repo, read from fig5/config.py (NOT set in the
    notebook -- refit.py reads the same module, and the two must agree) -- a
    depletion-rank BED (panel A's third curve) and a GeneHancer BED (the enhancer half
    of the "neutral" window definition). Both default to None; the figure builds
    without them.
"""
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
    A refit table, verified to have been built under the CURRENT config.GENEHANCER_BED.
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
    """A memory-capped duckdb connection. Public because diagnostics.py uses it too."""
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
                 genehancer_bed: str | None = config.GENEHANCER_BED) -> pl.DataFrame:
    """
    The analyzed window population: noncoding, pass_qc, autosome/PAR, with GC content
    as a 0-1 fraction. This is both the test set Gnocchi is scored on and (in panels
    C-E) the population the retrained adjustment is fit on.

    The default comes from fig5/config.py, which fig5/refit.py reads too -- so the
    population fit on and the population scored on cannot drift apart. Do not pass this
    explicitly unless you also rerun the refits with the same value.
    """
    return W.build_window_table(cache_dir, genehancer_bed=genehancer_bed)


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
    Per-window expected-count components: e1, e2 (totals over all 32 contexts) and
    e1_cpg, e2_cpg (the same sums restricted to ACG/CCG/GCG/TCG). Non-CpG is then
    e1 - e1_cpg, which is why only the CpG slice of the two multi-GB per-context files
    is ever joined -- an 85M x 85M join becomes 10M x 10M.

    The published pipeline writes its per-context r to a local dir, not the bucket, so
    the refit's rr table stands in. That substitution is validated per GC bin in
    r_eff_by_gc, since the published r_eff is separately computable as E2/E1.
    """
    ctx = ", ".join(f"'{c}'" for c in M.CPG_CONTEXTS)
    percontext = W.download(M.GENOME_EXPECTED_PERCONTEXT_FILE, cache_dir)
    step1 = W.download(W.REMOTE_FILES["step1_expected"], cache_dir)
    query = f"""
        WITH cpg AS (
            SELECT e.element_id AS element_id,
                   SUM(e.expected) AS e1_cpg,
                   SUM(e.expected * COALESCE(r.rr, 1.0)) AS e2_cpg
            FROM (SELECT element_id, context, expected
                  FROM read_csv_auto('{percontext}', delim='\t', header=True)
                  WHERE context IN ({ctx})) e
            LEFT JOIN (SELECT element_id, context, rr
                       FROM read_csv_auto('{refit_path("rr", pop, refits_dir)}',
                                          delim='\t', header=True)
                       WHERE context IN ({ctx})) r
              ON e.element_id = r.element_id AND e.context = r.context
            GROUP BY e.element_id)
        SELECT t1.element_id AS element_id, t1.expected AS e1, t2.expected AS e2,
               COALESCE(cpg.e1_cpg, 0.0) AS e1_cpg, COALESCE(cpg.e2_cpg, 0.0) AS e2_cpg
        FROM read_csv_auto('{step1}', delim='\t', header=True) t1
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


# ------------------------------------------------------------------- panel C

def dnm0_composition(edges: np.ndarray, cache_dir: str = CACHE_DIR,
                     coding_prop_threshold: float = 0.0, force: bool = False,
                     memory_limit: str = "8GB") -> pl.DataFrame:
    """
    Panel C's table. Each GC bin's non-CpG background training sites, mapped to their
    containing 1 kb tile and split three ways:

        n_analyzed  tile passes pass_qc, coding_prop <= threshold, autosome/PAR
        n_coding    tile is annotated but fails one of those
        n_noannot   tile has no row in the constraint table at all (no gnomAD coverage)

    These partition the total exactly, which is asserted rather than assumed.
    """
    def build() -> pl.DataFrame:
        b = sql_bin_expr("ft.GC_content_1k / 100.0", edges)
        ctx = ", ".join(f"'{c}'" for c in M.CPG_CONTEXTS)
        features = W.download(W.REMOTE_FILES["features"], cache_dir)
        annot = W.download(W.REMOTE_FILES["annot"], cache_dir)
        dnm0 = W.download(M.TRAINING_FILES["dnm0_sites"], cache_dir)
        query = f"""
            WITH ft AS (SELECT element_id, GC_content_1k
                        FROM read_csv_auto('{features}', delim='\t', header=True)),
            an AS (SELECT element_id, pass_qc, coding_prop
                   FROM read_csv_auto('{annot}', delim='\t', header=True)),
            d0 AS (SELECT context, {ELEMENT_ID_FROM_LOCUS} AS element_id
                   FROM read_csv_auto('{dnm0}', delim='\t', header=True)
                   WHERE locus NOT LIKE 'chrX:%')
            SELECT {b} AS gc_bin, COUNT(*) AS n_total,
                   CAST(SUM(CASE WHEN an.pass_qc
                                  AND an.coding_prop <= {coding_prop_threshold}
                                  AND ft.element_id NOT LIKE 'chrX-%'
                                  AND ft.element_id NOT LIKE 'chrY-%'
                             THEN 1 ELSE 0 END) AS BIGINT) AS n_analyzed,
                   CAST(SUM(CASE WHEN an.element_id IS NOT NULL
                                  AND NOT (an.pass_qc
                                           AND an.coding_prop <= {coding_prop_threshold})
                             THEN 1 ELSE 0 END) AS BIGINT) AS n_coding,
                   CAST(SUM(CASE WHEN an.element_id IS NULL
                             THEN 1 ELSE 0 END) AS BIGINT) AS n_noannot
            FROM d0 JOIN ft ON d0.element_id = ft.element_id
            LEFT JOIN an ON d0.element_id = an.element_id
            WHERE d0.context NOT IN ({ctx})
            GROUP BY 1
        """
        df = duck(memory_limit).execute(query).pl().sort("gc_bin")
        resid = df["n_total"] - df["n_analyzed"] - df["n_coding"] - df["n_noannot"]
        assert int(resid.abs().sum()) == 0, "composition categories do not partition the total"
        return df.with_columns([
            bin_centres(edges, df["gc_bin"]),
            (pl.col("n_analyzed") / pl.col("n_total")).alias("frac_analyzed"),
            (pl.col("n_coding") / pl.col("n_total")).alias("frac_coding"),
            (pl.col("n_noannot") / pl.col("n_total")).alias("frac_noannot"),
        ])

    # The bin count is in the cache name because this table is keyed by gc_bin: unlike
    # the per-window r_eff components, it would be silently stale if N_BINS changed.
    df = cached(f"dnm0_window_composition.{len(edges) - 1}bins.parquet", build, force)
    # Deliberately no fraction range here: the lowest-GC bins hold a handful of sites
    # that are essentially all uncovered, so an unrestricted min() reads 0.00 and would
    # get copied into a caption. The notebook reports the range over the plotted bins.
    print(f"dnm0 composition: {int(df['n_total'].sum()):,} non-CpG background sites "
          f"over {df.height} GC bins")
    return df


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
