"""
Depletion Rank (Halldorsson et al. 2022) as the third curve of panel A.

A SEPARATE WINDOW SET, NOT A COLUMN. Depletion rank is defined on Halldorsson's own
windows, not Chen's 1 kb windows, and McHale et al.'s regression notebook keeps them as
separate files. So this curve is not an element_id join onto the Gnocchi table: it is an
independent window set, ranked within itself, binned by its own GC content, and
overlaid. That is legitimate for a conditional-mean-rank plot precisely because the rank
statistic is uniform on (0,1) by construction for every curve -- the comparison is about
how each metric's uniform mass redistributes across GC, not about absolute score values.
The caption must say the DR curve comes from a different window set.

SIGN CONVENTION, the easy thing to get backwards. Gnocchi's z is signed so HIGH means
constrained; depletion rank runs the other way (LOW means more depleted, i.e. more
constrained). experiment.1.ipynb takes the complement, 1 - DR, and so does this module by
default. Get it wrong and the curve flips upside down.

STATUS: the input file is not in this repo (it lives on the constraint-tools
CONSTRAINT_TOOLS_DATA path), so the parsing below is defensive -- explicit column
resolution, GC unit auto-detection, loud errors -- but has NEVER been run against the
real file. Check the printed summary the first time it runs.
"""
import os

import polars as pl

from gnocchi_bias import windows as W

DR_NAME_CANDIDATES = ["depletion_rank", "DR", "depletionRank", "dr"]
GC_NAME_CANDIDATES = ["GC_content", "GC_content_1000bp", "pct_gc", "gc_content", "GC"]


def _resolve(df: pl.DataFrame, requested: str | None, candidates: list[str], role: str) -> str:
    """An explicit name wins and must exist; else the first candidate present wins.
    Raises with the full column list, rather than a KeyError deep inside binning."""
    if requested is not None:
        if requested not in df.columns:
            raise KeyError(f"{role} column {requested!r} not in file; columns are {df.columns}")
        return requested
    for name in candidates:
        if name in df.columns:
            return name
    raise KeyError(f"no {role} column found automatically (tried {candidates}); pass it "
                   f"explicitly. File columns are: {df.columns}")


def load_depletion_rank_windows(bed_path: str, dr_col: str | None = None,
                                gc_col: str | None = None, complement: bool = True,
                                has_header: bool = True) -> pl.DataFrame:
    """
    Returns gc (0-1 fraction), constraint (oriented HIGHER == MORE CONSTRAINED), and
    rank_dr, the standardized rank (rank - 0.5)/n computed WITHIN this window set --
    exactly as for the Gnocchi curves.

    GC units are detected, not assumed: McHale et al. compute GC with `bedtools nuc`,
    whose pct_gc column is already a 0-1 fraction, while this repo's own
    genomic_features13 columns are 0-100 percentages, and this may be pointed at either.
    """
    if not os.path.exists(bed_path):
        raise FileNotFoundError(
            f"depletion-rank BED not found: {bed_path}\nThis file is not in the repo -- "
            "it comes from the constraint-tools CONSTRAINT_TOOLS_DATA path.")

    df = pl.read_csv(bed_path, separator="\t", has_header=has_header,
                     infer_schema_length=10000, null_values=["", ".", "NA", "NaN"])
    dr_name = _resolve(df, dr_col, DR_NAME_CANDIDATES, "depletion-rank")
    gc_name = _resolve(df, gc_col, GC_NAME_CANDIDATES, "GC-content")
    df = df.select([pl.col(dr_name).cast(pl.Float64).alias("_dr"),
                    pl.col(gc_name).cast(pl.Float64).alias("_gc")]).drop_nulls()

    gc_max = df["_gc"].max()
    as_percent = gc_max is not None and gc_max > 1.5
    df = df.with_columns([
        (pl.col("_gc") / 100.0 if as_percent else pl.col("_gc")).alias("gc"),
        (1.0 - pl.col("_dr") if complement else pl.col("_dr")).alias("constraint"),
    ])
    df = df.with_columns(((pl.col("constraint").rank() - 0.5) / df.height).alias("rank_dr"))

    print(f"depletion rank: {df.height:,} windows from {os.path.basename(bed_path)}")
    print(f"  score column = {dr_name!r}" + ("  (complemented: 1 - DR)" if complement else ""))
    print(f"  GC column    = {gc_name!r}, max {gc_max:.3g}, "
          f"{'divided by 100' if as_percent else 'used as a fraction'}")
    print(f"  GC range     = {df['gc'].min():.3f} - {df['gc'].max():.3f}")
    return df.select(["gc", "constraint", "rank_dr"])


def bin_depletion_rank(df_dr: pl.DataFrame, n_bins: int = 20) -> pl.DataFrame:
    """GC-bin the DR windows with the same code the Gnocchi curves use, so the curves
    are directly comparable. Returns bin_by_gc's shape: n, gc_mid, mean_dr, se_dr."""
    return W.bin_by_gc(df_dr, "gc", n_bins, "fixed", value_cols={"dr": "rank_dr"})
