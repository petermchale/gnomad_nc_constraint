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

NOT THE ONLY SOURCE, and the other one is a trap. config.NEUTRAL_WINDOWS_BED --
constraint-tools' `Supplementary_Data_2.features.constraint_scores.bed` -- carries
`depletion_rank_constraint_score_complement`, i.e. depletion rank mapped onto Chen et
al.'s own 1 kb windows and ALREADY complemented. Reading that column here would be wrong
twice over: it would flip an already-oriented score (mirroring the curve about y = 0.5),
and it would rank a per-Chen-window quantity within a set this module treats as
independent. Using it properly means joining it onto the window table so all three of
panel A's curves share one population and one set of GC bins -- a different code path,
deliberately not taken (2026-08-21), to keep the panel matching how McHale et al.'s own
notebook treats the two files. `_resolve` raises rather than let the first mistake
happen silently.

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

# A column already oriented as a constraint score -- high == more constrained -- rather
# than as a raw depletion rank. config.NEUTRAL_WINDOWS_BED carries one,
# `depletion_rank_constraint_score_complement`: depletion rank mapped onto Chen et al.'s
# 1 kb windows and complemented there. It is deliberately NOT in DR_NAME_CANDIDATES (this
# module reads Halldorsson's own windows, see the docstring), and it must never be read
# with complement=True, which would flip it back. _resolve refuses that combination
# rather than trusting the caller, because a double complement produces a curve that is
# wrong but perfectly plausible -- mirrored about y = 0.5, which is where an unbiased
# metric sits.
ALREADY_COMPLEMENTED = "complement"


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
                                has_header: bool = True,
                                exclude_enhancer_windows: bool = True) -> pl.DataFrame:
    """
    Returns gc (0-1 fraction), constraint (oriented HIGHER == MORE CONSTRAINED), and
    rank_dr, the standardized rank (rank - 0.5)/n computed WITHIN this window set --
    exactly as for the Gnocchi curves.

    THE ENHANCER FILTER IS NOT OPTIONAL IN PRACTICE. McHale et al.'s
    `9.regression/experiment.1.ipynb` filters BOTH files to
    `window overlaps enhancer == False` before plotting, so leaving it off here would
    rank depletion rank over every window while the Gnocchi curves are ranked over
    enhancer-excluded ones. The panel compares how each metric's uniform mass
    redistributes across GC, so two populations would show up as a difference between
    metrics -- a population mismatch of precisely the kind this figure is about. Hence
    the default, and hence the raise when the column is absent rather than a quiet
    no-op.

    `constraint` here IS their `depletion_rank_constraint_score_complement`:
    complement=True computes `1 - depletion_rank`, the same quantity under a shorter
    name. Column names are normalized on read (`window overlaps enhancer` ->
    `window_overlaps_enhancer`), matching windows.load_mchale_neutral_element_ids, and
    an explicitly passed dr_col/gc_col is normalized the same way.

    GC units are detected, not assumed: McHale et al. compute GC with `bedtools nuc`,
    whose pct_gc column is already a 0-1 fraction, while this repo's own
    genomic_features13 columns are 0-100 percentages, and this may be pointed at either.
    """
    if not os.path.exists(bed_path):
        raise FileNotFoundError(
            f"depletion-rank BED not found: {bed_path}\nThis file is not in the repo -- "
            "it comes from the constraint-tools CONSTRAINT_TOOLS_DATA path.")

    df = (pl.read_csv(bed_path, separator="\t", has_header=has_header,
                      infer_schema_length=10000, null_values=["", ".", "NA", "NaN"])
            .rename(lambda c: c.strip().replace(" ", "_")))

    if exclude_enhancer_windows:
        if W.MCHALE_ENHANCER_COLUMN not in df.columns:
            raise ValueError(
                f"{bed_path} has no {W.MCHALE_ENHANCER_COLUMN!r} column (found "
                f"{df.columns[:8]}...), so the enhancer-overlapping windows cannot be "
                "excluded. McHale et al. exclude them from this file as well as from "
                "the Gnocchi window set, and ranking over the two different populations "
                "would put the difference into the curve. Pass "
                "exclude_enhancer_windows=False only if you mean to depart from that.")
        n_before = df.height
        df = df.filter(~pl.col(W.MCHALE_ENHANCER_COLUMN).cast(pl.Boolean))
        print(f"depletion rank: {n_before:,} windows -> {df.height:,} with "
              f"{W.MCHALE_ENHANCER_COLUMN} == False")

    norm = lambda c: c.strip().replace(" ", "_") if c else c        # noqa: E731
    dr_name = _resolve(df, norm(dr_col), DR_NAME_CANDIDATES, "depletion-rank")
    if ALREADY_COMPLEMENTED in dr_name.lower() and complement:
        raise ValueError(
            f"{dr_name!r} is already a constraint score (high == more constrained), so "
            "complementing it again would mirror the curve about y = 0.5. Pass "
            "complement=False -- and note this module ranks within THIS file's own "
            "windows, so a per-Chen-window column belongs in a join onto the window "
            "table, not here.")
    gc_name = _resolve(df, norm(gc_col), GC_NAME_CANDIDATES, "GC-content")
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
    print(f"  score column = {dr_name!r}"
          + ("  (complemented: 1 - DR, i.e. their "
             "depletion_rank_constraint_score_complement)" if complement else ""))
    print(f"  GC column    = {gc_name!r}, max {gc_max:.3g}, "
          f"{'divided by 100' if as_percent else 'used as a fraction'}")
    print(f"  GC range     = {df['gc'].min():.3f} - {df['gc'].max():.3f}")
    return df.select(["gc", "constraint", "rank_dr"])


def bin_depletion_rank(df_dr: pl.DataFrame, n_bins: int = 20) -> pl.DataFrame:
    """GC-bin the DR windows with the same code the Gnocchi curves use, so the curves
    are directly comparable. Returns bin_by_gc's shape: n, gc_mid, mean_dr, se_dr."""
    return W.bin_by_gc(df_dr, "gc", n_bins, "fixed", value_cols={"dr": "rank_dr"})
