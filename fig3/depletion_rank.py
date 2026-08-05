"""
Depletion Rank (Halldorsson et al. 2022) as a third curve on Fig. 3 panel A.

WHY THIS IS A SEPARATE MODULE, AND NOT A COLUMN ON THE CHEN WINDOW TABLE
-----------------------------------------------------------------------
Depletion rank is defined on Halldorsson's OWN windows, not on Chen's 1kb
windows. McHale et al.'s regression notebook
(constraint-tools/papers/neutral_models_are_biased/9.regression/experiment.1.ipynb)
keeps them as three separate window sets, read from three separate files:

    Chen windows        .../chen-et-al-2023-published-version/41586_2023_6045_MOESM4_ESM/
                            Supplementary_Data_2.features.constraint_scores.bed
    Halldorsson windows .../depletion_rank_scores/
                            41586_2022_4965_MOESM3_ESM.noncoding.enhancer.BGS.gBGC.GC_content.bed
    CDTS windows        .../CDTS/CDTS.gnomAD.hg38.noncoding.enhancer.BGS.gBGC.GC_content.bed

So the DR curve is NOT an element_id join onto the Gnocchi table. It is an
independent window set, ranked within itself, binned by its own GC content, and
overlaid on the same axes. That is legitimate for a conditional-mean-rank plot
precisely because the rank statistic is scale-free and uniform on (0,1) by
construction for every curve -- the comparison is about how each model's uniform
mass redistributes across GC, not about the scores' absolute values. It does
mean the Fig. 3 caption should say the DR curve comes from a different window
set (and a different window size) than the two Gnocchi curves.

SIGN CONVENTION -- the easy thing to get backwards
--------------------------------------------------
Gnocchi's z is signed so that HIGH z == constrained
(run_nc_constraint_gnomad_v31_main.py:278-280 negates the chi when observed >=
expected). Depletion rank runs the other way: a LOW depletion rank means the
window is more depleted of variation, i.e. more constrained. experiment.1.ipynb
handles this by taking the complement:

    (1-pl.col('depletion_rank')).alias('depletion_rank_constraint_score_complement')

so this module does the same by default (complement=True). Get this wrong and
the DR curve flips upside down and appears to disagree with the paper.

STATUS: the input file is not in this repo (it lives on the CONSTRAINT_TOOLS_DATA
HPC path and is not fetchable here), so the parsing below is written defensively
-- explicit column resolution, unit auto-detection, and loud errors -- but has
NOT been run against the real file. Check load_depletion_rank_windows()'s printed
summary against expectations the first time it runs.
"""
import os

import numpy as np
import polars as pl

# Column-name candidates, in preference order. The real file's header is
# unknown here; resolve_column() reports what it found so a wrong guess is
# visible immediately rather than silently plotting the wrong column.
DR_NAME_CANDIDATES = ["depletion_rank", "DR", "depletionRank", "dr"]
GC_NAME_CANDIDATES = ["GC_content", "GC_content_1000bp", "pct_gc", "gc_content", "GC"]


def resolve_column(df: pl.DataFrame, requested: str | None, candidates: list[str], role: str) -> str:
    """
    Pick the column to use for `role`. An explicit `requested` name wins and
    must exist; otherwise the first candidate present in the file wins. Raises
    with the full column list if nothing matches -- better than a KeyError deep
    in the binning code.
    """
    if requested is not None:
        if requested not in df.columns:
            raise KeyError(f"{role} column {requested!r} not in file; columns are {df.columns}")
        return requested
    for name in candidates:
        if name in df.columns:
            return name
    raise KeyError(
        f"could not find a {role} column automatically (tried {candidates}); "
        f"pass it explicitly. File columns are: {df.columns}"
    )


def to_gc_fraction(gc: pl.Series) -> tuple[pl.Series, str]:
    """
    Return GC content as a 0-1 fraction, plus a note on what was assumed.

    McHale et al. compute GC with `bedtools nuc`, whose pct_gc column is
    already a 0-1 fraction (see CLAUDE.md, "GC content units"), so a
    constraint-tools-derived file should need no conversion. This repo's own
    genomic_features13 columns are 0-100 percentages. Detect rather than
    assume, since this module may be pointed at either.
    """
    gc_max = gc.max()
    if gc_max is not None and gc_max > 1.5:
        return gc / 100.0, f"treated as 0-100 percent (max {gc_max:.3g}), divided by 100"
    return gc, f"treated as a 0-1 fraction (max {gc_max:.3g}), unchanged"


def load_depletion_rank_windows(bed_path: str, dr_col: str | None = None,
                                 gc_col: str | None = None, complement: bool = True,
                                 has_header: bool = True) -> pl.DataFrame:
    """
    Read the Halldorsson depletion-rank window file and return a table with:
        gc               GC content, 0-1 fraction
        constraint       the DR-derived constraint score, oriented so that
                         HIGHER == MORE CONSTRAINED (see module docstring)
        rank_dr          standardized rank of `constraint` in (0,1), computed
                         WITHIN this window set, exactly as for the Gnocchi
                         curves: (rank - 0.5) / n

    complement=True applies the 1 - depletion_rank flip from experiment.1.ipynb.
    Set it False only if you are handing in a score that is already oriented
    "higher == more constrained".
    """
    if not os.path.exists(bed_path):
        raise FileNotFoundError(
            f"depletion-rank BED not found: {bed_path}\n"
            "This file is not in the repo -- it comes from the constraint-tools "
            "CONSTRAINT_TOOLS_DATA path (see this module's docstring)."
        )

    df = pl.read_csv(bed_path, separator="\t", has_header=has_header,
                      infer_schema_length=10000, null_values=["", ".", "NA", "NaN"])
    dr_name = resolve_column(df, dr_col, DR_NAME_CANDIDATES, "depletion-rank")
    gc_name = resolve_column(df, gc_col, GC_NAME_CANDIDATES, "GC-content")

    df = df.select([
        pl.col(dr_name).cast(pl.Float64).alias("_dr"),
        pl.col(gc_name).cast(pl.Float64).alias("_gc"),
    ]).drop_nulls()

    gc_frac, gc_note = to_gc_fraction(df["_gc"])
    df = df.with_columns(gc_frac.alias("gc"))

    constraint = (1.0 - pl.col("_dr")) if complement else pl.col("_dr")
    df = df.with_columns(constraint.alias("constraint"))

    n = df.height
    df = df.with_columns(((pl.col("constraint").rank() - 0.5) / n).alias("rank_dr"))

    print(f"depletion rank: {n:,} windows from {os.path.basename(bed_path)}")
    print(f"  score column   = {dr_name!r}" + ("  (complemented: 1 - DR)" if complement else ""))
    print(f"  GC column      = {gc_name!r}, {gc_note}")
    print(f"  GC range       = {df['gc'].min():.3f} - {df['gc'].max():.3f}")

    return df.select(["gc", "constraint", "rank_dr"])


def bin_depletion_rank(df_dr: pl.DataFrame, n_bins: int = 20,
                        bin_method: str = "fixed") -> pl.DataFrame:
    """
    GC-bin the DR windows with the same binning code the Gnocchi curves use,
    so the two are directly comparable. Returns bin_by_gc()'s usual shape:
    n, gc_mid, mean_dr, se_dr.
    """
    from gnocchi_bias.windows import bin_by_gc
    return bin_by_gc(df_dr, "gc", n_bins, bin_method, value_cols={"dr": "rank_dr"})
