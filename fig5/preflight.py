"""
Check the two hand-supplied inputs BEFORE the expensive run.

    .venv/bin/python fig5/preflight.py

Neither `NEUTRAL_WINDOWS_BED` nor `DEPLETION_RANK_BED` is in this repo -- both live on
the constraint-tools HPC path -- so neither loader has ever run against its real file.
Everything downstream is expensive (two refits at ~6 min and ~4 GB each, then the
notebook), and the failure modes are quiet: a chromosome-naming mismatch reads as a very
strict filter, a depletion-rank column in the wrong orientation just flips a curve.
This script reads both files, asserts what the loaders assume, prints what it cannot
assert, and exits non-zero on anything that would produce a wrong figure rather than an
error.

It checks SCHEMA and CONVENTION only -- nothing here needs `published/`, so it runs in
seconds on a login node. The join against Chen et al.'s tables is checked separately, by
`restrict_to_mchale_neutral_windows`, which prints its own diagnostics at build time.
"""
import os
import sys

import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402
from gnocchi_bias import windows as W  # noqa: E402
import depletion_rank as DR  # noqa: E402

WINDOW_SIZE = 1000
problems: list[str] = []
notes: list[str] = []


def fail(msg: str) -> None:
    problems.append(msg)
    print(f"  FAIL  {msg}")


def ok(msg: str) -> None:
    print(f"  ok    {msg}")


def note(msg: str) -> None:
    notes.append(msg)
    print(f"  NOTE  {msg}")


def check_neutral(path: str) -> None:
    print(f"\nNEUTRAL_WINDOWS_BED = {path}")
    if not os.path.exists(path):
        return fail("file does not exist")

    df = (pl.read_csv(path, separator="\t", infer_schema_length=10_000,
                      schema_overrides={"chrom": pl.String})
            .rename(lambda c: c.strip().replace(" ", "_")))
    print(f"  {df.height:,} rows x {df.width} columns")

    # 1. the columns the loader indexes by name
    for col in ("chrom", "start", "end", W.MCHALE_ENHANCER_COLUMN):
        if col in df.columns:
            ok(f"column {col!r} present")
        else:
            fail(f"column {col!r} missing; first columns are {df.columns[:8]}")
    if problems:
        return

    # 2. the enhancer flag must be castable to Boolean, and both values must occur --
    #    an all-True or all-False column means the annotation did not land.
    try:
        flag = df[W.MCHALE_ENHANCER_COLUMN].cast(pl.Boolean)
    except Exception as exc:                                    # noqa: BLE001
        return fail(f"{W.MCHALE_ENHANCER_COLUMN!r} will not cast to Boolean: {exc}")
    n_true = int(flag.sum())
    if n_true in (0, df.height):
        fail(f"{W.MCHALE_ENHANCER_COLUMN!r} is constant ({n_true:,} True of {df.height:,})"
             " -- the enhancer annotation is missing, so the 'neutral' set is not one")
    else:
        ok(f"{W.MCHALE_ENHANCER_COLUMN}: {n_true:,} True / {df.height - n_true:,} False")

    # 3. THE window count. Not fatal -- a different vintage is a real possibility -- but
    #    it must be resolved before either count is quoted in the paper.
    n_neutral = df.height - n_true
    if n_neutral == W.MCHALE_NEUTRAL_WINDOW_COUNT:
        ok(f"neutral count is exactly {n_neutral:,}, their Fig. 1 window set")
    else:
        note(f"neutral count {n_neutral:,} != {W.MCHALE_NEUTRAL_WINDOW_COUNT:,} "
             "(their Fig. 1 set) -- a different vintage of the file")

    # 4. coordinate convention: 0-based half-open 1 kb tiles on a 1 kb grid, which is
    #    what makes `chrom-start-end` equal Chen et al.'s element_id. Off-by-one here
    #    produces an empty join, not an error.
    span = (df["end"].cast(pl.Int64) - df["start"].cast(pl.Int64))
    if span.min() == span.max() == WINDOW_SIZE:
        ok(f"every window spans exactly {WINDOW_SIZE} bp")
    else:
        fail(f"window spans run {span.min()}-{span.max()}, not a constant {WINDOW_SIZE} "
             "-- these are not Chen et al.'s 1 kb tiles")
    off_grid = int((df["start"].cast(pl.Int64) % WINDOW_SIZE != 0).sum())
    if off_grid:
        fail(f"{off_grid:,} rows have start % {WINDOW_SIZE} != 0 -- either 1-based "
             "coordinates or a different tiling; element_ids will not match")
    else:
        ok(f"every start is on the {WINDOW_SIZE} bp grid")

    # 5. chromosome naming. `1` vs `chr1` is THE silent failure: the join returns almost
    #    nothing and looks like a strict filter.
    chroms = df["chrom"].unique().to_list()
    if all(str(c).startswith("chr") for c in chroms):
        ok(f"chrom uses the 'chr' prefix ({len(chroms)} distinct)")
    else:
        fail(f"chrom values look like {chroms[:4]} -- Chen et al.'s element_ids are "
             "'chr1-10000-11000', so this join would return near-nothing")

    # 6. the depletion-rank column this file also carries. Not an error -- it is simply
    #    not the source panel A uses -- but worth naming, because wiring it into
    #    depletion_rank.py would double-complement it.
    dr_cols = [c for c in df.columns if "depletion_rank" in c.lower()]
    if dr_cols:
        note(f"this file also carries {dr_cols} -- depletion rank on Chen et al.'s "
             "windows, already complemented. Panel A does NOT use it: it reads "
             "DEPLETION_RANK_BED and ranks within Halldorsson's own windows. Do not "
             "point depletion_rank.py at this column (it would flip the curve; the "
             "loader raises if you try).")

    # 7. what the figure will actually key on
    ids = (df.filter(~flag).select(
        (pl.col("chrom").cast(pl.String) + "-"
         + pl.col("start").cast(pl.Int64).cast(pl.String) + "-"
         + pl.col("end").cast(pl.Int64).cast(pl.String)).alias("element_id"))["element_id"])
    dupes = ids.len() - ids.n_unique()
    if dupes:
        fail(f"{dupes:,} duplicate element_ids among the neutral windows")
    else:
        ok(f"{ids.len():,} unique element_ids, e.g. {ids[0]!r}")


def check_depletion_rank(path: str) -> None:
    print(f"\nDEPLETION_RANK_BED = {path}")
    if not os.path.exists(path):
        return fail("file does not exist")

    # This file needs the enhancer flag too: McHale et al. filter BOTH files to
    # `window overlaps enhancer == False`, and the loader now refuses to skip it.
    head = (pl.read_csv(path, separator="\t", n_rows=1)
              .rename(lambda c: c.strip().replace(" ", "_")))
    if W.MCHALE_ENHANCER_COLUMN in head.columns:
        ok(f"column {W.MCHALE_ENHANCER_COLUMN!r} present (both files get filtered on it)")
    else:
        fail(f"no {W.MCHALE_ENHANCER_COLUMN!r} column, so the enhancer-overlapping "
             "windows cannot be excluded here as they are from the Gnocchi window set; "
             f"columns are {head.columns[:8]}")

    # The loader resolves both columns by name, so run IT rather than a copy of it: what
    # this script must not do is pass a check the real loader would fail.
    try:
        df = DR.load_depletion_rank_windows(path)
    except Exception as exc:                                    # noqa: BLE001
        return fail(f"load_depletion_rank_windows raised: {exc}")

    ok(f"{df.height:,} windows loaded")
    gc_lo, gc_hi = df["gc"].min(), df["gc"].max()
    if 0.0 <= gc_lo and gc_hi <= 1.0:
        ok(f"GC is a 0-1 fraction after unit detection ({gc_lo:.3f}-{gc_hi:.3f})")
    else:
        fail(f"GC runs {gc_lo:.3g}-{gc_hi:.3g} after unit detection, not a 0-1 fraction")
    if df.height < 100_000:
        note(f"only {df.height:,} windows -- Halldorsson's set is genome-wide; a small "
             "count suggests a subset or a parse that dropped rows")

    # ORIENTATION is the one thing no schema check can settle, because the curve is a
    # RANK: any monotone transform of the score gives the identical curve, so units and
    # range are irrelevant and only the DIRECTION matters. Print what the loader did and
    # make the reader confirm it.
    note("orientation is not checkable from the data: the panel ranks within this set, "
         "so scale is irrelevant and only direction matters. depletion_rank.py takes "
         "1 - depletion_rank, which is McHale et al.'s own "
         "depletion_rank_constraint_score_complement -- so this matches their notebook "
         "as long as the resolved column above is the RAW rank. A column already named "
         "'...complement' is refused by the loader for exactly this reason.")
    q = df["constraint"].quantile
    ok(f"constraint (higher == more constrained after complement): "
       f"min {df['constraint'].min():.3g}, median {q(0.5):.3g}, max {df['constraint'].max():.3g}")
    r = df["rank_dr"]
    if abs(float(r.mean()) - 0.5) < 1e-6 and r.min() > 0 and r.max() < 1:
        ok(f"rank_dr is uniform on (0,1) by construction, mean {float(r.mean()):.6f}")
    else:
        fail(f"rank_dr mean {float(r.mean()):.6f}, range {r.min():.3g}-{r.max():.3g} "
             "-- expected mean 0.5 strictly inside (0,1)")


def main() -> int:
    print("Preflight for the two hand-supplied inputs (fig5/config.py)")
    for name, path, check in (
            ("NEUTRAL_WINDOWS_BED", config.NEUTRAL_WINDOWS_BED, check_neutral),
            ("DEPLETION_RANK_BED", config.DEPLETION_RANK_BED, check_depletion_rank)):
        if path is None:
            print(f"\n{name} = None -- not set, nothing to check "
                  "(that arm of the figure is simply not built)")
        else:
            check(path)

    print(f"\n{'-' * 70}")
    if problems:
        print(f"{len(problems)} problem(s) -- fix before running the refits:")
        for p in problems:
            print(f"  - {p}")
    else:
        print("no problems found")
    if notes:
        print(f"{len(notes)} note(s) needing your judgement:")
        for n in notes:
            print(f"  - {n}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
