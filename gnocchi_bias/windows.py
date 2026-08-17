"""
Genome-wide 1kb window table for the GC-bias analyses, plus the Figure-2A-style
rank statistic computed on it.

Extracted verbatim (2026-08-04) from the repo-root
compute_gc_bias_step1_vs_step2.py, which was the command-line entry point until
it was deleted (2026-08-07, superseded by fig5 panel A; recoverable from git
history). Every docstring below keeps its original citation trail into
CLAUDE.md / McHale et al.'s Methods -- those citations are the point of this
code, so do not trim them.

The only NEW code here is the N-curve generalization of the z/rank computation
(add_z_column, add_rank_columns, binned_rank_curves): the statistic
parameterized over an arbitrary list of (label, expected_col) curves instead of
the hardcoded step1/step2 pair, with every curve z-filtered jointly and ranked
only after that filter, so all curves describe one identical window population.
Lifted from the training-set-size experiment, which had already worked this out
because it needs one curve per DNM training fraction. add_z_columns() is kept
as a two-curve (step1/step2) wrapper so the original CLI path was unchanged by
the extraction.

This module deliberately does NOT call matplotlib.use(): it must import cleanly
inside a notebook running an interactive backend. Callers that need "Agg" set
it themselves.
"""
import os
import shutil
import subprocess
import tempfile
import time

import duckdb
import numpy as np
import polars as pl

BUCKET_URL = "https://storage.googleapis.com/gnomad-nc-constraint-v31-paper"

REMOTE_FILES = {
    "step1_expected": "expected_counts_by_context_methyl_genome_1kb.txt",  # element_id, possible, expected  (step-1, r==1)
    "features": "misc/genomic_features13_genome_1kb.txt",                 # element_id, GC_content_1k, + 51 other cols
    "annot": "fig_tables/constraint_z_genome_1kb.annot.txt",              # element_id, possible, expected (step-2, r-adjusted), observed, oe, z, pass_qc, coding_prop, ...
}

# ---------------------------------------------------------------- plot style
# Shared by every figure in this repo so the panels read as one system.
HEATMAP_LINE_COLOR = "0.9"  # light grey/near-white -- closer than plain dark grey to how
                             # the paper's line actually reads over its mostly dark
                             # hexbin cells; see CLAUDE.md, "Heat map".

RANK_YLABEL = "Standardized rank of constraint metric"
AXIS_LABEL_FONTSIZE = 13
TICK_LABEL_FONTSIZE = 14
TITLE_FONTSIZE = 16
LEGEND_FONTSIZE = 13

# Figure 2A's x-axis range, READ VISUALLY from the published figure -- not a
# value stated anywhere in the paper's text. See CLAUDE.md, "Axis ranges", for
# the method and the caveat.
PAPER_XRANGE = (0.2, 0.73)


def download(relpath: str, dest_dir: str) -> str:
    """
    curl `relpath` from BUCKET_URL into dest_dir if not already present locally.
    Return the local path. Streams straight to disk (curl, not a buffered
    Python download) given the file sizes involved (up to 1.44 GB), downloads
    to a .part sidecar first and renames on success so a half-finished
    download is never mistaken for a complete one on a later run.
    """
    local_path = os.path.join(dest_dir, os.path.basename(relpath))
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return local_path

    os.makedirs(dest_dir, exist_ok=True)
    url = f"{BUCKET_URL}/{relpath}"
    tmp_path = local_path + ".part"
    print(f"downloading {url} -> {local_path}")
    t0 = time.time()
    subprocess.run(["curl", "-fL", "-s", "-o", tmp_path, url], check=True)
    os.rename(tmp_path, local_path)
    print(f"  done in {time.time() - t0:.1f}s ({os.path.getsize(local_path) / 1e6:.1f} MB)")
    return local_path


def load_joined_table(local_paths: dict) -> pl.DataFrame:
    """
    Use duckdb to build the analysis table without loading full files into
    memory at once: column-pruned scans of the 1.44 GB features file and the
    325 MB annot file, inner-joined with the (already small) step-1 expected
    file on element_id, pulled out as polars via `.pl()` (not `.df()`).

    Return a polars DataFrame with columns:
      element_id, possible_step1, expected_step1, possible_step2,
      expected_step2, observed, pass_qc, coding_prop, GC_content_1k,
      z_published (the official, already-computed Gnocchi z, used only as a
      sanity check against this module's own from-scratch z for the published
      curve -- see check_z_against_published(), which binned_rank_curves()
      runs automatically whenever this column is present)
    """
    query = f"""
        SELECT
            s1.element_id      AS element_id,
            s1.possible         AS possible_step1,
            s1.expected         AS expected_step1,
            an.possible         AS possible_step2,
            an.expected         AS expected_step2,
            an.observed         AS observed,
            an.pass_qc          AS pass_qc,
            an.coding_prop      AS coding_prop,
            an.z                AS z_published,
            ft.GC_content_1k    AS GC_content_1k
        FROM read_csv_auto('{local_paths["step1_expected"]}', header=True) s1
        INNER JOIN (
            SELECT element_id, possible, expected, observed, pass_qc, coding_prop, z
            FROM read_csv_auto('{local_paths["annot"]}', header=True)
        ) an USING (element_id)
        INNER JOIN (
            SELECT element_id, GC_content_1k
            FROM read_csv_auto('{local_paths["features"]}', header=True)
        ) ft USING (element_id)
    """
    con = duckdb.connect()
    return con.execute(query).pl()


def exclude_sex_chromosomes(df: pl.DataFrame) -> pl.DataFrame:
    """
    Drop chrX/chrY windows (in practice, 2,497 chrX pseudoautosomal-region
    rows; chrY is already absent). See CLAUDE.md, "Chromosome filtering", for
    the Methods citation and why PAR-on-chrX is the only remnant possible.
    """
    chrom = df["element_id"].str.extract(r"^(chr[^-]+)-")
    return df.filter(~chrom.is_in(["chrX", "chrY"]))


def add_gc_content_fraction(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add GC_content = GC_content_1k / 100 (this repo's 0-100 percentage ->
    McHale et al.'s 0-1 fraction). See CLAUDE.md, "GC content units", for the
    citation trail (bedtools nuc's pct_gc column).
    """
    return df.with_columns((pl.col("GC_content_1k") / 100.0).alias("GC_content"))


# The "noncoding" half of McHale et al.'s neutral definition, as ONE number: a window is
# noncoding if at most this fraction of it is coding exon. Their Methods say "don't
# significantly overlap merged exons" without giving a value (CLAUDE.md, "Noncoding
# restriction"), so 0.0 -- no coding overlap at all -- is the conservative reading.
# It lives here rather than as a bare literal at each use because fig5's panel C has to
# label training sites by the same criterion this filter selects windows by; two spellings
# of "noncoding" that can drift apart is exactly the bug that panel is about. Change it
# here and both move together.
NONCODING_MAX_CODING_PROP = 0.0


def restrict_to_noncoding(df: pl.DataFrame,
                          coding_prop_threshold: float = NONCODING_MAX_CODING_PROP
                          ) -> pl.DataFrame:
    """
    Filter to noncoding 1kb windows (coding_prop <= threshold) -- half of
    McHale et al.'s "neutral" window definition. See
    restrict_to_neutral_genehancer() for the other half, and CLAUDE.md,
    "Noncoding restriction", for the Methods citation and the still-unconfirmed
    exact threshold.
    """
    return df.filter(pl.col("coding_prop") <= coding_prop_threshold)


def restrict_to_neutral_genehancer(
    df: pl.DataFrame,
    genehancer_bed_path: str | None,
    min_frac_covered: float | None = None,
) -> pl.DataFrame:
    """
    Exclude windows overlapping a GeneHancer enhancer -- the other half of
    McHale et al.'s "neutral" definition. No-op (with a printed warning)
    unless genehancer_bed_path is given: GeneHancer isn't freely
    downloadable, so this can't run end-to-end automatically. See CLAUDE.md,
    "GeneHancer enhancer exclusion", for the full citation trail and why.

    genehancer_bed_path: a standard BED file (tab-separated, chrom/start/end in
    the first three columns, 0-based half-open like the element_ids parsed here;
    extra columns and `#`/`track`/`browser` lines are bedtools' problem, not
    ours).

    min_frac_covered: None (default) excludes a window on ANY overlap.
    Otherwise a window is excluded when GeneHancer covers at least this fraction
    of it -- CUMULATIVE coverage, which is what `bedtools coverage` reports:
    bases of the window covered by at least one element, so the heavy mutual
    overlap between GeneHancer elements is counted once rather than summed.

    That is deliberately NOT `bedtools intersect -f` semantics, which ask
    whether ONE element covers that fraction of the window: three elements
    covering 35% each pass `-f 0.5` while covering the window entirely.
    Cumulative is the reading that answers "how much of this window is
    enhancer", which is what "significantly overlap" has to mean about a window.
    The distinction is moot at the default, where any overlap excludes and the
    two agree; it matters only if a fraction is ever supplied. CLAUDE.md notes
    McHale et al. use `-f 0.5` in a DIFFERENT intersect step -- if that turns
    out to be this step's rule too, this becomes `bedtools intersect -v -f`.

    Needs bedtools on PATH (`brew install bedtools`, or the module on an HPC).
    An earlier version did the interval arithmetic in duckdb to avoid the
    dependency; the machine that has the licensed BED will have bedtools, and
    `bedtools coverage` states the intent in one line where the SQL needed a
    gaps-and-islands merge to say the same thing.

    UNTESTED against real GeneHancer data -- no such file is available in this
    environment; verify directly before relying on it for the rebuttal/paper.
    The chromosome-naming check and the before/after counts below exist because
    the characteristic failure here is silent: a `chr1` vs `1` mismatch makes
    bedtools report zero coverage everywhere, which looks like a clean run.
    """
    if genehancer_bed_path is None:
        print(
            "WARNING: genehancer_bed_path is None -- 'neutral' here is only "
            "noncoding + pass_qc (+ non-sex-chromosome), NOT excluding "
            "GeneHancer-enhancer-overlapping windows. See CLAUDE.md, "
            "'GeneHancer enhancer exclusion'."
        )
        return df
    if shutil.which("bedtools") is None:
        raise RuntimeError(
            "bedtools is not on PATH, and the GeneHancer exclusion needs it "
            "(`brew install bedtools`, or load the module on an HPC).")

    # element_id is 0-based half-open (chr-start-end), the same convention BED
    # uses -- which is the whole reason these coordinates can go straight to
    # bedtools. The row number rides along as the BED name field so the result
    # can be joined back to the input.
    windows = df.select([
        pl.col("element_id").str.extract(r"^(chr[^-]+)-").alias("chrom"),
        pl.col("element_id").str.extract(r"^chr[^-]+-(\d+)-").cast(pl.Int64).alias("start"),
        pl.col("element_id").str.extract(r"^chr[^-]+-\d+-(\d+)$").cast(pl.Int64).alias("end"),
        pl.int_range(pl.len()).alias("rid"),
    ])

    with tempfile.TemporaryDirectory() as tmp:
        windows_bed = os.path.join(tmp, "windows.bed")
        covered_bed = os.path.join(tmp, "covered.bed")
        windows.write_csv(windows_bed, separator="\t", include_header=False)

        # coverage, not intersect: it reports bases of A covered by B and the
        # fraction they make up, counting a base once however many B features
        # cover it. Not -sorted -- that needs both inputs in the same chromosome
        # order and a -g file to check it, and B here is small enough to hold.
        with open(covered_bed, "w") as out_fh:
            subprocess.run(["bedtools", "coverage",
                            "-a", windows_bed, "-b", genehancer_bed_path],
                           stdout=out_fh, check=True)

        cov = pl.read_csv(
            covered_bed, separator="\t", has_header=False,
            new_columns=["chrom", "start", "end", "rid",
                         "n_elements", "bases_covered", "window_length", "frac_covered"],
            schema_overrides={"rid": pl.Int64, "n_elements": pl.Int64,
                              "bases_covered": pl.Int64, "frac_covered": pl.Float64})

    if cov.height != df.height:
        raise RuntimeError(
            f"bedtools coverage returned {cov.height:,} rows for "
            f"{df.height:,} windows -- expected one row per window.")

    # Compare chromosome NAMES, not whether anything overlapped: a BED that
    # legitimately misses every window is a real (if odd) answer, while
    # `chr1` against `1` is a bug that produces the same zero. bedtools warns
    # about it on stderr as well, but only warns.
    bed_chroms = set(pl.read_csv(
        genehancer_bed_path, separator="\t", has_header=False, columns=[0],
        new_columns=["chrom"], comment_prefix="#", truncate_ragged_lines=True,
        schema_overrides={"chrom": pl.String})["chrom"].unique())
    win_chroms = set(windows["chrom"].unique())
    if not (bed_chroms & win_chroms):
        raise ValueError(
            "GeneHancer BED and the window table share no chromosome name "
            f"(windows: {sorted(win_chroms)[:3]}, BED: {sorted(bed_chroms)[:3]}). "
            "A chr-prefix mismatch would silently filter nothing.")

    keep = (pl.col("n_elements") == 0 if min_frac_covered is None
            else pl.col("frac_covered") < float(min_frac_covered))
    out = df.filter(cov.sort("rid").select(keep).to_series())

    rule = ("any overlap" if min_frac_covered is None
            else f"cumulative coverage >= {min_frac_covered}")
    print(f"GeneHancer exclusion ({rule}): {df.height:,} windows -> "
          f"{out.height:,} ({df.height - out.height:,} dropped)")
    if out.height in (0, df.height):
        print("  WARNING: that filter removed all windows or none -- the "
              "signature of a coordinate or chromosome-naming mismatch.")
    return out


def maybe_downsample(df: pl.DataFrame, frac: float | None, n: int | None, seed: int) -> pl.DataFrame:
    """
    Escape hatch for compute: if `frac` or `n` is given, randomly (uniformly)
    subsample `df` before binning; otherwise return df unchanged. At most one
    of frac/n may be set.
    """
    if frac is not None and n is not None:
        raise ValueError(
            "specify at most one of frac / n "
            "(build_window_table: downsample_frac / downsample_n)")
    if frac is not None:
        return df.sample(fraction=frac, seed=seed)
    if n is not None:
        return df.sample(n=n, seed=seed)
    return df


def add_bias_columns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add per-window bias_step1 = expected_step1 - observed and
    bias_step2 = expected_step2 - observed, matching McHale et al.'s residual
    sign convention (predicted - y, not y - predicted). Fed to bin_by_gc() as
    value_cols for the residual metric -- the statistic Supp. Fig. 1 is defined
    on, which the deleted CLI exposed as -bias_metric residual (recoverable
    from git; nothing in the repo computes it today).
    """
    return df.with_columns([
        (pl.col("expected_step1") - pl.col("observed")).alias("bias_step1"),
        (pl.col("expected_step2") - pl.col("observed")).alias("bias_step2"),
    ])


# -------------------------------------------------------- z-scores and ranks

# The r-adjusted expected count Chen et al. actually published. A curve built on
# this column is the published Gnocchi, whatever the caller labels it (fig5 says
# "step2", dnm_training_size says "step2_published"), so the sanity check below
# keys on the column, not the label.
PUBLISHED_EXPECTED_COL = "expected_step2"


def z_expr(expected_col: str, observed_col: str = "observed") -> pl.Expr:
    """
    The Gnocchi z-score, exactly as run_nc_constraint_gnomad_v31_main.py lines
    278-280 computes it:
        oe = observed / expected
        chisq = (observed - expected)**2 / expected
        z = -sqrt(chisq) if oe >= 1 else sqrt(chisq)
    Note the sign: MORE observed variation than expected (oe >= 1, i.e.
    unconstrained) gives a NEGATIVE z, so high z == constrained.
    """
    oe = pl.col(observed_col) / pl.col(expected_col)
    chisq = (pl.col(observed_col) - pl.col(expected_col)) ** 2 / pl.col(expected_col)
    return pl.when(oe >= 1).then(-chisq.sqrt()).otherwise(chisq.sqrt())


def add_z_column(df: pl.DataFrame, label: str, expected_col: str,
                  observed_col: str = "observed") -> pl.DataFrame:
    """
    Add z_{label} from (expected_col, observed_col) using z_expr(). The
    N-curve generalization of add_z_columns()'s hardcoded step1/step2.
    """
    return df.with_columns(z_expr(expected_col, observed_col).alias(f"z_{label}"))


def filter_z_in_range(df: pl.DataFrame, labels: list[str],
                       lo: float = -10.0, hi: float = 10.0) -> pl.DataFrame:
    """
    Replicate run_nc_constraint_gnomad_v31_main.py line 281's filtering
    (`df_z[df_z['z'].between(-10,10)].drop(columns=['chisq']).dropna()` -- the
    between() and dropna() are the parts reproduced here), applied jointly across
    EVERY named curve: a window is kept only if all of its z-scores are
    finite and in [lo, hi]. Filtering jointly (rather than per curve) is what
    makes the curves an apples-to-apples comparison on one identical window
    population, rather than several differently-filtered sets.
    """
    for label in labels:
        df = df.filter(
            pl.col(f"z_{label}").is_between(lo, hi) & pl.col(f"z_{label}").is_finite()
        )
    return df


def add_rank_columns(df: pl.DataFrame, labels: list[str]) -> pl.DataFrame:
    """
    Standardized rank of each z_{label}, in (0, 1) with mean exactly 0.5:
    rank = (rank(z) - 0.5) / n -- matches Figure 2's y-axis definition (see
    CLAUDE.md for the exact quoted caption text). Each curve is ranked WITHIN
    ITSELF, so every curve is uniform on (0,1) by construction and the
    comparison between them is purely about how that uniform mass is
    distributed across GC bins.
    """
    n = df.height
    return df.with_columns([
        ((pl.col(f"z_{label}").rank() - 0.5) / n).alias(f"rank_{label}")
        for label in labels
    ])


def check_z_against_published(df: pl.DataFrame, label: str, tol: float = 1e-6) -> None:
    """
    Check that this module's from-scratch z for the PUBLISHED curve reproduces
    Chen et al.'s own z column (z_published, joined in by load_joined_table).

    Both sides start from the same (expected_step2, observed) pair, so this
    tests the join and z_expr()'s transcription of lines 278-280 -- NOT the
    model, and not anything about r. Measured directly over all 1,984,900 rows
    of fig_tables/constraint_z_genome_1kb.annot.txt the two agree exactly
    (max |diff| = 0.0), which is why a violation raises rather than warns: at
    this tolerance the only way to fail is a mis-joined or mislabelled column.

    No-op when z_published is absent, so callers scoring only refits (or a
    window table built without the annot file) are unaffected.
    """
    if "z_published" not in df.columns or f"z_{label}" not in df.columns:
        return
    max_diff = (df[f"z_{label}"] - df["z_published"]).abs().max()
    print(f"  sanity check: z_{label} vs published z, max |diff| = {max_diff} "
          f"over {df.height:,} windows")
    if max_diff is not None and max_diff > tol: # type: ignore
        raise ValueError(
            f"self-computed z_{label} disagrees with Chen et al.'s published z by "
            f"{max_diff} (> {tol}); both derive from the same "
            f"({PUBLISHED_EXPECTED_COL}, observed) pair, so this means the join "
            f"or the z formula is wrong, not the model.")


def add_z_columns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Two-curve (step1/step2) wrapper over add_z_column/filter_z_in_range, kept
    so the original CLI path was unchanged by the extraction. Adds
    z_step1 (from expected_step1, i.e. r==1) and z_step2 (from expected_step2,
    the real r-adjusted Gnocchi expected count -- the official pipeline never
    computes a step-1-only z, so z_step1 is entirely self-computed here).

    The z_step2-vs-z_published sanity check now lives in
    check_z_against_published() and is shared with binned_rank_curves(), so it
    runs on the live fig5 path too rather than only here. Measured agreement is
    exact, not merely close -- see that function.
    """
    df = add_z_column(df, "step1", "expected_step1")
    df = add_z_column(df, "step2", PUBLISHED_EXPECTED_COL)
    check_z_against_published(df, "step2")
    return filter_z_in_range(df, ["step1", "step2"])


def bin_by_gc(df: pl.DataFrame, gc_col: str, n_bins: int, bin_method: str,
               value_cols: dict[str, str]) -> pl.DataFrame:
    """
    Assign each window to a GC-content bin (fixed-width edges spanning the
    observed range of `gc_col`, or equal-count quantile edges), via
    numpy.digitize -- avoids depending on a specific polars cut()/qcut() API
    version.

    gc_col is the column to bin on: "GC_content" (0-1 fraction, paper units)
    in rank mode, or "GC_content_1k" (0-100 percentage, this repo's native
    units) in residual mode -- see add_gc_content_fraction().

    value_cols maps an output suffix to the per-window column to average within
    each bin. The two need NOT be the same string, and for the rank statistic
    they deliberately are not: binned_rank_curves() passes
    {label: f"rank_{label}"} -- e.g. {"step1": "rank_step1"} -- so the outputs
    are mean_step1/se_step1, indexable by curve label alone. fig5/data.py and
    fig5/depletion_rank.py both read them back that way, so changing this
    convention breaks them silently. The residual metric is passed the same
    way, from add_bias_columns()'s output. One binning implementation
    therefore serves both metrics without duplicating the digitize/group_by
    logic.

    Per bin, computes:
      - n      = window count
      - gc_mid = mean gc_col value in the bin (x-axis value for plotting)
      - for each (suffix, col) in value_cols:
          mean_{suffix} = mean(col) in the bin
          se_{suffix}   = std(col) / sqrt(n) in the bin
    (per-window value averaged, not a difference of per-bin averages, so the
    residual metric matches the "local bias" definition used in Supp Fig 1 /
    the simulation script; the rank metric matches Figure 2A's conditional-
    mean-rank line)

    Return the binned summary DataFrame (polars), one row per GC bin, sorted
    by gc_mid.
    """
    gc = df[gc_col].to_numpy()
    if bin_method == "quantile":
        edges = np.quantile(gc, np.linspace(0, 1, n_bins + 1))
    else:
        edges = np.linspace(gc.min(), gc.max(), n_bins + 1)
    edges = np.unique(edges)
    # Inert, and measured to be so on both sides of the fence: digitize() below
    # only ever sees the INTERIOR edges (edges[1:-1]), and while fig5/data.py's
    # sql_bin_expr() does read edges[-1] to derive its bin width, its LEAST(...,
    # n-1) clip already puts the max value in the last bin. Removing the bump
    # changed no bin assignment in either implementation over 500k GC values.
    # Kept only so gc_edges() there stays a literal mirror of this branch --
    # what matters is that the two build the SAME edges, not this epsilon.
    edges[-1] += 1e-9

    bin_idx = np.digitize(gc, edges[1:-1], right=False)
    df = df.with_columns(pl.Series("gc_bin", bin_idx))

    aggs = [pl.len().alias("n"), pl.col(gc_col).mean().alias("gc_mid")]
    for suffix, col in value_cols.items():
        aggs.append(pl.col(col).mean().alias(f"mean_{suffix}"))
        aggs.append((pl.col(col).std() / pl.len().sqrt()).alias(f"se_{suffix}"))

    binned = df.group_by("gc_bin").agg(aggs).sort("gc_mid")
    return binned


# ------------------------------------------------------ end-to-end pipelines

def build_window_table(cache_dir: str, exclude_sex: bool = True,
                        noncoding: bool = True, apply_qc: bool = True,
                        genehancer_bed: str | None = None,
                        genehancer_min_frac_covered: float | None = None,
                        downsample_frac: float | None = None,
                        downsample_n: int | None = None,
                        random_seed: int = 0) -> pl.DataFrame:
    """
    Download (if needed), join, and filter the genome-wide 1kb window table,
    returning it with a GC_content (0-1 fraction) column ready for binning.

    This is the download -> join -> filter -> GC-units chain that used to live
    inline in compute_gc_bias_step1_vs_step2.py's main() (since deleted); extracting it is
    what makes the analysis usable from a notebook. Defaults match that
    script's own defaults, i.e. McHale et al.'s window definition as far as it
    is reproducible here (see restrict_to_neutral_genehancer for the part that
    is not).
    """
    os.makedirs(cache_dir, exist_ok=True)
    local_paths = {k: download(v, cache_dir) for k, v in REMOTE_FILES.items()}

    df = load_joined_table(local_paths)
    if exclude_sex:
        df = exclude_sex_chromosomes(df)
    if noncoding:
        df = restrict_to_noncoding(df)
    if apply_qc:
        df = df.filter(pl.col("pass_qc"))
    df = restrict_to_neutral_genehancer(df, genehancer_bed, genehancer_min_frac_covered)
    df = maybe_downsample(df, downsample_frac, downsample_n, random_seed)
    df = add_gc_content_fraction(df)
    return df


def binned_rank_curves(df: pl.DataFrame, curves: list[tuple[str, str]],
                        n_bins: int = 20, bin_method: str = "fixed",
                        gc_col: str = "GC_content") -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Compute the Figure-2A rank statistic for an arbitrary set of curves on one
    shared window population, and bin it by GC content.

    curves: list of (label, expected_col) pairs, e.g.
        [("step1", "expected_step1"), ("step2", "expected_step2")]
    All curves are z-filtered JOINTLY and ranked AFTER that filter, so they
    describe the same windows (see filter_z_in_range).

    Return (df, binned): the per-window table (with z_/rank_ columns added,
    needed for heat maps) and the per-GC-bin summary (with n, gc_mid, and
    mean_/se_ per curve, needed for the line plots).
    """
    labels = [label for label, _ in curves]
    for label, expected_col in curves:
        df = add_z_column(df, label, expected_col)
        if expected_col == PUBLISHED_EXPECTED_COL:
            check_z_against_published(df, label)
    df = filter_z_in_range(df, labels)
    df = add_rank_columns(df, labels)

    binned = bin_by_gc(df, gc_col, n_bins, bin_method,
                        value_cols={label: f"rank_{label}" for label in labels})
    return df, binned
