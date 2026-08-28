# Runbook — rebuilding fig5 with `NEUTRAL_WINDOWS_BED` and `DEPLETION_RANK_BED` set

For the run this repo cannot do on a laptop: the two hand-supplied files live on the
constraint-tools `CONSTRAINT_TOOLS_DATA` path, so the figure has only ever been built on
the 1,843,559-window set with two curves in panel A. This is the ordered procedure for
building it on McHale et al.'s 693,270 putatively neutral windows with all three.

`README.md` explains what each file does and why; this one is just the order, the costs,
and what to check. Neither loader has ever run against its real file, so step 2 is not
optional.

## 1. Set the two paths in `fig5/config.py`

```python
DEPLETION_RANK_BED  = f"{CONSTRAINT_TOOLS_DATA}/depletion_rank_scores/41586_2022_4965_MOESM3_ESM.noncoding.enhancer.BGS.gBGC.GC_content.bed"
NEUTRAL_WINDOWS_BED = f"{CONSTRAINT_TOOLS_DATA}/chen-et-al-2023-published-version/41586_2023_6045_MOESM4_ESM/Supplementary_Data_2.features.constraint_scores.bed"
```

**There and nowhere else.** `refit.py` decides what the model is *fit* on from this module
and the notebook decides what the panels are *evaluated* on from the same module; setting
it in the notebook instead is how you end up training on one population and scoring on
another, which is the defect the figure is about.

**Nothing gets overwritten.** Everything whose content depends on the window set carries
`config.WINDOW_SET_SUFFIX`, so this run lands beside the existing results:

| | already there | this run writes |
|---|---|---|
| refits | `…scored.txt` | `…scored.neutral.txt` |
| panels | `fig5A.pdf` | `fig5A.neutral.pdf` |
| supporting figure | `supp_fig7.pdf` | `supp_fig7.neutral.pdf` |
| `full` refit | `…full.txt` | reused as-is, never retagged |

The parquet caches in `output/` need no suffix: they already carry a fingerprint of the GC
edges and the window set.

## 2. Preflight the two files

```
.venv/bin/python fig5/preflight.py
```

Seconds, needs no `published/`, exits non-zero on anything that would yield a wrong figure
rather than an error. Run it on a login node the moment the paths are set.

* **neutral file** — the four columns read by name; the enhancer flag casts to Boolean and
  is not constant; every window spans exactly 1000 bp and starts on the 1 kb grid (1-based
  coordinates produce an empty join, not an error); the `chr` prefix; element_id
  uniqueness; the 693,270 count, as a note rather than a failure since a different vintage
  is possible. It also reports `depletion_rank_constraint_score_complement` if present:
  that column is real, already oriented, and deliberately not what panel A uses.
* **depletion-rank file** — that it carries `window overlaps enhancer` too, then runs the
  real loader and checks GC resolved to a 0-1 fraction and `rank_dr` is uniform with mean
  0.5.

## 3. Environment and resources

`requirements.txt`: duckdb, polars, pandas, scikit-learn, statsmodels, matplotlib.
**Hail and pyspark are not needed** -- nothing in fig5 reads a `.ht`. Budget ~32 GB RAM
(duckdb is capped at 8-10 GB in `data.py`; the genome-wide apply is the peak), ~8 GB for
`published/`, ~8 GB for the two new refits. Set `MPLBACKEND=Agg`.

## 4. Put `published/` in shared space, and pre-stage it if compute nodes have no egress

`published/` defaults to the repo root, which is wrong on a cluster: it is ~8 GB of
someone else's public data, identical for every lab member, and the checkout may be on a
small filesystem. Point `GNOCCHI_PUBLISHED_DIR` at shared space instead — `gnocchi_bias.
windows.CACHE_DIR` reads it, and every `-cache_dir`/`-dest_dir` default, `fig5/data.py`
and the notebook all resolve through that one definition:

```bash
mkdir -p /scratch/ucgd/lustre-labs/quinlan/data-shared/gnomad_nc_constraint
chmod g+ws /scratch/ucgd/lustre-labs/quinlan/data-shared/gnomad_nc_constraint  # group-writable, setgid
export GNOCCHI_PUBLISHED_DIR=/scratch/ucgd/lustre-labs/quinlan/data-shared/gnomad_nc_constraint
```

Put the `export` in the job script and in `.bashrc`: it has to be set for the login-node
download AND for the batch job, and a job that silently loses it refetches 8 GB into the
checkout instead of failing.

Everything is fetched on demand from `storage.googleapis.com`: 1.44 GB features, 3.3 GB
per-context expected, 325 MB annot, 107 MB step-1 expected, plus the DNM training tables.
`rsync` a working `published/` across, or run one refit on a login node to pull it.

`refits/` (~12 GB, and written per run rather than shared) still lives in the checkout;
`fig5/refit.py -output_dir` moves it if the checkout's filesystem is too small.

## 5. Rerun the two window-dependent refits

```
python fig5/refit.py -population scored
python fig5/refit.py -population sizematched
```

~6 min and ~4 GB each. Each prints the tag it is writing and the `NEUTRAL_WINDOWS_BED` it
saw, then stamps `refits/provenance.json`.

**Do not rerun `full`.** It never builds the window table (`refit.py`'s
`if args.population != "full":` guard is the only reader of `NEUTRAL_WINDOWS_BED` there),
so its tables are identical under either setting and one copy serves both -- it is the
one population a window-set switch does not invalidate. It does have to be present:
`data.r_eff_by_gc(pop="full")` is the control that reproduces published Gnocchi through
this repo's own fitting code, and the notebook will not build without it.

So copy the five `full` tables across rather than spending 6 min regenerating identical
bytes. **From the Mac**, with the HPC checkout already created:

```bash
rsync -avP \
  ~/gnomad_nc_constraint/refits/*.full.txt \
  u6018199@father:/scratch/ucgd/lustre-labs/quinlan/u6018199/gnomad_nc_constraint/refits/
```

`-P` (`--partial --progress`), not `--info=progress2`: macOS ships rsync 2.6.9, which
predates that flag. ~4.0 GB, dominated by `rr_by_context.full.txt` (3.97 GB); the other four are
`expected_counts_by_context_methyl_genome_1kb` (127 MB),
`training_reliability_predictions` (144 MB), `coef_univariate` and `selected` (KBs).
If outbound ssh from the laptop is blocked, run the same command in the pull direction
from `father` with the Mac as the source.

**Do not copy `refits/provenance.json`.** The `*.full.txt` glob excludes it, which is the
point: `full` is outside `config.WINDOW_DEPENDENT` so it is never stamped or checked, and
copying the Mac's file would overwrite the `scored.neutral` / `sizematched.neutral` stamps
that this run's own refits wrote.

Verify before trusting it -- `rsync` will resume a partial transfer but nothing downstream
re-checks these sizes:

```bash
ls -l /scratch/ucgd/lustre-labs/quinlan/u6018199/gnomad_nc_constraint/refits/*.full.txt
```

## 6. Build the figure

```
jupyter nbconvert --to notebook --execute --inplace fig5/fig5.ipynb
```

## 7. Read these four before believing any panel

* **The neutral join.** `N -> M` with M near 693,270; the join *raises* below half the
  file's windows, which is the chromosome-naming signature rather than a strict filter.
  Then `X of the M kept have coding_prop > 0`: **0 means their set nests inside QC-pass
  noncoding**, so panel C's `coding` band is exactly QC-pass coding; anything else and
  that band is "QC-pass coding *outside their set*", which changes the caption. Finally
  the unmatched count -- their windows with no row in Chen et al.'s tables, expected to be
  almost all QC failures.
* **Depletion rank.** The enhancer filter line (`… -> … with window_overlaps_enhancer ==
  False`), which column resolved as the score, whether GC was divided by 100, and that the
  complement was applied. McHale et al. filter *both* files on enhancers; the loader
  raises if that column is missing rather than ranking a different population.
* **Panel A/E `mean |rank - 0.5|`.** Expect movement, and expect most of it to be bins
  dropping under the 100-window floor rather than the bias changing:
  `window_set_sensitivity.py` saw 0.212 -> 0.182 that way while step 2's per-bin curve
  stayed put. Quote the per-bin curve, or say which bins the average covers. The real
  narrowed run came in at 0.168 for published Gnocchi, 0.046 for the context-only model
  and 0.026 for the retrained one, against 0.212 / 0.093 / 0.046 on the wider set -- the
  whole triple shifts down together, and the *contrast* is what carries over.
* **Panel C's new middle band** (`other_noncoding`), the territory given up in the
  narrowing. Flat at 0 in the lower row means the narrowing costs sample size and nothing
  else, and the rest of the figure carries over; climbing with GC makes it a third
  population change that belongs in the caption beside the other two. **Measured: flat**,
  0.94-1.03x across the plotted range.

## 8. Assembly comes back to the Mac

`resave_ai.py` drives Illustrator through `osascript` and has no HPC equivalent (on Linux
it prints a notice and returns). Copy `fig5/output/*.neutral.{pdf,png}` back.

`fig5.ai` links the **unsuffixed** panels, so the neutral set is a second figure to
assemble, not a relink of the first. `resave_ai` is parameterized by assembly:

```bash
.venv/bin/python fig5/resave_ai.py                    # fig5.ai <- fig5A.pdf ...
.venv/bin/python fig5/resave_ai.py -suffix .neutral   # fig5.neutral.ai <- fig5A.neutral.pdf ...
```

Each assembly has its own PNG and its own links manifest (`fig5.neutral.ai.links.json`),
and each sees only its own panels, so neither reports the other's as stale. Until you
build `fig5.neutral.ai` in Illustrator, the neutral run's last notebook cell prints
`no such file ... -- nothing to refresh` and moves on.

Note the consequence of `NEUTRAL_WINDOWS_BED` being set: the notebook passes
`config.WINDOW_SET_SUFFIX`, so a neutral run refreshes the *neutral* assembly and leaves
`fig5.ai` alone. Refreshing `fig5.ai` is then a separate, explicit command.

## 9. Rebuild the supporting text

The submittable PDF of the notebook is built from the executed notebook AND the panel
PDFs, so it is stale after any rebuild of either:

```
.venv/bin/python fig5/make_supporting_pdf.py
```

Mac-side, like the assembly -- it wants xelatex and pandoc, not `published/`. See
`README.md` for what it filters out and why.

## Two caveats for the caption that no check can enforce

1. **The DR curve comes from a different window set** than the two Gnocchi curves. It is
   ranked within Halldorsson's own windows and overlaid, which is legitimate only because
   every rank is uniform on (0,1) by construction -- the comparison is about how each
   metric's uniform mass redistributes across GC, not about score values.
2. **Its orientation** rests on low depletion rank meaning more constrained, via the
   `1 - depletion_rank` complement that matches McHale et al.'s notebook. A DR curve
   mirrored about y = 0.5 is that assumption failing, and y = 0.5 is exactly where an
   unbiased metric sits -- so the wrong version looks entirely plausible.
