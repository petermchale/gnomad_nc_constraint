"""
The two inputs that are not in this repo, in ONE place -- plus the provenance stamp that
keeps the expensive refits honest about them.

WHY THIS FILE EXISTS RATHER THAN A CONSTANT IN THE NOTEBOOK. NEUTRAL_WINDOWS_BED defines
the analyzed window set, and that set is used twice, in two separate processes:

  * fig5/refit.py -population scored (and sizematched) -- it decides which training
    sites survive the restriction, i.e. what the model is FIT on;
  * fig5/fig5.ipynb -- it decides which windows every panel is EVALUATED on.

If those two disagree, the model is trained on one population and scored on another,
which is precisely the defect this figure is about. Reading both from here makes them
impossible to desynchronise within a single run.

That is not sufficient on its own, because the refits are expensive and live on disk
across edits: changing this file after a refit would leave a stale refit that was built
under the old setting. So refit.py stamps the value it used into refits/provenance.json
and data.refit_path calls check() before handing a path back. Editing this file then
produces a loud error naming the refit to rerun, instead of a silent mismatch.

DEPLETION_RANK_BED has no such hazard -- nothing is fit on it -- but it lives here too
so both hand-supplied inputs are in one place.
"""
import json
import os

# --- inputs that must be supplied by hand (neither is fetchable here) ---

# Panel A's third curve. From the constraint-tools CONSTRAINT_TOOLS_DATA path:
# {CONSTRAINT_TOOLS_DATA}/depletion_rank_scores/
#     41586_2022_4965_MOESM3_ESM.noncoding.enhancer.BGS.gBGC.GC_content.bed
# None -> panel A builds with two curves instead of three.
DEPLETION_RANK_BED = None

# McHale et al.'s putatively neutral window set -- the 693,270 windows behind their
# Fig. 1 -- as their own analysis reads it:
#   {CONSTRAINT_TOOLS_DATA}/chen-et-al-2023-published-version/
#       41586_2023_6045_MOESM4_ESM/Supplementary_Data_2.features.constraint_scores.bed
# (filtered to `window overlaps enhancer == False`; see
# windows.load_mchale_neutral_element_ids, and their
# papers/neutral_models_are_biased/9.regression/experiment.1.ipynb).
#
# SET IT AND THE WHOLE FIGURE RECOMPUTES ON THEIR WINDOWS. None -> the analyzed set is
# the 1,843,559 windows that are noncoding + pass_qc + autosome/PAR, consistently
# everywhere. Both are meant to be run: the second is this repo's reproduction of their
# definition from public data, the first is their definition itself, and a result that
# holds on both is a result that does not depend on which one is right.
NEUTRAL_WINDOWS_BED = None

# Populations whose training set is defined BY the analyzed window set, and so by
# NEUTRAL_WINDOWS_BED. "full" is not one of them -- it never builds the window table --
# so a change to NEUTRAL_WINDOWS_BED does not invalidate it.
WINDOW_DEPENDENT = ("scored", "sizematched")

# Every artefact whose CONTENT depends on the window set carries this, so the two sets'
# outputs land BESIDE each other instead of overwriting: the refit tables, their
# provenance entries, and the panel PDFs. Empty for the default set, so nothing that
# exists today is renamed and fig5.ai's links keep resolving; the narrowed set's files
# gain `.neutral`. The parquet caches in fig5/output/ do not need it -- they already
# carry a fingerprint of the GC edges and the window set itself.
WINDOW_SET_SUFFIX = "" if NEUTRAL_WINDOWS_BED is None else ".neutral"


def tagged(population: str) -> str:
    """
    `population` as it appears in refit filenames and provenance keys.

    ONLY WINDOW_DEPENDENT populations are tagged. `full` never builds the window table,
    so its tables are identical under either setting and one copy serves both -- tagging
    it would send every reader looking for a file no run ever writes, since `full` is
    also the one population a window-set switch does NOT require rerunning.
    """
    return f"{population}{WINDOW_SET_SUFFIX}" if population in WINDOW_DEPENDENT else population


_PROVENANCE = "provenance.json"


def _read(refits_dir: str) -> dict:
    path = os.path.join(refits_dir, _PROVENANCE)
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return json.load(fh)


def record(refits_dir: str, population: str, neutral_windows_bed: str | None) -> None:
    """Stamp the setting a refit was built under. Called by refit.py after it writes."""
    prov = _read(refits_dir)
    prov[tagged(population)] = {"neutral_windows_bed": neutral_windows_bed}
    with open(os.path.join(refits_dir, _PROVENANCE), "w") as fh:
        json.dump(prov, fh, indent=1, sort_keys=True)


def check(refits_dir: str, population: str) -> None:
    """
    Raise if `population`'s refit was built under a different NEUTRAL_WINDOWS_BED than
    the one set above, or if it carries no stamp at all. Populations outside
    WINDOW_DEPENDENT are unaffected by the setting and are not checked.

    A stamp written before this key existed (it was `genehancer_bed`, and always null)
    reads as None here, which is the correct answer for it: those refits were built with
    no neutral restriction. The next refit rewrites its own entry outright.
    """
    if population not in WINDOW_DEPENDENT:
        return
    stamped = _read(refits_dir).get(tagged(population))
    if stamped is None:
        raise RuntimeError(
            f"{os.path.join(refits_dir, _PROVENANCE)} has no entry for "
            f"{tagged(population)!r}, so the training population it was built on is "
            f"unknown.\nRerun:  .venv/bin/python fig5/refit.py -population {population}")
    if stamped.get("neutral_windows_bed") != NEUTRAL_WINDOWS_BED:
        raise RuntimeError(
            f"the {population!r} refit was built with NEUTRAL_WINDOWS_BED="
            f"{stamped.get('neutral_windows_bed')!r}, but fig5/config.py now says "
            f"{NEUTRAL_WINDOWS_BED!r}. It would be fit on one window population and "
            f"scored on another.\nRerun:  .venv/bin/python fig5/refit.py "
            f"-population {population}")
