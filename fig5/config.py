"""
The two inputs that are not in this repo, in ONE place -- plus the provenance stamp that
keeps the expensive refits honest about them.

WHY THIS FILE EXISTS RATHER THAN A CONSTANT IN THE NOTEBOOK. GENEHANCER_BED defines the
analyzed window set, and that set is used twice, in two separate processes:

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

# The enhancer-exclusion half of McHale et al.'s "neutral" window definition.
# GeneHancer is licensed and cannot be downloaded here. None -> "neutral" is
# noncoding + pass_qc + autosome/PAR, consistently everywhere.
GENEHANCER_BED = None

# Populations whose training set is defined BY the analyzed window set, and so by
# GENEHANCER_BED. "full" is not one of them -- it never builds the window table -- so a
# change to GENEHANCER_BED does not invalidate it.
WINDOW_DEPENDENT = ("scored", "sizematched")

_PROVENANCE = "provenance.json"


def _read(refits_dir: str) -> dict:
    path = os.path.join(refits_dir, _PROVENANCE)
    return json.load(open(path)) if os.path.exists(path) else {}


def record(refits_dir: str, population: str, genehancer_bed: str | None) -> None:
    """Stamp the setting a refit was built under. Called by refit.py after it writes."""
    prov = _read(refits_dir)
    prov[population] = {"genehancer_bed": genehancer_bed}
    with open(os.path.join(refits_dir, _PROVENANCE), "w") as fh:
        json.dump(prov, fh, indent=1, sort_keys=True)


def check(refits_dir: str, population: str) -> None:
    """
    Raise if `population`'s refit was built under a different GENEHANCER_BED than the
    one set above, or if it carries no stamp at all. Populations outside
    WINDOW_DEPENDENT are unaffected by the setting and are not checked.
    """
    if population not in WINDOW_DEPENDENT:
        return
    stamped = _read(refits_dir).get(population)
    if stamped is None:
        raise RuntimeError(
            f"{os.path.join(refits_dir, _PROVENANCE)} has no entry for {population!r}, "
            f"so the training population it was built on is unknown.\n"
            f"Rerun:  .venv/bin/python fig5/refit.py -population {population}")
    if stamped.get("genehancer_bed") != GENEHANCER_BED:
        raise RuntimeError(
            f"the {population!r} refit was built with GENEHANCER_BED="
            f"{stamped.get('genehancer_bed')!r}, but fig5/config.py now says "
            f"{GENEHANCER_BED!r}. It would be fit on one window population and scored "
            f"on another.\nRerun:  .venv/bin/python fig5/refit.py -population {population}")
