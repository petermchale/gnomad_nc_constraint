"""
Two claims about misc/generic.py, misc/constraint_basics.py and
misc/nc_constraint_utils.py, checked by downloading the real files and reading them:

1. They are NOT missing from the bucket (CLAUDE.md used to say they were) -- they
   are at misc/*.py and are what run_nc_constraint_gnomad_v31_main.py:23-25 imports.
2. None contains the multivariate PCA + regularized-logistic fit that computes
   r(w); only the apply side is published (that script's lines 231-249). The gap is
   real, not an artifact of an incomplete local checkout.

Claim 2 is the premise of validate.py -- the reason there is anything to reimplement.
See CLAUDE.md, "The paper's Methods do not match the code".

Outcome of the last run: preconditions/output/STATUS.md (transcript in the .log beside it).
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnocchi_bias.windows import download  # noqa: E402
from preconditions.report import Report  # noqa: E402

# Repo-root cache, shared with every other script here; resolved from __file__ so
# running from inside preconditions/ reuses it rather than refetching multi-GB files.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DEST_DIR = os.path.join(_REPO_ROOT, "published")

FILES = ["generic.py", "constraint_basics.py", "nc_constraint_utils.py"]
MAIN_SCRIPT = "run_nc_constraint_gnomad_v31_main.py"
GAP_PATTERNS = ["IncrementalPCA", r"\bPCA\b", "fit_regularized"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-dest_dir", default=DEFAULT_DEST_DIR,
        help=f"directory to download the three files into (default: {DEFAULT_DEST_DIR})")
    args = parser.parse_args()

    with Report("verify_missing_utils_files") as rep:
        # Via the shared downloader (curl -fL, check=True), NOT a bare curl: a silent
        # download failure here would leave empty files, find zero pattern matches, and
        # print "confirmed" -- a false confirmation of the very claim under test.
        paths = {fname: download(f"misc/{fname}", args.dest_dir) for fname in FILES}

        print("\n=== Claim 1: these modules exist and are what the main script imports ===")
        with open(MAIN_SCRIPT) as f:
            main_src = f.read()
        all_imported, total_lines = True, 0
        for fname, path in paths.items():
            with open(path) as f:
                n_lines = sum(1 for _ in f)
            total_lines += n_lines
            imported = f"from {fname[:-3]} import" in main_src
            all_imported = all_imported and imported and n_lines > 0
            print(f"  misc/{fname}: {os.path.getsize(path)} bytes, {n_lines} lines -- "
                  f"imported by {MAIN_SCRIPT}? {imported}")

        print("\n=== Claim 2: none of the three contains the multivariate-fit step ===")
        any_hit = False
        for fname, path in paths.items():
            with open(path) as f:
                hits = [p for p in GAP_PATTERNS if re.search(p, f.read())]
            any_hit = any_hit or bool(hits)
            print(f"  misc/{fname}: matches for {GAP_PATTERNS} -> {hits or 'none'}")

        print("\nConclusion: " + (
            "unexpected match found -- re-examine the gap claim in CLAUDE.md" if any_hit else
            "confirmed -- zero matches for PCA/IncrementalPCA/fit_regularized in any of the "
            "three. The multivariate fit behind r(w) has no published source in this bucket; "
            "only the apply side (that script's lines 231-249) is available."))

        # Claim 1 is what makes claim 2 mean anything: zero pattern matches in three
        # empty or wrong files would look identical to zero matches in the real ones.
        rep.claim(all_imported,
                  f"all {len(FILES)} modules downloaded non-empty ({total_lines} lines "
                  f"total) and each is imported by {MAIN_SCRIPT} -- they are NOT missing")
        rep.claim(not any_hit,
                  f"none of the three matches {GAP_PATTERNS} -- the multivariate PCA+logit "
                  "fit behind r(w) is unpublished, so reimplementing it was necessary")


if __name__ == "__main__":
    main()
