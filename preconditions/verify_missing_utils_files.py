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
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnocchi_bias.windows import download  # noqa: E402

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

    # Via the shared downloader (curl -fL, check=True), NOT a bare curl: a silent
    # download failure here would leave empty files, find zero pattern matches, and
    # print "confirmed" -- a false confirmation of the very claim under test.
    paths = {fname: download(f"misc/{fname}", args.dest_dir) for fname in FILES}

    print("\n=== Claim 1: these modules exist and are what the main script imports ===")
    with open(MAIN_SCRIPT) as f:
        main_src = f.read()
    for fname, path in paths.items():
        with open(path) as f:
            n_lines = sum(1 for _ in f)
        imported = f"from {fname[:-3]} import" in main_src
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


if __name__ == "__main__":
    main()
