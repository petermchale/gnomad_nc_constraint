"""
Verify two claims in CLAUDE.md about misc/generic.py, misc/constraint_basics.py,
and misc/nc_constraint_utils.py, by downloading the real files from the public
bucket and inspecting them directly:

1. These three modules are NOT missing from the published bucket (CLAUDE.md
   used to say they were) -- they're at misc/*.py, and are exactly what
   run_nc_constraint_gnomad_v31_main.py imports (`from generic import *`, etc.,
   lines 23-25).
2. None of the three contain the multivariate PCA + regularized-logistic fit
   step that actually computes Gnocchi's r(w) -- only the apply/predict side
   of that model is published (run_nc_constraint_gnomad_v31_main.py:231-249).
   So that specific gap is real, not just an artifact of an incomplete local
   checkout.

Background / write-up: see CLAUDE.md, "The next experiment: DNM training-set
size vs. Gnocchi's local bias" -> "The fitting code that's actually here --
and the gap".
"""
import argparse
import os
import re

BUCKET_URL = "https://storage.googleapis.com/gnomad-nc-constraint-v31-paper"
FILES = ["generic.py", "constraint_basics.py", "nc_constraint_utils.py"]
MAIN_SCRIPT = "run_nc_constraint_gnomad_v31_main.py"
GAP_PATTERNS = ["IncrementalPCA", r"\bPCA\b", "fit_regularized"]


def download(dest_dir: str) -> dict:
    os.makedirs(dest_dir, exist_ok=True)
    paths = {}
    for fname in FILES:
        dest_path = os.path.join(dest_dir, fname)
        if not os.path.exists(dest_path):
            url = f"{BUCKET_URL}/misc/{fname}"
            print(f"downloading {url} -> {dest_path}")
            os.system(f"curl -s -o '{dest_path}' '{url}'")
        paths[fname] = dest_path
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-dest_dir", default="tmp",
        help="local directory to download the three files into (default: ./tmp)")
    args = parser.parse_args()

    paths = download(args.dest_dir)

    print("\n=== Claim 1: these modules exist and match run_nc_constraint_gnomad_v31_main.py's imports ===")
    for fname, path in paths.items():
        size = os.path.getsize(path)
        n_lines = sum(1 for _ in open(path))
        module_name = fname[:-3]
        with open(MAIN_SCRIPT) as f:
            imported = any(f"from {module_name} import" in line for line in f)
        print(f"  misc/{fname}: {size} bytes, {n_lines} lines -- "
              f"imported by {MAIN_SCRIPT}? {imported}")

    print("\n=== Claim 2: none of the three contain the missing multivariate-fit step ===")
    any_hit = False
    for fname, path in paths.items():
        text = open(path).read()
        hits = [p for p in GAP_PATTERNS if re.search(p, text)]
        if hits:
            any_hit = True
        print(f"  misc/{fname}: matches for {GAP_PATTERNS} -> {hits or 'none'}")

    print(
        f"\nConclusion: {'unexpected match found -- re-examine the gap claim in CLAUDE.md' if any_hit else 'confirmed -- zero matches across all three files for PCA/IncrementalPCA/fit_regularized. The multivariate PCA+logistic fit step behind Gnocchi r(w) genuinely has no published source in this bucket; only run_nc_constraint_gnomad_v31_main.py:231-249 (the apply/predict side) is available.'}"
    )


if __name__ == "__main__":
    main()
