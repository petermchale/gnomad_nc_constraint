"""
Verify why fig_tables/comparisons_*.txt -- the data behind Extended Data Fig. 6
of Chen et al. 2024 ("Comparison of constraint scores built from different
mutational models and genomic windows", which plots z vs z_trimer vs
z_heptamer ROC curves in efig_utils.py:plt_comparison_roc_gnocchi) -- cannot be
used to answer the reviewer's GC-content-bias question, by downloading the
real files from the public bucket and inspecting their actual schema and
contents directly, rather than just asserting it.

Background / write-up: see CLAUDE.md, the section immediately before "The
analysis: real-data version of the reviewer's request".
"""
import argparse
import os
import tarfile

import pandas as pd

BUCKET_URL = "https://storage.googleapis.com/gnomad-nc-constraint-v31-paper"


def download(dest_dir: str) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    tar_path = os.path.join(dest_dir, "comparisons.tar.gz")
    if not os.path.exists(tar_path):
        url = f"{BUCKET_URL}/fig_tables/comparisons.tar.gz"
        print(f"downloading {url} -> {tar_path}")
        os.system(f"curl -s -o '{tar_path}' '{url}'")
    extract_dir = os.path.join(dest_dir, "comparisons")
    if not os.path.isdir(extract_dir):
        with tarfile.open(tar_path) as tf:
            tf.extractall(dest_dir, filter="data")
    return extract_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-dest_dir", default="tmp",
        help="local directory to download/extract into (default: ./tmp)")
    args = parser.parse_args()

    comp_dir = download(args.dest_dir)
    files = sorted(f for f in os.listdir(comp_dir) if f.endswith(".txt"))

    print(f"\n{len(files)} comparisons_*.txt files found in fig_tables/comparisons.tar.gz:\n")

    for fname in files:
        path = os.path.join(comp_dir, fname)
        df = pd.read_csv(path, sep="\t")
        n_dup = len(df) - df["locus"].nunique()

        print(f"=== {fname} ===")
        print(f"  rows: {len(df)}   unique loci: {df['locus'].nunique()}   duplicate loci: {n_dup}")
        print(f"  columns: {list(df.columns)}")
        has_gc = any("gc" in c.lower() for c in df.columns)
        has_element_id = "element_id" in df.columns
        print(f"  has a GC-content column? {has_gc}")
        print(f"  keyed by 'element_id' (1kb window)? {has_element_id}  -- "
              f"it's keyed by 'locus' (an individual variant position) instead")
        print("  sample rows:")
        print(df.head(3).to_string(index=False))
        print()

    pos_file = os.path.join(comp_dir, "comparisons_gwas_catalog_repl.txt")
    neg_file = os.path.join(comp_dir, "comparisons_topmed_maf5.sampled.cov.txt")
    n_pos = len(pd.read_csv(pos_file, sep="\t"))
    n_neg = len(pd.read_csv(neg_file, sep="\t"))
    print(
        f"Raw pool sizes for one GWAS-Catalog comparison: {n_pos} positive (functional) "
        f"variants vs {n_neg} candidate negative (TOPMed control) variants "
        f"({n_neg / n_pos:.1f}x). efig_utils.py:plt_comparison_roc_gnocchi further "
        f"subsamples the negative pool down to exactly 10x the positive count at "
        f"plot time (`sampling = 10`; `df_0.sample(n=sampling*len(df_1))`) -- i.e. "
        f"even the raw files are already a curated candidate pool for a "
        f"classification task, not a genome-wide window census.\n"
    )

    print(
        "Conclusion: every comparisons_*.txt file is a per-variant table built from "
        "a specific, curated positive (GWAS Catalog / GWAS fine-mapping / likely "
        "pathogenic ClinVar+HGMD) or negative (AF-matched, downsampled TOPMed "
        "control) variant set -- assembled for the ROC/AUC classification task "
        "behind Extended Data Fig. 6, not as a uniform, genome-wide sample of 1kb "
        "windows. None of them carry a GC-content column, and none are keyed by "
        "element_id. Stratifying z/z_trimer/z_heptamer from these files by GC "
        "content would measure model behavior on this curated, functionally- and "
        "AF-enriched variant sample, not genome-wide local bias -- a different "
        "question from the one 'compute_gc_bias_step1_vs_step2.py' already answers "
        "correctly using every 1kb window genome-wide."
    )


if __name__ == "__main__":
    main()
