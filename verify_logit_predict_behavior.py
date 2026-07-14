"""
Empirically verify the discrepancy between the paper's stated adjustment-factor
formula (a ratio of raw logits) and what run_nc_constraint_gnomad_v31_main.py
actually computes (a ratio of predicted probabilities), by downloading one real
fitted per-context logistic-regression model from the public bucket and calling
its .predict() directly.

Background / write-up: see CLAUDE.md, "Confirmed finding: the paper's Methods
text does not match the code".
"""
import argparse
import os

import pandas as pd
import statsmodels.api as sm

BUCKET_URL = "https://storage.googleapis.com/gnomad-nc-constraint-v31-paper"


def download(context: str, dest_dir: str) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    fname = f"logit_regularized_dnm01_{context}_pbonf_pca.pkl"
    dest_path = os.path.join(dest_dir, fname)
    if not os.path.exists(dest_path):
        url = f"{BUCKET_URL}/logit_pickles/{fname}"
        print(f"downloading {url} -> {dest_path}")
        os.system(f"curl -s -o '{dest_path}' '{url}'")
    return dest_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-context", default="AAA",
        help="trinucleotide context whose fitted model to test (default: AAA)")
    parser.add_argument(
        "-dest_dir", default="tmp",
        help="local directory to download the pickle into (default: ./tmp)")
    args = parser.parse_args()

    pkl_path = download(args.context, args.dest_dir)

    logit = pd.read_pickle(pkl_path)
    print("\ntype(logit):", type(logit))
    print("MRO:", [c.__name__ for c in type(logit).__mro__])

    n_params = len(logit.params) - 1  # exclude the intercept
    zero_row = sm.add_constant(
        pd.DataFrame([[0] * n_params]), has_constant="add")

    prob = logit.predict(zero_row).iloc[0]
    linear = logit.predict(zero_row, which="linear").iloc[0]
    intercept = logit.params.iloc[0]

    print(f"\nlogit.predict(zero_row)                    = {prob:.4f}  "
          f"(a probability, in (0,1))")
    print(f"logit.predict(zero_row, which='linear')    = {linear:.4f}  "
          f"(== logit.params[0] == {intercept:.4f}, the intercept)")

    import math
    sigmoid_of_linear = 1 / (1 + math.exp(-linear))
    print(f"\nsigma(linear) = {sigmoid_of_linear:.4f}  "
          f"(matches predict()'s default output: {prob:.4f})")

    print(
        "\nConclusion: logit.predict() defaults to sigma(linear predictor), a "
        "probability, not the raw logit the paper's Methods text describes. "
        "The pipeline's r = pred_ctx / ave is therefore a ratio of predicted "
        "probabilities, sigma(beta0 + beta.z(w)) / sigma(beta0), not a ratio "
        "of logits beta.x(w) / beta.xbar as stated in the paper."
    )


if __name__ == "__main__":
    main()
