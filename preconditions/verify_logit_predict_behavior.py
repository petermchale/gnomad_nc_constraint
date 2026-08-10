"""
The paper's Methods state the adjustment factor as a ratio of raw logits; the code
computes a ratio of predicted probabilities. Checked here on a real fitted
per-context model from the bucket, by calling its .predict() directly.

The whole "a level error cancels" argument in fig5 panel B needs r to be that ratio.
See CLAUDE.md, "The paper's Methods do not match the code".
"""
import argparse
import math
import os
import sys

import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnocchi_bias.windows import download  # noqa: E402

# Repo-root cache, shared with every other script here; resolved from __file__ so
# running from inside preconditions/ reuses it rather than refetching multi-GB files.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DEST_DIR = os.path.join(_REPO_ROOT, "published")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-context", default="AAA",
        help="trinucleotide context whose fitted model to test (default: AAA)")
    parser.add_argument(
        "-dest_dir", default=DEFAULT_DEST_DIR,
        help=f"directory to download the pickle into (default: {DEFAULT_DEST_DIR})")
    args = parser.parse_args()

    pkl_path = download(
        f"logit_pickles/logit_regularized_dnm01_{args.context}_pbonf_pca.pkl",
        args.dest_dir)

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
