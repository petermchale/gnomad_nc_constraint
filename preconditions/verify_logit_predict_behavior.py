"""
The paper's Methods state the adjustment factor as a ratio of raw logits; the code
computes a ratio of predicted probabilities. Checked here on a real fitted
per-context model from the bucket, by calling its .predict() directly.

The whole "a level error cancels" argument in fig5 panel B needs r to be that ratio.
See CLAUDE.md, "The paper's Methods do not match the code".

Outcome of the last run: preconditions/output/STATUS.md (transcript in the .log beside it).
"""
import argparse
import math
import os
import sys

import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnocchi_bias.windows import CACHE_DIR as W_CACHE_DIR, download  # noqa: E402
from preconditions.report import Report  # noqa: E402

# Repo-root cache, shared with every other script here -- or wherever
# $GNOCCHI_PUBLISHED_DIR points, so a run here reuses the same multi-GB files as
# every other entry point rather than refetching them.
DEFAULT_DEST_DIR = W_CACHE_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-context", default="AAA",
        help="trinucleotide context whose fitted model to test (default: AAA)")
    parser.add_argument(
        "-dest_dir", default=DEFAULT_DEST_DIR,
        help=f"directory to download the pickle into (default: {DEFAULT_DEST_DIR})")
    args = parser.parse_args()

    with Report("verify_logit_predict_behavior") as rep:
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

        # Three claims rather than one, because "predict() returns a probability" is
        # only convincing alongside the two that pin down what it is a probability OF:
        # that the linear predictor at z = 0 is the bare intercept, and that the
        # default output is its sigmoid. Together they identify the denominator of
        # r as sigma(beta0), which is the quantity the Methods text gets wrong.
        rep.claim(0 < prob < 1 and abs(prob - linear) > 1e-6,
                  f"predict() on the real {args.context} model returns {prob:.4f}, a "
                  f"probability in (0,1) -- not the linear predictor {linear:.4f}")
        rep.claim(abs(linear - intercept) < 1e-12,
                  f"predict(which='linear') at z = 0 equals the fitted intercept "
                  f"({linear:.6f} vs {intercept:.6f}), so r's denominator is sigma(beta0)")
        rep.claim(abs(sigmoid_of_linear - prob) < 1e-12,
                  f"sigma(linear predictor) reproduces predict()'s default output to "
                  f"{abs(sigmoid_of_linear - prob):.1e}, confirming the sigmoid is applied")


if __name__ == "__main__":
    main()
