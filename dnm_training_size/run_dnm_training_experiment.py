"""
NOTE ON THE FULL-SCALE RUN. `-subsample_frac 1.0 -tag full` is superseded by
`fig5/refit.py -population full`, which writes the same tables into the shared
repo-root refits/ as {table}.full.txt. That is where fig5/ now reads the
full-scale refit from, so running it here with `-tag full` writes a second, unread
copy (~4 GB). The subsampled runs (frac0.01, frac0.1) are this experiment's own and
still belong in dnm_training_size/output/.

Resize the DNM training set (regime 1 from
okf/dnm-training-set-experiment/training-data.md: shrink BOTH dnm0 and dnm1
by a fixed random subsample rate), refit the per-context regional-feature
logistic regression (feature selection -> standardize -> PCA -> multivariate
logit), and apply it genome-wide -- to test claim 1 of
chen_formula.tex's hypothesis (okf/dnm-training-set-experiment/hypothesis.md):
that Gnocchi's GC-content bias comes from DNM-training-set sparseness in the
tails of the regional-feature space x, and should shrink toward the
context-only model's bias as the training set shrinks.

FULL METHODS NARRATIVE: see okf/dnm-training-set-experiment/pipeline.md for
the step-by-step plan this script implements, training-data.md for the input
files (with confirmed row counts/schemas), missing-code.md for what's being
reconstructed here (the multivariate PCA+logit fit has no published fitting
code anywhere in this repo or the public bucket -- only the apply/predict
side does, run_nc_constraint_gnomad_v31_main.py lines 231-249),
reusable-code.md for exactly which lines of that script are reused
unchanged vs. replaced, and open-questions.md for the hyperparameters this
script has to assume in the absence of published ground truth (statsmodels'
default L1 regularization strength, ddof=0/population standardization,
IncrementalPCA with all components kept -- verified for context AAA in
missing-code.md).

Two modes:

Lives in dnm_training_experiment/ (moved here from the repo root 2026-07-21,
alongside plot_dnm_bias_comparison.py). Downloaded bucket files are cached in
the repo-root tmp/ (-cache_dir, shared with compute_gc_bias_step1_vs_step2.py
etc. -- avoids re-downloading multi-GB files); this experiment's own computed
outputs go to dnm_training_experiment/output/ (-output_dir) -- see
DEFAULT_CACHE_DIR/DEFAULT_OUTPUT_DIR below.

-mode validate: runs the univariate feature-selection fit
  (analyze_individual_feature_effects.py's own logic -- that script has
  undefined-name bugs: missing `import os`/`csv` and an undefined
  `output_dir`, fixed here by threading a real cache_dir through properly) on
  the FULL, unmodified training data, and diffs the result against the
  published
  genomic_features/dnm01_10x_ft_logit_regularized_coef_z_3mer_context_flnk_1k-1M.txt.
  Run this first (pipeline.md step 0) before trusting any subsampled refit
  below -- if this doesn't match, a subsampled refit could be diverging from
  the published pipeline for an unrelated reason (e.g. a bug in this
  reimplementation), not because of subsampling.

-mode refit -subsample_frac F: regime 1 -- randomly subsample both dnm0 and
  dnm1 at rate F (independently), then (1) refit univariate feature
  selection on the subsample; (2) refit the multivariate PCA+logit model per
  context, using ONLY that subsample's own selected features and its own
  standardization mean/std (not the published ones -- reusable-code.md,
  "Needs replacement"); (3) apply the new per-context models genome-wide,
  reusing run_nc_constraint_gnomad_v31_main.py lines 236-270 unchanged
  (reusable-code.md, "Fully reusable, unchanged"); (4) write a new
  element_id/possible/expected table, same shape as the published
  expected_counts_by_context_methyl_genome_1kb.txt (the r==1 table), for
  direct use by plot_dnm_bias_comparison.py. A context with no significant
  features under the subsample (or a fit that fails to converge) is left out
  of the adjustment entirely and defaults to r==1 for that context, exactly
  matching run_nc_constraint_gnomad_v31_main.py line 260's own fallback --
  which is also, mechanistically, exactly what hypothesis.md claim 1
  predicts happens as training data shrinks.

  -mode refit ALSO reports, per context, exactly which (feature, window)
  pairs were Bonferroni-selected (see refit_and_apply()'s printed output and
  the written selected.dnm_refit_{tag}.txt), plus a feature-selection
  frequency table across contexts -- e.g. to check whether GC_content
  specifically gets selected more often as the training set grows. And it
  writes a two-panel "training-set GC diagnostic" plot
  (gc_diagnostic.dnm_refit_{tag}.pdf, see plot_training_gc_diagnostic()):
  top panel is the GC-content distribution of the (possibly subsampled)
  dnm1/dnm0 training examples themselves -- directly showing how sparse the
  tails get at small subsample sizes; bottom panel is a POOLED (all
  contexts combined, GC content only, ignoring every other feature)
  univariate logistic fit of mutation status on GC content, plotted against
  binned empirical mutation proportions from the same training pool. This
  is deliberately NOT the actual per-context multivariate model Gnocchi
  uses (that can't be reduced to a single GC-only curve, since it's fit
  jointly on several PCA-whitened features and differs by context/window)
  -- it's a simplified, context-agnostic diagnostic to visualize, across
  subsample sizes, how a GC-content-driven fit's sensitivity to the tails
  (slope out where training points are sparse) changes as more (sparse,
  noisy) tail data becomes available, without necessarily tracking the
  tails' true underlying rate any better -- the qualitative mechanism this
  whole experiment is testing.

-mode reliability -subsample_frac F: a training-set-only reliability
  diagram / calibration curve, using the REAL per-context multivariate
  model (not the pooled GC-only proxy above). For every dnm1/dnm0 site in
  the (subsampled) training set, refits that site's own context's
  multivariate model exactly as in -mode refit, then predicts on the
  SITE'S OWN feature vector (not a window's aggregated feature values --
  see apply_genome_wide_context vs predict_training_set) and compares the
  mean fitted probability against the mean empirical label rate, binned by
  the site's own GC content. Deliberately stays within the training
  population (never touches the genome-wide features/expected-count
  files), which sidesteps a real problem with comparing predictions to
  genome-wide rates directly: dnm0:dnm1 is a fixed ~10:1 case-control
  design, not the true (much rarer) genome-wide DNM rate, so a fitted
  model's raw predicted probability carries a case-control intercept bias
  that isn't calibrated to genome-wide rates. Comparing prediction to label
  within the same case-control-sampled population cancels that bias out,
  since it affects both sides equally -- see
  okf/dnm-training-set-experiment/log.md's reliability-diagram entry.
"""

import argparse
import os
import sys
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

# The training data, the fitting core, and the genome-wide apply step now live
# in gnocchi_bias.dnm_model, shared with fig5/ (which needs the same
# per-context multivariate fit for its calibration panel). This script keeps
# the CLI and the plots that only the training-set-size experiment uses.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from gnocchi_bias.dnm_model import (
    TRAINING_FILES, PUBLISHED_COEF_FILE, MUTATION_RATE_FILE,
    GENOME_FEATURES_FILE, GENOME_EXPECTED_PERCONTEXT_FILE,
    FT_COLS, WINDOWS, CPG_CONTEXTS, FT_CORR_MET,
    load_training_data, subsample_regime1,
    fit_univariate, bonferroni_select, select_features_for_context,
    fit_multivariate_context, apply_genome_wide_context, combine_and_predict,
    report_selected_features, predict_training_set, bin_training_reliability,
    refit_and_apply, validate_against_published,
)
from gnocchi_bias.windows import download

DEFAULT_CACHE_DIR = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", "tmp"))
DEFAULT_OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "output")


def plot_training_gc_diagnostic(df_dnm1: pd.DataFrame, df_dnm0: pd.DataFrame, gc_col: str,
                                 output_path: str, n_bins: int, tag: str) -> None:
    """
    Two-panel diagnostic of the (possibly subsampled) training pool's
    coverage of GC content, deliberately POOLED across all contexts and
    ignoring every feature but GC content -- see the module docstring for
    why this is a simplified companion visualization, not the actual
    per-context multivariate model Gnocchi uses.

    Top panel: histogram of gc_col (density-normalized, since dnm0 outnumbers
    dnm1 ~10:1) for dnm1 (mutated) vs dnm0 (non-mutated) training examples --
    shows directly how sparse the GC-content tails of the training pool are
    at this subsample size.

    Bottom panel: a pooled univariate logistic regression of mutation status
    (1=dnm1, 0=dnm0) on z-scored gc_col (same fit_regularized() convention as
    fit_univariate(), just pooled across contexts instead of per-context),
    plotted as a smooth probability curve against the *training pool's own*
    binned empirical mutation proportion (points, with binomial standard-error
    bars) -- so the fitted curve's behavior in the tails can be visually
    compared to how few/noisy the points actually backing it are there.
    """
    gc1 = df_dnm1[gc_col].dropna()
    gc0 = df_dnm0[gc_col].dropna()
    if len(gc1) == 0 or len(gc0) == 0:
        print(f"  skipping GC diagnostic plot: no non-null {gc_col} in dnm1 and/or dnm0")
        return

    gc_all = pd.concat([gc1, gc0], ignore_index=True)
    group_all = pd.Series([1] * len(gc1) + [0] * len(gc0))

    bins = np.linspace(gc_all.min(), gc_all.max(), n_bins + 1)

    fig, (ax_hist, ax_fit) = plt.subplots(2, 1, figsize=(7, 9))

    ax_hist.hist(gc0, bins=bins, alpha=0.5, density=True, color="tab:blue",
                 label=f"dnm0 non-mutated background, n={len(gc0):,}")
    ax_hist.hist(gc1, bins=bins, alpha=0.5, density=True, color="tab:orange",
                 label=f"dnm1 real DNM, n={len(gc1):,}")
    ax_hist.set_xlabel(gc_col)
    ax_hist.set_ylabel("density")
    ax_hist.set_title(f"Training-set GC content distribution ({tag})")
    ax_hist.legend(fontsize=9)

    gc_mean, gc_std = gc_all.mean(), gc_all.std(ddof=0)
    df_fit = pd.DataFrame({'gc_z': (gc_all - gc_mean) / gc_std})
    logit = None
    if gc_std > 0:
        try:
            logit = sm.Logit(group_all.values, sm.add_constant(df_fit, has_constant='add')).fit_regularized(disp=0)
        except Exception:
            logit = None

    bin_idx = np.clip(np.digitize(gc_all, bins[1:-1]), 0, n_bins - 1)
    df_bin = pd.DataFrame({'bin': bin_idx, 'group': group_all.values, 'gc': gc_all.values})
    summary = df_bin.groupby('bin').agg(n=('group', 'size'), n1=('group', 'sum'), gc_mid=('gc', 'mean'))
    summary['prop'] = summary['n1'] / summary['n']
    summary['se'] = np.sqrt(summary['prop'] * (1 - summary['prop']) / summary['n'])

    ax_fit.errorbar(summary['gc_mid'], summary['prop'], yerr=summary['se'], fmt='o', color='black',
                     capsize=3, markersize=4, label='binned empirical P(mutated), +/-1 SE')
    if logit is not None:
        gc_grid = np.linspace(bins[0], bins[-1], 200)
        df_grid = pd.DataFrame({'gc_z': (gc_grid - gc_mean) / gc_std})
        p_grid = logit.predict(sm.add_constant(df_grid, has_constant='add'))
        ax_fit.plot(gc_grid, p_grid, color='crimson', linewidth=2,
                    label=f'pooled univariate logistic fit (z-coef={logit.params.iloc[1]:.3f})')
    else:
        ax_fit.text(0.5, 0.5, "fit failed to converge", transform=ax_fit.transAxes,
                    ha='center', color='crimson')
    ax_fit.set_xlabel(gc_col)
    ax_fit.set_ylabel("P(mutated)")
    ax_fit.set_title(f"Pooled logistic fit vs GC content only ({tag})")
    ax_fit.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  wrote GC diagnostic plot -> {output_path}")




def plot_training_reliability_diagram(df_pred: pd.DataFrame, output_path: str, n_bins: int, tag: str) -> pd.DataFrame:
    """
    Reliability diagram / calibration curve on the training set itself: bins
    sites (pooled across all fitted contexts) by their own GC content, and
    compares the mean FITTED probability (each site's own context-specific
    multivariate model prediction) against the mean EMPIRICAL label rate
    (fraction of dnm1 among all sites) in that bin, with binomial standard
    error on the empirical side.

    The binning now lives in gnocchi_bias.dnm_model.bin_training_reliability()
    so fig5/ can reuse it without importing this script's plotting; this
    function is just the plot around it, and still returns the same binned
    summary it always did (written as training_reliability_binned.*.txt).
    """
    if df_pred.empty:
        print("  skipping reliability diagram: no contexts fit")
        return pd.DataFrame()

    summary = bin_training_reliability(df_pred, n_bins)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(summary['gc_mid'], summary['empirical_prop'], yerr=summary['se'], fmt='o-',
                color='black', capsize=3, markersize=4, label='empirical P(DNM), training set, +/-1 SE')
    ax.plot(summary['gc_mid'], summary['mean_pred'], 'o-', color='crimson',
            label='mean fitted P(DNM) (per-context multivariate model)')
    ax.set_xlabel('GC content (site-level)')
    ax.set_ylabel('P(site is a DNM)')
    ax.set_title(f'Training-set reliability diagram vs GC content ({tag})')
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  wrote reliability diagram -> {output_path}")
    return summary



def main():
    # Non-interactive backend set here, not at import time, so the notebooks
    # in this directory and in fig5/ can import this module's plotters while
    # keeping their own inline backend.
    matplotlib.use("Agg")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-mode", choices=["validate", "refit", "reliability"], required=True)
    parser.add_argument("-cache_dir", default=DEFAULT_CACHE_DIR,
                         help="local dir for downloaded bucket files, shared with other scripts in this repo "
                              f"(default: repo-root tmp/, resolved as {DEFAULT_CACHE_DIR})")
    parser.add_argument("-output_dir", default=DEFAULT_OUTPUT_DIR,
                         help="local dir for this experiment's own computed outputs -- coefficients, selected "
                              "features, rr, diagnostic plots, final adjusted expected-count tables "
                              f"(default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("-subsample_frac", type=float, default=None,
                         help="refit mode only: regime-1 random subsample rate applied independently to "
                              "both dnm0 and dnm1 (e.g. 0.01 for 1%%)")
    parser.add_argument("-random_seed", type=int, default=0)
    parser.add_argument("-tag", default=None,
                         help="refit mode only: label for output filenames; default derived from "
                              "-subsample_frac/-random_seed")
    parser.add_argument("-max_contexts", type=int, default=None,
                         help="DEBUG ONLY: restrict to the first N contexts (alphabetical), for a fast "
                              "smoke test of the code -- not a real result at any N < 32")
    parser.add_argument("-gc_col", default="GC_content_1k",
                         help="refit mode only: which regional-feature column to use for the training-set "
                              "GC diagnostic plot (default: GC_content_1k, matching the rest of this repo's "
                              "GC-binned bias analysis -- see compute_gc_bias_step1_vs_step2.py)")
    parser.add_argument("-gc_diagnostic_bins", type=int, default=20)
    parser.add_argument("-plot_gc_diagnostic", action="store_true", default=True,
                         help="refit mode only: write the training-set GC-distribution + pooled-logistic-fit "
                              "diagnostic plot (see plot_training_gc_diagnostic()); on by default")
    parser.add_argument("-no_plot_gc_diagnostic", dest="plot_gc_diagnostic", action="store_false")
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    df_dnm1, df_dnm0 = load_training_data(args.cache_dir)
    mutation_rate_path = download(MUTATION_RATE_FILE, args.cache_dir)
    contexts = sorted(pd.read_csv(mutation_rate_path, sep='\t')['context'].unique())
    if args.max_contexts is not None:
        contexts = contexts[:args.max_contexts]
    print(f"{len(contexts)} contexts, dnm1={len(df_dnm1):,} rows, dnm0={len(df_dnm0):,} rows")

    if args.mode == "validate":
        print("\nfitting univariate feature selection on FULL, unmodified training data "
              "(pipeline.md step 0)...")
        t0 = time.time()
        df_coef = fit_univariate(df_dnm1, df_dnm0, contexts)
        print(f"done in {time.time() - t0:.1f}s")
        out_path = os.path.join(args.output_dir, "coef_univariate.validate.txt")
        df_coef.to_csv(out_path, sep='\t', index=False)
        print(f"wrote {out_path}")

        published_path = download(PUBLISHED_COEF_FILE, args.cache_dir)
        validate_against_published(df_coef, published_path)
        return

    if args.mode == "reliability":
        if args.subsample_frac is None:
            raise ValueError("-subsample_frac is required in -mode reliability")
        tag = args.tag or f"frac{args.subsample_frac}_seed{args.random_seed}"

        df_dnm1_sub, df_dnm0_sub = subsample_regime1(df_dnm1, df_dnm0, args.subsample_frac, args.random_seed)
        print(f"\nsubsampled (regime 1, frac={args.subsample_frac}, seed={args.random_seed}): "
              f"dnm1 {len(df_dnm1):,}->{len(df_dnm1_sub):,}, dnm0 {len(df_dnm0):,}->{len(df_dnm0_sub):,}")

        print("refitting univariate feature selection...")
        t0 = time.time()
        df_coef = fit_univariate(df_dnm1_sub, df_dnm0_sub, contexts)
        print(f"  done in {time.time() - t0:.1f}s")
        df_sel = bonferroni_select(df_coef)
        print(f"  selected {len(df_sel)} (context,feature) rows across "
              f"{df_sel['context'].nunique()}/{len(contexts)} contexts")
        report_selected_features(df_sel)

        print("fitting multivariate models + predicting on the training set itself...")
        t0 = time.time()
        df_pred = predict_training_set(df_dnm1_sub, df_dnm0_sub, contexts, df_sel, args.gc_col)
        print(f"  done in {time.time() - t0:.1f}s ({len(df_pred):,} site predictions)")

        pred_path = os.path.join(args.output_dir, f"training_reliability_predictions.dnm_refit_{tag}.txt")
        df_pred.to_csv(pred_path, sep='\t', index=False)
        print(f"  wrote per-site predictions -> {pred_path}")

        plot_path = os.path.join(args.output_dir, f"training_reliability.dnm_refit_{tag}.pdf")
        summary = plot_training_reliability_diagram(df_pred, plot_path, args.gc_diagnostic_bins, tag)
        summary_path = os.path.join(args.output_dir, f"training_reliability_binned.dnm_refit_{tag}.txt")
        summary.to_csv(summary_path, sep='\t', index=False)
        print(f"  wrote binned summary -> {summary_path}")
        return

    # -mode refit
    if args.subsample_frac is None:
        raise ValueError("-subsample_frac is required in -mode refit")
    tag = args.tag or f"frac{args.subsample_frac}_seed{args.random_seed}"

    df_dnm1_sub, df_dnm0_sub = subsample_regime1(df_dnm1, df_dnm0, args.subsample_frac, args.random_seed)
    print(f"\nsubsampled (regime 1, frac={args.subsample_frac}, seed={args.random_seed}): "
          f"dnm1 {len(df_dnm1):,}->{len(df_dnm1_sub):,}, dnm0 {len(df_dnm0):,}->{len(df_dnm0_sub):,}")

    if args.plot_gc_diagnostic:
        gc_diag_path = os.path.join(args.output_dir, f"gc_diagnostic.dnm_refit_{tag}.pdf")
        plot_training_gc_diagnostic(df_dnm1_sub, df_dnm0_sub, args.gc_col, gc_diag_path,
                                     args.gc_diagnostic_bins, tag)

    print("\nloading genome-wide features table (large; cached after first run)...")
    genome_ft_path = download(GENOME_FEATURES_FILE, args.cache_dir)
    t0 = time.time()
    df_ft_genome = pd.read_csv(genome_ft_path, sep='\t')
    print(f"  loaded {len(df_ft_genome):,} windows in {time.time() - t0:.1f}s")

    genome_exp_path = download(GENOME_EXPECTED_PERCONTEXT_FILE, args.cache_dir)

    t0 = time.time()
    df_out = refit_and_apply(df_dnm1_sub, df_dnm0_sub, contexts, df_ft_genome, genome_exp_path, args.output_dir, tag)
    print(f"\nrefit+apply total: {time.time() - t0:.1f}s")

    out_path = os.path.join(args.output_dir, f"expected_counts_by_context_methyl_genome_1kb.dnm_refit_{tag}.txt")
    df_out.to_csv(out_path, sep='\t', index=False)
    print(f"wrote {out_path} ({len(df_out):,} rows)")


if __name__ == "__main__":
    main()
