"""
The DNM training set and Gnocchi's per-context regional-feature mutation model.

Extracted verbatim (2026-08-04) from dnm_training_experiment/
run_dnm_training_experiment.py, so that both figure directories can import the
same fitting code:

  fig3/              needs fit_multivariate_context + predict_training_set, for
                     the training-set reliability/calibration gap (panel B).
  dnm_training_size/ needs all of it, plus apply_genome_wide_context /
                     combine_and_predict, to refit under a resized training set
                     and re-derive genome-wide expected counts.

Provenance, unchanged from the original and load-bearing for the manuscript:
fit_univariate + bonferroni_select reimplement the published feature-selection
step (analyze_individual_feature_effects.py) and validate against the published
coefficient table to max |coef diff| = 2.6e-4. fit_multivariate_context is the
one step with NO published source anywhere -- reconstructed from the apply-side
code in run_nc_constraint_gnomad_v31_main.py:231-249 -- and is validated
end-to-end instead: at frac=1.0 it reproduces the published Gnocchi `expected`
column with Pearson r = 1.0 over 1,984,900 windows. See CLAUDE.md.

This module deliberately does NOT call matplotlib.use() or import pyplot: it is
the model, not the plots, and it must import cleanly inside a notebook.
"""
import time
from functools import reduce

import duckdb
import numpy as np
import pandas as pd
import scipy.stats
import statsmodels.api as sm
from sklearn.decomposition import IncrementalPCA

from .windows import download

TRAINING_FILES = {
    "dnm1_sites": "genomic_features/DNM_decode_psychencode_site_context.mutation_rate.txt",
    "dnm1_features": "genomic_features/genomic_features13_dnm1_flnk_1k-1M.txt",
    "dnm0_sites": "genomic_features/context_prefiltered_nonmutated-dnm_sites10xdnm.mutation_rate.txt",
    "dnm0_features": "genomic_features/genomic_features13_dnm0_10x_flnk_1k-1M.txt",
}
PUBLISHED_COEF_FILE = "genomic_features/dnm01_10x_ft_logit_regularized_coef_z_3mer_context_flnk_1k-1M.txt"
MUTATION_RATE_FILE = "fig_tables/mutation_rate_by_context_methyl.txt"
GENOME_FEATURES_FILE = "misc/genomic_features13_genome_1kb.txt"
GENOME_EXPECTED_PERCONTEXT_FILE = "expected_counts_per_context_methyl_genome_1kb.txt"

# analyze_individual_feature_effects.py lines 22-24
FT_COLS = ['dist2telo', 'dist2cent', 'LCR', 'SINE', 'LINE', 'GC_content',
           'recomb_male', 'recomb_female', 'met_sperm', 'Nucleosome', 'CpG_island',
           'cDNM_maternal_05M', 'cDNM_paternal_05M']
WINDOWS = ['1k', '10k', '100k', '1M']
CPG_CONTEXTS = ['ACG', 'CCG', 'GCG', 'TCG']
# run_nc_constraint_gnomad_v31_main.py line 217: methylation-correlated features,
# excluded from CpG-context models (line 227)
FT_CORR_MET = ['GC_content', 'SINE', 'met_sperm', 'Nucleosome', 'CpG_island']


def load_contexts(cache_dir: str) -> list:
    """The 32 trinucleotide contexts, read from the published mutation-rate table."""
    path = download(MUTATION_RATE_FILE, cache_dir)
    return sorted(pd.read_csv(path, sep='\t')['context'].unique())


def load_training_data(dest_dir: str):
    """
    Load and join dnm1 (mutated) and dnm0 (non-mutated background) site
    tables with their regional-feature tables -- replicates
    analyze_individual_feature_effects.py lines 13-20 exactly (bug-fixed:
    that script never defines `output_dir` or imports `os`/`csv`).
    """
    paths = {k: download(v, dest_dir) for k, v in TRAINING_FILES.items()}

    df_dnm1 = pd.read_csv(paths["dnm1_sites"], sep='\t')
    df_ft_1 = pd.read_csv(paths["dnm1_features"], sep='\t').drop_duplicates()
    df_dnm1 = df_dnm1.merge(df_ft_1.rename(columns={'element_id': 'locus'}), how='left', on='locus')

    df_dnm0 = pd.read_csv(paths["dnm0_sites"], sep='\t')
    df_dnm0 = df_dnm0[~df_dnm0['locus'].str.contains('chrX:')]
    df_ft_0 = pd.read_csv(paths["dnm0_features"], sep='\t').drop_duplicates()
    df_dnm0 = df_dnm0.merge(df_ft_0.rename(columns={'element_id': 'locus'}), how='left', on='locus')

    return df_dnm1, df_dnm0


def subsample_regime1(df_dnm1: pd.DataFrame, df_dnm0: pd.DataFrame, frac: float, seed: int):
    """training-data.md regime 1: shrink both dnm0 and dnm1, independently, at the same rate."""
    return (df_dnm1.sample(frac=frac, random_state=seed),
            df_dnm0.sample(frac=frac, random_state=seed))


def fit_univariate(df_dnm1: pd.DataFrame, df_dnm0: pd.DataFrame, contexts: list) -> pd.DataFrame:
    """
    Per (context, window, feature): univariate logistic regression of
    mutation status (dnm1=1 vs dnm0=0) on that one z-scored feature --
    analyze_individual_feature_effects.py lines 31-57, verbatim except for
    being a function over caller-supplied (possibly subsampled) dnm0/dnm1,
    and disp=0 to silence statsmodels' per-fit convergence printout (cosmetic
    only -- doesn't change any fitted value).
    """
    rows = []
    for context in contexts:
        df_1_ = df_dnm1[df_dnm1['context'] == context]
        df_0_ = df_dnm0[df_dnm0['context'] == context]
        for flnk in WINDOWS:
            for ft_ in FT_COLS:
                ft = ft_ + '_' + flnk

                df_1 = df_1_[['locus', ft]].dropna().copy()
                df_1['group'] = 1
                df_0 = df_0_[['locus', ft]].dropna().copy()
                df_0['group'] = 0
                df_01 = pd.concat([df_1, df_0])

                df_y = df_01[['group']]
                df_x = df_01[[ft]].apply(scipy.stats.zscore)

                try:
                    logit = sm.Logit(df_y, sm.add_constant(df_x[[ft]], has_constant='add')).fit_regularized(disp=0)
                except Exception:
                    coef, se, pval = np.nan, np.nan, np.nan
                else:
                    coef = logit.params[ft]
                    pval = logit.pvalues[ft]
                    lci, hci = logit.conf_int().transpose()[ft].to_list()
                    se = (hci - coef) / 1.96
                rows.append({'context': context, 'window': flnk, 'feature': ft_,
                             'coef': coef, 'se': se, 'pval': pval})
    return pd.DataFrame(rows)


def bonferroni_select(df_coef: pd.DataFrame) -> pd.DataFrame:
    """
    analyze_individual_feature_effects.py lines 62-68. Note: drop_duplicates
    keeps the FIRST significant (context, feature) row in iteration order
    (window ascending: 1k, 10k, 100k, 1M) -- i.e. the smallest significant
    window scale wins when a feature is significant at more than one window.
    This replicates the published selection logic exactly, quirk included.
    """
    df = df_coef.copy()
    df['bonf'] = np.where(df['context'].isin(CPG_CONTEXTS), 0.05 / 4 / 8, 0.05 / 4 / 13)
    df_sig = df[df['pval'] <= df['bonf']]
    df_sig = df_sig.drop_duplicates(subset=['context', 'feature'])
    return df_sig[['context', 'feature', 'window', 'coef', 'se', 'pval']].reset_index(drop=True)


def select_features_for_context(df_sel: pd.DataFrame, context: str) -> list:
    """run_nc_constraint_gnomad_v31_main.py lines 226-228 (+217, 227 for the CpG special case)."""
    df_ft_sel = df_sel[df_sel['context'] == context]
    if context in CPG_CONTEXTS:
        df_ft_sel = df_ft_sel[~df_ft_sel['feature'].isin(FT_CORR_MET)]
    return list(df_ft_sel['feature'] + '_' + df_ft_sel['window'])


def fit_multivariate_context(df_dnm1: pd.DataFrame, df_dnm0: pd.DataFrame, context: str, ft_sel: list):
    """
    The missing step (missing-code.md): fit a multivariate, PCA-whitened
    logistic regression for one context on its selected features. Mean/std
    (open-questions.md item 3: ddof=0, matching scipy.stats.zscore's default
    used by fit_univariate above) and PCA (open-questions.md item 2:
    IncrementalPCA with default -- i.e. all -- components, verified for
    context AAA in missing-code.md) are both fit fresh on this context's own
    (possibly subsampled) training pool -- NOT the published values.

    Returns (logit, pca, ft_mean, ft_std), or None if there are no selected
    features, fewer than 2 outcome classes, a zero-variance feature, too few
    rows, or fit_regularized() fails to converge.
    """
    if not ft_sel:
        return None

    df_1 = df_dnm1[df_dnm1['context'] == context][['locus'] + ft_sel].dropna().copy()
    df_1['group'] = 1
    df_0 = df_dnm0[df_dnm0['context'] == context][['locus'] + ft_sel].dropna().copy()
    df_0['group'] = 0
    df_01 = pd.concat([df_1, df_0])
    if df_01['group'].nunique() < 2 or len(df_01) < len(ft_sel) + 2:
        return None

    df_x = df_01[ft_sel]
    ft_mean = df_x.mean()
    ft_std = df_x.std(ddof=0)
    if (ft_std == 0).any():
        return None
    df_x_z = (df_x - ft_mean) / ft_std

    pca = IncrementalPCA()
    df_x_pca = pca.fit_transform(df_x_z)

    df_y = df_01[['group']]
    try:
        logit = sm.Logit(df_y, sm.add_constant(df_x_pca, has_constant='add')).fit_regularized(disp=0)
    except Exception:
        return None

    return logit, pca, ft_mean, ft_std


def apply_genome_wide_context(df_ft_genome: pd.DataFrame, context: str, ft_sel: list, fitted) -> pd.DataFrame:
    """
    run_nc_constraint_gnomad_v31_main.py lines 236-249, unchanged apart from
    taking the fitted objects/ft_sel as arguments instead of loading them
    from disk. `ave` is deliberately predicted directly on an all-zero
    (n_features,) row WITHOUT going through pca.transform -- matching the
    original code's shortcut, which is valid because the training features
    were already standardized to zero mean before PCA, so PCA (an orthogonal
    rotation) maps the origin to the origin: ave == sigma(intercept), the
    CLAUDE.md-documented "r(w) = sigma(beta0 + beta.z(w)) / sigma(beta0)" formula.
    """
    logit, pca, ft_mean, ft_std = fitted
    df_x = df_ft_genome[['element_id'] + ft_sel].drop_duplicates().dropna()
    x = df_x[ft_sel]
    x_z = (x - ft_mean) / ft_std
    x_pca = pca.transform(x_z)

    df_adj = df_x[['element_id']].copy()
    df_adj['pred'] = logit.predict(sm.add_constant(x_pca, has_constant='add'))
    ave = logit.predict(sm.add_constant(pd.DataFrame([[0] * len(ft_sel)]), has_constant='add'))[0]
    df_adj['rr'] = df_adj['pred'] / ave
    return df_adj[['element_id', 'rr']]


def combine_and_predict(df_adj_by_context: dict, contexts: list, genome_expected_percontext_path: str) -> pd.DataFrame:
    """
    run_nc_constraint_gnomad_v31_main.py lines 250-270, generalized to
    handle contexts with no fitted model (defaults to r==1, via COALESCE)
    and to avoid loading the 79M-row / 3.1 GB per-context expected-count
    file into pandas: the final join-and-sum runs in duckdb, streamed
    straight from that file on disk, against a small in-memory wide table of
    per-context rr values (reduce-merged exactly as in the original code).
    """
    if df_adj_by_context:
        df_adj_l = [df.rename(columns={'rr': f'rr_{context}'}) for context, df in df_adj_by_context.items()]
        df_adj = reduce(lambda l, r: pd.merge(l, r, on='element_id', how='outer'), df_adj_l).fillna(1)
    else:
        df_adj = pd.DataFrame(columns=['element_id'])

    con = duckdb.connect()
    con.register('rr_wide', df_adj)

    case_terms = []
    for context in contexts:
        if context in df_adj_by_context:
            case_terms.append(f"WHEN '{context}' THEN COALESCE(w.rr_{context}, 1)")
        else:
            case_terms.append(f"WHEN '{context}' THEN 1")
    case_expr = "CASE e.context " + " ".join(case_terms) + " ELSE 1 END"

    query = f"""
        SELECT e.element_id AS element_id,
               SUM(e.possible) AS possible,
               SUM(e.expected * {case_expr}) AS expected
        FROM read_csv_auto('{genome_expected_percontext_path}', delim='\t', header=True) e
        LEFT JOIN rr_wide w ON e.element_id = w.element_id
        GROUP BY e.element_id
    """
    return con.execute(query).df()


def report_selected_features(df_sel: pd.DataFrame) -> None:
    """
    Print exactly which (feature, window) pairs were Bonferroni-selected,
    per context, plus a feature-selection frequency table across contexts --
    e.g. to see whether GC_content specifically gets selected more often as
    the training set grows.
    """
    if df_sel.empty:
        print("  (no features selected in any context)")
        return

    print("  selected features by context:")
    for context, g in df_sel.sort_values(['context', 'window']).groupby('context'):
        feats = ', '.join(f"{r.feature}_{r.window}" for r in g.itertuples())
        print(f"    {context}: {feats}")

    print("  feature-selection frequency (contexts selecting this feature, any window):")
    freq = df_sel['feature'].value_counts()
    for feat, cnt in freq.items():
        marker = "  <-- GC content" if feat == "GC_content" else ""
        print(f"    {feat}: {cnt}{marker}")


# ------------------------------------------------- training-set calibration

def predict_training_set(df_dnm1: pd.DataFrame, df_dnm0: pd.DataFrame, contexts: list,
                          df_sel: pd.DataFrame, gc_col: str = "GC_content_1k") -> pd.DataFrame:
    """
    Reliability-diagram data: for every training-set site (the same dnm1/dnm0
    rows used to fit each context's multivariate model), predict that site's
    own probability from its own context's fitted model, evaluated on the
    SITE'S OWN feature vector -- unlike apply_genome_wide_context, which
    evaluates once per (window, context) at the window's aggregated feature
    values and applies that single number to every possible site in the
    window. Because both the prediction and the label being compared against
    it come from the same case-control-sampled (dnm0:dnm1 ~10:1) population,
    the intercept bias that sampling scheme induces cancels out of the
    comparison.

    Contexts with no significant selected feature (or a fit that fails to
    converge) contribute no rows, matching how they're excluded from the
    genome-wide r(w) adjustment (default r=1) in refit_and_apply().
    """
    rows = []
    n_fit, n_skip = 0, 0
    for context in contexts:
        ft_sel = select_features_for_context(df_sel, context)
        fitted = fit_multivariate_context(df_dnm1, df_dnm0, context, ft_sel)
        if fitted is None:
            n_skip += 1
            continue
        n_fit += 1
        logit, pca, ft_mean, ft_std = fitted

        extra_cols = [] if gc_col in ft_sel else [gc_col]
        cols = ['locus'] + ft_sel + extra_cols
        df_1 = df_dnm1[df_dnm1['context'] == context][cols].dropna().copy()
        df_1['label'] = 1
        df_0 = df_dnm0[df_dnm0['context'] == context][cols].dropna().copy()
        df_0['label'] = 0
        df01 = pd.concat([df_1, df_0], ignore_index=True)
        if df01.empty:
            continue

        x_z = (df01[ft_sel] - ft_mean) / ft_std
        x_pca = pca.transform(x_z)
        pred = logit.predict(sm.add_constant(x_pca, has_constant='add'))

        rows.append(pd.DataFrame({
            'context': context,
            'gc': df01[gc_col].values,
            'label': df01['label'].values,
            'pred': np.asarray(pred),
        }))

    print(f"  training-set predictions: {n_fit}/{len(contexts)} contexts fit "
          f"({n_skip} skipped: no significant features or fit failure)")
    if not rows:
        return pd.DataFrame(columns=['context', 'gc', 'label', 'pred'])
    return pd.concat(rows, ignore_index=True)


def bin_training_reliability(df_pred: pd.DataFrame, n_bins: int = 20) -> pd.DataFrame:
    """
    Bin per-site predictions (from predict_training_set) by the site's own GC
    content, and compare mean FITTED probability against the mean EMPIRICAL
    label rate, with binomial standard error on the empirical side.

    No possible-weighting needed here, unlike a genome-wide window-level
    comparison -- each row is one real training-set site, the natural unit,
    not a window standing in for many possible sites.

    This is the binning half of what used to be
    plot_training_reliability_diagram(); the plotting half now lives with the
    figure that uses it. Returns columns:
      bin, n, n1, gc_mid, mean_pred, empirical_prop, se
    """
    if df_pred.empty:
        return pd.DataFrame(columns=['bin', 'n', 'n1', 'gc_mid', 'mean_pred', 'empirical_prop', 'se'])

    bins = np.linspace(df_pred['gc'].min(), df_pred['gc'].max(), n_bins + 1)
    bin_idx = np.clip(np.digitize(df_pred['gc'], bins[1:-1]), 0, n_bins - 1)
    df = df_pred.assign(bin=bin_idx)
    summary = df.groupby('bin').agg(
        n=('label', 'size'), n1=('label', 'sum'),
        gc_mid=('gc', 'mean'), mean_pred=('pred', 'mean'),
    ).reset_index()
    summary['empirical_prop'] = summary['n1'] / summary['n']
    summary['se'] = np.sqrt(summary['empirical_prop'] * (1 - summary['empirical_prop']) / summary['n'])
    return summary


def bin_training_calibration(df_pred: pd.DataFrame, n_bins: int = 20,
                              stratify_cpg: bool = True) -> pd.DataFrame:
    """
    Per-GC-bin calibration of the DNM model, optionally split by whether the
    site's trinucleotide context is a CpG context (ACG/CCG/GCG/TCG).

    WHY STRATIFY. bin_training_reliability() pools all 32 per-context models.
    That pooling is not a naive composition artifact -- the pooled gap is
    exactly the n-weighted mean of within-context gaps, since composition
    scales the fitted and empirical sides identically -- but it hides that the
    high-GC signal is almost entirely a CpG effect. CpG contexts are 0.9% of
    sites at GC 0.25 and 32% at GC 0.74, and they are the only contexts whose
    models are forbidden GC_content (FT_CORR_MET, applied in
    select_features_for_context) despite it clearing Bonferroni selection in
    all four of them.

    Note also that every per-context model is exactly calibrated IN THE MEAN
    on its own training data (max |mean(pred) - mean(label)| measured at 1e-7)
    -- that is the logistic-regression intercept score equation, not a
    result. So all structure here is within-context, GC-conditional
    miscalibration; none of it is between-context.

    WHY THE INFLATION FACTOR. `gap` (mean_pred - empirical_prop) is on an
    absolute probability scale, where a context with a ~0.5 baseline rate can
    show a large gap for the same relative error as a tiny gap in a context
    with a ~0.09 baseline. But the quantity the pipeline actually applies to
    expected counts is the RATIO r(w) = sigma(b0 + b.z(w)) / sigma(b0). Its
    multiplicative error is

        inflation = r_model / r_true
                  = [mean_pred / sigma(b0)] / [empirical_prop / sigma(b0)]
                  = mean_pred / empirical_prop

    -- sigma(b0) cancels exactly, so `inflation` is scale-free, directly
    comparable across contexts with different baseline rates, and reads
    straight off as "expected counts here are inflated by this factor".
    inflation > 1 inflates expected, depresses observed/expected, and pushes
    Gnocchi's z (and hence its rank) up.

    Returns one row per (group, bin) with: group, bin, n, n1, gc_mid,
    mean_pred, empirical_prop, gap, se (binomial SE of empirical_prop),
    inflation, se_log_inflation (delta-method SE of log inflation, treating
    mean_pred as fixed -- it is a deterministic function of already-fixed
    feature vectors, not resampled).
    """
    if df_pred.empty:
        return pd.DataFrame(columns=['group', 'bin', 'n', 'n1', 'gc_mid', 'mean_pred',
                                      'empirical_prop', 'gap', 'se', 'inflation',
                                      'se_log_inflation'])

    bins = np.linspace(df_pred['gc'].min(), df_pred['gc'].max(), n_bins + 1)
    df = df_pred.assign(bin=np.clip(np.digitize(df_pred['gc'], bins[1:-1]), 0, n_bins - 1))
    if stratify_cpg:
        df = df.assign(group=np.where(df['context'].isin(CPG_CONTEXTS), 'CpG', 'non-CpG'))
    else:
        df = df.assign(group='all')

    summary = df.groupby(['group', 'bin']).agg(
        n=('label', 'size'), n1=('label', 'sum'),
        gc_mid=('gc', 'mean'), mean_pred=('pred', 'mean'),
    ).reset_index()
    summary['empirical_prop'] = summary['n1'] / summary['n']
    summary['gap'] = summary['mean_pred'] - summary['empirical_prop']
    summary['se'] = np.sqrt(summary['empirical_prop'] * (1 - summary['empirical_prop']) / summary['n'])

    # inflation is undefined where no site in the bin is a DNM
    with np.errstate(divide='ignore', invalid='ignore'):
        summary['inflation'] = summary['mean_pred'] / summary['empirical_prop']
        summary['se_log_inflation'] = summary['se'] / summary['empirical_prop']
    summary.loc[summary['empirical_prop'] == 0, ['inflation', 'se_log_inflation']] = np.nan
    return summary


def refit_and_apply(df_dnm1, df_dnm0, contexts, df_ft_genome,
                     genome_expected_percontext_path, output_dir=None, tag=""):
    """
    Full regime-1 pipeline for one training-set size: univariate selection ->
    Bonferroni -> per-context multivariate PCA+logit -> genome-wide r(w)
    apply -> join against per-context expected counts. Returns the
    element_id/possible/expected table.

    If output_dir is given, the intermediate coefficient, selected-feature,
    and per-context rr tables are written there (as the CLI has always done);
    pass None to run purely in memory, e.g. from a notebook.
    """
    import os

    print("refitting univariate feature selection...")
    t0 = time.time()
    df_coef = fit_univariate(df_dnm1, df_dnm0, contexts)
    print(f"  done in {time.time() - t0:.1f}s")
    if output_dir:
        df_coef.to_csv(os.path.join(output_dir, f"coef_univariate.dnm_refit_{tag}.txt"), sep='\t', index=False)

    df_sel = bonferroni_select(df_coef)
    if output_dir:
        sel_path = os.path.join(output_dir, f"selected.dnm_refit_{tag}.txt")
        df_sel.to_csv(sel_path, sep='\t', index=False)
        print(f"  selected {len(df_sel)} (context,feature) rows across "
              f"{df_sel['context'].nunique()}/{len(contexts)} contexts -> {sel_path}")
    report_selected_features(df_sel)

    df_adj_by_context = {}
    n_fit, n_skip = 0, 0
    t0 = time.time()
    for context in contexts:
        ft_sel = select_features_for_context(df_sel, context)
        fitted = fit_multivariate_context(df_dnm1, df_dnm0, context, ft_sel)
        if fitted is None:
            n_skip += 1
            continue
        n_fit += 1
        df_adj_by_context[context] = apply_genome_wide_context(df_ft_genome, context, ft_sel, fitted)
    print(f"  multivariate fit + genome-wide apply: {n_fit}/{len(contexts)} contexts fit "
          f"({n_skip} defaulted to r=1: no significant features or fit failure) in {time.time() - t0:.1f}s")

    if output_dir and df_adj_by_context:
        rr_path = os.path.join(output_dir, f"rr_by_context.dnm_refit_{tag}.txt")
        df_rr_long = pd.concat(
            [df.assign(context=context) for context, df in df_adj_by_context.items()], ignore_index=True)
        df_rr_long.to_csv(rr_path, sep='\t', index=False)
        print(f"  wrote per-context rr -> {rr_path}")

    print("joining against genome-wide per-context expected counts (duckdb, streamed from disk)...")
    t0 = time.time()
    df_out = combine_and_predict(df_adj_by_context, contexts, genome_expected_percontext_path)
    print(f"  done in {time.time() - t0:.1f}s ({len(df_out):,} windows)")
    return df_out


def validate_against_published(df_coef: pd.DataFrame, published_path: str):
    df_pub = pd.read_csv(published_path, sep='\t')
    merged = df_coef.merge(df_pub, on=['context', 'window', 'feature'], suffixes=('_new', '_pub'), how='outer')
    n_total = len(merged)
    n_both_nan = (merged['coef_new'].isna() & merged['coef_pub'].isna()).sum()
    n_nan_mismatch = (merged['coef_new'].isna() != merged['coef_pub'].isna()).sum()

    ok = merged.dropna(subset=['coef_new', 'coef_pub'])
    for col in ['coef', 'se', 'pval']:
        ok = ok.assign(**{f'{col}_diff': (ok[f'{col}_new'] - ok[f'{col}_pub']).abs()})

    print(f"\nrows: {n_total}  both-NaN (fit failed in both): {n_both_nan}  "
          f"NaN-mismatch (fit failed in only one): {n_nan_mismatch}  comparable: {len(ok)}")
    if len(ok):
        print(f"max |coef diff| = {ok['coef_diff'].max():.3e}   "
              f"max |se diff|   = {ok['se_diff'].max():.3e}   "
              f"max |pval diff| = {ok['pval_diff'].max():.3e}")
        print(f"coef matches to <1e-6: {(ok['coef_diff'] < 1e-6).sum()}/{len(ok)}   "
              f"<1e-3: {(ok['coef_diff'] < 1e-3).sum()}/{len(ok)}")
    if n_nan_mismatch:
        print("rows where fit succeeded in exactly one of new/published:")
        print(merged[merged['coef_new'].isna() != merged['coef_pub'].isna()]
              [['context', 'window', 'feature', 'coef_new', 'coef_pub']].to_string(index=False))
    return merged
