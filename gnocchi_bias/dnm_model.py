"""
The DNM training set and Gnocchi's per-context regional-feature mutation model.

Extracted verbatim (2026-08-04) from dnm_training_size/
run_dnm_training_experiment.py, so that all three consumers can import the
same fitting code:

  fig5/              refits on three training populations (refit_and_apply) and
                     evaluates each on its own sites (predict_training_set), for
                     panels B, D and E.
  dnm_training_size/ the same refit under a randomly shrunk training set.
  preconditions/     fit_univariate + bonferroni_select, which validate.py diffs
                     against Chen et al.'s published fitted parameters. The comparison
                     itself lives THERE, not here: it is a test, and the module under
                     test should not ship its own grader. (Contrast
                     windows.check_z_against_published, which is a runtime guard on the
                     live fig5 path and so belongs beside the code it guards.)

Provenance, unchanged from the original and load-bearing for the manuscript:
fit_univariate + bonferroni_select reimplement the published feature-selection
step (analyze_individual_feature_effects.py) and validate against the published
coefficient table: every one of the 1,664 rows agrees to within 0.021 of its own
published standard error (max |coef diff| 2.6e-4 in absolute terms, against a
median |coef| of 0.027 -- quote the SE-normalized figure, since the absolute one
cannot be read without knowing that scale), and the selection those coefficients
drive reproduces Chen et al.'s own misc/genomic_features13_sel.txt exactly, 239
rows with none on either side alone. fit_multivariate_context is the
one step with NO published source anywhere -- reconstructed from the apply-side
code in run_nc_constraint_gnomad_v31_main.py:231-249 -- and is validated
end-to-end instead: at frac=1.0 it reproduces the published Gnocchi `expected`
column with Pearson r = 1.0 over 1,984,900 windows. See CLAUDE.md.

This module deliberately does NOT call matplotlib.use() or import pyplot: it is
the model, not the plots, and it must import cleanly inside a notebook.
"""
import time
from functools import reduce
from typing import cast

import duckdb
import numpy as np
import pandas as pd
import scipy.stats
import statsmodels.api as sm
from sklearn.decomposition import IncrementalPCA

from .windows import download

# Chen et al.'s step-2 training set, as published. Four files: the mutated (dnm1) and
# non-mutated (dnm0) SITE tables, each paired with a FEATURE table joined on locus.
# Sizes/row counts measured from the bucket; none of the four carries a GCS customTime,
# so unlike the expected-count tables there is no creation-date provenance to check.
#
# Paper's Methods, "Adjustment of the effects of regional genomic features": 413,304
# unique DNMs "compiled from two large-scale family-based whole-genome sequencing
# studies" -- deCODE (Halldorsson et al., recombination/sequence-level genetic map) and
# PsychENCODE (An et al., de novo risk score in autism), hence the filename -- against
# "an exclusive set of 4,104,879 genomic sites (~10x the DNMs) randomly drew from the
# genome" as the non-mutated background.
#
#   dnm1_sites     24.7 MB,   410,542 rows. locus, alleles, context, ref, alt,
#                  methyl_level, sid, 3mer. No chrX rows. `sid` is
#                  "{context}-{methyl_level}"; `3mer` is the step-1 context-only
#                  per-site mutation probability, i.e. fitted_po summed over the three
#                  alt alleles for that (context, methyl_level) -- verified to 15 digits
#                  against fig_tables/mutation_rate_by_context_methyl.txt
#                  (CCC-0: 0.2625636943172292).
#   dnm0_sites    190.2 MB, 4,107,802 rows. Same columns minus alleles/ref/alt, which do
#                  not apply to an unmutated site. Ratio to dnm1: 10.006:1.
#   *_features    206.4 MB / 2.05 GB, 413,273 / 4,105,163 rows, 53 columns = key +
#                  13 features x 4 scales (1k/10k/100k/1M), the panel and method of
#                  generate_genomic_features.sh (UCSC tools, bedtools, CrossMap) but
#                  centred on each training site rather than on genome tiles. NOTE the
#                  key column is named `element_id` yet holds a LOCUS ("chr10:100003712"),
#                  not a 1 kb tile id -- which is why load_training_data renames it.
#
# BOTH published counts are reproduced by these files exactly, though by different
# tables of the pair -- check against the right one before concluding an N is wrong:
#
#   413,304 DNMs        = the dnm1 FEATURE table's 413,273 rows + the 31 loci that carry
#                         two DNMs each (different alleles, so they collapse to one row
#                         in a locus-keyed feature table). 413,273 + 31 = 413,304.
#   4,104,879 background = the dnm0 SITE table's 4,107,802 rows minus its 2,924 chrX
#                         rows, which load_training_data drops below (upstream line 18):
#                         4,104,878, i.e. the published figure to within one row.
#
# The other table of each pair does not match, for reasons that are not defects: the
# dnm1 site table holds 410,542 rows, 2,762 fewer loci than the feature table (all
# autosomal, so not the chrX filter -- presumably sites where trinucleotide context or
# methylation could not be assigned), and since load_training_data left-joins features
# ONTO sites, 410,542 is the effective training N. Every one of them finds a feature
# row; nothing is dropped by the join itself.
TRAINING_FILES = {
    "dnm1_sites": "genomic_features/DNM_decode_psychencode_site_context.mutation_rate.txt",
    "dnm1_features": "genomic_features/genomic_features13_dnm1_flnk_1k-1M.txt",
    "dnm0_sites": "genomic_features/context_prefiltered_nonmutated-dnm_sites10xdnm.mutation_rate.txt",
    "dnm0_features": "genomic_features/genomic_features13_dnm0_10x_flnk_1k-1M.txt",
}
PUBLISHED_COEF_FILE = "genomic_features/dnm01_10x_ft_logit_regularized_coef_z_3mer_context_flnk_1k-1M.txt"
# Chen et al.'s own Bonferroni-surviving rows -- their selection output, not our
# recomputation of it. 239 rows of (context, feature, window, coef, se, pval).
PUBLISHED_SEL_FILE = "misc/genomic_features13_sel.txt"
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
    """
    Regime 1: shrink both dnm0 and dnm1, independently, at the same rate -- so the
    case-control ratio (dnm0:dnm1 ~ 10:1) is preserved and only the training-set SIZE
    changes. Named for the numbering in okf/dnm-training-set-experiment/training-data.md,
    deleted at 2a07dc9; the sentence above is the whole of what that file said about it.
    """
    return (df_dnm1.sample(frac=frac, random_state=seed),
            df_dnm0.sample(frac=frac, random_state=seed))


def locus_to_element_id(locus: pd.Series) -> pd.Series:
    """
    1-based "chr1:137548" -> the 0-based 1kb tile id "chr1-137000-138000" that the
    window table, the features file and the expected-count exports are all keyed by.
    Mirrors the SQL in fig3/empirical_r.py:78 (ELEMENT_ID_FROM_LOCUS, removed with
    fig3/ at c913d87 -- readable at c913d87^), including the -1 for the 1-based to
    0-based conversion. That -1 moves only sites at position = 0 mod 1000: position
    1000 lands in chr1-0-1000, not chr1-1000-2000.
    """
    chrom = locus.str.split(':').str[0]
    pos = locus.str.split(':').str[1].astype('int64')
    start = ((pos - 1) // 1000) * 1000
    return chrom + '-' + start.astype(str) + '-' + (start + 1000).astype(str)


def restrict_to_analyzed_windows(df_dnm1: pd.DataFrame, df_dnm0: pd.DataFrame,
                                  element_ids) -> tuple:
    """
    Keep only training sites whose containing 1kb window is in the analyzed set --
    i.e. make the training population match the population the model is applied to
    and scored on.

    WHY. The published models are fit on the whole genome but r(w) is applied to,
    and judged on, noncoding pass_qc autosome/PAR windows. Measured directly
    (fig5/data.py), those two populations agree in the GC bulk
    and come apart in the GC-rich tail: the fraction of training sites inside the
    analyzed set falls from 0.82 at GC 0.37 to 0.28 by GC 0.68, as GC-rich sequence
    turns coding or fails gnomAD's window QC. (CLAUDE.md and the README quote the same
    curve as 0.84 -> 0.28, at its sparser first and last plotted bins. One curve, two
    endpoint pairs; both read off fig5/output/dnm_rate_by_stratum.20bins.<fp>.parquet -- the
    suffix fingerprints the GC edges and the window population it was built over -- whose
    per-stratum site counts are the composition.) Restricting here is the
    intervention that tests whether that mismatch is what makes the fitted non-CpG
    adjustment climb with GC.

    `element_ids` is any iterable of analyzed element_ids -- in practice
    windows.build_window_table(...)["element_id"]. Sites whose window is absent from
    the constraint table (it failed gnomAD's variant-call QC) are dropped too, since they
    are absent from that set.

    Returns the filtered (df_dnm1, df_dnm0) and prints the retained fractions, which
    are worth reading: they are not equal, because DNMs and background sites are not
    distributed the same way across the excluded territory.
    """
    keep = set(element_ids)

    def _filter(df, name):
        eid = locus_to_element_id(df['locus'])
        mask = eid.isin(keep)
        print(f"  {name}: {int(mask.sum()):,} / {len(df):,} sites retained "
              f"({mask.mean():.1%})")
        return df[mask.values].copy()

    print("restricting training set to the analyzed window population:")
    return _filter(df_dnm1, "dnm1 (DNMs)"), _filter(df_dnm0, "dnm0 (background)")


def count_in_analyzed_windows(df_dnm1: pd.DataFrame, df_dnm0: pd.DataFrame,
                               element_ids) -> tuple[int, int]:
    """
    How many sites restrict_to_analyzed_windows would keep, without doing the filter.

    Exists so the size-matched control can draw exactly that many sites uniformly at
    random from the WHOLE genome -- separating "less training data" from "training
    data drawn from the population the model is applied to", which is the obvious
    alternative explanation for anything the restriction improves.
    """
    keep = set(element_ids)
    n1 = int(locus_to_element_id(df_dnm1['locus']).isin(keep).sum())
    n0 = int(locus_to_element_id(df_dnm0['locus']).isin(keep).sum())
    return n1, n0


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
                # The cast is for the type checker, not the reader: pandas-stubs' first
                # pd.concat overload takes `Iterable[None]` and returns Never, and a list
                # whose element type came out of a chained df[...][...] as Unknown matches
                # it -- which makes everything below here dead flow, so an editor greys out
                # the rest of the loop. Naming the element type picks the right overload.
                # A plain annotation would do it, but errors under the stubless pandas the
                # repo actually installs, where the same expression is DataFrame | Series.
                # Same fix in predict_training_set.
                df_01 = pd.concat(cast("list[pd.DataFrame]", [df_1, df_0]))

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
    The step with NO published source anywhere: fit a multivariate, PCA-whitened
    logistic regression for one context on its selected features. Both preprocessing
    choices are reconstructions, so the reasoning is recorded here -- the notes it
    came from (okf/dnm-training-set-experiment/{missing-code,open-questions}.md) were
    deleted at 2a07dc9:

      - std with ddof=0, because fit_univariate above standardizes with
        scipy.stats.zscore, whose default is ddof=0. The two stages must agree.
      - IncrementalPCA with default (i.e. all) components, so the transform is a
        pure rotation that discards nothing. Verified for context AAA; the bucket
        ships a fitted .pca.pkl per context if that check needs redoing.

    Both are fit fresh on this context's own (possibly subsampled) training pool --
    NOT the published values.

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
        # cast as in fit_univariate
        df01 = pd.concat(cast("list[pd.DataFrame]", [df_1, df_0]), ignore_index=True)
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


def refit_and_apply(df_dnm1, df_dnm0, contexts, df_ft_genome,
                     genome_expected_percontext_path, output_dir=None, tag=""):
    """
    Full regime-1 pipeline (see subsample_regime1) for one training-set size:
    univariate selection ->
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
