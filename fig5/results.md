# Results paragraph — insert after L381 of `constraint_paper.pdf`

Follows the paragraph ending "…primarily responsible for Gnocchi's GC-content-dependent
bias." (L371–381), which introduces Figure 5A. Every number is quoted from a printed cell
of `fig5/fig5.ipynb`; see *Provenance* below.

---

Having localized the bias to the DNM submodel, we asked which part of that submodel carries it and whether it could be removed. The submodel multiplies each window's context-only expected count by a factor fit separately for each trinucleotide context; aggregated over the windows of a GC bin, the multiplier Gnocchi actually applies rises from 0.95 to 1.44 across the GC range (Figure 5B). That rise is entirely attributable to non-CpG contexts, which climb from 0.95 to 1.78, while the CpG contexts stay within 2% of one — and holding the non-CpG factor at one leaves a curve flat to within 0.6%, even though CpG contexts carry 43% of the expected counts in the highest GC bin. A CpG factor of one is the correct behavior rather than a failure, because CpG mutability is dominated by methylation, which Chen et al. model in the context-only submodel (Supporting Text S1). We next asked why the fitted non-CpG factor climbs with GC content. The DNM submodel is fit on de novo mutations and matched background sites drawn from the whole genome, whereas Gnocchi is scored on noncoding windows that pass gnomAD's variant-call QC, and those two populations diverge as GC content increases: the fraction of background training sites that lie in scored windows falls from 0.85 to 0.29 (Figure 5C, upper). The excluded sequence is not merely absent but different — in windows failing QC the DNM rate is 1.55 times that of scored noncoding windows in the bulk of the GC distribution and 4.06 times at GC content 0.61, whereas coding windows track scored noncoding windows to within about 10% across the whole range (Figure 5C, lower). Restricting the training set to the scored population shrank the empirical GC dependence of the DNM probability from 2.45-fold and non-monotonic to 1.57-fold and monotonic, which the regional-feature model could then track instead of missing in opposite directions by 26% and 29% (Figure 5D). Refitting the DNM submodel on that population and rescoring the genome reduced Gnocchi's GC bias from 0.212 to 0.046, below the 0.093 of the context-only model (Figure 5E). Two controls, run through the same code, exclude the obvious alternatives: refitting on the full training set reproduced published Gnocchi (0.212), and refitting on a size-matched random subsample of the whole genome did too (0.210), so the improvement is attributable to which sites the submodel is fit on rather than to our reimplementation or to the amount of data.

---

## Provenance of every number

All from `fig5/fig5.ipynb`, "Numbers for the caption" cell unless noted. Bias is the mean
over GC bins of |mean standardized rank − 0.5|, the statistic of Figure 2A, computed over
bins holding ≥ 100 windows.

| Claim | Value | Cell |
|---|---|---|
| applied multiplier across GC | 0.95 → 1.44 | panel B table, `r_eff` |
| non-CpG multiplier | 0.95 → 1.78 | `r_non` spans 0.951–1.785 |
| CpG multiplier | 0.984–1.004 ("within 2% of one") | `r_cpg` |
| counterfactual, non-CpG factor held at 1 | 0.994–1.001 ("within 0.6%") | `r_counterfactual` |
| CpG share of expected counts | 0.025 → 0.426 | `pi_cpg` |
| training sites in scored windows | 0.85 at GC 0.23 → 0.29 at GC 0.72 | panel C |
| QC-fail DNM rate vs scored noncoding | 1.55× in bulk, 4.06× at GC 0.61 | `diagnostics.stratum_ratios` |
| coding DNM rate vs scored noncoding | 0.896–0.995 ("within about 10%") | same |
| empirical DNM-rate GC dependence | 2.45× (original) → 1.57× (scored) | panel D |
| fitted-vs-empirical miss, original set | 26% and 29%, opposite directions | panel D prose |
| bias, context-only / published / retrained | 0.093 / 0.212 / 0.046 | panel E |
| controls: full refit / size-matched | 0.212 / 0.210 | panel E |

## Two things the paragraph deliberately does not claim

- **Panel D is in-sample** — the retrained model is scored against the labels of the sites
  it was fit on. The paragraph therefore uses panel D only for the *shape* of the
  empirical curve and the fit's ability to track it, and rests the bias claim on panel E,
  which is measured on gnomAD polymorphism counts the DNM submodel never sees.
- **This is not a corrected Gnocchi.** The refit demonstrates attribution; it does not
  propose a score, because the scored population would have to be fixed before fitting and
  that choice is not obvious.

## If the paragraph needs to be split

The natural seam is before "We next asked why the fitted non-CpG factor climbs with GC
content": Figure 5B is *what the submodel applies*, Figures 5C–E are *why it is wrong and
what fixes it*.
