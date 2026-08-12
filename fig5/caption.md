# Figure 5 caption

---

**Figure 5. Gnocchi's GC-dependent bias is introduced by its DNM submodel, and arises because that submodel is fit on a population of sites that differs from the one Gnocchi is scored on.** Unless stated otherwise, all panels use the 1,843,559 autosomal 1 kb windows that are noncoding and pass gnomAD's variant-call QC — the windows Gnocchi is scored on — binned into 20 equal-width GC-content bins, and drop bins holding fewer than 100 windows (fewer than 500 training sites in **C** and **D**). **(A)** Mean standardized rank of each constraint metric per GC bin; the rank is uniform on (0,1) by construction, so an unbiased metric sits at 0.5 (dashed line) in every bin, and the vertical line marks the mean GC content of the window set. Gnocchi as published (orange) is compared with the same pipeline with its DNM submodel removed, i.e. expected counts from sequence context and methylation alone (blue), and with Depletion Rank (green), which is computed on its own windows and therefore ranked within its own set rather than joined. Both Gnocchi curves are computed on identical windows and ranked after a joint filter on the constraint score, so they differ only in whether the DNM submodel was applied. **(B)** The multiplier the DNM submodel applies to a GC bin's expected counts, R(g) = ΣE₂/ΣE₁ summed over the bin's windows, decomposed exactly as R_eff = Π·R_CpG + (1 − Π)·R_non, where Π is the CpG contexts' share of the bin's context-only expected counts. The dashed grey curve is the counterfactual in which the non-CpG contexts receive no adjustment, Π·R_CpG + (1 − Π), i.e. what Gnocchi would apply if it adjusted CpG contexts alone; its flatness places the entire GC dependence of the applied multiplier in the non-CpG contexts. **(C)** Upper: composition of the non-CpG background DNM-training sites per GC bin, by the window each site falls in — QC-pass noncoding (the scored population), QC-pass coding, or QC-fail (absent from the published constraint table; a mixture of coding and noncoding windows, in the same proportion as the QC-pass ones). Lower: each of the other two strata's empirical non-CpG DNM probability relative to the QC-pass noncoding stratum's, per GC bin, with delta-method binomial error bars on the log ratio; a ratio of one would mean the excluded sites are exchangeable with the scored ones as far as mutation rate is concerned. **(D)** Fitted and empirical DNM probability per GC bin over the non-CpG training sites of three training populations: the published training set, the same set restricted to the scored population, and a control drawn at random from the whole genome with the same number of sites as that restriction. Each curve is divided by its own site-weighted mean, because the three populations have different case-control ratios; only shape is comparable across populations, whereas fitted versus empirical *within* a population is a reliability diagram. **(E)** Panel A's statistic with Gnocchi rescored after refitting the DNM submodel on the scored population (violet). Nothing else in the pipeline differs. The two controls are reported in the text rather than plotted, being indistinguishable from published Gnocchi: the same code refit on the full training set, and on a size-matched random subsample.

---

## Build notes — not part of the caption

- **The Depletion Rank curve in (A) is not in the committed figure.** It needs
  `DEPLETION_RANK_BED` in `fig5/config.py`, which points at the constraint-tools
  `CONSTRAINT_TOOLS_DATA` path and is unavailable in this repository; the panel builds
  with two curves and prints a notice. Drop the clause naming it if the figure ships that
  way. `depletion_rank.py` has never been run against the real file.
- **"Noncoding" here is `coding_prop = 0` + `pass_qc` + autosome/PAR, without the
  GeneHancer enhancer exclusion** used elsewhere in the paper, because GeneHancer is
  licensed and `GENEHANCER_BED` is unset. The same definition is used in every panel,
  including the population the retrained submodel is fit on, so the comparison is internally
  consistent; but the window count (1,843,559) is larger than the paper's neutral set.
- **A supporting figure** (`output/fig5_supp_cpg.pdf`) backs the caption's claim that
  R_CpG ≈ 1 is correct: the methylation effect step 1 already absorbs (3.0–4.3× within one
  trinucleotide context), the hypomethylated character of high-GC CpGs (90–100% above GC
  0.70), and the resulting 2.7-fold fall in their DNM rate. Cite it wherever that claim is
  made in the text.
- Panel letters here match the files `output/fig5{A..E}.pdf`, each a standalone vector
  page for assembly.
