---
type: Hypothesis
title: Sparse DNM training data causes Gnocchi's local GC-content bias
description: >
  The specific claim from chen_formula.tex, "Predictions of the hypothesis",
  that this experiment is designed to test.
resource: /Users/petermchale/gnomad_nc_constraint/chen_formula/chen_formula.tex
tags: [gnocchi, gc-bias, dnm, hypothesis]
timestamp: 2026-07-20T00:00:00Z
---

# Hypothesis

Source: `chen_formula/chen_formula.tex`, section "Predictions of the hypothesis"
(quoted directly, red text in the original marks claims the rebuttal asserts
were already empirically shown, with no code or data published for them):

1. **Very small DNM training set** → the regional-feature model `r_c(x)` can't
   sample the tails of `x` (e.g. extreme GC content) at all, so it collapses
   toward its value at the mean, `r_c(x̄) = 1` by construction — Gnocchi
   reduces to the context-only model and should inherit *its* bias, not more.
   > "We performed this experiment and found not only that bias in Gnocchi was
   > reduced, but also that it approached the levels of bias we saw in other
   > context-only constraint metrics."
2. **Growing training set, more extreme `x` sampled** → `r_c(x)` starts fitting
   the true function near `x̄` better, "but this may come at the expense of
   deviating from the true function away from `x̄`" — this is what the tex
   claims is the actual mechanism behind Gnocchi's real, large bias.
3. **Densifying only the background (non-mutated) sites, without adding more
   real DNMs** → extreme `x` gets covered without adding more noisy positive
   (mutated) examples, so tail bias should shrink again.
   > "We approximated this regime by increasing the number of background
   > sites (only) in the DNM training set, retrained `r_c(x)`, and observed
   > that bias below the level we reported in our paper."

## What this experiment does

Attempts to reproduce claim (1) and/or claim (3) directly on real data in this
repo, since neither the code nor the intermediate results behind the red text
above are published anywhere the authors have access to. See
[pipeline](pipeline.md) for how. Claim (2) is the *default/baseline* case (the
published training set as-is) already partially addressed by the existing
step-1-vs-step-2 analysis documented in the root `CLAUDE.md` (section "The
analysis that does work").
