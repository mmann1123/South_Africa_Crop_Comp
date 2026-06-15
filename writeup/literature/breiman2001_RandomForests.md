# Random Forests

**Citation:** Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32. DOI: 10.1023/A:1010933404324 (verified via Crossref). (PDF reviewed is the author's technical-report/preprint version, Statistics Dept., UC Berkeley, January 2001; no DOI printed on the preprint, canonical journal DOI used.)

## Objectives

This is the **foundational theory-and-methods paper** introducing Random Forests
(RF) — an ensemble of tree classifiers in which each tree is grown on a bootstrap
sample using a *random selection of features* at each node, and the forest classifies
by majority vote. The paper's aims are (1) to define random forests formally and
prove they do not overfit as trees are added (the generalization error converges
almost surely to a limit); (2) to derive an upper bound on generalization error in
terms of individual-tree **strength** and inter-tree **correlation**; (3) to show
empirically that random-feature forests match AdaBoost's accuracy while being more
robust to noise and outliers; and (4) to develop **out-of-bag (OOB) internal
estimates** of error, strength, correlation, and variable importance. Results extend
to regression.

Per our rules, the evaluation-protocol question is recast for a theory/methods paper:
the focus below is on the conceptual/inductive-bias implications for our work.

## Methods

**Definition.** A random forest is a collection of tree classifiers
{h(x, Θ_k), k = 1, 2, ...} where the Θ_k are i.i.d. random vectors and each tree casts
a unit vote for the most popular class at input x.

**Theory.** Using the Strong Law of Large Numbers, Breiman proves the generalization
error PE* converges as the number of trees grows, so forests *do not overfit* with
more trees (Theorem 1.2/2.1). He derives the bound **PE* <= ρ̄(1 − s²)/s²**, where
s is the *strength* (expected margin) of the individual trees and ρ̄ is the *mean
correlation* between trees' raw margin functions. The c/s² ratio (correlation over
squared strength) is the governing quantity: lower correlation and higher strength
both reduce the error bound. This formalizes the bias–variance intuition that an
ensemble of *strong but decorrelated* learners generalizes well — random feature
selection injects the decorrelation.

**Forest-RI.** The practical algorithm selects, at each node, a small random group of
F input variables to split on, grows unpruned CART trees, and votes. Two F values are
tried: F = 1 (single random feature) and F = int(log₂M + 1). Accuracy is shown to be
**insensitive to F** — usually one or two features per split gives near-optimal
results.

**Out-of-bag estimates.** Because each bootstrap leaves out ~1/3 of instances, the
left-out (out-of-bag) predictions yield *unbiased* internal estimates of
generalization error, strength, correlation, and variable importance — removing the
need for a set-aside test set. Breiman notes OOB estimates are unbiased, unlike
cross-validation where bias is present but of unknown extent.

**Evaluation protocol (empirical sections).** Benchmarking is on UCI and synthetic
tabular datasets (Table 1). For the 13 small sets, accuracy is estimated by repeated
**random 10%-holdout, averaged over 100 runs**; the 3 larger sets and 4 synthetic
sets use fixed train/test splits. This is **standard random-holdout/repeated-split
tabular evaluation** — appropriate for the i.i.d. UCI benchmarks but carrying no
spatial or temporal structure. The contribution that matters for our spatial work is
not these benchmark splits but the **OOB methodology** and the **strength/correlation
generalization theory**.

## Key Findings

- **Forests do not overfit with more trees**; PE* converges to a limiting value
  (Theorem 1.2).
- **Generalization is governed by strength and correlation**: PE* <= ρ̄(1 − s²)/s².
  Injected randomness should *minimize inter-tree correlation while preserving tree
  strength* — the core design principle.
- **Random-feature forests rival AdaBoost** in accuracy and are *more robust to noise
  and outliers*, faster than bagging/boosting, easily parallelized, and accuracy is
  insensitive to the number of features per split (F = 1 often near-optimal).
- **Out-of-bag estimates** give unbiased, free internal monitoring of error, strength,
  correlation, and variable importance — no separate test set required.
- The method scales to high-dimensional problems (the paper notes a synthetic case
  with 1,000 input variables reaching near-Bayes accuracy).

## Relevance to Our Crop-Classification Study

Random Forests is the **canonical ancestor of the sparse, feature-selecting,
tree-like classifiers our manuscript finds transfer best** out-of-sample, and several
of its ideas underpin our methodology:

- **Inductive bias = decorrelated feature-selecting trees.** Our central empirical
  result is that gradient-boosted trees and TabNet (attentive sparsemax masks)
  transfer to a disjoint tile while dense temporal/patch nets collapse. Breiman's
  strength/correlation theory is the conceptual root of *why* tree ensembles
  generalize from limited, structured data: per-node random feature selection
  produces strong-but-decorrelated learners. This is the theoretical scaffold for our
  inductive-bias argument, even though boosting (our GBT) differs from RF's
  randomization.
- **Built-in feature selection.** RF's node-level random subspace and its
  variable-importance machinery embody the "select a sparse subset of informative
  features" behavior we argue aids spatial transfer — the same principle behind
  TabNet's sparse masks and our feature-selection preprocessing.
- **OOB vs. spatial holdout — an instructive contrast.** RF's OOB error is unbiased
  *under the i.i.d. assumption*. But satellite pixels/fields are **not** i.i.d. —
  spatial autocorrelation means OOB (and any random k-fold) leaks neighboring
  information and overstates transfer, exactly the failure our spatially disjoint
  holdout exposes. Breiman's OOB is a perfect foil: it is rigorous for tabular i.i.d.
  data and misleading for spatially structured remote-sensing data. We can cite RF to
  make precisely this point — random/internal estimates are not spatial-transfer
  estimates.
- **Robustness to noise and high dimensionality** rationalizes RF/GBT as strong
  classical baselines for our many-feature xr_fresh design, and their robustness is
  part of why they degrade gracefully out-of-sample relative to dense nets.

Cite this as the theoretical/methodological foundation for the tree-ensemble family
and for the strength–correlation account of generalization, and as the contrast case
showing why internal (OOB/random) error is not a substitute for spatial-holdout
evaluation in autocorrelated data.

## Evaluation Caveats

- **Theory/methods paper on i.i.d. tabular benchmarks.** Its empirical accuracies are
  on UCI/synthetic data with random holdout — no spatial or temporal structure, no
  crop or remote-sensing data — so they are not comparable to our macro-F1 and serve
  only to validate the algorithm in the abstract.
- **OOB unbiasedness assumes i.i.d.** The paper's claim that OOB removes the need for
  a test set holds under independence; for spatially autocorrelated satellite data it
  does **not**, and treating OOB/random-CV as a transfer estimate is exactly the
  misranking trap our study documents. This caveat is the load-bearing one for us.
- **Class imbalance not addressed.** The 2001 paper does not treat class-imbalanced
  minority recall (later RF variants add class weighting / balanced sampling); it
  reports error rates, not macro-F1 or per-class metrics, so it offers no guidance on
  the lucerne/medics-dominated skew central to our problem.
- **Silences.** No spatial transfer, no cross-domain robustness, no temporal-sequence
  modeling, and no treatment of field-level aggregation — all outside its 2001 scope.
  Its enduring contribution to us is conceptual (sparse decorrelated trees, variable
  importance, the strength/correlation bound), not benchmark numbers.
