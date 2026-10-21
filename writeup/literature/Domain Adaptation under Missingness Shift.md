# Domain Adaptation under Missingness Shift

**Citation:** Helen Zhou, Sivaraman Balakrishnan, and Zachary C. Lipton (2023). *Proceedings of
the 26th International Conference on Artificial Intelligence and Statistics (AISTATS 2023)*,
Valencia, Spain, PMLR volume 206. DOI: not found in PDF. Crossref title searches
(`query.title=Domain Adaptation under Missingness Shift` and
`query.bibliographic=Domain Adaptation under Missingness Shift Zhou Balakrishnan Lipton`) on
2026-10-21 returned no matching record (top hits were other Lipton-group papers on label shift and
unrelated domain-adaptation work), consistent with PMLR proceedings not being DOI-registered. Cite
via PMLR. The PDF carries no arXiv identifier.

## Objectives

A theory paper from the statistical machine-learning community. It introduces *missingness shift*:
source and target domains share the same underlying distribution `P(X, Y)` but differ in the
mechanism by which covariates go missing. The motivating case is clinical: a hospital's vaccination
field read 40.5 percent positive when the true regional rate was 75 to 90 percent, because
vaccinations given elsewhere were never recorded; a later change to the intake form would raise the
recorded rate overnight with no change in patient health. The paper asks how to train a predictor
on labelled source data and unlabelled target data when the only thing that changed is how often
each feature is recorded.

Objectives:

1. Define Domain Adaptation under Missingness Shift (DAMS) and the special case of
   *underreporting completely at random* (UCAR), where each feature is independently zeroed with an
   unknown, constant, domain-specific probability and *no missingness indicator is observed* (a
   recorded zero is indistinguishable from a true zero).
2. Show that when indicators *are* observed, missingness shift reduces to covariate shift
   (Proposition 1).
3. Show that without indicators the optimal source predictor can be arbitrarily worse on the target
   than predicting the label mean (Remarks 1 and 2), so adaptation is required.
4. Show that although the missingness rates themselves are not identifiable, the *relative* rates
   are (Remark 5), and that the labelled target distribution is identifiable from the labelled
   source distribution given those relative rates (Theorem 5.2).
5. Derive a closed-form adjusted linear predictor (Proposition 2, Algorithm 2) and a non-parametric
   adjustment that re-masks source data to match the target's missingness (Algorithm 1).
6. Validate on synthetic, semi-synthetic (UCI adult, bank, thyroid), and real (eICU, two
   hospitals) data against oracle, unadjusted source, and MissForest-imputed baselines, for linear
   regression, XGBoost, and MLP model classes.

## Methods

### Problem definition (Section 3)

Observed covariates `X_tilde = X * xi`, `xi in {0,1}^d`. Missingness shift: `P^s(X, Y) = P^t(X, Y)`
but `P^s(xi | .) != P^t(xi | .)`. UCAR: `xi^s ~ Bernoulli(1 - m^s)`, `xi^t ~ Bernoulli(1 - m^t)`
independently per feature, with `m^s, m^t` unknown and `xi` unobserved. Labels are never missing.
The target quantity is the optimal predictor `E[Y | X_tilde^t]` under squared error.

### Theory (Sections 4 to 6)

- Example 1 (redundant features, `X1 = X2 = Z`, `m^s = [1-eps, eps]`, `m^t = [eps, 1-eps]`): as
  `eps -> 0` the source predictor weights the feature that is absent in the target and its target
  risk tends to `Var(Y)`.
- Example 2 (confounded features): for any `tau > 1` there exist parameters such that the source
  predictor's target risk exceeds `tau Var(Y)`.
- Proposition 1: with indicators appended as features, `P^s(Y | X_tilde, xi) = P^t(Y | X_tilde, xi)`,
  so standard covariate-shift tools apply (and if supports overlap, often no adaptation is needed).
- Theorem 4.1: for generalized linear models, UCAR on inputs rescaled by `1 / (1 - m)` is
  equivalent to an L2-type penalty on the clean-data loss, i.e. underreporting acts like dropout
  regularization.
- Lemma 5.1 and Remark 4: the clean distribution is recoverable from the corrupted one if `m` is
  known, but `m` is not identifiable from one domain (a Bernoulli(0.5) feature with 50 percent
  underreporting and a Bernoulli(0.25) feature fully reported are indistinguishable).
- Remark 5: `(1 - m^t) / (1 - m^s) = P^t(X_tilde != 0) / P^s(X_tilde != 0)`, estimable from the
  proportion of non-zeros in each domain; define relative missingness `r^{s->t}`.
- Theorem 5.2: `p_tilde^t` is identifiable from `p_tilde^s` and `r^{s->t}` alone.
- Proposition 2: optimal linear target coefficients
  `beta_t = E[X_tilde^t' X_tilde^t]^{-1} (r^{s->t} * E[X_tilde^s' Y^s])`, with the target second
  moment estimable from unlabelled target data or from source data via Proposition 3.
- Algorithm 1 (non-parametric, any model class): estimate `r^{s->t}`, apply an extra Bernoulli mask
  with those rates to the *source* data so it is distributed like the target, then fit any model.
  Proper only when `m^s <= m^t` element-wise (one can only add missingness); otherwise an
  "improper" clipped version is used.
- Theorem 6.1: Hoeffding bound on the estimation error of `r^{s->t}`.

**Evaluation protocol:** Synthetic: 10,000 samples from two generators, split 4:1:4:1 into source
train / source test / target train / target test, with `eps` swept from 0.05 to 0.95 (20 runs
each). Semi-synthetic: UCI covariates with synthetic linear labels, missingness sampled either with
`m^s <= m^t` (proper regime) or both uniform on [0, 0.9] (general regime), target metric
MSE / Var(Y) averaged over missingness draws, bootstrap CIs in the appendix. Real: eICU 48-hour
mortality, FIDDLE binary features, Hospital 1 (n = 3,006) vs Hospital 2 (n = 2,663) in both
directions, MSE / AUROC / AUPRC with CIs. Every experiment is, by construction, a *mismatched*
train/test missingness regime: that is the object of study. Imputation is compared at inference in
one form, MissForest applied to both domains treating all zeros as missing. Five arms per model
class: Oracle (trained on labelled target), Source (no adjustment), Imputed, Closed-form
adjustment (linear only), Non-parametric adjustment.

## Key Findings

### The cost of ignoring missingness shift (Figure 2a, Table 1)

Synthetic, target MSE / Var(Y), `eps = 0.1`:

| Model | Redundant: Oracle / Source / Closed-form / Non-param | Confounded: Oracle / Source / Closed-form / Non-param |
|---|---|---|
| Linear | 0.178 / 1.259 / 0.186 / 0.473 | 0.206 / 1.103 / 0.209 / 0.492 |
| XGBoost | 0.166 / 0.166 / n.a. / 0.425 | 0.200 / 0.475 / n.a. / 0.473 |
| MLP | 0.166 / 0.184 / n.a. / 0.436 | 0.201 / 0.321 / n.a. / 0.470 |

The unadjusted linear source predictor is worse than predicting the mean (ratio above 1) in both
scenarios; the closed-form adjustment recovers oracle performance. The non-linear source models are
far more robust in the redundant case (XGBoost source equals its oracle) but not in the confounded
case (XGBoost 0.475 vs 0.200; MLP 0.321 vs 0.201). MissForest imputation treating zeros as missing
is the worst arm almost everywhere (0.918 to 1.157).

### Semi-synthetic UCI (Table 1, right columns)

Gaps are smaller but consistent. Linear: adult `m^s <= m^t` source 0.437 vs oracle 0.420 vs
closed-form 0.422; bank general 0.480 vs 0.433 vs 0.442; thyroid general 0.320 vs 0.251 vs 0.291.
Non-parametric adjustment of the MLP matches its oracle on adult and bank in the proper regime
(0.389 vs 0.389; 0.294 vs 0.295). The source predictor's error grows faster with
`||m^s - m^t||_2` than any adjusted model (Figure 2b). For XGBoost the non-parametric adjustment
often ties with the unadjusted source.

### Real eICU data (Table 2)

All adjustments *fail*: for H2 -> H1, Source AUROC 0.593 vs Closed-form 0.540 vs Non-param 0.555
vs Oracle 0.713; Imputed collapses to 0.500 because all features are binary and MissForest turns
every zero into a one. Estimated relative missingness shows `m^s` is not element-wise below `m^t`,
so the non-parametric adjustment is improper, and the authors state that hospitals differ in
`P(X, Y)` and in covariate-dependent missingness, so "fundamental assumptions of our adaptation
methods are likely violated."

### Conceptual results

- Without indicators, missingness shift is neither covariate shift nor label shift (Section 1).
- Corrupting a linear relationship with missingness makes the observed relationship non-linear,
  which is why flexible models partly self-correct (Section 8, citing Le Morvan et al. 2020b).
- "Learning without missing data indicators is fundamentally more difficult than learning with
  them, and methods which might make sense when missing data indicators are present (e.g.
  imputation) can be ill-defined when the indicators are absent."

## Relevance to Our Crop-Classification Study

Our revised findings are: (1) the holdout tile lacks September for 41 percent of pixels while
September is never missing in training; the old pipeline zero-filled it before standardization for
every raw-sequence deep model, cutting holdout macro-F1 by 0.08 to 0.14 and producing an apparent
in-region / out-of-region ranking inversion; with the month filled from training-tile medians the
dense temporal models lead (CNN-BiLSTM 0.66, TempCNN 0.65, L-TAE 0.65, Transformer 0.64, TimesNet
0.63), TabNet 0.62, tree / linear / MiniRocket 0.59 to 0.61, no inversion; (2) a sparsemax channel
gate gives +0.06 macro-F1 robustness to the zero-filled month on CNN-BiLSTM and L-TAE-S but no gain
on complete inputs, and TabNet is not robust (0.61 to 0.51); (3) learning curves on the corrected
holdout show about 0.55 macro-F1 for every model at 10 percent of fields, deep models rising to
about 0.65 at 100 percent, logistic regression plateauing at 0.61.

**This paper names and formalizes the thing we found.** Our old pipeline satisfied the DAMS
definition almost literally: the same crops and the same bands (the authors' "fixed underlying
P(X, Y)" is an idealization we only approximately meet across tiles), source missingness rate zero
for every month, target missingness rate 0.41 for one month, and, critically, *no indicator*: the
zero-filled September entered the network looking like a legitimate standardized value. The paper's
central negative result, that an optimal source predictor can be arbitrarily bad on the target when
indicators are absent (Remarks 1 and 2, Table 1 linear source 1.259 vs oracle 0.178), is the
general form of what we measured as a 0.08 to 0.14 macro-F1 loss. It supports revised finding (1)
and reframes it: the inversion was not a property of dense temporal models but a property of
*deploying any predictor under an unindicated missingness shift*.

**It predicts which model families suffer most, and the prediction matches our old ranking.** In
the redundant-features scenario, XGBoost trained on the source matches its oracle (0.166 vs 0.166)
while the linear model collapses (1.259); the MLP is in between (0.184). The authors' explanation
is that missingness turns a linear relation into a non-linear one that flexible, piecewise models
partly absorb, and that trees route a zeroed feature down a branch rather than multiplying it into
every output. Our tree and xr_fresh-feature models were the least affected by the September gap in
the old pipeline, and the dense networks, which push every input through affine layers, were the
most affected; that is the paper's ordering. The same table warns against over-reading tree
robustness: in the confounded scenario XGBoost source is 2.4x its oracle (0.475 vs 0.200). The
apparent "sparse models transfer best" story of the original manuscript was this
scenario-dependent robustness to an unindicated shift, not a transfer property.

**Its two remedies correspond to the two fixes we tested.** Algorithm 1 adds missingness to the
*source* data so it is distributed like the target, then trains any model; that is exactly the
training-time temporal-dropout augmentation of Wang et al. (2025) and the temporal random masking
of Zhang et al. (2024), applied with a rate matched to the deployment tile. Our median fill does the
reverse: it moves the *target* toward the source by replacing the missing month with a plausible
source-distribution value. Both restore exchangeability between training and test inputs, which
Theorem 5.2 says is what matters. The paper's Proposition 1 adds a third, cleaner option that we
have not tried: append the missingness indicator as an input. With the indicator present,
missingness shift becomes covariate shift, and the authors note that with overlapping support
"often no adaptation is needed." A masked-input model (Che et al. 2024's masking Bi-LSTM and
Transformer) is this option in architectural form; for TempCNN and other models that cannot skip
steps, an explicit indicator channel is the equivalent. The paper's MissForest arm is a caution
against the one thing we did *not* do: treating every zero as missing and imputing it, which
collapsed to chance on eICU (AUROC 0.500) and was the worst arm on every synthetic set.

**On revised finding (2).** Theorem 4.1 says that random zeroing of inputs acts as L2
regularization, i.e. dropout. A sparsemax channel gate trained on complete inputs has no
regularization pressure from missingness, which is consistent with our observation that it gives
nothing on complete inputs; under zero fill it can learn, from the corrupted channel's implausible
value, to switch that channel off, which is a learned stand-in for the missing indicator and
explains a partial (+0.06) recovery rather than a full one. The TabNet result (0.61 to 0.51) is
the paper's Example 1 in miniature: a sparse selector that learned under `m^s = 0` to rely on a
feature that is zeroed under `m^t` has no mechanism to re-route, and sparsity makes the dependence
on a few features *more* concentrated, not less.

**On revised finding (3).** The paper has no learning-curve results. Theorem 6.1 does bound the
error in estimating relative missingness as a function of source and target sample sizes, which
is irrelevant to our fill (we know the month is missing) but would matter if we estimated gap
rates from data for an Algorithm 1-style augmentation.

**What the paper does not give us.** It is regression-first (squared error), linear-first (the
closed-form result), tabular (no sequence structure), and its only real-data test failed. Its
UCAR assumption (independent per-feature Bernoulli missingness) is violated by our gap, which is
one whole month of every band for 41 percent of pixels, i.e. perfectly correlated across the six
bands and deterministic given the pixel's position in the cloud mask. The identification results
therefore do not apply literally; the conceptual results (indicator absence is the problem;
exchangeability is the fix; flexible models partly self-correct) do.

**What to cite it for.** (a) The definition of missingness shift and the statement that without
indicators it is neither covariate nor label shift. (b) Remarks 1 and 2 and Table 1: an unadjusted
source predictor can be worse than the label mean under missingness shift, with linear models
hurt most and tree models least in the redundant case. (c) Proposition 1: with indicators,
missingness shift reduces to covariate shift. (d) Algorithm 1: re-masking source data to match
target missingness, the formal justification for training-time temporal dropout at a matched rate.
(e) Section 8: imputation is ill-defined without indicators.

**Matched or mismatched regime.** Mismatched by design in every experiment; that is the paper's
subject.

**Imputation strategies compared at inference.** One (MissForest on all zeros, both domains),
included as a negative control; zero fill is the *given* corruption, not a strategy, and mean /
median / indicator strategies are not evaluated.

## Evaluation Caveats

1. **Not a remote-sensing study.** No spatial, temporal, or crop data; no F1; no class imbalance
   analysis. Its relevance is conceptual.
2. **UCAR is a strong and, for us, false assumption.** Our missingness is month-wise, band-coherent,
   and spatially structured (cloud mask), not independent Bernoulli per feature. The identification
   and closed-form results require UCAR; only Proposition 1 and the qualitative conclusions carry
   over.
3. **Regression and MSE throughout** (except eICU AUROC/AUPRC). No classification-margin or
   macro-F1 analogues of the theorems.
4. **Real-data test failed.** On eICU every adjustment under-performed the unadjusted source
   model, and the authors concede that hospital differences in `P(X, Y)` and covariate-dependent
   missingness violate the assumptions. The paper's positive evidence is entirely synthetic or
   semi-synthetic with synthetic labels.
5. **Semi-synthetic labels are linear in clean covariates**, which favours the linear closed-form
   method and makes the XGBoost and MLP oracles an upper bound on an artificially simple problem.
6. **Non-parametric adjustment is only proper one way** (`m^s <= m^t`). In our setting that
   direction holds (training has no gap, holdout has one), so Algorithm 1 would be proper; in the
   reverse deployment (training with gaps, test complete) it would not.
7. **No sequence models.** The MLP is the only neural arm; LSTMs, CNNs, and attention models,
   which are our concern, are not studied, and the interaction between missingness shift and
   temporal position is not addressed.
8. **The Oracle arm requires labelled target data**, which we have only for scoring; the gaps
   between Source and Oracle in Table 1 are upper bounds on what any unsupervised adaptation can
   recover.
9. **Sample sizes in the real experiment are small** (about 3,000 per hospital) with wide CIs
   (Oracle AUROC 0.652 to 0.775), so the eICU negative result is also weakly powered.
10. **Variance across missingness draws is reported only in the appendix**; Table 1 averages over
    draws without CIs in the main text.
