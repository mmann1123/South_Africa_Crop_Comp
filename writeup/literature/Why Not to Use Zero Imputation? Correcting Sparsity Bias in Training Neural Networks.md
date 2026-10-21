# Why Not to Use Zero Imputation? Correcting Sparsity Bias in Training Neural Networks

**Citation:** Joonyoung Yi, Juhyuk Lee, Kwang Joon Kim, Sung Ju Hwang, and Eunho Yang (2020).
*International Conference on Learning Representations (ICLR 2020)*. arXiv:1906.00150v5 [cs.LG],
6 February 2020 (the PDF in this folder is the arXiv v5 camera-ready marked "Published as a
conference paper at ICLR 2020"). DOI: not found in PDF. Crossref title searches
(`query.title=Why Not to Use Zero Imputation Correcting Sparsity Bias in Training Neural Networks`
and `query.bibliographic=Why Not to Use Zero Imputation`) on 2026-10-21 returned no matching record
(top hits were unrelated imputation papers), consistent with ICLR proceedings not being
DOI-registered. Cite via arXiv / OpenReview.

## Objectives

This is a theory-plus-benchmarks machine-learning paper, not a remote-sensing paper. Its question is
why the simplest treatment of missing inputs to a neural network, setting them to zero, degrades
performance, a fact that had been reported empirically (Hazan et al. 2015; Luo et al. 2018; Smieja
et al. 2018) but not explained.

Objectives:

1. Define and demonstrate the *Variable Sparsity Problem* (VSP): the expected output of a network
   fed zero-imputed inputs varies systematically with the *fraction of missing entries* in the input,
   independent of the values that are present.
2. Prove that VSP arises under zero imputation for linear, affine, and convex non-decreasing (ReLU,
   leaky ReLU, ELU, softplus) activations, assuming missing-completely-at-random masks.
3. Propose *Sparsity Normalization* (SN), a one-line input rescaling by the number of observed
   entries, and prove it removes the dependence for affine networks.
4. Show empirically that VSP occurs, and SN helps, across collaborative filtering, electronic
   medical records (tabular and LSTM time series), single-cell RNA imputation, dropout in hidden
   layers, and autoregressive density estimation, and compare SN against standard imputation
   methods (mean, median, kNN, MICE, SoftImpute, GMMC, GAIN).

## Methods

### Setup and theory (Section 2)

Input `h^0 = x_tilde * m` where `x_tilde` is the (possibly unobserved) feature value and `m` is a
binary mask with mean `mu_m` (MCAR, independent of values and of other mask entries). Weights are
i.i.d. with mean `mu_w^i != 0`, biases with mean `mu_b^i`. For an `L`-layer network
`h^i = sigma(W^i h^{i-1} + b^i)`:

- Theorem 1 (identity activation, no bias): `E[h^L] = prod_i n_{i-1} mu_w^i * mu_x * mu_m`. The
  expected output is *proportional* to the fraction of observed entries.
- Theorem 2 (affine activation): `E[h^L] = f_L o ... o f_1(mu_x mu_m)` with
  `f_i(x) = sigma(n_{i-1} mu_w^i x + mu_b^i)`.
- Theorem 3 (convex non-decreasing activation, positive weight means):
  `E[h^L] >= f_L o ... o f_1(mu_x mu_m)`, a lower bound that still moves with `mu_m`.

The practical reading: two instances with identical observed values but different numbers of
zeroed entries get systematically different activations and therefore different predictions.

### Sparsity Normalization (Section 3, Algorithm 1)

Replace the input by `h^0_SN = K * h^0 / ||m||_1`, where `||m||_1` is the number of observed
entries in that instance and `K` is a constant, recommended as the training-set average of
`||m||_1` so that input scale is unchanged on average. Theorem 4 shows that for affine networks
`E[h^L]` then no longer depends on `mu_m`. The method is applied per instance (and per time step
for sequences), needs no learned parameters, and reduces to standard inverted-dropout scaling
when applied to hidden layers with a fixed drop rate.

### Experiments (Section 4 and appendices)

- Collaborative filtering: AutoRec, CF-NADE, CF-UIcA on MovieLens 100K / 1M / 10M, with and
  without SN. Five runs, 95 percent CIs.
- Electronic medical records: NHIS (about 300,000 Korean patients, 34 exams over 5 years, five
  binary disease outcomes, 2-layer MLP 50/30 units, min-max inputs, class-weighted loss, AUROC)
  and PhysioNet Challenge 2012 (4,000 ICU stays, 48 h at 288 steps x 35 exams, single-layer LSTM
  with 108 units, features standardized to zero mean and unit variance *then* zero-filled, class
  weights, early stopping on validation AUROC, SN applied per time step).
- Single-cell RNA: AutoImpute on six datasets, imputation RMSE vs training fraction.
- Dropout as hidden-layer VSP: three UCI regression sets, RMSE vs drop rate.
- Density estimation: MADE on binarized MNIST, NLL vs number of masks.
- Imputation comparison on NHIS (Table 3) and on UCI binary tasks with and without MCAR (Tables 9
  and 10).

**Evaluation protocol:** Standard i.i.d. train/test splits within each dataset (MovieLens
canonical splits; NHIS 3:1 random split; PhysioNet's official split; UCI random splits), five or
ten repeated runs with 95 percent confidence intervals. The missingness in every experiment is
either intrinsic to the data and shared by train and test (ratings, medical exams, RNA dropout) or
artificially injected at the same rate in both. The train/test missingness regime is therefore
*matched* throughout; the paper is about within-dataset variation in sparsity across instances,
not about a shift in sparsity between training and deployment. Imputation strategies are compared
(Table 3: zero, zero+SN, zero+BatchNorm, zero+LayerNorm, dropout, mean, median, kNN, MICE,
SoftImpute, GMMC, GAIN), but each strategy is applied identically to training and test data. There
is no spatial, temporal, or domain transfer component.

## Key Findings

### VSP is real and visible before training (Figure 1)

For a randomly chosen test instance with its mask re-sampled at controlled sparsity levels, the
zero-imputed model's prediction moves monotonically with the number of observed entries in all
three domains (predicted MovieLens rating rises with number of known ratings; PhysioNet LSTM
in-hospital death probability rises with number of exams; AutoImpute expression rises with number
of detected genes), whereas the ground-truth target shows no such trend. With SN the prediction is
flat in sparsity and its variance shrinks as more entries are observed. A footnote states the
effect "is observed throughout the entire learning process (even before the training)," so it is a
property of the forward pass under zero fill, not a learned artefact.

### SN improves accuracy where sparsity varies a lot (Tables 1 to 3)

- MovieLens AutoRec user-vector RMSE: 0.9346 to 0.9208 (100K), 0.8831 to 0.8742 (1M), 0.8859 to
  0.8462 (10M). Gains are consistent but modest for item-vector encodings.
- NHIS AUROC, zero imputation without vs with SN: cardiovascular 0.7057 to 0.7106, fatty liver
  0.6750 to 0.6911, hypertension 0.7977 to 0.8096, heart failure 0.7834 to 0.7914, diabetes 0.9121
  to 0.9283. SN beats every other listed imputation or normalization on all five tasks; MICE is
  second (0.7075, 0.6902, 0.7957, 0.7875, 0.9224). Batch norm and layer norm on zero-imputed inputs
  do *not* fix VSP (BN is worse than plain zero fill on four of five tasks).
- Mean and median imputation are statistically indistinguishable from zero imputation on NHIS
  (for example diabetes 0.9117 and 0.8975 vs 0.9121).

### SN does not always help accuracy (Table 6)

On PhysioNet with a single-layer LSTM fed standardized-then-zero-filled inputs, AUROC is 0.8177
without SN and 0.8152 with SN: no gain, although Figure 1 shows the bias in predicted mortality vs
number of exams is removed. The authors keep SN for calibration reasons rather than accuracy.

### Severity scales with sparsity variance (Figure 4, Section 4.4)

Interpreting dropout as hidden-layer zero imputation, the RMSE gap between with-SN and without-SN
widens as the drop rate rises; they define the degree of VSP as `sqrt(p / (n (1 - p)))`, the inverse
signal-to-noise ratio of the active-unit count.

### SN matters most with little data (Figure 3)

On six single-cell datasets the SN gain in imputation RMSE grows as the training fraction shrinks.

## Relevance to Our Crop-Classification Study

Our revised findings are: (1) the holdout tile lacks September for 41 percent of pixels while
September is never missing in training; the old pipeline zero-filled it *after* standardization for
every raw-sequence deep model, cutting holdout macro-F1 by 0.08 to 0.14 and producing an apparent
in-region / out-of-region ranking inversion; with the month filled from training-tile medians the
dense temporal models lead (CNN-BiLSTM 0.66, TempCNN 0.65, L-TAE 0.65, Transformer 0.64, TimesNet
0.63), TabNet 0.62, tree / linear / MiniRocket 0.59 to 0.61; (2) a sparsemax channel gate gives
+0.06 macro-F1 robustness to the zero-filled month (CNN-BiLSTM, L-TAE-S) but no gain on complete
inputs, and TabNet is not robust to zero fill (0.61 to 0.51); (3) learning curves on the corrected
holdout show about 0.55 macro-F1 for every model at 10 percent of fields, deep models reaching
about 0.65 at 100 percent, logistic regression plateauing at 0.61.

**This paper supplies the mechanism behind revised finding (1).** Theorems 1 to 3 say that under
zero imputation the expected pre-activation of every layer is a function of the observed fraction
`mu_m`. In our training tiles `mu_m = 1` for every pixel; on the holdout tile 41 percent of pixels
have one month of every band zeroed (for six bands over roughly ten months, about 10 percent of the
input entries). The network's internal activations for those pixels are therefore pushed toward a
regime it never encountered, by an amount that Theorem 1 says is proportional to the missing
fraction in the linear case and that the convex-activation bound says is at least that in the ReLU
case. The paper's Figure 1 PhysioNet panel is a near-exact analogue of our pipeline: an LSTM over
standardized-then-zero-filled multivariate time steps, where predicted class probability drifts
with the count of zeroed entries even though the label does not. Our zero-fill damage is the
cross-domain version of that drift: training never contained the drift, so no part of the decision
boundary accounts for it. The paper therefore *anticipates* finding (1) and gives it a theoretical
footing: the penalty we observed is not evidence of dense-model fragility to transfer but the
predicted consequence of a sparsity level unseen in training.

**It also explains why the median fill works.** SN works by restoring the expected activation to
its training value. A training-median fill after standardization does the same thing more crudely:
it places the missing month at (approximately) the training mean of each band, so `E[h^1]` for the
affected pixels is nearly what it would have been with the true value, and the downstream layers
see no sparsity-driven shift. Mean and median imputation performed no better than zero on NHIS
(Table 3), but that is because NHIS inputs were min-max scaled to [0, 1], so zero was already an
in-range value and the mean was not much more informative; in our pipeline the inputs are
standardized, so zero is the mean and the "zero" we inserted was the raw-scale zero, far outside
the band's distribution. The paper's result that batch norm and layer norm do *not* repair zero
imputation (Table 3) is worth citing against the objection that our networks' normalization layers
should have absorbed the shift.

**On revised finding (2), the sparsemax gate.** SN and a learned channel gate are both input-side
re-weightings. SN rescales all observed entries by `K / ||m||_1`; a sparsemax gate can learn to
assign zero weight to a channel whose value is implausible, which for the single missing month
amounts to a learned, data-dependent mask. The paper gives a reason to expect the gate's robustness
gain to show up only under corruption: on complete inputs `||m||_1` is constant and SN is an
identity up to scale, which matches our observation that the gate gives nothing on complete inputs
and +0.06 under zero fill. It also gives a reason to expect the gate to be an incomplete fix: SN
corrects the expected activation but not the information lost, and the paper's own LSTM experiment
(PhysioNet, 0.8177 vs 0.8152) shows that removing the bias does not necessarily recover accuracy.
Our TabNet result (0.61 to 0.51 under zero fill, no robustness despite sparse attention) is
consistent with the paper's framing: TabNet's attentive transformer sees the zero-filled xr_fresh
or raw features through ordinary affine layers first, so VSP acts before sparsity can help, and a
sparse selector that was trained with `mu_m = 1` has no reason to drop the corrupted entries.

**A concrete, cheap experiment the paper suggests.** Apply SN (per time step, as in their PhysioNet
setup) to our raw-sequence models and evaluate on the zero-filled holdout. If SN closes most of the
0.08 to 0.14 gap, that is an independent confirmation that the inversion was VSP; if it closes
little, as on PhysioNet, the information loss dominates and the median fill (which supplies a
plausible value rather than only a scale correction) is the right remedy. Either outcome
strengthens the revision.

**On revised finding (3).** The paper reports that SN's benefit grows as the training set shrinks
(Figure 3, single-cell RNA). Our learning curves are computed on the *corrected* (median-filled)
holdout, so VSP is not in play there, and the paper has no direct bearing on the 0.55 floor at 10
percent of fields.

**What to cite it for.** (a) Theorems 1 to 3: under zero imputation the expected network output
depends on the input's missing fraction. (b) Figure 1, PhysioNet panel: an LSTM over
standardized-then-zero-filled time steps drifts with the number of zeroed entries even before
training. (c) Table 3: normalization layers do not repair zero imputation; mean / median fills are
equivalent to zero fill only when zero is already in-range. (d) Table 6: removing the bias does not
guarantee recovering accuracy when information is genuinely absent.

**Matched or mismatched regime.** Matched. All experiments vary sparsity *across instances* within
a dataset whose train and test splits share the same missingness distribution. The paper does not
test a model trained at one sparsity level and deployed at another, which is our case and is the
harder one (no instance in training spans the deployment sparsity).

**Imputation strategies compared at inference.** Yes, extensively (Table 3, Tables 7, 9, 10), but
each strategy is applied consistently to train and test; there is no cross-condition evaluation.

## Evaluation Caveats

1. **Not a remote-sensing or transfer study.** No spatial, temporal, or domain shift anywhere; all
   accuracy claims are i.i.d. within-dataset results. The relevance is mechanistic, not empirical
   comparability.
2. **MCAR assumption in the theory.** Theorems 1 to 4 assume masks independent of values. Our
   September gap is not random: it is tile-wide and month-specific (missing-not-at-random at the
   pixel level, deterministic at the tile level). The paper shows empirically that VSP persists
   without MCAR (MovieLens, single-cell), so the mechanism is still expected to apply, but the
   proportionality constants do not transfer.
3. **Theory covers the forward pass in expectation only.** The theorems describe `E[h^L]`, not
   decision boundaries or classification error; the link from activation shift to F1 loss is
   empirical.
4. **The LSTM result is the weakest in the paper.** The one sequence-model experiment (PhysioNet)
   shows no accuracy gain from SN, which is the setting closest to ours. The paper's strongest
   gains are on very high-dimensional, very sparse inputs (ratings, RNA) unlike ours.
5. **Normalization choice confounds the imputation comparison.** NHIS inputs were min-max scaled,
   so "zero imputation" there is an in-range value; the Table 3 equivalence of zero, mean, and
   median does not carry over to standardized inputs where raw zero is an outlier.
6. **Model sizes are small.** MLPs of 50/30 units and a single-layer LSTM of 108 units; whether
   deeper networks with residual connections or attention show VSP of the same magnitude is not
   tested (the theorems do not bound it for attention).
7. **Class imbalance handled by class weights only**; AUROC is reported rather than any F1, so
   minority-class behaviour under VSP is not characterized.
8. **SN hyperparameter `K`.** Set to the training-mean sparsity; the paper notes dying-ReLU
   problems if `K` is too small, so SN is not entirely parameter-free.
9. **No evaluation of indicator channels.** Passing the mask as an extra feature, the standard
   alternative in clinical time series (Lipton et al. 2016; Che et al. 2018), is cited but not
   compared against SN.
