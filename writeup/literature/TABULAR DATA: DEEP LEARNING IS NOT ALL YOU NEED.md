# Tabular Data: Deep Learning Is Not All You Need

**Citation:** Shwartz-Ziv, R., & Armon, A. (2022). Tabular data: Deep learning is not all you need.
*Information Fusion*, 81, 84–90. DOI: 10.1016/j.inffus.2021.11.011 (verified via Crossref: title
"Tabular data: Deep learning is not all you need", *Information Fusion* vol. 81, pp. 84–90, issued
2022-05, authors Shwartz-Ziv and Armon). The PDF in this folder is the arXiv preprint
`arXiv:2106.03253v2` (dated 24 November 2021), not the journal typeset version; section numbering
and table numbers below follow the preprint.

> Methodology / benchmark paper. Per authoring rule 7, the spatial-evaluation-protocol question does not apply: there is no scene, tile, field, year, or sensor. The paper is a controlled re-benchmark of deep tabular architectures (`TabNet`, `NODE`, `DNF-Net`, `1D-CNN`) against `XGBoost` under one shared tuning and preprocessing protocol. Its value to us is methodological: what happens to a model's apparent advantage when every competitor is treated identically.

## Objectives

The paper asks whether any of the recently proposed deep architectures for tabular data should be a
recommended default over gradient-boosted decision trees. The authors frame this as two concrete
questions. First, are the deep models more accurate, and in particular are they more accurate on
datasets that did not appear in the paper that proposed them? Second, how long do training and
hyperparameter search take relative to `XGBoost`?

The motivation is that each deep tabular paper had evaluated on its own choice of datasets, there
was no shared benchmark comparable to ImageNet or GLUE, some models lacked open implementations,
and, in the authors' words, "papers that attempted to compare these models often did not optimize
them equally." The study therefore sets out to re-run all models under the same tuning budget and
the same preprocessing so that reported advantages can be separated from pipeline advantages.

A secondary objective is constructive: to test whether an ensemble of deep models with `XGBoost`
improves on `XGBoost` alone, and whether the deep components are actually needed in that ensemble or
whether an ensemble of classical models (`XGBoost`, `SVM`, `CatBoost`) would do as well.

## Methods

**Models compared.** Four deep tabular architectures that had claimed to beat tree ensembles:
`TabNet` (sequential attention with sparsemax masks for instance-wise feature selection), `NODE`
(ensembles of differentiable oblivious decision trees), `DNF-Net` (soft disjunctive-normal-form
blocks), and a `1D-CNN` that won a Kaggle tabular competition. The baseline is `XGBoost`. Two
ensemble variants combine all five: a uniformly weighted mixture of predicted probabilities (Eq. 1),
and a weighted average with weights proportional to normalized validation loss (Eq. 2). A separate
"classical" ensemble (`XGBoost`, `SVM`, `CatBoost`) and a "deep ensemble without `XGBoost`" are also
reported.

**Datasets.** Eleven tabular datasets spanning 10 to 2,000 features, 1 to 7 classes (regression
included), and 7k to 1M samples. Nine come from the three deep papers, three apiece: Rossmann Store
Sales, Forest Cover Type, and Higgs Boson from the `TabNet` paper; Gas Concentrations, Eye
Movements, and Gesture Phase from the `DNF-Net` paper; YearPrediction, Microsoft MSLR, and Epsilon
from the `NODE` paper. Two Kaggle datasets, Shrutime and Blastchar, appear in none of the source
papers. The `1D-CNN` was proposed for a single Kaggle competition dataset that is not used, so every
dataset is "unseen" for it.

**Terminology.** A model's "unseen datasets" are those not reported in its original paper, not
datasets it was never trained on. The question is whether the model's advantage survives on data its
authors did not choose.

**Shared preprocessing.** Each dataset is preprocessed as described in its source paper. Inputs are
standardized to zero mean and unit variance, "and the statistics for the standardization are
calculated based on the training data." This is applied identically to every model. The
train/validation/test splits follow the original papers: random stratified 80/20 train/validation
for Epsilon, YearPrediction, MSLR, Shrutime, and Blastchar; 70/10/20 for Eye, Gesture, and Gas; the
dataset authors' split for Forest Cover Type; and for Rossmann a temporal split (2014 for training
and validation, 2015 for test). Where the source split was random, three repetitions are run and the
mean and standard error reported; otherwise four random seeds on the fixed partition.

**Shared tuning protocol.** Hyperparameters are optimized with HyperOpt (Bayesian optimization) for
1,000 steps per dataset on the validation set, starting from the hyperparameters published in each
model's paper. Each model has 6 to 9 tuned hyperparameters (learning rate, depth or layers, width,
regularization, batch size); the full search spaces are in Appendix B. Deep models use Adam without
schedules; training stops after 100 epochs without validation improvement. Early stopping on the
validation set applies to all models.

**Metrics.** Cross-entropy loss (reported with a 100x factor) for classification, RMSE or MSE for
regression. Lower is better throughout. Table 3 summarizes each model's average relative
deterioration versus the best model on each of its unseen datasets (geometric mean of relative
performance). The Friedman rank test at the 95% level, followed by pairwise comparisons, is used for
significance.

**Tuning-difficulty analysis.** Rather than wall-clock time (which the authors argue is confounded
by software optimization, since `XGBoost` is a mature library and the deep models are research
code), they compare the number of HyperOpt iterations each model needs before its validation loss
plateaus, on Shrutime. This is offered as a proxy for how far a model's published default
hyperparameters sit from good ones on a new dataset.

**Ensemble subset analysis.** On Shrutime, ensemble members are added in three orders: by validation
loss, by per-example confidence, or at random, to see how many models are needed to approach the
full ensemble.

**Evaluation protocol:** Not spatial. All benchmarks are i.i.d. random or dataset-provided splits of
standard tabular datasets, with the single exception of Rossmann, which uses a temporal train/test
split (2014 to 2015) and so carries a mild distribution shift. There is no field, tile, region, or
sensor dimension. The relevant protocol feature for us is not where test data come from but that
every model faces the same preprocessing, the same split, and the same tuning budget, with
standardization statistics computed from training data only.

## Key Findings

**Deep models lose their advantage on datasets they were not introduced on.** Table 3 gives the
average relative deterioration from the best model on each model's unseen datasets:

| Model | Average relative deterioration on unseen datasets |
|---|---|
| Deep ensemble with `XGBoost` | 2.32% |
| Simple (classical) ensemble | 3.15% |
| `XGBoost` | 3.34% |
| Deep ensemble without `XGBoost` | 6.91% |
| `1D-CNN` | 7.56% |
| `TabNet` | 10.51% |
| `DNF-Net` | 11.96% |
| `NODE` | 14.21% |

(The running text rounds these to 3.4, 7.5, 10.5, 11.8, and 14.2; the `DNF-Net` figure differs
slightly between text and table.) `XGBoost` alone is roughly three to four times closer to the best
model than `TabNet`, `DNF-Net`, or `NODE` are. `XGBoost` beat every deep model on 8 of the 11
datasets when those datasets were outside the deep model's own paper, with p < 0.005 by the Friedman
test.

**Per-dataset detail (Table 2, lower is better).** On `TabNet`'s own datasets `TabNet` edges
`XGBoost` (Rossmann 485.1 vs 490.2; CoverType 3.01 vs 3.13; Higgs 21.14 vs 21.62). On the `DNF-Net`
datasets `TabNet` trails badly (Eye Movements 67.1 vs 56.1, about 20% worse; Gesture 96.4 vs 80.6,
about 20% worse). On the `NODE` datasets it trails again (YearPrediction 83.2 vs 78.0), and on the
two neutral Kaggle datasets it loses (Shrutime 14.94 vs 13.82; Blastchar 23.72 vs 20.39, about 16%
worse). `NODE` is 33% worse than `XGBoost` on CoverType (4.15 vs 3.13) and about 22% worse on Eye
Movements. `DNF-Net` is 37% worse on Blastchar (27.91 vs 20.39) and, notably, loses to `XGBoost` on
two of its own three datasets (Eye 68.4 vs 56.1; Gesture 87.0 vs 80.6), winning only on Gas (1.44 vs
2.18). `NODE` likewise loses to `XGBoost` on one of its own three (MSLR 55.72 vs 55.43). So the
paper's summary line, "each deep model was better only on the datasets that appeared in its own
paper," is if anything generous to the deep models.

**The authors' two explanations for the gap.** Selection bias (each paper reports the datasets on
which its model happened to work) and unequal tuning (each paper's authors searched their own
model's hyperparameters more extensively than the baselines'). They rule out implementation error
because their re-runs on each model's original datasets reproduce the published numbers.

**Ensembling.** The deep-plus-`XGBoost` ensemble is the best model on average (2.32%) and
significantly better than any single deep model on 7 of 11 datasets (p < 0.005). The deep ensemble
without `XGBoost` (6.91%) is much worse than `XGBoost` alone, so the trees are the load-bearing
component. The text states that the classical ensemble "performed much worse" than the
deep-plus-`XGBoost` ensemble; Table 3 shows the gap as 3.15% vs 2.32%, so "much worse" is an
overstatement, and the classical ensemble is in fact slightly better than `XGBoost` alone.

**Subset selection.** Adding ensemble members in order of validation loss reaches near-full-ensemble
performance with three models; random order is worst (p < 0.005 for the first three additions).

**Tuning cost.** `XGBoost` plateaus in far fewer HyperOpt iterations than the deep models on
Shrutime, and each iteration is shorter. The authors note three caveats: results may depend on the
Bayesian optimizer used; `XGBoost`'s published defaults may simply have been tuned across more
datasets historically; and the behavior may be intrinsic to the model. They report `XGBoost` as
"more than an order of magnitude" faster in wall-clock time but decline to lean on that number
because the deep implementations were research code.

**Overall conclusion.** "Deep learning is currently not all we need for tabular data." `XGBoost` is
the recommended default under time constraints; a deep-plus-tree ensemble is the recommended option
when the extra compute is affordable; and future tabular papers should evaluate on several diverse
datasets with a shared tuning protocol and report tuning difficulty alongside accuracy.

## Relevance to Our Crop-Classification Study

This paper is directly relevant to the revision of our manuscript, but for a different reason than
it would have been to the originally submitted version. The submitted version cited the "sparse,
feature-selecting, tree-like models transfer best" result as our central finding, and Shwartz-Ziv
and Armon would have read as supporting evidence that `XGBoost` and `TabNet`-style sparsity are the
safe choice. The re-analysis in `experiments/holdout_fill_check/` has since shown that our ranking
was an artifact of an asymmetric preprocessing step, and this paper is the cleanest statement in the
literature of why that kind of asymmetry matters.

**What we found.** The holdout tile `34S_20E_259N` has no September observation for 40.6% of its
pixels (39.3% of fields); the two training tiles have none missing. The published pipeline handled
that gap differently by model family. The raw-sequence deep models (`L-TAE`, `L-TAE-S`, `TempCNN`,
`Transformer`, `CNN-BiLSTM`, `TimesNet`, `MiniRocket`) received a zero reflectance at the missing
month before standardization, a value that never occurs in training. Pixel `TabNet` received the
holdout's own column medians. The xr_fresh classical models computed their statistics on the
shortened series. The two model families therefore faced different inputs on the same fields, a
train/test and model-to-model missingness shift. On the 1,352 holdout fields where September was
observed, the published predictions already ranked the dense networks first (`CNN-BiLSTM` 0.66,
`TempCNN` 0.65 macro-F1) and `TabNet` fourteenth (0.62). On the 1,063 fields with September missing,
the dense networks fell to 0.21 to 0.36 while `TabNet` (0.58) held up. Pooled over all fields, that
produced the apparent inversion: `TabNet` first at 0.603, `CNN-BiLSTM` nineteenth at 0.513.

**What the re-run shows.** Filling the missing month from training-tile medians for every model,
with no retraining, the dense pixel networks rise by 0.08 to 0.14 macro-F1 (`CNN-BiLSTM` 0.51 to
0.66, `L-TAE` 0.54 to 0.65, `TempCNN` 0.57 to 0.65, `MiniRocket` 0.50 to 0.61), pixel `TabNet`
barely moves (0.603 to 0.616), and the xr_fresh classical models gain 0.02 to 0.05. Under one
protocol the top six are dense pixel temporal models at 0.63 to 0.66, and trees, logistic
regression, and `TabNet` sit at 0.59 to 0.62. `TabNet` drops from rank 1 to rank 7. The revision
presents this as a re-analysis: name the asymmetric step, re-run every model under one fill, and
report the before and after ranking.

**The parallel to this paper is exact in structure.** Shwartz-Ziv and Armon's thesis is that a
model's published advantage can be a property of its pipeline rather than of the model. In their
setting the pipeline advantage was tuning effort and dataset selection; in ours it was the fill
operator. `TabNet`'s published holdout lead of 0.03 over the next model, and its 0.09 lead over
`CNN-BiLSTM`, was not a property of sparse attentive feature selection. It was a property of
`TabNet` having been handed a plausible value where the dense models were handed a value from
outside the training distribution. When the authors here equalized the tuning protocol, `TabNet`
fell from winning on its own benchmarks to 10.5% behind the best model on others; when we equalized
the fill, `TabNet` fell from first to seventh. The quantity that changed was the comparison, not the
model.

**The "unequal tuning" argument generalizes to unequal preprocessing.** The paper's diagnosis is
that "papers that attempted to compare these models often did not optimize them equally," and its
remedy is a single tuning budget and a single standardization computed from training statistics for
every model. Our manuscript's remedy is the same in kind: one fill operator, computed from
training-tile statistics, applied to every model before inference. The paper's explicit note that
standardization statistics come from training data only is the detail that matters. Our old `TabNet`
path used holdout column medians, which is a holdout statistic; our new fill uses training-tile
medians, which is the leakage-safe direction and the one this paper's protocol prescribes.

**The ensemble result still bears on our design.** The paper finds that the deep-plus-`XGBoost`
ensemble is best (2.32%) and that the deep ensemble without trees (6.91%) is much worse than
`XGBoost` alone (3.34%). Our final submission is a vote over `out_of_sample/predictions_*.csv`,
which under the published pipeline mixed models that had seen different holdout inputs. After the
re-analysis, the question of whether a tree-plus-dense ensemble beats the best dense model on the
filled holdout is open and worth reporting; the paper's finding that validation-loss-ordered subsets
of three models suffice is a practical recipe for that experiment. The paper also finds that the
trees are the component that cannot be dropped, which is consistent with the classical models' 0.02
to 0.05 gains on the filled holdout being smaller but real.

**What this paper does not license.** It is tempting to read the paper as "trees beat deep models"
and leave it there. The careful reading is narrower: deep tabular models lose their edge when their
authors' pipeline advantages are removed, and no single model wins everywhere. Our revised result is
the mirror image, dense temporal models winning once the fill is equalized, and it does not
contradict the paper because our inputs are raw temporal sequences rather than static tabular rows.
The paper's lesson is about procedure, and the procedure is what we now follow.

**Reporting implications for the revision.** The paper's closing recommendation, that model
comparisons report tuning difficulty alongside accuracy, maps onto a practice we should adopt for
preprocessing: a per-model table stating the fill operator, the statistic it was computed from
(training tile vs holdout), and the order of fill and standardization. The paper's reproduction
check (re-running each model on its own published datasets and matching the published numbers before
trusting the cross-dataset results) is also the check we ran: the original-file re-run reproduces
the published predictions field for field for every deep model except pixel `TabNet`, whose
published file predates its saved checkpoints and differs on 221 of 2,415 fields (0.603 vs 0.609).
That discrepancy should be stated in the revision.

**Equal treatment of baselines as the general lesson.** The paper's critique of the deep tabular
literature is that baselines were under-tuned relative to the proposed model. Our critique of our
own submitted manuscript is that the dense baselines were under-served by preprocessing relative to
`TabNet`. Both are instances of the same error: a comparison that attributes to architecture what
was actually caused by the pipeline. The fix in both cases is to make the pipeline identical and
only then compare.

## Evaluation Caveats

- **No spatial, temporal, or sensor transfer is tested.** Ten of eleven datasets use i.i.d. splits;
  only Rossmann has a temporal train/test split. The paper cannot speak to whether `XGBoost`'s
  robustness extends to out-of-region transfer, and we should not cite it as evidence on that
  question.
- **Static tabular rows, not sequences.** None of the datasets is a time series in the sense of our
  pixel-level monthly reflectance stacks. Temporal convolutions, attention over time steps, and
  recurrent encoders are absent from the comparison, so the paper's "deep models" are not the dense
  temporal models in our revised ranking.
- **No missing-data handling is studied.** The paper states that datasets are preprocessed as in
  their source papers and standardized with training statistics, but it does not vary imputation or
  examine robustness to missingness shift. Its relevance to our fill-operator finding is by analogy
  of protocol, not by direct experiment.
- **Metrics are cross-entropy and RMSE, not macro-F1.** Class imbalance is not addressed; no
  balanced metric is reported. The relative-deterioration figures in Table 3 are geometric means of
  loss ratios and are not comparable to our macro-F1 gaps.
- **Internal inconsistency in the ensemble claim.** The text says the classical ensemble "performed
  much worse" than the deep-plus-`XGBoost` ensemble, but Table 3 gives 3.15% vs 2.32%, and the
  classical ensemble is better than `XGBoost` alone. The Table 2 column labelled "Simple Ensemble"
  is presumably the classical ensemble, but the labelling is not explicit.
- **Tuning-difficulty analysis is on one dataset** (Shrutime) with one optimizer (HyperOpt), and the
  authors themselves list three confounds. The order-of-magnitude runtime claim is explicitly not
  relied upon.
- **The 1D-CNN result is not symmetric.** All eleven datasets are unseen for it, so its 7.56% figure
  is not comparable to the other deep models' figures, which average over fewer, harder datasets.
  The authors note this.
- **Preprint vs journal version.** The PDF here is `arXiv:2106.03253v2`. The Information Fusion
  version may differ in wording, table numbering, or minor figures; the DOI and citation above refer
  to the journal article, and quotations are from the preprint.
- **Selection bias cuts both ways.** The paper argues each deep paper chose favorable datasets. The
  two "new" Kaggle datasets (Shrutime, Blastchar) are small (7k to 10k rows) and conventionally
  tree-friendly, so the authors' own dataset choice is not neutral either.

