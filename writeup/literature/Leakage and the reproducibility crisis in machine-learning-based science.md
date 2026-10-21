# Leakage and the Reproducibility Crisis in Machine-Learning-Based Science

**Citation:** Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in
machine-learning-based science. *Patterns*, 4(9), 100804. DOI: 10.1016/j.patter.2023.100804
(verified via Crossref: title "Leakage and the reproducibility crisis in machine-learning-based
science", *Patterns* vol. 4 issue 9, article 100804, issued 2023-09, authors Kapoor and Narayanan).
Open access, CC BY. Published online 4 August 2023; issue date 8 September 2023. Code and data for
the case study are on CodeOcean (DOI 10.24433/CO.4899453.v1, as printed in the paper; not separately
verified).

> Methodology paper. Per authoring rule 7, the spatial-evaluation-protocol question does not apply to the paper itself: it is a cross-disciplinary survey, a taxonomy, a reporting proposal, and a reproducibility case study in political science. Its relevance to us is entirely methodological: it supplies the vocabulary (leakage types L1 to L3, with imputation named explicitly under L1.2), the documentation instrument (model info sheets), and a worked example of an ML-versus-baseline ranking that reversed once preprocessing was fixed.

## Objectives

The paper sets out to establish that data leakage is a systemic, cross-disciplinary cause of
irreproducible results in "ML-based science," defined as research that makes a scientific claim
using the performance of an ML model as evidence. The authors distinguish this from ML methods
research, ML engineering, and modeling contests, and argue that leakage mitigations developed for
those settings (limited production deployment, third-party holdout sets) do not transfer to science,
where the researcher usually has the whole dataset in hand while modeling.

Three contributions are stated. First, a survey of prior reviews to quantify how widespread leakage
is, and from it a taxonomy of eight leakage types "ranging from textbook errors to open research
problems." Second, a reporting instrument, the model info sheet, with 21 questions that force the
author to argue explicitly that none of the eight types is present. Third, a reproducibility study
in civil war prediction, a subfield where complex ML models (random forests, AdaBoost,
gradient-boosted trees) were believed to vastly outperform logistic regression, to test whether that
claim survives once leakage is corrected.

The authors define leakage as "a spurious relationship between the independent variables and the
target variable that arises as an artifact of the data collection, sampling, or pre-processing
strategy." Because that relationship is absent from the distribution about which the scientific
claim is made, leakage "usually leads to inflated estimates of model performance." A finding is
reproducible, in their broader sense, if code and data are available and the data are correctly
analyzed; this is wider than computational reproducibility (re-running the exact code).

## Methods

**Survey.** The authors searched for published reviews that document pitfalls in ML adoption within
a scientific field. Because fields use different vocabulary for the same errors, they present the
result as a lower bound rather than a systematic meta-review. They found 22 reviews across 17
fields, collectively covering 294 papers; Figure 1 tabulates each review against the leakage types
in the taxonomy plus four other issue categories: computational reproducibility, data quality (small
samples, large amounts of missing data), metric choice (accuracy under heavy class imbalance), and
problems arising despite use of standard datasets. Remote sensing is among the fields: Nalepa et al.
(2019), "Validating Hyperspectral Image Segmentation," IEEE GRSL, is reference 24 in the survey,
flagging train/test pixel overlap in hyperspectral benchmarks. Most reviews inspected only paper
text, not code, which the authors note undercounts leakage.

**Taxonomy.** Eight leakage types grouped under three headings, reproduced here because the revision
will cite them by label.

[L1] Lack of clean separation of training and test data. The model has access to information from
the test set during preprocessing, modeling, or evaluation.

- [L1.1] No test set. Training and testing on the same data.
- [L1.2] Preprocessing on training and test set. "Using the entire dataset for any pre-processing
  steps, such as imputation or over/under sampling, results in leakage." The paper's example is
  oversampling before the split, which places synthetic copies of training rows in the test set.
- [L1.3] Feature selection on training and test set. Selecting features using information about
  which features perform well on the test set.
- [L1.4] Duplicates in datasets. The same record in both partitions.

[L2] Model uses features that are not legitimate. A feature that is a proxy for the outcome, or that
would not be available at prediction time. No subcategories; the judgement is domain-specific, and
the authors ask researchers to justify every feature.

[L3] Test set is not drawn from the distribution of scientific interest. The evaluation distribution
differs from the one about which claims are made.

- [L3.1] Temporal leakage. Test data predating training data when the claim is about forecasting.
- [L3.2] Nonindependence between training and test samples. Observations from the same unit
  (patient, protein family) split across partitions. The recommended remedy is block
  cross-validation that partitions along the dependency structure. The authors call the general
  case, without assumptions on the dependency structure, "a hard problem."
- [L3.3] Sampling bias in test distribution. The paper's first example is spatial bias: "choosing
  the test data from a geographic location but making claims about model performance in other
  geographic locations." A second example is selection bias (dropping borderline cases). A third is
  Zech et al.'s finding that pneumonia classifiers trained at one hospital failed at another because
  of subtle differences in image generation.

The authors add a general remark under L3 that is important for us: "leakage can result from a
difference between the distribution of the test set and the distribution of scientific interest,"
and "even slight shifts in the target distribution can cause performance estimates to change
drastically." They treat robustness to distribution shift as an open problem and give the current
best practice as testing on data from the distribution one wants to make claims about.

**Model info sheets.** A template (supplemental section S4) of 21 questions organized to elicit
three arguments: that train/test separation was clean throughout preprocessing, modeling, and
evaluation (L1); that each feature is legitimate (L2); and that the test set is drawn from the
distribution of scientific interest with no selection or sampling bias (L3). The sheets are inspired
by Mitchell et al.'s model cards and Gebru et al.'s datasheets but target leakage specifically. The
authors note that most ML-based science papers present none of the three arguments, and at best
partially address L1 by reporting an out-of-sample number. Stated limitations: claims on a sheet
cannot be verified without computational reproducibility; incorrect sheets could give false
assurance; and filling one out needs ML expertise. The sheets are versioned so they can evolve.

**Civil war case study.** A systematic search (supplemental S2.1) returned 124 papers; 12 focused on
predicting civil war, used a train/test split, and shared complete code and data. For each of the
12, the authors read the text and the code, identified errors, and re-ran the analysis with the
errors corrected. Significance testing used the Robin et al. smoothed-ROC comparison with
bootstrapped test-set resampling.

**Evaluation protocol:** Not applicable in the spatial sense. The case study re-scores published
civil war models on their own published out-of-sample splits after fixing the preprocessing; it does
not introduce new data. The survey is a literature synthesis. What the paper evaluates is a
procedure (how train and test were separated, how features were chosen, what distribution the test
set represents) rather than a model, and that is precisely the lens the revision needs.

## Key Findings

**Scale of the problem.** 22 reviews, 17 fields, 294 affected papers. "A prominent finding that
emerges is that data leakage is a pitfall in every single case." Six of the 22 reviews report
multiple leakage types in their field. Ten reviews flag data-quality issues including unaddressed
missing values; four flag metric choice (accuracy under imbalance); five flag lack of computational
reproducibility; seven report issues despite standard datasets, usually because the split and metric
were not fixed.

**Civil war: errors found in exactly the papers claiming ML superiority.** Of the 12 papers with
code and data, 4 had leakage, "exactly the 4 papers that claimed superior performance of complex ML
models over baseline LR models." All four were in top-10 political science or international
relations journals. Figure 3 gives the per-paper detail:

| Paper | Claim | Error | Result after correction |
|---|---|---|---|
| Muchlinski et al. (2016) | Random forests drastically outperform logistic regression | [L1.2] Imputation of training and test data together; 95% of the out-of-sample dataset was missing and filled by the incorrect imputation | Random forests perform no better than logistic regression |
| Colaresi and Mahmood (2017) | Random forests drastically outperform logistic regression | [L1.2] Re-use of Muchlinski et al.'s incorrectly imputed dataset | Random forests perform no better than logistic regression |
| Wang (2019) | AdaBoost and gradient-boosted trees drastically outperform other models | [L1.2] Re-use of the incorrectly imputed dataset; [L3.1] k-fold cross-validation on temporal data | AUC difference between AdaBoost and logistic regression drops from 0.14 to 0.01 |
| Kaufman et al. (2019) | AdaBoost outperforms other models | [L2] Proxy variables for the outcome (`colwars`, `cowwars`, `sdwars`) used as predictors | AdaBoost no longer outperforms logistic regression; no model beats a baseline that predicts last year's outcome |

Three of the four cases trace to a single imputation performed on the pooled training and test data,
and two of them inherited that imputed file from the first. The only quantitative before/after
figure in the main text is Wang's AUC gap of 0.14 falling to 0.01; the other reversals are reported
as "no better than" without AUC values in the text (the values are in Figure 3's bars and in
supplemental S2). The authors emphasize that the logistic regression baselines were explanatory
models never tuned for prediction, which makes the parity after correction more striking.

**Capacity and spurious correlation.** "Leakage affects both simple and complex models for civil war
prediction. However, because of higher model capacity, complex ML models tend to over-fit to
spurious correlations more easily in this case." The imputation artifact did not help all models
equally; it helped the flexible ones most.

**No uncertainty quantification.** 9 of the 12 papers reported no significance tests or confidence
intervals. Blair and Sambanis (2020) reported a smoothed AUC of 0.85 on a test set with only 11
civil war onsets; the bootstrapped 95% CI is 0.66 to 0.95, and the model is not significantly better
than any of four baselines (p = 0.26, 0.14, 0.34, 0.25, one-tailed, uncorrected for multiple
comparisons).

**Info sheets would have caught every case.** The authors applied the 21-question sheet
retrospectively to the four papers and report (supplemental S3) that it surfaces each identified
leakage type.

**Five diagnoses and five recommendations (Discussion).** D1 lack of understanding of limits to
prediction, with R1 to establish task-specific ceilings; D2 hype and publication bias, with R2 to
treat ML-based science results as tentative; D3 inadequate expertise, with R3 interdisciplinary
teams; D4 lack of standardization, with R4 to adopt the common task framework (fixed split, fixed
metric, secret holdout, leaderboard) where possible, while noting that it narrows attention to one
metric; D5 lack of computational reproducibility, with R5 to use platforms like CodeOcean.

## Relevance to Our Crop-Classification Study

This paper gives the revision its framing and its vocabulary. The submitted manuscript reported that
sparse, tree-like models (gradient-boosted trees, `TabNet`) transferred best to the spatially
disjoint holdout tile and that dense temporal networks overfit the training region. The re-analysis
in `experiments/holdout_fill_check/` shows that this ranking came from an asymmetric preprocessing
step, and Kapoor and Narayanan's taxonomy lets us name it precisely.

**The asymmetric step, in the paper's terms.** The holdout tile `34S_20E_259N` is missing the
September observation for 40.6% of pixels (39.3% of fields); neither training tile has any missing
month. That is a train/test distribution difference in the missingness pattern, a form of the L3
situation the paper describes where "even slight shifts in the target distribution can cause
performance estimates to change drastically." The published pipeline then handled that difference
differently by model family. The raw-sequence deep models (`L-TAE`, `L-TAE-S`, `TempCNN`,
`Transformer`, `CNN-BiLSTM`, `TimesNet`, `MiniRocket`) received a zero reflectance before
standardization, a value that never occurs in training. Pixel `TabNet` received the holdout's own
column medians. The xr_fresh classical models computed their features on the shortened series. So
three fill operators were in play, and the two fastest-rising model families on the re-run had faced
the harshest one.

The `TabNet` path is also a mild L1.2 instance: its fill statistic was computed on the holdout set.
It is not label leakage, and it did not inflate `TabNet` in the usual direction (the re-run with
training-tile medians gives 0.616 vs 0.603 to 0.609, a change inside the bootstrap interval), but it
is exactly the kind of test-set preprocessing the taxonomy says to report, and the revised pipeline
removes it by using training-tile medians for every model.

**The ranking inversion and its reversal.** On the 1,352 holdout fields where September was
observed, the published predictions already placed the dense networks first (`CNN-BiLSTM` 0.66,
`TempCNN` 0.65 macro-F1) and pixel `TabNet` fourteenth (0.62). On the 1,063 fields where September
was missing, the dense networks fell to 0.21 to 0.36 while `TabNet` held at 0.58 and `L-TAE-S` at
0.54. Pooled, that gave `TabNet` first at 0.603 and `CNN-BiLSTM` nineteenth at 0.513. With the month
filled from training-tile medians for every model and no retraining, the dense pixel networks gain
0.08 to 0.14 (`CNN-BiLSTM` 0.51 to 0.66, `L-TAE` 0.54 to 0.65, `TempCNN` 0.57 to 0.65, `MiniRocket`
0.50 to 0.61), `TabNet` moves from 0.603 to 0.616, and the xr_fresh classical models gain 0.02 to
0.05. The top six under one protocol are dense pixel temporal models at 0.63 to 0.66; trees,
logistic regression, and `TabNet` sit at 0.59 to 0.62. The revision presents this as a re-analysis:
it names the asymmetric step, re-runs every model under one fill, and reports the before and after
ranking side by side.

**How this differs from the civil war case, and why that matters for how we write it.** In
Muchlinski et al., pooled imputation inflated the flexible model and the correction removed its
advantage. In our case the asymmetric fill deflated the flexible models and the correction restored
their advantage. The direction differs; the structure is the same. A comparative claim about model
families was driven by a preprocessing step that treated the families differently, and the claim did
not survive equal treatment. The paper's observation that high-capacity models "over-fit to spurious
correlations more easily" has a direct counterpart: our dense temporal models were the most
sensitive to an out-of-distribution input value, which is why a zero that `TabNet` never saw cost
`CNN-BiLSTM` 0.14 macro-F1 and `TabNet` almost nothing. The revision should say plainly that the
original sparse-transfers-best conclusion was an artifact of unequal fill, and should cite this
paper for the general principle rather than present the change as a minor correction.

**The spatial-holdout design is the right one, and the paper says so.** L3.3 names spatial bias as
the first example of sampling bias, and the recommended practice is to evaluate on data from the
distribution of interest. Our holdout tile is that design. The lesson of the re-analysis is not that
the holdout was wrong but that a holdout from a different distribution can carry a different
missingness pattern, and that pattern must be handled identically for every model before any ranking
is read off it. The paper's field-level examples (hospital-to-hospital image differences,
borderline-case exclusion) are the same phenomenon at a different scale.

**Documenting each model's fill operator: a model info sheet entry.** The paper's instrument is the
right vehicle for the revision's supplement. For every model in Table 1 we should record, in one
row, the following: the fill operator applied at the missing month (zero, holdout column median,
training-tile median, or NaN-aware feature computation on the shortened series); the dataset from
which any fill statistic was computed (training tiles only, or holdout); the order of fill and
standardization; whether the model was retrained or only re-scored; and whether the re-run
reproduced the published predictions field for field. That last item matters because the paper ties
info sheets to computational reproducibility: the original-file re-run reproduces every published
deep-model prediction exactly except pixel `TabNet`, whose published file (2026-02-10) predates its
saved checkpoints (2026-02-17 and 18) and differs on 221 of 2,415 fields (0.603 vs 0.609); the
classical field features were rebuilt from a current pixel grid and differ on 85 fields; and the
xr_fresh `autocorr_lag_k` features are copies of `absolute_sum_of_changes` because of a
`function_mapping` lookup that returns None. These are exactly the details the paper says reviewers
cannot find without a sheet, and they belong in the supplement with the before/after table.

**Which of the 21 questions we can now answer.** For L1 (clean separation): the fill statistic is
computed on training tiles only; standardization statistics are computed on training tiles only; the
FID-wise split with `random_state=42` is the same across scripts; no SMOTE or resampling touches the
holdout. For L2 (feature legitimacy): the six bands and xr_fresh statistics are available at
prediction time; the September fill value is a training-derived constant, not a holdout-derived one.
For L3 (distribution of interest): the holdout is a spatially disjoint tile; the missingness shift
is documented and handled uniformly; the published per-subset scores (September observed vs missing)
are reported so that readers can see the shift's effect directly.

**Uncertainty reporting.** The paper's second case-study finding, that 9 of 12 papers gave no
significance tests, is a reminder that the revision should report the paired bootstrap intervals
from `score.py` (B = 10,000, seed 42) for every before/after delta, and McNemar p-values for the
changed-field counts, rather than only the point estimates. The holdout fill check already computes
these; the revision should print them.

**Metric choice.** The survey's metric-choice column (accuracy under imbalance) supports our use of
macro-F1 as the primary metric and Kappa and weighted F1 as secondary, since lucerne/medics
dominates the holdout and accuracy alone would hide the minority-class collapse that the zero fill
caused.

## Evaluation Caveats

- **The paper's own evidence base is a survey of reviews, not a systematic sample.** The 294-paper
  figure is a lower bound assembled from 22 reviews found by keyword search across differently named
  pitfalls; the authors say so. It cannot be used to estimate leakage prevalence in remote sensing
  specifically, although remote sensing is represented (reference 24, hyperspectral segmentation).
- **The case study is one subfield with 12 papers.** The reversal of the
  ML-versus-logistic-regression claim is demonstrated in civil war prediction and should not be
  generalized to "ML never beats simple baselines"; the authors cite other fields where ML does
  help.
- **Before/after AUC values are mostly in the figure and supplement.** The main text gives only
  Wang's 0.14 to 0.01 gap and Blair and Sambanis's 0.85 with CI 0.66 to 0.95; the Muchlinski and
  Colaresi corrected AUCs are in Figure 3's bars and supplemental S2, not in the text we extracted.
- **Leakage is defined as producing overoptimism.** Our asymmetric fill produced pessimism for one
  model family and a relative advantage for another. The taxonomy's L1.2 and L3 labels still apply
  to the mechanism (preprocessing that differs between test and train, and between models), but a
  reader applying the paper's definition literally will note that our case is a comparative artifact
  rather than an inflated absolute score.
- **Remote sensing is not the paper's focus.** Spatial autocorrelation, FID-wise splitting, and
  tile-level holdouts are covered only by L3.2 and L3.3 in general terms; the paper does not discuss
  missing observations in satellite time series or imputation of a missing acquisition.
- **Model info sheets are a proposal, not a validated instrument.** The retrospective check on four
  papers is reported in the supplement; there is no prospective evidence that sheets reduce leakage
  in practice, and the authors list verifiability, false assurance, and expertise as limitations.
- **No class-imbalance metric guidance beyond the survey flag.** The paper notes that accuracy under
  imbalance is a pitfall but does not recommend a specific balanced metric.
- **Computational reproducibility of the case study** rests on a CodeOcean capsule whose DOI we did
  not verify separately.

