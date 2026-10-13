# Bake off redux: a review and experimental evaluation of recent time series classification algorithms

**Citation:** Matthew Middlehurst, Patrick Schäfer, and Anthony Bagnall (2024).
*Data Mining and Knowledge Discovery*, 38(4):1958-2031. Published online 19 April 2024.
DOI: 10.1007/s10618-024-01022-1 (from Crossref title match; verified against
`https://api.crossref.org/works/10.1007/s10618-024-01022-1`, title, authors, journal,
volume and pages match). Preprint: arXiv:2304.13029v3 (8 May 2024), which is the
version in our PDF.
**Note:** The PDF is the arXiv v3 preprint and carries no DOI; the DOI above comes from a
Crossref title search. A separate correction notice exists (DOI 10.1007/s10618-024-01040-z,
*DMKD* 38(6):4236-4237); cite the main article, not the correction. Not yet present in
`writeup/literature/references.bib`, `README.md`, or `writeup/tgrs-article.tex`.
**Domain:** General time-series classification (TSC) benchmark on the UCR archive.
Not remote sensing, not crop mapping. Briefed here as methodological background for the
recent-baseline additions in `experiments/recent_baselines/` (reviewer R2-Q3a).

## Objectives

This is the sequel to the 2017 "great time series classification bake off"
(Bagnall et al., *DMKD* 31(3):606-660), which compared 18 TSC algorithms on 85 UCR
datasets and found only nine significantly better than the 1-NN Dynamic Time Warping
(`1NN-DTW`) and Rotation Forest benchmarks. Six years on, the authors set out to
answer four questions.

First, how has each algorithm family advanced since 2017? They re-describe the field
with an extended taxonomy of eight categories, adding three new ones (feature based,
convolution based, deep learning) to the original distance, interval, shapelet,
dictionary and hybrid groups.

Second, within each category, do the new algorithms beat the previous best of
category, under a common, reproducible protocol on the 112 equal-length univariate
problems of the 2019 UCR archive?

Third, do the conclusions hold on fresh data? They release 30 new univariate datasets
(donated or reformatted) specifically to counter the risk of implicit overfitting to a
UCR archive that every algorithm in the comparison was developed against.

Fourth, what drives performance (series length, train-set size, number of classes,
problem type, design choices such as dilation, differencing, ensembling), and what is
the accuracy versus train-time trade-off for practitioners?

The stated audience is both TSC researchers and practitioners choosing a classifier
for a new problem, and the paper closes with an explicit default recommendation.

## Methods

**Scope.** Univariate TSC only. Multivariate TSC (MTSC) is explicitly excluded because
of computational cost and because only a subset of the algorithms has a multivariate
version. Unequal-length and missing-value series are excluded from the main 112 and
handled only by simple padding or case deletion in the new-dataset set.

**Algorithm inclusion criteria.** An algorithm must have been published post-2017 in a
high-quality venue (or extend one that was), must have been evaluated on UCR/UEA data
(or a justified subset), and must have source code adaptable to Java or Python. Almost
all implementations are run through the `aeon` toolkit and the `tsml-eval` harness.
Three algorithms are Java-only.

**Algorithms compared, by category.** The paper benchmarks about 40 classifiers. The
best of category are what matter for us:

| Category | Members evaluated | Best of category |
|---|---|---|
| Distance based | `1NN-DTW`, Elastic Ensemble (EE), Proximity Forest (PF), ShapeDTW, GRAIL | PF |
| Feature based | Catch22, TSFresh (+FRESH selection, RF), Generalised Signatures, FreshPRINCE | FreshPRINCE |
| Interval based | TSF, RISE, STSF, R-STSF, CIF, DrCIF, QUANT | QUANT (tied with DrCIF, R-STSF) |
| Shapelet based | STC, RDST, RSF, MrSQM | RDST |
| Dictionary based | BOSS, cBOSS, WEASEL 1.0, TDE, WEASEL 2.0 | WEASEL 2.0 |
| Convolution based | ROCKET, MiniROCKET, MultiROCKET, Arsenal, Hydra, MultiROCKET-Hydra | MultiROCKET-Hydra (MR-Hydra) |
| Deep learning | CNN, ResNet, InceptionTime, H-InceptionTime, LITETime | H-InceptionTime |
| Hybrid | HIVE-COTE 1.0 (HC1), HIVE-COTE 2.0 (HC2), TS-CHIEF, RIST | HC2 |

Some design details matter for how we might use these as baselines.

ROCKET draws thousands of random dilated convolution kernels (length in {7, 9, 11}),
pools each activation map by max and proportion of positive values (`PPV`), and feeds
the resulting 20k features to a cross-validated ridge classifier. It is equivalent to a
single-layer CNN with frozen random kernels. MiniROCKET makes it almost deterministic
(kernel length fixed at 9, two weight values, PPV only, about 10k features) and an order
of magnitude faster with no significant accuracy loss. MultiROCKET adds first-order
differences and three more pooling operators (MPV, MIPV, LSPV), about 50k features.
Hydra groups random kernels into dictionaries and counts "best-matching kernel" hits;
MR-Hydra concatenates Hydra and MultiROCKET features before the ridge classifier.

QUANT uses a single feature type, quantiles, over fixed dyadic intervals of the raw
series, its first and second differences, and its Fourier coefficients, then fits an
Extra Trees classifier. It is the fastest strong method in the study.

FreshPRINCE is the full TSFresh feature set (about 800 summary statistics, no FRESH
feature selection) followed by a Rotation Forest. This is the closest conceptual
analogue to our `xr_fresh` plus gradient-boosted-tree pipeline.

InceptionTime is an ensemble of five Inception-module networks with residual
connections and global average pooling; H-InceptionTime adds hand-crafted trend and
peak filters in parallel with the first module; LITETime is a lightweight depthwise
separable variant that matches InceptionTime at a fraction of the cost.

HC2 is a heterogeneous meta-ensemble of four modules (STC shapelets, TDE dictionary,
DrCIF intervals, and the Arsenal, an ensemble of ROCKET classifiers), weighted by
CAWPE using out-of-bag accuracy estimates.

**Metrics.** Primary metric is test accuracy. Also reported: balanced accuracy
(explicitly to diagnose class-imbalance problems), weighted F1, AUROC (multiclass AUROC
weighted by class frequency in the training data), and negative log-likelihood (NLL)
for probability calibration. Significance is assessed by critical difference diagrams
built from pairwise one-sided Wilcoxon signed-rank tests with Holm correction (instead of
the Nemenyi post-hoc test), supplemented by pairwise scatter plots, violin plots of
deviation from the median classifier, and the Multiple Comparison Matrix (MCM) of
Ismail-Fawaz et al. (2023), because CD diagrams can be unstable and depend on which
comparates are included.

**Evaluation protocol:** Each of the 112 UCR datasets (and the 30 new ones) is
evaluated over 30 stratified train/test resamples, seeded by resample ID, with resample
0 being the archive's original split; accuracy is averaged across resamples and then
ranked across datasets. This sits outside the spatial spectrum this manuscript is built
around, because UCR problems have no geography: each dataset is an i.i.d.-style pool of
labelled series from one source, and both train and test resamples are drawn from that
same pool. In our terms it is closest to repeated random hold-out within a single
distribution, i.e. in-distribution fit, not transfer. Generalization is assessed across
*problems* (ranks over 112 or 142 datasets, plus 30 unseen datasets as a guard against
algorithm-design overfitting to the archive), never across a domain shift within a
problem. No spatial, temporal (cross-year), or cross-sensor holdout exists or is
attempted. The paper also notes that the GRAIL authors' code let the test set influence
an unsupervised clustering step, which the bake-off authors patched out, and it
criticises deep-learning TSC papers for selecting models on test data (choosing the
best epoch by test accuracy). Both are leakage concerns analogous to, but distinct from,
the spatial leakage we guard against.

**Compute.** All non-deep models were timed on the same CPU (Intel Xeon Gold 5220R);
H-InceptionTime ran on two GPU types and is omitted from timing comparisons.

**Reproducibility.** All results files, code, and parameter settings are public via
`aeon`, `tsml-eval`, and timeseriesclassification.com. The arXiv v3 also reproduces the
full reviewer reports and responses from a rejected *Machine Learning* submission and
the accepted *DMKD* submission.

## Key Findings

**Overall winners.** On the 112 UCR problems the best-of-category ranking by accuracy
is HC2 (0.891), MR-Hydra (0.884), RDST (0.876), H-InceptionTime (0.874), WEASEL 2.0
(0.874), QUANT (0.867), FreshPRINCE (0.855), Proximity Forest (0.837), and `1NN-DTW`
(0.756). HC2 is about 0.5 points more accurate than MR-Hydra, over 6 points better
than PF, and over 12 points better than `1NN-DTW`. HC2 and MR-Hydra are not
significantly different from each other in the MCM pairwise test (p = 0.10), but both
are significantly better than all others. Balanced accuracy and weighted F1 give the
same top two.

**New datasets.** On the 30 new datasets the top accuracy clique widens to MR-Hydra,
HC2, RDST, QUANT and FreshPRINCE, and MR-Hydra ranks first on average. On the combined
142 problems HC2 and MR-Hydra are in a clique of their own. The authors read this as
HC2's edge on UCR possibly reflecting a smaller sample, different data, or embedded
design bias from years of tuning against UCR. This is the paper's own acknowledgement
that rankings on a familiar benchmark can shift on unseen problems.

**Recommended defaults.** In the absence of prior information, HC2 and MR-Hydra are
the recommended starting point for a new TSC problem. MR-Hydra is "on average as
accurate as HC2 but orders of magnitude faster" (1.33 h total train time on 142
problems vs 263.89 h for HC2). HC2 is preferred for small training sets, long series,
balanced classes, or when calibrated probabilities and rankings are needed (MR-Hydra,
RDST and WEASEL 2.0 use ridge classifiers that output hard 0/1 predictions, so their
AUROC and NLL are poor). QUANT is singled out as a sensible default for exploratory work
and very large problems because it is about an order of magnitude faster again (0.22 h
total) while staying competitive. The authors stress this is not a universal
prescription: no algorithm dominates every problem type.

**Imbalance and data size.** HC2 does worse than MR-Hydra on imbalanced data and on
larger training sets; MR-Hydra improves in relative rank as training size grows
(rank 1 in the 300-699 and 700+ bins). H-InceptionTime also improves relative to the
others as the training set grows, which the authors offer as one reason deep learning
underperforms on the comparatively small UCR problems.

**Deep learning.** No deep model is an "AlexNet for TSC". Within the deep category
CNN is no better than `1NN-DTW`, ResNet beats CNN, InceptionTime beats ResNet,
H-InceptionTime gives a small significant gain over InceptionTime, and LITETime matches
InceptionTime far more cheaply. H-InceptionTime is competitive on average but has the
widest spread of deviations from the median classifier: it "often performs terribly,"
which drags its average down. The authors attribute the lack of a deep-learning
breakthrough mainly to the absence of a common underlying structure across TSC problems
for networks to exploit. They also report that, after surveying recent deep TSC papers
(Table 11 lists 17, including several Transformers), they found none that could
realistically claim to beat InceptionTime or H-InceptionTime, and they list recurring
problems in that literature: model selection on test data, missing code, unexplained
dataset subsets, MTSC-only evaluation, and weak straw-man baselines.

**Feature-based pipelines.** All four feature pipelines significantly beat `1NN-DTW`.
Using the full TSFresh set without FRESH feature selection and with Rotation Forest
instead of Random Forest improved accuracy by more than 0.05, reinforcing earlier
evidence that Rotation Forest is strong for continuous features. FreshPRINCE is the
best feature-based method but only seventh of the eight best-of-category classifiers,
and it has a wide accuracy spread across problems.

**Design factors.** One-factor ANOVAs over design choices find significant benefit from
dilation and from using differenced series, but no significant effect of ensembling,
frequency-domain features, or discretisation (with the caveat that test assumptions
are not met). Pipelines that combine a large random or summary transform with a linear
ridge classifier are as competitive as large ensembles and scale better, at the cost of
probability estimates.

**Scalability.** Apart from QUANT, none of the methods scale well to large problems,
especially in memory; the authors note they are building larger problems that none of
the classifiers could fit in reasonable time or memory.

## Relevance to Our Crop-Classification Study

This is a review and benchmark from a different field, so per rule 7 the relevance is
methodological: which baselines a reviewer will expect, how to justify the ones we
chose, and how its protocol contrasts with ours.

**What it recommends as strong baselines.** The paper names three tiers. As
state-of-the-art defaults, MultiROCKET-Hydra and HIVE-COTE 2.0. As the fast, strong
option, QUANT (and the ROCKET family generally: ROCKET, MiniROCKET, MultiROCKET). As the
strongest deep learner, H-InceptionTime / InceptionTime (and LITETime as a cheap
equivalent). If our paper is criticised for omitting "modern time-series classifiers,"
this is the citation that defines the modern frontier for general TSC, and it states in
plain terms that recent Transformer-style TSC models have not been shown to beat
InceptionTime. That supports reporting a ROCKET-family model (MiniROCKET or
MultiROCKET) as the most-cited fast, strong non-deep TSC baseline, with InceptionTime as
the reference deep TSC baseline, ahead of more recent general-purpose architectures.

**Bearing on the TimesNet baseline.** The TimesNet run in
`experiments/recent_baselines/train_timesnet.py` responds to reviewer R2-Q3a. The bake
off does not evaluate TimesNet (it postdates the cut-off and is a general time-series
model evaluated mainly on the multivariate UEA archive). It does give us a principled
reason to describe TimesNet as a recent general-purpose deep TSC model rather than an
established TSC state of the art. The bake off's own default recommendations would be
MiniROCKET or MultiROCKET-Hydra and InceptionTime. If reviewers press for the
established TSC SOTA, adding a MiniROCKET or MultiROCKET-Hydra run (both in `aeon`, both
support multivariate input) is the cheapest and most defensible addition. HC2 is
probably not worth running: about 200 times slower than MR-Hydra in this study, and
our pixel-level training set is orders of magnitude larger than any UCR problem.

**A structural mismatch to flag before running ROCKET-family models.** Our inputs are
short multivariate series: 6 bands (`B2`, `B6`, `B11`, `B12`, `EVI`, `hue`) over 10
monthly steps (months 05 and 06 excluded), i.e. 60 values per pixel. The bake off is
univariate only, and its series are mostly 24 to several thousand steps long (only one
of the 30 new datasets, MelbournePedestrian, is as short as 24). ROCKET kernels have
length 7 to 11 with exponentially sampled dilation, and MiniROCKET fixes length 9, so on
a 10-step series almost all dilations collapse and the transform reduces to a handful
of near-global features. QUANT's dyadic interval pyramid is similarly shallow at length
10. Any result we report for these models should note that their published strengths
come from long univariate series and do not automatically carry over.

**Protocol contrast, and why it matters for our argument.** The bake off measures
accuracy over 30 stratified random resamples of each dataset's pooled train and test
data. That is in-distribution evaluation, the analogue of our in-region protocol,
not of our spatially disjoint holdout tile (`34S_20E_259N`). Its rankings therefore
answer "which algorithm fits a new i.i.d. problem best on average," not "which
algorithm survives a domain shift." Our central result is that rankings measured
in-distribution invert under spatial transfer: dense temporal nets lead in-region
(L-TAE and Transformer 0.78 macro-F1) and fall to 0.58 on the holdout, while TabNet and
L-TAE-S lead under transfer (0.60). The bake off's rankings are thus a statement about
in-distribution accuracy and should not be read as a prediction of transfer rankings.
Two elements of the paper support this framing. The authors themselves observe that
HC2's lead on the familiar UCR archive shrinks on 30 new problems, attributing it in
part to embedded design bias, which is a benchmark-level version of the validation
artifact our paper describes. And the arXiv v3 appendix reproduces a reviewer arguing
that apparent TSC progress may be overfitting to UCR or that UCR is a poor proxy for
real problems; the lead author's reply defends UCR but concedes that benchmark rankings
are "only a starting point." Cite this as reviewer commentary, not as the authors'
finding.

**Imbalance and calibration.** The paper's use of balanced accuracy to diagnose
imbalance parallels our macro-F1 and Kappa reporting. Its finding that the
ridge-classifier family (ROCKET, MR-Hydra, WEASEL 2.0, RDST) gives hard predictions with
poor AUROC and NLL is directly relevant to us because the AI4FoodSecurity official
metric is field-level cross-entropy (our "Xent" column). If we add a ROCKET-family
baseline with `RidgeClassifierCV`, it cannot produce meaningful cross-entropy without
swapping in logistic regression or a calibration step; the paper itself notes logistic
regression as the suggested replacement on larger datasets. The finding that MR-Hydra
beats HC2 on imbalanced data is a mild argument for the ROCKET family over HIVE-COTE in
our majority-class-dominated setting.

**Feature engineering versus raw sequences.** FreshPRINCE, the TSFresh pipeline, is
the closest published analogue to our `xr_fresh` plus tree pipeline, and it lands in the
lower-middle of the best-of-category ranking under in-distribution evaluation. That is
consistent with our reviewer 4.2 experiment, where raw reflectance roughly matches
`xr_fresh` features under transfer (XGBoost ST macro-F1 0.59 vs 0.56): engineered
summary features are competitive but not the accuracy frontier. Its result that using
the full TSFresh feature set beat FRESH-selected features is worth noting alongside our
sparse-feature-selection thesis. The two are not in conflict, because our claim concerns
selection learned inside the model (tree splits, sparsemax masks) rather than a
univariate hypothesis-test filter applied before training.

**Sparsity and inductive bias.** The paper's design-factor analysis shows that
ensembling per se did not significantly affect ranks, and that random-transform plus
regularised linear pipelines match big ensembles. This parallels our ablation finding
that the gain is architectural and aggregation-driven, not an ensembling effect, and
fits the broader point that strong TSC models over-produce features and let a
regularised selector choose among them, conceptually close to the sparse, axis-aligned
selection we argue drives transfer. The authors' explanation for weak deep learning
(no shared structure across TSC problems for networks to exploit) is a useful
in-distribution counterpart to our explanation of dense nets failing to transfer.

**Suggested manuscript use.** One or two sentences in the Introduction or Related Work
(or in the response to reviewers) of the form: "In the general TSC literature, large
benchmarks on the UCR archive identify the ROCKET family and HIVE-COTE 2.0 as the
strongest classifiers and InceptionTime as the strongest deep learner, with no recent
deep architecture shown to surpass it [Middlehurst et al., 2024]; these benchmarks,
however, use repeated random resampling within each dataset and therefore measure
in-distribution accuracy rather than transfer to new regions." This justifies why our
baseline set includes TempCNN and L-TAE (the remote-sensing analogues of convolutional
and attention TSC models) plus a recent general TSC model, and frames any added
ROCKET-family result appropriately.

## Evaluation Caveats

The entire study is univariate. Our problem is multivariate (6 channels by 10 steps),
and the paper explicitly declines to rank MTSC methods, so its rankings cannot be
transferred to our setting without that caveat.

The evaluation is repeated random stratified resampling within each dataset. There is
no domain shift of any kind (spatial, temporal, sensor), so none of its accuracy
figures bear on transfer. The 30 new datasets test generalization of *algorithm design*
across problems, which is a different axis from generalization of a *fitted model*
across space. Its absolute accuracies (0.75-0.89) are averages over unrelated problems
and are not comparable to our holdout macro-F1.

Series lengths and sample sizes differ sharply from ours. UCR problems are mostly long
(hundreds to thousands of steps) and small (tens to a few thousand cases), while ours
are very short (10 steps) and large at the pixel level, with field-level dependence
(pixels within a field are autocorrelated, so the effective sample size is closer to the
number of fields). Rankings by series length and training size (Tables 15-16) suggest
relative performance shifts with both, so extrapolation to short, large, clustered data
is uncertain.

Spatial leakage is not applicable in the usual sense, but the analogous concerns are
present: the paper notes that the GRAIL reference code let the test set inform
clustering, and criticises deep TSC papers for epoch selection on test accuracy. Both
reinforce our practice of field-wise (FID) splitting with validation-only early
stopping.

Class imbalance is addressed only through balanced accuracy and weighted F1 averaged
over datasets; there is no per-class or minority-crop-style analysis, and no macro-F1.
The imbalance comparison between HC2 and MR-Hydra is an aggregate statement over
datasets.

Compute comparisons exclude the deep model (different GPUs) and reflect specific
implementations (for example, TDE inside HC2 is not `numba`-optimised), so the HC2
versus MR-Hydra time gap partly reflects engineering, not just algorithm design.

Hyperparameters are mostly library defaults rather than tuned per dataset, which is
fair across methods but means published accuracies are not necessarily each method's
ceiling. Deep-learning coverage is restricted to CNN-based models (no RNN/LSTM or
Transformer was benchmarked), so the claim that no recent deep model beats
InceptionTime rests on a literature survey, not on head-to-head runs.

When citing, use the *DMKD* version (DOI 10.1007/s10618-024-01022-1) and attribute
recommendations to the authors (HC2 and MR-Hydra as defaults, QUANT for fast or large
problems, H-InceptionTime as best deep learner). Do not cite the reviewer appendix as
the paper's finding, and do not present its UCR rankings as evidence about transfer.
