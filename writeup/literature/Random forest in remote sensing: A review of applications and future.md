# Random forest in remote sensing: A review of applications and future directions

**Citation:** Belgiu, M. & Drăguț, L. (2016). *Random forest in remote sensing: A review of applications and future directions.* ISPRS Journal of Photogrammetry and Remote Sensing, 114, 24–31. DOI: `10.1016/j.isprsjprs.2016.01.011` (verified via Crossref).

## Objectives

This is a review article, not an empirical study. Its stated objective is to summarize the use of the Random Forest (`RF`) classifier across remote-sensing classification tasks, with particular attention to (1) parameterization (`Ntree`, `Mtry`), (2) sensitivity to sampling design, training-set size, and noise, (3) the variable-importance (`VI`) machinery and its use for feature-space optimization, and (4) how `RF` compares to other mainstream classifiers (SVM, ANN, decision trees, AdaBoost). It deliberately excludes regression and treats "variable" and "feature" interchangeably. The article is pitched partly as a tutorial for readers with limited ML background, so much of it restates Breiman (2001).

Because this is a review, the per-paper "evaluation protocol" question in our usual template does not apply directly (authoring rule 7). Instead, the load-bearing content for our manuscript is what the review says about *transferability across study areas* and *feature selection* — both of which speak straight to our central argument.

## Methods

No new experiments. The paper synthesizes ~100 prior studies into thematic sections:

- **Ensemble background** — bagging vs. boosting; `RF` as a bagging ensemble of unpruned `CART` trees, each grown on a bootstrap (~2/3 in-bag) sample with `Mtry` features tried per split, the held-out ~1/3 forming the out-of-bag (`OOB`) sample used for an internal error estimate.
- **Parameterization** — `Ntree` (number of trees, commonly 500, the `randomForest` R default; accuracy is largely insensitive to it once errors stabilize) and `Mtry` (features per split, typically √(n_features); the more sensitive parameter).
- **Variable importance** — Mean Decrease in Gini (`MDG`) and Mean Decrease in Accuracy (`MDA`, permutation-based on `OOB`); most reviewed studies use `MDA`.
- **Feature selection** — filter (PCA, ICA, MNF, stepwise LDA), embedded (`MDA`-based pre-selection, `varSelRF` backward elimination), and wrapper (`Boruta`) methods, with the observation that pre-filtering rarely beats `RF`'s embedded selection.
- **Comparisons** — narrative meta-comparison of accuracy, training time, and stability against SVM, ANN, LDA, decision trees, and boosting ensembles.

**Evaluation protocol (of the reviewed corpus, as characterized by the authors):** The review notes that most reviewed studies measure accuracy with internal `OOB` error or with held-out validation samples drawn from the *same* scene/study area. The review does **not** systematically audit whether reviewed studies used spatially disjoint train/test splits, and it treats `OOB` error as a "reliable measure of classification accuracy" (citing Lawrence et al. 2006; Zhong et al. 2014) while flagging that this claim "needs to be further tested." This is exactly the in-region-validation assumption our manuscript exists to interrogate: `OOB` error is computed on bootstrap-excluded samples drawn from the *same* spatially autocorrelated field population, so it measures resubstitution-like generalization within a region, not transfer to a new region.

## Key Findings

- **`RF` handles high dimensionality and multicollinearity well, is fast, and is "insensitive to overfitting"** — but is **sensitive to sampling design** and to **imbalanced training data**. Dalponte et al. (2013) and Millard & Richardson (2015) report `RF` "fails to cope with imbalanced training data and tends to favor the most representative classes." This is the class-imbalance artifact our crop dataset embodies (lucerne/medics majority): overall accuracy can look strong while minority-crop recall collapses.
- **Sampling-design results are contradictory.** Some studies find `RF` robust to mislabeled/imbalanced data (Mellor et al. 2015); others find it sensitive to spatial autocorrelation of training classes and to class proportions. The review recommends per-study sensitivity analysis rather than assuming robustness.
- **`Mtry` matters more than `Ntree`.** `Ntree`=500 is an acceptable default; several studies found accuracy plateaus well below that (e.g., 70 trees for SAR oil-spill mapping).
- **`RF` ≈ SVM in accuracy**, with `RF` slightly ahead on high-dimensional/hyperspectral input, faster to train, fewer parameters, and more stable; SVM more sensitive to feature selection and harder to tune. `RF` outperforms single decision trees, LDA, BHC, and ANN. Versus AdaBoost the evidence is mixed (and confounded by different base learners, C5.0 vs. CART).
- **Variable importance is the most exploited "extra" capability** — used to reduce hyperspectral dimensionality, rank multi-source/ancillary layers, and pick the most discriminative season for a target class. Embedded `MDA` selection generally matches or beats external filters.
- **Critically for us — Section 6.1, "Stability of the RF classifier":** the review explicitly reports that **overall accuracy decreases when `RF` is trained on one study area and applied to another** (Vetrivel et al. 2015), and that Juel et al. (2015) found an `RF` vegetation-mapping model **"was not transferable to new areas."** The proposed remedies are (a) hybrid models that encode object semantics and (b) **spectral indices, "which have been shown to be more stable when applied to new study areas."**
- **Future directions** the authors flag as under-studied: robustness of selected feature space across changing sample sizes/noise, `OBIA`+`RF` coupling, validating `OOB` error as an accuracy proxy, and proximity-based outlier detection in training samples.

## Relevance to Our Crop-Classification Study

This review is one of the clearest pieces of prior support for our manuscript's thesis, despite predating it by a decade.

1. **`RF` non-transferability is documented here, not just in our results.** Section 6.1 states plainly that `RF` accuracy "decreases when the algorithm is trained on different study areas" and cites a model that "was not transferable to new areas." Our finding — that in-region field-wise k-fold misranks models and a spatially disjoint holdout tile (`34S_20E_259N`) reorders them — is the systematic, quantified version of an instability the `RF` literature already acknowledged anecdotally. We can cite Belgiu & Drăguț (2016) §6.1 to establish that spatial non-transferability of tree ensembles was a known-but-unsystematized concern.

2. **The "use spectral indices for stability" recommendation aligns with our feature design.** The review notes spectral indices generalize across areas better than raw bands. We use `EVI` and `hue` alongside `B2`, `B6`, `B11`, `B12`, and engineered xr_fresh time-series statistics — exactly the kind of derived, scene-invariant features the review predicts will transfer better. This is a mechanism-level argument for why our gradient-boosted and sparse models hold up out-of-region.

3. **Variable importance / feature selection is the through-line of our argument.** The review's core message — that `RF`'s embedded `VI`-based feature selection is its most valuable asset for taming high-dimensional remote-sensing input — is the same inductive bias that, in our manuscript, separates the models that transfer (gradient-boosted trees, TabNet — sparse, feature-selecting) from the dense temporal/patch nets that overfit the training region. Belgiu & Drăguț supply the classical-side rationale for why feature-selecting tree models should generalize.

4. **Class imbalance is named explicitly.** The review's citation that `RF` "tends to favor the most representative classes" under imbalance is directly applicable: our lucerne/medics-dominated dataset is why we report macro-F1 and Cohen's Kappa rather than overall/weighted accuracy. We can cite this as motivation for our balanced-metric choice.

5. **Parameterization defaults.** Where we run `RF` baselines (pixel-level classical comparators), the review justifies `Ntree`≈500 and `Mtry`≈√n as defensible defaults, and warns that `OOB` error is *not* a substitute for an independent (and ideally spatially disjoint) test set — supporting our decision to score on the holdout tile rather than on internal `OOB`/CV.

## Evaluation Caveats

- **It is a review, not a benchmark.** No headline F1/OA to cite as a comparator, and no controlled experiment of its own; all numeric claims are second-hand and span heterogeneous sensors, classes, and protocols.
- **`OOB`-error-as-accuracy is endorsed with only a soft caveat.** The review repeats the claim that `OOB` error is a reliable accuracy measure. For our purposes this is precisely the in-region validation conflation to flag: `OOB` samples come from the same spatially autocorrelated population as the training data, so `OOB` error overstates transfer. A reader taking the review at face value would *not* anticipate the spatial-holdout collapse our paper documents.
- **No field-wise (`FID`-disjoint) splitting discussion.** The review predates the now-standard concern about pixels from one field appearing in both train and test. Spatial leakage at the field/pixel level is not addressed; "sensitivity to spatial autocorrelation of training classes" is the closest it comes, and it is framed as a sampling-design nuisance rather than a validation-integrity problem.
- **Pre-deep-learning vintage (2016).** No CNN/RNN/transformer/temporal-attention comparisons; the "other classifiers" are SVM, ANN, LDA, AdaBoost. The review cannot speak to the dense-deep-net overfitting half of our story, only to the tree-ensemble half.
- **Cross-year and computational-cost transfer are absent.** Stability is discussed only across study areas, not across seasons/years, and runtime is mentioned qualitatively ("`RF` is fast") without systematic profiling.
- **Contradictory sampling-design evidence is left unresolved**, so the review offers guidance ("run a sensitivity analysis") rather than a settled recommendation on handling imbalance — the practitioner is left to decide.
