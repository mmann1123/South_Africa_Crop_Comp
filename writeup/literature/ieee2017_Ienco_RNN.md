# Land Cover Classification via Multitemporal Spatial Data by Deep Recurrent Neural Networks

**Citation:** Dino Ienco, Raffaele Gaetano, Claire Dupaquier, Pierre Maurel (2017). "Land Cover Classification via Multitemporal Spatial Data by Deep Recurrent Neural Networks." *IEEE Geoscience and Remote Sensing Letters*, 14(10), 1685–1689. DOI: `10.1109/lgrs.2017.2728698` (verified via Crossref; the PDF is the arXiv preprint `1704.04055`, whose title omits "Deep" but matches the same work).

## Objectives

The letter asks whether Recurrent Neural Networks (RNNs), specifically Long Short-Term Memory (LSTM) units, can perform land-cover classification on satellite image time series (SITS) better than the then-standard practice of stacking dates and feeding them to Random Forest (RF) or SVM. The authors argue that stacking treats time stamps as independent features and ignores temporal correlation, whereas an LSTM explicitly models the temporal sequence by recursion. A secondary objective is to test the LSTM as a *representation learner*: use the last hidden state (a 512-dimensional vector) as engineered features to feed standard classifiers (`RF(LSTM)`, `SVM(LSTM)`).

## Methods

A single-layer LSTM (512 hidden units) ingests a sequence of per-timestamp feature vectors and feeds a final SoftMax layer for multi-class prediction. Trained 200 epochs, RMSprop, batch size 20, Keras/Theano. Two datasets exercise both granularities:

- **THAU basin (France)** — object-based. A Pléiades VHSR (2 m) time series of only 3 dates; objects from multiresolution segmentation, each described by mean/std of 4 bands + NDVI (10 features/date). 15,196 objects, 11 classes (including `Winter crops`, `Summer crops`, `Vineyards`).
- **REUNION ISLAND** — pixel-based. 23-date annual Landsat-8 (30 m) series, 7 surface reflectances + NDVI/NDWI/BI (10 features/timestamp), cloud-filled by linear interpolation. 37,900 pixels, 9 classes, deliberately rebalanced by random sampling to an "almost balanced ground truth."

Sensor/feature design: optical-only, multispectral reflectance plus vegetation/water/brightness indices; engineered per-date statistics (object case) or interpolated pixel reflectance sequences (pixel case). No SAR. Baselines RF (400 trees, depth 10) and SVM (RBF).

**Evaluation protocol:** 5-fold cross-validation pooled over all objects/pixels within each single study scene. This is **pooled in-region k-fold**, not a spatially disjoint holdout. For the THAU object dataset there is no statement that folds respect spatial separation between segments, and for the REUNION pixel dataset pixels are split randomly — neighboring pixels of the same field can land in both train and test, so spatial autocorrelation can inflate scores. The protocol measures *within-scene, within-year* discrimination only; it tells us nothing about transfer to a disjoint tile, a different year, or a different sensor.

## Key Findings

- LSTM matched or modestly beat RF/SVM on aggregate metrics. THAU: LSTM F-measure 74.63% vs SVM 71.35%, RF 71.58%; Accuracy 75.15%; Kappa 0.69. REUNION: LSTM F-measure 83.56%, but `SVM(LSTM)` won at 84.41% — i.e., the LSTM's biggest contribution was as a feature extractor feeding SVM, not as an end-to-end classifier.
- The headline strength is on **rare and mixed classes**: for THAU classes with few samples (Tree crops, Summer crops, Truck farming) RF/SVM sometimes scored ~0 (missed the class entirely) while the LSTM recovered them. The authors used average and per-class F-measure precisely because the data are imbalanced — a balanced-metric choice worth noting.
- Feeding the LSTM's learned 512-d representation into RF/SVM improved those classifiers over the raw features, supporting the "learned temporal embedding as features" idea.

## Relevance to Our Crop-Classification Study

This is a foundational reference for temporal deep nets on SITS and an early articulation of the inductive-bias argument we invert. Ienco et al. show an LSTM helping most on *minority* classes under *in-region* validation — exactly the regime where our manuscript finds dense temporal nets (CNN-BiLSTM, L-TAE, TempCNN) look strong before a spatially disjoint holdout reorders them. Their best result on the harder pixel dataset comes not from the end-to-end LSTM but from `SVM(LSTM)`, an LSTM-as-feature-extractor handed to a margin-based classifier; this prefigures our finding that sparse, feature-selecting models transfer better and that dense temporal encoders are most useful when their output is consumed downstream rather than trusted directly. The paper also validates two design choices we use: per-field/object aggregation (THAU) as a variance-reduction lever, and reliance on macro/per-class F-measure rather than overall accuracy given a skewed class distribution (our majority class is lucerne/medics).

Anchored to the sparsity/inductive-bias thesis: the LSTM here has no sparsity prior — it is a dense recurrent encoder whose gains evaporate to a tie with SVM on the cleaner pixel set. Our argument is that under spatial covariate shift, the lack of a feature-selecting bottleneck is a liability, not the asset it appears to be under pooled CV.

## Evaluation Caveats

- **In-region 5-fold CV only.** No spatially disjoint tile, no cross-year, no cross-sensor test. Reported numbers are an upper bound on transferable performance; they cannot speak to out-of-sample generalization, which is our manuscript's central axis.
- **Spatial leakage risk.** REUNION uses random pixel splits with no FID-disjoint or spatial-block constraint; THAU folds are over segmented objects with no stated spatial separation. Both admit train/test adjacency.
- **REUNION ground truth was rebalanced by sampling** to "almost balanced," which softens the very imbalance the LSTM is credited with handling and makes the pixel-level F-measure less comparable to operational, skewed settings like ours.
- **Tiny temporal depth in one case** (THAU: 3 dates) limits how much an LSTM's sequence modeling can matter; gains there may reflect representation capacity more than temporal recursion.
- **Silences:** no measurement of training-set-size sensitivity, no spatial transfer, no field-vs-pixel aggregation comparison on the same scene, no calibration or probability outputs. The paper does not test whether the LSTM advantage survives a domain shift — the question our study is built around.
