# A CNN-SVM Study Based on Selected Deep Features for Grapevine Leaves Classification

**Citation:** Koklu, M., Unlersen, M. F., Ozkan, I. A., Aslan, M. F., & Sabanci, K. (2022). A CNN-SVM study based on selected deep features for grapevine leaves classification. *Measurement*, 188, 110425. DOI: 10.1016/j.measurement.2021.110425 (verified via Crossref).
**BibTeX key:** `koklu2022cnn`

## Objectives

The paper develops an automatic image-classification system to distinguish five
species of edible grapevine leaves photographed in the Central Anatolia region of
Turkey. The motivation is economic: edible grapevine leaves vary in market price
and culinary quality by species, and visual species discrimination is difficult even
for experts. The authors' stated contributions are threefold: (1) classify the five
leaf species with a fine-tuned `MobileNetv2` CNN; (2) extract deep features from
the CNN's `Logits` layer and classify them with several SVM kernels (the "CNN-SVM"
hybrid); and (3) apply a feature-selection algorithm to show that a reduced feature
subset can match or exceed the full set.

This is an RGB-leaf-image study, not a satellite or temporal study. It sits in our
literature as a methodological touchstone for the *hybrid pipeline* idea — a deep
network used as a feature extractor feeding a classical, margin-based classifier
(SVM) over a feature-selected subspace — rather than as a directly comparable
crop-mapping result.

## Methods

A custom self-illuminating image-acquisition rig photographed 100 leaves for each
of 5 species (500 originals), against a black background to ease segmentation. Data
augmentation (rotation, scaling, translation) expanded the set to 2,500 images
(500 per class). Three classification systems were compared:

1. **End-to-end CNN.** A `MobileNetv2` pre-trained on ImageNet, fine-tuned on the
   leaf images (input `224x224x3`).
2. **CNN-SVM.** Deep features extracted from `MobileNetv2`'s `Logits` layer (1000
   features) and classified with SVMs using Linear, Quadratic, Cubic, and Gaussian
   kernels.
3. **CNN-SVM with feature selection.** The 1000 `Logits` features were ranked by the
   Chi-Squares method and reduced to 250, then classified with the same SVM kernels.

The best result — **97.60% classification accuracy** — came from the Chi-Squares-
selected feature subset fed to a Cubic SVM. Notably, feature selection *increased*
accuracy while *reducing* dimensionality, the result most relevant to our work.
Hardware: a single laptop GPU (GTX 1050 4 GB).

**Evaluation protocol.** The reported accuracy is **single-scene, random
train/test partition over an augmented image set** — the weakest rung on the
generalization spectrum for our purposes. There is no spatial holdout, no temporal
holdout, and (critically) no separation between a leaf and its augmented copies: a
rotated/scaled/translated version of a training leaf can land in the test fold, so
the 2,500-image pool is not 2,500 independent samples but ~500 leaves × 5 augmented
views. This is the image-domain analogue of the spatial-leakage problem we flag in
satellite work (correlated near-duplicate samples straddling train and test inflate
accuracy). The paper does not state whether the augmented variants of a leaf were
constrained to the same fold. The class balance is exactly even (500 per class), so
class-imbalance artifacts are absent here — but that also means the study tells us
nothing about minority-class behavior under skew, which is central to our
lucerne/medics-dominated problem.

## Key Findings

- A fine-tuned `MobileNetv2` can separate five visually similar grapevine-leaf
  species; the CNN-SVM hybrid and feature-selected variant outperform the bare CNN.
- **Feature selection (Chi-Squares, 1000→250) raised accuracy** to 97.60% with the
  Cubic SVM, demonstrating that a sparse, selected deep-feature subspace can beat
  the full feature vector. The authors frame this as deep features being partially
  redundant/noisy and a margin classifier benefiting from a pruned input space.
- The strongest SVM kernel was Cubic; the pipeline runs on modest hardware.
- The work is positioned against prior leaf-classification literature where CNN
  features beat hand-crafted morphometric features (e.g., 94.88% vs. 66.55% in a
  cited 43-species study).

## Relevance to Our Crop-Classification Study

The conceptual payload aligns with our central thesis even though the domain
(RGB leaf photos) does not. Our manuscript argues that **sparse, feature-selecting
classifiers transfer best** — gradient-boosted trees and TabNet's attentive
sparsemax masks. This paper is an independent, different-domain demonstration that
*a deep network as a feature extractor + an explicit feature-selection step + a
margin classifier (SVM)* outperforms the end-to-end deep model. That is structurally
the same inductive-bias argument: imposing sparsity/selection on top of learned
representations improves the downstream classifier. It is a useful citation for the
"feature selection helps, even on deep features" point and for motivating our
TabNet/tree results, but it must be cited carefully — it shows accuracy *gain on a
held-in random split*, not transfer.

For feature/sensor design, the relevance is limited: this is optical RGB at the
leaf scale, no temporal sequence, no spectral indices, no field/patch granularity.
It does not inform our `B2,B6,B11,B12,EVI,hue` band choice or our temporal-feature
design. Its value is purely methodological (hybrid pipeline + feature selection),
not empirical.

## Evaluation Caveats

- **Cite the protocol, not the 97.60%.** That figure is overall accuracy on a
  random partition of an augmented single-session image set — not comparable to our
  spatial-holdout macro-F1. Do not present it as evidence of generalization.
- **Augmentation leakage risk.** With 500 originals expanded 5× and no stated
  constraint keeping a leaf's augmented copies in one fold, near-duplicate
  correlated samples can straddle train/test, inflating accuracy — the image-domain
  cousin of our field-wise (FID) leakage concern.
- **Balanced classes only.** Exactly 500 images per class means the study reports
  accuracy under perfect balance; it does not probe minority recall, weighted vs.
  macro-F1 divergence, or behavior under the heavy skew that dominates our data.
- **Silences.** No spatial transfer, no cross-session/cross-device robustness, no
  per-class breakdown reported in the extractable text, and only single-session
  acquisition. Compute cost is noted but not benchmarked against alternatives.
- **Metric.** Only overall accuracy is foregrounded; with balanced classes this is
  acceptable internally but offers no balanced-metric (macro-F1/Kappa) anchor for
  cross-study comparison.
