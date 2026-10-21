# Exploring Self-Attention for Crop-type Classification Explainability

**Citation:** Ivica Obadic, Ribana Roscher, Dario Augusto Borges Oliveira, and Xiao Xiang Zhu
(2022, revised 2025). arXiv preprint `arXiv:2210.13167` (v2, 20 Apr 2025). DOI:
`10.48550/arXiv.2210.13167` **(unverified on Crossref: `api.crossref.org/works/10.48550/arXiv.2210.13167`
returns 404 because arXiv DOIs are registered with DataCite, not Crossref; the DataCite record
`api.datacite.org/dois/10.48550/arXiv.2210.13167` resolves to the identical title and year 2022).** A
Crossref title search for "Exploring Self-Attention for Crop-type Classification Explainability" and
an author-plus-title search for Obadic returned no journal version as of 2026-10-21; cite as an arXiv
preprint. The manuscript's bibliography entry `obadic2022` (arXiv:2210.13167) is consistent with this.
Code: `github.com/IvicaObadic/crop-type-classification-explainability`. Note the full title includes
"Explainability", which the PDF filename drops.

## Objectives

The paper asks whether the attention weights of a standard transformer encoder trained for crop-type
classification are a faithful explanation of the model, and whether they reveal agronomically
meaningful phenological events. Three stated contributions: (i) identify the key dates and
phenological events the model relies on by relating sparse attention patterns to NDVI-based domain
knowledge; (ii) test quantitatively, by removing attended dates and retraining, whether those dates
are actually critical for crop disambiguation; (iii) a sensitivity analysis that removes one crop
class at a time from training and measures how the attention distribution of the remaining classes
shifts, to learn whether attention characterizes each crop's phenology or only the contrasts needed
to separate the classes present. The motivating context is the EU Common Agricultural Policy's use of
Sentinel-based classifiers and the need to connect model decisions to agricultural knowledge.

## Methods

**Model and data.** The transformer encoder of Rußwurm and Körner (2020) (positional encoding, linear
input projection, encoder layers, max-pool over time, dense classifier) trained on the BavarianCrops
dataset from the same paper: parcel-mean Sentinel-2 time series over 2018, 13 bands, from the three
Bavarian regions (Hollfeld, Krumbach, Bavarian Forest), in the 12-class land-cover version. The five
most frequent crops cover 89 % of parcels. The authors reuse Rußwurm and Körner's dataset split and
training procedure, substitute focal loss for class imbalance, right-pad sequences to a fixed length
of 144 (the maximum number of observations per parcel), and use a positional encoding based on
acquisition date rather than sequence index. After a small hyperparameter search (Appendix A) the
selected model has 1 encoder layer, 1 attention head, embedding dimension 128, mean F1 0.64 and
average class accuracy 0.62 on the test set, which the authors say matches Rußwurm and Körner. The
confusion matrix (Fig. 2) shows good accuracy on frequent crops and on rapeseed, low accuracy on some
rare crops. All explanation experiments use a fixed seed of 0 so that attention changes can be
attributed to the data manipulation rather than to initialization.

**Explainability module (Sec. 4.2).**
- Date importance for a parcel: column mean of the `T x T` attention matrix (Eq. 4), justified by the
  observation that attention is sparse, with "the entire attention being assigned only to a few
  columns" so that outputs are built from a handful of attended dates.
- Date importance per crop: average over test parcels of that crop (Eq. 5); overall key dates: average
  over all test parcels (Eq. 6).
- Phenological events: attended dates are related to parcel imagery (B11/B8/B2 composites) and to the
  average NDVI per crop (threshold 0.4 for sparse vegetation).

**Attention evaluation module (Sec. 4.3).**
- *Is attention an explanation?* Following Hooker et al.'s remove-and-retrain benchmark, dates are
  ranked by Eq. 6; new datasets are built with the top-`t` dates removed (`t` in {1, 3, 5, 10, 15, ...,
  T}) or with *only* the top-`t` dates kept; the model is retrained several times with different
  seeds on each and class accuracy is compared with the full-data reference.
- *Can attention describe crop phenology?* For each of the three frequent crops (grassland, winter
  barley, corn), a new dataset is built with that crop occluded (removed), the model is retrained from
  scratch, and the change in per-date importance for every remaining crop is computed (Eq. 8).

**Evaluation protocol.** Everything is measured on the test partition of the Rußwurm and Körner
block-wise split within the three Bavarian regions (4.5 km blocks, 500 m margins, random 4:1:1
assignment, all regions in all partitions). No other region, tile, year, or sensor is used. The
classification accuracy itself is incidental to the paper; the measured quantities are attention
statistics and the accuracy of retrained models under date or class removal. **The only experimental
factor that is varied is the set of crop classes in the training data (and, in the
remove-and-retrain test, the set of dates). Region is held fixed throughout.** This is the point that
matters for how our manuscript cites the paper (see Relevance).

## Key Findings

**Attention is sparse and crop-specific in time (Fig. 3).** Winter barley is attended almost
exclusively at the beginning of July; corn at several dates in late spring (May); grassland
throughout spring and summer; outside spring and summer attention is near zero for every crop. The
top-3 key dates overall are 07 May, 01 July, and 03 July.

**The attended dates coincide with discriminative phenological events (Figs. 4–5).** On 01 and 03
July the highly attended winter-barley parcels are brown (harvested, NDVI below 0.4) while the other
crops are green; on 07 May the attended corn parcels are bare soil (sowing/emergence) while winter
cereals and grassland are at peak greenness. Attention therefore lands on dates where *one crop's
NDVI differs from all the others*, i.e. on contrasts, which the authors describe as harvesting
(winter barley) and growing (corn) events.

**Remove-and-retrain (Fig. 6).** Removing the five most-attended dates and retraining drops class
accuracy by about 5 points relative to the reference model; a further 2-point drop appears after the
25th removed date; yet after 25 removals the retrained model still reaches about 85 % of the
reference accuracy, and accuracy stays high as more dates are removed. Conversely, training on *only*
the top-3 dates gives the sharpest accuracy rise, and the top-15 dates are enough to match the
reference. The authors conclude that attention identifies "one sparse set of dates crucial for crop
disambiguation" but that "multiple sparse sets of critical dates" exist, so attention is "an
explanation rather than the explanation" (quoting Wiegreffe and Pinter).

**Class-occlusion sensitivity (Fig. 7, Table B.2), the paper's headline limitation.** Removing
grassland from the training set "drastically changes the resulting attention weights for the other
crops": winter barley's attention moves from early July (harvest) to late April and May. Removing
winter barley or corn also changes the attention of the remaining crops, less dramatically. The
retrained models' scores are: full dataset F1 0.64 / class accuracy 0.62; grassland occluded 0.67 /
0.66; winter barley occluded 0.59 / 0.58; corn occluded 0.62 / 0.60. The authors' conclusion: "the
interpretation of the attention weights is conditional to the classes in the dataset and focuses on
events that help to disambiguate them"; the model "does not comprehensively characterize crop-specific
phenological cycles" and "does not utilize this information to disambiguate crops based on their
specific vegetation dynamics throughout the vegetation season."

**Preprocessing note (Appendix A).** Right padding and random subsampling of observations give
similar scores (right padding slightly higher); weekly averaging of observations is "significantly"
worse. Different hyperparameter configurations yield near-identical F1 (0.67 in all top-3 rows of
Table A.1).

**Future work.** A multi-layer encoder with local-window attention so that events spanning
consecutive observations can be captured, which the authors expect would let the model learn
crop-specific temporal evolution rather than single-date contrasts.

**What is not measured.** Any change of region, tile, year, or sensor; whether attention patterns
differ between the three Bavarian regions inside the dataset (they are pooled); transfer accuracy of
any kind; comparison with non-attention models (gradient attribution, trees); per-class F1 under
occlusion beyond the mean; the effect of class *frequency* as opposed to class *presence*; robustness
to missing or corrupted observations.

### Numbers to cite

Retrained-model scores under class occlusion (Table B.2, seed 0, BavarianCrops 12-class test
partition):

| Training set | Mean F1 | Class accuracy |
| --- | --- | --- |
| All classes (reference) | 0.64 | 0.62 |
| Grassland removed | 0.67 | 0.66 |
| Winter barley removed | 0.59 | 0.58 |
| Corn removed | 0.62 | 0.60 |

Remove-and-retrain (Fig. 6, class accuracy, several seeds): top-5 attended dates removed, about -5
points; top-25 removed, a further -2 points, model still at about 85 % of reference; training on the
top-3 dates only gives the steepest gain; top-15 dates only recover the reference. Hyperparameter
search (Table A.1): three configurations (1 or 3 layers, 1 head, dim 128) all reach F1 0.67 on
validation; weekly averaging of observations is markedly worse than right padding or random
subsampling (Fig. A.8).

Experimental factors varied: set of training classes (three single-class occlusions); set of dates
(remove-and-retrain). Factors held fixed: region (three pooled Bavarian landscapes), year (2018),
sensor (Sentinel-2), split (Rußwurm and Körner's block split), seed (0 for the explanation model).

Citable sentences, with the scope they actually carry:
- "attention patterns strongly relate to key dates, which are often associated with critical
  phenological events for crop-type classification" (in-region, one dataset).
- "the identified phenological events depend on the other crops considered during training"
  (class-set dependence; this is the sentence our manuscript paraphrases).
- "the interpretation of the attention weights is conditional to the classes in the dataset and
  focuses on events that help to disambiguate them."
- "the standard transformer encoder model does not utilize this information to disambiguate crops
  based on their specific vegetation dynamics throughout the vegetation season."
- Nothing in the paper mentions region, tile, location, or spatial transfer as a factor; the words
  "region" and "transfer" do not occur in the results or discussion in that sense.

Suggested replacement wording for the manuscript sentence that currently reads "the dates such
models attend to are themselves contingent on the training region \citep{obadic2022}": "the dates
such models attend to are contingent on the set of classes they were trained to separate
\citep{obadic2022}", or remove the clause entirely now that the dense-model gap it explained has
been shown to be an imputation artifact.

## Relevance to Our Crop-Classification Study

Our manuscript's revised findings are: (1) the holdout tile is missing September for 41 % of pixels,
never missing in training; zero-filling it before standardization in every raw-sequence deep model
created a train/test missingness shift that cost those models 0.08–0.14 holdout macro-F1 and produced
an apparent in-region/out-of-region ranking inversion; with September median-filled from the training
tiles, dense deep temporal pixel models lead (CNN-BiLSTM 0.66, TempCNN 0.65, L-TAE 0.65, Transformer
0.64, TimesNet 0.63), TabNet 0.62, tree/linear/MiniRocket 0.59–0.61, no inversion. (2) A sparsemax
channel gate gives no transfer gain on complete inputs and no consistent low-data gain, but about
+0.06 robustness to the zero-filled month; TabNet is not robust to zero fill (0.61 to 0.51). (3) At
10 % of training fields every model is near 0.55 macro-F1 (XGBoost 0.51); deep models reach about
0.65 at 100 %, logistic regression plateaus at 0.61.

**The current citation is mis-used and should be reworded or dropped.** Both `cea-article.tex`
(line 268) and `tgrs-article.tex` (line 235) write that the large in-region-to-holdout gaps of the
dense temporal nets are "consistent with evidence that the dates such models attend to are themselves
contingent on the training region \citep{obadic2022}." Obadić et al. never vary the region. Their
sole dataset is BavarianCrops, one year, one pooled set of three nearby Bavarian landscapes, one
block-wise split; the only factor they manipulate is the *set of crop classes* present in training
(and, separately, the set of dates). What they show is that attended dates are contingent on *which
other classes the model must separate*: remove grassland and winter barley's attention moves from
July to spring. Reading "class set" as "training region" is an extrapolation the paper does not make
and does not test. It is a plausible extrapolation (a new region has a different class composition,
so a model trained there would attend differently), but it is ours, not theirs, and the sentence as
written presents it as their evidence. Two acceptable rewrites: "the dates such models attend to are
contingent on the set of classes present in training \citep{obadic2022}, so a holdout region with a
different crop mix need not share them"; or simply drop the clause. In the revised manuscript the
sentence's purpose (explaining why dense nets transfer worst) no longer exists, since the gap it was
explaining was an imputation artifact and dense nets now lead on the holdout, so the cleanest fix is
to remove it from that paragraph. The paper remains citable elsewhere for what it does show (below).

**What the paper does support, and where it still fits.** Its real finding is that a single-layer
transformer encoder on parcel series is a *contrast detector*: it finds the few dates on which one
class looks unlike the rest and ignores the rest of the season, and the particular contrasts it picks
depend on the class set. For our Western Cape winter-crop problem (lucerne/medics versus small-grain
grazing, barley, canola, wheat), where the discriminative contrasts are presumably canola flowering
and cereal senescence timing, this predicts that a dense model will concentrate its evidence on a
few months. That makes the identity of those months a single point of failure: if the attended month
is the one that is missing or corrupted in the holdout, the loss is large; if not, it is small. This
is a cleaner mechanistic explanation for the *size* of the zero-fill damage (0.08–0.14, varying by
model) than anything about regional overfitting, and it suggests a diagnostic we can run cheaply: the
Eq. 4–6 column-mean attention of our Transformer and L-TAE on the training tiles, to see how much
weight September carries. If September is a key date, the artifact's magnitude is explained and the
median fill is doing real work; if it is not, the residual sensitivity must come from the
standardization pathology rather than from attention.

**On the sparsemax channel gate (finding 2).** Obadić et al. report, without a sparsity prior, that
softmax attention over time is already nearly one-hot per crop ("the entire attention being assigned
only to a few columns"). This is consistent with our observation that imposing sparsemax on a gate
adds nothing on complete inputs: the dense model is already sparse where it matters. Their
remove-and-retrain curve adds a second point: because many alternative sparse date sets give similar
accuracy (85 % of reference after removing the top 25 dates), a model that is forced to be sparse in
one place can usually re-route through another, which is one reading of why our gated and ungated
variants converge on clean data and only separate under corruption.

**On learning curves (finding 3).** Not addressed; no sample-size experiment. The remove-and-retrain
result that 15 well-chosen dates recover full accuracy is about temporal, not label, sparsity.

**On attention as evidence in general.** The paper's quantitative point, that attention weights are
one of several equally valid sparse explanations, cautions against using attention maps in our
supplement as if they identified "the" months the model uses. If we show attention over months, we
should pair it with a remove-and-retrain or gradient-attribution check (Rußwurm and Körner's `dy/dX`),
as this paper does.

## Evaluation Caveats

- **Region is never varied.** One dataset (BavarianCrops), one year (2018), three pooled Bavarian
  landscapes with a block-wise random split. No spatial, temporal, or sensor transfer is measured,
  and attention is not compared between the three regions. Any claim about regional dependence of
  attention attributed to this paper is unsupported.
- **In-region protocol.** The test blocks are inside the training landscapes (Rußwurm and Körner's
  4.5 km blocks with 500 m margins); this controls field-level leakage but not shared climate, soils,
  or crop calendars.
- **Single seed for the explanation model.** The attention analysis and class-occlusion comparison
  use seed 0 throughout, by design, so that changes can be attributed to the data; the cost is that
  attention variability across seeds is not characterized, even though the paper's own
  remove-and-retrain result implies that many alternative attention solutions exist. The
  remove-and-retrain accuracies are averaged over several seeds.
- **One layer, one head.** The selected model is the smallest configuration; multi-layer or
  multi-head attention would not reduce to a single column-mean per date, and the authors' own future
  work proposes a different architecture.
- **Class imbalance.** Five crops cover 89 % of parcels; focal loss is used; F1 (0.64) and class
  accuracy (0.62) are reported but no per-class table, so the occlusion effects on minority crops are
  invisible. The headline attention plots are for the five frequent crops only.
- **Occlusion changes the task.** Removing grassland raises F1 (0.64 to 0.67) because an easily
  confused majority class is gone, so part of the attention shift may reflect a simpler task rather
  than a different phenological focus. The paper does not separate these.
- **Metric for the ablation is class accuracy, not F1,** and the full-data reference is a single
  model; standard deviations are drawn but not tabulated.
- **No comparison explanation method.** Gradient-based attribution, which Rußwurm and Körner used on
  the same data, is cited but not run, so "attention is one of several explanations" is not tested
  against the alternatives.
- **Parcel-mean inputs.** As in the parent dataset, pixels are averaged per parcel; nothing is said
  about pixel-level models or field aggregation.
