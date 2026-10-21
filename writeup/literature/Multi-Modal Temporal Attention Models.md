# Multi-Modal Temporal Attention Models for Crop Mapping from Satellite Time Series

**Citation:** Vivien Sainte Fare Garnot, Loic Landrieu, and Nesrine Chehata (2022). *ISPRS Journal of
Photogrammetry and Remote Sensing* 187, 294–305. DOI: `10.1016/j.isprsjprs.2022.03.012` (verified:
Crossref returns "Multi-modal temporal attention models for crop mapping from satellite time series",
ISPRS JPRS vol. 187, pp. 294–305, issued 2022-05). The PDF in the folder is the 21-page arXiv preprint
`arXiv:2112.07558v1` (14 Dec 2021), which contains the same experiments as the journal version plus
the full appendix figures; page and table numbers below refer to the arXiv text. Dataset:
PASTIS-R (`doi:10.5281/ZENODO.5735646`); code: `github.com/VSainteuf/pastis-benchmark`,
`github.com/VSainteuf/utae-paps`, `github.com/VSainteuf/lightweight-temporal-attention-pytorch`.

## Objectives

The paper asks how temporal-attention crop-mapping networks (PSE+L-TAE for parcel classification,
U-TAE for pixel segmentation, U-TAE+PaPs for panoptic segmentation) should be extended to consume
Sentinel-1 radar time series alongside Sentinel-2 optical series. Three questions are posed: which
fusion scheme (early, mid, late, decision) works best for each task; whether two cheap training
enhancements, auxiliary per-modality supervision and *temporal dropout*, improve fusion; and whether
multimodality buys robustness when optical observations are removed at test time to simulate heavier
cloud cover. A fourth, practical contribution is PASTIS-R, the PASTIS dataset augmented with
spatially aligned Sentinel-1 ascending and descending time series.

## Methods

**Data: PASTIS-R.** 2433 patches of 1.28 km x 1.28 km sampled from four Sentinel-2 tiles in
metropolitan France, 2019 season, with per-pixel semantic (18 crop classes plus background) and
instance (parcel) annotations from the French LPIS (RPG). Sentinel-2: all available acquisitions, no
cloud filtering ("certain patches can be partially or entirely obstructed by clouds"), about 115k
images. Sentinel-1: GRD sigma-0 backscatter in dB, orthorectified to 10 m with Orfeo Toolbox, no
speckle filtering or terrain correction, assembled as `VV`, `VH`, `VV/VH`; ascending and descending
orbits kept as two separate modalities (`S1A`, `S1D`) because incidence angle differs, roughly 70
acquisitions each, 339k radar images in total. Modalities are thus `M = 3`.

**Fusion schemes (Sec. 3.1), reformulated for sequence encoders.**
- *Early fusion*: interpolate S1 to the S2 dates, concatenate channels per date, one spatio-temporal
  encoder. Requires an interpolation step; cheapest model (117k parameters for parcel classification).
- *Mid fusion* (new): one spatial encoder (PSE) per modality, the per-date embeddings of all modalities
  are stacked along time into a single sequence, one shared temporal encoder (L-TAE). No interpolation,
  no duplicated temporal encoder. Not applicable to U-TAE, whose spatial and temporal encoders are
  intertwined.
- *Late fusion*: one full spatio-temporal encoder per modality, embeddings concatenated, one decoder.
- *Decision fusion*: independent encoder and decoder per modality, predictions averaged.

**Enhancements (Secs. 3.2–3.3).**
- *Auxiliary supervision*: an extra cross-entropy term per modality (weight `lambda_m = 0.5`) through
  modality-specific decoders (and extra temporal encoders in mid fusion), discarded at inference.
- *Temporal dropout*: during training each observation of modality `m` is dropped with probability
  `p_m` (`p_S2 = 0.4`, `p_S1A = p_S1D = 0.2`); at inference all observations are used. Observations are
  *removed from the sequence*, not replaced by a fill value, which L-TAE and U-TAE tolerate because
  they take variable-length sequences with day-of-year positional encoding. The scheme also cuts
  training memory, and was *necessary* to fit the late- and decision-fusion segmentation models into
  32 GB of GPU memory.

**Backbones.** Parcel classification: PSE + L-TAE with the published hyperparameters, MLP decoder,
cross-entropy, 100 epochs, batch 128. Semantic segmentation: U-TAE, 2-layer conv decoder, 19 classes.
Panoptic: U-TAE + Parcel-as-Points (PaPs) head, learning rate 0.01 for 50 epochs then 0.001.

**Evaluation protocol.** The official five cross-validation folds of PASTIS are used for every
number in the paper, and results are 5-fold cross-validated means. Metrics are overall accuracy and
class-averaged intersection-over-union (mIoU) for classification and segmentation, and class-averaged
SQ/RQ/PQ for panoptic. The paper itself does not describe how the PASTIS folds were constructed; the
fold definition is inherited from the PASTIS paper (Garnot and Landrieu, ICCV 2021), and readers must
consult that work to establish whether folds are spatially blocked. Because all four tiles contribute
to the training data of every fold, this is in-region cross-validation at the patch level and not a
disjoint-region holdout. There is no cross-year test. The cloud-robustness experiment is an
*artificial perturbation of the test set*: random optical acquisitions are deleted at inference time
(10 % to 90 % remaining) from the same folds, with radar left intact; crop growth and the model are
unchanged, which the authors flag as a limitation.

## Key Findings

**Parcel classification (Table 1, mIoU, 5-fold CV).** Single modality: `S2` 73.9 (74.5 with
temporal dropout), `S1D` 64.5, `S1A` 63.3. Optical alone beats radar alone by about 10 mIoU. Without
enhancements, early (74.9) and mid (75.1) fusion improve on S2-only, while late (73.0) and decision
(72.5) fusion are slightly worse, consistent with Ofori-Ampofo et al. (2021). With both enhancements
the order flips: late fusion with auxiliary supervision and temporal dropout reaches 77.2 (+3.3 over
S2-only), decision 75.8, mid 76.5, early 76.5 (temporal dropout only). Per-class gains concentrate in
hard classes (potatoes, winter triticale). Parameter counts: 114k (unimodal), 117k (early), 152k/185k
(mid without/with auxiliary heads), 254k/287k (late), 259k (decision). Mid fusion is about 20 %
faster at inference than late fusion.

**Temporal dropout alone helps the optical-only model.** S2-only rises from 73.9 to 74.5 mIoU
(classification) and 63.1 to 63.6 (segmentation) with no other change. The authors state that
temporal dropout "was beneficial to all fusion schemes and the optical baseline across our
experiments."

**Semantic segmentation (Table 2).** `S2` 63.1 (63.6 with Tdrop), `S1D` 54.9, `S1A` 53.8; early
fusion 64.9 (65.8 with Tdrop); late fusion 65.8 with Tdrop and 66.3 with auxiliary + Tdrop; decision
64.7/64.3. Radar sharpens parcel boundaries (volumetric scattering at field edges). Segmentation is
about 10 points below parcel classification because parcel contours are unknown and background is a
class.

**Panoptic (Table 3).** `S2` PQ 40.4; early fusion + Tdrop 42.0; late fusion + Tdrop 41.6. Gains come
from recognition quality (more parcels detected) rather than segmentation quality.

**Test-time optical-date removal (Fig. 8), the result most relevant to us.** As the fraction of
available optical observations is reduced from 90 % to 10 %:
- The S2-only model "drops drastically" for both parcel classification and semantic segmentation and
  falls *below the radar-only model* once about 70 % of optical observations are removed. Note that
  this S2-only curve is for a model trained *with temporal dropout* (`S2 + Tdrop` in the legend); the
  segmentation panel also shows a plain `S2` curve that degrades faster.
- All fusion models hold "an almost constant level of performance for up to 50 % missing optical
  acquisitions."
- Beyond 50 % the schemes separate by how much the modalities interact inside the network: early
  fusion is least robust, mid fusion next, late fusion better, and decision fusion (independent
  classifiers, averaged) is most resilient, still beating radar-only with 90 % of optical images gone.
- Auxiliary supervision and temporal dropout make *both unimodal and multimodal* models more
  resilient to missing optical acquisitions (shown for segmentation; "the same phenomenon can be
  observed for parcel classification").
- Panoptic with only 30 % of optical observations: S2-only PQ falls to 33.0, late fusion keeps 37.6.

**Recommendations (Sec. 5.1).** Late fusion with both enhancements by default; mid fusion when
inference speed and memory matter and clouds are moderate; decision fusion where "only a few optical
observations may be available for inference"; early fusion is compact but needs interpolation and is
least robust.

**What is not measured.** No transfer to a region or year outside the four French tiles (the
authors say their results "are most relevant for crop mapping applications with the same
meteorological context, terrain conditions, and crop types"); no comparison with non-attention
baselines (no RF, no TempCNN, no LSTM); no per-class F1 or kappa (only mIoU); no experiment with a
fill value substituted for missing dates, only deletion; the perturbation is random dates, not a
systematically missing calendar month; no interferometric or polarimetric SAR features.

### Numbers to cite

Parcel classification, mIoU (Table 1; OA in parentheses for the base column):

| Model | Base | + Temporal dropout | + Auxiliary | + Aux & Tdrop | Params |
| --- | --- | --- | --- | --- | --- |
| `S2` only | 73.9 (91.7) | 74.5 | - | - | 114k |
| `S1D` only | 64.5 (87.0) | 64.7 | - | - | 114k |
| `S1A` only | 63.3 (86.4) | 62.9 | - | - | 114k |
| Early fusion | 74.9 (91.8) | 76.5 | - | - | 117k |
| Mid fusion | 75.1 (92.0) | 75.9 | 75.0 | 76.5 | 152k / 185k |
| Late fusion | 73.0 (91.1) | 73.6 | 76.1 | 77.2 | 254k / 287k |
| Decision fusion | 72.5 (91.0) | 72.8 | 75.2 | 75.8 | 259k |

Semantic segmentation, mIoU (Table 2): `S2` 63.1 / 63.6 (Tdrop); early 64.9 / 65.8; late 65.8
(Tdrop) / 66.3 (aux + Tdrop); decision 64.7 / 64.3. Panoptic PQ (Table 3): `S2` 40.4; early + Tdrop
42.0; late + Tdrop 41.6; with only 30 % of optical dates, `S2` 33.0 versus late fusion 37.6.

Test-time date removal (Fig. 8, read from the curves, no table given): S2-only (with Tdrop) degrades
monotonically and crosses below `S1D`-only at roughly 30 % optical observations remaining; all fusion
models are flat to about 50 % remaining; decision fusion stays above `S1D`-only at 10 % remaining.

Citable sentences: "Multimodal fusion models can maintain an almost constant level of performance for
up to 50 % missing optical acquisitions"; "Early fusion proves the least robust to missing optical
observations"; "the decision fusion scheme comprises independent classifiers and proves to be the most
resilient"; temporal dropout "was beneficial to all fusion schemes and the optical baseline across our
experiments"; and, on scope, "our results are most relevant for crop mapping applications with the
same meteorological context, terrain conditions, and crop types as this region."

Terminology: in this paper "temporal dropout" means deleting observations from the input sequence at
training time with probability `p_m`; it is not dropout on hidden units and it does not insert a
fill value.

## Relevance to Our Crop-Classification Study

Our manuscript's revised findings are: (1) the holdout tile is missing September for 41 % of pixels,
never missing in training; zero-filling it before standardization in all raw-sequence deep models
imposed a train/test missingness shift that cut their holdout macro-F1 by 0.08–0.14 and produced an
apparent ranking inversion. With September filled from training-tile medians, dense deep temporal
pixel models lead (CNN-BiLSTM 0.66, TempCNN 0.65, L-TAE 0.65, Transformer 0.64, TimesNet 0.63),
TabNet 0.62, tree/linear/MiniRocket 0.59–0.61; no inversion. (2) A sparsemax channel gate gives no
transfer gain on complete inputs and no consistent low-data gain, but about +0.06 robustness to the
zero-filled month; TabNet is not robust to zero fill (0.61 to 0.51). (3) At 10 % of training fields
every model is near 0.55 macro-F1 (XGBoost 0.51); deep models reach about 0.65 at 100 %, logistic
regression plateaus at 0.61.

**This paper supplies the missing-data protocol our pipeline should have used, and a training-time
remedy we have not yet tried.** Garnot et al. never *fill* a missing optical date; they *delete* it
from the sequence, which the L-TAE/U-TAE family handles natively through day-of-year positional
encoding and variable-length attention. Their test-time experiment is therefore a clean measurement
of "how much does the model lose when an observation is genuinely absent," which is the question our
holdout tile actually poses for September. Our old pipeline instead inserted an out-of-distribution
constant (zero after standardization is the training mean, but a whole month pinned at the mean of
every band is a pattern no training pixel exhibits) at a fixed grid slot, which is a different and
harsher perturbation. The median fill we now use is closer in spirit to their setting but still
asserts a value. A direct replication of their protocol on our models, mask September in the
sequence for the L-TAE and Transformer (both already carry positional encodings) and compare with
median fill, would tell us whether any of the remaining 0.59–0.66 holdout spread is still an
imputation effect.

**Temporal dropout is the obvious next ablation.** With `p = 0.4` on optical dates, their S2-only
PSE+L-TAE gained +0.6 mIoU on clean test data *and* became markedly more resilient to deleted dates.
Our deep models were trained with no observation dropout at all, so the missingness shift found them
maximally brittle. Training CNN-BiLSTM, TempCNN, L-TAE, and the Transformer with month-level
temporal dropout (drop each month with probability 0.2–0.4, fixed monthly grid, mask or median fill
at the dropped slots) and then evaluating on the zero-filled, median-filled, and masked September
variants of the holdout would (a) show whether the 0.08–0.14 loss could have been prevented without
touching test-time imputation, and (b) give a fairer comparison for the sparsemax gate's +0.06
robustness, which may be smaller than, or redundant with, what temporal dropout alone provides. Given
finding 2 (no gain on complete inputs), the gate's whole case rests on robustness to missing months;
it should be benchmarked against this cheaper, standard augmentation before it is presented as a
contribution.

**Their robustness ordering supports our reinterpretation of the inversion.** Early fusion, the
scheme in which the whole network sees a single interleaved stream, is least robust to deleted dates;
decision fusion, in which independent sub-models vote, is most robust. Our pre-fix inversion looked
like "dense models transfer worse," but in their framing it reads as "a single end-to-end sequence
model is most exposed to a corrupted input stream," which is a statement about input handling, not
about regional overfitting. Our field-level majority vote over pixels is itself a form of decision
fusion across pixels and may be part of why field-level aggregation damped the artifact.

**Dense attention models are not fragile in this paper.** PSE+L-TAE, U-TAE, and their fusion
variants are the only models tested, and with clean inputs and standard regularization they are
stable across folds (standard deviations are not reported in Table 1 but 5-fold means are used).
Nothing here suggests that dense temporal attention should lose to trees on a neighbouring tile with
the same crop calendar, which is consistent with our corrected holdout where L-TAE (0.65) and the
Transformer (0.64) sit above the tree models (0.59–0.61).

**On finding 3 (learning curves).** No label-fraction experiment is reported. The parameter counts
(114k for a PSE+L-TAE that reaches 73.9 mIoU on 18 classes) are a reminder that the attention models
we use are small; our deep models reaching only about 0.55 at 10 % of fields alongside every other
family is therefore unlikely to be a capacity effect.

**SAR fusion as a structural fix for missing optical months.** Our study is optical-only (B2, B6,
B11, B12, EVI, hue). Their central result is that a second modality lets the model keep performance
with half the optical dates gone and that decision fusion keeps it with 90 % gone. For an operational
Western Cape product where winter-season cloud and tile-edge coverage gaps are recurrent, adding
Sentinel-1 would remove the need to impute September at all. This belongs in our discussion as the
principled long-term answer to the missingness problem rather than as an experiment we can run now.

**What this paper cannot support.** It is in-region 5-fold CV on four French tiles; it does not
measure spatial transfer and should not be cited for it. Its robustness experiment deletes random
dates uniformly, not a contiguous calendar month, and does not test a fill-value perturbation, so it
does not directly quantify the artifact we found; it quantifies the ceiling of what a well-trained
attention model can tolerate when the perturbation is handled correctly.

## Evaluation Caveats

- **Protocol.** 5-fold cross-validation on PASTIS folds; all four tiles are in every training set.
  In-region CV, not out-of-region holdout. Fold construction is not described in this paper (see the
  PASTIS paper). No cross-year test.
- **Metrics.** mIoU and OA only for classification and segmentation; no macro-F1, kappa, or per-class
  tables in the text (per-class IoU appears only as bar charts, Figs. 5 and 10). No standard
  deviations across folds in Tables 1–3, so the 0.5–1 mIoU differences between fusion variants cannot
  be tested for significance.
- **Class imbalance.** PASTIS is dominated by meadow and winter cereals; mIoU is class-averaged,
  which is a balanced metric, but OA (91–92 %) is not and is reported alongside it.
- **Cloud experiment is synthetic.** Random deletion of dates on the same patches; does not change
  crop development, does not model systematic (seasonal, tile-edge) gaps, does not test a fill value.
  The authors note that a rigorous test would vary regions and years with real cloud statistics.
- **No non-attention baselines.** No RF, gradient boosting, TempCNN, or LSTM in any table, so the
  paper cannot speak to tree-versus-deep comparisons.
- **Fusion experiments are not fully factorial.** Auxiliary supervision is not evaluated for early
  fusion (by design) or for panoptic late fusion (cost); decision fusion is not run for panoptic;
  mid fusion is not run for segmentation. Late- and decision-fusion segmentation models could only be
  trained *with* temporal dropout, so their "base" row is empty and the enhancement's effect is
  confounded with feasibility.
- **Radar preprocessing is minimal** (no speckle filtering, no terrain correction), which the authors
  defend but which may understate the radar modality's value.
- **Compute.** Single V100 32 GB; inference times reported for one fold (Table 4); late fusion is
  2x slower than early fusion for panoptic (819 s vs 414 s per 500 patches).
