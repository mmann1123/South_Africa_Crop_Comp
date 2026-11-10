# A deep learning approach for mapping and dating burned areas using temporal sequences of satellite images

**Citation:** Miguel M. Pinto, Renata Libonati, Ricardo M. Trigo, Isabel F. Trigo, and Carlos C. DaCamara
(2020). "A deep learning approach for mapping and dating burned areas using temporal sequences of
satellite images." *ISPRS Journal of Photogrammetry and Remote Sensing*, 160, 260-274.
DOI: `10.1016/j.isprsjprs.2019.12.014` (extracted from the PDF first page; verified against Crossref
on 2026-11-09: title, journal, volume 160, pages 260-274 and author list Pinto, Libonati, Trigo,
Trigo, DaCamara all match).

Sources for this briefer: the full text of the PDF in this folder (`pdftotext -layout`, 15 pages). The
statements about the authors' released code (`mnpinto/banet`) come from the audit's code reading
(`banet/data.py`) and were not re-checked here.

This is a burned-area segmentation paper on daily VIIRS image sequences, not a crop-classification
paper. Per authoring rule 7 the crop-specific questions (field splits, crop imbalance, band design
for crops) are mostly not applicable, and the briefer concentrates on the evaluation protocol and on
how clouds and gaps in the daily sequences are handled.

## Objectives

The paper proposes `BA-Net`, an end-to-end deep model that maps burned area and assigns a day of
burning from daily multispectral image sequences. It is a 3D U-Net in which each 3D convolution is
split into a spatial and a temporal part (after De Fauw et al. 2018), with a unidirectional LSTM
after the first spatial convolution "allowing the model to capture longer-term temporal
dependencies in the data and to better filter the noise" (p. 263). The stated advantages over
MCD64A1 Collection 6 and FireCCI51 are: "(1) No clouds/shadows mask, filtering of lower quality
images or use of temporal composites is required"; (2) the structure depends on the existence of
daily image sequences, not on a particular sensor; and (3) active fires enter as an extra input
channel with no preprocessing (p. 261). The authors also claim near-real-time operation, with the
output converging to the final solution within days.

The paper claims to be the first application of these techniques to burned-area mapping and dating,
and one of the first uses of VIIRS for global burned-area mapping.

## Methods

### Regions, inputs and targets

- **Five regions:** California, Portugal, the Brazilian Cerrado, Mozambique and north Australia,
  chosen to span different fire regimes (large fires, low ignition density in Australia; many small
  fires in Mozambique; Brazil in between). Portugal and California were also chosen for recent
  extreme fire years and the authors' domain knowledge (p. 262).
- **Inputs:** daily VIIRS Level 1B top-of-atmosphere Red (`M5`), NIR (`M7`) and MIR (derived from
  `M12` and `M15`) reflectances, 750 m, bilinearly resampled to a 0.01 degree grid, taking the
  image with the lower satellite zenith angle when several exist for a day. A fourth channel holds
  VIIRS 375 m active-fire radiative power summed per grid cell. Years 2012-2018.
- **Seasonal windows:** only each region's fire season is used: June to September for California
  and Portugal, July to October for Brazil and Mozambique, September to December for Australia,
  with extensions for Portugal 2017 and California 2018.
- **Targets:** `MCD64A1C6` daily burned fractions on the same grid ("classification with soft
  labels"). The model is therefore trained to imitate another satellite product, not a reference
  survey.
- **Tiles and sequences:** 128 by 128 tiles with 28 pixels of overlap; input tensors of 64 days by
  4 channels. Training sequences are sampled around days with at least 10 burned pixels, which
  biases training towards burned-rich periods.

### Clouds and gaps

No cloud or shadow mask is applied in the paper: "No clouds/shadows mask, filtering of lower quality
images or use of temporal composites is required" (p. 261). The model is instead made robust by
architecture and augmentation (p. 264): the LSTM is justified as a way "to better filter the
noise", and three augmentations are used, namely random brightness and contrast on 75 percent of
images, random constant-valued rectangular patches on the reflectance channels (probability 0.5,
5 to 50 pixels, 1 to 5 patches) described as aimed "to make the model robust to any signal that is
not persistent in time, such as clouds", and random erasing of the active-fire channel to simulate
omission. The paper does not describe any treatment of missing or invalid pixels (swath gaps, fill
values, NaN). According to the audit's reading of `banet/data.py`, the loader reads no QA flags,
applies the union of the `I`-band netCDF masks, and runs `mask2nan` then `nan2zero`, so masked
observations become zero. The paper text never states this, and no cloud fraction, valid-observation
count or filled-pixel fraction is reported for any region or year.

### Evaluation protocol

**Evaluation protocol:** Leave-one-year-out cross-validation over 2012-2017 (six folds) within the
same five regions, plus a single held-out test year (2018) in the same regions, with a held-out
validation of the burned-area maps against independent medium-resolution reference maps. It sits at
the cross-year rung of the spectrum this manuscript is built around, but there is no spatially
disjoint region: all five regions are both trained on and evaluated on, and the paper does not say
that any region is excluded from training (it appears to be one model over all regions; the text
does not state this explicitly). No region-to-region transfer is claimed. The paper does say its
future work is "to investigate how the model performs in other regions" (p. 272), which confirms
that region transfer is untested.

Details of the protocol (p. 265):

- Cross-validation: "leaving one year out each time for the training period (2012-2017), resulting
  in six CV folds", with three training runs per fold to measure random-augmentation variance.
- The 2018 test scores are averaged over the six CV fold models.
- Metrics are Dice (equal to F1 for the burned class), overall accuracy, omission and commission
  error, and, for dating, the mean absolute error (MAE) of the burning date against VIIRS active
  fires. A fixed 0.5 threshold converts BA-Net outputs to a binary mask in the intercomparison.
- The first and last month of each season are dropped from evaluation because of the 15-day
  context windows.
- Burned-area accuracy against reference maps (Table 6) is evaluated only where reference data exist
  (Brazil 2015 from 53 Landsat-8 scenes, Mozambique 2016 from the Sentinel-2 FireCCISFD11, California
  2012-2016 MTBS perimeters, Australia 2014-2016 TERN AusCover, Portugal 2017 ICNF maps).
- For BA-Net, MCD64A1C6 and FireCCI51 alike, the burn-fraction threshold that balances omission and
  commission is chosen per region and year, using the same reference data it is then scored on.
  This favours all three products but is the same tuning for each.

Spatial leakage is not a field-versus-pixel issue here. The relevant overlap is that tiles are 128
by 128 with 28 pixels of overlap, sampled from the same regions and seasons across years, and that
the training target (`MCD64A1C6`) is also the benchmark in the intercomparison, so agreement with
it is partly by construction.

## Key Findings

### Agreement with MCD64A1C6 (Table 4, Dice)

| Region | CV 2012-2017 BA-Net vs MCD64A1C6 | Test 2018 BA-Net vs MCD64A1C6 |
|---|---|---|
| California | 0.884 | 0.906 |
| Portugal | 0.878 | 0.940 |
| Brazil | 0.831 | 0.780 |
| Mozambique | 0.742 | 0.749 |
| Australia | 0.886 | 0.881 |

The authors say the fluctuations "are to be expected since the Dice score is usually higher when
large fires occur" and attribute the 2018 values to fire size: "California had several large fires
in 2018, while the same year in Portugal was generally mild with a single large fire, explaining
the Dice score of 0.94" (p. 266). They give no explanation for Brazil falling from 0.831 to 0.780,
and clouds are not invoked for any regional difference in these tables.

### Validation against reference maps (Table 6, Dice at burned fraction 0.5)

BA-Net's overall Dice is 0.678 against 0.687 for `MCD64A1C6` and 0.656 for `FireCCI51`. Per-region
values range from 0.582 (Mozambique 2016) and 0.729 to 0.837 (Australia 2014-2016) up to 0.907
(Portugal 2017), with Brazil 2015 at 0.754 and California 2012-2016 at 0.786 to 0.920. Lower scores
in Mozambique and Brazil are attributed to small, fragmented fires, which the 750 m inputs resolve
less well, not to clouds. At low reference burned fractions BA-Net beats both products; in
California the opposite holds, attributed to the reference data being fire perimeters (pp. 268-269).

### Dating (Table 5)

Overall MAE against active-fire dates is 1.13 days for BA-Net against 1.80 for `MCD64A1C6` and 4.25
for `FireCCI51` in cross-validation, and 1.11 against 1.80 for 2018. Dating here is scored against
VIIRS active fires, which are also a model input, so the authors note the model "was optimized to
reproduce the MCD64A1C6 daily burned fractions and not the dates of the active fires".

### Ablation (Tables 7 and 8, 2015 held out)

Removing the LSTM lowers Dice against `MCD64A1C6` in all five regions (for example Mozambique 0.727
to 0.656), removing the U-Net structure lowers it in all regions, and removing both lowers it most.
Against the reference maps the ablations help in California, which the authors again attribute to the
perimeter-style reference. The ablation is run on a single held-out year.

### Cloud case studies (Section 3.4, pp. 269-271)

The only treatment of clouds is qualitative. For Central Portugal in October 2017 "the presence of
clouds during the fire event did not allow a clear view of the surface during the first few days",
with usable images only a week later, and 19 to 20 October "completely covered by clouds". The
near-real-time output initially shows commission errors and converges once better images arrive. The
conclusion says NRT convergence "depends primarily on cloud coverage". No numbers on cloud fraction
or valid observations are given.

## Relevance to Our Crop-Classification Study

### A different task, and no region transfer to compare

This paper is not a comparator for our holdout-tile macro-F1. Its generalization test is
leave-one-year-out in regions that are also in training, so it neither supports nor contradicts our
finding that in-region rankings of dense temporal deep nets reverse under a spatially disjoint
holdout. Its architecture (a 3D convolutional U-Net with an LSTM, 3.5 million parameters) is the
dense-temporal family our study finds overfits in a new region, but nothing here tests that.

### Sequence handling relevant to our missingness analysis

Two design choices bear on the observation-dropout argument made elsewhere in this project. First,
the paper handles clouds by leaving them in the input and training through them with augmentation
(random constant-valued patches meant to mimic non-persistent signals), with an LSTM as the denoiser.
This is a cousin of the temporal random masking in Zhang et al. 2024 and the observation dropout in
Pham et al. 2024: the model is trained on corrupted inputs it will meet at test time. Second, the
paper reports no per-region or per-year cloud, valid-observation or filled-pixel statistics, and
never says how gaps are filled. If the audit's code reading is right, masked days become zeros with
no indicator, which is the zero-fill pattern behind our September finding (zero-filling that month,
not the region change, drove most of the CNN-BiLSTM's transfer drop).

### Evaluation design points worth borrowing or flagging

- Reporting the standard deviation over three seeds per fold (and averaging six fold models for the
  test year) is the variance reporting our study also needs and that Mask-PSTIN lacks.
- Scoring against independent higher-resolution reference maps, in addition to a product-to-product
  comparison, separates agreement with the training target from accuracy. Our analogue is the hidden
  field-label holdout.
- The threshold is optimised on the reference data it is scored on, for all products. This is a mild
  optimistic bias to remember when comparing with any number computed under a fixed threshold.

### What the paper does not measure

No region or sensor transfer (the authors list other regions as future work), no per-region cloud or
valid-observation statistics, no test where the amount of cloud is varied, no comparison against a
tree-based or other simple baseline (the only ablations are within the BA-Net family), and no
computational comparison beyond training time (about 4.3 hours) and inference time (155 ms per
sequence on a GTX 1080). No class-imbalance issue arises in the crop sense, though burned pixels are
rare and Dice, not accuracy, is the headline metric.

## Evaluation Caveats

- **Cross-year, not cross-region.** All five regions appear in both training and evaluation. The
  leave-one-year-out design measures year-to-year robustness only.
- **Target equals benchmark.** BA-Net is trained to reproduce `MCD64A1C6`, and Table 4 and the dating
  scores are measured against that product or against the active fires used as input, so these
  results are not independent. Table 6 against reference maps is the independent evidence.
- **Reference maps are partial and heterogeneous.** They differ by region (Landsat-8 scenes,
  Sentinel-2 maps, perimeters, a provisional Portuguese map), cover only some years, and the
  California reference is perimeter-based, which the authors invoke repeatedly to explain the
  deviations there.
- **Thresholds tuned on the scoring data** (balanced omission and commission, per region and year)
  for all three products.
- **Unexplained regional change.** Brazil's Dice falls from 0.831 (cross-validation) to 0.780 (2018)
  with no explanation, and no missingness statistic is offered that could test whether cloud cover
  differed between the years.
- **Cloud handling is untested.** The claim that no cloud masking is needed is supported only by
  qualitative case studies and by the augmentation design, not by an experiment that varies cloud
  cover or compares masked and unmasked inputs.
- **Single-run ablations.** The ablation uses one held-out year (2015) and its California results
  contradict the other regions without a tested explanation.
- **Missingness audit.** The audit report ranks this paper "2 (prov.)" from code alone; the full
  text reports no transfer loss attributed to anything other than fire size and reference-map
  characteristics, and no per-domain cloud or valid-observation statistics.
