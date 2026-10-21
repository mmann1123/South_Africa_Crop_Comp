# TimeMatch: Unsupervised cross-region adaptation by temporal shift estimation

**Citation:** Joachim Nyborg, Charlotte Pelletier, Sébastien Lefèvre, and Ira Assent (2022). *ISPRS
Journal of Photogrammetry and Remote Sensing* 188, 301–313. DOI: `10.1016/j.isprsjprs.2022.04.018`
(verified: extracted from the PDF's first page; Crossref returns the identical title, ISPRS JPRS,
issued 2022-06). Received 4 Nov 2021, accepted 20 Apr 2022; CC BY 4.0. Code and dataset:
`github.com/jnyborg/timematch`; dataset archive `doi:10.5281/zenodo.5636422`.

## Objectives

The paper addresses cross-region unsupervised domain adaptation (UDA) for satellite image time
series: a crop classifier trained with labels in a source region must classify an unlabeled target
region whose crop calendar is shifted in time by climate. The authors argue that existing UDA methods
(adversarial feature alignment, discrepancy minimization, self-training) ignore the temporal
dimension and therefore give "only limited benefits" for SITS. Three contributions: (i) an
unsupervised estimator of the temporal shift (in days) between target and source, computed from the
predictions of a source-trained model on temporally shifted target data; (ii) TimeMatch, a
self-training algorithm that uses the estimated shift to generate accurate target pseudo-labels and
retrains the model on shifted source data plus pseudo-labeled target data; (iii) the TimeMatch
dataset, over 300,000 LPIS-labeled parcels from four Sentinel-2 tiles in Denmark, France, and Austria
for benchmarking cross-region adaptation.

## Methods

**Data (Sec. 4.2).** Four Sentinel-2 tiles for 2017: `32VNH` Denmark (`DK1`), `30TXT` mid-west
France (`FR1`), `31TCJ` southern France (`FR2`), `33UVP` Austria (`AT1`). Level-1C top-of-atmosphere
reflectance, all acquisitions with cloud cover at most 80 % and tile coverage at least 50 %, 10 bands
(atmospheric bands 1, 9, 10 removed; 20 m bands bilinearly resampled to 10 m). Acquisition dates and
sequence lengths differ between tiles (Fig. 5), and no cloud masking or gap filling is applied; cloudy
dates are left in the series. Labels from the national LPIS registers; 15 major European crop classes
plus "unknown"; parcels eroded by 20 m and those under 1 ha removed to limit edge label noise. Per
tile, parcels are split 70/10/20 into train/validation/test. For a source tile only classes with at
least 200 parcels are kept as the `K - 1` crop classes, the rest become "unknown"; target evaluation
uses the same source class list regardless of target frequency.

**Model.** PSE+LTAE (Garnot et al. 2020; Garnot and Landrieu 2020): a pixel-set encoder samples
`S = 64` pixels per parcel per date during training (all pixels at test time), and a lightweight
temporal attention encoder with sinusoidal day-of-year positional encoding outputs a single parcel
embedding classified by an MLP. The parcel geometry features normally fed to PSE are deliberately
omitted "to avoid biasing the model towards the shapes of parcels in the source region." Source
training: focal loss (`gamma = 1`), Adam 1e-3 with cosine annealing, 100 epochs, batch 128, weight
decay 1e-4, random subsampling of 30 time steps as augmentation, model selection on the *source*
validation set.

**Temporal shift estimation (Sec. 3.3).** For each candidate shift `delta` in `[-60, +60]` days, the
target series are shifted by adding `delta` to every day-of-year and passed through the source-trained
model. A good shift should make the model confident (low expected entropy of the conditional
prediction) and diverse (marginal prediction matching the class distribution). Two scores borrowed
from generative-model evaluation are repurposed: the Inception Score (IS; confident and uniform
marginal) and the AM score (confident and marginal matching a given class distribution). Because the
target class distribution is unknown, Algorithm 1 first picks a shift with IS, pseudo-labels the
target under that shift to estimate the class distribution, then re-estimates the shift with AM.

**TimeMatch learning (Sec. 3.4).** Student and teacher are both initialized from the source model. The
teacher pseudo-labels weakly augmented target series shifted by `delta_{t->s}`; the student is
trained with a FixMatch loss on strongly augmented (time-subsampled) target series using confident
pseudo-labels (`epsilon = 0.9`) plus a supervised loss on source series shifted by
`delta_{s->t} = -delta_{t->s}` with true source labels (`lambda = 2`). The teacher is an exponential
moving average of the student (`alpha = 0.9999`), so the shift needs re-estimation only once per
epoch (500 iterations); 20 epochs at learning rate 1e-4. Domain-specific batch normalization and a
class-balanced source sampler are used.

**Comparisons (Sec. 4.3).** Source-trained PSE+LTAE applied without adaptation (lower bound);
target-trained PSE+LTAE with target labels (upper bound); FixMatch (TimeMatch without shift
estimation); MMD; DANN; CDAN+E; ALDA; JUMBOT, all re-implemented with the PSE+LTAE backbone. Each
competitor is also run with *ShiftAug*, a data augmentation that applies a random shift uniform in
`[-60, 60]` days to every training series (source and target) to train shift-invariant models. PAN
(Wang et al. 2021) could not be obtained; MMD stands in for it.

**Evaluation protocol.** Five cross-region tasks: `DK1->FR1`, `DK1->FR2`, `DK1->AT1`, `FR1->DK1`,
`FR1->FR2`. Target labels are used only for the final test set; hyperparameters are tuned on the
target validation set of `DK1->FR1` alone and reused for every task, and all results are the model at
the end of training (no target-label model selection). Reported metric: macro F1 (%), mean and standard
deviation over three runs with different dataset splits. This is a genuine disjoint-region (indeed
disjoint-country) spatial holdout with parcel-level labels, the top of our manuscript's spectrum;
parcels within a tile are split 70/10/20 at the parcel level, so there is no pixel-level leakage.
Cross-year transfer is not tested (all tiles are 2017).

## Key Findings

**Source-only transfer collapses, and the collapse is the baseline against which everything is
measured (Table 1, macro-F1 %).** PSE+LTAE trained in one region and applied to another without any
adaptation scores `DK1->FR1` 28.3 ± 1.9, `DK1->FR2` 29.0 ± 5.2, `DK1->AT1` 43.4 ± 4.0, `FR1->DK1`
24.9 ± 2.0, `FR1->FR2` 70.3 ± 1.9, average 39.2 ± 3.0. The same architecture trained with target labels
reaches 74.6, 72.4, 86.9, 90.6, 85.7, average 82.0 ± 1.9. The source-only deficit is therefore about
43 F1 points on average, from 15 points (`FR1->FR2`) to 66 points (`FR1->DK1`). Training the source
model with ShiftAug alone raises source-only transfer to 40.9 / 37.4 / 48.9 / 47.3 / 70.5, average
49.0 ± 1.8, a 10-point gain with no target data at all.

**Generic UDA methods barely help; TimeMatch recovers much of the gap.** Without ShiftAug the best
competitor is CDAN+E at 47.4 average (from 39.2); with ShiftAug, FixMatch reaches 53.7 and CDAN+E 52.9.
TimeMatch reaches 57.4 / 47.0 / 61.7 / 52.1 / 73.0, average 58.2 ± 1.8, 11 points above the best
non-ShiftAug competitor and 5 points above the best ShiftAug competitor, and still 24 points below
target-trained. Plain FixMatch without ShiftAug (37.3) is *worse* than source-only, "indicating
semi-supervised learning cannot address the cross-region task alone."

**The estimated shifts are physically sensible.** `FR1->DK1` yields `delta_{t->s} = 32` days (crops in
mid-west France mature earlier than in Denmark) and the reverse task yields `-35` days, roughly
inverse estimates from two independently trained source models. `FR1->FR2`, the geographically closest
pair, has the smallest shift and is the one task where ShiftAug slightly *hurts* most competitors,
which the authors read as shift-invariance trading short-range accuracy for long-range robustness.

**Ablation on `DK1->FR1` (Table 2).** Full TimeMatch 57.4; without domain-specific batch
normalization 56.9; entropy instead of AM 56.9; IS instead of AM 56.3; no balanced source sampler
53.3; no source shift (`delta_{s->t} = 0`) 51.9; no EMA teacher 49.9. Shift-variant TimeMatch beats
the shift-invariant ShiftAug strategy, which the authors attribute to the fact that for some pairs
(spring vs winter barley) the temporal offset *is* the class signal, so invariance destroys it.

**Sensitivity (Fig. 10).** EMA decay 0.9999 is best (1.0, a frozen teacher, is worse); pseudo-label
threshold 0.9 is best (0.95 starves rare classes); `lambda` matters less.

**Qualitative (Figs. 8–9).** Source-only confuses winter barley with winter wheat in the target;
TimeMatch separates them without target labels; t-SNE of target features is better clustered than
CDAN+E's, which is barely different from source-only.

**What is not measured.** Any non-deep baseline (no RF, gradient boosting, or linear model is run
in any configuration); pixel-level classification (the authors note `S = 1` would enable it but do
not test it); cross-year transfer; the contribution of cloud contamination (cloudy dates are left in
and never analysed); any experiment with missing or filled dates; spectral (as opposed to temporal)
shift, which the authors acknowledge exists and leave unaddressed; the effect of the class-list
mismatch between source and target on the macro-F1 denominator.

### Numbers to cite (Table 1, macro-F1 %, mean ± s.d. over three splits)

| Method | ShiftAug | DK1->FR1 | DK1->FR2 | DK1->AT1 | FR1->DK1 | FR1->FR2 | Avg. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Source-trained (no adaptation) | no | 28.3 | 29.0 | 43.4 | 24.9 | 70.3 | 39.2 ± 3.0 |
| Source-trained | yes | 40.9 | 37.4 | 48.9 | 47.3 | 70.5 | 49.0 ± 1.8 |
| FixMatch | no / yes | 24.2 / 48.2 | 28.2 / 44.2 | 37.4 / 57.4 | 26.2 / 51.3 | 70.4 / 67.7 | 37.3 / 53.7 |
| MMD | no / yes | 36.6 / 42.2 | 35.5 / 39.5 | 49.7 / 48.9 | 32.5 / 42.8 | 61.6 / 59.0 | 43.2 / 46.5 |
| DANN | no / yes | 38.7 / 45.3 | 37.3 / 44.1 | 52.0 / 52.4 | 34.0 / 42.9 | 71.0 / 68.7 | 46.6 / 50.7 |
| CDAN+E | no / yes | 39.3 / 46.5 | 37.9 / 45.2 | 51.5 / 55.0 | 36.5 / 46.9 | 71.7 / 70.7 | 47.4 / 52.9 |
| ALDA | no / yes | 36.9 / 42.8 | 33.1 / 36.2 | 47.2 / 51.5 | 35.0 / 40.7 | 55.3 / 53.8 | 41.5 / 45.0 |
| JUMBOT | no / yes | 36.8 / 42.7 | 33.6 / 38.3 | 50.5 / 49.7 | 35.6 / 41.5 | 63.7 / 62.2 | 44.0 / 46.9 |
| TimeMatch | (shift-variant) | 57.4 | 47.0 | 61.7 | 52.1 | 73.0 | 58.2 ± 1.8 |
| Target-trained (upper bound) | - | 74.6 | 72.4 | 86.9 | 90.6 | 85.7 | 82.0 ± 1.9 |

Source-only deficit relative to target-trained, by task: 46.3, 43.4, 43.5, 65.7, 15.4 points. Share of
the deficit recovered by TimeMatch: 63 %, 41 %, 42 %, 41 %, 18 %. Estimated shifts: `FR1->DK1`
+32 days, `DK1->FR1` -35 days; maximum searched shift 60 days.

Citable sentences: "source-trained models transfer very poorly to new target regions, with an average
F1-score of 39 % on target data"; "training shift-invariant models with ShiftAug improves domain
generalization, leading to an increased average score of 49 %"; "existing UDA methods, however, only
slightly increase the performance of source-trained models"; "accounting for the temporal shift is a
key component for the cross-region adaptation problem of SITS."

## Relevance to Our Crop-Classification Study

Our manuscript's revised findings are: (1) the holdout tile (`34S_20E_259N`) is missing September for
41 % of pixels, never missing in training; zero-filling it before standardization in every
raw-sequence deep model imposed a train/test missingness shift that cost those models 0.08–0.14
holdout macro-F1 and produced an apparent in-region/out-of-region ranking inversion; with September
median-filled from the training tiles, dense deep temporal pixel models lead (CNN-BiLSTM 0.66, TempCNN
0.65, L-TAE 0.65, Transformer 0.64, TimesNet 0.63), TabNet 0.62, tree/linear/MiniRocket 0.59–0.61, no
inversion. (2) A sparsemax channel gate gives no transfer gain on complete inputs and no consistent
low-data gain, but about +0.06 robustness to the zero-filled month; TabNet is not robust to zero fill
(0.61 to 0.51). (3) At 10 % of training fields every model is near 0.55 macro-F1 (XGBoost 0.51); deep
models reach about 0.65 at 100 %, logistic regression plateaus at 0.61.

**TimeMatch is the strongest published demonstration that a dense temporal attention model can fail
catastrophically across regions, and it attributes the failure to the data, not the architecture.**
The old version of our manuscript could have cited the 39.2-versus-82.0 source-only collapse of
PSE+LTAE as support for "dense temporal nets overfit the training region." Reading the paper
carefully, that would have been a mistake of the same kind as the one we have just corrected. Nyborg
et al. diagnose the collapse as a *temporal shift* of 30–35 days between Denmark and France, show
that it is largely removable by shifting the day-of-year axis (ShiftAug alone recovers 10 points;
explicit estimation recovers 19), and never test whether a tree or linear model would have done
better. The architecture is held fixed; the finding is about a specific, physically interpretable
distribution shift. Our revised finding is structurally the same story: our dense models looked like
poor transferers because of a specific input shift (a month present in training but not in the
holdout), and once that shift was handled they led. The lesson in both cases is to name the shift
before blaming the inductive bias.

**Our holdout is a `FR1->FR2`-type task, not a `DK1->FR1`-type task.** Of their five pairs, the
geographically closest (mid-west to southern France, small estimated shift) retains 70.3 of 85.7
macro-F1 with no adaptation, a 15-point loss, while the cross-climate pairs lose 44–66 points. Our
holdout tile is the next tile east (`34S_20E` versus `34S_19E`) in the same Western Cape winter-rain
zone and the same season, so crop calendars are essentially aligned and no temporal shift of the
TimeMatch kind should exist. Our corrected in-region-to-holdout gaps (dense models around 0.65 on the
holdout against higher in-region scores) are therefore of the magnitude TimeMatch predicts for a
near-neighbour pair, and well below what a true cross-climate transfer would produce. This is worth
stating explicitly in the manuscript: our holdout tests spatial generalization under a nearly shared
phenology, which is the operational case for scaling a provincial map, not the cross-continent case
TimeMatch targets. If we ever evaluate on a tile with a different rainfall regime (summer-rainfall
maize areas, for instance), the TimeMatch collapse is the expected outcome for every model that
encodes absolute timing, including trees on month-indexed xr_fresh features.

**ShiftAug and temporal subsampling are the data-level robustness tools we have not used.** The
source-only model trained with ShiftAug gains 10 F1 points on average on unseen regions with zero
target data. Together with the 30-step random temporal subsampling they already apply during source
training, this is a demonstration that augmentation along the time axis, not sparsity in the
architecture, is what makes a dense attention model transfer. Finding 2 (no transfer gain from the
sparsemax gate on complete inputs) is consistent with this: the gate changes what the model can
ignore across *channels*, whereas the shifts and gaps that break transfer are along *time*. If we
want a robustness contribution, random month-dropout and small month-shift augmentations during
training, evaluated on the holdout with September masked, median-filled, and zero-filled, would be the
TimeMatch-style control against which the gate's +0.06 should be measured.

**Irregular sequences with day-of-year encoding sidestep imputation entirely.** TimeMatch's inputs
have different lengths and dates in every tile (Fig. 5), cloudy dates are left in, and nothing is
filled; the LTAE positional encoding over DOY absorbs the differences. The same design in Garnot et
al. (2022) and SITS-BERT. Our fixed 10-month grid is what turned a missing September into a value
that had to be invented. Rebuilding the L-TAE and Transformer inputs as `(observation, month)` tokens
with a padding mask would make the holdout's missing month an absent token rather than an imputed
one, and it would let us report a "no imputation" row in Table 1 of the fill-check experiment.

**Source-only numbers are the right baseline to report, and we already do.** The paper's practice of
reporting the un-adapted source model as the lower bound, the target-trained model as the upper
bound, and every method in between is the layout our holdout table should keep: in-region CV score,
holdout score with the corrected fill, and (if we run them) holdout scores with the augmentations
above. Note that their ShiftAug and TimeMatch models still use *target* unlabeled data; our holdout
protocol uses none, so the comparable rows are only "Source-trained" and "Source-trained + ShiftAug."

**On finding 3 (learning curves).** Not addressed; TimeMatch always uses the full source training
set. The class-list rule (at least 200 parcels per source class) is a reminder that at 10 % of our
fields several minority classes fall below any such threshold, which may be why all models converge
to the same macro-F1 there: the per-class terms for rare crops are near zero for everyone.

**What this paper cannot support.** It has no classical baseline, so it cannot be cited for the
tree-versus-deep question in either direction. It should be cited for: the magnitude of
source-only collapse under a real phenological shift; the demonstration that the collapse is
removable by handling the shift; the value of temporal augmentation for zero-target-data transfer;
and the irregular-sequence input design.

## Evaluation Caveats

- **Spatial protocol is strong.** Disjoint tiles in different countries, parcel-level splits, no
  target labels used for model selection, hyperparameters tuned on one task only. This is a true
  out-of-region holdout, better than almost everything else in the corpus.
- **No non-deep baseline.** Every row is PSE+LTAE. Nothing can be said about whether trees, linear
  models, or engineered features would have transferred better or worse under the same shift.
- **Macro-F1 over a class list fixed by the source.** Target classes with few or no parcels still
  enter the macro average, and the "unknown" class is heterogeneous; this depresses absolute scores
  and inflates variance (the authors note the target-trained variance for this reason). Numbers are
  not comparable across tasks with different class lists.
- **Cloud contamination is not controlled.** Dates with up to 80 % cloud are retained and no mask is
  applied; the paper does not separate spectral noise from temporal shift in the source-only
  collapse.
- **Temporal shift only.** Spectral shift (soil, management) is acknowledged but not modelled; the
  shift is a single scalar per region pair, not class-wise or non-linear (the authors propose
  class-wise alignment and time warping as future work).
- **Single year.** All tiles are 2017; cross-year transfer, which the DOY encoding should support, is
  not tested.
- **Object-based only.** Requires parcel boundaries in the target region; the pixel-based variant
  (`S = 1`) is mentioned but not evaluated.
- **Three runs with different splits** give the reported standard deviations (1–5 points); some
  pairwise differences among competitors are within that range.
- **Hyperparameter tuning on `DK1->FR1` target validation labels** is a mild use of target labels
  for one task; the authors are explicit about it and apply the same values elsewhere.
- **Compute.** Single 1080 Ti; shift estimation requires a forward pass over target data for each of
  121 candidate shifts per re-estimation, made tractable only by the once-per-epoch schedule.
