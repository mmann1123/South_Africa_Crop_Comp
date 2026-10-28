# Did published transfer failures have a missing-data contribution?

Paper-by-paper audit (steps 1, 2 and 4 of the plan), 14 papers that report cross-region
crop-classification failure. Per-paper evidence, with section, table and code-line citations, is in
`part1.md` (7 papers) and `part2.md` (7 papers). Step 3, comparing missingness between source and
target in the public data, is still to do for the "possible" cases. 2026-10-28.

## Result

No paper reaches "likely", and none fills a target-region gap with an out-of-distribution
constant the way our pipeline did. Every pipeline either drops missing dates, interpolates the
same way in both regions, or leaves cloudy observations in raw. Where a missing-data contribution
is possible, the route is a different observation pattern between regions (cloud contamination,
gappier composites, stale nearest-date values, sample selection), not a constant fill. Every
"possible" case is also confounded by phenology, class shift or different years, and none reports
per-region observation statistics.

| Paper | Claimed failure | Missing-data handling | Verdict |
|---|---|---|---|
| Sen4AgriNet (Sykas 2022) | Catalonia→France: Transformer 39.2%, TempCNN 36.9%, LSTM 91.6% | Scenes <10% cloud, monthly medians, empty months interpolated, ends extrapolated, raw values | Possible (strongest); LSTM holds on the same inputs, large label shift |
| TimeMatch (Nyborg 2022) | Source-only macro-F1 39.2 vs 82.0 target-trained | Scenes up to 80% cloud kept, no pixel mask, irregular dates, nothing filled | Possible; estimated shifts are physically sensible, shift augmentation alone recovers ~10 F1 |
| T3S (Turkoglu 2025) | PSE+L-TAE 31.6% FR→DK, 36.5% DK→FR | TimeMatch data; least-cloudy date per bin, method on L1C not stated | Possible; gain may mix cloud filtering with thermal alignment |
| Orynbaikyzy 2022 | 7 German sites, median F1 loss S2 −0.08 | Nearest-in-time value copied onto 29 anchor dates, no mask | Possible; authors blame optical availability, sites on different orbits |
| Chen & Liu 2021 | Lindian→Hailun OA −7.2 to −41.0% | "Interpolation" of empty pixels, min-max scaling; 41 vs 48 images | Possible; authors cite temporal composition (abstract only) |
| Valinho 2024 | Mali→Uganda, accuracy declines | L1C scenes <20% cloud, pixel masking not verified | Possible; full text unreachable |
| Wijesingha 2024 | Macro-F1 loss 0.02 to 0.18 | Masked, bimonthly medians, field-years with an empty period dropped | Possible (temporal), unlikely (spatial) |
| Mohammadi 2024 | ~72–83 across years → ~45–52 across regions | ≥2 valid obs per season third, linear interpolation to weekly, same everywhere | Unlikely; label shift much larger |
| Pandžić 2026 | Naive Slovenia→Serbia F1 RF 46.5, Transformer 66.8 | Cloud Score+ mask, spline plus edge padding, same in both | Unlikely; largest drop is Sentinel-1 only |
| Hoppe 2024 | F1 82% (one date) → 61% (full season) | Single cloud-free observations, no series, no fill | Unlikely |
| Rusňák 2023 | ~5–7 OA below temporal holdout, RF worst | SCL mask, monthly medians, same for all | Unlikely; authors tested observation counts vs loss, no link |
| Xie 2025 | Up to 21.6 OA spread across source states | QA mask, monthly median, interpolation, same on both continents | Unlikely; same target data for all, explained by temperature similarity |
| EuroCropsML (Reuss) | Portugal in pre-training lowers Estonian few-shot accuracy | Irregular dates, padding masked, nothing filled | Unlikely; not a zero-shot test |
| Kerner 2024 / 2020 | 2024: map assessment, no transfer test; 2020: Togo only | 2020 code: NaN to raw 0 before the training normaliser | Not assessable / unlikely |

## Our mechanism in operational code

The Togo crop-mask code (Kerner et al. 2020, nasaharvest/togo-crop-mask at commit 61da135) contains
a partial version of our mismatch, checked against the code: at inference `predict`
(`src/models/models.py` 245–263) fills missing values with raw 0 (`nan_fill=0`) and then applies the
training mean and standard deviation (`src/models/utils.py` 52–55), with no limit on how much is
missing. Training fills with 0 the same way but drops samples more than 30% missing
(`src/engineer/base.py` `maxed_nan_to_num`, `src/engineer/togo.py` 77–79), so zeros reach training
only in moderation. The paper tests only inside Togo, and least-cloudy compositing keeps missing
timesteps rare, so this shows the pattern exists in operational code, not that it caused a
published failure.

## Also found (not missingness)

Sen4AgriNet's public code (`utils/OAD_datamodule.py` 90–99) appears to load the `_test` file as the
training set for the object-level models, which also get no input scaling. Not verified whether the
published runs used this code; if they did, its object-level table does not measure transfer.

## What the paper can say

Gap handling in the published transfer literature is usually consistent between regions; our
failure needs a hand-built monthly grid with a constant fill at inference, which none of these
papers shows causing a reported failure. Different observation patterns between regions remain an
untested confound in several well-cited failures (Sen4AgriNet, TimeMatch, T3S, Orynbaikyzy), and
almost no transfer paper reports per-region observation statistics. Recommending that they do is
a fair, modest contribution.

Counter-example and precedent (from the user's separate review, `reports/Missingness confound audit
of transfer papers.md`): Pham et al. 2024 (IJAEOG 129:103867) report annual clear-observation
counts, test missingness as the main hypothesis, and show cross-year loss tracks observation
density and is fixed by training-time observation dropout. It is the published demonstration of
the mechanism (across years, not regions) and must be cited, not audited. Briefer:
`writeup/literature/Temporally transferable crop mapping with temporal encoding and deep learning augmentations.md`.

## Data checks for step 3 (priority order)

1. Sen4AgriNet: count filled and extrapolated months per parcel by region; compare French parcels
   with zero vs one or more filled months.
2. TimeMatch / T3S: cloudy-pixel fraction per tile and date from the Zenodo archive; re-score
   source-only models with cloudy dates removed; least-cloudy selection without thermal time.
3. Orynbaikyzy: per site and anchor date, the distance to the clear date actually used, against
   per-site loss.
