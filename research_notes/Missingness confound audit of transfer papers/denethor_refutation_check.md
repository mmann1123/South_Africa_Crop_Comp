# DENETHOR refutation check (Kondmann et al. 2021, NeurIPS Datasets and Benchmarks)

Claim under test (from `experiments/external_audits/denethor/README.md`): DENETHOR's roughly 12 point out-of-year drop (train 18E-242N 2018, test 17E-243N 2019), which the paper attributes to "weather and growing cycles", is measured across an unreported shift in cloud cover and missingness. 2019 had fewer clear Sentinel-2 acquisitions than 2018: 42 vs 35 per field from April to October, and 40 vs 31 for the full year under the `cld<40` rule. The tile contributes nothing to this shift. In Planet Fusion (PF), May was 41% synthetic in 2018 and 81% in 2019.

Sources read: the main paper PDF (13 pp., proceedings, https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/5b8add2a5d98b1a652ea7fd72d942dac-Paper-round2.pdf), the supplementary material (3 pp., inside `...-Supplemental-round2.zip`), the GitHub `readme.md` and `random_forest.py` (lukaskondmann/DENETHOR, master), and the Figure 1 page rendered at 200 dpi. The prompt's proceedings hash (a8f15eda...) is wrong. The correct hash is 5b8add2a5d98b1a652ea7fd72d942dac (OpenReview id uUa4jNMLjrL).

## Verdict

The paper leaves the claim **intact as stated** (an unseparated confound) and **weakens any stronger reading** (that clouds or missingness drive the drop). Nothing in the paper or supplement disproves it. The paper reports no per-year cloud statistic, does not use the PF QA layer, and gives no evidence for "weather and growing cycles" beyond one illustrative NDVI figure.

Three things limit how far the claim can be pushed:

1. The decisive SAR test cannot be run from the paper. No Sentinel-1 (or Sentinel-2, or random forest) model is evaluated out of year. The only out-of-year decomposition uses a single Planet-only model (PselTae).
2. The same decomposition gives a 9 point drop from the tile change alone. The audit shows the tile change carries no cloud shift. So a drop of comparable size arises without any missingness change, and a 12 point drop does not need clouds to explain it.
3. 2018, the training year, was the clearest year in Brandenburg's record, and it was also a drought year. Cloudiness and weather are therefore collinear across this one year pair. Figure 1 shows a visible meadow-phenology difference between the years. Clouds and weather cannot be separated with two years, in either direction. That favours the "unseparated" framing over any "cloud-driven" framing.

## 1. All results reported (per model, per input)

### Out-of-year and out-of-region decomposition: only PselTae, Planet Fusion only

| Evaluation | Tile, year | Accuracy (Table 2, p. 8) | Macro F1 (Table A1, supp. p. 1) |
|---|---|---|---|
| PselTae (Val) | train tile 18E-242N, 2018, held-out fields | 88.02% | 83.23% |
| PselTae (2018) | test tile 17E-243N, 2018 (ablation) | 78.77% | 71.76% |
| PselTae (test) | test tile 17E-243N, 2019 | 67.25% | 58.12% |
| Spatial term (Val minus 2018 test tile) | | 9.25 pp | 11.47 pp |
| Temporal term (2018 test tile minus 2019) | | 11.52 pp | 13.64 pp |

Paper, Sec. 5, p. 8: "To split up this difference into a spatial and temporal shift, we evaluate PselTae as well on the test set in 2018 which results in 78.77% accuracy. Therefore, the spatial shift to a new tile accounts for about 9 p.p. and the temporal shift for 12 p.p. which is about 60% of the drop in total."

Table 2 caption, p. 8: "The majority of this drop can be attributed to out-of-year prediction rather than the spatial shift."

### 2019 test only, no 2018 reference: deep models with Planet Fusion (Table 2 accuracy / Table A1 macro F1)

| Spatial \ Temporal | TempCNN | MSResNet | LSTM | Transformer |
|---|---|---|---|---|
| ResNet18 | 52.22 / 30.96 | 49.53 / 29.30 | 44.64 / 18.44 | 43.61 / 14.72 |
| SqueezeNet | 53.94 / 34.73 | 49.78 / 25.10 | 35.89 / 5.86 | 42.58 / 11.19 |
| MobileNetv3 | 53.20 / 32.79 | 54.33 / 34.17 | 43.46 / 19.89 | 48.06 / 26.24 |
| Pixel average | 64.46 / 49.17 | 58.83 / 48.64 | 48.40 / 23.82 | 52.56 / 33.83 |
| PselTae | 67.25 / 58.12 | | | |
| PseTae | 64.95 / 54.50 | | | |

### 2019 test only: random forest on hand-crafted features (Table 3, p. 9; no 2018 or validation numbers reported)

| Input | # features | Accuracy | Macro F1 |
|---|---|---|---|
| Sentinel-1 | 42 | 0.58 | 0.43 |
| Sentinel-2 (cloud-masked, `cld < 40`) | 91 | 0.59 | 0.42 |
| Planet (PL) | 35 | 0.37 | 0.12 |
| S1 + S2 | 133 | 0.62 | 0.46 |
| S1 + PL | 77 | 0.60 | 0.42 |
| S2 + PL | 126 | 0.59 | 0.41 |
| S1 + S2 + PL | 168 | 0.63 | 0.46 |

Sec. 4, p. 6: "The listed models are evaluated only with Planet data as inputs. When training solely on S2 inputs, models did not converge."

**Critical SAR test: not possible.** S1 appears only in Table 3, which reports the 2019 test score alone. S1 has no validation, 2018-test-tile or 2018 train-tile score, so its out-of-year drop is unknown. Indirectly, the cloud-immune S1 forest (0.58 / 0.43) does no better on the cloudier 2019 test than the cloud-masked S2 forest (0.59 / 0.42). With no 2018 baseline for either, this parity cannot discriminate. It is mildly unhelpful to the claim: if 2019 clouds had badly degraded S2, S1 would be expected to beat it. The audit cannot run this test either, because Planet is the only input with a 2018 test-tile score. The S1 and S2 ablation tile-years (17E-243N 2018, 18E-242N 2019) are in the Source Cooperative bucket, so a 2x2 for S1 vs S2 random forests could be run locally to settle it.

## 2. Does the design separate tile and year?

Partly. The paper reports train-tile 2018 (validation), test-tile 2018 and test-tile 2019, all for PselTae on Planet only. It mentions train-tile 2019 data but runs no experiment on it.

Sec. 3, p. 4: "We will further release the 2018 test tile data and 2019 training tile data for future ablation studies."

Repo readme: "ablation study data (train tile year 2019 and test tile year 2018) ... download of the ablation study data is temporarily disabled."

The decomposition comes from one model with no seeds or intervals, and it is sequential (spatial first, then temporal). It is not a full 2x2: train-2019 → test-2019 and train-2018 → train-tile-2019 are missing. The paper does not say which labels were used for the 2018 evaluation of the test tile. The shipped bucket has only `crops_test_2019` labels for that tile, and the Planet imagery for 17E-243N 2018 is not in the bucket (see `recon_timematch_denethor_data.md`), so this number cannot be reproduced now.

Weakening point: the 9 pp spatial term arises from a tile change that, per our audit, has zero clear-count difference (tile effect 0 to +1 under all rules). A drop of the same order therefore occurs with no missingness shift. The 12 pp temporal term does not need a missingness explanation, though missingness remains possible.

## 3. Cloud handling and cloud statistics

S2 collection, Sec. 3, p. 6: "No observations are filtered because of cloud cover (maxcc=1) to maximize temporal coverage. We provide 12 bands ... together with the valid data mask, scene classification (SCL) band and the s2cloudless probability map (SCP)."

Random forest baseline, Sec. 4, p. 7: "For the 12-band Sentinel 2 and the 4-band RGB+NIR Planet data, we average all pixels of one field and consider only cloud-free observations which discards around 55% of S2 observations effectively cutting temporal resolution in half." In `random_forest.py` the mask (`msk = cld < 40`) is applied only in the S2 branch. The Planet branch uses all 365 days unmasked. The 55% figure is pooled, with no per-year split.

Related work, p. 3: "Still, on average 55% of the earth's surface is covered by clouds [19]." This is a separate, global MODIS statistic, not a DENETHOR measurement. The two 55% figures should not be conflated.

Consistency flag for our audit: our median clear counts imply discard rates above 55%. Under the literal rule they are 40/144 and 31/144 (72% and 78% discarded). Under s2cloudless < 40% they are 54/144 and 48/144 (62.5% and 67%). Under SCL they are 55/144 and 52/144 (62% and 64%). None reproduces "around 55%". The paper's rule may be applied differently (per pixel, a different CLP scale, or a different denominator). This does not touch the year contrast, but we should not claim to reproduce their 55%.

Planet Fusion gap filling, Sec. 3, p. 5: "Potential gaps because of clouds are filled with different points in time. The data also includes quality assurance (QA) information that underlines from which source any pixel is taken from together with a confidence score if the observation is gap-filled. This may be particularly useful in combination with machine learning techniques since observation could be prioritized by their confidence measure." This supports the claim: the authors knew the QA existed and called it useful, but did not use it or report it per year.

Sec. 5, p. 8: "When cloudy and raw Sentinel-2 data is used, convolutions may struggle to identify the relevant observations. In well-prepared, declouded images, temporal convolutions seem to excel." This treats PF as clean, declouded input.

The paper contains no per-year cloud statistic or discussion of cloud differences between 2018 and 2019, in the main text, the supplement or the readme.

## 4. Figures of per-year time series (Figure 1, p. 5)

Caption: "The bottom plots shows the Normalized Difference Vegetation Index NDVI [41] averaged over all pixels of these parcels across the years 2018 (above) and 2019 (below) ... wheat and meadow vary systematically between day 180 and 240 in both years. However, the vegetation activity varies notably between the years."

Text, pp. 5 to 6: "Comparing the NDVI across 2018 and 2019 underlines the difficulty of out-of-year generalization in crop type mapping. While patterns show some similarity across years, there are also significant differences, particularly for meadows in the middle of the year."

What the rendered figure shows. It plots Planet Fusion NDVI for the training-tile parcels, so the 2019 panel is 18E-242N 2019, not the test tile.

1. Wheat is similar in both years: green-up, peak around 0.85 at doy 130 to 160, and harvest decline around doy 180 to 200.
2. Meadows differ. In 2018 the highlighted meadow and its peers drop to about 0.45 around doy 185 and stay at 0.45 to 0.7 through doy 250. In 2019 meadows stay at about 0.85 from doy 160 to 220. This fits 2018 drought browning or cutting, which is a real agronomic or weather difference and not a gap artefact.
3. The 2019 spring rise is more gradual (doy 50 to 140) than the sharp 2018 rise (doy 100 to 130). Some 2019 segments are plateau-and-step shaped: the meadow plateau doy 160 to 215, then a sharp drop around doy 225. That could reflect gap filling, but the figure cannot show it.

The figure is the paper's only evidence for "weather and growing cycles". It is consistent with a genuine phenology or management difference in meadows, the largest class (954 of 2,530 training fields). That is a credible non-cloud explanation for part of the drop, and it weakens a cloud-driven reading. It does not separate the two, because 2018's clear skies and its drought share a cause.

## 5. Authors' explanation and evidence; class distribution

Sec. 6, p. 9: "Since the weather and growing cycles can vary notably from year to year, out-of-year generalization provides a challenge." Beyond Figure 1, the paper offers no evidence for this: no weather data, no phenology dates, no per-class or per-year analysis.

Class distribution: only training counts are given. Sec. 3, p. 4 lists "Wheat (305), Rye (276), Barley (137), Oats (45), Corn (251), Oil Seeds (201), Root Crops (23), Meadows (954) and Forage Crops (339)", and the readme gives the same. Neither the paper nor the supplement reports the test tile's 2019 class counts or any per-year class table, and label shift between years is not discussed. The supplement says only "Over what timeframe was the data collected? 2018-2019" (C.3). The test-tile class distribution could be computed from `brandenburg_crops_test_2019.geojson`, which we already hold, but it is not in the paper. Label or prior shift between tiles is a third uncontrolled factor besides clouds and weather. It sits in both the spatial and the temporal term, because the 2019 evaluation changes the field set's labels as well as the imagery.

## 6. Limitations and reviewer comments

Stated limitations (Sec. 6, p. 10): "First, the area covered by our dataset is comparably small ... Second, crop type datasets often lack geographic diversity." Neither clouds, gap filling nor the confounding of year effects is listed.

Datasheet (supp. C.2): "Are there any errors, sources of noise, or redundancies in the dataset? Not to our knowledge." The PF synthetic days are not mentioned as a source of noise.

OpenReview reviews: **not accessed** (Cloudflare-style challenge on both the openreview.net page and the API).

## Strongest evidence each way

Against or weakening:
1. There are no S1 out-of-year numbers, so the cloud-independent control the claim would need does not exist.
2. A 9 pp drop occurs on a tile change with no cloud shift.
3. Figure 1 shows a real meadow-phenology difference between 2018 and 2019, consistent with the 2018 drought, which is collinear with 2018 being clear.
4. The S1 random forest is no better than the S2 random forest on 2019.
5. Our clear counts do not reproduce the paper's "around 55%" discard figure.

Supporting:
1. The decomposition rests on one Planet-Fusion-only model, so it applies to the input whose synthetic share changed most by month.
2. The authors note the PF QA and gap-filled confidence (p. 5) but never use or report them.
3. No per-year cloud statistic appears anywhere. The 55% figure is pooled and applied only to the S2 random forest.
4. "Weather and growing cycles" is asserted with no supporting analysis beyond one illustrative figure.
5. Test-year class counts are not reported, so the authors did not separate label shift either.

## Could not access

1. OpenReview forum https://openreview.net/forum?id=uUa4jNMLjrL (reviews, meta-review, author responses): browser challenge blocks both the page and the API. Please download or print to PDF from a browser.
2. The ablation-study data on LRZ Sync&Share (Planet for test tile 2018 and train tile 2019): the readme says download is disabled. It would be needed to reproduce the 78.77% figure or to measure PF-QA for 17E-243N 2018.
3. Test-tile 2018 labels: not identifiable in the paper or the bucket, so it is unknown which labels the 78.77% evaluation used.
