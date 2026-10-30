# Mai et al. 2025: do the weekly composites of the US source and the Hexi target have different gap structure?

Mai, Feng, Fu, Wang, Zhang, Zhang and Liang, Remote Sensing 17:1494 (2025), doi 10.3390/rs17091494.
Instance-based transfer of five crops from CDL 2022 (Iowa, Missouri, Kansas, Minnesota, North and South Dakota) to field-survey points in the Hexi Corridor, Gansu, with Sentinel-1, Sentinel-2 and Landsat-8 features and tree models.
Naive-transfer XGBoost reaches macro F1 0.448, and the domain gap is explained with KS tests and NDVI DTW distances. No code; data "available on request".
This audit measures the gap structure their Sentinel-2 recipe leaves in each domain. No model is re-run.

## What the paper pins down (step 0)

Pinned: calendar year 2022; Sentinel-2 cloud removal with QA60; weekly maximum-value composites; linear interpolation of the removed values; source = CDL 2022 pixels above 95 percent confidence in the six named states, five crops (maize, alfalfa, oats, canola, spring wheat);
target extent 36 29'-42 48' N, 92 20'-104 16' E; target samples "affected by cloud cover were discarded", source samples were only confidence-filtered.
Not pinned: the collection (Table A1 lists B9 and B10, which only L1C carries, so `COPERNICUS/S2_HARMONIZED` is used), week edges, the Hexi cropland mask (unnamed), the rule behind "affected by cloud cover", whether QA60 was applied as an update mask.
That is enough to measure empty and interpolated weeks and residual cloud in both domains; the numbers below are for this reconstruction, not the authors' files.

## Method

1. `01_points.py`: per state up to 60 seeded CDL 2022 pixels of each of the five crops (confidence above 95), 1,131 points (Iowa has no canola or spring wheat points, Missouri 68 points in all). Target: 1,800 seeded ESA WorldCover v200 cropland pixels inside Jiuquan, Jiayuguan, Zhangye, Jinchang and Wuwei (GAUL level 2) and the paper's extent. Draws are on a 300 m grid (30 m was too slow over whole states).
2. `02_weekly_counts.py` (`eelib.py`): for 2021, 2022 and 2023, 52 seven-day bins from 1 January (the paper gives no week edges; 31 December is dropped). Per point and week it counts L1C images inside the footprint (n), passing QA60 (bits 10 and 11 clear, the paper's mask), Cloud Score+ `cs_cdf` clear (>= 0.5 and >= 0.65), and passing both. A QA60 value that is absent is treated as no flag.
3. `03_qa60_images.py`: per image, share of 60 points per group (Hexi, Iowa, North Dakota) where QA60 flags cloud, the Cloud Score+ cloud share and the ESA granule cloud percentage, for L1C and L2A, 2021 to 2023.
4. `04_summarise.py` (`hexilib.py`): a weekly composite is valid under four definitions: no mask, QA60 (the paper), Cloud Score+, both. Valid weeks, interpolated weeks (empty between first and last valid), edge weeks, longest runs, a growing-season window (weeks 14 to 44), residual cloud (share of QA60-passed observations that Cloud Score+ flags, and weeks that contain at least one), monthly QA60 flag rates, and how many points a cloud rule would drop.
5. `05_figure.py`: `figures/hexi_gap_structure.png` and `.pdf`.

## Results

QA60 check. In Earth Engine the QA60 cloud polygons stopped being produced in early 2022 (catalogue: "masked out between February 2022 and February 2024"). The data agree for Hexi and only partly for the Midwest. Share of Sentinel-2 images with any QA60 cloud flag at the sample points (`qa60_images.csv`, L1C; L2A is the same):

| Domain | 2021 | 2022 Jan | 2022 Feb to Dec | 2023 |
|---|---|---|---|---|
| Hexi | 11 to 39 percent a month | 11 | 0 | 0 to 4 |
| Iowa, North Dakota | 25 to 90 percent | 71 to 85 | Feb to Apr 65 to 82, then 36 to 40 in May falling to 1 to 12 percent in Aug to Oct, 31 to 59 in Nov to Dec | mixed, 0 to 73, mostly 0 from July |

At pixel level (`monthly_qa60.csv`) the share of observations flagged is 0.64 in Jan to Mar 2022 over the Midwest, 0.02 in September and 0.41 in December, against 0.00 for Hexi from February. QA60 values are present (zeros), not masked, in both domains, so a literal `updateMask` would keep every pixel. So the QA60 mask was a no-op for Hexi in 2022 and only partly effective in the Midwest, and it was working in both in 2021 (flagged observations 53 percent Midwest, 20 percent Hexi).

Weekly series, 2022, 52 weeks (point means; `weekly_gap_summary.csv`):

| Definition | Domain | Valid weeks | Interpolated share | Edge share | Longest interpolated run | Season weeks valid of 31 |
|---|---|---|---|---|---|---|
| No mask | Source | 51.9 | 0.00 | 0.00 | 0.1 | 31.0 |
| No mask | Hexi | 52.0 | 0.00 | 0.00 | 0.0 | 31.0 |
| QA60 (paper) | Source | 41.9 | 0.18 | 0.02 | 2.9 | 27.8 |
| QA60 (paper) | Hexi | 52.0 | 0.00 | 0.00 | 0.04 | 31.0 |
| Cloud Score+ 0.5 | Source | 41.3 | 0.20 | 0.01 | 2.3 | 25.0 |
| Cloud Score+ 0.5 | Hexi | 45.9 | 0.12 | 0.00 | 2.1 | 25.9 |
| QA60 and Cloud Score+ | Source | 36.0 | 0.28 | 0.03 | 3.4 | 23.8 |
| QA60 and Cloud Score+ | Hexi | 45.8 | 0.12 | 0.00 | 2.1 | 25.9 |

By state (QA60, 2022) the interpolated share runs from 0.14 (Kansas) to 0.20 (South Dakota); with Cloud Score+ added, 0.20 (Kansas) to 0.31 (Minnesota). In 2021, when QA60 worked in both, the paper's recipe gives 0.30 interpolated weeks for the source and 0.09 for Hexi. Images per week are alike (2.7 source, 2.5 Hexi, overlapping tiles included).

Residual cloud (2022, `residual_cloud.csv`): of the QA60-passed observations, Cloud Score+ (0.5) flags 26.7 percent in the source and 24.1 in Hexi; 37.7 and 37.8 percent of QA60-valid weeks contain at least one such observation (the contributor a maximum composite would pick up); 14.2 and 11.8 percent of weeks have only flagged observations. At 0.65: 30.0 and 27.2, 41.5 and 41.9. In 2021 the residual was 5 percent in the source and 8 in Hexi.

A cloud-affected rule (`discard_rules.csv`, 2022, growing-season weeks 14 to 44). Under the paper's QA60 the weekly series of every Hexi point is complete, so no target sample is dropped by an empty-week rule (source: 93 percent have at least one empty season week). With Cloud Score+ in place of QA60: 94 percent of Hexi points have at least one empty season week, 30 percent have at least a quarter of them empty, 17 percent a gap of 4 weeks or more (source with both masks: 99, 43 and 21 percent).

## Finding

The hypothesis is supported in direction and refuted in part. The source series are more gappy and more interpolated than the target series under every definition that removes cloud: in 2022 the paper's QA60 recipe leaves 18 percent of source weeks interpolated against about zero for Hexi, and with a Cloud Score+ cloud definition 28 against 12 percent (longest runs 3.4 against 2.1 weeks); in 2021, when QA60 worked in both, 30 against 9 percent. So the source and target feature series differ in gap structure before phenology, which a KS test on week-indexed features would register.
But the 2022 contrast under the paper's own recipe is partly a product of QA60, not of climate. QA60 cloud flags were still delivered over the Midwest through spring and fell through summer, and were absent over Hexi from February, so the paper's mask cleaned the source and left Hexi untouched, and Hexi's weekly series have no interpolated weeks at all. The residual cloud is the same: about a quarter of QA60-passed observations are cloud in both domains and 38 percent of weeks in both carry a contaminated contributor, so the prediction that the source carries more cloud that QA60 misses fails; both carry plenty. The paper's phrase that cloud-affected target samples were discarded cannot have been a QA60 rule in 2022, since QA60 flags almost nothing over Hexi (zero empty weeks); it must have been visual or NDVI-curve screening, which is unmeasurable here.
The confound is therefore a real but different one: the source series are 2.3 times as interpolated (Cloud Score+) and the target series are effectively unmasked and cloud-contaminated at the same rate, with the amount depending on an undocumented QA60 outage that the paper does not mention. Whether it contributes to the naive-transfer macro F1 of 0.448 is not testable without their data. Verdict: supported as a gap-structure difference, not shown to explain the loss; confidence medium.

## Limitations

Points are CDL pixels in six states (a 300 m grid, up to 60 per crop per state, Iowa and Missouri thin) and WorldCover cropland in five Gansu prefectures, not the authors' 1,131 source and 893 to 1,272 target samples, and the Hexi points are not stratified by crop. The collection is assumed to be L1C; week edges are assumed (7 days from 1 January); the QA60 mask is assumed to have been applied literally with a missing value read as no flag, and a differently coded mask could instead have emptied 2022. Cloud Score+ is a model, not truth: at 0.5 it flags 24 percent of Hexi observations and 58 percent in August, and may over-flag bright desert or thin cloud, so the Cloud Score+ rows are an upper bound on gaps; the 0.65 variant is in `residual_cloud.csv`. A maximum composite of NDVI is less hurt by cloud than one of reflectance, so contamination in the table is a potential, not a measured F1 effect. The QA60 flag pattern in the Midwest (a decline from May and a mixed 2023) is reported as seen and not explained; images per week count overlapping tiles. Sentinel-1, Landsat-8 (monthly composites) and the interpolation scheme beyond linear are not measured. `run_log_02_weekly_counts.json` records only the last of the per-year runs (2021); all three years are in the chunk files and in `weekly_gap_summary.csv`. 03 is limited to Hexi, Iowa and North Dakota.
