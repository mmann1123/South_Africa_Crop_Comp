# Sen4AgriNet refutation check and measurement specs (Sykas, Sdraka, Zografakis, Papoutsis 2022, IEEE JSTARS 15)

Claim under test (card in `reports/Missingness confound audit of transfer papers.md`, row 3 and paragraph at line 114): Sen4AgriNet keeps Sentinel-2 scenes under a 10% scene-cloud filter and applies no pixel cloud mask. The model loader builds monthly medians and fills empty months with `interpolate_na(dim='time_bins', method='linear', fill_value='extrapolate')`, which the paper does not describe. Catalonia and France differ in clear-scene availability, so Scenario 2 (Catalonia 2019+2020 to France 2019) and Scenario 3 (France 2019 to Catalonia 2020) mix different interpolation and extrapolation rates. The paper attributes the drops to phenological stretching and label imbalance.

Sources read: the arXiv PDF 2204.00951 (20 pp., same text and tables as the JSTARS article, which is open access at doi 10.1109/JSTARS.2022.3164771; page numbers below are arXiv pages); `Orion-AI-Lab/S4A-Models` at commit ae5a376 (2022-09-20, full git history checked); `Orion-AI-Lab/S4A` at commit 7ac66dc; the Hugging Face dataset `orion-ai-lab/S4A` (file listing, sizes, and the `time` coordinate of 979 patch NetCDFs, read by HTTP range requests); the authors' Dropbox reproduction bundle (`S4A-Models.zip`, 13.7 GB), from which I streamed and kept only the nine published COCO split files and the Scenario 2 and 3 OAD test CSVs; and AWS Earth Search STAC (`sentinel-2-l1c`) granule metadata for all 16 tile-years. No downloaded code was executed. Downloads are in `/tmp/claude-1000/-home-mmann1123-Documents-github-South-Africa-Crop-Comp/6a949ad2-1d7c-45e4-b537-97b8e63decc6/scratchpad/s4a_read/`; my read-only analysis scripts are in `.../scratchpad/s4a_scripts/`.

## Verdict

**Mechanism confirmed and measured. The "different rates" part is partly wrong. The confound survives as "different months filled". There is also one finding that matters more than missingness for the OAD table.**

1. **The undescribed fill is real and it fires on the published inputs.** It fires for both PAD and OAD, not only PAD. The OAD CSV builder applies the same monthly-median, linear-interpolation and extrapolation chain over all 13 bands. The paper describes OAD as aggregating "each available timestamp" (p. 7). The code does not do that.
2. **Measured fill rates.** These come from published split membership plus patch time stacks, and the OAD test CSVs cross-check them directly.
   - PAD window (Apr to Sep). Scenario 2: 9.8% of patch-months are filled in the Catalonia train set and 14.9% in the France test set. Scenario 3: 15.8% in the France train set and 15.0% in the Catalonia 2020 test set.
   - Overall rates in Scenario 3 are therefore equal. **The filled months differ.** Catalonia 2020 is missing April in 90% of test patches and never June. France 2019 is missing April in 48% and June in 44%.
   - France also has fewer observations: about 24 per patch-year against about 37 to 38 in Catalonia.
   - The hypothesis as worded ("different interpolation/extrapolation rates") holds for Scenario 2 and is wrong for PAD Scenario 3. "Different months interpolated" holds for both.
3. **The OAD window is May to Nov in the code, not Apr to Sep.** `OAD_datamodule.py` line 53 slices month indices 4 to 10. On that window the asymmetry is large:

   | Scenario | Train, filled patch-months | Train, extrapolated | Test, filled patch-months | Test, extrapolated |
   |---|---|---|---|---|
   | 2 | Catalonia 1.4% | n/a | France 19.8% | 9.3% (Oct to Dec trailing) |
   | 3 | France 20.7% | n/a | Catalonia 0.0% | n/a |

   The Transformer and TempCNN collapse in Scenario 2 (F1 31.5 and 30.1). That fits a model that never saw interpolated months meeting a test set where 93% of patches have at least one.
4. **This is the strongest result against a clean missingness reading of Table IV, and against the paper too.** As released, `OAD_datamodule.py` loads `{stem}_test.csv.gz` for the fit, validate and test stages (lines 91 to 117). This has been true since the initial commit; the full history shows no change. If the published OAD runs used this code, the OAD models were trained on the test file. Table IV would then not be a transfer result at all, and neither phenology nor missingness explains it.
   - Two things are consistent with a leak: LSTM Scenario 2 scores higher than Scenario 1 (F1 91.34 vs 88.03), and its Scenario 2 confusion matrix looks near-diagonal (Fig. 12b).
   - Two things are not: the Scenario 3 OAD scores (57 to 64) and the Transformer and TempCNN scores in Scenario 2 (about 30) are low for train-equals-test. Short training (10 epochs) could explain them, or the published runs may have used a different local datamodule.
   - **I cannot settle which.** Either way, Table IV should not carry weight in the audit.
5. **Nothing in the paper disproves the confound.** It reports no per-tile or per-year observation counts beyond "30 to 50 timestamps depending on the region" (p. 6). It runs no cloud-controlled or fill-controlled experiment. It gives no per-month analysis. The cloud-mask option (Hollstein) exists in the loader but is off by default and is never used by any model or loss.
6. **What does weaken the confound** is phenology and label evidence that does not depend on the fill:
   - Fig. 5 shows France 2019 NDVI peaking about a month later than Catalonia for wheat, oats and peas.
   - Fig. 10 shows wheat pixels "two orders of magnitude" rarer in the Scenario 2 train set.
   - The UMAP (Fig. 9) separates Catalonia from France.
   - The Scenario 3 PAD overall fill rate matches between train and test, yet PAD Scenario 3 is the worst result (F1 34 to 44). A rate-based story alone cannot produce that gap.
   - None of this separates the two causes, because the filled months (April, June) are exactly where the phenology shift sits.

**Bottom line:** the card's claim should be restated.
- Drop "the domains differ in interpolation rate" for PAD Scenario 3. Replace it with "the domains have different months filled: Catalonia 2020 is missing April almost everywhere, France 2019 is missing April or June in about half its patches, and the OAD window has large train/test fill asymmetry in both scenarios".
- Add the OAD train-on-test-file defect as a separate and more serious caveat.
- Change the product level from L2A to L1C.
- Confidence that the confound is present and unseparated: high. Confidence that it drives the drop: low to moderate, because a phenology shift sits in the same months.

## 1. Results tables (exact numbers)

Table III, p. 9, PAD (pixel-level, weighted metrics, monthly medians Apr to Sep, bands B02 B03 B04 B08):

| Scenario | Model | Acc. W (%) | F1 W (%) | Precision W (%) |
|---|---|---|---|---|
| 1 | U-Net | 93.70 | 82.61 | 86.64 |
| 1 | ConvLSTM | 94.72 | 85.18 | 86.86 |
| 1 | ConvSTAR | 92.78 | 80.38 | 83.33 |
| 2 (Cat 19+20 to FR 19) | U-Net | 83.12 | 57.85 | 61.57 |
| 2 | ConvLSTM | 82.53 | 56.56 | 60.57 |
| 2 | ConvSTAR | 79.52 | 52.15 | 58.98 |
| 3 (FR 19 to Cat 20) | U-Net | 72.11 | 43.54 | 68.42 |
| 3 | ConvLSTM | 69.86 | 40.47 | 66.17 |
| 3 | ConvSTAR | 69.07 | 34.45 | 67.43 |

Table IV, p. 10, OAD (parcel mean and std per band; per code, months May to Nov, 13 bands):

| Scenario | Model | Acc. W (%) | F1 W (%) | Precision W (%) |
|---|---|---|---|---|
| 1 | LSTM | 88.52 | 88.03 | 87.85 |
| 1 | Transformer | 88.36 | 88.10 | 87.90 |
| 1 | TempCNN | 90.08 | 89.97 | 90.01 |
| 2 | LSTM | 91.55 | 91.34 | 91.31 |
| 2 | Transformer | 39.17 | 31.45 | 58.52 |
| 2 | TempCNN | 36.90 | 30.14 | 60.71 |
| 3 | LSTM | 60.60 | 63.96 | 70.55 |
| 3 | Transformer | 51.21 | 56.71 | 67.76 |
| 3 | TempCNN | 52.32 | 57.38 | 68.35 |

The S4A-Models README reproduces both tables identically. All metrics are sample-weighted ("Due to high class imbalance, we weigh all metrics with the samples of each class", p. 8). No macro F1 and no per-class numbers are given. The confusion matrices (Fig. 11 PAD p. 18, Fig. 12 OAD p. 19) are images without readable numbers at PDF resolution.

## 2. Quotes bearing on the claim (arXiv pages)

**Data and cloud filter, p. 6:** "All Sentinel-2 L1C images with less than 10% cloud coverage are selected for download and each image is split into 900 non-overlapping patches." This is L1C, not L2A. The S4A-Models README also says "sub-datasets of Sentinel-2 L1C images".

**Timestamps, p. 6:** "Each 'spectral' variable contains all the ntime data acquisitions (depending on the region ntime varies from 30 to 50 timestamps for one year)." This is the only availability statistic in the paper.

**Empty months deferred to code, p. 6:** "Please note that the aforementioned code repository explains both how instances of the same patch transform through time and how empty-data months are treated."

**Monthly medians, p. 7:** "we aggregated the data into 12 time-bins by calculating the median of each month, and used a fixed window including the medians of months 4 (April) through 9 (September) for model training. Only bands Red, Green, Blue and Near-Infrared were selected".

**OAD described as per-timestamp, which contradicts the code, pp. 6 to 7:** "The pixels contained within each individual parcel are aggregated via the mean function to a single number. This is repeated for each available timestamp within the netCDF variable."

**Phenology attribution, Scenario 2 OAD, pp. 10 to 11:** "The adaptation of LSTM can be attributed to the fact that time is encoded as part of the feedback process, while transformers handle data as a sequence. ... the phenological cycles do not simply shift across time, but are also stretched or shrunk down depending on the soil-climate conditions of the area."

**Scenario 3 attribution, p. 11:** "the growing season across the regions is different for the same crops (mainly due to different climatological conditions). The second component is that different years have slightly different starting dates for crop cultivation."

**Label imbalance, p. 11:** "in scenario 2 the number of wheat pixels are two orders of magnitude less in the train set than in the test set. The inverse is observed in scenario 3."

**PAD class pattern, p. 9:** Scenario 2 has "wheat, rapeseed and sunflower being the most correctly classified crops". In Scenario 3, "only maize and rapeseed display the highest accuracy". p. 12: "The most challenging classes to distinguish in scenario 2 are cereal-based crops, such as wheat, maize, sorghum, barley."

**Repo S4A README:** "If you plot right now, you might notice that some months are empty". The example patch `2020_31TCG_patch_14_14` (43 timestamps, 2020-01-08 to 2020-11-18) has April and December empty (`images/per_month_empty.png`). April is inside the training window, and this is a Scenario 3 test tile-year.

## 3. Evidence against the hypothesis (adversarial read)

1. **Phenology evidence exists independently of the fill.** Fig. 5 (p. 7) plots monthly mean NDVI for wheat, sorghum, oats and peas by region-year. France 2019 peaks around May at higher NDVI. Catalonia peaks earlier, around March and April. Fig. 9 (UMAP, p. 12) separates Catalonia from France. This is genuine evidence of a phenology shift. However, Fig. 5 is itself computed from monthly medians, and April is the month most often interpolated in Catalonia. The Catalonia April points are therefore partly linear interpolations of March and May.
2. **The label shift is large and documented** (Fig. 10, p. 13). Wheat is 2 orders of magnitude rarer in Scenario 2 train pixels. Our OAD test CSVs confirm it: Scenario 3 test (Catalonia 2020) has 155 wheat parcels against 50,461 barley, while Scenario 2 test (France) has 39,413 wheat. Weighted F1 under this shift can collapse without any input artifact.
3. **Overall PAD fill rates are equal in Scenario 3.** Train is 15.8% and test is 15.0% of patch-months. If transfer loss tracked the fill rate, PAD Scenario 3 should not be the worst scenario. It is the worst (F1 34.5 to 43.5).
4. **Within-class fill does not track the class pattern.** On the France test parcels (Scenario 2), sunflower has the highest PAD fill share (26%; April filled 86%, June 72%). Sunflower is one of the best-classified Scenario 2 classes. Potatoes have the lowest fill (3.6%). This is weak evidence against a "filled parcels fail" mechanism, but per-class scores are not published, so it cannot be tested further.
5. **Each domain has internal gaps of its own.** Every Scenario 2 and 3 train and test split contains filled months. No experiment pairs a gap-free source with a gappy target, except OAD Scenario 2 (train 1.4%, test 19.8%) and its mirror, OAD Scenario 3.

Things I looked for and did not find: per-tile-year observation counts, per-month analysis, any cloud-masked or fill-free control, a raw-timestamp input variant, and per-class or macro scores. The paper and repos contain none of these.

## 4. Evidence for the hypothesis

1. Code path (all files in `Orion-AI-Lab/S4A-Models` at ae5a376):
   - `utils/PAD_dataset.py` lines 293 to 299: `groupby_bins('time', bins=date_range, right=True, include_lowest=False, labels=date_range[:-1]).median(dim='time')`.
   - Line 302: `resample(time_bins='1MS').median`.
   - Line 307: `interpolate_na(dim='time_bins', method='linear', fill_value='extrapolate')`.
   - The same chain appears in `export_medians_multi.py` lines 126, 135 and 140 (the precomputed-median path the README command uses, `--saved_medians`), in `object-based-csv.py` lines 69 to 83 (the OAD builder), and in `S4A/patches_dataset.py` lines 284, 293 and 298.
   - `max_gap` is not set, so any run of empty months is filled.
   - With `fill_value='extrapolate'`, xarray uses scipy `interp1d` linear extrapolation, so leading and trailing months are extrapolated linearly from the two nearest months and can leave the data range.
   - The class docstring (`PAD_dataset.py` lines 31 to 38) wrongly says gaps are filled by "the median of all data from previous and following bins" and edge months by replication.
2. No cloud mask:
   - `pad_experiments.py` line 167 has `--return_masks` defaulting to False. Lines 169 to 175 have `--clouds`, `--cirrus`, `--shadow` and `--snow` all defaulting to False.
   - `PAD_dataset.py` lines 510 to 517 compute Hollstein masks only as an extra output. No model or loss reads `masks` (searched across the repo).
   - The README training command passes no mask flags.
   - The OAD path has no mask option at all.
   - Under a 10% granule-cloud filter, residual clouds and shadows over a 3.66 km patch enter the medians unmasked.
3. Measured asymmetries (Section 6): France has fewer observations, about 24 vs 37 to 38 per patch-year. The filled months sit in different places. The OAD window has a 1.4% vs 19.8% (Scenario 2) and 20.7% vs 0% (Scenario 3) train/test fill asymmetry, with Oct to Dec extrapolation on the France side.
4. The paper's "30 to 50 timestamps depending on region" understates the spread. Patch stacks range from 9 to 67 per year (Section 6).

## 5. Measurement specs (what to reproduce)

| Item | Value | Source |
|---|---|---|
| Product level | Sentinel-2 **L1C** (top of atmosphere), DN 0 to 10000 stored as uint16; normalised by `NORMALIZATION_DIV = 10000` | paper p. 6; S4A-Models README; `utils/settings/config.py` line 186; NetCDF dtype uint16 |
| Scene filter | "less than 10% cloud coverage", granule level, downloaded from SciHub and/or AWS (p. 11). The exact metadata field is not stated. Earth Search `eo:cloud_cover` < 10 (the granule CLOUDY_PIXEL_PERCENTAGE) reproduces the full-coverage patch stacks month by month (Section 7) | paper pp. 6, 11 |
| Catalonia tiles | 31TBF, 31TCF, 31TCG, 31TDF, 31TDG (2019 and 2020) | `utils/settings/mappings/mappings_cat.py` lines 4 to 6; HF listing |
| France tiles | 31TCJ, 31TCL, 31TDK, 31TDM, 31UCP, 31UDR (2019 only) | `mappings_fr.py` lines 4 to 6; HF listing |
| Scenario definitions | Exp 2: train/val = Catalonia tiles, years 2019 and 2020; test = France tiles, 2019. Exp 3: train/val = France 2019; test = Catalonia 2020 | `coco_data_split.py` lines 147 to 156 |
| Published split sizes (COCO) | Exp 2: train 3025, val 1009, test 1014 patches. Exp 3: train 3000, val 1001, test 1003 | Dropbox `coco_files/exp{2,3}_patches5000_strat_coco_{train,val,test}.json` |
| Exp 2 train composition | 2019: TBF 188, TCF 297, TCG 506, TDF 52, TDG 459. 2020: TBF 201, TCF 300, TCG 509, TDF 53, TDG 460 | COCO |
| Exp 2 test (France 2019) | TCJ 153, TCL 173, TDK 179, TDM 163, UCP 184, UDR 162 | COCO |
| Exp 3 train (France 2019) | TCJ 516, TCL 505, TDK 507, TDM 506, UCP 491, UDR 475 | COCO |
| Exp 3 test (Catalonia 2020) | TBF 135, TCF 193, TCG 338, TDF 43, TDG 294 | COCO |
| Patch geometry | 366x366 at 10 m (3.66 km); 30x30 = 900 patches per tile; files named `{year}_{tile}_patch_{row}_{col}.nc`; 20 m and 60 m bands repeated to 10 m | paper p. 6; `PAD_dataset.py` lines 316 to 321 |
| Per-patch time stack | Only acquisitions with **complete** data over the patch are kept. Every stack I checked has 0% zero pixels at every timestamp. Patches at swath edges therefore lose a relative orbit: inside one tile-year, stacks range from 14 to 32 (31TCL), 19 to 50 (31TCJ) or 33 to 67 (31TCF) | HF NetCDF reads (Section 6) |
| Time bins | Calendar months, `pd.date_range(year-01-01, year+1-01-01, freq='1MS')`, intervals (start, end] with `right=True` (so an acquisition at exactly 00:00 on the 1st falls in the previous month; S2 overpasses are about 10:40 to 11:05 UTC, so this never bites) | `PAD_dataset.py` lines 282 and 293 to 299 |
| PAD window | Months 4 to 9 (Apr to Sep), 6 steps, via `--fixed_window`: `load_medians` uses bins 3 to 8 | `PAD_dataset.py` lines 359 to 366; README command |
| PAD caveat | `fixed_window` is honoured only on the `--saved_medians` path (`load_medians`). On the on-the-fly path, `get_medians` uses a rolling `window_len` (default 6) from `start_bin` | `PAD_dataset.py` lines 418 to 437 |
| OAD window | Month indices 4 to 10 = **May to Nov**, 7 steps, 13 bands x {mean, std} = 26 features per step | `utils/OAD_datamodule.py` line 53 (`iloc[:, 4*26 : -(26+3)]` after sorting columns `MM_BAND_stat`) |
| OAD split defect | Train, val, test and predict all load `{stem}_test.csv.gz` | `utils/OAD_datamodule.py` lines 91 to 117 (unchanged since initial commit db68141) |
| When interpolation fires | Per patch-file and band, for each month bin with **no acquisition in that patch's stack**. The patch's stack is shared by all its pixels, and stored data have no NaN or zeros, so in practice the fill is patch-wide (never per pixel). Interior empty months are interpolated linearly in time. Months before the first or after the last acquisition of the calendar year are linearly extrapolated. | code above plus NetCDF checks |
| Granularity of emptiness | Mostly tile-year and orbit driven: a month is empty for every full-coverage patch of a tile if no granule passes the < 10% filter. Edge patches add extra empty months because they miss one orbit. | Section 6 |

## 6. Measured fill rates for the published splits

Method: I sampled 25 random patches per tile-year in each split (979 patch NetCDFs, time coordinate read by range requests) and weighted them by the split's tile-year composition from the COCO files. I cross-checked against the OAD test CSVs, flagging a month as filled when all 13 band means lie exactly (relative error < 1e-6) on the time-weighted line through the neighbouring months.

| Split | Obs per patch-year (mean) | PAD Apr to Sep: patch-months filled | PAD: patches with at least one filled month | OAD May to Nov: filled | OAD: patches with at least one filled month | Main empty months |
|---|---|---|---|---|---|---|
| Exp 2 train (Cat 19+20) | 38.2 | 9.8% | 58.6% | 1.4% | 9.5% | Apr 58%, Dec 40%, Mar 10% |
| Exp 2 test (FR 19) | 24.3 | 14.9% | 64.8% | 19.8% (9.3% extrapolated) | 93.2% | Nov 65%, Dec 61%, Apr 45%, Jun 41%, Jan 34%, Oct 29% |
| Exp 3 train (FR 19) | 23.6 | 15.8% | 65.9% | 20.7% (10.0% extrapolated) | 91.5% | as Exp 2 test (Apr 48%, Jun 44%) |
| Exp 3 test (Cat 20) | 37.2 | 15.0% | 89.8% | 0.0% | 0.0% | Apr 90%, Dec 34% |

CSV cross-check:
- Exp 2 test, 309,010 parcel rows: PAD-window filled share 14.2% (all parcels) or 13.8% (the 11 study classes). OAD window 17.5% / 17.3%. By month: Apr 36%, Jun 45%, Oct 33%, Nov 42%.
- Exp 3 test, 329,839 rows: PAD 15.7% / 15.3%. April filled in 94.5% / 91.6% of parcels. OAD window 0.1% / 0.0%.

Per tile (patch sample):
- **Catalonia 2019:** 31TBF has Apr, Nov and Dec empty in 100% of patches (27 obs). 31TCF and 31TDF have none. 31TCG has Apr empty in 44%. 31TDG has Mar empty in 64%.
- **Catalonia 2020:** Apr is empty in 100% of 31TBF, 31TCF and 31TCG patches, 74% of 31TDG and 43% of 31TDF.
- **France 2019:**
  - 31TCL (median 14 obs): Jan, Apr and Nov empty in 100% of patches; Jun 74%; Oct 72%.
  - 31TCJ: Apr, Jun and Nov empty in 81%.
  - 31TDM: Apr empty in 100%.
  - 31UCP: Jun empty in 100%.
  - 31UDR: Oct empty in 100%.
  - 31TDK: Nov and Dec empty in 100%.
  - One 31UDR patch (`patch_29_29`) has only 9 acquisitions, ending 2019-04-21. The loader extrapolates May to Dec for it.

Class-level fill on the France test parcels (PAD window share filled): Sunflower 26%, Sorghum 23%, Grapes 18%, Oats 17%, Rye 15%, Wheat 13%, Rapeseed 13%, Barley 13%, Maize 9%, Peas 9%, Potatoes 4%. On Catalonia 2020 test parcels every class is between 12% and 17%, because April is missing for all of them.

## 7. Can the empty-month rate be reproduced from scene metadata (GEE) or does it need the NetCDFs?

**Tile-level: yes, from metadata.**
- I tested Element84 Earth Search `sentinel-2-l1c`, granule `eo:cloud_cover` < 10, filtered by MGRS tile, with 2019 de-duplicated by sensing time because reprocessed baselines appear twice.
- It reproduces the monthly counts of full-coverage patches exactly or within 1 for most tile-years. Exact matches include 31TCL 2019 `[0,6,5,0,1,1,7,4,4,1,0,3]`, 31UCP 2019, 31UDR 2019, and 31TCG 2020 `[3,5,2,0,3,3,11,8,7,3,3,0]`. 31TDF and 31TDG 2020 match from March onward.
- The only discrepancies are 1 to 3 extra granules in January and February. These are probably granules the authors' pipeline did not ingest.
- The equivalent GEE recipe:
  - Collection: `COPERNICUS/S2_HARMONIZED` (L1C; the dataset is L1C, so not `S2_SR_HARMONIZED`).
  - Filters: `MGRS_TILE`, then `CLOUDY_PIXEL_PERCENTAGE < 10`. The paper does not say whether the field was `CLOUD_COVERAGE_ASSESSMENT`; for L1C the two are normally equal.
  - De-duplicate by `GENERATION_TIME` / sensing date.
  - Count per calendar month in Apr to Sep (PAD) or May to Nov (OAD).

**Patch-level: metadata plus geometry.** Patch stacks keep only acquisitions whose valid-data footprint fully covers the 3.66 km patch, so edge patches lose an orbit. In GEE this means testing `system:footprint.contains(patch_polygon)`. The patch polygon is the tile's UTM origin plus (row, col) x 3660 m. Each NetCDF stores `transform` and `crs` attributes per band, so exact polygons can be taken from one file per patch.

Two caveats limit reproduction from GEE alone:
- **Split membership:** which patches are in each split comes only from the Dropbox COCO files. They are large because they embed annotations: 112 to 689 MB each, inside the 13.7 GB `S4A-Models.zip` (Dropbox ignores Range requests, so the zip must be streamed to about 13.5 GB to reach them).
- **Parcel weighting:** this needs the parcel rasters inside the NetCDFs, or the OAD CSVs (about 0.7 GB per test split; about 2.1 to 2.3 GB per train split).

**Dataset size if the NetCDFs are needed:**
- The Hugging Face `orion-ai-lab/S4A` holds 10,013 patch files, 281 GB in total.
  - Catalonia 2019 (5 tiles): 83 GB.
  - Catalonia 2020: 78 GB.
  - France 2019 (6 tiles): 120 GB.
  - Patches are 7 to 67 MB each, and size scales with the number of timestamps.
- Full files are not necessary for the time axis. Reading only `B02/time` by HTTP range (h5py over fsspec) costs a few hundred KB per file but is throttled to about 20 files per minute, so all 7,659 Scenario 2 and 3 patches would take about 6 hours. A stratified sample (as here) takes about 30 minutes.

## 8. Could not access or verify

- **Whether the published Table IV runs used the released `OAD_datamodule.py`** (train on the `_test` file) or a corrected local version. The published model weights in the Dropbox bundle (`model_weights/`) were not downloaded past the first PAD checkpoint. The bundle as streamed contains only PAD weights in its directory listing, under `model_weights/PAD/{UNet,ConvLSTM,ConvSTAR}/Scenario {1,2,3}/`, and no OAD weights directory.
- **Per-class or macro F1.** Not published. Confusion-matrix figures are unreadable at PDF resolution.
- **The exact cloud metadata field** and whether the 10% cut is `<` or `<=`. Not stated.
- **Why 1 to 3 January and February 2019/2020 granules** present in Earth Search are absent from some patch stacks.
- **Train-side OAD CSVs:** streamed past but not kept. The train-side fill rates above come from patch time stacks, not parcel CSVs.
- **The JSTARS typeset PDF** was not fetched separately. Page numbers refer to arXiv 2204.00951. Table and figure numbers are the same in both.
