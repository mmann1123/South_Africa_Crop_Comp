# Kerner et al. 2020 (growth-stage normalization): refutation check and measurement specs

Prepared 28 October 2026 for the missingness confound audit (card 3 in `reports/Missingness confound audit of transfer papers.md`). Sources read: the arXiv v1 PDF of Kerner et al. 2020 (arXiv 2009.10189, 9 pp.), the full repository `nasaharvest/croptype-mapping-gsn` (three commits, head `77042c1`; it contains only `README.md` and `lstm-cnn.ipynb`), the Zenodo data record 10.5281/zenodo.3997350 (one 7.27 GB `data.zip`), and NASA CMR granule metadata for HLS v2.0 tile 16TBL. All downloads are in `/tmp/claude-1000/-home-mmann1123-Documents-github-South-Africa-Crop-Comp/6a949ad2-1d7c-45e4-b537-97b8e63decc6/scratchpad/kerner_read/`. Analysis scripts are in the sibling `kerner_scripts/` folder. No downloaded code was executed. The notebook was parsed as JSON.

## Verdict

**The zero-fill version of the hypothesis is refuted for the released inputs, and the broader missingness version is weakened to the point where Kerner should drop from rank 3 to rank 1 or 2.**

The Zenodo archive holds the exact growth-stage-normalized input arrays the notebook loads (`data/inputs_3doy/pheno_timeseries_16TBL_{2017,2018,2019}.npy`). I measured them directly on the same northern strip in each year: rows 3 to 1020 at full width, which is 28 percent of the tile and about 3.7 million pixels per year. They contain **no NaN at all**. Exact zeros make up 0.001 to 0.003 percent of values, with no year pattern that matters. The notebook's `X[np.where(np.isnan(X))] = 0` therefore does nothing to these arrays. Gaps were filled upstream by the unreleased Savitzky-Golay step. 2019 does not carry more zero-filled values than 2017 or 2018 in the growth-stage inputs.

The fixed-DOY arrays (`fixed_timeseries_*`, named only in commented-out code) were not released, so their NaN count cannot be measured. They are almost certainly samples from the same gap-filled per-acquisition cube, which would make them NaN-free as well. This is an inference.

What remains is a softer "filled versus observed" shift: a value at a cloudy acquisition date is a smoothed interpolation, not a measurement. Granule cloud cover from CMR (HLS v2.0, used as a proxy for v1.4) works against that version too. The training year 2018 has its fixed dates on scenes that are 100 percent (DOY 149) and 97 percent (DOY 271) cloudy, and 2018 has no v2.0 granule at all on DOY 209. 2017's fixed dates are clear (1, 0, and 50 to 75 percent). 2019's are 67, 67 and 75 percent. The training mixture already spans fully observed and fully interpolated fixed-date inputs, and 2019 lies inside that range, not beyond it.

The paper's own random-forest result points the same way. With fixed-DOY inputs the forest loses only 0.9 points (82.8 versus 83.7 percent), so the 2019 fixed-date inputs still hold almost all the class information. The 25 to 30 point collapse belongs to the neural networks, not to the inputs.

The remaining critique is about reporting. The paper does not state the Savitzky-Golay window, how gaps were bridged before smoothing, or the share of time-series values that are interpolated in each year. The fixed-DOY arrays and the preprocessing code are missing, so the filled-value share at the fixed dates cannot be checked from released material.

## Evidence against our hypothesis (strongest first)

**1. Released inputs have no NaN in any year (direct measurement).** Script: `kerner_scripts/prefix_measure.py`. I decompressed the first 480 MB of each year's deflate stream from the Zenodo zip using HTTP range requests. That gives 1,021 complete rows in all three years. The array header is `{'descr': '<f8', 'fortran_order': False, 'shape': (3660, 3660, 6, 3)}`.

| Year | NaN fraction (all bands, all 3 stages) | Exact-zero fraction | Pixels with any NaN |
|---|---|---|---|
| 2017 | 0.00000 | 0.00001 | 0.00000 |
| 2018 | 0.00000 | 0.00002 | 0.00000 |
| 2019 | 0.00000 | 0.00003 | 0.00000 |

Values are confined to [0, 1]: none are negative and none exceed 1. The `X[np.where(X==0)] = 0.00000001` line in `load_data` (`lstm-cnn.ipynb`, cell 4) therefore also touches essentially nothing.

**2. The random forest barely cares about fixed versus growth-stage dates.** Table 1, p. 6, gives overall accuracy for growth-stage versus fixed inputs: CNN-LSTM 82.8 vs 55.9, CNN 82.1 vs 65.1, LSTM 76.3 vs 48.8, Random Forest 83.7 vs 82.8. Section 5.1, p. 7: "the difference between the random forest with growth-stage normalized and fixed DOY inputs is much lower than for the neural network models. This suggests that random forests may extrapolate better to changes in planting timelines than neural network approaches." If fill shift had destroyed information in the 2019 fixed-date inputs, the forest would also degrade. It does not. The loss reflects how the neural networks respond to a covariate shift, whatever caused that shift.

There is one caveat. Section 5.1, p. 5, says "The input to the random forest included the 6 optical bands, 2 SAR bands, 2 spectral index values, and δ," where δ is the pixel's senescence-minus-greenup DOY. If the fixed-DOY forest also received the per-pixel δ, it carried growth-stage information. The text is ambiguous: δ "would be constant across all inputs" is said only of the CNN-LSTMδ fixed run.

**3. Granule cloud cover at the fixed dates is not worse in 2019 than in training (CMR, HLS v2.0).** Script: inline Python over `kerner_read/cmr/rows.json`. The fixed DOYs come from Section 5.1, p. 5: "DOY 153, 149, and 152 for 2017, 2018, and 2019 ... 210, 209, and 211 ... 274, 271, and 274."

| Year | Fixed DOY: granule cloud % on that exact day |
|---|---|
| 2017 | 153: L30 1 · 210: L30 0 · 274: L30 75, S30 50 |
| 2018 | 149: L30 100 · 209: no v2.0 granule (nearest 211, L30 59) · 271: S30 97 |
| 2019 | 152: L30 67 (S30 on DOY 151 is 17) · 211: S30 67 · 274: S30 75 |

Every fixed DOY is an actual L30 or S30 overpass date, except 2018 DOY 209 in v2.0. The authors evidently picked real acquisitions near late May, late July and late September without regard to cloud.

**4. Growth-stage dates are heavily interpolated too, in training.** Weighting each pixel's selected DOY by that day's granule cloud cover gives these shares of corn and soybean pixels whose input falls on a scene that is at least 80 percent cloudy: 2017 greenup 0.705, peak 0.539, senescence 0.325; 2018 0.001, 0.139, 0.004; 2019 0.017, 0.311, 0.001. The model with the best result (85.4 percent) was trained partly on 2017 inputs, most of which are smoothed fills. Interpolated inputs therefore did not by themselves prevent transfer.

**5. Phenology really is late in 2019.** These are medians from `data/phenology/16TBL_{yr}_phenology.npy` over CDL corn and soybean pixels.

| | 2018 corn | 2019 corn | 2018 soy | 2019 soy |
|---|---|---|---|---|
| Greenup DOY | 140 | 174 | 165 | 181 |
| Peak DOY | 199 | 231 | 214 | 239 |
| Senescence DOY | 251 | 280 | 268 | 280 |

Corn shifts by about 30 days at every stage. Section 1 of the paper says "forced farmers to plant crops 1-2 months later than normal." A one-month shift moves the fixed late-May and late-July samples to a different crop stage, which is enough to explain a large neural-network loss without any gap effect.

**6. Figure 2 (p. 4) shows a class-specific shift.** Caption: "Histograms of NDVI for pixels classified as corn, soybean, or other in the CDL show the distribution in 2019 is shifted significantly lower than 2017-2018 for corn and soybeans, but mostly consistent for other." Fill or cloud contamination would shift the other class too. The true-colour DOY 154 panels show visibly less green in 2019. This is weaker evidence: DOY 154 in 2019 is an S30 granule with 45 percent cloud, the figure covers only a subset inset, and the histograms are from that single date.

**7. Gap-free SAR channels.** Two of the ten model channels are Sentinel-1 VV and VH (Section 3, p. 2). The notebook also adds incidence angle. These channels are gap-free and identically processed in every year, which limits how much optical fill could drive the result.

## Evidence for the hypothesis (or at least not refuting it)

Six points remain on this side. (a) The preprocessing code is absent and the paper says only "To fill in no-data pixels and mitigate outliers in the time series data, we smoothed the time series for each pixel and each band using the Savitzky-Golay method [7]" (Section 3, p. 3). Window, polynomial order, any pre-interpolation, and the per-year fill fraction are all unreported. (b) The fixed-DOY arrays were not released, so their fill state is unmeasured. (c) The 2019 fixed dates are all on 67 to 75 percent cloudy granules. Interpolated values there are anchored on neighbouring acquisitions, and June 2019 had only 2 granules at 20 percent cloud or less, against 9 in 2018 and 4 in 2017 (CMR, DOY 152 to 181). Sparse anchors make Savitzky-Golay fills less faithful, so a "fill quality" shift at DOY 152 in 2019 is plausible. It is not zero-fill, though, and 2018's DOY 149 sits on a 100 percent cloudy scene. (d) The 2017 phenology looks degenerate. Greenup is DOY 186 for 70 percent and DOY 51 for 28 percent of pixels, with identical medians for corn, soybean and other. That points to sparse 2017 data (only 43 S30 v2.0 granules) controlling the phenology fit, so one training year's "growth-stage" inputs are near fixed-date anyway. (e) The paper reports no per-year valid-observation counts. (f) The inputs measured here are from a 28 percent northern strip; the remaining rows were not decoded. With exactly zero NaN in 11 million pixel-years, a different result elsewhere is unlikely.

## Exact measurement specs

| Item | Specification | Source |
|---|---|---|
| Product | HLS, both L30 and S30. "allows the Landsat-8 and Sentinel-2 satellites to be treated as a virtual constellation with 30 m/pixel spatial resolution and 2-3 days revisit time" | Paper Section 3, p. 2; selected DOYs match both L30 and S30 v2.0 dates (93 to 100 percent of corn/soy pixels within 1 day) |
| Version | Not stated. Almost certainly v1.4: data in Zenodo 23 Aug 2020, work done 2019 to 2020, before the v2.0 release. Cites Claverie et al. 2018 (the v1.4 description) | Inference |
| Tile | MGRS 16TBL, northern Illinois, 12,056 km², 3660 × 3660 px at 30 m | Paper Fig. 1 caption, p. 3; notebook cell 2 `HEIGHT = 3660`, `WIDTH = 3660` |
| Years | Train 2017 and 2018 (east half plus NW quadrant), validate on SW quadrant of 2017 and 2018, test 2019 full tile. Zenodo also holds an unused `pheno_timeseries_16TBL_2016.npy` | Paper Section 5, p. 5; notebook cells 5 to 17, 92 |
| Date window | Full calendar year, DOY 0 to about 346. Stage DOYs range from 21 to 346; mean greenup 144 ± 37, peak 216 ± 27, senescence 278 ± 28 | Paper Sections 5.1 and 5.2; phenology arrays |
| Optical bands | Blue, green, red, NIR, SWIR1, SWIR2. Paper text says "Landsat-8 bands B01-B06 and Sentinel-2 bands B02, 03, 04, 08A, 11, and 12". B01 is coastal aerosol on OLI, so B02 to B07 is meant. Array band order confirmed from medians: index 0 blue, 1 green, 2 red, 3 NIR, 4 SWIR1, 5 SWIR2. Reflectance is scaled to [0, 1] | Paper Section 3, p. 2; notebook cell 3 ("All of the reflectance values are in the range [0,1]") |
| Indices | NDWI = (GREEN − SWIR1)/(GREEN + SWIR1), LSWI = (NIR − SWIR1)/(NIR + SWIR1), Eqs. 1 and 2. NDVI is used only for phenology detection; NDVI and GCVI as inputs "decreased" performance | Paper p. 3; notebook cell 4 `add_ndwi_channel`, `add_lswi_channel` (constants `GREEN`, `NIR`, `SWIR1` are never defined in the notebook) |
| SAR | Sentinel-1 VV and VH, "speckle-filtered and normalized by incidence angle". The notebook also adds incidence angle (`ia_timeseries`), giving 11 channels, while the paper says M = 10 | Paper p. 2 and p. 4; notebook cell 4 `add_sar_channel` |
| QA bits masked | HLS QA bits 0 to 3 ("remove pixels with clouds and cloud shadows (bit numbers 0-3)"). In the v1.4 QA layout these are cirrus, cloud, adjacent cloud and cloud shadow. Snow/ice (bit 4), water (bit 5) and aerosol (bits 6 and 7) are not masked | Paper Section 3, p. 3 |
| Temporal axis | Not a regular grid. Every acquisition date (L30 and S30) appears to be a time step: stage DOYs take 24 to 139 distinct values, and 93 to 100 percent of corn/soy pixels' DOYs fall within 1 day of a v2.0 granule date. The same-day L30 + S30 merge rule is unknown | Phenology arrays vs CMR (inline analysis) |
| Savitzky-Golay | Per pixel and per band, used to "fill in no-data pixels and mitigate outliers". Window and order are not given; the cited method is Chen et al. 2004. NaN handling is not stated, but the output is NaN-free in all years, so gaps were bridged before or during smoothing | Paper p. 3; measurement above |
| Phenology | From the SG-smoothed NDVI series. Greenup is the day after which slope is highest "within the first half of the time series"; senescence is the day after greenup with the most negative slope; peak is the maximum NDVI between them | Paper Section 4.1, pp. 3 to 4 |
| Model inputs | Values of the 6 bands at the 3 pixel-specific stage DOYs (`pheno_timeseries_*.npy`, float64, shape 3660×3660×6×3). Fixed-DOY variant: `fixed_timeseries_*.npy`, not released | Notebook cells 4 and 6; commented `path.replace('fixed_timeseries', ...)` |
| NaN to 0 | `load_data` (cell 4): `if convert_nans: X[np.where(np.isnan(X))] = 0`, then `X[np.where(X==0)] = 0.00000001`. The same `_X[np.where(np.isnan(_X))] = 0` appears after each added channel in `add_lswi_channel`, `add_ndwi_channel` and `add_sar_channel`. Measured fraction affected in the released arrays is 0 NaN, with 0.001 to 0.003 percent exact zeros, in every year | Notebook cell 4; measurement |
| Labels | CDL for each year, aggregated to corn (0), soybean (1) and other (2). A 3×3 homogeneity mask is used for training. `make_inputs` (cell 19) actually rejects a sample if any pixel in its 5×5 mask window is non-homogeneous. The 2019 evaluation uses all pixels with a 3-pixel border trimmed (`[3:-3,3:-3]`, cell 130) and no homogeneity mask. The paper adds a QGIS Sieve on CDL before metrics; that step is not in the notebook | Paper p. 3 and p. 6; notebook cells 13, 19, 130 |
| Sampling | Random row and column index sets: `subset=400` (400 × 400 = 160,000 candidate pixels) for each east-half year, `subset=100` (10,000) for each NW quadrant year and for each SW validation year. The notebook comments say these cells were re-run "2 times" and "4 times". The paper says 520,000 training and 10,000 validation pixels were drawn, then classes were balanced by truncation to 96,459 training and 5,517 validation per class. The notebook counts do not reproduce these numbers | Notebook cells 21 to 24, 31, 34 and 35; paper p. 6 |
| Split by year | Train and validation both come from 2017 and 2018, separated spatially (validation = SW quadrant). Test is the entire 2019 tile, so the spatial and temporal shifts are confounded for three of the four quadrants | Notebook cells 15 and 17; paper p. 5 |
| Normalisation | A separate StandardScaler per timestep for the pixel (LSTM) input, and global per-feature mean and standard deviation for patches. Both are fit on training data and applied to 2019 | Notebook cells 54 to 59, 100 and 101 |
| Training | 25 epochs, batch 4096, Adam, categorical cross-entropy, label smoothing 0.1 | Notebook cells 2 and 49 to 51; paper p. 5 |

Two notebook defects bear on reproducibility but not on missingness. Cell 29 builds `delta_train_p` from `east_2018_delta_s, east_2017_delta_s, nw_2018_delta_s, nw_2017_delta_p`, so the mid-season δ in training mixes senescence and peak deltas. Also, `N_BANDS = 6` is used for model input shapes while `load_data` produces 11 channels. The published notebook is Python 2.7, has no saved outputs, and cannot run as-is, so it is not the exact code behind Table 1. The random forest code is not in the repository.

## Reproducibility in Google Earth Engine

**The recipe can be reproduced approximately in Earth Engine, not exactly.** It would use `NASA/HLS/HLSL30/v002` (bands B2, B3, B4, B5, B6, B7) and `NASA/HLS/HLSS30/v002` (bands B2, B3, B4, B8A, B11, B12) filtered to tile 16TBL, which covers the same 3660 × 3660 30 m UTM 16N grid. The Fmask bits 1 to 3 would be masked, the series kept on all acquisition dates, SG-smoothed per pixel and band, and phenology and stage values extracted from it. Earth Engine has no native per-pixel Savitzky-Golay, so smoothing is best done after export, for example with `scipy.signal.savgol_filter` on a linearly gap-filled series.

The differences from the original would be the following. (1) Version 2.0 uses Fmask 4 for both sensors, and bit 0 (cirrus) is reserved but unpopulated in v2.0, so the "bits 0-3" mask loses its cirrus component. Cloud and shadow masks will differ from v1.4. (2) The v2.0 acquisition set differs from v1.4. v2.0 has more S30 granules (2018: 101 S30, 2019: 98, 2017: 43), and DOY 209 in 2018 has no v2.0 granule, so the time steps, and hence the phenology DOYs, will not match exactly. (3) v2.0 applies a revised BRDF normalisation and a new atmospheric correction, which shifts reflectance slightly. (4) The SG window and order, the pre-smoothing gap bridging, the rule for merging same-day L30 and S30, and the exact first-half definition for greenup are unspecified and must be chosen. (5) Sentinel-1 VV, VH and incidence angle would come from `COPERNICUS/S1_GRD`, with unknown speckle filter and angle normalisation. (6) The fixed-DOY inputs can be rebuilt in Earth Engine, which would allow measuring the filled-value share at DOY 149/152, 209/211 and 271/274 directly. That is the one measurement still missing.

There is a cheaper route for the main question. The released Zenodo phenology and growth-stage arrays can be reused as-is, and Earth Engine (or v2.0 Fmask through CMR) can count per-pixel clear observations within ±8 days of each fixed and each growth-stage DOY in each year. That gives the per-year interpolated-input share without re-implementing the smoother.

## Not accessed

**Kerner et al. 2022, Science of Remote Sensing 6:100059 (PII S2666017222000219).** Crossref and OpenAlex list it as gold open access under CC-BY 4.0. ScienceDirect and DOAJ returned HTTP 403 to automated fetches, and no repository copy was found (Semantic Scholar, OpenAlex, CORE, Europe PMC). An attempt through the Elsevier text API was refused by the session's permission policy. The OpenAlex abstract confirms that it uses HLS plus Sentinel-1 with test data from Iowa 2019 ("late planting year") and Illinois 2020 ("standard year"), reporting accuracies of 71/72, 80/84 and 81/85 percent for early, mid and late season respectively. Its preprocessing, gap handling, and any fixed-date comparison remain unread. Because it is CC-BY, a browser download by the user should be straightforward.

The full Zenodo archive (7.27 GB) was not downloaded. Only the phenology arrays, the CDL label arrays, and the first 28 percent of rows of each growth-stage input array were read. HLS v1.4 granule metadata, which would give the exact v1.4 acquisition dates and cloud cover, was not retrieved. The v2.0 cloud percentages above are a proxy for it.
