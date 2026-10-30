# Reconnaissance: TimeMatch and DENETHOR data for a cloud-fraction audit

Date: 2026-10-28. Scope: metadata, README and code reads only. No data archives were downloaded. The only remote bytes pulled beyond documents and code were (a) the 161.8 MB Zip64 central directory of the TimeMatch zip plus its four `metadata.pkl` members (13.3 MB) and four `.zarray` headers, all fetched by HTTP byte range from Zenodo, and (b) the kilobyte-sized `timestamp.pkl`, `meta_info.pkl`, `bbox.pkl` and `NORM_FACTORS.npy` files of the seven Sentinel-2 EOPatches and two Sentinel-1 EOPatches on Source Cooperative. Everything marked **derived** below is computed from those metadata reads; everything marked **inferred** is reasoned from file sizes and has not been checked against array contents.

Supporting files written next to this note: `recon_timematch_metadata_summary.json` (dates and tile-level cloud percentages for the four TimeMatch tiles) and `recon_fusion_competition_s3_listing.tsv` (every object in the `esa/fusion-competition` bucket with its size in bytes).

---

## 1. TimeMatch dataset (Nyborg, Pelletier, Lefevre, Assent 2022, ISPRS JPRS 188:301-313)

### 1.1 Zenodo records, sizes, licence

- Two versions exist under concept DOI 10.5281/zenodo.5636421. Version 1 is record 5636422 (DOI 10.5281/zenodo.5636422, publication date 2021-11-01). Version 2 is record 6542639 (created 2022-05-12, same publication date), which adds `thermal_time.zip` for the follow-up CVPRW 2022 paper. Both: licence `cc-by-4.0`, creators Joachim Nyborg and Charlotte Pelletier. Source: https://zenodo.org/api/records/5636422 and https://zenodo.org/api/records/6542639 (JSON `metadata.license`, `files`).
- Files (identical bytes and md5 in both versions): `timematch_data.zip` 73,904,625,097 bytes (73.9 GB, md5 b7909ba1d03e5c1abdac1c8c6a23f9ad); `trained_models_and_results.zip` 113,398,374 bytes; version 2 only: `thermal_time.zip` 193,567,818 bytes. Total v2 = 74.2 GB. Download URL pattern: `https://zenodo.org/api/records/6542639/files/timematch_data.zip/content`. The README states "The extracted size is about 78 GB" (https://github.com/jnyborg/timematch/blob/main/README.md).
- Zenodo's zip previewer refuses the file ("ZIP file is too large to be previewed", https://zenodo.org/records/6542639/preview/timematch_data.zip), but the download endpoint honours HTTP `Range` requests (verified; `content-length: 73904625097`). The zip is Zip64 with 1,201,894 entries; the central directory is 161,764,514 bytes at offset 73,742,860,485 (**derived** from the end-of-central-directory records).
- Per-tile sizes (**derived** from the central directory; compressed = bytes inside the zip, uncompressed = bytes on disk):

| tile path inside zip | zip entries | parcels (`.zarr` dirs) | compressed | uncompressed |
|---|---|---|---|---|
| `timematch_data/austria/33UVP/2017` | 168,961 | 49,168 | 8.20 GB | 8.96 GB |
| `timematch_data/denmark/32VNH/2017` | 251,953 | 60,332 | 18.98 GB | 20.66 GB |
| `timematch_data/france/30TXT/2017` | 454,116 | 115,755 | 28.10 GB | 30.55 GB |
| `timematch_data/france/31TCJ/2017` | 326,856 | 89,985 | 18.33 GB | 20.04 GB |

Total 315,240 parcels. (The thermal-time paper says "The total amount of parcels is 280K with 15 crop classes", presumably after dropping unknowns: https://arxiv.org/abs/2203.09175.) 315,275 members are stored uncompressed (method 0, the zarr chunk files `0.0.0` etc.) and 886,619 are deflated.

### 1.2 Tiles, years, product level, cloud filtering

- Exactly four tile-years, hard-coded in `train.py` line 264: `available_tiles = ['denmark/32VNH/2017', 'france/30TXT/2017', 'france/31TCJ/2017', 'austria/33UVP/2017']` (https://github.com/jnyborg/timematch/blob/main/train.py).
- Paper section 4.2: "The TimeMatch dataset [33] contains SITS from Sentinel-2 Level-1C products in top-of-atmosphere reflectance. Four Sentinel-2 tiles are chosen in various climates: 33UVP (Austria), 32VNH (Denmark), 30TXT (mid-west France), and 31TCJ (southern France) ... We use all available observations with cloud coverage <= 80% and coverage >= 50% between January 2017 and December 2017. Figure 5 shows the resulting acquisition dates for the four tiles. We leave out the atmospheric bands (1, 9, and 10), keeping C = 10 spectral bands. The 20m bands are bilinearly interpolated to 10m." (https://arxiv.org/pdf/2111.02682, p. 8).
- The filter is tile-level, from the AWS L1C `tileInfo.json`: `preprocessing/downloader.py` lines 156-157 `cloudy_pct = tileinfo['cloudyPixelPercentage']; coverage = tileinfo['dataCoveragePercentage']`, applied at lines 74-75; `prepare_pixels.py` argparse defaults `--max_cloudy_percentage 80.0`, `--min_coverage 50.0`, `--data_collection l1c` (lines 318-323 of the file as counted from 1). Imagery source: "the AWS bucket managed by Sinergise" (README credits), inventory via `https://roda.sentinel-hub.com/sentinel-s2-l1c` (downloader.py line 16; line 141 comments "use l1c api for l2a data as l2a cloudy percentage is buggy").
- **No per-pixel cloud mask or cloud probability is shipped.** For L1C the downloader sets `self.cloud_mask = None` (downloader.py line 32) and `read_s2_image` concatenates only `[x10, x20]` (prepare_pixels.py line 185). The L2A branch (not used for the release) would have rasterised the vector `qi/MSK_CLOUDS_B00.gml` to a 65535-valued 11th band (prepare_pixels.py lines 160-171, 184). `dataset.py` line 76 sets `"valid_pixels": np.ones((pixels.shape[0], pixels.shape[-1]), dtype=np.float32)`, so the model sees every date as valid.

### 1.3 On-disk layout, shapes, dtype

- One zarr v2 store per parcel: `timematch_data/<country>/<tile>/<year>/data/<parcel_idx>.zarr/` containing `.zarray` and chunk files; plus `meta/metadata.pkl`, `meta/blocks/blocks_<country>_<tile>_<year>.{shp,shx,dbf,prj,cpg}` and (Austria and the two France tiles only, not Denmark) `meta/parcels/parcels_<country>_<tile>_<year>.{shp,...}` (**derived** from the central directory; Denmark's `meta/` has 9 entries, the others 14). Creation code: prepare_pixels.py lines 75-83 `parcel_file = str(config.data_dir / f"{parcel_idx}.zarr") ... pixel_set_shape = (len(dates), n_bands, n_pixels[parcel_idx])  # (T, C, S); z = zarr.open(parcel_file, mode='a', shape=pixel_set_shape, dtype=np.uint16)`.
- `.zarray` headers read by range request (**derived**): `"dtype": "<u2"`, `"order": "C"`, `"fill_value": 0`, compressor `{"id": "blosc", "cname": "lz4", "clevel": 5, "shuffle": 1}`, `zarr_format 2`. Shape `(T, 10, S)`: Austria `[58, 10, 102]` with a single chunk `[58, 10, 102]`; France 30TXT `[62, 10, 145]` single chunk; Denmark `[52, 10, 258]` chunked `[26, 10, 258]` (two time chunks, so Denmark parcels have `0.0.0` and `1.0.0`). Loader: `dataset.py` line 70 `pixels = zarr.load(path)  # (T, C, S)`; `get_shapes` (lines 56-60) hard-codes 10 channels.
- Per-parcel pixel counts (**derived** from `metadata.pkl`): Austria median 180 (min 1, max 50,212; 11.55 M pixels total); Denmark median 321 (min 89, max 46,292; 29.94 M); France 30TXT median 250 (min 91, max 8,431; 39.21 M); France 31TCJ median 275 (min 93, max 7,744; 36.64 M). Parcels were eroded by 20 m and filtered to >= 10,000 m2 before erosion-based pixel counting (prepare_pixels.py `get_ground_truth_parcels(..., erosion_m=20, min_area_sqm=10000)`).

### 1.4 Band list, order and scaling

- Downloaded L1C bands: `['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']` (prepare_pixels.py `config.bands`, downloader.py line 56 and line 184).
- **Channel order in the arrays is not that list.** `read_s2_image` stacks the 10 m bands first and the 20 m bands second: downloader.py lines 29-30 define `bands_10m = ['B02','B03','B04','B08']`, `bands_20m = ['B05','B06','B07','B8A','B11','B12']`; prepare_pixels.py line 185 `image = np.concatenate([x10, x20], axis=-1)`. So channel index 0..9 = **B02, B03, B04, B08, B05, B06, B07, B8A, B11, B12**. The 20 m bands are upsampled with `cv2.resize(x20, x10.shape[:2], interpolation=cv2.INTER_LINEAR)` (line 182).
- Values are the raw L1C JP2 digital numbers (uint16; TOA reflectance x 10,000; 2017 data predate the 04.00 baseline offset). Training normalisation divides by 65,535, not 10,000: `transforms.py` lines 98 and 110 `def __init__(self, max_pixel_value=65535)` ... `sample['pixels'] = np.clip(sample['pixels'], 0, self.max_pixel_value).astype(np.float32) / self.max_pixel_value`.
- **s2cloudless cannot be run on the shipped arrays**: B01, B09 and B10 are absent (section 3 below lists the required bands). Seven of the ten s2cloudless bands are present (B02, B04, B05, B08, B8A, B11, B12).

### 1.5 Dates and the tile-level cloud statistic that is shipped

- `metadata.pkl` keys (**derived**): `['start_date', 'end_date', 'dates', 'cloudy_pct', 'parcels']`, written by prepare_pixels.py lines 70-71 and 105-120: `dates = [int(tile.yyyymmdd) for tile in s2_images]` (ints like 20170113) and `'cloudy_pct': cloudy_pct` (the per-date tile `cloudyPixelPercentage`). `dataset.py` lines 52-53, 127-136 convert to `date_positions` = days after `start_date` (DOY-like ints from 20170101).
- Number of dates per tile and the tile-level cloud percentages (**derived**; this is the only cloud statistic shipped with TimeMatch, and it is per tile-date, not per parcel):

| tile | T (dates) | first, last | dates per month Jan..Dec | tile cloudy_pct mean / median / max | dates > 50 % cloudy | dates > 20 % |
|---|---|---|---|---|---|---|
| AT1 33UVP | 58 | 2017-01-01, 2017-12-25 | 2,4,2,4,4,5,9,9,6,5,4,4 | 36.4 / 41.3 / 79.3 | 19 | 37 |
| DK1 32VNH | 52 | 2017-01-13, 2017-12-29 | 1,1,4,3,3,3,6,8,6,6,6,5 | 40.9 / 42.2 / 78.4 | 20 | 40 |
| FR1 30TXT | 62 | 2017-01-19, 2017-12-28 | 2,3,3,6,4,6,6,7,5,8,7,5 | 34.6 / 30.8 / 79.2 | 17 | 41 |
| FR2 31TCJ | 39 | 2017-01-26, 2017-12-27 | 1,2,2,1,3,3,2,5,4,6,5,5 | 27.3 / 11.9 / 74.0 | 13 | 17 |

Monthly mean tile cloudiness (same source): AT1 Jan 43, Feb 46, Mar 76, Apr 31, May 42, Jun 16, Jul 25, Aug 23, Sep 35, Oct 47, Nov 50, Dec 60; DK1 8, 42, 35, 20, 62, 56, 46, 38, 46, 40, 40, 37; FR1 23, 31, 35, 24, 41, 34, 54, 34, 39, 32, 23, 44; FR2 24, 3, 25, 12, 26, 28, 0.4, 13, 58, 16, 17, 66. Full per-date lists are in `recon_timematch_metadata_summary.json`.
- The paper publishes no numeric cloud statistics; Figure 5 ("Acquisition dates for each Sentinel-2 tile in our dataset. The inputs are irregularly sampled with variable temporal length") is a dot plot of dates only. The ISPRS paper and the CVPRW thermal-time paper (https://arxiv.org/pdf/2203.09175) both describe only the 80 %/50 % filter. **Could not determine** any published per-pixel or per-parcel cloud fraction for the TimeMatch tiles.
- Training samples 30 time steps at random (`train.py` line 291 `--seq_length 30`; `transforms.py` `RandomSampleTimeSteps` raises `NotImplementedError` when T < seq_length, so FR2 with T = 39 is fine) and 64 pixels (`--num_pixels 64`).

### 1.6 Parcels and labels

- `metadata['parcels']` is a list of dicts `{'id', 'label', 'n_pixels', 'block', 'geometric_features'}` (prepare_pixels.py lines 110-118), where `label` is the raw LPIS crop code (Denmark `afgkode` as float string e.g. `'1.0'`, France `code_cultu` e.g. `'BTH'`, Austria `snar_code` e.g. `'105'`; column names in `utils/label_utils.py` lines 50-63) and `geometric_features = [perimeter, area, perimeter/area, cover_ratio]` (lines 136-141). `block` indexes 1098 x 1098 pixel spatial blocks (`--block_size 1098`, 100 blocks per tile; `train.py --num_blocks 100`) used for the 70/10/20 split.
- Raw code to class mapping is in `class_mapping/<country>_class_mapping.yml` and `<country>_codification_table.csv` in the repo (https://github.com/jnyborg/timematch/tree/main/class_mapping); `dataset.py` lines 94-109 map unknown codes to class `unknown`, and `train.py` lines 39-46 keep only source classes with >= 200 parcels. Top raw codes (**derived**): AT1 `717` (13,219), `105`, `138`; DK1 `1.0` (13,114), `11.0`, `260.0`; FR1 `BTH` (20,548), `PTR`, `PPH`; FR2 `BTH` (18,140), `TRN`, `BDH`.
- LPIS sources credited in README: IGN RPG (France), NaturErhvervstyrelsen "Marker" (Denmark), AMA INVEKOS (Austria).

### 1.7 Access notes for an audit that does not retrain

- Zenodo range requests work, the central directory lists every member's offset and size, and zarr chunk files are stored uncompressed, so individual parcels can be fetched without the 74 GB download (one parcel is one or two members of ~50-500 KB). The four `metadata.pkl` files (dates, tile cloudiness, parcel ids, pixel counts) total 13.3 MB compressed and were already retrieved this way.
- Because B01/B09/B10 are missing, any per-pixel cloud score for TimeMatch must come from a reduced-band detector trained or calibrated elsewhere, or from a brightness/whiteness proxy on B02/B03/B04; s2cloudless itself is not applicable (section 3).

---

## 2. DENETHOR (Kondmann et al. 2021, NeurIPS Datasets and Benchmarks)

### 2.1 Where the data actually live, access, licence

- Repo: https://github.com/lukaskondmann/DENETHOR (default branch `master`, no licence file, last push 2023-04-18). README download section: the AI4EO link is struck through; "Update 2: Access the full challenge dataset including the test data via Radiant MLHub [here](https://mlhub.earth/data/dlr_fusion_competition_germany). Due to a quota limit at Sync&Share, download of the ablation study data is temporarily disabled." The earlier LRZ Sync&Share link (password `dailycrops`) for the 2019-train/2018-test ablation tiles is also struck through.
- Radiant MLHub is defunct; `https://mlhub.earth/data/dlr_fusion_competition_germany` now returns `301 -> https://beta.source.coop/esa/fusion-competition -> 307 -> https://source.coop/esa/fusion-competition` (verified with `curl -sIL`). The dataset page is **Source Cooperative, "Fusion Competition", owner European Space Agency, DOI https://doi.org/10.34911/rdnt.z9y7vu, created 5 Jul 2023, last updated 21 Aug 2025, public** (https://source.coop/esa/fusion-competition). Data base URL `https://data.source.coop/esa/fusion-competition/`; S3 bucket `us-west-2.opendata.source.coop`, prefix `esa/fusion-competition/`. Object listing works anonymously with `https://data.source.coop/esa/?list-type=2&delimiter=/&prefix=fusion-competition/<folder>/` (S3 ListObjectsV2 XML); `--no-sign-request` style anonymous access, **no registration or token was needed** for listing or for fetching documentation and metadata files. (Note: the local `aws` CLI is broken by a urllib3 mismatch; plain curl works.)
- Licence: the Source Cooperative page shows no licence field. The paper: "DENETHOR will be released under a CC-BY license to encourage widespread use and adoption" (proceedings PDF p. 3, https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/5b8add2a5d98b1a652ea7fd72d942dac-Paper-round2.pdf). The label documentation `Crops_GT_Brandenburg_Doc.pdf` (https://data.source.coop/esa/fusion-competition/Crops_GT_Brandenburg_Doc.pdf): "License - (c) GeoBasis-DE/LGB (2018/19), dl-de/by-2-0, Original data was altered ... This dataset is available under the DL-DE->BY-2.0 license". **Could not determine** the licence text attached to the Planet Fusion imagery itself (the MLHub record that carried it is gone); `documentation.html` in the bucket is only a redirect to the GitHub repo.
- The same bucket also holds the AI4FoodSecurity South Africa track (tiles 34S-19E-258N, 34S-19E-259N, 34S-20E-259N, 2017), i.e. this project's own data, with the same EOPatch structure.

### 2.2 Total and per-component sizes (**derived** from the full object listing, 3,361 objects, 781.34 GB)

| component | objects | size |
|---|---|---|
| `planet/UTM-24000/` daily PF-SR + PF-QA, all 5 tiles | 2,924 | 500.18 GB |
| of which `33N/18E-242N/PF-SR` 2018 (365 daily GeoTIFFs, 2018-01-01..2018-12-31, 326-332 MB each) | 365 | 121.69 GB |
| of which `33N/18E-242N/PF-QA` 2018 (365 daily, ~28 MB each) | 365 | 5.93 GB |
| of which `33N/17E-243N/PF-SR` 2019 (2019-01-01..2019-12-31) | 365 | 124.60 GB |
| of which `33N/17E-243N/PF-QA` 2019 | 365 | 5.92 GB |
| of which `34S/{19E-258N,19E-259N,20E-259N}/PF-SR` 2017 (244 daily each, 2017-04-01..2017-11-30) | 3 x 244 | 78.24 + 77.25 + 77.47 GB |
| of which `34S/.../PF-QA` 2017 | 3 x 244 | 3.08 + 3.08 + 2.93 GB |
| `planet-5-day/` (PF-SR only, median-filtered; Brandenburg 73 files per tile-year 2018-01-03..12-29 / 2019-01-03..12-29; SA 48 files 2017-04-03..11-24) | 290 | 94.91 GB |
| `sentinel-2/` 7 EOPatches | 56 | 125.04 GB (each Brandenburg patch 22.395 GB; each SA patch 11.820 GB) |
| `sentinel-1/` 11 EOPatches (asc + des) | 76 | 59.62 GB |
| `brandenburg-gt/` + root label GeoJSONs + docs | 13 | ~0.19 GB |
| `esa-food-security-challenge/sentinel-1/.../data/MASK.npy` (two stray files) | 2 | 1.39 GB |

DENETHOR proper (train tile 18E-242N 2018, test tile 17E-243N 2019) is therefore about 258.1 GB Planet daily + 44.8 GB Sentinel-2 + ~27.5 GB Sentinel-1, consistent with the paper's "254.5" GB in Table 1 (p. 3, which lists DENETHOR as "PF+S2+S1, 3m, Daily, 4,500 fields, 254.5 GB"). Only the Brandenburg tiles for 2018 train and 2019 test are in the DENETHOR benchmark; the bucket also carries the ablation tile-years 17E-243N-2018 and 18E-242N-2019 for S1 and S2 but **not** Planet imagery for those (Planet exists only for 18E-242N-2018 and 17E-243N-2019).

### 2.3 Tile footprints and dates

- Paper: "we restrict our focus to two tiles. Both tiles are identical in size with 24km x 24km. One tile is used for training and validation, the other for testing. For the training tile, we include field masks and crop types from the year 2018 ... Test evaluations are based on the 2019 data." (p. 4).
- EOPatch `bbox.pkl` (**derived**, EPSG:32633): train `18E-242N` x 431,999.73-455,999.98, y 5,808,000.14-5,832,000.45; test `17E-243N` x 408,000.11-432,000.08, y 5,831,999.78-5,856,000.36 (adjacent, test tile is NW of train tile). S1 `meta_info.pkl` footprints in WGS84 start near 12.25 E, 51.45-51.51 N. Planet tile naming follows the Planet Fusion "UTM-24000" 24 km grid (spec section on file naming: `UTM-2400/15N/17E-192N/PF-SR/2006-01-02.tif`).
- Sentinel-2 dates (**derived** from `timestamp.pkl`, tz UTC):
  - `s2-utm-33N-18E-242N-2018`: **T = 144**, 2018-01-02 10:24 to 2018-12-30 10:16; per month Jan..Dec 12,12,12,12,11,12,13,12,12,13,11,12; gaps 2-5 days (median 3).
  - `s2-utm-33N-17E-243N-2019`: **T = 144**, 2019-01-02 10:26 to 2019-12-30 10:16; identical monthly counts 12,12,12,12,11,12,13,12,12,13,11,12.
  - The ablation patches `17E-243N-2018` and `18E-242N-2019` also have T = 144; the two 2018 patches have byte-identical timestamp lists and the two 2019 patches share dates with slightly different acquisition times (same orbits, adjacent tiles).
  - **Consequence for the audit: 2018 and 2019 have the same number of Sentinel-2 acquisitions, because `maxcc=1` keeps every pass.** Any between-year difference in usable observations is entirely inside the CLP/SCL layers, not in the time axis.
  - SA 2017 patches: T = 76, 2017-04-01 08:49 to 2017-11-29 08:33; per month Apr 6, May 7, Jun 6, Jul 12, Aug 12, Sep 9, Oct 12, Nov 12 (single-satellite cadence until Sentinel-2B ramped up mid-2017). The three SA patches share the same 76 dates.
- `meta_info.pkl` (S2, **derived** via `pickletools`): `size_x 2400, size_y 2400, time_interval (YYYY-01-01 00:00, YYYY-12-31 23:59:59) [SA: 2017-04-01 to 2017-11-30], maxcc 1, time_difference timedelta(seconds=7200)`, i.e. Sentinel Hub requests at 10 m over the 24 km tile with same-pass mosaicking within 2 h.
- Sentinel-1 (**derived**): asc 18E-242N-2018 T = 122 (2018-01-03..2018-12-30), asc 17E-243N-2019 T = 120; `meta_info.pkl` carries `card4l_metadata` with S1 scene ids (e.g. `S1A_IW_GRDH_1SDV_20180103T165201_...`).

### 2.4 Sentinel-2 product shipped

- Paper p. 6: "The S2 data is downloaded from Sentinel Hub at processing level L2A. No observations are filtered because of cloud cover (maxcc=1) to maximize temporal coverage. We provide 12 bands - all resampled at 10m resolution - together with the valid data mask, scene classification (SCL) band and the s2cloudless probability map (SCP)." The RF script lists the band order: `random_forest.py` line 13 `BANDS = ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B11","B12"]` (B10 is absent from L2A: Sentinel Hub docs, "The cirrus band B10 is excluded as it does not contain any 'bottom of the atmosphere' information", https://docs.sentinel-hub.com/api/latest/data/sentinel-2-l2a/).
- EOPatch files per patch (eo-learn pickle/npy layout; listing): `bbox.pkl`, `timestamp.pkl`, `meta_info.pkl`, `data/BANDS.npy`, `data/CLP.npy`, `mask/IS_DATA.npy`, `mask/SCL.npy`, `scalar/NORM_FACTORS.npy`. Brandenburg sizes: `BANDS.npy` 19,906.56 MB, `CLP.npy` 829.44 MB, `IS_DATA.npy` 829.44 MB, `SCL.npy` 829.44 MB; SA: 10,506.24 / 437.76 / 437.76 / 437.76 MB.
- **Inferred** dtypes from sizes: 144 x 2400 x 2400 = 829,440,000 cells, so `CLP`, `IS_DATA` and `SCL` are 1 byte per cell, shape `(144, 2400, 2400, 1)`, and `BANDS` = 829,440,000 x 12 x 2 bytes, i.e. `(144, 2400, 2400, 12)` uint16 digital numbers. `NORM_FACTORS.npy` (**derived**, read) is `(144, 1) float32`, all `1e-4`, so DN x 1e-4 = reflectance; `random_forest.py` line 159 does `X = X * 1e-4`. Loader shape convention: `s2dataset.py` line 54 `_, width, height, _ = self.bands.shape` and line 80 `X = self.bands.transpose(0, 3, 1, 2)[:, :, field_mask]` (T, C, pixels).
- CLP semantics per Sentinel Hub: "CLP: Cloud probability, based on s2cloudless ... UINT8, DN, 0-255 (divide by 255 to get to the [0-1] range)", native 160 m; "CLM: ... 0 - no clouds, 1 - clouds, 255 - no data"; "SCL: Scene classification data, based on Sen2Cor processor ... 0 - No data through 11 - Snow / Ice", 20 m; `dataMask` 0/1 (https://docs.sentinel-hub.com/api/latest/data/sentinel-2-l2a/). The RF baseline thresholds CLP at 40: `random_forest.py` lines 157-158 `def transform(X, cld): msk = cld < 40`, which on a 0-255 scale is a 0.157 probability cut (the authors may have assumed 0-100; **could not determine** which).
- **Cloud layer handling in the loaders, exact lines** (https://github.com/lukaskondmann/DENETHOR/blob/master/s2dataset.py):
  - `S2DatasetV2.__init__` lines 51-53: `self.bands = np.load(os.path.join(rootpath, "data", "BANDS.npy"))` / `self.clp = np.load(os.path.join(rootpath, "data", "CLP.npy"))` / `# bands = np.concatenate([bands, clp], axis=-1) # concat cloud probability` (commented out).
  - `__getitem__` lines 80-84: `X = self.bands.transpose(0, 3, 1, 2)[:, :, field_mask]` / `clp = self.clp.transpose(0, 3, 1, 2)[:, :, field_mask]` / `if self.data_transform is not None: X = self.data_transform(X, clp)` (CLP is only ever passed to an optional transform; the default returns pixels with clouds untagged).
  - `setup()` lines 149-151: `bands = np.load(... "BANDS.npy")` / `clp = np.load(... "CLP.npy")` / `#bands = np.concatenate([bands, clp], axis=-1) # concat cloud probability`; line 175 `clouddata.append(clp.transpose(0,3,1,2)[:,:,field_mask].mean(-1))` stores the field-mean CLP separately in the `.npz` cache; `S2Dataset.__getitem__` line 117 passes it as `cld` to the transform.
  - `s1dataset.py` lines 148-149 (copy of the same comment for S1, where no CLP exists): `#clp = np.load(os.path.join(rootpath, "data", "CLP.npy"))` / `#bands = np.concatenate([bands, clp], axis=-1)`.
  - `IS_DATA.npy` and `SCL.npy` are never opened by any script in the repo.
  - The AI4Food starter pack reader has the same pattern with a flag: `notebook/utils/sentinel_2_reader.py` lines 117-120 `bands = np.load(os.path.join(rootpath, "bands.npy")); clp = np.load(os.path.join(rootpath, "clp.npy")) #CLOUD PROBABILITY; if include_cloud: bands = np.concatenate([bands, clp], axis=-1)` with `include_cloud=False` default (line 27), and line 156 saves `cloud_stack` separately (https://github.com/AI4EO/tum-planet-radearth-ai4food-challenge). Note the starter expects lower-case `bands.npy`/`clp.npy` at the patch root and a `.tar.gz`, whereas the bucket has `data/BANDS.npy`; the MLHub tarballs evidently differed from the Source Cooperative layout.
- Paper's own cloud statistic (p. 7, RF baseline): "we average all pixels of one field and consider only cloud-free observations which discards around 55% of S2 observations effectively cutting temporal resolution in half." No split by year is given; **could not determine** per-year cloud fractions from any published source.

### 2.5 Planet Fusion product and quality flags

- Paper p. 4-5: 3 m, 4 bands RGB+NIR, daily; "it delivers a temporally consistent collection of images with removed clouds and shadows ... Potential gaps because of clouds are filled with different points in time. The data also includes quality assurance (QA) information that underlines from which source any pixel is taken from together with a confidence score if the observation is gap-filled."
- Planet Fusion specification (March 2021, shipped in the bucket, https://data.source.coop/esa/fusion-competition/Planet_fusion_specification_March_2021.pdf): PF-SR "records gridded (3 m) and gap-free orthorectified data in four spectral bands (blue, green, red, NIR) at a daily interval. The data is stored in 16-bit integer format (with a multiplication factor of 10,000) as cloud optimized geotiffs compressed using LZW compression" (valid range 1-10,000, scale 0.0001). PF-QA is "a 9 layer thematic raster using the same spatial grid": Layer 1 "Percentage of synthetic (gap-filled)" 1-100 ("A value of 1 indicates no gap-filling (i.e., real PS/S2/L8 data) whereas a value of 100 indicates an entirely gap-filled pixel value"); Layer 2 "Days to closest scene with actual observation" -900..900 (0 = real observation); Layer 3 PlanetScope cloud and cloud shadow mask (1-7, -999); Layer 4 pixel traceability; Layer 5 number of FORCE L8/S2 reference scenes used; Layers 6-9 per-band uncertainty of synthetic values (0-200, real data = 0). Section 4.4: "The Planet Fusion gap-filling ensures a spatially complete and temporally continuous (daily) 3 m product ... users are advised to utilize the associated QA metrics (Table 5) as measures of the synthetic data retrieval confidence"; a "three-point (i.e., 3-day) temporal smoothing" is applied to the gap-free series.
- **So yes, an observed-vs-synthesised flag exists (PF-QA layers 1 and 2) and is shipped for the daily product** (365 x ~27 MB GeoTIFFs per Brandenburg tile-year, 244 x ~10-16 MB for SA), but **not** for the 5-day product (`planet-5-day/.../PF-SR` only). The DENETHOR Planet loader never opens it: `dataset.py` lines 17-18 glob `tifs = [f for f in os.listdir(self.tifroot) if f.endswith(".tif")]` from `--datapath .../PlanetL3H/Train/PF-SR` (README) and line 56 stacks them as `T x D x H x W`; the AI4Food `planet_reader.py` line 96 likewise globs `input_dir + '/*/*.tif'`.
- Planet daily file naming in the bucket: `planet/UTM-24000/33N/18E-242N/PF-SR/2018-01-01.tif` ... `2018-12-31.tif`, and `PF-QA/` with the same names; each PF-SR file 307-350 MB (8000 x 8000 x 4 int16 LZW), each PF-QA 9-28 MB.

### 2.6 Labels and fields

- GeoJSON (fid, crop_id, crop_name, geometry): `brandenburg-gt/brandenburg_crops_train_2018.geojson` 14.13 MB and `brandenburg_crops_test_2019.geojson` 8.38 MB (duplicated at bucket root as `br-18E-242N-crop-labels-train-2018.geojson` / `br-17E-243N-crop-labels-test-2019.geojson`). Loader: `dataset.py` lines 9-10 `CLASSES = ["Wheat", "Rye", "Barley", "Oats", "Corn", "Oil Seeds", "Root Crops", "Meadows", "Forage Crops"]; CROP_IDS = [1..9]`, fields `< 1000 m2` dropped (`min_area=1000`, line 13 and 29-31); `s2dataset.py` lines 58-59 rasterise `fid` with `all_touched=True`. Paper training counts: Wheat 305, Rye 276, Barley 137, Oats 45, Corn 251, Oil Seeds 201, Root Crops 23, Meadows 954, Forage Crops 339 (p. 4; README says the same). The README's "Fields with areas below 1km2 are excluded" is a typo for the paper's "1000m2".
- Label doc lists the 1-9 class codes and "Spatial clip of 2018 and 2019 vector files to training and test tiles".

### 2.7 Access notes for an audit that does not retrain

- Per-date, per-field Sentinel-2 cloud fractions for 2018 vs 2019 need only `timestamp.pkl` (15 KB), `CLP.npy` (829 MB), `SCL.npy` and `IS_DATA.npy` (829 MB each) and the label GeoJSON per tile-year; `BANDS.npy` (19.9 GB) is not needed. `.npy` is a flat C-order array behind a small header, so HTTP range reads of per-date slabs (5.76 MB each) are possible; the bucket serves range requests over HTTPS.
- Planet observed-vs-synthetic fractions need the PF-QA GeoTIFFs (5.9 GB per tile-year as a whole; as COGs they can be read windowed with GDAL `/vsicurl/`), reading layer 1 (percent synthetic) or layer 2 (days to real observation) per field per day.
- The SA 2017 patches in the same bucket give this project's own three tiles the same CLP/SCL layers (76 dates), which can be read the same way.

---

## 3. Tooling: s2cloudless

- Package `s2cloudless` current version **1.7.3** (PyPI upload 2024-09-27; GitHub release v1.7.3 same day), `requires_python >= 3.8`, `requires_dist`: `lightgbm>=2.0.11`, `numpy>=1.13.3`, `opencv-python-headless`, `sentinelhub>=3.9.0`, `typing-extensions`. Licence CC BY-SA 4.0. Sources: https://pypi.org/pypi/s2cloudless/json, https://github.com/sentinel-hub/sentinel2-cloud-detector.
- Single shipped model: `s2cloudless/models/pixel_s2_cloud_detector_lightGBM_v0.1.txt` (10,975,392 bytes), loaded as `lightgbm.Booster(model_file=...)` (`cloud_detector.py` lines 15, 74). No reduced-band model exists in the repo.
- Required bands and order (`utils.py` lines 13-15): `S2_BANDS = ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B10","B11","B12"]; MODEL_BAND_IDS = [0, 1, 3, 4, 7, 8, 9, 10, 11, 12]`, i.e. **B01, B02, B04, B05, B08, B8A, B09, B10, B11, B12** in that order. Input scaling (README): "the cloud detector requires the following 10 Sentinel-2 band reflectances: B01, B02, B04, B05, B08, B8A, B09, B10, B11, B12, which are obtained from raw reflectance values in the following way: `B_i/10000`. From product baseline `04.00` onward additional harmonization factors have to be applied". The model is for L1C ("Module for pixel-based classification on Sentinel-2 L1C imagery", `cloud_detector.py` line 1).
- Interface: `S2PixelCloudDetector(threshold=0.4, all_bands=False, average_over=1, dilation_size=1)`; `get_cloud_probability_maps(data)` requires `data.ndim == 4` with shape `(N, height, width, 10)` (or 13 with `all_bands=True`) and returns `(N, height, width)` float32 (lines 87-110); `get_cloud_masks` applies the disk convolution and dilation (lines 112-154). **Arbitrary pixel arrays work** because `PixelClassifier.image_predict_proba` flattens: `pixels = data.reshape((-1, data.shape[-1])); proba = self.classifier.predict(pixels, **kwargs)` (`pixel_classifier.py` lines 76-84); reshape a `(P, 10)` matrix to `(1, P, 1, 10)` or `(P, 1, 1, 10)`, or call `Booster.predict` directly, and set `average_over=None, dilation_size=None` to skip spatial post-processing.
- Applicability: **TimeMatch lacks B01, B09, B10 (and is TOA L1C as required); DENETHOR's Sentinel-2 is L2A (no B10, and bottom-of-atmosphere) but ships s2cloudless output as `CLP.npy`.** The audit can read CLP for DENETHOR but cannot recompute s2cloudless for either dataset from the shipped bands.

## 4. Pre-existing cloud statistics for these regions

- TimeMatch tiles: none published beyond the 80 %/50 % tile filter; the shipped `cloudy_pct` (section 1.5) is the only quantitative statistic and is tile-level.
- DENETHOR: "discards around 55% of S2 observations" (paper p. 7) for the RF baseline using `CLP < 40`; no per-year figure; "the performance drop in out-of-year evaluation from 2018 and 2019 ... amounts to 12 percentage points in accuracy" and "the spatial shift to a new tile accounts for about 9 p.p. and the temporal shift for 12 p.p." (pp. 1 and 8).
- Pham, Tetteh, Thiel, Erasmi, Schwieder, Frantz, van der Linden 2024, IJAEOG 129:103867 (Brandenburg, Landsat 5/7/8 + Sentinel-2 A/B, FORCE clear-sky observations): "there are large differences in annual CSO in terms of density (Fig. 1b) and distribution (see Supplement 1)"; training year 2018 chosen as "the year that had the highest number of clear sky observations or highest temporal data density"; "In years with high numbers of CSO (e.g., on average, pixels have more than 80 CSO in 2018), the daily encoded pixels still contain more than 75 % features with no-data values"; "Sentinel-2B data in 2017 was reduced, because the nominal phase started late in the year" (Fig. 9 caption). The per-year counts are only in the Fig. 1b bar chart and Supplement 1 (not fetched); **could not determine** a numeric 2019 CSO count from the text. Open-access PDF: https://literatur.thuenen.de/digbib_extern/dn068235.pdf; DOI https://doi.org/10.1016/j.jag.2024.103867.
- AI4FoodSecurity challenge pages (https://platform.ai4eo.eu/ai4food-security-germany/data) describe Planet Fusion as "clean (i.e. free from clouds and shadows), daily gap-filled" with "accompanying QA layers" and give no cloud statistics.

## 5. Items that could not be determined

- TimeMatch: per-pixel or per-parcel cloud fraction (nothing shipped, no published figure); exact content of `trained_models_and_results.zip` and `thermal_time.zip` (not opened); whether Denmark's missing `meta/parcels` shapefile is intentional.
- DENETHOR: licence text for the Planet imagery as distributed; exact dtypes of `BANDS.npy`/`CLP.npy`/`SCL.npy`/`IS_DATA.npy` (inferred from byte counts only); whether the `CLP < 40` threshold was meant on a 0-255 or 0-100 scale; the number of PF-QA days per tile-year that are mostly synthetic (requires reading the QA rasters); the original MLHub tarball layout versus the bucket layout.
- Pham 2024: numeric annual CSO counts for 2018 and 2019 (figure/supplement only).
