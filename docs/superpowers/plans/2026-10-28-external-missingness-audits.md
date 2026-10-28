# External Missingness Audits Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure, from each paper's own data or deposited recipe, the cloud and gap difference between the source and target domains of TimeMatch, DENETHOR and DACCN, and write up whether it supports or weakens each paper's attribution of its transfer loss.

**Architecture:** A small shared package (`experiments/external_audits/common/`) handles Earth Engine initialisation, HTTP range downloads with a SHA-256 manifest, field-polygon zonal statistics on EOPatch grids, run logs and plot theming. Each paper is an independent folder of numbered scripts (fetch, compute, summarise, cleanup) that read from one external data root and write small CSVs, one figure and a README with a finding paragraph. A `run_all.sh` chains everything and deletes downloaded data at the end, leaving manifests.

**Tech Stack:** Python 3 in the `ee` conda env (earthengine-api 1.7.50, geopandas, rasterio, numpy, pandas, matplotlib) for Earth Engine and raster work; `ml_field` is an acceptable alternative for the DENETHOR numpy scripts. pytest for tests. No new conda packages except pytest if missing.

**Spec:** `docs/superpowers/specs/2026-10-28-external-missingness-audits-design.md`

## Global Constraints

- No model training or re-scoring; no download of TimeMatch `data/*.zarr` beyond at most a handful of parcels, and never `BANDS.npy` or PF-SR imagery.
- Downloaded data live under `/mnt/bigdrive/external_audits/<paper>/`; nothing under `experiments/external_audits/` may hold raw downloads; committed CSVs under 20 MB each.
- Every download recorded in `manifest.json` (key, url, bytes, sha256, fetched_at); every script writes `run_log.json` (script, git commit, args, seed, counts, timings, EE collection ids and image ids used).
- Every script is idempotent: skips existing outputs unless `--force`; long EE loops are resumable row by row.
- Fixed seed 42 everywhere a sample is drawn.
- Earth Engine project id read from `~/.config/earthengine/credentials` (`project` key, currently 327687702789); reductions use `getInfo` on modest samples; Drive export only as a documented fallback.
- Cloud definitions: Cloud Score+ `cs_cdf < 0.6` cloudy; s2cloudless probability `>= 40` cloudy (shared cross-paper rule) and `>= 16` (DENETHOR's literal `cld < 40` on a 0 to 255 scale); DENETHOR SCL clear classes {4, 5, 6, 7} with IS_DATA true.
- DACCN recipe constants: simple cloud score threshold 0.2, 10-day composites from DOY 0 step 10 (37 composites), interpolation targets list indices 9 to 30 (22 steps), search 9 composites each side, `unmask(0)`. Variant A uses the whole year of imagery; Variant B filters imagery to April 1 to October 31.
- Prose in READMEs: no em dashes, concise, manuscript style; figures follow the `dataviz` skill.
- Commit after each task; the repo is on branch `missingness-shift`.

## Review Focus

1. A TimeMatch shipped date with no matching `COPERNICUS/S2_HARMONIZED` granule for that MGRS tile: the row must still be written with `n_images = 0` and `frac_* = NaN`, never silently skipped (Task 5 test `test_rows_for_dates_without_images`).
2. DENETHOR CLP stored as uint8 0 to 255 versus 0 to 100: the scale must be detected from the array maximum and recorded, and the 40 percent rule must map to 102 on a 0 to 255 scale (Task 8 test `test_clp_scale_detection`).
3. Label polygons that fall outside the EOPatch grid or rasterise to zero pixels: excluded and counted in the run log, never given a clear count of 0 (Task 2 test `test_field_index_map_drops_empty_fields`).
4. A missing or unreadable PF-QA daily COG: that day is counted as "no QA" and excluded from the synthetic fraction denominator, not counted as synthetic (Task 9 test `test_missing_day_excluded`).
5. GAUL region name resolving to zero or several features (for example "Jilin" matching a city and a province): the script must assert exactly one feature and fail loudly (Task 12 test `test_region_lookup_single_feature`).

---

### Task 1: Package scaffold, paths, manifest, run log, EE init

**Files:**
- Create: `experiments/external_audits/__init__.py`, `experiments/external_audits/common/__init__.py`
- Create: `experiments/external_audits/common/paths.py`
- Create: `experiments/external_audits/common/runlog.py`
- Create: `experiments/external_audits/common/fetch.py`
- Create: `experiments/external_audits/common/ee_init.py`
- Create: `experiments/external_audits/README.md` (skeleton: purpose, layout, how to run; filled in Task 14)
- Test: `experiments/external_audits/tests/test_common.py`

**Interfaces:**
- Produces: `paths.EXTERNAL_ROOT: Path = Path('/mnt/bigdrive/external_audits')`; `paths.paper_root(paper: str) -> Path` (creates dir); `paths.AUDIT_DIR: Path` (the `experiments/external_audits` folder); `paths.results_dir(paper) -> Path`, `paths.figures_dir(paper) -> Path`.
- Produces: `runlog.RunLog(script: str, args: dict, seed: int = 42)` context manager with `.add(key, value)`, `.count(key, n)`, writes `<results_dir>/run_log_<script>.json` on exit including `git_commit`, `started`, `ended`, `python`, `packages` (versions of ee, geopandas, rasterio, numpy, pandas).
- Produces: `fetch.download(url: str, dest: Path, manifest: Path, key: str, expected_bytes: int | None = None, force: bool = False) -> Path` (streams, computes SHA-256, appends `{key, url, bytes, sha256, fetched_at}` to manifest JSON list, skips if dest exists and manifest entry matches size); `fetch.range_get(url: str, start: int, length: int) -> bytes`; `fetch.manifest_entries(manifest: Path) -> list[dict]`.
- Produces: `ee_init.init() -> str` returning the project id from `~/.config/earthengine/credentials` after `ee.Initialize(project=...)`; raises with a clear message if the file or key is missing.

- [ ] **Step 1: Write failing tests**

```python
def test_download_writes_manifest_and_sha(tmp_path, httpserver):  # pytest-httpserver if available, else monkeypatch requests.get
    ...
    assert entries[0]['sha256'] == hashlib.sha256(b'hello').hexdigest()
    assert entries[0]['bytes'] == 5

def test_download_skips_when_present(tmp_path, ...):
    # second call with same size does not re-fetch (server hit count stays 1)

def test_range_get_sends_range_header(...):
    # request header 'Range' == 'bytes=10-19'

def test_runlog_writes_json(tmp_path, monkeypatch):
    # monkeypatch paths.results_dir -> tmp_path; with RunLog('x', {'a':1}) as log: log.count('rows', 3)
    # json has keys script, args, seed==42, git_commit, counts=={'rows':3}, packages

def test_paper_root_creates_dir(tmp_path, monkeypatch):
    # monkeypatch paths.EXTERNAL_ROOT -> tmp_path; paper_root('denethor').is_dir()
```

If `pytest-httpserver` is not installed, monkeypatch `requests.get` with a stub returning an iterable of chunks and recording headers; do not add the package.

- [ ] **Step 2: Run tests, verify they fail**

Run: `cd /home/mmann1123/Documents/github/South_Africa_Crop_Comp && /home/mmann1123/miniconda3/envs/ee/bin/python -m pytest experiments/external_audits/tests/test_common.py -v`
Expected: FAIL with ModuleNotFoundError / ImportError. If pytest is missing in `ee`, install it with `pip install pytest` in that env and record it in Task 14's environment export.

- [ ] **Step 3: Implement the four modules with the signatures above**

`fetch.download` uses `requests.get(url, stream=True, timeout=120)` with 8 MiB chunks and writes to `dest.with_suffix(dest.suffix + '.part')` then renames. Manifest is a JSON list; replace an existing entry with the same `key`.

- [ ] **Step 4: Run tests, verify they pass**

Run: same command. Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add experiments/external_audits docs/superpowers
git commit -m "external_audits: scaffold with paths, manifest downloads, run log, EE init"
```

### Task 2: Zonal statistics on EOPatch grids

**Files:**
- Create: `experiments/external_audits/common/zonal.py`
- Test: `experiments/external_audits/tests/test_zonal.py`

**Interfaces:**
- Produces: `zonal.field_index_map(gdf: gpd.GeoDataFrame, transform: affine.Affine, shape: tuple[int,int], crs) -> tuple[np.ndarray, pd.DataFrame]`: reprojects `gdf` to `crs`, rasterises with `rasterio.features.rasterize` (all_touched False), background -1, returns int32 index raster and a frame `fields` with columns `field_pos` (0..n-1 position in the raster), `field_id` (gdf index), `n_pixels`, keeping only fields with `n_pixels > 0`; fields dropped are returned via `fields.attrs['n_dropped']`.
- Produces: `zonal.field_means(cube: np.ndarray, index_map: np.ndarray, n_fields: int) -> np.ndarray`: `cube` shape `(T, H, W)`, returns `(T, n_fields)` means via `np.bincount` per time step, ignoring pixels where `index_map == -1`.
- Produces: `zonal.field_fraction(cube_bool: np.ndarray, index_map, n_fields) -> np.ndarray` same shape, fraction of True per field.
- Produces: `zonal.grid_from_bbox(minx, miny, maxx, maxy, size: int = 2400) -> affine.Affine` (north-up, 10 m).

- [ ] **Step 1: Write failing tests**

```python
def test_field_index_map_two_squares():
    # two 2x2 squares in a 10x10 grid at 10 m; index raster has 4 pixels of 0 and 4 of 1; fields.n_pixels == [4,4]

def test_field_index_map_drops_empty_fields():
    # a polygon fully outside the grid -> not in fields, fields.attrs['n_dropped'] == 1

def test_field_means_matches_numpy():
    # random cube (3,10,10); compare field_means against cube[:, mask].mean(axis=1) per field

def test_field_fraction_bool():
    # cube_bool with known True counts -> fractions 0.25 / 1.0
```

- [ ] **Step 2: Run tests, verify fail**

Run: `python -m pytest experiments/external_audits/tests/test_zonal.py -v` (in `ee` env). Expected: FAIL, module missing.

- [ ] **Step 3: Implement `zonal.py`**

- [ ] **Step 4: Run tests, verify pass**

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add experiments/external_audits/common/zonal.py experiments/external_audits/tests/test_zonal.py
git commit -m "external_audits: field index map and per-field means on EOPatch grids"
```

### Task 3: Plot theme and summary helpers

**Files:**
- Create: `experiments/external_audits/common/plotting.py`
- Create: `experiments/external_audits/common/tables.py`
- Test: `experiments/external_audits/tests/test_tables.py`

**Interfaces:**
- Produces: `plotting.theme() -> None` applying matplotlib rcParams per the `dataviz` skill (load the skill before writing this file; colour-blind safe categorical palette, 300 dpi save default, no top/right spines); `plotting.save(fig, path: Path) -> Path` saving PNG at 300 dpi and PDF beside it.
- Produces: `tables.write_csv(df: pd.DataFrame, path: Path, script: str, inputs: list[str]) -> Path` writing the CSV plus `path.with_suffix('.meta.json')` with `{script, git_commit, run_date, inputs}`.
- Produces: `tables.longest_gap_days(dates: pd.Series, clear: pd.Series, start: str, end: str) -> int` longest run of consecutive days with no clear acquisition inside `[start, end]`, counting from the window start to the first clear date and from the last clear date to the window end.

- [ ] **Step 1: Write failing tests**

```python
def test_write_csv_writes_sidecar(tmp_path): ...
def test_longest_gap_days_simple():
    # clear on 2018-04-10 and 2018-05-20 within 2018-04-01..2018-10-31 -> 164 (2018-05-20 to 2018-10-31)
def test_longest_gap_days_no_clear():
    # no clear dates -> 213 (full window length Apr 1 to Oct 31)
```

- [ ] **Step 2: Run, verify fail.** - [ ] **Step 3: Implement.** - [ ] **Step 4: Run, verify pass (3 passed).**

- [ ] **Step 5: Commit**

```bash
git commit -am "external_audits: plot theme, CSV sidecar metadata, longest-gap helper"
```

### Task 4: TimeMatch footprints and shipped dates

**Files:**
- Create: `experiments/external_audits/timematch/01_fetch_footprints.py`
- Create: `experiments/external_audits/timematch/zipindex.py`
- Test: `experiments/external_audits/tests/test_zipindex.py`

**Interfaces:**
- Consumes: `fetch.range_get`, `fetch.download`, `paths`, `RunLog`.
- Consumes: `research_notes/Missingness confound audit of transfer papers/recon_timematch_metadata_summary.json` (dict keyed by `timematch_data/<country>/<tile>/2017/meta/metadata.pkl` with `dates` as int yyyymmdd and `cloudy_pct` lists) and `recon_timematch_denethor_data.md` for the zip URL and Zip64 central-directory notes.
- Produces: `zipindex.ZipMember(name: str, header_offset: int, compressed_size: int, uncompressed_size: int, method: int)`; `zipindex.read_central_directory(read: Callable[[int,int], bytes], total_size: int) -> list[ZipMember]` (handles Zip64 EOCD locator); `zipindex.member_data_offset(read, member) -> int` (parses the local header to skip name and extra); `zipindex.extract_member(read, member, dest: Path) -> Path` (method 0 copies bytes; method 8 inflates with `zlib.decompressobj(-15)`).
- Produces: `/mnt/bigdrive/external_audits/timematch/footprints/<country>_<tile>/blocks.{shp,shx,dbf,prj}` for all four tiles and `parcels.*` for Austria and the two French tiles if present in the archive; `experiments/external_audits/timematch/results/timematch_dates.csv` with columns `country, tile, date (YYYY-MM-DD), tile_cloudy_pct`; `manifest.json` under the paper root listing every member fetched.
- Zip URL: `https://zenodo.org/records/5636422/files/timematch_data.zip?download=1`; total size 73,904,625,097 bytes (verify with a HEAD request and record).

- [ ] **Step 1: Write failing tests**

```python
def test_read_central_directory_local_zip(tmp_path):
    # build a zip with zipfile (two members, one stored, one deflated); read = lambda s,l: data[s:s+l]
    # members' names and sizes match zipfile.ZipFile(...).infolist()

def test_extract_member_stored_and_deflated(tmp_path):
    # extracted bytes equal originals for both methods

def test_read_central_directory_zip64(tmp_path):
    # zipfile with force_zip64=True on a member -> still parsed (EOCD64 path)
```

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement `zipindex.py`, then `01_fetch_footprints.py`**

The script: HEAD for size; `read = lambda s, l: fetch.range_get(url, s, l)`; fetch the central directory once and cache it as `<paper_root>/central_directory.json` (list of members) so re-runs skip the 162 MB read; filter members whose name matches `meta/blocks/` or `meta/parcels/`; extract; write `timematch_dates.csv` from the recon JSON; log counts (members total, members fetched, dates per tile: expect AT1 58, DK1 52, FR1 62, FR2 39).

- [ ] **Step 4: Run tests (3 passed), then run the script**

Run: `python experiments/external_audits/timematch/01_fetch_footprints.py`
Expected: four `blocks.shp` files exist; `geopandas.read_file` opens each; `timematch_dates.csv` has 211 rows.

- [ ] **Step 5: Commit** (`results/timematch_dates.csv`, scripts, tests; not the footprints)

```bash
git commit -m "timematch audit: range-fetch block footprints from Zenodo zip, shipped date table"
```

### Task 5: TimeMatch per-date cloud fraction in Earth Engine

**Files:**
- Create: `experiments/external_audits/timematch/02_ee_cloud_by_date.py`
- Create: `experiments/external_audits/timematch/eecloud.py`
- Test: `experiments/external_audits/tests/test_eecloud.py`

**Interfaces:**
- Consumes: Task 4 outputs; `ee_init.init()`; footprints via `geopandas`.
- Produces: `eecloud.support_geometry(tile_dir: Path) -> ee.Geometry` from `parcels.shp` if present else `blocks.shp` dissolved, simplified to 50 m, as `ee.Geometry` (upload via `ee.Geometry(geojson)`; if the geometry exceeds the request limit, use the bounding box of blocks and let the WorldCover mask do the cropland selection, and record which path was taken).
- Produces: `eecloud.date_images(tile: str, date: str) -> ee.ImageCollection` of `COPERNICUS/S2_HARMONIZED` with `MGRS_TILE == tile` on that UTC date, joined to `GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED` (band `cs_cdf`) and `COPERNICUS/S2_CLOUD_PROBABILITY` (band `probability`) by `system:index`.
- Produces: `eecloud.cloud_fractions(images: ee.ImageCollection, support: ee.Geometry, cropland: ee.Image) -> dict` computing over `support ∩ cropland` (WorldCover v100 `Map == 40`) at scale 20 m: `n_images`, `n_pixels`, `frac_cloudy_csplus` (mean of `cs_cdf < 0.6`), `frac_cloudy_s2cp40` (mean of `probability >= 40`), `frac_cloudy_s2cp16` (`>= 16`), `frac_nodata` (fraction of support pixels with no observation in the day's mosaic). Mosaic multiple granules of the same date with `.mosaic()` and record `image_ids`.
- Produces: `eecloud.row_from_info(tile, date, info: dict | None) -> dict` pure function: `info None` means no images, yields `n_images = 0` and NaN fractions.
- Produces: `experiments/external_audits/timematch/results/cloud_by_date.csv` with columns `country, tile, date, tile_cloudy_pct, n_images, image_ids, n_pixels, frac_cloudy_csplus, frac_cloudy_s2cp40, frac_cloudy_s2cp16, frac_nodata, support_kind`.

- [ ] **Step 1: Write failing tests (pure parts only)**

```python
def test_rows_for_dates_without_images():
    r = row_from_info('32VNH', '2017-06-01', None)
    assert r['n_images'] == 0 and math.isnan(r['frac_cloudy_csplus'])

def test_row_from_info_maps_keys():
    info = {'n_images': 2, 'n_pixels': 1000, 'csplus_cloudy': 0.42, 's2cp40': 0.4, 's2cp16': 0.6, 'nodata': 0.01, 'image_ids': ['a','b']}
    r = row_from_info('32VNH', '2017-06-01', info)
    assert r['frac_cloudy_csplus'] == 0.42 and r['image_ids'] == 'a;b'
```

- [ ] **Step 2: Run, verify fail.** - [ ] **Step 3: Implement `eecloud.py` and the script**

The script loops tiles then dates from `timematch_dates.csv`, appends one row per date to the CSV after each `getInfo`, and on start skips `(tile, date)` pairs already present (resumable). `--smoke` runs the first two dates of `32VNH` only. `--tiles` and `--force` flags. Wrap each `getInfo` in a retry (3 attempts, exponential backoff) and on final failure write the row with `n_images = -1` and the error in the run log.

- [ ] **Step 4: Run tests (2 passed), then smoke, then full run**

Run: `python experiments/external_audits/timematch/02_ee_cloud_by_date.py --smoke`
Expected: 2 rows, `n_images >= 1`, fractions in [0, 1].
Run: `python experiments/external_audits/timematch/02_ee_cloud_by_date.py`
Expected: 211 rows; run log lists any dates with `n_images == 0`.

- [ ] **Step 5: Commit** (`results/cloud_by_date.csv` and its sidecar, run log, code)

```bash
git commit -m "timematch audit: Cloud Score+ and s2cloudless fractions over fields per shipped date"
```

### Task 6: TimeMatch summary, figure, finding

**Files:**
- Create: `experiments/external_audits/timematch/03_summarise.py`
- Create: `experiments/external_audits/timematch/README.md`

**Interfaces:**
- Consumes: `results/cloud_by_date.csv`, `results/timematch_dates.csv`, `plotting`, `tables`.
- Produces: `results/tile_summary.csv` with one row per tile: `n_dates, mean_tile_cloudy_pct, median_tile_cloudy_pct, dates_tile_over50, mean_field_cloudy_csplus, median_field_cloudy_csplus, dates_field_over50_csplus, dates_field_over80_csplus, expected_cloudy_obs_per_series_csplus (= sum over dates of frac_cloudy_csplus), same four for s2cp40, same four for s2cp16, mean_frac_nodata`.
- Produces: `results/monthly_summary.csv`: per tile and month, `n_dates`, `expected_cloudy_obs_csplus`, `expected_clear_obs_csplus`.
- Produces: `figures/timematch_cloud_by_date.png` and `.pdf`: four panels (AT1, DK1, FR1, FR2), x = date, y = fraction cloudy over fields (Cloud Score+ as filled markers, s2cp40 as open markers), tile-level `tile_cloudy_pct/100` as a thin grey line, horizontal guide at 0.5; shared y-axis; panel titles with tile and `n_dates`.
- Produces: README with sections: What was measured, How to run, Results (the two tables rendered as markdown), Finding (manuscript style, states DK1 versus FR1/FR2 numbers and whether the difference is large, small or absent, and that the paper reports none of these), Limitations of this audit (support asymmetry for Denmark, WorldCover 2020 versus 2017 fields, Cloud Score+ versus the raw inputs the model saw).

- [ ] **Step 1: Write a test for the summary arithmetic**

```python
def test_tile_summary_expected_cloudy_obs():
    # 3 dates with frac 0.2, 0.5, 1.0 -> expected_cloudy_obs 1.7, dates_over50 == 1 (strict >0.5) ... document the strictness chosen
```

- [ ] **Step 2: Run, fail.** - [ ] **Step 3: Implement.** - [ ] **Step 4: Run tests pass; run the script; open the figure and check it reads at phone width (two columns of panels).**

- [ ] **Step 5: Commit**

```bash
git commit -m "timematch audit: per-tile and monthly cloud summaries, figure, finding"
```

### Task 7: DENETHOR mask and label fetch

**Files:**
- Create: `experiments/external_audits/denethor/01_fetch_masks.py`
- Create: `experiments/external_audits/denethor/catalogue.py`
- Test: `experiments/external_audits/tests/test_denethor_catalogue.py`

**Interfaces:**
- Consumes: `fetch.download`, `paths`, `RunLog`; the listing `research_notes/Missingness confound audit of transfer papers/recon_fusion_competition_s3_listing.tsv` (columns `size_bytes`, `key`; keys begin `fusion-competition/`).
- Produces: `catalogue.BASE_URL = 'https://data.source.coop/esa/'`; `catalogue.url(key: str) -> str` (`BASE_URL + key`); `catalogue.S2_TILE_YEARS = ['s2-utm-33N-18E-242N-2018', 's2-utm-33N-17E-243N-2018', 's2-utm-33N-18E-242N-2019', 's2-utm-33N-17E-243N-2019']`; `catalogue.SA_TILE_YEARS = ['s2-utm-34S-19E-258N-2017', 's2-utm-34S-19E-259N-2017', 's2-utm-34S-20E-259N-2017']`; `catalogue.LABELS = {'18E-242N': 'fusion-competition/br-18E-242N-crop-labels-train-2018.geojson', '17E-243N': 'fusion-competition/br-17E-243N-crop-labels-test-2019.geojson', '19E-258N': 'fusion-competition/sa-19E-258N-crop-labels-train-2017.geojson', '19E-259N': 'fusion-competition/sa-19E-259N-crop-labels-train-2017.geojson', '20E-259N': 'fusion-competition/sa-20E-259N-crop-labels-test-2017.geojson'}`; `catalogue.s2_keys(tile_year: str) -> dict[str, str]` returning keys for `CLP` (`data/CLP.npy`), `SCL` (`mask/SCL.npy`), `IS_DATA` (`mask/IS_DATA.npy`), `timestamp` (`timestamp.pkl`), `bbox` (`bbox.pkl`), `meta_info` (`meta_info.pkl`); `catalogue.listing_size(key) -> int` from the TSV; `catalogue.pfqa_key(tile: str, date: str) -> str` giving `fusion-competition/planet/UTM-24000/33N/<tile>/PF-QA/<date>.tif`; `catalogue.tile_of(tile_year) -> str` (`'18E-242N'`).
- Produces: files under `/mnt/bigdrive/external_audits/denethor/<tile_year>/{CLP.npy, SCL.npy, IS_DATA.npy, timestamp.pkl, bbox.pkl, meta_info.pkl}` and `/mnt/bigdrive/external_audits/denethor/labels/<tile>.geojson`; `manifest.json`. Flag `--sa` adds the three South Africa tile-years (optional goal 4; default off).

- [ ] **Step 1: Write failing tests**

```python
def test_s2_keys_paths():
    assert s2_keys('s2-utm-33N-18E-242N-2018')['CLP'] == 'fusion-competition/sentinel-2/s2-utm-33N-18E-242N-2018/data/CLP.npy'
def test_listing_size_known():
    assert listing_size('fusion-competition/sentinel-2/s2-utm-33N-18E-242N-2018/data/CLP.npy') == 829440128
def test_pfqa_key():
    assert pfqa_key('17E-243N', '2019-01-01') == 'fusion-competition/planet/UTM-24000/33N/17E-243N/PF-QA/2019-01-01.tif'
def test_url(): assert url('fusion-competition/x') == 'https://data.source.coop/esa/fusion-competition/x'
```

- [ ] **Step 2: Run, fail.** - [ ] **Step 3: Implement; the script downloads with `expected_bytes` from the listing and verifies `np.load(..., mmap_mode='r').shape[0] == len(timestamps)` after fetching.**

- [ ] **Step 4: Run tests (4 passed); run `python experiments/external_audits/denethor/01_fetch_masks.py`**

Expected: about 10 GB fetched (4 tile-years x 3 arrays of 829 MB plus small files); run log records shapes `(144, 2400, 2400, 1)` for all four and dtype of each array.

- [ ] **Step 5: Commit** (code, tests, run log; not data)

```bash
git commit -m "denethor audit: Source Cooperative catalogue and fetch of CLP, SCL, IS_DATA, labels"
```

### Task 8: DENETHOR per-field clear Sentinel-2 counts

**Files:**
- Create: `experiments/external_audits/denethor/02_s2_clear_counts.py`
- Create: `experiments/external_audits/denethor/s2clear.py`
- Test: `experiments/external_audits/tests/test_s2clear.py`

**Interfaces:**
- Consumes: Task 7 files; `zonal.field_index_map`, `zonal.field_means`, `zonal.field_fraction`, `tables.longest_gap_days`.
- Produces: `s2clear.load_bbox(path: Path) -> tuple[float,float,float,float,str]` returning `(minx, miny, maxx, maxy, crs)`; unpickle `bbox.pkl` with a custom `pickle.Unpickler` whose `find_class` returns a stub for `sentinelhub` classes, then read its coordinate attributes; cross-check against the 24 km UTM grid implied by the tile name (`easting = E * 24000`, `northing = N * 24000`, EPSG:32633 for 33N, EPSG:32734 for 34S) and against the label GeoJSON bounds; raise if they disagree by more than 100 m, and record both in the run log.
- Produces: `s2clear.detect_clp_scale(clp: np.ndarray) -> int` returning 255 if `clp.max() > 100` else 100 (sample a stride of the array, not all 829 MB at once).
- Produces: `s2clear.clear_rules(clp_mean: np.ndarray, scl_mode_clear_frac: np.ndarray, isdata_frac: np.ndarray, scale: int) -> dict[str, np.ndarray]` with boolean arrays `(T, n_fields)` for `literal40` (`clp_mean < 40` on the stored scale), `s2cp40` (`clp_mean < 0.40 * scale`), `scl` (`scl_mode_clear_frac >= 0.5` and `isdata_frac >= 0.99`), where `scl_mode_clear_frac` is the per-field fraction of pixels in SCL classes {4, 5, 6, 7}.
- Produces: `results/s2_field_dates.csv` (long: `tile_year, tile, year, field_id, date, clp_mean, scl_clear_frac, isdata_frac, clear_literal40, clear_s2cp40, clear_scl`) and `results/s2_field_summary.csv` (per `tile_year, field_id`: `n_pixels, crop (label column as shipped), n_clear_literal40, n_clear_s2cp40, n_clear_scl, n_clear_s2cp40_aprOct, longest_gap_days_s2cp40_aprOct, clear_by_month_s2cp40 (12 ints joined by ';')`). Labels for a tile are reused across its two years and the summary notes `labels_from_year`.
- Chunk the arrays along T with `np.load(mmap_mode='r')`, 8 time steps at a time, to keep memory under 4 GB.

- [ ] **Step 1: Write failing tests**

```python
def test_clp_scale_detection():
    assert detect_clp_scale(np.array([0, 3, 250], dtype=np.uint8)) == 255
    assert detect_clp_scale(np.array([0, 3, 99], dtype=np.uint8)) == 100
def test_clear_rules_thresholds():
    # clp_mean 39.9 -> literal40 True on scale 255 and s2cp40 True (threshold 102); clp_mean 101 -> literal False, s2cp40 True; 103 -> both False
def test_load_bbox_grid_consistency(tmp_path):
    # pickle a stub object with min_x.. attributes = 18*24000 etc.; load_bbox returns them and crs 'EPSG:32633'
```

- [ ] **Step 2: Run, fail.** - [ ] **Step 3: Implement.** - [ ] **Step 4: Run tests (3 passed); run the script on all four tile-years**

Run: `python experiments/external_audits/denethor/02_s2_clear_counts.py`
Expected: summary rows equal the number of label polygons that rasterise to at least one pixel for each tile (two tile-years per tile share the count); run log reports `clp_scale`, dropped fields, T = 144 for all four.

- [ ] **Step 5: Commit** (`results/*.csv` if under 20 MB, else keep `s2_field_dates.csv` under the external root and commit only the summary; say which in the run log)

```bash
git commit -m "denethor audit: per-field clear Sentinel-2 counts under three cloud rules, 2x2 tile-years"
```

### Task 9: DENETHOR Planet Fusion synthetic fraction

**Files:**
- Create: `experiments/external_audits/denethor/03_pf_qa_synthetic.py`
- Create: `experiments/external_audits/denethor/pfqa.py`
- Test: `experiments/external_audits/tests/test_pfqa.py`

**Interfaces:**
- Consumes: `catalogue.pfqa_key`, `catalogue.url`, labels, `zonal` helpers; season April 1 to October 31 for `('18E-242N', 2018)` and `('17E-243N', 2019)`.
- Produces: `pfqa.read_day(url_or_path: str, window: rasterio.windows.Window) -> tuple[np.ndarray, np.ndarray] | None` returning `(percent_synthetic, days_to_real)` as layers 1 and 2 (1-based band indices 1 and 2) of the int16 raster inside the window, or `None` if the file is missing (HTTP 404) or unreadable; opens with `rasterio.open('/vsicurl/' + url)` when given a URL.
- Produces: `pfqa.field_day_stats(percent_synthetic, days_to_real, index_map, n_fields, synthetic_threshold: int = 50) -> pd.DataFrame` with `field_pos, frac_synthetic (fraction of pixels with percent_synthetic > 50), mean_percent_synthetic, mean_days_to_real`.
- Produces: `pfqa.summarise(daily: pd.DataFrame) -> pd.DataFrame`: per field and month `n_days_with_qa, share_days_mostly_synthetic (frac_synthetic > 0.5), mean_percent_synthetic, mean_days_to_real`; days with no QA are excluded from denominators and counted in `n_days_missing`.
- Produces: `results/pfqa_field_day.csv` (or under the external root if over 20 MB) and `results/pfqa_field_month.csv`; per-day cache `/mnt/bigdrive/external_audits/denethor/pfqa_cache/<tile>/<date>.npz` so re-runs skip fetched days.
- The window is the union bounding box of the tile's label polygons in the PF-QA raster's CRS at 3 m; the index map is rasterised at that 3 m grid. If `/vsicurl/` reads are slower than about 20 s per day, download the season's COGs (about 214 files per tile-year) with `fetch.download` and read locally; record which path was used.

- [ ] **Step 1: Write failing tests**

```python
def test_missing_day_excluded(tmp_path):
    # daily frame with one day frac_synthetic NaN (no QA) -> n_days_with_qa excludes it, n_days_missing == 1
def test_field_day_stats_threshold():
    # 4 pixels with percent_synthetic [1, 60, 100, 1] -> frac_synthetic 0.5, mean 40.5
def test_read_day_returns_none_on_missing(tmp_path):
    # nonexistent path -> None
```

- [ ] **Step 2: Run, fail.** - [ ] **Step 3: Implement.** - [ ] **Step 4: Run tests (3 passed); run with `--smoke` (first 3 days) then the full season for both tile-years.**

Expected: 214 days per tile-year attempted; run log lists missing days; `pfqa_field_month.csv` has 7 months x fields rows.

- [ ] **Step 5: Commit**

```bash
git commit -m "denethor audit: per-field Planet Fusion synthetic fraction from PF-QA, 2018 vs 2019"
```

### Task 10: DENETHOR summary, figure, finding

**Files:**
- Create: `experiments/external_audits/denethor/04_summarise.py`
- Create: `experiments/external_audits/denethor/README.md`

**Interfaces:**
- Consumes: `results/s2_field_summary.csv`, `results/pfqa_field_month.csv`, `plotting`, `tables`.
- Produces: `results/denethor_summary.csv`: one row per tile-year with `n_fields, median_n_clear_s2cp40, iqr_n_clear_s2cp40, median_n_clear_literal40, median_n_clear_scl, share_fields_under_20_clear_aprOct_s2cp40, median_longest_gap_days, mean_clear_by_month (12 values)`, and for the two benchmark tile-years `mean_share_days_mostly_synthetic_season, by_month (7 values)`.
- Produces: `results/denethor_2x2.csv`: the 2x2 of tile x year on `median_n_clear_s2cp40` and the two marginal differences (year effect within tile, tile effect within year).
- Produces: `figures/denethor_clear_and_synthetic.png/.pdf`: panel (a) distributions (violin or histogram) of `n_clear_s2cp40_aprOct` per field for the four tile-years, benchmark pair highlighted; panel (b) monthly share of mostly-synthetic Planet Fusion days, 2018 train tile versus 2019 test tile, April to October.
- Produces: README with What was measured, How to run, Results tables, Finding (states 2018 versus 2019 and tile versus tile numbers, cites the paper's "around 55 percent of S2 observations discarded" and Pham 2024's "more than 80 clear-sky observations in 2018", says whether the year difference is large, small or absent), Limitations (labels reused across years, CLP scale assumption, PF-QA definition of synthetic).

- [ ] **Step 1: Test `denethor_2x2` arithmetic on a toy frame.** - [ ] **Step 2: Run, fail.** - [ ] **Step 3: Implement.** - [ ] **Step 4: Run tests pass; run the script; check the figure.**

- [ ] **Step 5: Commit**

```bash
git commit -m "denethor audit: 2x2 clear-count summary, synthetic-day figure, finding"
```

### Task 11: DACCN recipe port with validity and zero counters

**Files:**
- Create: `experiments/external_audits/daccn/you2021_recipe.py`
- Create: `experiments/external_audits/daccn/reference.py`
- Test: `experiments/external_audits/tests/test_daccn_reference.py`
- Copy for provenance: `experiments/external_audits/daccn/you2021_GEE_code_for_crop_classification.js` (from `research_notes/Missingness confound audit of transfer papers/you2021_gee_code/`, unchanged)

**Interfaces:**
- Produces: `reference.zero_flags(valid: np.ndarray, targets: range = range(9, 31), search: int = 9) -> tuple[np.ndarray, np.ndarray]` numpy model of the script's interpolation for one pixel: `valid` is a boolean array over the 37 composites (pixel has a valid median in that composite); returns `(is_zero, is_valid)` over the 22 targets where `is_zero[i]` is True iff the target composite is invalid for the pixel AND (no valid composite among the 9 before OR no valid composite among the 9 after). This mirrors `targetImg.unmask(Y)` followed by `unmask(0)`, since `Y` is masked wherever `before` or `after` is masked.
- Produces: `you2021_recipe.simple_cloud_score(toa: ee.Image) -> ee.Image` (exact port of `cloudMask` score with the six `rescale` terms; bands renamed as in `sentinel2toa`: B1 aerosol, B2 blue, B3 green, B4 red, B8A red4, B10 cirrus, B11 swir1); `you2021_recipe.masked_collection(region: ee.Geometry, year: int, variant: str) -> ee.ImageCollection` on `COPERNICUS/S2_HARMONIZED` divided by 10000, filtered to the whole year for variant `'A'` and to `YYYY-04-01` to `YYYY-10-31` inclusive for `'B'`, masked where `score > 0.2`; `you2021_recipe.composites(collection, year, region) -> ee.List` of 37 images each with band `valid` (the median's mask as 0/1, built from one band, `red`) and properties `doy`, `length`; `you2021_recipe.counter_image(composites: ee.List, targets=(9, 30), search=9) -> ee.Image` with bands `valid_00..valid_21`, `zero_00..zero_21`, `n_valid`, `n_zero`, implementing the before/after mosaic logic on the `valid` band exactly as the script does on data bands (`filter(gt('length', 0))`, `.mosaic()`, masked arithmetic), then `unmask(0)` semantics as `zero = target_invalid AND (before_masked OR after_masked)`.
- Produces: `you2021_recipe.smoke(region: ee.Geometry, year: int, variant: str) -> dict` running `counter_image(...).reduceRegion(ee.Reducer.mean(), region, 10)` on a 1 km box and returning the means.

- [ ] **Step 1: Write failing tests for the numpy reference**

```python
def test_zero_flags_all_valid(): valid = np.ones(37, bool) -> is_zero all False, is_valid all True
def test_zero_flags_isolated_gap(): valid[15] = False, rest True -> is_zero[15-9] False (interpolated), is_valid False there
def test_zero_flags_edge_gap_variant_b_like():
    # valid False for indices 0..8 and 31..36 (no winter imagery), True 9..30 except 9 -> target 9 (step 0) is zero (no valid before)
def test_zero_flags_long_gap():
    # valid False for 12..30 inclusive -> targets 22..30 (steps 13..21) zero (no valid after within 9); targets 12..21 interpolated from 11 and 31? only if 31 valid and within 9 -> check 12..21: after search reaches index 31 only for targets >= 22; so 12..21 are zero too? compute by hand and assert exact pattern
```

Compute the expected patterns by hand before writing the asserts; put the derivation in a comment.

- [ ] **Step 2: Run, fail.** - [ ] **Step 3: Implement `reference.py`, then `you2021_recipe.py`.** - [ ] **Step 4: Run tests (4 passed); run the smoke**

Run: `python -c "from experiments.external_audits.daccn.you2021_recipe import smoke; import ee; from experiments.external_audits.common.ee_init import init; init(); print(smoke(ee.Geometry.Rectangle([-93.5,44.0,-93.49,44.01]), 2018, 'A'))"`
Expected: a dict with `n_valid` between 0 and 22, `n_zero` between 0 and 22, and `n_zero` for variant `'B'` on the same box greater than or equal to variant `'A'`.

- [ ] **Step 5: Commit**

```bash
git commit -m "daccn audit: faithful port of You et al. 2021 compositing/interpolation with validity and zero counters"
```

### Task 12: DACCN regions, sampling frames, and counter runs

**Files:**
- Create: `experiments/external_audits/daccn/regions.py` (importable module) and `experiments/external_audits/daccn/01_regions.py` (CLI wrapper that prints each region's area and frame label; numbered files cannot be imported, so all importable code lives in unnumbered modules)
- Create: `experiments/external_audits/daccn/02_run_counters.py`
- Test: `experiments/external_audits/tests/test_daccn_regions.py`

**Interfaces:**
- Produces: `regions.REGIONS = [('Minnesota','United States of America'), ('Nebraska','United States of America'), ('Wisconsin','United States of America'), ('Iowa','United States of America'), ('Heilongjiang','China'), ('Jilin','China')]`; `regions.lookup(name: str, country: str) -> ee.Feature` from `FAO/GAUL/2015/level1` filtering `ADM1_NAME == name` and `ADM0_NAME == country`, asserting `.size().getInfo() == 1`; `regions.frame_image(region_key: str, year: int) -> tuple[ee.Image, str]` returning a class image and a frame label: US states use `USDA/NASS/CDL/<year>` band `cropland` restricted to DACCN's classes (read the class list from the DACCN card in the audit report; corn code 1 and soybeans code 5 at minimum; record the list used in the run log) with band `confidence > 90`, frame label `'cdl_conf90'`; Chinese provinces use `ESA/WorldCover/v100/2020` where `Map == 40` as a single class, frame label `'worldcover40'`.
- Produces: `02_run_counters.py`: for each region, year in (2018, 2019), variant in ('A', 'B'): `counter_image(...)` plus the class band, `stratifiedSample(numPoints = 20000 // n_classes, classBand = 'class', region, scale = 10, seed = 42, geometries = False, tileScale = 4)`, `getInfo` in chunks if needed (sample per class separately when the single call exceeds limits), and appends to `results/daccn_counters.csv` with columns `region, country, year, variant, frame, class, n_valid, n_zero, zero_00..zero_21, valid_00..valid_21`. Resumable by `(region, year, variant)`; `--smoke` runs Iowa 2018 A with 200 points; falls back to `Export.table.toDrive` with a documented folder name `external_audits_daccn` if `getInfo` fails three times, and records the task id.

- [ ] **Step 1: Write failing tests**

```python
def test_region_lookup_single_feature(monkeypatch):
    # monkeypatch ee.FeatureCollection to return objects whose size().getInfo() is 2 -> lookup raises ValueError
def test_regions_list_has_six(): assert len(REGIONS) == 6
```

- [ ] **Step 2: Run, fail.** - [ ] **Step 3: Implement.** - [ ] **Step 4: Run tests (2 passed); smoke; full run of 24 jobs**

Expected: `daccn_counters.csv` has about 24 x 20,000 rows (fewer where a class is scarce); run log records points per job, the CDL class list, and any Drive fallbacks.

- [ ] **Step 5: Commit** (CSV if under 20 MB; otherwise gzip it and commit the `.csv.gz`)

```bash
git commit -m "daccn audit: GAUL regions, CDL/WorldCover frames, stratified counter samples for 24 region-year-variants"
```

### Task 13: DACCN summary, figure, finding

**Files:**
- Create: `experiments/external_audits/daccn/03_summarise.py`
- Create: `experiments/external_audits/daccn/README.md`

**Interfaces:**
- Consumes: `results/daccn_counters.csv(.gz)`, `plotting`, `tables`.
- Produces: `results/daccn_summary.csv`: per `region, year, variant`: `n_points, median_n_valid, mean_n_valid, share_any_zero (n_zero > 0), mean_n_zero, share_zero_step01, share_zero_step22, share_zero_apr (steps 1 to 3), share_zero_oct (steps 20 to 22)`.
- Produces: `results/daccn_pairs.csv`: for the DACCN transfer pairs (MN to WI, NE to IA, MN to HLJ, NE to JL, each region 2018 to 2019), variant A and B: `source_share_any_zero, target_share_any_zero, difference, source_mean_n_valid, target_mean_n_valid`.
- Produces: `figures/daccn_zero_by_step.png/.pdf`: small multiples, one panel per region, x = step 1 to 22 (DOY 90 to 300), y = share of pixels zero at that step, lines for 2018 and 2019, variant A solid and variant B dashed; shared y-axis.
- Produces: README with What was measured, How to run, the deviation list from the deposited script (stop before Savitzky-Golay; one carrier band; 22 versus DACCN's 21 steps; cubic versus quadratic noted), Results tables, Finding, Limitations.

- [ ] **Step 1: Test pair-table arithmetic on a toy frame.** - [ ] **Step 2: Run, fail.** - [ ] **Step 3: Implement.** - [ ] **Step 4: Run tests pass; run; check figure.**

- [ ] **Step 5: Commit**

```bash
git commit -m "daccn audit: zero-fill share by step and transfer-pair table, figure, finding"
```

### Task 14: Clean-up, run_all, environments, report note

**Files:**
- Create: `experiments/external_audits/common/cleanup.py`
- Create: `experiments/external_audits/timematch/99_cleanup.py`, `denethor/99_cleanup.py`, `daccn/99_cleanup.py`
- Create: `experiments/external_audits/run_all.sh`, `timematch/run.sh`, `denethor/run.sh`, `daccn/run.sh`
- Create: `experiments/external_audits/env/environment.ee.yml`, `env/environment.ml_field.yml`, `env/pip_freeze_ee.txt`
- Modify: `experiments/external_audits/README.md` (complete it)
- Modify: `reports/Missingness confound audit of transfer papers.md` (add a "Measured" note to the TimeMatch, DENETHOR and DACCN rows of Table 1 and a short paragraph at the end of each paper's evidence block pointing to the audit README and its finding)
- Test: `experiments/external_audits/tests/test_cleanup.py`

**Interfaces:**
- Produces: `cleanup.expected_outputs(paper: str) -> list[tuple[Path, int]]` listing required result files and minimum row counts (`timematch`: `cloud_by_date.csv` 200 rows, `tile_summary.csv` 4; `denethor`: `s2_field_summary.csv` 1000, `denethor_summary.csv` 4, `pfqa_field_month.csv` 1000; `daccn`: `daccn_counters.csv[.gz]` 100000, `daccn_summary.csv` 24); `cleanup.complete(paper) -> tuple[bool, list[str]]`; `cleanup.purge(paper: str, keep: tuple[str, ...] = ('manifest.json', 'central_directory.json')) -> int` deleting everything else under `paper_root(paper)` and returning bytes freed; refuses to run if `complete(paper)` is False unless `--force`.
- `run_all.sh`: activates `ee` via `source ~/miniconda3/etc/profile.d/conda.sh && conda activate ee`, runs the three `run.sh` in order, then `99_cleanup.py` for each unless `--keep-data`; exits non-zero on the first failure; prints the three README finding paragraphs at the end.

- [ ] **Step 1: Write failing tests**

```python
def test_complete_false_when_missing(tmp_path, monkeypatch): ...
def test_purge_keeps_manifest(tmp_path, monkeypatch):
    # create manifest.json and two data files; purge -> only manifest.json remains, returned bytes equal the two files' sizes
def test_purge_refuses_when_incomplete(tmp_path, monkeypatch): raises RuntimeError
```

- [ ] **Step 2: Run, fail.** - [ ] **Step 3: Implement; write the env exports with `conda env export --from-history -n ee` and `pip freeze` from the `ee` env; complete the README (layout, one-command run, data root, clean-up behaviour, Drive export caveat).**

- [ ] **Step 4: Run tests (3 passed); run `bash experiments/external_audits/run_all.sh --keep-data` end to end from a clean state of results (move existing results aside first), confirm identical summary CSVs to within sampling noise; then run `99_cleanup.py` for each paper and confirm `ls /mnt/bigdrive/external_audits/<paper>` shows only the manifest (and the TimeMatch central directory cache).**

- [ ] **Step 5: Update the audit report rows and evidence blocks with the measured numbers; commit**

```bash
git add experiments/external_audits reports/"Missingness confound audit of transfer papers.md"
git commit -m "external_audits: cleanup, run_all, pinned environments, measured findings into audit report"
```
