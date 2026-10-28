# External missingness audits: TimeMatch, DENETHOR, DACCN

Date: 2026-10-28. Status: builder check done; four open points settled below as the author's calls (marked "my call").

## Intent

Three published crop-type transfer papers were ranked "highly suspect" in
`reports/Missingness confound audit of transfer papers.md` because a
difference in cloud or gap structure between their training and holdout
domains could explain part of the transfer loss they attribute to phenology
or spectral shift. This project measures that cloud or gap difference
directly from each paper's own data or preprocessing recipe, so the
missingness paper can point to concrete numbers ("the target had N times the
cloudy observations of the source") rather than a plausibility argument.

No model is re-trained or re-run. The deliverable per paper is a small set
of tables and one figure describing the missingness structure of the source
and target domains, plus a short written finding that either supports or
weakens the paper's attribution. A null result (missingness matched) is a
valid outcome and gets written up with the same care.

## Goals

1. TimeMatch (Nyborg et al. 2022): per tile and per shipped acquisition
   date, the fraction of cropland or parcel pixels that are cloudy; from it,
   per tile, the expected number of cloudy observations per parcel series,
   its monthly distribution, and the share of dates above 50 and 80 percent
   cloud over fields. Headline comparison: Denmark 32VNH versus France
   30TXT and 31TCJ, the pair with the 43 point source-only collapse.
2. DENETHOR (Kondmann et al. 2021): per field and per year (2018 train tile
   18E-242N, 2019 test tile 17E-243N), the number of clear Sentinel-2
   acquisitions out of 144 using the shipped CLP and SCL layers, and the
   fraction of Planet Fusion daily inputs that were synthesised rather than
   observed using the shipped PF-QA "percent synthetic" layer. Headline
   comparison: 2018 versus 2019 over the growing season, beside Pham et al.
   2024's statement that 2018 was Brandenburg's clearest year.
3. DACCN (Wang, Feng, Zhang and Tian 2023): re-run the compositing and
   interpolation stage of the You et al. 2021 Earth Engine recipe that DACCN
   states it followed, for 2018 and 2019 over Minnesota, Nebraska,
   Wisconsin, Iowa, Heilongjiang and Jilin, and export the two counters the
   recipe never computes: per cropland pixel, the number of valid 10-day
   composites out of 22, and the number of composites that ended up as
   `unmask(0)` zeros. Headline comparisons: each DACCN transfer pair
   (Minnesota to Wisconsin, Nebraska to Iowa, each region 2018 to 2019, US
   to China).
4. Optional, same code path as goal 2: the three AI4FoodSecurity South
   Africa 2017 tiles sit in the same bucket with identical CLP and SCL
   layers, so the per-field clear-count comparison between the two training
   tiles and the holdout tile (the sub-tile check the forensics researcher
   could not run) falls out of the DENETHOR script with a different tile
   list. Build it only after goals 1 to 3 are done.

## Non-goals and anti-goals

- No model training, fine-tuning or re-scoring of any paper's method.
- No download of the TimeMatch 74 GB archive or the DENETHOR 500 GB Planet
  surface-reflectance imagery. Only metadata, mask and QA layers are read.
- Do not claim the papers' conclusions are wrong. The finding is the size
  of the missingness difference and whether the papers measured it, not a
  re-estimate of their transfer loss.
- Anti-goal: an audit whose own missing-data handling is undocumented.
  Every script must print and save the counts it is built on (dates
  found, pixels sampled, fields matched, nodata encountered).

## Constraints

- Earth Engine access works from the `ee` conda environment with the
  project id stored in `~/.config/earthengine/credentials` (project
  327687702789). Default scopes were granted, so Export tasks to Drive are
  allowed. Stay within the free interactive and batch quota: reduce over
  stratified pixel samples, not full-tile rasters, and prefer
  `reduceRegion` or `sampleRegions` with modest sample sizes (order 20,000
  pixels per region-year) returned via `getInfo`; fall back to a Drive
  export only if a reduction times out.
- DENETHOR data are on Source Cooperative
  (`https://source.coop/esa/fusion-competition`), anonymous HTTPS, S3-style
  listing. The local `aws` CLI is broken (urllib3 mismatch); use `curl` or
  Python `requests` with Range headers, and GDAL `/vsicurl/` for windowed
  COG reads. The full S3 object listing with sizes is already saved at
  `research_notes/Missingness confound audit of transfer papers/recon_fusion_competition_s3_listing.tsv`.
- TimeMatch per-tile date lists and tile-level cloudy percentages are
  already extracted to
  `research_notes/Missingness confound audit of transfer papers/recon_timematch_metadata_summary.json`;
  the block shapefiles (all four tiles) and parcel shapefiles (Austria and
  France only) are inside the Zenodo zip and can be range-fetched without
  downloading the archive (offsets are in the zip central directory; see
  `recon_timematch_denethor_data.md`).
- Python environments: Earth Engine work runs in `ee` (has earthengine-api
  1.7.50, geopandas, rasterio). DENETHOR numpy and raster work runs in
  `ml_field` (geopandas 1.0.1, rasterio 1.4.4) or `ee`; either is fine, the
  builder picks one per script and records it in the README.
- All numbers in prose and figures follow the manuscript style: no em
  dashes, concise wording. Figures follow the `dataviz` skill (load it
  before writing plotting code).

## Layout (my call)

```
experiments/external_audits/
  README.md                      overview, how to run, where results land
  common/                        shared helpers (EE init, S3 range fetch,
                                 field-polygon zonal stats, plotting theme)
  timematch/
    01_fetch_footprints.py       range-fetch block/parcel shapefiles from zip
    02_ee_cloud_by_date.py       Cloud Score+ and S2_CLOUD_PROBABILITY per
                                 shipped date over cropland in footprint
    03_summarise.py              tables and figure
    results/  figures/  README.md (findings)
  denethor/
    01_fetch_masks.py            CLP.npy, SCL.npy, IS_DATA.npy,
                                 timestamp.pkl, labels GeoJSON per tile-year
    02_s2_clear_counts.py        per-field clear S2 acquisitions
    03_pf_qa_synthetic.py        per-field percent-synthetic from PF-QA COGs
    04_summarise.py              tables and figure
    results/  figures/  README.md
  daccn/
    you2021_recipe.py            faithful Python port of the compositing and
                                 interpolation stage (cloud score 0.2,
                                 10-day medians, +/- 9 composite search,
                                 unmask(0)), plus the two counters
    01_regions.py                region geometries and cropland masks
    02_run_counters.py           per region-year stratified sample, getInfo
    03_summarise.py              tables and figure
    results/  figures/  README.md
```

Downloaded external data go under `/mnt/bigdrive/external_audits/<paper>/`
(outside Dropbox to avoid syncing several gigabytes), with the path set in
one place in `experiments/external_audits/common/paths.py`. Results, figures
and READMEs are committed; downloaded data are not.

## Decisions already made (from the conversation and recon)

### TimeMatch

- s2cloudless cannot be run on the shipped arrays: bands B01, B09 and B10
  are absent (channel order B02, B03, B04, B08, B05, B06, B07, B8A, B11,
  B12). No per-pixel cloud layer is shipped; `valid_pixels` is all ones.
- Therefore the per-pixel cloud fraction is computed in Earth Engine over
  the exact shipped dates (52 for DK1 32VNH, 62 for FR1 30TXT, 39 for FR2
  31TCJ, 58 for AT1 33UVP, all 2017) from the dataset's `metadata.pkl`
  date lists already extracted to the recon JSON. Match each shipped
  yyyymmdd to the Sentinel-2 granules of that MGRS tile on that date
  (`COPERNICUS/S2_HARMONIZED`, `MGRS_TILE` property).
- Cloud layers: primary `GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED` band
  `cs_cdf` with the documented threshold 0.6 (clear if cs_cdf >= 0.6);
  secondary `COPERNICUS/S2_CLOUD_PROBABILITY` (s2cloudless) at 40 percent,
  which is the shared cross-paper definition (DENETHOR's CLP is the same
  s2cloudless product), and additionally at about 16 percent, which is
  what DENETHOR's literal `cld < 40` on a 0 to 255 scale amounts to.
  Report all three.
- Spatial support: the dataset's block polygons (shipped for all four
  tiles) intersected with a cropland mask (ESA WorldCover 2020 class 40,
  or the shipped parcel polygons where available for AT and FR). Denmark
  has no shipped parcels; blocks plus WorldCover cropland is the support
  there, and the README states the asymmetry.
- Outputs: `results/cloud_by_date.csv` (tile, date, n_pixels, frac_cloudy
  csplus, frac_cloudy s2cp, tile_cloudy_pct from metadata), per-tile
  summary table (dates, mean and median field-level cloud fraction, dates
  above 50 and 80 percent, expected cloudy observations per parcel series,
  monthly breakdown), and one figure: four-panel timeline of field-level
  cloud fraction per shipped date, one panel per tile, with the tile-level
  metadata percentage overlaid so the gap between "what the filter saw"
  and "what the fields saw" is visible.
- Also report the tile-level statistics already in hand (dates per tile,
  mean and median cloudy_pct, dates above 50 percent) as the zero-cost
  baseline.

### DENETHOR

- Fetch only `data/CLP.npy` (829 MB), `mask/SCL.npy`, `mask/IS_DATA.npy`,
  `timestamp.pkl`, `bbox.pkl` or equivalent georeference, and the label
  GeoJSON. Do not fetch `BANDS.npy` (19.9 GB each).
- Tile-years (my call, after the builder check): all four Sentinel-2
  tile-years in the bucket, not just the two benchmark ones. The 2x2 of
  {18E-242N, 17E-243N} x {2018, 2019} separates the tile change from the
  year change, which matches the paper's own split of its loss into about
  9 points spatial and 12 points temporal. Labels exist only for
  18E-242N-2018 and 17E-243N-2019, so each tile's labelled polygons are
  reused for its other year and the README says so. Adds about 5 GB. The
  Planet Fusion QA part stays on the two benchmark tile-years, the only
  ones with Planet data.
- Clear-observation definition (my call, after the builder check): first
  read the CLP value range from the array; the recon infers Sentinel Hub
  uint8 0 to 255, in which case the authors' `cld < 40` is a 15.7 percent
  cut, not 40 percent. Report three rules per field and acquisition:
  (i) the authors' literal rule, field-mean CLP below 40 on the stored
  scale; (ii) s2cloudless probability below 40 percent (CLP below 102 if
  the scale is 0 to 255), which is the shared cross-paper definition used
  in the TimeMatch audit; (iii) field-majority SCL class in {4, 5, 6, 7}
  with IS_DATA true. Rule (ii) is primary in the cross-paper comparison;
  rule (i) is what DENETHOR actually applied and is shown beside it.
- Per field and year: number of clear acquisitions out of 144, monthly
  counts, longest clear-free gap in days within April to October. Compare
  2018 versus 2019 distributions (median, interquartile range, share of
  fields with fewer than N clear acquisitions in the season).
- PF-QA: layer 1 is percent synthetic (1 = observed, 100 = fully
  gap-filled), layer 2 is days to nearest real observation. Read these two
  layers windowed through `/vsicurl/` for every day in April to October of
  each year (about 214 COGs per year at 27 MB each; windowed reads over the
  field bounding boxes are far smaller) and compute per field the fraction
  of days with percent synthetic above 50, and the mean days to nearest
  real observation, per month. If windowed reads are slow, download the
  PF-QA COGs for the season (about 6 GB per year) to the external data root.
- Outputs: `results/s2_clear_counts_by_field.csv`,
  `results/pf_qa_by_field_month.csv`, a summary table (2018 vs 2019), and
  one two-panel figure: (a) distribution of clear Sentinel-2 acquisitions
  per field, 2018 vs 2019; (b) monthly share of synthetic Planet Fusion
  days, 2018 vs 2019.
- Cite beside it the paper's own "around 55 percent of S2 observations
  discarded" (no per-year split) and Pham et al. 2024's "more than 80
  clear-sky observations on average in 2018".

### DACCN

- Port the You et al. 2021 `GEE_code_for_crop_classification.js`
  compositing and interpolation stage to the Earth Engine Python API,
  faithfully: `COPERNICUS/S2_HARMONIZED` L1C, simple cloud score with
  threshold 0.2 and no shadow test, 10-day median composites, linear
  interpolation searching at most 9 composites each side, then
  `unmask(0)`.
- Compositing window (my call, after the builder check): run two variants
  as a sensitivity pair. Variant A, "recipe as deposited", is the literal
  script: 37 composites over the whole calendar year (DOY 0 to 360) with
  interpolation applied to list indices 9 to 30, so the edge searches can
  reach into winter composites. Variant B, "as DACCN describes", feeds only
  April to October imagery, so the 22 composites have nothing outside the
  window to interpolate from. A is primary in the tables and figure; B is
  reported beside it and the README explains why the two differ at steps 1
  and 22. Cost is 24 small Earth Engine jobs instead of 12. Stop before Savitzky-Golay (private module, and
  smoothing only spreads zeros). Keep the JS file and port side by side and
  document every deviation in the README.
- Add two per-pixel counters: `n_valid` (composites with a valid median
  before interpolation) and `n_zero` (composites that are zero after
  `unmask(0)`, meaning no valid composite within 9 steps on at least one
  side). Also export a 22-element per-step zero flag so the seasonal
  position of zeros can be plotted.
- Regions: state and province boundaries from `FAO/GAUL/2015/level1`.
  Cropland support: USDA CDL (`USDA/NASS/CDL`) cropland classes for the
  four US states in the matching year; for Heilongjiang and Jilin, ESA
  WorldCover 2020 class 40 (the You et al. 2021 crop maps are not in the
  EE catalogue; if they can be found as an asset, prefer them and note it).
- Sampling frame (my call): mirror DACCN's own training frame where the
  inputs exist. In the four US states, sample CDL pixels of the classes
  DACCN used (read them from the report's DACCN card; corn and soybean at
  minimum) with the CDL `confidence` band above 90, stratified by class
  with equal counts per class, about 20,000 pixels per region-year in
  total, fixed seed. In Heilongjiang and Jilin, where the You et al. 2021
  maps are not in the catalogue, sample WorldCover 2020 class 40 stratified
  by region only, and if a copy of the You et al. maps can be ingested as
  an asset, prefer it and say so. Every README table states which frame
  each row used. `sampleRegions` returns `n_valid`, `n_zero` and the step
  flags; aggregate locally.
- Outputs: `results/daccn_counters_by_region_year.csv` (per sample pixel),
  summary table per region-year (median n_valid, share of pixels with
  n_zero > 0, mean n_zero, share zero at step 1 and step 22), a pairwise
  table for the DACCN transfer pairs, and one figure: per region-year, the
  share of pixels zero-filled at each of the 22 steps (small multiples,
  2018 vs 2019 lines).
- Caveat recorded in the README: DACCN reports 21 steps and a quadratic
  Savitzky-Golay against the script's 22 and cubic, so the port measures
  the recipe as deposited, and the edge behaviour is an inference about
  how closely DACCN reproduced it.

### Shared

- Replicability (user requirement). Each audit must be re-runnable from a
  clean checkout by someone else with one command: an `experiments/
  external_audits/run_all.sh` that calls each paper's numbered scripts in
  order with the right conda environment, and per-paper `run.sh`. Pin the
  software: export `environment.ee.yml` and `environment.ml_field.yml`
  (`conda env export --from-history` plus the resolved earthengine-api,
  geopandas, rasterio, numpy, pandas, matplotlib versions) into
  `experiments/external_audits/env/`. Fix every random seed and write it
  to the run log. Record the exact Earth Engine collection ids and the
  image ids or system:index values actually used per date, so a later run
  can detect catalogue changes. Record the Source Cooperative object keys,
  byte sizes and SHA-256 of every file downloaded in a `manifest.json` per
  paper. Every CSV in `results/` carries a header comment or sidecar JSON
  naming the script, git commit, run date and input manifest that produced
  it. No notebook-only steps; no hand edits to results.
- Every script is idempotent, skips work whose output exists unless
  `--force`, and writes a `run_log.json` with inputs, counts and timings.
- Clean-up (user requirement). Downloaded external data are temporary.
  Each paper has a `99_cleanup.py` that deletes everything under
  `/mnt/bigdrive/external_audits/<paper>/` once `results/` and `figures/`
  exist and pass a completeness check (expected files present, row counts
  above a minimum), leaving only `manifest.json` so the download is
  reproducible. `run_all.sh` runs clean-up by default at the end and takes
  `--keep-data` to skip it. Earth Engine Drive exports, if any were needed,
  are listed in the run log with their Drive paths so they can be removed
  by hand (the API cannot delete Drive files with the granted scopes), and
  the README says so. Nothing under `experiments/external_audits/` may
  contain raw downloaded data; only derived CSVs small enough to commit
  (under 20 MB each; larger intermediates go to the external root and are
  deleted by clean-up).
- Each paper's README ends with a "Finding" paragraph in manuscript style
  stating the measured difference, its direction relative to the paper's
  transfer pairs, and whether it supports or weakens the paper's
  attribution, plus a "Limitations of this audit" paragraph.
- A final step adds a short "Measured" column or note to the three papers'
  rows in `reports/Missingness confound audit of transfer papers.md`.

## Left to the builder

- Exact zip range-fetch implementation for the TimeMatch shapefiles (the
  recon note documents the central directory approach).
- Whether to use `sampleRegions` or `reduceRegion` histograms for DACCN,
  depending on what returns within `getInfo` limits.
- Figure styling within the dataviz skill.
- How to handle Sentinel-2 dates where two granules of the same tile exist
  (take the union, document it).

## Success criteria

- A second person can run `run_all.sh` on a clean checkout with the two
  conda environments and Earth Engine credentials and obtain the same CSVs
  to within sampling noise, with no manual steps.
- After `run_all.sh` finishes, the external data root for each paper holds
  only `manifest.json`.

- Three READMEs with findings, three figures, and CSVs that another person
  could re-derive from the scripts without asking questions.
- Each finding states numbers for source and target (or 2018 and 2019) and
  says plainly whether the difference is large, small or absent.
- Total new external data on disk under 30 GB.
