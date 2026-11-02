# External missingness audits

Descriptive audits of the cloud and gap structure behind six published
crop-type transfer papers that the audit report
(`reports/Missingness confound audit of transfer papers.md`) ranked as suspect
of a missingness confound: TimeMatch (Nyborg et al. 2022), DENETHOR
(Kondmann et al. 2021), DACCN (Wang et al. 2023, via the You et al. 2021
recipe it states it followed), Kerner et al. 2020, Sen4AgriNet (Sykas et al.
2022) and Stoian et al. 2019. Each audit measures, from the paper's own data
or deposited recipe, how cloud contamination, clear-observation counts or
zero-fill rates differ between the paper's source and target domains. No
model is re-run, except in `timematch_rerun/`, which evaluates TimeMatch's
released models with cloud-flagged target dates removed (GPU and the full
74 GB archive; not part of `run_all.sh`). Design: `docs/superpowers/specs/2026-10-28-external-missingness-audits-design.md`;
plan: `docs/superpowers/plans/2026-10-28-external-missingness-audits.md`.

Each paper folder has a README that ends with a Finding paragraph and a
Limitations paragraph. Those are the deliverables.

## Layout

```text
common/      paths, SHA-256 manifest downloads, run logs, EE init, zonal stats, plot theme, cleanup
timematch_rerun/ 01 parcel tables, 02 EE per-parcel Cloud Score+, 03 reproduce Table 1, 04 band features, 05 DK1 cloud emulator, 06 conditions (one GPU process at a time), 07 summarise, 08 figure
timematch/   01 range-fetch footprints from the Zenodo zip, 02 Earth Engine cloud fractions per shipped date, 03 summarise
denethor/    01 fetch CLP/SCL/IS_DATA + labels, 02 per-field clear counts (2x2 tile-years), 03 Planet Fusion QA synthetic fraction, 04 summarise
daccn/       you2021_recipe (EE port) + reference (numpy model), 01 regions, 02 counter samples per random 5 km cell, 03 summarise, snow check
kerner/      01 Zenodo phenology/CDL members, 02 HLS acquisitions, 03 pixel sample, 04 EE per-acquisition Fmask states, 05 summarise
sen4agrinet/ 01 pinned code facts, 02 split files streamed from the Dropbox bundle, 03-06 patch time stacks, fill rates, OAD CSV cross-check, 07 STAC cloud cover, 08 summarise
stoian/      01 digitised Figure 19 kappas, 02 Table 1 class areas, 03 Planetary Computer SCL clear counts, 04 L1C effective clear, 05 correlations
gallo/       01 Kaggle deposit file listing, 02 dates per area-year, 03 patch footprints, 04 EE Cloud Score+ per date, 05 raw-raster cloud check, 06 summarise
wijesingha/ 01 saved notebook outputs, 02 cropland points, 03 EE Landsat valid counts per bimonth, 04 summarise
verhulst/   01 digitised Figure 7 OA, 02 forest points, 03 EE valid dekads, 04 gap summary, 05 OA-loss correlations
maleki/     01 Ramsar site geometry and inventory, 02 EE per-date validity sample, 03 6-day grid interpolation metrics, 04 summarise
hao/        01 digitised in-season OA figures, 02 CDL and cropland points, 03 EE 15-day window validity (HLS and L8+L1C streams), 04 summarise, 05 relate to OA gaps, 06 figure
hexi/       01 CDL and Hexi cropland points, 02 EE weekly validity under four cloud rules, 03 QA60 population per image, 04 summarise, 05 figure
masolele/    01 Hansen 2001-05 loss pixels at tropical confluence units, 02 EE valid Landsat 5/7 obs per six-month window, 03 summarise and figure
tests/       pytest unit tests for every pure function
env/         conda env exports and pip freeze of the `ee` environment used
run_all.sh   everything in order, then clean-up
```

Results (`<paper>/results/*.csv` with `.meta.json` sidecars naming the script,
git commit and inputs, plus `run_log_<script>.json` per script) and figures
(`<paper>/figures/`, PNG at 300 dpi and PDF) are committed. Downloaded data
are not.

## How to run

```bash
conda env create -f experiments/external_audits/env/environment.ee.yml   # once; or use an existing env with the pip_freeze_ee.txt versions
conda activate ee
earthengine authenticate --auth_mode=notebook                              # once; store a Cloud project when asked
bash experiments/external_audits/run_all.sh                                # add --keep-data to keep downloads
```

Individual steps are the numbered scripts in each folder, run from the repo
root; each skips work whose output exists and takes `--force` to redo it. The
Earth Engine loops are resumable row by row (TimeMatch by tile and date, DACCN by
region, year and variant from `results/daccn_counters.csv`), and a run that finds
nothing to do leaves the earlier run log untouched. Fixed seed 42 throughout.

## Data root and clean-up

Downloads go to `/mnt/bigdrive/external_audits/<paper>/` (set in
`common/paths.py`), about 10 GB for DENETHOR masks, 12 GB of transient Planet
Fusion QA rasters (deleted day by day as they are read), about 5 GB for
Sen4AgriNet (one streamed pass over the authors' Dropbox bundle plus patch
time stacks), about 80 MB for Kerner, about 200 MB for Gallo (mostly the 90 MB Kaggle file
listing, which takes about four hours to re-crawl), and under 100 MB each for
TimeMatch, Stoian, Wijesingha, Verhulst, Maleki, Hao and Hexi. `run_all.sh` ends by running `<paper>/99_cleanup.py`, which checks
the results are complete and then deletes everything under the paper's data
root except `manifest.json` (object key, URL, bytes, SHA-256, fetch time of
every file) so the download is reproducible. The DACCN sampler computes each cell interactively
(`getInfo`) and writes nothing to Earth Engine assets or Drive.

## Replicability notes

Earth Engine collection ids and thresholds are written to the run logs, the
Sentinel-2 image ids used per date to `timematch/results/cloud_by_date.csv`, and
the DACCN cell coordinates to `daccn/results/run_log_02_run_counters.json`; a
later run can diff them against the catalogue.
Source Cooperative object keys and byte sizes come from the saved listing
`research_notes/Missingness confound audit of transfer papers/recon_fusion_competition_s3_listing.tsv`.
`pip_freeze_ee.txt` pins every package version used for the committed results.
