# External missingness audits

Descriptive audits of the cloud and gap structure behind three published
crop-type transfer papers that the audit report
(`reports/Missingness confound audit of transfer papers.md`) ranked as highly
suspect of a missingness confound: TimeMatch (Nyborg et al. 2022), DENETHOR
(Kondmann et al. 2021) and DACCN (Wang et al. 2023, via the You et al. 2021
recipe it states it followed). Each audit measures, from the paper's own data
or deposited recipe, how cloud contamination, clear-observation counts or
zero-fill rates differ between the paper's source and target domains. No
model is re-run. Design: `docs/superpowers/specs/2026-10-28-external-missingness-audits-design.md`;
plan: `docs/superpowers/plans/2026-10-28-external-missingness-audits.md`.

Each paper folder has a README that ends with a Finding paragraph and a
Limitations paragraph. Those are the deliverables.

## Layout

```text
common/      paths, SHA-256 manifest downloads, run logs, EE init, zonal stats, plot theme, cleanup
timematch/   01 range-fetch footprints from the Zenodo zip, 02 Earth Engine cloud fractions per shipped date, 03 summarise
denethor/    01 fetch CLP/SCL/IS_DATA + labels, 02 per-field clear counts (2x2 tile-years), 03 Planet Fusion QA synthetic fraction, 04 summarise
daccn/       you2021_recipe (EE port) + reference (numpy model), 01 regions, 02 counter samples via batch export, 03 summarise
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
Earth Engine loops are resumable row by row and the DACCN sampler resumes from
`results/ee_tasks.json`. Fixed seed 42 throughout.

## Data root and clean-up

Downloads go to `/mnt/bigdrive/external_audits/<paper>/` (set in
`common/paths.py`), about 10 GB for DENETHOR masks, 12 GB of transient Planet
Fusion QA rasters (deleted day by day as they are read), and under 100 MB for
TimeMatch. `run_all.sh` ends by running `<paper>/99_cleanup.py`, which checks
the results are complete and then deletes everything under the paper's data
root except `manifest.json` (object key, URL, bytes, SHA-256, fetch time of
every file) so the download is reproducible. The DACCN sampler writes its
samples to Earth Engine assets under
`projects/mmann1123-malawi/assets/external_audits_daccn/` and deletes each
asset after reading it (`--keep-assets` to retain). No Drive exports are used.

## Replicability notes

Earth Engine collection ids, thresholds and the image ids used per date are
written to the run logs; a later run can diff them against the catalogue.
Source Cooperative object keys and byte sizes come from the saved listing
`research_notes/Missingness confound audit of transfer papers/recon_fusion_competition_s3_listing.tsv`.
`pip_freeze_ee.txt` pins every package version used for the committed results.
