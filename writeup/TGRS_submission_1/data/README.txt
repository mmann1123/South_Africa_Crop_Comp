================================================================================
SOUTH AFRICA CROP CLASSIFICATION — PROCESSED DATASET
Companion data for the IEEE TGRS submission
================================================================================

Paper : Mann, M. L., Venkatachalam, S., Kacha, D., Sheth, D., Engstrom, R.,
        and Jafari, A. "Sparse Feature Selection, Not Deep Learning Capacity,
        Drives Spatial Transfer in Crop Classification." IEEE Transactions on
        Geoscience and Remote Sensing (under review).
Code  : https://github.com/mmann1123/South_Africa_Crop_Comp
Author: Michael L. Mann, The George Washington University (mmann1123@email.gwu.edu)

--------------------------------------------------------------------------------
1. WHAT THIS IS
--------------------------------------------------------------------------------
This package contains the model-ready, processed datasets used to train and
evaluate every model in the paper. Together with the public code above, it
reproduces all reported results end to end.

The study classifies five winter crops in the Western Cape, South Africa from
multi-temporal Sentinel-2 optical imagery, and evaluates each model twice:
  (a) IN-REGION  — field-wise cross-validation within two adjacent training tiles
                   (34S_19E_258N, 34S_19E_259N; 4,150 labeled fields), and
  (b) SPATIAL TRANSFER — on a disjoint holdout tile (34S_20E_259N; 2,417 fields)
                   that no training pixel touches.

Crops (crop_id : crop_name):
  1 : Wheat        2 : Barley        3 : Canola
  4 : Lucerne/Medics                 5 : Small grain grazing

Sensor / period: Sentinel-2 L2A surface reflectance, Jan–Dec 2017, cloud-masked
(s2cloudless + a shadow mask) and composited monthly. Six features per month —
bands B2, B6, B11, B12 plus EVI and Hue. June 2017 had no usable imagery and May
was discarded for cloud, leaving 10 usable months (see DATA_DICTIONARY.txt for how
May appears in the raw files but is dropped in modeling). All labels are at the
field level; pixel/patch predictions are aggregated to fields.

--------------------------------------------------------------------------------
2. CONTENTS (three model families, train + holdout for each)
--------------------------------------------------------------------------------
Raw monthly Sentinel-2 time series (deep pixel models):
  merged_dl_train.parquet          pixel-level, TRAIN  (with labels)
  merged_dl_test.parquet           pixel-level, HOLDOUT (labels via test_fields.geojson)

Automated xr_fresh time-series features (classical models):
  final_data.parquet               pixel-level xr_fresh, TRAIN (with labels)
  combined_test_features.parquet   field-level  xr_fresh, HOLDOUT (with labels)

Patch tensors (patch 2D/3D CNNs):
  patch_level_data.parquet         patch pixels, TRAIN
  test_patch_data.parquet          patch pixels, HOLDOUT

Field / patch geometries and ground-truth labels:
  combined_training_fields.geojson  4,150 training field polygons + labels
  test_fields.geojson               2,417 holdout field polygons + labels
                                    (**the holdout ground truth for OOS scoring**)
  patch_level.geojson               training patch tiling geometry
  test_patches.geojson              holdout patch tiling geometry

Total size ~7.6 GB. Per-file sizes, row/column counts, and checksums are in
FILE_MANIFEST.txt; column-level schemas in DATA_DICTIONARY.txt.

--------------------------------------------------------------------------------
3. QUICK START
--------------------------------------------------------------------------------
1. Clone the code repo and place these files in its  data/  directory (the paths
   in deep_learn/src/config.py expect exactly these names).
2. Set up the two conda environments (deep_field, ml_field) per the repo README.
3. Run the numbered orchestrators: 1_train_all_models.py -> 4_run_inference.py
   -> 5_compare_and_submit.py. Holdout scoring reads test_fields.geojson.
See REPRODUCIBILITY.txt for the file-to-script map and exact commands.

--------------------------------------------------------------------------------
4. PROVENANCE & LICENSE (summary — full detail in PROVENANCE_AND_LICENSE.txt)
--------------------------------------------------------------------------------
Ground-truth labels derive from Radiant Earth MLHub / the AI4FoodSecurity (South
Africa) Track-1 challenge; imagery was extracted from Copernicus Sentinel-2 via
Google Earth Engine; features were computed with the open-source xr_fresh package.
These files are DERIVED products; all upstream sources permit redistribution with
attribution (labels are CC-BY-4.0, DOI 10.34911/rdnt.j0co8q). Please cite all
upstream sources (see CITATION.txt); full terms are in PROVENANCE_AND_LICENSE.txt.
