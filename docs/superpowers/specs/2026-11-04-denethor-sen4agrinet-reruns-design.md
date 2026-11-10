# DENETHOR and Sen4AgriNet causal re-runs: design

Date: 2026-11-04. Status: the user asked for the re-runs and opted out of design questions; every choice below is Claude's call unless marked otherwise.

## Why

The audit report (`reports/Missingness confound audit of transfer papers.md`) keeps five papers at rank 3, all "measured but unproven": each shows a source-target missingness difference the paper never reports, none shows that difference caused the transfer loss. The only causal test so far, `experiments/external_audits/timematch_rerun/`, came back mostly negative. DENETHOR and Sen4AgriNet are the two rank-3 cases with public data and code, so they are the two where the measured difference can be turned into a causal estimate. The deliverable is the same shape as `timematch_rerun/`: a folder per paper with numbered resumable scripts, committed CSV results with `.meta.json` sidecars, one figure, and a README ending in a Finding and a Limitations paragraph; then the report cards and the summary table are updated with the result, whichever way it falls.

## Goals and non-goals

Goal: for each paper, estimate how much of the published transfer loss the measured missingness difference accounts for, using the authors' own models (Sen4AgriNet) or the authors' own training recipe (DENETHOR), with a matched-random control so that "fewer or filled observations" is separated from "these particular observations".

Non-goals: improving either model; re-running every architecture or scenario; the OAD half of Sen4AgriNet (its released datamodule trains on the test file, already reported, and needs no re-run to stand).

Anti-goal: an experiment that can only confirm. Each condition below has a stated outcome that would count against the missingness reading, and the README reports that outcome if it occurs, as the TimeMatch re-run did.

## Sen4AgriNet (`experiments/external_audits/sen4agrinet_rerun/`)

Models: the released PAD checkpoints in the authors' Dropbox bundle (`model_weights/PAD/{UNet,ConvLSTM,ConvSTAR}/Scenario {1,2,3}/*.ckpt`), extracted with the existing forward-only `sen4agrinet/zipstream.py`. Code: `Orion-AI-Lab/S4A-Models` at `ae5a376` (clone under `/mnt/bigdrive/external_audits/sen4agrinet_rerun/repo`). The model classes subclass `pytorch_lightning.LightningModule`; the harness puts a stub `pytorch_lightning` package (LightningModule = `torch.nn.Module` plus the few attributes the classes touch) on `sys.path` and loads `state_dict` from the checkpoint, so no package is installed into `deep_field` (my call). Inputs reproduce the published recipe: bands B02 B03 B04 B08, monthly medians April to September (`--fixed_window`), `--requires_norm` (divide by 10000), 61 by 61 subpatches of the 366 by 366 patch with the authors' padding, the authors' `get_medians` (calendar-month median, resample, `interpolate_na(linear, extrapolate)`) called on the patch NetCDF; test patches from the Hugging Face dataset at the pinned revision, only the variables needed, cached under `/mnt/bigdrive/external_audits/sen4agrinet_rerun/netcdf` (about 17 GB per scenario at most).

Gate (step 1): with no intervention, weighted F1 per checkpoint must reproduce the README PAD table (Scenario 2: U-Net 57.85, ConvLSTM 56.56, ConvSTAR 52.15; Scenario 3: 43.54, 40.47, 34.45) to within 1 point using the authors' metric code path (`test_step` / `test_epoch_end` arithmetic, parcel pixels only, class-sample weighting). If it does not, stop and report why before running conditions.

Bound: the Scenario 1 (random split) checkpoints give the in-domain weighted F1 (80.4 to 85.2); the gap is Scenario 1 minus Scenario 2 or 3 for the same architecture. This is the analogue of TimeMatch's target-trained bound and is imperfect (different test set); the README says so.

Conditions, all test-time, frozen models, both Scenarios 2 and 3, all three architectures:

- (a) as published.
- (b) fill imposition by month: on test patches where month m (April to September) is observed, delete m's acquisitions and refill with the authors' rule; one run per month. Gives the cost of a fill in each month, Δ_m. Against the reading: Δ_m near zero for April and June, the months that differ between domains.
- (c) source-pattern imposition: on test patches with no filled month, impose the empty-month pattern of a randomly drawn training-domain patch (10 draws, seed 42), against (c-ctrl) the same number of months emptied at random. Then the reverse comparison, imposing a target-domain pattern on those patches. The difference between imposing source and target patterns estimates how much of the gap the fill-pattern difference carries on otherwise identical inputs. Against the reading: source- and target-pattern imposition give the same F1, or no different from random months.
- (d) pixel cloud mask at test: the authors' own `hollstein_mask` (the `--clouds` path they ship but did not use) sets cloudy pixels to NaN before the median, so cloudy months fall back to interpolation. Against the reading: no change, or a drop.
- Predicted share: Σ_m (target fill rate_m minus source fill rate_m) × Δ_m from (b), with the fill rates already in `sen4agrinet/results/month_missing_by_split.csv`, divided by the Scenario 1 gap. Reported next to the direct (c) estimate.

## DENETHOR (`experiments/external_audits/denethor_rerun/`)

No weights are released, so models are retrained with the authors' recipe (`lukaskondmann/DENETHOR` at `55573a8`, `train_psetae.py`). Two arms.

### Arm A: PseLTae on Planet Fusion daily (the paper's headline 67.25 percent)

Data: PF-SR and PF-QA daily COGs for 18E-242N 2018 (train) and 17E-243N 2019 (test) from Source Cooperative, about 125 GB per tile-year, streamed one day at a time and deleted after extraction (peak disk about 1 GB). For each field above the authors' 1000 m² threshold, a fixed random set of 128 pixel positions inside the polygon (seed 0, rasterised `all_touched=True` as in `dataset.py`) is stored for all 365 days and 4 bands, plus PF-QA layer 1 at the same pixels (my call: storing all pixels would be about 80 GB; the authors resample 20 pixels per field per epoch, which the training loop still does, from these 128). Model, normalisation, 20-pixel draw, 25 percent validation split with `RandomState(0)`, optimiser and learning rate as in `train_psetae.py`; batch size 64 and 100 epochs as stated in the paper (the script defaults of 4 and 150 contradict the text; the paper's values are used and the choice reported); best epoch by validation loss as in the script. Three seeds per condition.

Gate: mean test accuracy within 3 points of 67.25 and validation accuracy within 3 points of 88.02. Retraining noise makes an exact match impossible; if the gate fails the conditions are still run but the README leads with the failure.

Conditions (trained and tested the same way):

- A0 as published (4 bands, all 365 days as consumed).
- A1 QA-aware: PF-QA percent synthetic added as a fifth input channel, so the model can tell observed from synthesised days.
- A2 control: the fifth channel holds the same field's QA values permuted across days, which keeps the channel's marginal distribution and the parameter count but destroys the day-to-QA link.
- Per-field stratification on A0: test accuracy by tercile of the field's May synthetic share and of its 2018-minus-2019 difference in synthetic share (descriptive only).

Against the reading: A1 does no better than A2 on the 2019 test tile. This arm cannot split spatial from temporal, because the public bucket has no PF for the test tile in 2018 (the paper's 78.77 percent ablation used data not released); the README says so.

### Arm B: the authors' Sentinel-2 random forest (`random_forest.py`, Table 3, accuracy 0.59, macro F1 0.42)

Data: S2 BANDS.npy for 18E-242N 2018 and 17E-243N 2018 and 2019 (about 22 GB each, downloaded one at a time, field pixels extracted, deleted), with CLP from the existing `denethor/` audit fetch. Features exactly as `random_forest.py` (field mean per acquisition, clear acquisitions only by `cld < 40` as coded, 7 statistics per band plus NDVI). This arm gives the 2 by 2 the PF arm cannot: train on the train tile in 2018, test on the test tile with 2018 imagery and with 2019 imagery (labels of the test tile reused for both years, as the paper does), so the temporal component is measured.

Conditions:

- B0 as published, both test years; gate: 2019 accuracy within 3 points of 0.59.
- B1 density matched: thin each test-tile 2018 field's clear acquisitions, month by month, to the count the same field has in 2019 (random choice, 10 draws), then evaluate. If the temporal component is density, B1 falls most of the way from the 2018 to the 2019 score.
- B1-ctrl: thin the same total number of acquisitions per field, but at random across months, so month placement is separated from count.
- B2 rule strictness: the same 2 by 2 under the s2cloudless 40 percent rule (CLP < 102) used elsewhere in the audit, since the literal rule's year effect is larger (40 to 31 against 42 to 35).

Against the reading: B1 leaves 2018 accuracy near its unthinned value, so fewer clear observations do not reproduce the 2019 drop.

## Constraints

GPU work in `deep_field` (RTX 3080 Ti, 12 GB), one GPU process at a time (the TimeMatch run froze the machine with six). Earth-Engine-free; all inputs are public HTTP. Downloads under `/mnt/bigdrive/external_audits/<folder>/` (327 GB free at start; peak use is under 40 GB because PF and S2 are streamed and deleted). Seeds fixed (42 for sampling, 0 where the authors fix 0, three training seeds 0 to 2). Every script resumable and `--force`-able, pure functions unit-tested in `experiments/external_audits/tests/`, results committed, data not. Neither folder joins `run_all.sh` (GPU and large downloads), as with `timematch_rerun/`.

## Left to the builder

Exact stub surface for the Lightning shim; batching and caching layout; how the Hugging Face NetCDFs are range-read; the figure design (one per folder, following `common/` plot theme).

## After the runs

Update `reports/Missingness confound audit of transfer papers.md` (the two Table rows, the per-paper cards, the opening paragraph and the conclusion), `experiments/external_audits/README.md`, and the revision log only if the manuscript text changes.
