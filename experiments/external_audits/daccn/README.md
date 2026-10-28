# DACCN audit: how often does the You et al. 2021 recipe zero-fill, and does it differ between DACCN's source and target domains?

Wang, Feng, Zhang and Tian 2023 (ISPRS JPRS 199, [10.1016/j.isprsjprs.2023.04.002](https://doi.org/10.1016/j.isprsjprs.2023.04.002)) transfer a crop classifier between Minnesota, Nebraska, Wisconsin, Iowa, Heilongjiang and Jilin and between 2018 and 2019, and attribute the unadapted losses to meteorology-driven phenological shift. They released no code but state that preprocessing followed You et al. 2021 (Scientific Data 8:41, [10.1038/s41597-021-00827-9](https://doi.org/10.1038/s41597-021-00827-9)), whose Earth Engine script is deposited on figshare ([13090442](https://figshare.com/articles/figure/The_10-m_crop_type_maps_in_Northeast_China_during_2017-2019/13090442), CC BY 4.0; copy in `you2021_GEE_code_for_crop_classification.js`). The script's "linear interpolation" of empty 10-day composites searches at most nine composites on each side and then calls `unmask(0)`, so any pixel without a valid composite on one side within 90 days is set to zero in every feature. Neither paper counts how often that happens. This audit re-runs the compositing and interpolation stage with two counters added.

## What was measured

`you2021_recipe.py` ports the script to the Earth Engine Python API: `COPERNICUS/S2_HARMONIZED` L1C scaled to reflectance, the Hancher, Hewig and Housman simple cloud score with threshold 0.2 and no shadow test, 37 ten-day median composites from DOY 0, interpolation of list indices 9 to 30 (22 steps, DOY 90 to 300) with the nine-composite search, and the `unmask(0)` semantics. For every sampled pixel it records `valid_k` (the step's own median was unmasked) and `zero_k` (the step ended as zero: invalid itself and no valid composite within nine steps before or after), plus `n_valid` and `n_zero`. `reference.py` is a numpy model of the same rule, unit-tested against hand-derived cases.

Two variants bracket what DACCN did. Variant A is the recipe as deposited: imagery for the whole calendar year, so the April and October edge steps can borrow from winter composites. Variant B feeds only April 1 to October 31 imagery, which is what DACCN says it used, so nothing outside the window exists to interpolate from. Variant B is a counterfactual, not a literal run of the deposited script: when a before or after window contains no composite with imagery, the script's `mosaic().select(bands)` is applied to an image with no bands and Earth Engine raises an error (verified: `Image.select: Band pattern 'red' was applied to an Image with no bands`), so the deposited script cannot run on April to October imagery alone. The port substitutes a fully masked image for an empty window, which lets the `unmask(0)` branch fire as it would for any masked neighbour. DACCN therefore either pulled imagery beyond its stated window (in which case Variant A describes its inputs and the zero-fill rate is small) or altered the edge handling (in which case Variant B describes the zeros its inputs carried, provided the `unmask(0)` fallback survived the alteration).

Sampling frame: in the four US states, CDL pixels of corn (1), soybeans (5) and a pooled "other" class (spring wheat, winter wheat, alfalfa, oats, dry beans, sugarbeets) with CDL confidence above 90, mirroring DACCN's own labelling; in the two Chinese provinces, ESA WorldCover 2020 cropland (class 40), because the You et al. maps are not in the Earth Engine catalogue. Design: 16 random 5 km cells per region (seed 42, the same cells for both years and both variants), 100 stratified points per class per cell at 10 m, computed interactively per cell. A whole-state request exceeds Earth Engine's memory limit and a batch export of 300 state-wide points ran for 40 minutes before it was cancelled.

## How to run

```bash
conda activate ee
python experiments/external_audits/daccn/01_regions.py       # GAUL lookups and frame labels (sanity check)
python experiments/external_audits/daccn/02_run_counters.py  # 24 region-year-variant jobs, ~25 min, resumable; results/daccn_counters.csv
python experiments/external_audits/daccn/03_summarise.py     # results/daccn_summary.csv, daccn_pairs.csv, figures/daccn_zero_by_step.png
```

Deviations from the deposited script, all deliberate: one carrier band (`red`) instead of the ten indices, since zero fill applies to every band identically; no Savitzky-Golay step (the module `users/nanshany1993/common:sentinel2` is private, and smoothing only spreads zeros across seven steps); the two counters; and a fully masked placeholder for an empty before or after window, where the deposited script would raise (see above). DACCN reports 21 steps and a quadratic smoother against the script's 22 and cubic, so the port measures the recipe as deposited and the edge behaviour is an inference about how closely DACCN reproduced it.

## Results

Share of sampled cropland pixels with at least one zero-filled step (`share_any_zero`), mean number of zero steps, and the share zero at the first and last step, generated from `results/daccn_summary.csv`. Cells is the number of the 16 cells that returned points (variant A jobs lost some cells to the Earth Engine memory limit; `results/run_log_02_run_counters.json` lists the 40 failed cells and every cell's coordinates).

| Region | Year | Variant A: any zero | A: mean zeros | A: cells | Variant B: any zero | B: mean zeros | B: zero at step 1 | B: zero at step 22 | B: cells | B: cell SD |
|---|---|---|---|---|---|---|---|---|---|---|
| Minnesota | 2018 | 0.041 | 0.05 | 9 | 0.906 | 1.52 | 0.49 | 0.74 | 11 | 0.23 |
| Minnesota | 2019 | 0.031 | 0.03 | 6 | 0.887 | 1.42 | 0.48 | 0.75 | 10 | 0.21 |
| Nebraska | 2018 | 0.000 | 0.00 | 9 | 0.592 | 1.47 | 0.59 | 0.43 | 12 | 0.46 |
| Nebraska | 2019 | 0.000 | 0.00 | 7 | 0.893 | 0.89 | 0.00 | 0.89 | 10 | 0.37 |
| Wisconsin | 2018 | 0.008 | 0.01 | 12 | 0.662 | 0.85 | 0.44 | 0.38 | 14 | 0.35 |
| Wisconsin | 2019 | 0.017 | 0.02 | 12 | 0.687 | 1.47 | 0.27 | 0.61 | 14 | 0.42 |
| Iowa | 2018 | 0.000 | 0.00 | 14 | 0.257 | 0.27 | 0.03 | 0.23 | 16 | 0.37 |
| Iowa | 2019 | 0.006 | 0.01 | 12 | 0.687 | 1.63 | 0.46 | 0.48 | 16 | 0.38 |
| Heilongjiang | 2018 | 0.005 | 0.00 | 10 | 0.746 | 0.79 | 0.04 | 0.75 | 13 | 0.43 |
| Heilongjiang | 2019 | 0.009 | 0.01 | 10 | 0.381 | 0.51 | 0.16 | 0.35 | 13 | 0.48 |
| Jilin | 2018 | 0.001 | 0.00 | 14 | 0.956 | 0.99 | 0.03 | 0.96 | 16 | 0.12 |
| Jilin | 2019 | 0.003 | 0.01 | 14 | 0.157 | 0.33 | 0.07 | 0.14 | 16 | 0.33 |

Mean valid composites out of 22 lie between 16.7 and 20.0 in every region-year under both variants; the recipe interpolates three to five steps per pixel in all domains.

DACCN's transfer pairs under Variant B (share of pixels with any zero, source then target):

| Pair | 2018 | 2019 |
|---|---|---|
| Minnesota to Wisconsin | 0.91 to 0.66 | 0.89 to 0.69 |
| Nebraska to Iowa | 0.59 to 0.26 | 0.89 to 0.69 |
| Minnesota to Heilongjiang | 0.91 to 0.75 | 0.89 to 0.38 |
| Nebraska to Jilin | 0.59 to 0.96 | 0.89 to 0.16 |
| Each region 2018 to 2019 | Minnesota 0.91 to 0.89, Nebraska 0.59 to 0.89, Wisconsin 0.66 to 0.69, Iowa 0.26 to 0.69, Heilongjiang 0.75 to 0.38, Jilin 0.96 to 0.16 | |

Under Variant A every pair differs by less than 0.04.

![Share of pixels zero-filled at each of the 22 steps, six regions, 2018 and 2019, both variants](figures/daccn_zero_by_step.png)

## Finding

The deposited recipe has two very different behaviours depending on what imagery it is given, and the paper's one-paragraph description does not say which it got. Fed the whole year (Variant A, the script as deposited), it almost never zero-fills: at most 4 percent of cropland pixels receive any zero and the mean is a few hundredths of a step, because the nine-composite search reaches into winter scenes on both sides of the April to October window. Restricted to April to October imagery, as DACCN states (Variant B, which requires an edge-handling change the deposited script lacks), the search has nothing to find beyond the window edges, and between 16 and 96 percent of pixels receive at least one zero-filled step, almost all at step 1 (first ten days of April) and step 22 (late October), with 0.3 to 1.6 zero steps per pixel on average. Those zeros are not random noise: they are a constant 0 in all twelve features at the start or end of the series, which the seven-step Savitzky-Golay smoother then spreads inward.

The rate differs sharply across exactly the pairs DACCN transfers. Nebraska to Jilin goes from 59 percent of source pixels zero-filled to 96 percent of target pixels in 2018 and from 89 to 16 percent in 2019; Minnesota to Heilongjiang from 91 to 38 percent in 2019; Nebraska to Iowa from 59 to 26 percent in 2018; and the temporal cases swing by up to 80 points (Jilin 96 to 16, Iowa 26 to 69, Heilongjiang 75 to 38). A classifier that learned, say, Jilin 2018 with a zero in nearly every pixel's final step and is then applied to Jilin 2019 where only one pixel in six has it, meets a feature distribution shift that has nothing to do with crop calendars, and the maximum mean discrepancy that DACCN minimises will absorb it just as readily as a real phenological shift. The audit cannot apportion DACCN's losses between the two, and it cannot tell which variant DACCN ran. What it shows is conditional: if DACCN used only the April to October imagery it describes, it must have changed the script's edge handling, and the zero-fill rate then differs between source and target by tens of percentage points in most pairs, in no consistent direction, uncounted by the recipe and unmentioned by the paper; if DACCN pulled year-round imagery, the deposited script applies and the zero-fill confound is small (under 4 percent of pixels). Either way the paper does not report which, and the difference between the two readings is the difference between a negligible confound and a large one.

## Limitations of this audit

The ranks of the six regions are clustered estimates from 16 random 5 km cells (fewer where cells exceeded the Earth Engine memory limit in Variant A, listed in the run log), so the per-region shares carry between-cell spread of 0.1 to 0.5; the pair differences reported above are far larger than that spread for the extreme pairs and of the same order for the mild ones. The Chinese frame is WorldCover cropland rather than DACCN's crop-map labels. The port stops before smoothing and uses one carrier band; the counters would be identical for any band. The 22-step window and the edge behaviour follow the deposited script, which DACCN describes as 21 steps. Nothing here re-runs DACCN.
