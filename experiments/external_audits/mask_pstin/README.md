# Mask-PSTIN audit: how often are the released composites zero-filled?

Zhang, Cai et al. 2024 (ISPRS JPRS 218:87-101, [10.1016/j.isprsjprs.2024.10.013](https://doi.org/10.1016/j.isprsjprs.2024.10.013); code and the region A data at [ZhiwenCai-RS/Mask-PSTIN](https://github.com/ZhiwenCai-RS/Mask-PSTIN), commit 4a03467, formerly `BruceKai/Mask-PSTIN`; supplement `1-s2.0-S0924271624003897-mmc1.docx` on ars.els-cdn.com) build 20-day Sentinel-2 L2A composites after removing pixels with Cloud Score+ below 0.65. The paper never says what an empty composite receives. The released loader (`dataset/mydataset.py`) sets `feat[np.isnan(feat)] = 0` at training and test time, and its temporal random masking augmentation writes the same value, `feat[mask] = 0`, for whole time steps. The only transfer experiments are in Supplement Part VI, run in region A: train on 2019 and 2020 and test on 2021 (temporal), and train on Allier and test on Puy-de-Dôme (spatial). Only the 2021 data are released.

## What was measured

`zero_composites.py` reads the released `data/{train,val,test}/data.npy` (uint16, 19 composites by 5 bands by 5 by 5 pixels) and counts composites whose five bands are all zero at the centre pixel. Because the arrays are unsigned integers they cannot hold NaN, so an empty composite appears as an all-zero one. The splits are exactly the paper's region A 2021 sample: 3,316 train, 1,919 validation and 2,168 test points, with the test class counts of Table 1.

```bash
git clone https://github.com/ZhiwenCai-RS/Mask-PSTIN /path/to/Mask-PSTIN
conda activate ee
python -I experiments/external_audits/mask_pstin/zero_composites.py /path/to/Mask-PSTIN
```

## Results

All three splits match to within a point (`results/split_summary.csv`). On the test split, 9.9 percent of centre-pixel composites are all zero (9.5 percent over the whole 5 by 5 patch), 87.6 percent of points have at least one zero composite, and a point has 1.9 on average and at most 7 of 19. The zeros are seasonal (`results/by_step.csv`): 44 percent of points are empty for 31 December to 19 January and 63 percent for 20 January to 8 February, with smaller peaks of 12 to 17 percent in early March, early May and early July. The last bin, 27 October to 1 November, covers five days and is 31 percent empty. Classes differ little: from 9.1 percent for sunflower to 11.3 percent for rye (`results/by_class.csv`).

## Finding

Empty composites are zero-filled, at about one composite in ten, mostly in the midwinter weeks when the winter cereals (wheat, rye and barley) are in the ground. That is the rubric's mechanism. In 2021 it is the same in training and test because the split is random within one year and one region, so it cannot affect the main results. Supplement Part VI reports a temporal transfer from 2019 and 2020 to 2021 in which wheat, rye and barley show "notable performance degradation", and attributes this to "year-to-year phenological shifts" without reporting zero-fill rates for 2019 or 2020. Those rates are not released, so the year-to-year difference is unmeasured. Two things weaken the case for a confound. Mask-PSTIN is trained by writing zeros over whole time steps (threshold 0.40 in region A, applied to a random half of training samples, so 20 percent of training steps on average), about twice the natural rate, so zero composites are in distribution for the transferred model. And the spatial transfer, to Puy-de-Dôme, the more mountainous department, shows no loss. Under the rubric this is rank 2: gap handling is applied identically to both domains, but the paper never reports how often it fired in each year.

## Limitations of this audit

Only the 2021 region A data are released, so the 2019 and 2020 rates that would test the temporal-transfer attribution would have to be rebuilt in Earth Engine from the paper's recipe (Cloud Score+ 0.65, 20-day composites from 1 November). The compositing reducer is not stated, and bin boundaries are inferred from the stated start date and 20-day period. Regions B and C (Inner Mongolia and Hubei) are not released. The RF baseline's gap handling is not in the code.
