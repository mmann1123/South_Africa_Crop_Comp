# Crop phenology (reviewer R2-Q2)

`phenology_curves.py` builds field-level monthly profiles (mean of each field's pixels) for EVI, B11, B12 and hue over the 10 model months, for the training region and the holdout tile. It writes `figures/phenology_curves.{pdf,png}` (copied to `writeup/figures/crop_phenology.pdf`, SI Fig. S8), `results/phenology_medians.csv` and `results/separability_*.csv`.

Key numbers (2026-10-09):

- **Largest |Cohen's d| over bands and months (training region):** lucerne/medics pairs 1.26 to 2.27 (B11, September); canola pairs 1.06 to 1.44 (EVI); cereal pairs barley–small-grain grazing 0.73, wheat–small-grain grazing 0.65, wheat–barley 0.50.
- **Peak median EVI:** lucerne/medics 0.32, annual crops 0.45 to 0.51.
- **Holdout minus training, July EVI:** canola −0.21, barley −0.13, wheat −0.13, small-grain grazing −0.11, lucerne/medics −0.03.
- **Holdout minus training, October EVI:** +0.02 to +0.08. The holdout season is later.
