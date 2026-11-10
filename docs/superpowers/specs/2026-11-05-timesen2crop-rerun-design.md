# TimeSen2Crop re-run: does hard zero fill break transfer when gentler fills do not?

Written before any model was trained. Only the empty-month table and acquisition counts had been seen.

## Why

Causal re-runs of TimeMatch, DENETHOR and Sen4AgriNet found missingness real but cheap. All three involve gentle handling (unmasked cloud, vendor reconstruction, linear interpolation). Our own AI4FoodSecurity case, where zero-filling a month costs dense pixel networks 9 to 15 macro-F1 points, involves hard zero fill. The hypothesis that zero fill specifically is what hurts was formed after seeing those results, so it needs a test whose pass mark is set in advance. TimeSen2Crop is the only public benchmark whose official recipe zero-fills whole months and whose labels cover a second year on the same tile.

This does not audit a claim by the paper, which reports no cross-year score. Its rank stays 2 regardless. The target is the hypothesis.

## Data

Zenodo 4715631 (md5 b5b7aad3...). Train 13 tiles 2017-18, validation 33TWN, test 33UVP 2017-18 (paper's split), and the labelled 2019_33UVP tile for the year transfer. Monthly clear medians, September first; the paper's rule zeroes empty pixel-months.

## Models

LSTM (paper OA 83.44) and TempCNN (paper OA 81.71), the dense month-indexed families that lost most in our case. The paper gives search ranges but not chosen values, so: random search over the paper's ranges (10 configurations per model, zero fill, 10 epochs), chosen on 33TWN accuracy as the paper does, then fixed for every condition. 30 epochs, best checkpoint on 33TWN, 5 seeds per trained condition. Weighted LSTM and the other paper models are out of scope.

Reproduction gate: mean OA on 33UVP 2018 under zero fill within 2 points of the paper for each model. A model that misses the gate is reported but not interpreted.

## Conditions (training fill to test fill)

Z to Z published. Z to M and Z to L frozen zero-trained model, test gaps filled with training-pool per-month band medians (M, the form of our own fix) or linear interpolation (L). M to M and L to L retrained on that fill. Zm to Zm zero fill plus a present-month mask channel. M to Z positive control: trained without ever seeing a zero, tested with zeros; this is the closest analogue of our case, where the month was never empty in training.

Test-time only, on frozen Z and M models: deleting each month in turn, and imposing on 33UVP 2018 pixels the empty-month pattern of random 2019 pixels against a matched random deletion of the same number of months (5 draws).

## Outcomes and pass marks, fixed now

Transfer loss: 33UVP 2018 minus 2019 under Z to Z, macro F1 and OA, with OA also reweighted to the 2018 class prior to remove label shift.

Primary contrast: on the 2019 tile, macro F1 of the best non-zero condition minus Z to Z, paired over seeds, per model, choosing the best on 33TWN, not on the test tile.

Assay check: M to Z must fall at least 10 macro-F1 points below M to M on the 2019 tile. If not, the assay cannot detect the effect and a null means nothing.

Verdict, applied to both models:
- Supported: lift at least 5 macro-F1 points with a 95 percent interval above 0, in both models, and the lift on the 2019 tile exceeds the lift on 33UVP 2018 (a fill that helps everywhere is a better input, not a transfer effect).
- Weak: lift between 1 and 5 points with an interval above 0, in at least one model.
- Null: otherwise.

## Limits known in advance

Training tiles contain zeros in every month, unlike our case, so the published model has already seen them; the positive control addresses this but is itself synthetic. Pixel samples have no field or location identifiers, so an in-domain bound cannot be built without leakage and the year gap is the only transfer loss measured. One tile pair, one year pair. Label sampling caps differ between years (15,001 and 20,001 per class).
