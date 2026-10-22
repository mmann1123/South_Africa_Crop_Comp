"""Per-model fill sheet: how each published pipeline handled the missing holdout month.

One row per model in the 27-model table: input representation, the operator applied to the
missing month in the published inference, where any fill statistic came from (training data,
holdout data, or none), the order of fill, aggregation and scaling, how May (missing in every
training and holdout pixel) entered the model, how the model was treated in the re-analysis, and
whether the re-analysis reproduced its published predictions. Modeled on the model info sheet of
Kapoor & Narayanan (2023).

The operator descriptions are from the inference code (paths in the "code" column); the
reproduction counts are read from experiments/holdout_fill_check/results/fill_comparison.csv.
Writes experiments/missingness/results/fill_sheet.csv and fill_sheet.md.
Run: python experiments/missingness/fill_sheet.py
"""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
FC = os.path.join(REPO, "experiments", "holdout_fill_check", "results", "fill_comparison.csv")

RAW_ZERO = dict(
    inputs="raw band x month sequence",
    operator="0 (raw reflectance)", statistic="none (constant)",
    order="fill, then standardize with the training scaler", may="column dropped",
)
RAW_ZERO_FIELD = dict(RAW_ZERO, order="fill, average pixels per field, then standardize")
XR = dict(
    inputs="174 xr_fresh statistics of each band's series",
    operator="none: statistics computed on the series without the missing month (NaN-aware)",
    statistic="none; remaining feature NaNs by a training-fitted imputer",
    order="features per pixel, then field mean, then imputer and scaler", may="skipped (NaN)",
)
XR_PIXEL = dict(XR, order="features per pixel, field mean, then 0 for feature NaNs and scaler "
                          "(pixel-trained model applied to field-mean features)",
                statistic="none; remaining feature NaNs set to 0")

# display name -> (fill_comparison model name, attributes, code)
MODELS = {
    "TabNet (raw), pixel": ("TabNet (raw), pixel", dict(
        inputs="raw band x month values as tabular features",
        operator="holdout column median, then 0", statistic="holdout (target) data",
        order="fill, then standardize with the training scaler",
        may="included as a feature; filled with 0 (median undefined)"), "out_of_sample/inference_tabnet.py"),
    "L-TAE-S, pixel": ("L-TAE-S, pixel", RAW_ZERO, "experiments/field_reduction/predict_oos.py"),
    "L-TAE-S, field": ("L-TAE-S, field", RAW_ZERO_FIELD, "experiments/field_reduction/predict_oos.py"),
    "L-TAE, pixel": ("L-TAE, pixel", RAW_ZERO, "out_of_sample/inference_ltae.py"),
    "L-TAE, field": ("L-TAE, field", RAW_ZERO_FIELD, "out_of_sample/inference_ltae_field.py"),
    "Transformer, pixel": ("Transformer, pixel", RAW_ZERO, "out_of_sample/inference_transformer.py"),
    "TempCNN, pixel": ("TempCNN, pixel", RAW_ZERO, "out_of_sample/inference_tempcnn.py"),
    "TempCNN, field": ("TempCNN, field", RAW_ZERO_FIELD, "out_of_sample/inference_tempcnn_field.py"),
    "CNN-BiLSTM, pixel": ("CNN-BiLSTM, pixel", dict(RAW_ZERO, order="fill; no scaling (raw reflectance)"),
                          "out_of_sample/inference_cnn_bilstm.py"),
    "CNN-BiLSTM, field": ("CNN-BiLSTM, field", dict(RAW_ZERO, order="fill, average pixels per field; no scaling"),
                          "experiments/field_reduction/predict_oos.py"),
    "TimesNet, pixel": ("TimesNet, pixel", RAW_ZERO, "experiments/recent_baselines/common.py"),
    "TimesNet, field": ("TimesNet, field", RAW_ZERO_FIELD, "experiments/recent_baselines/common.py"),
    "MiniRocket, pixel": ("MiniRocket, pixel", RAW_ZERO, "experiments/recent_baselines/common.py"),
    "MiniRocket, field": ("MiniRocket, field", RAW_ZERO_FIELD, "experiments/recent_baselines/common.py"),
    "TabNet (raw, field-avg), field": ("TabNet (raw, field-avg), field", RAW_ZERO_FIELD,
                                       "out_of_sample/inference_tabnet_temporal_field.py"),
    "XGBoost, field": ("XGBoost, field", XR, "out_of_sample/inference_xgboost.py"),
    "LightGBM, field": ("LightGBM, field", XR, "out_of_sample/inference_lgbm.py"),
    "Stacking, field": ("Stacking, field", XR, "experiments/field_reduction/stacking_consistent_oos.py"),
    "SMOTE Stacked, field": ("SMOTE Stacked, field", XR, "out_of_sample/inference_smote_stacked.py"),
    "TabNet (xr_fresh), field": ("TabNet (xr_fresh), field", XR, "out_of_sample/inference_tabnet_field.py"),
    "Logistic Regression, pixel": ("Logistic Regression, pixel", XR_PIXEL, "out_of_sample/inference_base_ml.py"),
    "Random Forest, pixel": ("Random Forest, pixel", XR_PIXEL, "out_of_sample/inference_base_ml.py"),
    "LightGBM, pixel": ("LightGBM, pixel", XR_PIXEL, "out_of_sample/inference_base_ml.py"),
    "XGBoost, pixel": ("XGBoost, pixel", XR_PIXEL, "out_of_sample/inference_base_ml.py"),
    "Voting, pixel": ("Voting, pixel", dict(XR, order="features per pixel, training-fitted imputer and "
                                                       "scaler, pixel votes per field"),
                      "experiments/field_reduction/voting_hybrid_oos.py"),
    "3D CNN, patch": (None, dict(inputs="image patches (band x month)", operator="0",
                                 statistic="none (constant)", order="fill per patch, no re-analysis",
                                 may="column dropped"), "out_of_sample/inference_3d_cnn.py"),
    "Multi-Channel 2D CNN, patch": (None, dict(inputs="image patches (band x month)", operator="0",
                                               statistic="none (constant)", order="fill per patch, no re-analysis",
                                               may="column dropped"), "out_of_sample/inference_multi_channel_cnn.py"),
}


def reanalysis(name, row):
    if name is None:
        return "not re-run (separate patch pipeline)", ""
    if "xr_fresh" in MODELS_FAMILY.get(name, ""):
        how = "xr_fresh features rebuilt from the filled series; inference re-run"
    elif name in ("Stacking, field", "Voting, pixel"):
        how = "retrained inside its script (fixed seeds) on rebuilt features"
    else:
        how = "re-scored with the saved checkpoints; no retraining"
    d = row["orig_fields_differ_from_pub"]
    if pd.isna(d):
        rep = ""
    elif d == 0:
        rep = "yes, field for field"
    else:
        rep = f"no: {int(d)} of 2,415 fields differ"
    return how, rep


if __name__ == "__main__":
    fc = pd.read_csv(FC).set_index("model")
    MODELS_FAMILY = {k: (v[1]["inputs"]) for k, v in MODELS.items()}
    rows = []
    for disp, (fc_name, attr, code) in MODELS.items():
        how, rep = reanalysis(fc_name, fc.loc[fc_name] if fc_name in fc.index else None) if fc_name else reanalysis(None, None)
        rows.append({"model": disp, **attr, "reanalysis": how, "reproduces_published": rep, "code": code})
    sheet = pd.DataFrame(rows)
    res = os.path.join(HERE, "results")
    os.makedirs(res, exist_ok=True)
    sheet.to_csv(os.path.join(res, "fill_sheet.csv"), index=False)
    cols = ["model", "inputs", "operator", "statistic", "order", "may", "reanalysis", "reproduces_published"]
    with open(os.path.join(res, "fill_sheet.md"), "w") as f:
        f.write("Fill sheet: how each published pipeline handled the missing holdout month "
                "(September, 41.5% of pixels; May missing everywhere). Generated by "
                "`experiments/missingness/fill_sheet.py`.\n\n"
                "Note on reproduction: for the xr_fresh models the re-analysis baseline uses features "
                "rebuilt from the raw series, which differ from the published feature file on 85 of 2,415 "
                "fields (an older pixel grid, mostly B12), and the inference ran under newer library "
                "versions; both contribute to the differing field counts. The fill comparison is still "
                "like for like, because every fill variant of a model uses the same rebuilt features and "
                "environment.\n\n")
        f.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for _, r in sheet[cols].iterrows():
            f.write("| " + " | ".join(str(v) for v in r) + " |\n")
    print(sheet[["model", "operator", "statistic", "reproduces_published"]].to_string(index=False))
