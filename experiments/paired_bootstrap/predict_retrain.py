"""Predict the holdout tile with the retrained L-TAE pixel ensemble
(ltae_pixel_retrain/frac_1.00) using the same predict_ltae_pixel() routine that
produced the L-TAE-S frac_1.00 predictions. Output stays in this folder."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "field_reduction"))
from predict_oos import predict_ltae_pixel
os.makedirs(os.path.join(HERE, "rerun"), exist_ok=True)
predict_ltae_pixel(os.path.join(HERE, "ltae_pixel_retrain", "frac_1.00"),
                   os.path.join(HERE, "rerun", "predictions_ltae_retrain.csv"))
