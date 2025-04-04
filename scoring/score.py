import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score
import base64
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from io import StringIO

# Load prediction
pred = pd.read_csv("submissions/prediction.csv")

# Decode and load ground truth from secret
gt = pd.read_csv(StringIO(base64.b64decode(os.environ["GROUND_TRUTH"]).decode()))

# Check if lengths match
if len(pred) != len(gt):
    raise ValueError(
        f"Error: Prediction file has {len(pred)} rows but ground truth has {len(gt)} rows. "
        f"Predictions should be at the field level, not pixel level, and match the length of the ground truth."
    )


label_name = "crop_name"


kappa = cohen_kappa_score(gt[label_name], pred[label_name])
f1 = f1_score(gt[label_name], pred[label_name], average="weighted")

# Score
le = LabelEncoder()
y_true = le.fit_transform(gt[label_name])


# Calculate the J score (log loss)
# First convert labels to one-hot encoding if predictions are probabilities

if "probability" in pred.columns:
    # If probabilities are provided
    J = log_loss(gt[label_name], pred[["probability"]])
else:
    # If only labels are provided, this is an approximation
    y_pred = le.transform(pred[label_name])
    y_pred_one_hot = np.zeros((len(y_pred), len(le.classes_)))
    for i, val in enumerate(y_pred):
        y_pred_one_hot[i, val] = 1
    J = log_loss(y_true, y_pred_one_hot)

print(f"Kappa: {kappa:.3f}, F1: {f1:.3f}, Cross Entropy: {J:.3f}")
