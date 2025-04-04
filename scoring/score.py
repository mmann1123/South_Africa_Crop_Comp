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

kappa = cohen_kappa_score(gt["label"], pred["label"])
f1 = f1_score(gt["label"], pred["label"], average="weighted")

# Score
le = LabelEncoder()
y_true = le.fit_transform(gt["label"])


# Calculate the J score (log loss)
# First convert labels to one-hot encoding if predictions are probabilities

if "probability" in pred.columns:
    # If probabilities are provided
    J = log_loss(gt["label"], pred[["probability"]])
else:
    # If only labels are provided, this is an approximation
    y_pred = le.transform(pred["label"])
    y_pred_one_hot = np.zeros((len(y_pred), len(le.classes_)))
    for i, val in enumerate(y_pred):
        y_pred_one_hot[i, val] = 1
    J = log_loss(y_true, y_pred_one_hot)

print(f"Kappa: {kappa:.3f}, F1: {f1:.3f}, Cross Entropy: {J:.3f}")
