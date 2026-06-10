"""
PREDICT.PY

Purpose:
Loads the saved model and tests prediction using a real ML-ready row
from crop_ml_ready.csv.

This ensures prediction input has the same format used during training.
"""

import os
import joblib
import pandas as pd

# -----------------------
# PATHS
# -----------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "models", "crop_model.pkl")
data_path = os.path.join(BASE_DIR, "data", "processed", "crop_ml_ready.csv")

# -----------------------
# LOAD MODEL
# -----------------------
model = joblib.load(model_path)

# -----------------------
# CROP LABEL MAPPING
# -----------------------
crop_mapping = {
    0: "apple",
    1: "banana",
    2: "blackgram",
    3: "chickpea",
    4: "coconut",
    5: "coffee",
    6: "cotton",
    7: "grapes",
    8: "jute",
    9: "kidneybeans",
    10: "lentil",
    11: "maize",
    12: "mango",
    13: "mothbeans",
    14: "mungbean",
    15: "muskmelon",
    16: "orange",
    17: "papaya",
    18: "pigeonpeas",
    19: "pomegranate",
    20: "rice",
    21: "watermelon"
}

# -----------------------
# PREDICT USING REAL ML-READY ROW
# -----------------------
def predict_sample_row(row_index=0):
    # Load ML-ready dataset
    df = pd.read_csv(data_path)

    # Separate features and target
    X = df.drop("target", axis=1)
    y = df["target"]

    # Select one row for testing
    # iloc[[row_index]] keeps it as a DataFrame, which sklearn expects
    sample = X.iloc[[row_index]]

    # Get the actual crop label for the selected row
    actual_encoded = y.iloc[row_index]

    # Predict crop for the selected row
    # model.predict(sample) returns an array like [20]
    # [0] extracts the first prediction from that array
    predicted_encoded = model.predict(sample)[0]

    # Convert encoded labels into crop names
    predicted_crop = crop_mapping[predicted_encoded]
    actual_crop = crop_mapping[actual_encoded]

    print("Prediction Test")
    print("----------------")
    print("Row Index:", row_index)

    print("\nInput Values Used:")
    print(sample)

    print("\nPredicted Encoded Label:", predicted_encoded)
    print("Actual Encoded Label:", actual_encoded)

    print("\nPredicted Crop:", predicted_crop)
    print("Actual Crop:", actual_crop)

    if predicted_crop == actual_crop:
        print("\nResult: Correct prediction")
    else:
        print("\nResult: Incorrect prediction")


if __name__ == "__main__":
    predict_sample_row(150)