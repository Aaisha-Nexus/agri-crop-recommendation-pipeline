"""
PREDICT.PY

Purpose:
This script loads the trained Random Forest model saved using Joblib
and performs crop prediction on new input data.

Why are sample values used?

The current model was trained using both original features and
engineered features (19 total inputs), including:

- N, P, K
- temperature
- humidity
- pH
- rainfall
- rainfall categories
- temperature categories
- humidity categories
- season categories

Because these engineered features must be supplied during prediction,
we currently use sample values from the dataset to verify that:

1. The saved model loads successfully
2. The deployment workflow works correctly
3. Predictions can be generated outside the notebook

Future Improvement (API Integration):
In the next project phase, the prediction process will be upgraded.

User Inputs:
- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- pH
- City / Location

Weather API will automatically provide:
- Temperature
- Humidity
- Rainfall information

The system will then create the required engineered features,
combine them with user inputs, and generate a real-time crop recommendation.

Current version = Deployment testing
Future version = Real-world prediction system
"""


import joblib
import numpy as np
import os

# -----------------------
# LOAD SAVED MODEL
# -----------------------

model_path = os.path.join(
    os.path.dirname(__file__),
    "../models/crop_model.pkl"
)

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
# SAMPLE INPUT
# -----------------------
# Replace with a real row from crop_ml_ready.csv
# for deployment testing

sample_data = np.array([
    [
        90,     # N
        42,     # P
        43,     # K
        20.87,  # temperature
        82.00,  # humidity
        6.50,   # ph
        202.93, # rainfall

        0,      # rainfall_category_Low
        0,      # rainfall_category_Medium
        1,      # rainfall_category_High

        0,      # temperature_category_Cool
        1,      # temperature_category_Moderate
        0,      # temperature_category_Hot

        0,      # humidity_category_Low
        0,      # humidity_category_Medium
        1,      # humidity_category_High

        0,      # season_type_Moderate
        0,      # season_type_Summer
        1       # season_type_Winter
    ]
])

# -----------------------
# MAKE PREDICTION
# -----------------------

prediction = model.predict(sample_data)

predicted_crop = crop_mapping[prediction[0]] # here prediction[0] means give me the first prediction from the array and then we will map it to the crop name using the crop_mapping dictionary

print("Recommended Crop:", predicted_crop)

