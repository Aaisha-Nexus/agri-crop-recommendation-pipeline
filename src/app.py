from flask import Flask, request, jsonify
import os
import joblib
import pandas as pd

app = Flask(__name__)

# Project root folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Saved files
model_path = os.path.join(BASE_DIR, "models", "crop_model.pkl")
scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")
columns_path = os.path.join(BASE_DIR, "models", "model_columns.pkl")

# Load model, scaler, and column order
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
model_columns = joblib.load(columns_path)

crop_mapping = {
    0: "apple", 1: "banana", 2: "blackgram", 3: "chickpea",
    4: "coconut", 5: "coffee", 6: "cotton", 7: "grapes",
    8: "jute", 9: "kidneybeans", 10: "lentil", 11: "maize",
    12: "mango", 13: "mothbeans", 14: "mungbean", 15: "muskmelon",
    16: "orange", 17: "papaya", 18: "pigeonpeas",
    19: "pomegranate", 20: "rice", 21: "watermelon"
}


def get_rainfall_category(rainfall):
    if rainfall <= 100:
        return "Low"
    elif rainfall <= 200:
        return "Medium"
    return "High"


def get_temperature_category(temperature):
    if temperature < 20:
        return "Cool"
    elif temperature < 30:
        return "Moderate"
    return "Hot"


def get_humidity_category(humidity):
    if humidity < 40:
        return "Low"
    elif humidity < 70:
        return "Medium"
    return "High"


def get_season_type(temperature):
    if temperature < 20:
        return "Winter"
    elif temperature < 30:
        return "Moderate"
    return "Summer"


def create_model_input(data):
    # Raw values coming from user/API request
    row = {
        "N": data["N"],
        "P": data["P"],
        "K": data["K"],
        "temperature": data["temperature"],
        "humidity": data["humidity"],
        "ph": data["ph"],
        "rainfall": data["rainfall"]
    }

    # Create category values from raw inputs
    rainfall_cat = get_rainfall_category(row["rainfall"])
    temp_cat = get_temperature_category(row["temperature"])
    humidity_cat = get_humidity_category(row["humidity"])
    season = get_season_type(row["temperature"])

    # Add one-hot encoded category columns
    row.update({
        "rainfall_category_Low": 0,
        "rainfall_category_Medium": 0,
        "rainfall_category_High": 0,
        "temperature_category_Cool": 0,
        "temperature_category_Moderate": 0,
        "temperature_category_Hot": 0,
        "humidity_category_Low": 0,
        "humidity_category_Medium": 0,
        "humidity_category_High": 0,
        "season_type_Moderate": 0,
        "season_type_Summer": 0,
        "season_type_Winter": 0
    })

    row[f"rainfall_category_{rainfall_cat}"] = 1
    row[f"temperature_category_{temp_cat}"] = 1
    row[f"humidity_category_{humidity_cat}"] = 1
    row[f"season_type_{season}"] = 1

    input_df = pd.DataFrame([row])

    # Scale numeric columns exactly like training
    num_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    # Ensure same column order as training
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    return input_df


@app.route("/")
def home():
    return jsonify({
        "message": "Crop Recommendation API is running",
        "predict_endpoint": "/predict",
        "required_inputs": ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        required_fields = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

        missing_fields = [
            field for field in required_fields
            if field not in data
        ]

        if missing_fields:
            return jsonify({
                "error": "Missing required fields",
                "missing_fields": missing_fields
            }), 400

        model_input = create_model_input(data)

        prediction = model.predict(model_input)[0]
        recommended_crop = crop_mapping[int(prediction)]

        return jsonify({
            "status": "success",
            "recommended_crop": recommended_crop,
            "encoded_prediction": int(prediction),
            "input_received": data,
            "message": f"{recommended_crop.title()} is recommended based on the given soil and environmental conditions."
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400


if __name__ == "__main__":
    app.run(debug=True)