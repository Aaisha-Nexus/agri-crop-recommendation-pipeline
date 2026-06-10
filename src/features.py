import os
import joblib
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


def create_features():

    # Load cleaned dataset
    df = pd.read_csv("data/processed/crop_clean.csv")

    # -----------------------------------
    # Rainfall Categories
    # -----------------------------------
    df["rainfall_category"] = pd.cut(
        df["rainfall"],
        bins=[0, 100, 200, 400],
        labels=["Low", "Medium", "High"]
    )

    # -----------------------------------
    # Temperature Categories
    # -----------------------------------
    df["temperature_category"] = pd.cut(
        df["temperature"],
        bins=[0, 20, 30, 50],
        labels=["Cool", "Moderate", "Hot"]
    )

    # -----------------------------------
    # Humidity Categories
    # -----------------------------------
    df["humidity_category"] = pd.cut(
        df["humidity"],
        bins=[0, 40, 70, 100],
        labels=["Low", "Medium", "High"]
    )

    # -----------------------------------
    # Season Feature
    # -----------------------------------
    def season_mapper(temp):
        if temp < 20:
            return "Winter"
        elif temp < 30:
            return "Moderate"
        else:
            return "Summer"

    df["season_type"] = df["temperature"].apply(season_mapper)

    # -----------------------------------
    # Encode Target
    # -----------------------------------
    le = LabelEncoder()
    df["target"] = le.fit_transform(df["label"])

    # -----------------------------------
    # Define X and y
    # -----------------------------------
    X = df.drop(
        columns=[
            "label",
            "target"
        ]
    )

    y = df["target"]

    # -----------------------------------
    # One-Hot Encode Categories
    # -----------------------------------
    X = pd.get_dummies(
        X,
        columns=[
            "rainfall_category",
            "temperature_category",
            "humidity_category",
            "season_type"
        ]
    )

    # -----------------------------------
    # Scale Numerical Features
    # -----------------------------------
    scaler = StandardScaler()

    num_cols = [
        "N",
        "P",
        "K",
        "temperature",
        "humidity",
        "ph",
        "rainfall"
    ]

    X[num_cols] = scaler.fit_transform(X[num_cols])

    # -----------------------------------
    # Save scaler and model columns
    # -----------------------------------
    os.makedirs("models", exist_ok=True)

    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(X.columns.tolist(), "models/model_columns.pkl")

    # -----------------------------------
    # Final ML Dataset
    # -----------------------------------
    ml_ready = X.copy()
    ml_ready["target"] = y

    ml_ready.to_csv(
        "data/processed/crop_ml_ready.csv",
        index=False
    )

    print("Feature engineering complete!")
    print("ML-ready dataset saved!")
    print("Scaler saved!")
    print("Model columns saved!")
    print("Shape:", ml_ready.shape)


if __name__ == "__main__":
    create_features()