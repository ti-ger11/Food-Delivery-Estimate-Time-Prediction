"""
Rshishabh - Shared prediction logic used by app.py, api.py and predict_test.py.

Raw order(s) -> same cleaning + features as training -> preprocessor -> best model.
New orders can't be dropped, so bad/missing values are FILLED with the
training data's medians/modes instead of removing the row.
"""
import json

import joblib
import numpy as np
import pandas as pd

from src.data_loader import fix_text_columns
from src.feature_engineering import haversine
from src.model_prep import CATEGORICAL_FEATURES, NUMERIC_FEATURES

FEATURED_PATH = "data/processed/featured_data.csv"
PREPROCESSOR_PATH = "models/preprocessor.pkl"
BEST_MODEL_INFO = "results/best_model.json"

_cache = {}


def load_artifacts():
    """Load preprocessor, best model and training defaults once."""
    if not _cache:
        with open(BEST_MODEL_INFO) as f:
            info = json.load(f)
        if info["path"].endswith(".keras"):
            import keras
            model = keras.models.load_model(info["path"])
        else:
            model = joblib.load(info["path"])

        train = pd.read_csv(FEATURED_PATH)
        defaults = {col: train[col].median() for col in NUMERIC_FEATURES}
        defaults.update({col: train[col].mode()[0] for col in CATEGORICAL_FEATURES})

        _cache.update(model=model, model_name=info["best_model"],
                      preprocessor=joblib.load(PREPROCESSOR_PATH), defaults=defaults)
    return _cache


def _to_minutes(series):
    """'19:45:00' or '19:45' -> 1185. Missing stays missing."""
    parts = series.astype("string").str.split(":", expand=True)
    hours = pd.to_numeric(parts[0], errors="coerce")
    minutes = pd.to_numeric(parts[1], errors="coerce")
    return hours * 60 + minutes


def build_features(raw_df):
    """Raw Kaggle-style rows -> the 14 model features (no rows dropped)."""
    defaults = load_artifacts()["defaults"]
    df = fix_text_columns(raw_df)

    # Coordinates: fix negatives, treat 0,0 as unknown
    for col in ["Restaurant_latitude", "Restaurant_longitude",
                "Delivery_location_latitude", "Delivery_location_longitude"]:
        df[col] = df[col].abs()
    valid = (df["Restaurant_latitude"] > 1) & (df["Restaurant_longitude"] > 1)
    df["distance_km"] = haversine(df["Restaurant_latitude"], df["Restaurant_longitude"],
                                  df["Delivery_location_latitude"], df["Delivery_location_longitude"])
    df.loc[~valid, "distance_km"] = np.nan

    # Impossible values -> unknown
    df.loc[df["Delivery_person_Ratings"] > 5, "Delivery_person_Ratings"] = np.nan
    df.loc[df["Delivery_person_Age"] < 18, "Delivery_person_Age"] = np.nan

    # Time features (order time can be missing -> use pickup hour)
    ordered = _to_minutes(df["Time_Orderd"])
    picked = _to_minutes(df["Time_Order_picked"])
    df["order_hour"] = (ordered // 60).fillna(picked // 60)
    df["prep_time_min"] = (picked - ordered) % 1440
    df["order_weekday"] = df["Order_Date"].dt.day_of_week

    # Fill every gap with training defaults
    features = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    for col in features.columns:
        features[col] = features[col].fillna(defaults[col])
    return features


def predict(features):
    """Features DataFrame -> predicted minutes (numpy array)."""
    art = load_artifacts()
    X = art["preprocessor"].transform(features)
    return np.asarray(art["model"].predict(X)).ravel()


def predict_order(order):
    """One order as a dict (Kaggle column names) -> predicted minutes."""
    features = build_features(pd.DataFrame([order]))
    return round(float(predict(features)[0]), 1)


if __name__ == "__main__":
    sample = {
        "Delivery_person_Age": 28, "Delivery_person_Ratings": 4.6,
        "Restaurant_latitude": 12.914264, "Restaurant_longitude": 77.678400,
        "Delivery_location_latitude": 12.994264, "Delivery_location_longitude": 77.758400,
        "Order_Date": "25-03-2022", "Time_Orderd": "19:45:00", "Time_Order_picked": "19:55:00",
        "Weatherconditions": "Sunny", "Road_traffic_density": "Jam",
        "Vehicle_condition": 2, "Type_of_order": "Meal", "Type_of_vehicle": "motorcycle",
        "multiple_deliveries": 1, "Festival": "No", "City": "Metropolitian",
    }
    print(f"Predicted delivery time: {predict_order(sample)} min")
