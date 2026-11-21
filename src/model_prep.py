"""
Chandrakant (Member 1) - Train/test split + encoding + scaling.
"""
import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Time_taken_min"

NUMERIC_FEATURES = [
    "Delivery_person_Age", "Delivery_person_Ratings", "distance_km",
    "order_hour", "order_weekday", "prep_time_min",
    "Vehicle_condition", "multiple_deliveries",
]
CATEGORICAL_FEATURES = [
    "Weatherconditions", "Road_traffic_density", "Type_of_order",
    "Type_of_vehicle", "Festival", "City",
]


def load_featured(path="data/processed/featured_data.csv"):
    df = pd.read_csv(path)
    print(f"Loaded featured data: {df.shape}")
    return df


def split_data(df, test_size=0.2):
    """80% train, 20% test. random_state=42 so every member gets the same split."""
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def build_preprocessor():
    """Scale numbers, one-hot encode categories."""
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ])


def prepare_data(path="data/processed/featured_data.csv", save_path="models/preprocessor.pkl"):
    """Full prep: load -> split -> fit preprocessor on TRAIN only -> transform both."""
    df = load_featured(path)
    X_train, X_test, y_train, y_test = split_data(df)

    preprocessor = build_preprocessor()
    X_train_p = preprocessor.fit_transform(X_train)   # learn from train only
    X_test_p = preprocessor.transform(X_test)          # apply same rules to test

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(preprocessor, save_path)
    print(f"Preprocessor saved to {save_path}  ({X_train_p.shape[1]} features after encoding)")

    feature_names = list(preprocessor.get_feature_names_out())
    return X_train_p, X_test_p, y_train.values, y_test.values, feature_names


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, names = prepare_data()
    print(names)