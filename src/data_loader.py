"""
Chandrakant Suman - Data loading, schema check and text cleaning.
"""
import numpy as np
import pandas as pd

EXPECTED_COLUMNS = [
    "ID", "Delivery_person_ID", "Delivery_person_Age", "Delivery_person_Ratings",
    "Restaurant_latitude", "Restaurant_longitude",
    "Delivery_location_latitude", "Delivery_location_longitude",
    "Order_Date", "Time_Orderd", "Time_Order_picked",
    "Weatherconditions", "Road_traffic_density", "Vehicle_condition",
    "Type_of_order", "Type_of_vehicle", "multiple_deliveries",
    "Festival", "City", "Time_taken(min)",
]

NUMERIC_COLUMNS = [
    "Delivery_person_Age", "Delivery_person_Ratings",
    "Restaurant_latitude", "Restaurant_longitude",
    "Delivery_location_latitude", "Delivery_location_longitude",
    "Vehicle_condition", "multiple_deliveries",
]


def load_data(path):
    """Read the raw CSV file."""
    df = pd.read_csv(path)
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")
    return df


def validate_schema(df):
    """Stop early if any expected column is missing."""
    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")
    print("Schema check passed")


def fix_text_columns(df):
    """Clean messy text and convert columns to the right types."""
    df = df.copy()

    # 1. Strip extra spaces from every text column ("Jam " -> "Jam")
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # 2. Text "NaN" -> real missing value
    df = df.replace("NaN", np.nan)

    # 3. "conditions Sunny" -> "Sunny"  ("conditions NaN" -> missing)
    df["Weatherconditions"] = df["Weatherconditions"].str.replace("conditions ", "", regex=False)
    df["Weatherconditions"] = df["Weatherconditions"].replace("NaN", np.nan)

    # 4. Target: "(min) 24" -> 24, and give it a clean name
    #    (test.csv and app inputs have no target, so only do this if it exists)
    if "Time_taken(min)" in df.columns:
        df["Time_taken_min"] = df["Time_taken(min)"].str.replace("(min)", "", regex=False).str.strip().astype(int)
        df = df.drop(columns=["Time_taken(min)"])

    # 5. Text numbers -> real numbers
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 6. Date text -> datetime (format in file is day-month-year)
    df["Order_Date"] = pd.to_datetime(df["Order_Date"], format="%d-%m-%Y")

    return df


if __name__ == "__main__":
    data = load_data("data/raw/dataset.csv")
    validate_schema(data)
    data = fix_text_columns(data)
    print(data.dtypes)
    print(data.isnull().sum())