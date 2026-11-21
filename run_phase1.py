"""
Phase 1: raw data -> cleaned_data.csv -> featured_data.csv
Run alone:  python run_phase1.py      (or through main.py)
"""
import os

from src.data_loader import load_data, validate_schema, fix_text_columns
from src.preprocessing import remove_outliers, handle_missing
from src.feature_engineering import add_distance, add_time_features, select_features
from src.eda import make_plots

RAW_PATH = "data/raw/dataset.csv"
CLEANED_PATH = "data/processed/cleaned_data.csv"
FEATURED_PATH = "data/processed/featured_data.csv"


def main():
    os.makedirs("data/processed", exist_ok=True)

    # Step 1: Load (Member 1)
    df = load_data(RAW_PATH)
    validate_schema(df)
    df = fix_text_columns(df)

    # Step 2: Preprocess (Member 2)
    df = remove_outliers(df)
    df = handle_missing(df)
    df.to_csv(CLEANED_PATH, index=False)
    print(f"Saved {CLEANED_PATH}  shape={df.shape}")

    # Step 3: Feature engineering (Member 3)
    df = add_distance(df)
    df = add_time_features(df)
    df = select_features(df)
    df.to_csv(FEATURED_PATH, index=False)
    print(f"Saved {FEATURED_PATH}  shape={df.shape}")

    # Step 4: Charts (Member 1)
    make_plots(df)

    print("Phase 1 complete")


if __name__ == "__main__":
    main()