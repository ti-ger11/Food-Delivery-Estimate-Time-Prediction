"""
Member 2 - Outlier removal and missing value handling.
"""


def remove_outliers(df):
    """Remove rows with impossible values."""
    df = df.copy()
    start = len(df)

    # 1. Some coordinates were typed as negative -> make them positive (India is +lat, +long)
    for col in ["Restaurant_latitude", "Restaurant_longitude",
                "Delivery_location_latitude", "Delivery_location_longitude"]:
        df[col] = df[col].abs()

    # 2. Restaurant location 0,0 is not a real place -> drop those rows
    df = df[(df["Restaurant_latitude"] > 1) & (df["Restaurant_longitude"] > 1)]

    # 3. Ratings must be between 1 and 5 (there are some 6s); keep missing for now
    df = df[~(df["Delivery_person_Ratings"] > 5)]

    # 4. Delivery partners must be adults (there are some age 15)
    df = df[~(df["Delivery_person_Age"] < 18)]

    # 5. Duplicate rows
    df = df.drop_duplicates()

    print(f"Outliers removed: {start - len(df)} rows dropped, {len(df)} left")
    return df


def handle_missing(df):
    """Fill or drop missing values."""
    df = df.copy()

    # Numbers -> median (not affected by extreme values)
    for col in ["Delivery_person_Age", "Delivery_person_Ratings", "multiple_deliveries"]:
        df[col] = df[col].fillna(df[col].median())

    # Categories -> mode (most common value)
    for col in ["Weatherconditions", "Road_traffic_density", "Festival", "City"]:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Order time is missing -> we cannot calculate prep time, so drop those rows
    before = len(df)
    df = df.dropna(subset=["Time_Orderd"])
    print(f"Dropped {before - len(df)} rows with missing order time")

    print(f"Missing values left: {df.isnull().sum().sum()}")
    return df


if __name__ == "__main__":
    from src.data_loader import load_data, fix_text_columns

    data = fix_text_columns(load_data("data/raw/train.csv"))
    data = remove_outliers(data)
    data = handle_missing(data)
    print(data.shape)