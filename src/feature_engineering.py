"""
Rshishabh Yadav - Feature engineering.
"""
import numpy as np

FINAL_COLUMNS = [
    "Delivery_person_Age", "Delivery_person_Ratings", "distance_km",
    "order_hour", "order_weekday", "prep_time_min",
    "Vehicle_condition", "multiple_deliveries",
    "Weatherconditions", "Road_traffic_density", "Type_of_order",
    "Type_of_vehicle", "Festival", "City",
    "Time_taken_min",  # target
]


def haversine(lat1, lon1, lat2, lon2):
    """Straight-line distance in km between two points on Earth."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def add_distance(df):
    df = df.copy()
    df["distance_km"] = haversine(
        df["Restaurant_latitude"], df["Restaurant_longitude"],
        df["Delivery_location_latitude"], df["Delivery_location_longitude"],
    ).round(2)
    return df


def _to_minutes(time_text):
    """'19:45:00' -> 1185 (minutes after midnight)."""
    parts = time_text.str.split(":", expand=True).astype(int)
    return parts[0] * 60 + parts[1]


def add_time_features(df):
    df = df.copy()
    ordered = _to_minutes(df["Time_Orderd"])
    picked = _to_minutes(df["Time_Order_picked"])

    df["order_hour"] = ordered // 60
    df["order_weekday"] = df["Order_Date"].dt.day_of_week  # 0 = Monday
    # % 1440 handles orders that cross midnight (23:55 -> 00:05 = 10 min)
    df["prep_time_min"] = (picked - ordered) % 1440
    return df


def select_features(df):
    """Keep only the columns the model will use."""
    return df[FINAL_COLUMNS]


if __name__ == "__main__":
    print(haversine(12.97, 77.59, 13.01, 77.64))  # about 7 km