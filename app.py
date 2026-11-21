"""
Om Singh Bisen - Streamlit UI.
Run:  streamlit run app.py
"""
import datetime as dt
import os

import pandas as pd
import streamlit as st

from src.predict import build_features, load_artifacts, predict, predict_order

st.set_page_config(page_title="Delivery Time Estimator", page_icon="🛵", layout="wide")


@st.cache_resource
def get_artifacts():
    return load_artifacts()


if not os.path.exists("results/best_model.json"):
    st.error("No trained model found. Run `python main.py` first.")
    st.stop()

art = get_artifacts()
st.title("🛵 Food Delivery Time Estimator")
st.caption(f"Model in use: **{art['model_name']}** · Team Neural Storm")

tab_single, tab_batch, tab_perf = st.tabs(["Single order", "Batch (CSV upload)", "Model performance"])

# ---------------- Tab 1: one order ----------------
with tab_single:
    with st.form("order_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.subheader("Courier")
            age = st.number_input("Courier age", 18, 60, 28)
            rating = st.slider("Courier rating", 1.0, 5.0, 4.6, 0.1)
            vehicle = st.selectbox("Vehicle type", ["motorcycle", "scooter", "electric_scooter"])
            vehicle_cond = st.selectbox("Vehicle condition (0 = poor, 2 = good)", [0, 1, 2], index=2)
            multi = st.selectbox("Multiple deliveries", [0, 1, 2, 3], index=1)

        with c2:
            st.subheader("Route")
            r_lat = st.number_input("Restaurant latitude", value=12.914264, format="%.6f")
            r_lon = st.number_input("Restaurant longitude", value=77.678400, format="%.6f")
            d_lat = st.number_input("Delivery latitude", value=12.994264, format="%.6f")
            d_lon = st.number_input("Delivery longitude", value=77.758400, format="%.6f")
            city = st.selectbox("City type", ["Metropolitian", "Urban", "Semi-Urban"])

        with c3:
            st.subheader("Order & conditions")
            order_date = st.date_input("Order date", dt.date(2022, 3, 25))
            order_time = st.time_input("Order time", dt.time(19, 45))
            pickup_time = st.time_input("Pickup time", dt.time(19, 55))
            order_type = st.selectbox("Order type", ["Meal", "Snack", "Drinks", "Buffet"])
            weather = st.selectbox("Weather", ["Sunny", "Cloudy", "Fog", "Windy", "Stormy", "Sandstorms"])
            traffic = st.selectbox("Traffic", ["Low", "Medium", "High", "Jam"], index=3)
            festival = st.radio("Festival day?", ["No", "Yes"], horizontal=True)

        submitted = st.form_submit_button("Predict delivery time", type="primary")

    if submitted:
        order = {
            "Delivery_person_Age": age, "Delivery_person_Ratings": rating,
            "Restaurant_latitude": r_lat, "Restaurant_longitude": r_lon,
            "Delivery_location_latitude": d_lat, "Delivery_location_longitude": d_lon,
            "Order_Date": order_date.strftime("%d-%m-%Y"),
            "Time_Orderd": order_time.strftime("%H:%M:%S"),
            "Time_Order_picked": pickup_time.strftime("%H:%M:%S"),
            "Weatherconditions": weather, "Road_traffic_density": traffic,
            "Vehicle_condition": vehicle_cond, "Type_of_order": order_type,
            "Type_of_vehicle": vehicle, "multiple_deliveries": multi,
            "Festival": festival, "City": city,
        }
        minutes = predict_order(order)
        features = build_features(pd.DataFrame([order]))

        m1, m2, m3 = st.columns(3)
        m1.metric("⏱ Estimated delivery time", f"{minutes} min")
        m2.metric("Distance", f"{features['distance_km'].iloc[0]:.1f} km")
        m3.metric("Prep time", f"{features['prep_time_min'].iloc[0]:.0f} min")
        st.caption("Typical error of this model is about ±3 minutes (MAE on the test set).")

# ---------------- Tab 2: CSV upload ----------------
with tab_batch:
    st.write("Upload a CSV in the Kaggle format (for example `data/raw/test.csv`).")
    file = st.file_uploader("CSV file", type="csv")
    if file is not None:
        raw = pd.read_csv(file)
        preds = predict(build_features(raw))
        out = pd.DataFrame({"ID": raw["ID"].astype(str).str.strip() if "ID" in raw else range(len(raw)),
                            "predicted_time_min": preds.round(1)})
        st.success(f"Predicted {len(out)} orders · average {out['predicted_time_min'].mean():.1f} min")
        st.dataframe(out, width="stretch", height=300)
        st.download_button("Download predictions", out.to_csv(index=False),
                           "predictions.csv", "text/csv")

# ---------------- Tab 3: model performance ----------------
with tab_perf:
    if os.path.exists("results/model_comparison.csv"):
        st.subheader("Model comparison (20% test set)")
        st.dataframe(pd.read_csv("results/model_comparison.csv"), width="stretch")
    for img, caption in [("images/model_comparison.png", "RMSE / MAE / R² by model"),
                         ("images/all_models_actual_vs_predicted.png", "Actual vs predicted"),
                         ("images/feature_importance.png", "What drives delivery time")]:
        if os.path.exists(img):
            st.image(img, caption=caption)
