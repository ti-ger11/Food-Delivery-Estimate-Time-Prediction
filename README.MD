# 🛵 Delivery Time Estimation — ML Regression System

> Predicting food delivery duration (in minutes) from route, traffic, courier and order characteristics using traditional ML and a neural network.

---

## 📌 Table of Contents
1. [Problem Statement](#-problem-statement)
2. [Use Case](#-use-case-delivery-time-estimation)
3. [Objectives](#-objectives)
4. [Dataset](#-dataset)
5. [System Design](#-system-design)
6. [Tech Stack](#-tech-stack)
7. [Project Structure](#-project-structure)
8. [How to Run](#-how-to-run)
9. [Results](#-results)
10. [Team](#-team)

---

## ❓ Problem Statement

Many industries rely on accurate numerical predictions to support decision-making. Regression models play a central role in forecasting continuous values such as sales, customer spend, equipment failure patterns and delivery times. Building an effective regression solution requires clean data, robust modeling, hyperparameter optimization and clear evaluation.

In food and parcel delivery, an inaccurate **Estimated Time of Arrival (ETA)** directly hurts the business:

- **Customers** lose trust when orders arrive later than promised.
- **Operations teams** cannot plan courier allocation or batch orders efficiently.
- **Restaurants** cannot time food preparation to courier arrival.

A simple rule such as *"distance ÷ average speed"* ignores traffic, weather, courier experience, multiple orders and festival rush, so it is frequently wrong.

**Goal:** build an end-to-end Machine Learning Regression System that learns from historical deliveries and predicts the delivery time of a new order.

---

## 🎯 Use Case: Delivery Time Estimation

We chose **Option C — Delivery Time Estimation** from the hackathon use cases.

| Item | Description |
|---|---|
| **Business question** | *"How many minutes will this order take to deliver?"* |
| **Prediction type** | Regression (continuous target) |
| **Target variable** | `Time_taken(min)` |
| **Inputs** | Route (pickup & drop location, city), traffic, weather, courier profile, vehicle, order type, festival flag, order time |
| **Consumers** | Customer app (ETA display), dispatch team (courier assignment), analysts (delay drivers) |

### Example
```
Input : 6.2 km, High traffic, Cloudy, Motorcycle, rating 4.7, 1 extra order, no festival, 8:15 PM
Output: Estimated delivery time ≈ 31 minutes
```

---

## ✅ Objectives

- Ingest and validate structured delivery data (CSV)
- Clean data: missing values, inconsistent text, invalid values and outliers
- Perform EDA and correlation analysis to identify key drivers of delay
- Engineer domain features (distance, preparation time, peak hour, etc.)
- Train and compare **traditional ML models** and an **Artificial Neural Network**
- Tune hyperparameters with **RandomizedSearchCV / GridSearchCV** and validate with **K-Fold Cross Validation**
- Evaluate using **RMSE, MAE and R²**, with residual and feature-importance plots
- Serve the best model through a **FastAPI REST API** and a **Streamlit UI**

---

## 📊 Dataset

**Source:** [Food Delivery Dataset — Kaggle (Gaurav Malik)](https://www.kaggle.com/datasets/gauravmalik26/food-delivery-dataset)
**Size:** ~45,593 orders × 20 columns
**Domain:** Food deliveries across Indian cities

### Column Description

| Group | Column | Description | Type |
|---|---|---|---|
| **Identifiers** | `ID` | Unique order ID (dropped) | string |
| | `Delivery_person_ID` | Courier ID (dropped) | string |
| **Courier** | `Delivery_person_Age` | Age of the courier | numeric |
| | `Delivery_person_Ratings` | Average customer rating (1–5) | numeric |
| | `Vehicle_condition` | 0 = Best, 1 = Good, 2 = Fair, 3 = Worst | ordinal |
| | `Type_of_vehicle` | motorcycle / scooter / electric_scooter / bicycle | categorical |
| | `multiple_deliveries` | Number of orders carried together (0–3) | numeric |
| **Route** | `Restaurant_latitude`, `Restaurant_longitude` | Pickup location | numeric |
| | `Delivery_location_latitude`, `Delivery_location_longitude` | Drop location | numeric |
| | `City` | Metropolitian / Urban / Semi-Urban | categorical |
| **Time** | `Order_Date` | Date of the order (DD-MM-YYYY) | date |
| | `Time_Orderd` | Time the order was placed | time |
| | `Time_Order_picked` | Time the courier picked up the order | time |
| **Conditions** | `Weatherconditions` | Sunny / Cloudy / Fog / Stormy / Sandstorms / Windy | categorical |
| | `Road_traffic_density` | Low / Medium / High / Jam | ordinal |
| | `Festival` | Yes / No | binary |
| **Package** | `Type_of_order` | Snack / Meal / Drinks / Buffet | categorical |
| **Target** | `Time_taken(min)` | Delivery duration in minutes | numeric |

### Data Quality Issues Identified

| Issue | Example | Handling |
|---|---|---|
| Target stored as text | `"(min) 24"` | Extract number with regex |
| Missing values stored as text | `"NaN "` | Convert to real `NaN`, then impute |
| Extra whitespace / prefixes | `"High "`, `"conditions Sunny"` | Strip and clean strings |
| Numbers stored as text | Age, Ratings | `pd.to_numeric(errors="coerce")` |
| Invalid values | Rating = 6, Age < 18, coordinates = 0 / negative | Remove or correct rows |
| Unrealistic distances | Thousands of km | Remove using distance threshold |
| Target outliers | Extreme delivery times | IQR-based removal |

### Engineered Features

| Feature | Logic | Why it matters |
|---|---|---|
| `distance_km` | Haversine distance between restaurant and customer | Core driver of travel time |
| `prep_time_min` | `Time_Order_picked − Time_Orderd` | Restaurant preparation delay |
| `order_hour` | Hour from `Time_Orderd` | Time-of-day effect |
| `day_of_week`, `is_weekend` | From `Order_Date` | Weekly demand pattern |
| `is_peak_hour` | Lunch (12–15) or dinner (19–23) | Rush-hour congestion |
| `traffic_level` | Low=0, Medium=1, High=2, Jam=3 | Preserves natural order |

### Conceptual Model
```
Delivery time ≈ Restaurant prep time
              + Travel time (distance × traffic × weather × vehicle)
              + Delays (multiple orders, festival, courier rating/experience)
```

---

## 🏗️ System Design

### High-Level Architecture

```
┌──────────────┐   ┌────────────────────┐   ┌──────────────────────┐
│ Data Ingestion│──►│ Data Preprocessing │──►│ Feature Engineering   │
│  (CSV + schema│   │ • clean text / NaN │   │ • distance_km         │
│   validation) │   │ • invalid values   │   │ • prep_time, hour,    │
└──────────────┘   │ • outlier removal  │   │   peak, weekend       │
                   └────────────────────┘   │ • correlation analysis│
                                            └──────────┬───────────┘
                                                       ▼
┌──────────────────────────────────────────────────────────────────┐
│ Model Training & Optimization (sklearn Pipeline)                   │
│   Impute → Encode → Scale → Model                                  │
│   Models: Linear Regression  · XGBoost · MLP Regressor (ANN)       │
│                                                │
│   Tuning: RandomizedSearchCV / GridSearchCV · 5-Fold Cross Val     │
└──────────────────────────────┬───────────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│ Evaluation: RMSE · MAE · R² · Residual plots · Pred vs Actual ·  │
│ Feature importance → reports/metrics.json + reports/plots/       │
└──────────────────────────────┬───────────────────────────────────┘
                               ▼
              models/best_model.pkl (full pipeline)
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
     FastAPI REST API                    Streamlit UI
     POST /predict                       Order form → ETA
     GET  /metrics                       Model comparison & plots
     GET  /health                        Feature importance
```


---

## Team
- Member 1: Rshishabh Yadav
- Member 2: Chandrakant Suman
- Member 3: Om Singh BIsen