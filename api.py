"""
Rshishabh - FastAPI service.
Run:   uvicorn api:app --reload
Test:  open http://127.0.0.1:8000/docs
"""
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.predict import load_artifacts, predict_order

app = FastAPI(title="Food Delivery Time Prediction API")
load_artifacts()  # load model once at start-up, not on every request


class Order(BaseModel):
    Delivery_person_Age: float = Field(28, ge=18, le=60)
    Delivery_person_Ratings: float = Field(4.6, ge=1, le=5)
    Restaurant_latitude: float = 12.914264
    Restaurant_longitude: float = 77.678400
    Delivery_location_latitude: float = 12.994264
    Delivery_location_longitude: float = 77.758400
    Order_Date: str = Field("25-03-2022", description="dd-mm-yyyy")
    Time_Orderd: str = Field("19:45", description="HH:MM")
    Time_Order_picked: str = Field("19:55", description="HH:MM")
    Weatherconditions: str = Field("Sunny", description="Sunny, Cloudy, Fog, Windy, Stormy, Sandstorms")
    Road_traffic_density: str = Field("Jam", description="Low, Medium, High, Jam")
    Vehicle_condition: int = Field(2, ge=0, le=3)
    Type_of_order: str = Field("Meal", description="Snack, Meal, Drinks, Buffet")
    Type_of_vehicle: str = Field("motorcycle", description="motorcycle, scooter, electric_scooter")
    multiple_deliveries: int = Field(1, ge=0, le=3)
    Festival: str = Field("No", description="Yes or No")
    City: str = Field("Metropolitian", description="Metropolitian, Urban, Semi-Urban")


@app.get("/")
def home():
    return {"message": "Delivery Time Prediction API is running",
            "model": load_artifacts()["model_name"], "docs": "/docs"}


@app.post("/predict")
def predict(order: Order):
    minutes = predict_order(order.model_dump())
    return {"predicted_minutes": minutes, "model": load_artifacts()["model_name"]}
