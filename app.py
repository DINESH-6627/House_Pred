from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="House Price Prediction API")

model = joblib.load("house_price_model.pkl")

class HouseInput(BaseModel):
    area: int
    bedrooms: int
    bathrooms: int

@app.post("/predict")
def predict_price(data: HouseInput):
    features = np.array([[data.area, data.bedrooms, data.bathrooms]])
    prediction = model.predict(features)[0]

    return {
        "predicted_price_lakhs": round(float(prediction), 2)
    }
