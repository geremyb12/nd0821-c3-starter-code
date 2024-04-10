from fastapi import FastAPI
import pandas as pd
import joblib
from src.ml.data import process_data
from pydantic import BaseModel, Field
from typing import List

app = FastAPI()

cat_features = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]


class Input(BaseModel):
    age: int = Field(..., example=35)
    capital_gain: int = Field(..., example=1500)
    capital_loss: int = Field(..., example=100)
    education: str = Field(..., example="HS-grad")
    education_num: int = Field(..., example=9)
    fnlgt: int = Field(..., example=200000)
    hours_per_week: int = Field(..., example=40)
    marital_status: str = Field(..., example="Married-civ-spouse")
    native_country: str = Field(..., example="United-States")
    occupation: str = Field(..., example="Exec-managerial")
    race: str = Field(..., example="White")
    relationship: str = Field(..., example="Husband")
    sex: str = Field(..., example="Male")
    workclass: str = Field(..., example="Private")


# Define route for root
@app.get("/")
async def root():
    return {"message": "Predict if salary is greater or lower 50k based on census data"}


# Define route for model inference
@app.post("/predict/")
async def predict(data: List[Input]):
    df = pd.DataFrame([item.dict() for item in data])
    input_data, _, _, _ = process_data(df, cat_features, label=None, encoder=encoder, lb=lb, training=False)
    # Perform inference
    predictions = lr_model.inference(lr_model, input_data)
    return {"predictions": predictions.tolist()}


@app.on_event("startup")
async def startup_event():
    global lr_model, encoder, lb
    lr_model = joblib.load("src/model/lr_model.joblib")
    encoder = joblib.load("src/model/encoder.joblib")
    lb = joblib.load("src/model/label_binarizer.joblib")
