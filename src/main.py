from fastapi import FastAPI
import pandas as pd
import joblib
from src.ml import model
from src.ml.data import process_data
from pydantic import BaseModel, Field
from typing import List


app = FastAPI()
lr_model = joblib.load("src/model/lr_model.joblib")
encoder = joblib.load("src/model/encoder.joblib")
lb = joblib.load("src/model/label_binarizer.joblib")

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

class InputData(BaseModel):
    age: int
    workclass: str
    fnlwgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str


# Define route for root
@app.get("/")
async def root():
    return {"message": "Predict if salary is greater or lower 50k based on census data"}


# Define route for model inference
@app.post("/predict/")
async def predict(data: List[InputData]):
    df = pd.DataFrame([item.model_dump() for item in data])
    print(df)
    input_data, _, _, _ = process_data(df, cat_features, label=None, encoder=encoder, lb=lb, training=False)
    # Perform inference
    predictions = model.inference(lr_model,input_data)
    return {"predictions": predictions.tolist()}


