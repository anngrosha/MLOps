from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import numpy as np
import os

model_path = os.path.join('/app/models', 'model.pkl')
model = joblib.load(model_path)

app = FastAPI()

class LoanInfo(BaseModel):
    no_of_dependents: float
    education: float
    self_employed: float
    income_annum: float
    loan_amount: float
    loan_term: float
    cibil_score: float
    residential_assets_value: float
    commercial_assets_value: float
    luxury_assets_value: float
    bank_asset_value: float

@app.post("/predict")
def predict(input_data: LoanInfo):
    data = [[
        input_data.no_of_dependents, 
        input_data.education, 
        input_data.self_employed,
        input_data.income_annum,
        input_data.loan_amount,
        input_data.loan_term, 
        input_data.cibil_score,
        input_data.residential_assets_value,
        input_data.commercial_assets_value,
        input_data.luxury_assets_value,
        input_data.bank_asset_value
    ]]

    prediction = model.predict(data)
    return {"prediction": int(prediction[0])}
