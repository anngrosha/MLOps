import streamlit as st
import requests
import pandas as pd
import joblib
import os

def input_prep(data):
    data['education'] = 1 if data['education'] == 'Graduate' else 0
    data['self_employed'] = 1 if data['self_employed'] == 'Yes' else 0

    columns_to_normalize = [
        'no_of_dependents', 'income_annum', 'loan_amount', 
        'loan_term', 'cibil_score', 'residential_assets_value', 
        'commercial_assets_value', 'luxury_assets_value', 
        'bank_asset_value'
    ]
    
    input_df = pd.DataFrame([data])

    scaler_path = os.path.join('/app/models', 'scaler.pkl')
    scaler = joblib.load(scaler_path)
        
    input_df[columns_to_normalize] = scaler.transform(input_df[columns_to_normalize])

    processed_data = input_df.to_dict(orient='records')[0]
    
    return processed_data


FASTAPI_URL = "http://fastapi:80/predict"

st.title("Loan Prediction")

no_of_dependents = st.number_input("no_of_dependents", min_value=0)
education = st.selectbox("education", options=["Graduate", "Not Graduate"])
self_employed = st.selectbox("self_employed", options=["Yes", "No"])
income_annum = st.number_input("income_annum", min_value=0)
loan_amount = st.number_input("loan_amount", min_value=0)
loan_term = st.number_input("loan_term", min_value=0)
cibil_score = st.number_input("cibil_score", min_value=0)
residential_assets_value = st.number_input("residential_assets_value", min_value=0)
commercial_assets_value = st.number_input("commercial_assets_value", min_value=0)
luxury_assets_value = st.number_input("luxury_assets_value", min_value=0)
bank_asset_value = st.number_input("bank_asset_value", min_value=0)

if st.button("Predict"):

    input_data = {
        "no_of_dependents": no_of_dependents,
        "education": education,
        "self_employed": self_employed,
        "income_annum": income_annum,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "cibil_score": cibil_score,
        "residential_assets_value": residential_assets_value,
        "commercial_assets_value": commercial_assets_value,
        "luxury_assets_value": luxury_assets_value,
        "bank_asset_value": bank_asset_value
    }

    processed_input = input_prep(input_data)

    response = requests.post(FASTAPI_URL, json=processed_input)

    prediction = response.json()["prediction"]

    prediction_label = "Approved" if prediction == 1 else "Rejected"

    st.write(prediction_label)
