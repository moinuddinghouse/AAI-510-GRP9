import streamlit as st
import joblib
import pandas as pd

# Load model and scaler
model = joblib.load("loan_default_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Loan Default Prediction")

# Numerical Inputs
loan_amnt = st.number_input("Loan Amount", min_value=0.0)
int_rate = st.number_input("Interest Rate", min_value=0.0)
annual_inc = st.number_input("Annual Income", min_value=0.0)
dti = st.number_input("Debt-to-Income Ratio", min_value=0.0)
fico_range_high = st.number_input("FICO Score", min_value=300, max_value=850)
revol_util = st.number_input("Revolving Utilization", min_value=0.0)
installment = st.number_input("Installment Amount", min_value=0.0)

# Categorical Inputs
term = st.selectbox("Loan Term", ["36 months", "60 months"])

home_ownership = st.selectbox(
    "Home Ownership",
    ["RENT", "OWN", "MORTGAGE", "OTHER"]
)

purpose = st.selectbox(
    "Loan Purpose",
    [
        "debt_consolidation",
        "credit_card",
        "home_improvement",
        "major_purchase",
        "small_business",
        "other"
    ]
)

emp_length = st.selectbox(
    "Employment Length",
    [
        "< 1 year",
        "1 year",
        "2 years",
        "3 years",
        "4 years",
        "5 years",
        "6 years",
        "7 years",
        "8 years",
        "9 years",
        "10+ years"
    ]
)

addr_state = st.text_input("State (Example: CA, FL, TX)")

if st.button("Predict"):

    # NOTE:
    # These encodings MUST match the LabelEncoder mappings
    # used during training.
    # Replace these with your actual mappings.

    term_map = {"36 months": 0, "60 months": 1}

    home_map = {
        "RENT": 0,
        "OWN": 1,
        "MORTGAGE": 2,
        "OTHER": 3
    }

    data = pd.DataFrame({
    "loan_amnt": [loan_amnt],
    "term": [term_map.get(term, 0)],
    "int_rate": [int_rate],
    "installment": [installment],
    "emp_length": [0],          # replace later with actual encoding
    "home_ownership": [home_map.get(home_ownership, 0)],
    "annual_inc": [annual_inc],
    "purpose": [0],             # replace later with actual encoding
    "addr_state": [0],          # replace later with actual encoding
    "dti": [dti],
    "fico_range_high": [fico_range_high],
    "revol_util": [revol_util]
})

    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)

    if prediction[0] == 1:
        st.error("High Risk of Default")
    else:
        st.success("Likely to Fully Pay")