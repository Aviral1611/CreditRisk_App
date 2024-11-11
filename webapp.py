import streamlit as st
import pandas as pd
import pickle

# Load the trained Logistic Regression model
with open('logistic_model.pkl', 'rb') as model_file:
    logistic_model = pickle.load(model_file)

# Streamlit app layout
st.title("Credit Risk Loan Eligibility Predictor")
st.write("Provide your details (already transformed using WoE) to check loan eligibility.")

# Collect WoE-transformed user input
age_woe = st.number_input("WoE Value for Age", value=0.0)
income_woe = st.number_input("WoE Value for Income", value=0.0)
loan_amount_woe = st.number_input("WoE Value for Loan Amount", value=0.0)
credit_score_woe = st.number_input("WoE Value for Credit Score", value=0.0)
employment_status_woe = st.number_input("WoE Value for Employment Status", value=0.0)

# Create a DataFrame for prediction
input_data = pd.DataFrame([[age_woe, income_woe, loan_amount_woe, credit_score_woe, employment_status_woe]],
                          columns=['age', 'income', 'loan_amount', 'credit_score', 'employment_status'])

# Make a prediction
if st.button("Check Eligibility"):
    prediction = logistic_model.predict(input_data)
    prediction_proba = logistic_model.predict_proba(input_data)[0][1]

    if prediction[0] == 1:
        st.success(f"Loan Approved! Probability of default: {prediction_proba:.2%}")
    else:
        st.error(f"Loan Denied! Probability of default: {prediction_proba:.2%}")
