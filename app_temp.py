import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import tempfile

# Load models and scaler
def load_model():
    return joblib.load('xgboost_model.pkl'), joblib.load('scaler.pkl')

model, scaler = load_model()

def predict(data):
    input_df = pd.DataFrame([data])

    st.write("Raw Input Data:", input_df)

    # One-hot encode `occupation_type` and ensure column names are strings
    occupation_encoded = pd.get_dummies(input_df["occupation_type"], prefix="occupation")
    occupation_encoded.columns = occupation_encoded.columns.astype(str)
    input_df = pd.concat([input_df.drop(columns=["occupation_type"]), occupation_encoded], axis=1)

    # Align columns with model and ensure column names are strings
    missing_cols = [col for col in model.feature_names_in_ if col not in input_df.columns]
    for col in missing_cols:
        input_df[col] = 0
    input_df = input_df[model.feature_names_in_]
    input_df.columns = input_df.columns.astype(str)

    # Apply scaling and convert back to DataFrame
    input_df_scaled = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)

    # Get prediction probability
    probability = model.predict_proba(input_df_scaled)[0][1]
    threshold = 0.5  # Adjust threshold if needed
    prediction = "Not Eligible for Loan" if probability >= threshold else "Eligible for Loan"
    return prediction, probability, input_df_scaled


# Add SHAP visualization
def explain_prediction(input_df):
    explainer = shap.TreeExplainer(model)

    # Get the SHAP values as an Explanation object
    shap_values = explainer(input_df)

    st.subheader("Feature Contribution to Prediction (Waterfall Plot)")

    # Use the waterfall plot with the Explanation object
    plt.figure()
    shap.waterfall_plot(shap_values[0])
    st.pyplot(plt)

    # Display the SHAP summary plot using Matplotlib
    st.subheader("Feature Importance (Summary)")
    plt.figure()
    shap.summary_plot(shap_values, input_df, plot_type="bar", show=False)
    st.pyplot(plt)



# UI
st.title("Loan Eligibility Prediction")
with st.form("loan_form"):
    age = st.number_input("Age", min_value=18, max_value=100, value=30, help="Enter your current age in years.")
    gender = st.selectbox("Gender", ["M", "F"])
    owns_car = st.selectbox("Owns a Car?", ["Y", "N"])
    owns_house = st.selectbox("Owns a House?", ["Y", "N"])
    no_of_children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
    net_yearly_income = st.number_input("Net Yearly Income", min_value=1000.0, value=50000.0)
    no_of_days_employed = st.number_input("Number of Days Employed", min_value=0, value=1000)
    occupation_type = st.selectbox("Occupation Type", ["Unknown", "Laborers", "Sales staff", "Core staff", 
                                                       "Managers", "Drivers", "High skill tech staff",
                                                       "Accountants", "Medicine staff", "Security staff",
                                                       "Cooking staff", "Cleaning staff", "Private service staff",
                                                       "Low-skill Laborers", "Secretaries", "Waiters/barmen staff",
                                                       "Realty agents", "HR staff", "IT staff"])
    total_family_members = st.number_input("Total Family Members", min_value=1, max_value=20, value=2)
    migrant_worker = st.selectbox("Migrant Worker", ["Yes", "No"])
    yearly_debt_payments = st.number_input("Yearly Debt Payments", min_value=0.0, value=10000.0)
    credit_limit = st.number_input("Credit Limit", min_value=0.0, value=50000.0)
    credit_score = st.number_input("Credit Score", min_value=300.0, max_value=850.0, value=650.0)
    prev_defaults = st.number_input("Previous Defaults", min_value=0, max_value=10, value=0)
    default_in_last_6months = st.selectbox("Default in Last 6 Months?", ["Yes", "No"])
    
    submitted = st.form_submit_button("Predict")
    
    if submitted:
        user_data = {
            "age": age,
            "gender": 1 if gender == "F" else 0,
            "owns_car": 1 if owns_car == "Y" else 0,
            "owns_house": 1 if owns_house == "Y" else 0,
            "no_of_children": no_of_children,
            "net_yearly_income": net_yearly_income,
            "no_of_days_employed": no_of_days_employed,
            "occupation_type": occupation_type,
            "total_family_members": total_family_members,
            "migrant_worker": 1 if migrant_worker == "Yes" else 0,
            "yearly_debt_payments": yearly_debt_payments,
            "credit_limit": credit_limit,
            "credit_score": credit_score,
            "prev_defaults": prev_defaults,
            "default_in_last_6months": 1 if default_in_last_6months == "Yes" else 0,
        }
        
        prediction, probability, input_df = predict(user_data)
        st.success(f"Prediction: {prediction}")
        st.info(f"Probability of Default: {probability:.2f}")

        # Explain the prediction
        explain_prediction(input_df)
