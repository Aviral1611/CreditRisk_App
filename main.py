import streamlit as st
import pandas as pd
from credit_scoring import CreditScoreEvaluator, get_loan_decision

# Initialize the credit score evaluator
evaluator = CreditScoreEvaluator()

# Streamlit App UI
st.title("Loan Eligibility Prediction")

# Create two columns for input fields
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    gender = st.selectbox("Gender", ["M", "F"])
    owns_car = st.selectbox("Owns a Car?", ["Y", "N"])
    owns_house = st.selectbox("Owns a House?", ["Y", "N"])
    no_of_children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
    net_yearly_income = st.number_input("Net Yearly Income", min_value=1000.0, value=50000.0)
    no_of_days_employed = st.number_input("Number of Days Employed", min_value=0, value=1000)

with col2:
    occupation_type = st.selectbox("Occupation Type", [
        "Unknown", "Laborers", "Sales staff", "Core staff", "Managers",
        "Drivers", "High skill tech staff", "Accountants", "Medicine staff",
        "Security staff", "Cooking staff", "Cleaning staff", "Private service staff",
        "Low-skill Laborers", "Secretaries", "Waiters/barmen staff", "Realty agents", 
        "HR staff", "IT staff"
    ])
    yearly_debt_payments = st.number_input("Yearly Debt Payments", min_value=0.0, value=10000.0)
    credit_limit = st.number_input("Credit Limit", min_value=0.0, value=50000.0)
    credit_score = st.number_input("Credit Score", min_value=300.0, max_value=850.0, value=650.0)
    prev_defaults = st.number_input("Previous Defaults", min_value=0, max_value=10, value=0)
    default_in_last_6months = st.selectbox("Default in Last 6 Months?", ["No", "Yes"])

# Prepare data for prediction
user_data = {
    "age": age,
    "gender": 1 if gender == "F" else 0,
    "owns_car": 1 if owns_car == "Y" else 0,
    "owns_house": 1 if owns_house == "Y" else 0,
    "no_of_children": no_of_children,
    "net_yearly_income": net_yearly_income,
    "no_of_days_employed": no_of_days_employed,
    "occupation_type": occupation_type,
    "yearly_debt_payments": yearly_debt_payments,
    "credit_limit": credit_limit,
    "credit_score": credit_score,
    "prev_defaults": prev_defaults,
    "default_in_last_6months": 1 if default_in_last_6months == "Yes" else 0
}

# Display prediction
if st.button("Predict"):
    # Get detailed report
    report = evaluator.predict_eligibility(user_data)
    decision, confidence = get_loan_decision(report)
    
    # Display results
    st.header("Loan Assessment Results")
    
    # Display the main decision
    if decision == "Eligible for Loan":
        st.success(f"Decision: {decision} ({confidence})")
    else:
        st.error(f"Decision: {decision} ({confidence})")
    
    # Display detailed scores
    st.subheader("Detailed Assessment")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Payment History", 
                 f"{report['component_scores']['payment_history']:.2%}")
    with col2:
        st.metric("Credit Utilization", 
                 f"{report['component_scores']['credit_utilization']:.2%}")
    with col3:
        st.metric("Financial Stability", 
                 f"{report['component_scores']['financial_stability']:.2%}")
    with col4:
        st.metric("Personal Factors", 
                 f"{report['component_scores']['personal_factors']:.2%}")
    
    # Display overall score
    st.subheader("Overall Score")
    st.progress(report['final_score'])
    st.write(f"Final Score: {report['final_score']:.2%}")