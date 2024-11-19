import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained Naive Bayes model and scaler
nb_model = joblib.load('naive_bayes_model.pkl')  # Saved Naive Bayes model
scaler = joblib.load('scaler.pkl')  # Saved scaler used during training

# Function to make predictions
def predict_nb(data, trained_columns):
    input_df = pd.DataFrame([data])

    # Display raw input for debugging
    st.write("Raw Input Data:", input_df)

    # One-hot encode `occupation_type`
    occupation_encoded = pd.get_dummies(input_df["occupation_type"], prefix="occupation")
    input_df = pd.concat([input_df.drop(columns=["occupation_type"]), occupation_encoded], axis=1)

    # Align columns dynamically with the model's training data
    for col in trained_columns:
        if col not in input_df.columns:
            input_df[col] = 0  # Add missing columns with default value
    input_df = input_df[trained_columns]  # Ensure column order matches the training phase

    # Apply scaling
    input_df_scaled = scaler.transform(input_df)

    # Make predictions
    prediction = nb_model.predict(input_df_scaled)[0]
    probability = nb_model.predict_proba(input_df_scaled)[0][1]
    return prediction, probability

# Streamlit App UI
st.title("Loan Eligibility Prediction (Naive Bayes Classifier)")

# Collect user input
with st.form("loan_form"):
    age = st.number_input("Age", min_value=18, max_value=100, value=30, help="Enter your age in years.")
    gender = st.selectbox("Gender", ["M", "F"])
    owns_car = st.selectbox("Owns a Car?", ["Y", "N"])
    owns_house = st.selectbox("Owns a House?", ["Y", "N"])
    no_of_children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
    net_yearly_income = st.number_input("Net Yearly Income", min_value=1000.0, value=50000.0)
    no_of_days_employed = st.number_input("Number of Days Employed", min_value=0, value=1000)
    occupation_type = st.selectbox("Occupation Type", [
        "Unknown", "Laborers", "Sales staff", "Core staff", "Managers",
        "Drivers", "High skill tech staff", "Accountants", "Medicine staff",
        "Security staff", "Cooking staff", "Cleaning staff", "Private service staff",
        "Low-skill Laborers", "Secretaries", "Waiters/barmen staff", "Realty agents", "HR staff", "IT staff"
    ])
    total_family_members = st.number_input("Total Family Members", min_value=1, max_value=20, value=2)
    migrant_worker = st.selectbox("Migrant Worker", ["Yes", "No"])
    yearly_debt_payments = st.number_input("Yearly Debt Payments", min_value=0.0, value=10000.0, max_value=500000.0)
    credit_limit = st.number_input("Credit Limit", min_value=0.0, value=5000.0, max_value=100000.0)
    credit_limit_used_percent = st.number_input("Credit Limit Used (%)", min_value=0.0, max_value=100.0, value=50.0)
    credit_score = st.number_input("Credit Score", min_value=300.0, max_value=850.0, value=650.0)
    prev_defaults = st.number_input("Previous Defaults", min_value=0, max_value=10, value=0)
    default_in_last_6months = st.selectbox("Default in Last 6 Months?", ["Yes", "No"])

    submitted = st.form_submit_button("Predict")

    if submitted:
        # Prepare the user data dictionary
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
            "credit_limit_used(%)": credit_limit_used_percent,
            "credit_score": credit_score,
            "prev_defaults": prev_defaults,
            "default_in_last_6months": 1 if default_in_last_6months == "Yes" else 0,
        }

        # Get the columns from the trained model
        trained_columns = nb_model.feature_names_in_

        # Make predictions
        prediction, probability = predict_nb(user_data, trained_columns)
        result = "Eligible for Loan" if prediction == 0 else "Not Eligible for Loan"
        st.success(f"Prediction: {result}")
        st.info(f"Probability of Default: {probability:.2f}")

st.markdown("---")
st.markdown(
    """
    <p style="text-align: center;">Developed by <strong>Aviral Bansal</strong></p>
    <p style="text-align: center;">
        <a href="https://github.com/Aviral1611" target="_blank">
            <img src="https://img.icons8.com/ios-glyphs/30/000000/github.png"/>
        </a>
    </p>
    """, unsafe_allow_html=True
)
