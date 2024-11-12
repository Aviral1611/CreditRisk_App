CreditRisk_Analysis
This repository contains multiple Streamlit applications for predicting loan eligibility using different machine learning models. Each application utilizes a distinct model to make predictions based on user input.

Applications
1. Loan Eligibility Prediction (XGBoost)
File: app.py
Model: XGBoost
Description: This application uses an XGBoost model to predict loan eligibility. Users can enter their personal and financial details to receive a prediction.
2. Loan Eligibility Prediction (CatBoost)
File: app2.py
Model: CatBoost
Description: This application uses a CatBoost model to predict loan eligibility. Users can enter their personal and financial details to receive a prediction.
3. Loan Eligibility Prediction (KNN)
File: app3.py
Model: K-Nearest Neighbors (KNN)
Description: This application uses a KNN model to predict loan eligibility. Users can enter their personal and financial details to receive a prediction.
How to Run
1. Install Dependencies
Ensure Python is installed. Then, install the required packages using pip:

sh
Copy code
pip install streamlit pandas joblib numpy scikit-learn catboost
2. Run the Application
Use Streamlit to run the desired application. For example, to run the XGBoost application:

sh
Copy code
streamlit run app.py
Similarly, you can run the other applications:

sh
Copy code
streamlit run app2.py
streamlit run app3.py
User Input
Each application collects the following user inputs:

Age
Gender
Owns a Car
Owns a House
Number of Children
Net Yearly Income
Number of Days Employed
Occupation Type
Total Family Members
Migrant Worker
Yearly Debt Payments
Credit Limit
Credit Score
Previous Defaults
Default in Last 6 Months
Prediction
Each application displays the prediction result as either:

"Eligible for Loan"
"Not Eligible for Loan"
based on the input data.

Models
XGBoost: A powerful gradient boosting framework.
CatBoost: A gradient boosting framework well-suited for categorical features.
KNN: A simple, instance-based learning algorithm.
License
This project is licensed under the MIT License.