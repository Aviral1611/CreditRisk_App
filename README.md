# CreditRisk_Analysis

This repository contains multiple Streamlit applications for predicting loan eligibility using different machine learning models. Each application uses a distinct model to make predictions based on user input.

## Applications

### 1. Loan Eligibility Prediction (XGBoost)
- **File:** [`app.py`](app.py)
- **Model:** XGBoost
- **Description:** This application uses an XGBoost model to predict loan eligibility. Users can input their personal and financial details to get a prediction.

### 2. Loan Eligibility Prediction (CatBoost)
- **File:** [`app2.py`](app2.py)
- **Model:** CatBoost
- **Description:** This application uses a CatBoost model to predict loan eligibility. Users can input their personal and financial details to get a prediction.

### 3. Loan Eligibility Prediction (KNN)
- **File:** [`app3.py`](app3.py)
- **Model:** K-Nearest Neighbors (KNN)
- **Description:** This application uses a KNN model to predict loan eligibility. Users can input their personal and financial details to get a prediction.

### 4. Loan Eligibility Prediction (Random Forest)
- **File:** [`app4.py`](app4.py)
- **Model:** Random Forest
- **Description:** This application uses a Random Forest classifier to predict loan eligibility based on user inputs.

### 5. Loan Eligibility Prediction (Decision Tree)
- **File:** [`app5.py`](app5.py)
- **Model:** Decision Tree
- **Description:** This application uses a Decision Tree classifier to predict loan eligibility based on user inputs.

### 6. Loan Eligibility Prediction (Logistic Regression)
- **File:** [`app6.py`](app6.py)
- **Model:** Logistic Regression
- **Description:** This application uses Logistic Regression to predict loan eligibility based on user inputs.

### 7. Loan Eligibility Prediction (Naive Bayes)
- **File:** [`app7.py`](app7.py)
- **Model:** Naive Bayes
- **Description:** This application uses a Naive Bayes classifier to predict loan eligibility based on user inputs.

## How to Run


1. **Install Dependencies**  
   Ensure you have Python installed. Then install the required packages using pip:

   ```bash
   pip install streamlit pandas joblib numpy scikit-learn catboost


2. **Run the Application**
Use Streamlit to run the desired application. For example, to run the XGBoost application:

   ```bash
   streamlit run app.py

   ```

Similarly, you can run the other applications:

```bash
streamlit run app2.py
streamlit run app3.py
```

- **User Input**
- Each application collects the following user inputs:

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
Each application will display the prediction result as either:

- **"Eligible for Loan"**
-"Not Eligible for Loan"
-based on the input data.

- **XGBoost**: A powerful gradient boosting framework.  
- **CatBoost**: A gradient boosting framework that handles categorical features well.  
- **KNN**: A simple, instance-based learning algorithm.  
- **Random Forest**: An ensemble learning method based on decision trees.  
- **Decision Tree**: A tree-based classifier for decision-making.  
- **Logistic Regression**: A statistical model for binary classification.  
- **Naive Bayes**: A probabilistic classifier based on Bayes' theorem.  


-**License**
-This project is licensed under the MIT License