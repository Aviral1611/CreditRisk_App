import numpy as np
import pandas as pd

class CreditScoreEvaluator:
    def __init__(self):
        # Define weights for different factor categories (total = 100%)
        self.weights = {
            'payment_history': 0.30,  # Payment history (defaults, timely payments)
            'credit_utilization': 0.25,  # Credit utilization and limits
            'financial_stability': 0.25,  # Income and employment
            'personal_factors': 0.20,  # Personal factors (age, housing, etc)
        }
        
    def normalize_score(self, value, min_val, max_val):
        """Normalize a value between 0 and 1"""
        if max_val - min_val == 0:
            return 0
        return (value - min_val) / (max_val - min_val)
    
    def calculate_payment_history_score(self, data):
        """Calculate payment history score"""
        credit_score_norm = self.normalize_score(data['credit_score'], 300, 850)
        prev_defaults_norm = 1 - self.normalize_score(data['prev_defaults'], 0, 10)
        recent_default_impact = 0 if data['default_in_last_6months'] else 1
        
        return (0.4 * credit_score_norm + 
                0.3 * prev_defaults_norm + 
                0.3 * recent_default_impact)
    
    def calculate_credit_utilization_score(self, data):
        """Calculate credit utilization score"""
        if data['credit_limit'] == 0:
            utilization_ratio = 1
        else:
            utilization_ratio = min(data['yearly_debt_payments'] / data['credit_limit'], 1)
        utilization_score = 1 - utilization_ratio
        
        return utilization_score
    
    def calculate_financial_stability_score(self, data):
        """Calculate financial stability score"""
        income_score = self.normalize_score(data['net_yearly_income'], 1000, 300000)
        employment_score = self.normalize_score(data['no_of_days_employed'], 0, 10000)
        
        return (0.6 * income_score + 0.4 * employment_score)
    
    def calculate_personal_factors_score(self, data):
        """Calculate personal factors score"""
        age_score = self.normalize_score(data['age'], 18, 100)
        stability_score = (0.5 if data['owns_house'] else 0) + (0.3 if data['owns_car'] else 0)
        
        return (0.4 * age_score + 0.6 * stability_score)
    
    def predict_eligibility(self, data):
        """Calculate final eligibility score and make prediction"""
        # Calculate component scores
        payment_history = self.calculate_payment_history_score(data)
        credit_utilization = self.calculate_credit_utilization_score(data)
        financial_stability = self.calculate_financial_stability_score(data)
        personal_factors = self.calculate_personal_factors_score(data)
        
        # Calculate weighted final score
        final_score = (
            self.weights['payment_history'] * payment_history +
            self.weights['credit_utilization'] * credit_utilization +
            self.weights['financial_stability'] * financial_stability +
            self.weights['personal_factors'] * personal_factors
        )
        
        # Calculate prediction probability
        probability = final_score
        
        # Generate detailed report
        report = {
            'final_score': final_score,
            'probability': probability,
            'component_scores': {
                'payment_history': payment_history,
                'credit_utilization': credit_utilization,
                'financial_stability': financial_stability,
                'personal_factors': personal_factors
            }
        }
        
        return report

def get_loan_decision(report):
    """Determine loan eligibility based on the report"""
    probability = report['probability']
    
    if probability >= 0.7:
        return "Eligible for Loan", "High confidence"
    elif probability >= 0.5:
        return "Eligible for Loan", "Medium confidence"
    elif probability >= 0.3:
        return "Not Eligible for Loan", "Medium confidence"
    else:
        return "Not Eligible for Loan", "High confidence"