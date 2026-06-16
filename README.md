# Loan Default Prediction Using Machine Learning

## Project Overview

This project develops a machine learning framework for predicting loan defaults using the Lending Club dataset. The objective is to classify borrowers as either likely to fully repay a loan or likely to default based on financial, credit, employment, and loan-related characteristics.

The project includes data preprocessing, exploratory data analysis, feature engineering, class balancing using SMOTE, model development, explainability analysis, and deployment using Streamlit.

---

## Dataset

Source: Lending Club Loan Data

The dataset contains borrower information, loan characteristics, credit history, and loan repayment outcomes.

Target Variable:

* 0 = Fully Paid
* 1 = Charged Off

---

## Features Used

* loan_amnt
* term
* int_rate
* installment
* emp_length
* home_ownership
* annual_inc
* purpose
* addr_state
* dti
* fico_range_high
* revol_util

---

## Machine Learning Models

The following classification models were developed and evaluated:

* Logistic Regression
* Decision Tree
* Random Forest
* XGBoost

Class imbalance was addressed using SMOTE (Synthetic Minority Oversampling Technique).

---

## Model Performance

| Model               | ROC-AUC |
| ------------------- | ------- |
| Logistic Regression | 0.6991  |
| XGBoost             | 0.6925  |
| Random Forest       | 0.6876  |
| Decision Tree       | 0.6503  |

Logistic Regression achieved the highest ROC-AUC score and was selected as the final model.

---

## Explainability

The project includes:

* Feature Importance Analysis
* SHAP (SHapley Additive Explanations)

These techniques help explain model predictions and improve transparency in credit risk assessment.

---

## Deployment

The final Logistic Regression model was saved using Joblib and deployed through a Streamlit web application.

Files:

* loan_default_model.pkl
* scaler.pkl
* app.py

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd Loan-Default-Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Imbalanced-Learn (SMOTE)
* XGBoost
* SHAP
* Streamlit
* Joblib
* Matplotlib

---

## Author

Moinuddin K

MS in Artificial Intelligence
