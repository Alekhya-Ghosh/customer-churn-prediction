# Customer Churn Prediction Model

**Tools:** Python · Pandas · Scikit-learn · Plotly  
**Domain:** Telecom / Subscription Business  
**Type:** Binary Classification · Machine Learning

---

## Project Overview

Customer churn is one of the most costly problems for subscription-based businesses. This project builds a machine learning pipeline to predict which customers are likely to churn, enabling retention teams to intervene early with targeted strategies.

The analysis covers the full ML workflow — data exploration, feature engineering, model training, evaluation, and business interpretation of results.

---

## Key Findings

| Metric | Value |
|--------|-------|
| Dataset size | 10,000 customer records |
| Overall churn rate | 24.9% |
| Logistic Regression AUC-ROC | 0.73 |
| Random Forest AUC-ROC | 0.72 |
| Highest churn segment | Month-to-Month contracts (36.5%) |
| Lowest churn segment | Two Year contracts (7.4%) |

---

## Top Churn Drivers

1. Total charges and monthly charges
2. Tenure (newer customers churn more)
3. Contract type (Month-to-Month highest risk)
4. Number of support calls
5. Internet service type

---

## Business Recommendations

- **Prioritise Month-to-Month customers** for retention campaigns — 36.5% churn rate vs 7.4% on two-year contracts
- **Target early-tenure customers** (0–12 months) with onboarding support — highest churn window
- **High support call volume is a red flag** — customers with 5+ calls are significantly more likely to churn
- **Encourage contract upgrades** — shifting customers from monthly to annual contracts reduces churn risk by ~29 percentage points

---

## Project Structure

```
customer-churn-prediction/
├── data/
│   └── customer_churn.csv       # 10,000 customer records
├── outputs/
│   ├── 01_churn_distribution.html
│   ├── 02_churn_by_contract.html
│   ├── 03_churn_by_tenure.html
│   ├── 04_charges_vs_churn.html
│   ├── 05_roc_curve.html
│   ├── 06_feature_importance.html
│   └── 07_confusion_matrix.html
├── generate_data.py              # Dataset generation script
├── churn_analysis.py             # Main analysis and modelling script
└── README.md
```

---

## How to Run

```bash
pip install pandas numpy scikit-learn plotly
python generate_data.py
python churn_analysis.py
```

Charts are saved as interactive HTML files in the `outputs/` folder.

---

## Skills Demonstrated

- Exploratory data analysis and visualisation
- Feature engineering and encoding
- Binary classification with Logistic Regression and Random Forest
- Model evaluation using AUC-ROC, confusion matrix, and classification report
- Translating model outputs into actionable business recommendations

---

*Built by Alekhya Ghosh | [LinkedIn](https://linkedin.com/in/alekhya-ghosh-0a9963248)
