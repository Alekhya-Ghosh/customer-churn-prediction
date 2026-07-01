import pandas as pd
import numpy as np

np.random.seed(42)
n = 10000

age = np.random.randint(18, 70, n)
tenure = np.random.randint(1, 72, n)
monthly_charges = np.round(np.random.uniform(20, 120, n), 2)
total_charges = np.round(monthly_charges * tenure * np.random.uniform(0.85, 1.05, n), 2)
num_products = np.random.randint(1, 6, n)
support_calls = np.random.randint(0, 10, n)
contract = np.random.choice(["Month-to-Month", "One Year", "Two Year"], n, p=[0.55, 0.25, 0.20])
payment_method = np.random.choice(["Electronic Check", "Mailed Check", "Bank Transfer", "Credit Card"], n)
internet_service = np.random.choice(["DSL", "Fiber Optic", "No"], n, p=[0.35, 0.45, 0.20])
gender = np.random.choice(["Male", "Female"], n)
senior_citizen = np.random.choice([0, 1], n, p=[0.84, 0.16])
partner = np.random.choice(["Yes", "No"], n)
dependents = np.random.choice(["Yes", "No"], n)
paperless_billing = np.random.choice(["Yes", "No"], n)
tech_support = np.random.choice(["Yes", "No", "No internet service"], n, p=[0.29, 0.49, 0.22])
online_security = np.random.choice(["Yes", "No", "No internet service"], n, p=[0.28, 0.50, 0.22])

# Churn probability based on realistic business logic
churn_prob = (
    0.05
    + 0.25 * (contract == "Month-to-Month")
    - 0.12 * (contract == "Two Year")
    + 0.15 * (tenure < 12)
    - 0.10 * (tenure > 48)
    + 0.12 * (support_calls > 5)
    + 0.08 * (monthly_charges > 80)
    - 0.06 * (num_products >= 4)
    + 0.07 * (internet_service == "Fiber Optic")
    + 0.05 * (payment_method == "Electronic Check")
    - 0.04 * (tech_support == "Yes")
    - 0.03 * (online_security == "Yes")
    + 0.04 * (senior_citizen == 1)
)
churn_prob = np.clip(churn_prob, 0.02, 0.92)
churn = (np.random.rand(n) < churn_prob).astype(int)

df = pd.DataFrame({
    "customer_id": [f"CUST{str(i).zfill(5)}" for i in range(1, n+1)],
    "gender": gender,
    "senior_citizen": senior_citizen,
    "partner": partner,
    "dependents": dependents,
    "age": age,
    "tenure_months": tenure,
    "contract_type": contract,
    "payment_method": payment_method,
    "paperless_billing": paperless_billing,
    "internet_service": internet_service,
    "online_security": online_security,
    "tech_support": tech_support,
    "num_products": num_products,
    "support_calls": support_calls,
    "monthly_charges": monthly_charges,
    "total_charges": total_charges,
    "churn": churn
})

df.to_csv("data/customer_churn.csv", index=False)
print(f"Dataset created: {len(df)} records")
print(f"Churn rate: {df['churn'].mean()*100:.1f}%")
print(df.head())
