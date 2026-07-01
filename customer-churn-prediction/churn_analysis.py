"""
Customer Churn Prediction
Author: Alekhya Ghosh
Description: End-to-end churn prediction pipeline covering EDA,
             feature engineering, model training, and evaluation.
             Models: Logistic Regression + Random Forest
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, classification_report,
                             confusion_matrix, roc_curve)
import os, warnings
warnings.filterwarnings("ignore")

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("CUSTOMER CHURN PREDICTION")
print("=" * 60)

# ── LOAD DATA ─────────────────────────────────────────────────
df = pd.read_csv("data/customer_churn.csv")
print(f"\nDataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Churn rate: {df['churn'].mean()*100:.1f}%")
print(f"Missing values: {df.isnull().sum().sum()}")

# ── 1. CHURN DISTRIBUTION ─────────────────────────────────────
churn_counts = df['churn'].value_counts().reset_index()
churn_counts.columns = ['churn', 'count']
churn_counts['label'] = churn_counts['churn'].map({0: 'Retained', 1: 'Churned'})
churn_counts['pct'] = (churn_counts['count'] / len(df) * 100).round(1)

fig1 = px.pie(churn_counts, values='count', names='label',
              title='Customer Churn Distribution',
              color_discrete_map={'Retained': '#2a7c6f', 'Churned': '#c45c2a'})
fig1.update_traces(textinfo='percent+label+value')
fig1.update_layout(template='plotly_white')
fig1.write_html(f"{OUTPUT_DIR}/01_churn_distribution.html")

# ── 2. CHURN BY CONTRACT TYPE ─────────────────────────────────
contract_churn = df.groupby('contract_type')['churn'].agg(['sum','count']).reset_index()
contract_churn.columns = ['contract_type', 'churned', 'total']
contract_churn['churn_rate'] = (contract_churn['churned'] / contract_churn['total'] * 100).round(1)

fig2 = px.bar(contract_churn, x='contract_type', y='churn_rate',
              color='churn_rate', color_continuous_scale='Oranges',
              title='Churn Rate by Contract Type (%)',
              labels={'churn_rate': 'Churn Rate (%)', 'contract_type': 'Contract Type'},
              text='churn_rate')
fig2.update_traces(texttemplate='%{text}%', textposition='outside')
fig2.update_layout(template='plotly_white', coloraxis_showscale=False)
fig2.write_html(f"{OUTPUT_DIR}/02_churn_by_contract.html")
print("\n[2] Churn by Contract:")
print(contract_churn.to_string(index=False))

# ── 3. CHURN BY TENURE ────────────────────────────────────────
df['tenure_band'] = pd.cut(df['tenure_months'],
                            bins=[0,12,24,36,48,72],
                            labels=['0-12m','13-24m','25-36m','37-48m','49-72m'])
tenure_churn = df.groupby('tenure_band', observed=True)['churn'].mean().reset_index()
tenure_churn['churn_rate'] = (tenure_churn['churn'] * 100).round(1)

fig3 = px.line(tenure_churn, x='tenure_band', y='churn_rate',
               markers=True, title='Churn Rate by Customer Tenure',
               labels={'churn_rate': 'Churn Rate (%)', 'tenure_band': 'Tenure Band'})
fig3.update_traces(line_color='#c45c2a', marker_size=8)
fig3.update_layout(template='plotly_white')
fig3.write_html(f"{OUTPUT_DIR}/03_churn_by_tenure.html")

# ── 4. MONTHLY CHARGES VS CHURN ───────────────────────────────
fig4 = px.box(df, x=df['churn'].map({0:'Retained',1:'Churned'}),
              y='monthly_charges', color=df['churn'].map({0:'Retained',1:'Churned'}),
              title='Monthly Charges Distribution: Churned vs Retained',
              labels={'x': 'Customer Status', 'monthly_charges': 'Monthly Charges (USD)'},
              color_discrete_map={'Retained': '#2a7c6f', 'Churned': '#c45c2a'})
fig4.update_layout(template='plotly_white', showlegend=False)
fig4.write_html(f"{OUTPUT_DIR}/04_charges_vs_churn.html")

# ── 5. FEATURE ENGINEERING ────────────────────────────────────
print("\n[5] Preparing features...")
df_model = df.drop(columns=['customer_id', 'tenure_band'])

# Encode categorical columns
le = LabelEncoder()
cat_cols = df_model.select_dtypes(include='object').columns
for col in cat_cols:
    df_model[col] = le.fit_transform(df_model[col])

X = df_model.drop(columns=['churn'])
y = df_model['churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

print(f"   Training set: {X_train.shape[0]:,} records")
print(f"   Test set:     {X_test.shape[0]:,} records")

# ── 6. LOGISTIC REGRESSION ────────────────────────────────────
print("\n[6] Training Logistic Regression...")
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train_sc, y_train)
lr_probs = lr.predict_proba(X_test_sc)[:, 1]
lr_preds = lr.predict(X_test_sc)
lr_auc = roc_auc_score(y_test, lr_probs)
print(f"   AUC-ROC: {lr_auc:.4f}")
print(classification_report(y_test, lr_preds, target_names=['Retained','Churned']))

# ── 7. RANDOM FOREST ──────────────────────────────────────────
print("\n[7] Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_probs = rf.predict_proba(X_test)[:, 1]
rf_preds = rf.predict(X_test)
rf_auc = roc_auc_score(y_test, rf_probs)
print(f"   AUC-ROC: {rf_auc:.4f}")
print(classification_report(y_test, rf_preds, target_names=['Retained','Churned']))

# ── 8. ROC CURVE ──────────────────────────────────────────────
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)

fig8 = go.Figure()
fig8.add_trace(go.Scatter(x=lr_fpr, y=lr_tpr, mode='lines',
                           name=f'Logistic Regression (AUC={lr_auc:.2f})',
                           line=dict(color='#2a7c6f', width=2)))
fig8.add_trace(go.Scatter(x=rf_fpr, y=rf_tpr, mode='lines',
                           name=f'Random Forest (AUC={rf_auc:.2f})',
                           line=dict(color='#c45c2a', width=2)))
fig8.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
                           name='Random Baseline',
                           line=dict(color='gray', dash='dash')))
fig8.update_layout(title='ROC Curve Comparison',
                   xaxis_title='False Positive Rate',
                   yaxis_title='True Positive Rate',
                   template='plotly_white', legend=dict(x=0.6, y=0.1))
fig8.write_html(f"{OUTPUT_DIR}/05_roc_curve.html")

# ── 9. FEATURE IMPORTANCE ─────────────────────────────────────
feat_imp = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=True).tail(12)

fig9 = px.bar(feat_imp, x='importance', y='feature', orientation='h',
              title='Top 12 Churn Drivers (Random Forest Feature Importance)',
              labels={'importance': 'Importance Score', 'feature': 'Feature'},
              color='importance', color_continuous_scale='Oranges')
fig9.update_layout(template='plotly_white', coloraxis_showscale=False)
fig9.write_html(f"{OUTPUT_DIR}/06_feature_importance.html")

print("\n[9] Top Churn Drivers:")
print(feat_imp[::-1][['feature','importance']].to_string(index=False))

# ── 10. CONFUSION MATRIX ──────────────────────────────────────
cm = confusion_matrix(y_test, rf_preds)
fig10 = px.imshow(cm, text_auto=True,
                  labels=dict(x="Predicted", y="Actual"),
                  x=['Retained','Churned'], y=['Retained','Churned'],
                  title='Confusion Matrix — Random Forest',
                  color_continuous_scale='Oranges')
fig10.update_layout(template='plotly_white')
fig10.write_html(f"{OUTPUT_DIR}/07_confusion_matrix.html")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print(f"Logistic Regression AUC-ROC : {lr_auc:.4f}")
print(f"Random Forest AUC-ROC       : {rf_auc:.4f}")
print(f"Charts saved to /{OUTPUT_DIR}/")
print("=" * 60)
