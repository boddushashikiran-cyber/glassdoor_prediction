import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ── Load saved artifacts ──────────────────────────────────────────────────────
model   = joblib.load('glassdoor_salary_gbr_model.pkl')
scaler  = joblib.load('glassdoor_robust_scaler.pkl')
le      = joblib.load('glassdoor_label_encoder.pkl')
feature_cols = joblib.load('glassdoor_feature_columns.pkl')

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Glassdoor Salary Predictor", page_icon="💼", layout="centered")
st.title("💼 Tech Job Salary Predictor")
st.markdown("Estimate a tech job's average salary based on role, company, and location — powered by Glassdoor data (2017-2018).")
st.markdown("---")

# ── Sidebar inputs ────────────────────────────────────────────────────────────
st.sidebar.header("🔧 Job Details")

job_category = st.sidebar.selectbox("Job Category", [
    'Data Scientist', 'Data Engineer', 'Data Analyst',
    'ML Engineer', 'Manager/Director', 'BI Analyst', 'Other'
])

sector = st.sidebar.selectbox("Sector", [
    'Information Technology', 'Finance', 'Biotech & Pharmaceuticals',
    'Business Services', 'Insurance', 'Health Care',
    'Manufacturing', 'Aerospace & Defense', 'Education', 'Other'
])

ownership = st.sidebar.selectbox("Type of Ownership", [
    'Company - Private', 'Company - Public', 'Nonprofit Organization',
    'Government', 'Hospital', 'Subsidiary or Business Segment',
    'College / University', 'Other Organization'
])

size = st.sidebar.selectbox("Company Size", [
    '1 to 50 employees', '51 to 200 employees', '201 to 500 employees',
    '501 to 1000 employees', '1001 to 5000 employees',
    '5001 to 10000 employees', '10000+ employees'
])

state = st.sidebar.selectbox("State", [
    'CA', 'NY', 'MA', 'IL', 'TX', 'WA', 'PA', 'VA', 'NJ', 'NC',
    'GA', 'OH', 'CO', 'FL', 'MD', 'MN', 'AZ', 'MI', 'OR', 'DC',
    'CT', 'UT', 'IN', 'TN', 'MO', 'WI', 'KS', 'SC', 'AL', 'Other'
])

rating      = st.sidebar.slider("Company Glassdoor Rating", 1.0, 5.0, 3.8, 0.1)
founded_year = st.sidebar.number_input("Year Company Founded", min_value=1800, max_value=2018, value=2000)
salary_source = st.sidebar.radio("Salary Estimate Source", ['Glassdoor', 'Employer'])

# ── Feature engineering ───────────────────────────────────────────────────────
size_order = {
    '1 to 50 employees': 1, '51 to 200 employees': 2,
    '201 to 500 employees': 3, '501 to 1000 employees': 4,
    '1001 to 5000 employees': 5, '5001 to 10000 employees': 6,
    '10000+ employees': 7
}
size_encoded  = size_order[size]
company_age   = 2018 - founded_year

# Encode state (use -1 for unseen states)
try:
    state_encoded = le.transform([state])[0]
except ValueError:
    state_encoded = 0

# ── Build raw input row matching training columns ─────────────────────────────
raw = {
    'Rating'       : rating,
    'size_encoded' : size_encoded,
    'company_age'  : company_age,
    'state_encoded': state_encoded,
}

# One-hot dummies matching training encoding
raw[f'job_category_{job_category}']     = 1
raw[f'Sector_{sector}']                 = 1
raw[f'Type of ownership_{ownership}']   = 1
raw[f'salary_source_{salary_source}']   = 1

input_df = pd.DataFrame([raw])

# Add any missing columns from training (fill 0)
for col in feature_cols:
    if col not in input_df.columns:
        input_df[col] = 0

# Reorder to match training
input_df = input_df[feature_cols]

# Scale numeric features
numeric_features = ['Rating', 'size_encoded', 'company_age', 'state_encoded']
input_df[numeric_features] = scaler.transform(input_df[numeric_features])

# ── Predict ───────────────────────────────────────────────────────────────────
if st.sidebar.button("🔍 Predict Salary"):
    pred_log = model.predict(input_df)[0]
    pred_salary = np.expm1(pred_log)

    st.markdown("## 💰 Predicted Salary")
    st.metric(label="Estimated Average Annual Salary", value=f"${pred_salary:.0f}K")
    st.markdown(f"*Estimated range: **${pred_salary - 15:.0f}K – ${pred_salary + 15:.0f}K***")

    st.markdown("---")
    st.markdown("### 📋 Your Input Summary")
    summary = {
        "Job Category"     : job_category,
        "Sector"           : sector,
        "Ownership Type"   : ownership,
        "Company Size"     : size,
        "State"            : state,
        "Company Rating"   : rating,
        "Company Age"      : f"{company_age} years",
        "Salary Source"    : salary_source
    }
    st.table(pd.DataFrame.from_dict(summary, orient='index', columns=['Value']))
else:
    st.info("👈 Fill in the details on the left and click **Predict Salary**.")

st.markdown("---")
st.caption("Built using Glassdoor Tech Jobs Dataset (2017-2018) | Gradient Boosting Regressor | MAE ≈ $12K")
