import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.ensemble import GradientBoostingRegressor

df = pd.read_csv('glassdoor_jobs.csv')
if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)

def parse_salary(s):
    try:
        s = s.replace('$', '').replace('K', '').split('(')[0].strip()
        low, high = s.split('-')
        return int(low.strip()), int(high.strip())
    except:
        return np.nan, np.nan

df[['min_salary', 'max_salary']] = df['Salary Estimate'].apply(lambda x: pd.Series(parse_salary(x)))
df['avg_salary'] = (df['min_salary'] + df['max_salary']) / 2
df['Company Name'] = df['Company Name'].str.split('\n').str[0].str.strip()
df['state'] = df['Location'].str.split(',').str[-1].str.strip()
df['Rating'] = df['Rating'].replace(-1, np.nan)
df['Founded'] = df['Founded'].replace(-1, np.nan)

for col in ['Size', 'Type of ownership', 'Industry', 'Sector', 'Revenue', 'Headquarters', 'Competitors']:
    df[col] = df[col].replace('-1', np.nan)
df['Revenue'] = df['Revenue'].replace('Unknown / Non-Applicable', np.nan)
df['Size'] = df['Size'].replace('Unknown', np.nan)

size_order = {
    '1 to 50 employees': 1, '51 to 200 employees': 2,
    '201 to 500 employees': 3, '501 to 1000 employees': 4,
    '1001 to 5000 employees': 5, '5001 to 10000 employees': 6,
    '10000+ employees': 7
}
df['size_encoded'] = df['Size'].map(size_order)
df['company_age'] = 2018 - df['Founded']
df.loc[df['company_age'] < 0, 'company_age'] = np.nan
df['salary_source'] = df['Salary Estimate'].apply(lambda x: 'Employer' if 'Employer' in str(x) else 'Glassdoor')

def simplify_title(title):
    t = title.lower()
    if 'data scientist' in t: return 'Data Scientist'
    elif 'data engineer' in t: return 'Data Engineer'
    elif 'data analyst' in t or 'analytics' in t: return 'Data Analyst'
    elif 'machine learning' in t or 'ml engineer' in t: return 'ML Engineer'
    elif 'manager' in t or 'director' in t or 'head' in t: return 'Manager/Director'
    elif 'business intelligence' in t or 'bi ' in t: return 'BI Analyst'
    else: return 'Other'

df['job_category'] = df['Job Title'].apply(simplify_title)

model_df = df[['avg_salary', 'Rating', 'size_encoded', 'company_age',
               'job_category', 'Sector', 'Type of ownership', 'state', 'salary_source']].copy()

for col in ['Rating', 'size_encoded', 'company_age']:
    model_df[col] = model_df[col].fillna(model_df[col].median())
for col in ['Sector', 'Type of ownership', 'state']:
    model_df[col] = model_df[col].fillna(model_df[col].mode()[0])
model_df = model_df.dropna(subset=['avg_salary']).reset_index(drop=True)

Q1, Q3 = model_df['avg_salary'].quantile([0.25, 0.75])
IQR = Q3 - Q1
model_df['avg_salary'] = model_df['avg_salary'].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)

cat_cols = ['job_category', 'Sector', 'Type of ownership', 'salary_source']
model_encoded = pd.get_dummies(model_df, columns=cat_cols, drop_first=True)

le = LabelEncoder()
model_encoded['state_encoded'] = le.fit_transform(model_df['state'])
model_encoded.drop(columns=['state'], inplace=True)

bool_cols = model_encoded.select_dtypes(include='bool').columns
model_encoded[bool_cols] = model_encoded[bool_cols].astype(int)

y = np.log1p(model_encoded['avg_salary'])
X = model_encoded.drop(columns=['avg_salary'])

numeric_features = ['Rating', 'size_encoded', 'company_age', 'state_encoded']
scaler = RobustScaler()
X[numeric_features] = scaler.fit_transform(X[numeric_features])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor(
    n_estimators=200, learning_rate=0.1,
    max_depth=4, subsample=0.8, random_state=42
)
model.fit(X_train, y_train)

joblib.dump(model,  'glassdoor_salary_gbr_model.pkl')
joblib.dump(scaler, 'glassdoor_robust_scaler.pkl')
joblib.dump(le,     'glassdoor_label_encoder.pkl')
joblib.dump(list(X.columns), 'glassdoor_feature_columns.pkl')

print("Model trained and all 4 .pkl files saved successfully.")
