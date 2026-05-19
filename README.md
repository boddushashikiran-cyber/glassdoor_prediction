# 💼 Glassdoor Tech Job Salary Predictor

Predicts average tech job salaries using the Glassdoor Jobs Dataset (2017-2018).
Built as part of an AI/ML internship project.

## 🔧 Setup & Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/glassdoor-salary-prediction.git
cd glassdoor-salary-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model (generates .pkl files)
```bash
python train.py
```

### 4. Launch the app
```bash
streamlit run app.py
```

## 🚀 Deploy on Streamlit Cloud (Free)

1. Push this entire repo to GitHub (including the `.pkl` files after running `train.py`)
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud) and sign in with GitHub
3. Click **New app** → select your repo → set **Main file path** to `app.py`
4. Click **Deploy** — live in ~2 minutes

## 📁 Project Structure

| File | Description |
|------|-------------|
| `glassdoor_jobs.csv` | Raw dataset |
| `train.py` | Model training script — run once to generate .pkl files |
| `app.py` | Streamlit web app for salary prediction |
| `glassdoor_salary_prediction.ipynb` | Full EDA + ML notebook submission |
| `requirements.txt` | Python dependencies |

## 🧠 Model

Gradient Boosting Regressor (scikit-learn) with:
- Log-transformed target (avg_salary)
- RobustScaler on numeric features
- One-Hot Encoding for categorical features
- ~R² 0.55–0.60 on test set, MAE ≈ $12K
