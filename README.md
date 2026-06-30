# 🏥 Insurance Health Claim Fraud Detection
### End-to-End Fraud Detection System for Health Insurance Claims

> **Domain:** Insurance (BFSI)  
> **Tools:** Python · MySQL · Power BI · XGBoost · Scikit-learn · Pandas  
> **Author:** Dharmik Panchal

---

## 📌 Project Overview

This project builds a complete **Health Insurance Claim Fraud Detection** system — from database design and SQL-based fraud rules to ML-powered classification and provider risk ranking. It simulates a real insurance compliance workflow used by TPA (Third Party Administrators) and insurance companies to detect fraudulent claims before payout.

---

## 🎯 Fraud Patterns Detected

| Pattern | Description |
|---------|-------------|
| **Claim Inflation** | Claim amount 2x above disease category average |
| **Duplicate Claims** | Same patient, same disease, within 30 days |
| **Non-Empanelled Provider** | Claims through non-approved hospitals |
| **Unnecessary Procedures** | 8+ procedures with ≤2 days hospitalisation |
| **KYC Non-Compliance** | Unverified patients with high-value claims |

---

## 🗃️ Dataset

Synthetically generated dataset simulating a real insurance environment:

| Table | Records | Description |
|-------|---------|-------------|
| `patients` | 1,000 | Demographics, policy type, KYC status |
| `providers` | 100 | Hospitals, empanelment status |
| `claims` | 10,000 | Claims with amount, procedures, days |
| `investigations` | ~850 | Fraud investigation outcomes |

---

## 🗄️ MySQL — 8 Fraud Detection Queries

| Query | Purpose |
|-------|---------|
| Claim Inflation Detection | Claims 2x above disease average |
| Duplicate Claim Detection | SELF JOIN — same patient, 30-day window |
| Non-Empanelled Provider | High-value claims through unapproved hospitals |
| High Procedure + Short Stay | Fraud pattern — excessive billing |
| Provider Fraud Rank | CTE + RANK() Window Function |
| Monthly Fraud Trend | Month-over-month suspicious claim volume |
| Disease Category Analysis | Fraud rate & avg amount per disease |
| KYC Risk Patients | Unverified patients with large claims |

---

## 🤖 Machine Learning Models

**3 Models Compared:**

| Model | AUC-ROC |
|-------|---------|
| Logistic Regression | Baseline |
| Random Forest | High |
| **XGBoost** | **Best** |

**Key Features Used:**
- Claim amount, inflation ratio, cost per procedure
- Duplicate flag, empanelled provider flag, KYC risk
- Days hospitalised, number of procedures
- Suspicious pattern (high procedures + short stay)

---

## 📊 Visualizations

| File | Content |
|------|---------|
| `01_eda_dashboard.png` | Fraud distribution, amount analysis, disease trends, monthly volume |
| `02_fraud_patterns.png` | Inflation ratio, policy type fraud rate, recovery amounts |
| `03_ml_evaluation.png` | Confusion matrix, ROC curves (3 models), feature importance |
| `04_provider_risk.png` | Provider fraud rate distribution, top 15 high-risk providers |

---

## 📁 Project Structure

```
insurance-fraud-detection/
│
├── generate_data.py              # Synthetic dataset generation
├── fraud_analysis.py             # Main Python ML + EDA script
├── fraud_schema_and_queries.sql  # MySQL schema + 8 fraud queries
│
├── patients.csv
├── providers.csv
├── claims.csv
├── investigations.csv
├── provider_risk_scores.csv      # Output for Power BI
├── claims_analysis.csv           # Output for Power BI
│
├── 01_eda_dashboard.png
├── 02_fraud_patterns.png
├── 03_ml_evaluation.png
├── 04_provider_risk.png
└── powerbi_dashboard.png
```

---

## ⚙️ How to Run

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn faker

python generate_data.py
python fraud_analysis.py
```

---

## 💡 Key Insights

- **11.3%** of claims flagged as fraudulent
- **Cancer & Neurological** categories show highest fraud rates
- **774 claims** detected as inflated (2x above disease average)
- **Claim inflation ratio** is the strongest fraud predictor
- **₹19.3 Crore+** worth of fraud identified in investigations
- **XGBoost** outperforms all models with highest AUC score

---

## 🔗 Technologies Used

`Python` `MySQL` `Power BI` `XGBoost` `Scikit-learn` `Pandas` `NumPy` `Matplotlib` `Seaborn`

---

## 📬 Contact

**Dharmik Panchal** — Data Scientist  
📧 iamdharmik13@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/iamdharmik1334) · [GitHub](https://github.com/iamdharmik13)
