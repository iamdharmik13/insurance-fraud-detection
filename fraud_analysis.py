# ============================================================
#  INSURANCE HEALTH CLAIM FRAUD DETECTION
#  Author : Dharmik Panchal
#  Domain : Insurance (BFSI)
#  Tools  : Python, Pandas, Scikit-learn, XGBoost, Matplotlib
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, ConfusionMatrixDisplay)
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ── STYLE ─────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#1a1d2e",
    "axes.edgecolor":   "#3a3d5c",
    "axes.labelcolor":  "#e0e0e0",
    "xtick.color":      "#b0b0c0",
    "ytick.color":      "#b0b0c0",
    "text.color":       "#e0e0e0",
    "grid.color":       "#2a2d3e",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
})

ACCENT  = "#00d4ff"
DANGER  = "#ff4d6d"
WARNING = "#ffd166"
SUCCESS = "#06d6a0"
PURPLE  = "#a78bfa"

# ── LOAD DATA ─────────────────────────────────────────────────
print("=" * 60)
print("  INSURANCE HEALTH CLAIM FRAUD DETECTION")
print("=" * 60)

df_pat  = pd.read_csv("patients.csv")
df_prov = pd.read_csv("providers.csv")
df_clm  = pd.read_csv("claims.csv", parse_dates=["claim_date"])
df_inv  = pd.read_csv("investigations.csv")

print(f"\n📊 Dataset Loaded:")
print(f"   Patients       : {len(df_pat):,}")
print(f"   Providers      : {len(df_prov):,}")
print(f"   Claims         : {len(df_clm):,}")
print(f"   Investigations : {len(df_inv):,}")
print(f"   Fraud Rate     : {df_clm['is_fraud'].mean()*100:.1f}%")

# ── MERGE & FEATURE ENGINEERING ───────────────────────────────
df = df_clm.merge(df_pat[["patient_id","age","gender","policy_type","premium_tier","kyc_verified"]],
                  on="patient_id", how="left")
df = df.merge(df_prov[["provider_id","provider_type","empanelled"]],
              on="provider_id", how="left")

df["claim_month"]       = df["claim_date"].dt.month
df["claim_year"]        = df["claim_date"].dt.year
df["kyc_risk"]          = (df["kyc_verified"] == "No").astype(int)
df["not_empanelled"]    = (df["empanelled"] == "No").astype(int)
df["cost_per_procedure"]= df["claim_amount"] / df["num_procedures"].replace(0,1)
df["high_procedure"]    = (df["num_procedures"] >= 8).astype(int)
df["short_stay"]        = (df["days_hospitalised"] <= 2).astype(int)
df["suspicious_pattern"]= df["high_procedure"] & df["short_stay"]

# Disease avg amounts
disease_avg = df.groupby("disease_category")["claim_amount"].mean().reset_index()
disease_avg.columns = ["disease_category","disease_avg_amount"]
df = df.merge(disease_avg, on="disease_category", how="left")
df["inflation_ratio"] = df["claim_amount"] / df["disease_avg_amount"]
df["is_inflated"]     = (df["inflation_ratio"] > 2).astype(int)

print(f"\n✅ Features engineered. Shape: {df.shape}")

# ════════════════════════════════════════════════════════════
#  SECTION 1 — EDA
# ════════════════════════════════════════════════════════════
print("\n📈 Running EDA...")

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Insurance Claim Fraud — Exploratory Data Analysis",
             fontsize=18, fontweight="bold", color=ACCENT, y=0.98)

# 1a. Fraud vs Normal
counts = df["is_fraud"].value_counts()
axes[0,0].pie(counts, labels=["Legitimate","Fraud"],
              colors=[SUCCESS, DANGER], autopct="%1.1f%%", startangle=90,
              textprops={"color":"white","fontsize":11},
              wedgeprops={"edgecolor":"#0f1117","linewidth":2})
axes[0,0].set_title("Claim Distribution", fontsize=13, color=ACCENT)

# 1b. Claim amount by fraud
fraud_amt  = df[df["is_fraud"]==1]["claim_amount"]
legit_amt  = df[df["is_fraud"]==0]["claim_amount"]
axes[0,1].hist(legit_amt, bins=40, color=SUCCESS, alpha=0.7, label="Legitimate", density=True)
axes[0,1].hist(fraud_amt, bins=40, color=DANGER,  alpha=0.7, label="Fraud",      density=True)
axes[0,1].set_title("Claim Amount Distribution", fontsize=13, color=ACCENT)
axes[0,1].set_xlabel("Claim Amount (₹)")
axes[0,1].legend()

# 1c. Fraud rate by disease
disease_fraud = df.groupby("disease_category")["is_fraud"].mean().sort_values() * 100
axes[0,2].barh(disease_fraud.index, disease_fraud.values,
               color=[DANGER if v > 12 else WARNING if v > 8 else SUCCESS for v in disease_fraud.values])
axes[0,2].set_title("Fraud Rate by Disease Category", fontsize=13, color=ACCENT)
axes[0,2].set_xlabel("Fraud Rate (%)")

# 1d. Fraud by fraud type (from investigations)
fraud_types = df_inv["fraud_type"].value_counts()
axes[1,0].bar(fraud_types.index, fraud_types.values, color=PURPLE, edgecolor="#0f1117")
axes[1,0].set_title("Fraud Types Distribution", fontsize=13, color=ACCENT)
axes[1,0].tick_params(axis="x", rotation=30)
axes[1,0].set_ylabel("Count")

# 1e. Procedures vs Days (fraud vs legit)
axes[1,1].scatter(df[df["is_fraud"]==0]["days_hospitalised"],
                  df[df["is_fraud"]==0]["num_procedures"],
                  c=SUCCESS, alpha=0.3, s=10, label="Legitimate")
axes[1,1].scatter(df[df["is_fraud"]==1]["days_hospitalised"],
                  df[df["is_fraud"]==1]["num_procedures"],
                  c=DANGER, alpha=0.5, s=15, label="Fraud", marker="X")
axes[1,1].set_title("Procedures vs Days Hospitalised", fontsize=13, color=ACCENT)
axes[1,1].set_xlabel("Days Hospitalised")
axes[1,1].set_ylabel("Num Procedures")
axes[1,1].legend()

# 1f. Monthly fraud trend
monthly = df.groupby(df["claim_date"].dt.to_period("M")).agg(
    total=("is_fraud","count"), fraud=("is_fraud","sum")).reset_index()
monthly["fraud_rate"] = monthly["fraud"] / monthly["total"] * 100
x = range(len(monthly))
axes[1,2].bar(x, monthly["total"], color=SUCCESS, alpha=0.7, label="Legitimate")
axes[1,2].bar(x, monthly["fraud"], color=DANGER,  alpha=0.9, label="Fraud")
axes[1,2].set_title("Monthly Claims: Fraud vs Legitimate", fontsize=13, color=ACCENT)
axes[1,2].set_ylabel("Number of Claims")
axes[1,2].set_xticks(list(x)[::3])
axes[1,2].set_xticklabels([str(monthly["claim_date"].iloc[i]) for i in range(0,len(monthly),3)],
                           rotation=45, fontsize=7)
axes[1,2].legend()

plt.tight_layout()
plt.savefig("01_eda_dashboard.png", dpi=150, bbox_inches="tight", facecolor="#0f1117")
plt.close()
print("   ✅ 01_eda_dashboard.png saved")

# ════════════════════════════════════════════════════════════
#  SECTION 2 — FRAUD PATTERN ANALYSIS
# ════════════════════════════════════════════════════════════
print("\n🔍 Analyzing Fraud Patterns...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Fraud Pattern Deep Dive", fontsize=16, fontweight="bold", color=ACCENT)

# Inflation ratio
axes[0].hist(df[df["is_fraud"]==0]["inflation_ratio"], bins=30, color=SUCCESS,
             alpha=0.7, label="Legitimate", density=True)
axes[0].hist(df[df["is_fraud"]==1]["inflation_ratio"], bins=30, color=DANGER,
             alpha=0.7, label="Fraud", density=True)
axes[0].axvline(2.0, color=WARNING, linestyle="--", linewidth=2, label="2x Threshold")
axes[0].set_title("Claim Inflation Ratio", fontsize=13, color=ACCENT)
axes[0].set_xlabel("Amount / Disease Average")
axes[0].legend()

# Fraud rate by policy type
pol_fraud = df.groupby("policy_type")["is_fraud"].mean().sort_values() * 100
pol_colors = [DANGER if v > 12 else WARNING if v > 8 else SUCCESS for v in pol_fraud.values]
axes[1].bar(pol_fraud.index, pol_fraud.values, color=pol_colors, edgecolor="#0f1117")
axes[1].set_title("Fraud Rate by Policy Type", fontsize=13, color=ACCENT)
axes[1].set_ylabel("Fraud Rate (%)")
axes[1].tick_params(axis="x", rotation=15)

# Amount recovered in investigations
status_rec = df_inv.groupby("status")["amount_recovered"].sum().sort_values()
axes[2].barh(status_rec.index, status_rec.values,
             color=[SUCCESS, WARNING, DANGER, PURPLE][:len(status_rec)])
axes[2].set_title("Amount Recovered by Investigation Status", fontsize=13, color=ACCENT)
axes[2].set_xlabel("Amount Recovered (₹)")

plt.tight_layout()
plt.savefig("02_fraud_patterns.png", dpi=150, bbox_inches="tight", facecolor="#0f1117")
plt.close()
print("   ✅ 02_fraud_patterns.png saved")

# ════════════════════════════════════════════════════════════
#  SECTION 3 — ML MODELS
# ════════════════════════════════════════════════════════════
print("\n🤖 Training ML Models...")

FEATURES = ["claim_amount","days_hospitalised","num_procedures","duplicate_claim",
            "empanelled_provider","kyc_risk","not_empanelled","cost_per_procedure",
            "high_procedure","short_stay","suspicious_pattern","inflation_ratio",
            "is_inflated","age","claim_month"]

le_disease  = LabelEncoder()
le_gender   = LabelEncoder()
le_policy   = LabelEncoder()
le_premium  = LabelEncoder()
le_provtype = LabelEncoder()

df["disease_enc"]   = le_disease.fit_transform(df["disease_category"].fillna("Unknown"))
df["gender_enc"]    = le_gender.fit_transform(df["gender"].fillna("Unknown"))
df["policy_enc"]    = le_policy.fit_transform(df["policy_type"].fillna("Unknown"))
df["premium_enc"]   = le_premium.fit_transform(df["premium_tier"].fillna("Unknown"))
df["provtype_enc"]  = le_provtype.fit_transform(df["provider_type"].fillna("Unknown"))

FEATURES += ["disease_enc","gender_enc","policy_enc","premium_enc","provtype_enc"]

df_ml = df[FEATURES + ["is_fraud"]].dropna()
X = df_ml[FEATURES]
y = df_ml["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# Logistic Regression
lr = LogisticRegression(max_iter=500, random_state=42)
lr.fit(X_train_s, y_train)
lr_proba = lr.predict_proba(X_test_s)[:,1]
lr_pred  = lr.predict(X_test_s)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_proba = rf.predict_proba(X_test)[:,1]
rf_pred  = rf.predict(X_test)

# XGBoost
xgb = XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss",
                     verbosity=0, use_label_encoder=False)
xgb.fit(X_train, y_train)
xgb_proba = xgb.predict_proba(X_test)[:,1]
xgb_pred  = xgb.predict(X_test)

lr_auc  = roc_auc_score(y_test, lr_proba)
rf_auc  = roc_auc_score(y_test, rf_proba)
xgb_auc = roc_auc_score(y_test, xgb_proba)

print(f"\n   Logistic Regression AUC : {lr_auc:.4f}")
print(f"   Random Forest       AUC : {rf_auc:.4f}")
print(f"   XGBoost             AUC : {xgb_auc:.4f}")
print(f"\n   XGBoost Classification Report:")
print(classification_report(y_test, xgb_pred,
      target_names=["Legitimate","Fraud"], digits=3))

# ── MODEL EVALUATION CHARTS ───────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("ML Fraud Detection Models — Evaluation", fontsize=16,
             fontweight="bold", color=ACCENT)

# Confusion Matrix (XGBoost — best model)
cm = confusion_matrix(y_test, xgb_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["Legitimate","Fraud"])
disp.plot(ax=axes[0], cmap="Blues", colorbar=False)
axes[0].set_title(f"XGBoost Confusion Matrix\nAUC: {xgb_auc:.4f}", fontsize=12, color=ACCENT)
axes[0].set_facecolor("#1a1d2e")

# ROC Curves — all 3
fpr_lr,  tpr_lr,  _ = roc_curve(y_test, lr_proba)
fpr_rf,  tpr_rf,  _ = roc_curve(y_test, rf_proba)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_proba)
axes[1].plot(fpr_lr,  tpr_lr,  color=WARNING, lw=2, label=f"Logistic Reg (AUC={lr_auc:.3f})")
axes[1].plot(fpr_rf,  tpr_rf,  color=ACCENT,  lw=2, label=f"Random Forest (AUC={rf_auc:.3f})")
axes[1].plot(fpr_xgb, tpr_xgb, color=DANGER,  lw=2, label=f"XGBoost (AUC={xgb_auc:.3f})")
axes[1].plot([0,1],[0,1],"k--",alpha=0.4)
axes[1].set_title("ROC Curve — All 3 Models", fontsize=13, color=ACCENT)
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].legend(fontsize=9)

# Feature Importance (XGBoost)
fi = pd.Series(xgb.feature_importances_, index=FEATURES).sort_values(ascending=True).tail(12)
bars = axes[2].barh(fi.index, fi.values, color=PURPLE, edgecolor="#0f1117")
axes[2].set_title("Top 12 Feature Importances (XGBoost)", fontsize=13, color=ACCENT)
axes[2].set_xlabel("Importance")
for bar, val in zip(bars, fi.values):
    axes[2].text(bar.get_width()+0.001, bar.get_y()+bar.get_height()/2,
                 f"{val:.3f}", va="center", color="white", fontsize=7)

plt.tight_layout()
plt.savefig("03_ml_evaluation.png", dpi=150, bbox_inches="tight", facecolor="#0f1117")
plt.close()
print("   ✅ 03_ml_evaluation.png saved")

# ════════════════════════════════════════════════════════════
#  SECTION 4 — PROVIDER RISK RANKING
# ════════════════════════════════════════════════════════════
print("\n🏥 Provider Risk Analysis...")

prov_risk = df.groupby(["provider_id"]).agg(
    total_claims=("claim_id","count"),
    fraud_claims=("is_fraud","sum"),
    total_billed=("claim_amount","sum"),
    avg_claim=("claim_amount","mean")
).reset_index()
prov_risk["fraud_rate"] = prov_risk["fraud_claims"] / prov_risk["total_claims"] * 100
prov_risk = prov_risk[prov_risk["total_claims"] >= 5]
prov_risk["risk_tier"] = pd.cut(
    prov_risk["fraud_rate"],
    bins=[-1, 8, 18, 101],
    labels=["LOW","MEDIUM","HIGH"]
)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Provider Risk Analysis", fontsize=16, fontweight="bold", color=ACCENT)

# Fraud rate distribution
axes[0].hist(prov_risk["fraud_rate"], bins=20, color=PURPLE, edgecolor="#0f1117")
axes[0].axvline(8,  color=SUCCESS, linestyle="--", lw=2, label="LOW threshold (8%)")
axes[0].axvline(18, color=DANGER,  linestyle="--", lw=2, label="HIGH threshold (18%)")
axes[0].set_title("Provider Fraud Rate Distribution", fontsize=13, color=ACCENT)
axes[0].set_xlabel("Fraud Rate (%)")
axes[0].set_ylabel("Number of Providers")
axes[0].legend()

# Top 15 HIGH risk providers
top_prov = prov_risk[prov_risk["risk_tier"]=="HIGH"].nlargest(15,"fraud_rate")
axes[1].barh(top_prov["provider_id"], top_prov["fraud_rate"], color=DANGER, edgecolor="#0f1117")
axes[1].set_title("Top 15 High Risk Providers", fontsize=13, color=ACCENT)
axes[1].set_xlabel("Fraud Rate (%)")
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig("04_provider_risk.png", dpi=150, bbox_inches="tight", facecolor="#0f1117")
plt.close()
print("   ✅ 04_provider_risk.png saved")

# Save for Power BI
prov_risk.to_csv("provider_risk_scores.csv", index=False)
df[["claim_id","patient_id","provider_id","disease_category","claim_amount",
    "days_hospitalised","num_procedures","is_fraud","inflation_ratio",
    "suspicious_pattern","duplicate_claim"]].to_csv("claims_analysis.csv", index=False)

# ════════════════════════════════════════════════════════════
#  FINAL SUMMARY
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  PROJECT SUMMARY")
print("=" * 60)
print(f"  Total Claims Analyzed     : {len(df):,}")
print(f"  Fraud Claims Detected     : {df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.1f}%)")
print(f"  Inflated Claims           : {df['is_inflated'].sum():,}")
print(f"  Suspicious Pattern Claims : {df['suspicious_pattern'].sum():,}")
print(f"  XGBoost AUC Score         : {xgb_auc:.4f}")
print(f"  Random Forest AUC Score   : {rf_auc:.4f}")
print(f"  Amount in Investigations  : ₹{df_inv['amount_recovered'].sum():,.0f}")
print("\n  Output Files:")
print("  01_eda_dashboard.png")
print("  02_fraud_patterns.png")
print("  03_ml_evaluation.png")
print("  04_provider_risk.png")
print("  provider_risk_scores.csv")
print("  claims_analysis.csv")
print("=" * 60)
