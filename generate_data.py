import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

fake = Faker('en_IN')
np.random.seed(42)
random.seed(42)

# ── 1. PATIENTS ───────────────────────────────────────────────
NUM_PATIENTS = 1000

patients = []
for i in range(1, NUM_PATIENTS + 1):
    patients.append({
        "patient_id":    f"PAT{i:05d}",
        "name":          fake.name(),
        "age":           random.randint(18, 80),
        "gender":        random.choice(["Male", "Female"]),
        "city":          random.choice(["Ahmedabad", "Mumbai", "Delhi", "Surat", "Pune", "Bengaluru", "Jaipur"]),
        "policy_type":   random.choice(["Individual", "Family", "Group", "Senior Citizen"]),
        "premium_tier":  random.choice(["Basic", "Standard", "Premium"]),
        "policy_start":  fake.date_between(start_date="-5y", end_date="-6m"),
        "kyc_verified":  random.choice(["Yes", "Yes", "Yes", "No"])
    })

df_patients = pd.DataFrame(patients)

# ── 2. PROVIDERS (Hospitals/Clinics) ──────────────────────────
providers = []
for i in range(1, 101):
    providers.append({
        "provider_id":   f"PROV{i:04d}",
        "hospital_name": fake.company() + " Hospital",
        "city":          random.choice(["Ahmedabad", "Mumbai", "Delhi", "Surat", "Pune"]),
        "provider_type": random.choice(["Government", "Private", "Clinic", "Diagnostic Center"]),
        "empanelled":    random.choice(["Yes", "Yes", "Yes", "No"])
    })

df_providers = pd.DataFrame(providers)

# ── 3. CLAIMS ─────────────────────────────────────────────────
NUM_CLAIMS = 10000

disease_map = {
    "Cardiac":      (50000, 300000),
    "Orthopaedic":  (40000, 200000),
    "Maternity":    (30000, 150000),
    "General":      (5000,  80000),
    "Cancer":       (100000,500000),
    "Dental":       (2000,  20000),
    "Eye":          (5000,  50000),
    "Neurological": (80000, 400000),
}

claims = []
for i in range(1, NUM_CLAIMS + 1):
    pat      = df_patients.sample(1).iloc[0]
    prov     = df_providers.sample(1).iloc[0]
    disease  = random.choice(list(disease_map.keys()))
    lo, hi   = disease_map[disease]

    is_fraud = random.random() < 0.11   # ~11% fraud

    if is_fraud:
        claim_amount   = round(random.uniform(hi * 0.9, hi * 2.5), 2)  # inflated
        days_hosp      = random.randint(1, 2)                            # short stay, high bill
        num_procedures = random.randint(8, 15)                           # excessive procedures
        duplicate_flag = random.choice([True, True, False])
        provider_match = random.choice([True, False, False])             # non-empanelled
    else:
        claim_amount   = round(random.uniform(lo, hi), 2)
        days_hosp      = random.randint(2, 15)
        num_procedures = random.randint(1, 5)
        duplicate_flag = False
        provider_match = True

    claim_date = fake.date_between(start_date="-2y", end_date="today")

    claims.append({
        "claim_id":          f"CLM{i:06d}",
        "patient_id":        pat["patient_id"],
        "provider_id":       prov["provider_id"],
        "disease_category":  disease,
        "claim_amount":      claim_amount,
        "days_hospitalised": days_hosp,
        "num_procedures":    num_procedures,
        "claim_date":        claim_date,
        "duplicate_claim":   int(duplicate_flag),
        "empanelled_provider": int(provider_match),
        "claim_status":      random.choice(["Approved", "Approved", "Pending", "Rejected"]),
        "is_fraud":          int(is_fraud)
    })

df_claims = pd.DataFrame(claims)

# ── 4. INVESTIGATIONS ─────────────────────────────────────────
fraud_claims = df_claims[df_claims["is_fraud"] == 1].sample(frac=0.75)
investigations = []
for i, row in enumerate(fraud_claims.itertuples(), 1):
    investigations.append({
        "investigation_id": f"INV{i:05d}",
        "claim_id":         row.claim_id,
        "patient_id":       row.patient_id,
        "fraud_type":       random.choice(["Claim Inflation", "Duplicate Claim",
                                           "Non-Empanelled Provider", "Unnecessary Procedures",
                                           "Falsified Documents"]),
        "investigation_date": row.claim_date,
        "status":           random.choice(["Confirmed Fraud", "Confirmed Fraud",
                                           "Under Review", "Dismissed"]),
        "amount_recovered": round(row.claim_amount * random.uniform(0.4, 0.9), 2)
    })

df_investigations = pd.DataFrame(investigations)

# ── SAVE ──────────────────────────────────────────────────────
df_patients.to_csv("patients.csv",         index=False)
df_providers.to_csv("providers.csv",       index=False)
df_claims.to_csv("claims.csv",             index=False)
df_investigations.to_csv("investigations.csv", index=False)

print("✅ Insurance Fraud Dataset Generated!")
print(f"   Patients       : {len(df_patients):,}")
print(f"   Providers      : {len(df_providers):,}")
print(f"   Claims         : {len(df_claims):,}")
print(f"   Investigations : {len(df_investigations):,}")
print(f"   Fraud Claims   : {df_claims['is_fraud'].sum():,} ({df_claims['is_fraud'].mean()*100:.1f}%)")
