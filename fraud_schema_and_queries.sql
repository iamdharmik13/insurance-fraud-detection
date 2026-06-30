-- ============================================================
--  INSURANCE HEALTH CLAIM FRAUD DETECTION — MySQL
--  Project by: Dharmik Panchal
--  Domain: Insurance (BFSI)
--  Purpose: Health Claim Fraud Detection & Analysis
-- ============================================================

CREATE DATABASE IF NOT EXISTS insurance_fraud;
USE insurance_fraud;

-- ── TABLES ───────────────────────────────────────────────────

CREATE TABLE patients (
    patient_id    VARCHAR(10) PRIMARY KEY,
    name          VARCHAR(100),
    age           INT,
    gender        VARCHAR(10),
    city          VARCHAR(50),
    policy_type   VARCHAR(30),
    premium_tier  VARCHAR(20),
    policy_start  DATE,
    kyc_verified  VARCHAR(5)
);
select * from patients;
CREATE TABLE providers (
    provider_id   VARCHAR(10) PRIMARY KEY,
    hospital_name VARCHAR(100),
    city          VARCHAR(50),
    provider_type VARCHAR(30),
    empanelled    VARCHAR(5)
);

CREATE TABLE claims (
    claim_id             VARCHAR(12) PRIMARY KEY,
    patient_id           VARCHAR(10),
    provider_id          VARCHAR(10),
    disease_category     VARCHAR(30),
    claim_amount         DECIMAL(12,2),
    days_hospitalised    INT,
    num_procedures       INT,
    claim_date           DATE,
    duplicate_claim      TINYINT(1),
    empanelled_provider  TINYINT(1),
    claim_status         VARCHAR(20),
    is_fraud             TINYINT(1),
    FOREIGN KEY (patient_id)  REFERENCES patients(patient_id),
    FOREIGN KEY (provider_id) REFERENCES providers(provider_id)
);


CREATE TABLE investigations (
    investigation_id  VARCHAR(10) PRIMARY KEY,
    claim_id          VARCHAR(12),
    patient_id        VARCHAR(10),
    fraud_type        VARCHAR(50),
    investigation_date DATE,
    status            VARCHAR(30),
    amount_recovered  DECIMAL(12,2),
    FOREIGN KEY (claim_id)   REFERENCES claims(claim_id),
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);


-- ── IMPORT CSVs via MySQL Workbench Table Data Import Wizard ──


-- ════════════════════════════════════════════════════════════
--  QUERY 1: Claim Inflation Detection
--  Claims where amount is 2x above disease category average
-- ════════════════════════════════════════════════════════════
SELECT
    c.claim_id,
    c.patient_id,
    p.name,
    c.disease_category,
    c.claim_amount,
    ROUND(avg_by_disease.avg_amount, 2)  AS category_avg,
    ROUND(c.claim_amount / avg_by_disease.avg_amount, 2) AS inflation_ratio
FROM claims c
JOIN patients p ON c.patient_id = p.patient_id
JOIN (
    SELECT disease_category, AVG(claim_amount) AS avg_amount
    FROM claims
    GROUP BY disease_category
) avg_by_disease ON c.disease_category = avg_by_disease.disease_category
WHERE c.claim_amount > avg_by_disease.avg_amount * 2
ORDER BY inflation_ratio DESC;


-- ════════════════════════════════════════════════════════════
--  QUERY 2: Duplicate Claim Detection
--  Patients who filed multiple claims for same disease in 30 days
-- ════════════════════════════════════════════════════════════
SELECT
    c1.patient_id,
    p.name,
    c1.disease_category,
    c1.claim_id        AS claim_1,
    c2.claim_id        AS claim_2,
    c1.claim_amount    AS amount_1,
    c2.claim_amount    AS amount_2,
    c1.claim_date      AS date_1,
    c2.claim_date      AS date_2,
    DATEDIFF(c2.claim_date, c1.claim_date) AS days_apart
FROM claims c1
JOIN claims c2
    ON  c1.patient_id       = c2.patient_id
    AND c1.disease_category = c2.disease_category
    AND c1.claim_id         < c2.claim_id
    AND DATEDIFF(c2.claim_date, c1.claim_date) BETWEEN 1 AND 30
JOIN patients p ON c1.patient_id = p.patient_id
ORDER BY days_apart;


-- ════════════════════════════════════════════════════════════
--  QUERY 3: Non-Empanelled Provider Fraud
--  High-value claims through non-approved hospitals
-- ════════════════════════════════════════════════════════════
SELECT
    c.claim_id,
    c.patient_id,
    pat.name,
    pr.hospital_name,
    pr.provider_type,
    pr.empanelled,
    c.claim_amount,
    c.disease_category,
    c.claim_status
FROM claims c
JOIN patients  pat ON c.patient_id  = pat.patient_id
JOIN providers pr  ON c.provider_id = pr.provider_id
WHERE pr.empanelled = 'No'
  AND c.claim_amount > 50000
ORDER BY c.claim_amount DESC;


-- ════════════════════════════════════════════════════════════
--  QUERY 4: High Procedure Count with Short Stay (Fraud Pattern)
--  Many procedures but very short hospitalisation = suspicious
-- ════════════════════════════════════════════════════════════
SELECT
    c.claim_id,
    c.patient_id,
    p.name,
    c.disease_category,
    c.num_procedures,
    c.days_hospitalised,
    ROUND(c.claim_amount, 2) AS claim_amount,
    ROUND(c.claim_amount / c.num_procedures, 2) AS cost_per_procedure
FROM claims c
JOIN patients p ON c.patient_id = p.patient_id
WHERE c.num_procedures >= 8
  AND c.days_hospitalised <= 2
ORDER BY cost_per_procedure DESC;


-- ════════════════════════════════════════════════════════════
--  QUERY 5: Provider Fraud Score — Rank by Fraud Rate
--  Window function to rank hospitals by suspicious claim %
-- ════════════════════════════════════════════════════════════
WITH provider_stats AS (
    SELECT
        c.provider_id,
        pr.hospital_name,
        pr.provider_type,
        COUNT(*)                                              AS total_claims,
        SUM(c.is_fraud)                                       AS fraud_claims,
        ROUND(SUM(c.claim_amount), 2)                         AS total_billed,
        ROUND(AVG(c.claim_amount), 2)                         AS avg_claim
    FROM claims c
    JOIN providers pr ON c.provider_id = pr.provider_id
    GROUP BY c.provider_id, pr.hospital_name, pr.provider_type
),
ranked AS (
    SELECT *,
        ROUND(fraud_claims * 100.0 / total_claims, 2)         AS fraud_rate_pct,
        RANK() OVER (ORDER BY fraud_claims DESC)               AS fraud_rank
    FROM provider_stats
)
SELECT *
FROM ranked
WHERE total_claims >= 10
ORDER BY fraud_rank
LIMIT 20;


-- ════════════════════════════════════════════════════════════
--  QUERY 6: Monthly Fraud Trend
--  Track fraud volume month over month
-- ════════════════════════════════════════════════════════════
SELECT
    DATE_FORMAT(claim_date, '%Y-%m')                          AS month,
    COUNT(*)                                                   AS total_claims,
    SUM(is_fraud)                                              AS fraud_claims,
    ROUND(SUM(is_fraud) * 100.0 / COUNT(*), 2)                AS fraud_rate_pct,
    ROUND(SUM(CASE WHEN is_fraud=1 THEN claim_amount ELSE 0 END), 2) AS fraud_amount,
    ROUND(SUM(claim_amount), 2)                                AS total_amount
FROM claims
GROUP BY DATE_FORMAT(claim_date, '%Y-%m')
ORDER BY month;


-- ════════════════════════════════════════════════════════════
--  QUERY 7: Disease Category Fraud Analysis
--  Which diseases have highest fraud rate & inflated claims
-- ════════════════════════════════════════════════════════════
SELECT
    disease_category,
    COUNT(*)                                                    AS total_claims,
    SUM(is_fraud)                                               AS fraud_count,
    ROUND(SUM(is_fraud) * 100.0 / COUNT(*), 2)                 AS fraud_rate_pct,
    ROUND(AVG(claim_amount), 2)                                 AS avg_claim_amount,
    ROUND(AVG(CASE WHEN is_fraud=1 THEN claim_amount END), 2)   AS avg_fraud_amount,
    ROUND(MAX(claim_amount), 2)                                 AS max_claim_amount
FROM claims
GROUP BY disease_category
ORDER BY fraud_rate_pct DESC;


-- ════════════════════════════════════════════════════════════
--  QUERY 8: KYC Unverified Patients with High Claims
--  Compliance risk — unverified patients claiming large amounts
-- ════════════════════════════════════════════════════════════
SELECT
    p.patient_id,
    p.name,
    p.kyc_verified,
    p.policy_type,
    COUNT(c.claim_id)       AS total_claims,
    ROUND(SUM(c.claim_amount), 2) AS total_claimed,
    SUM(c.is_fraud)         AS fraud_count
FROM patients p
JOIN claims c ON p.patient_id = c.patient_id
WHERE p.kyc_verified = 'No'
GROUP BY p.patient_id, p.name, p.kyc_verified, p.policy_type
HAVING total_claimed > 100000
ORDER BY total_claimed DESC;
