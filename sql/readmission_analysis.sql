SELECT current_catalog() AS catalog, current_schema() AS schema;
SELECT COUNT(*) AS rows FROM diabetes_raw;
-- diabetes_base:
-- 1) keeps only the columns we care about
-- 2) creates a clean 30-day readmission label
-- 3) casts the numeric columns so modeling doesn't get messy later

CREATE OR REPLACE VIEW diabetes_base AS
SELECT
  -- identifiers
  CAST(encounter_id AS STRING) AS encounter_id,
  CAST(patient_nbr  AS STRING) AS patient_nbr,

  -- target: original + binary version
  readmitted,
  CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END AS readmit_30d,

  -- demographics-ish fields (keep as categories)
  COALESCE(race,   'Unknown') AS race,
  COALESCE(gender, 'Unknown') AS gender,
  COALESCE(age,    'Unknown') AS age_band,

  -- core utilization/severity proxies (force numeric)
  CAST(time_in_hospital   AS INT) AS time_in_hospital,
  CAST(num_lab_procedures AS INT) AS num_lab_procedures,
  CAST(num_procedures     AS INT) AS num_procedures,
  CAST(num_medications    AS INT) AS num_medications,

  CAST(number_outpatient  AS INT) AS number_outpatient,
  CAST(number_emergency   AS INT) AS number_emergency,
  CAST(number_inpatient   AS INT) AS number_inpatient,

  CAST(number_diagnoses   AS INT) AS number_diagnoses,

  -- diabetes / treatment indicators
  max_glu_serum,
  A1Cresult,
  insulin,
  change,
  diabetesMed

FROM diabetes_raw
WHERE encounter_id IS NOT NULL
  AND patient_nbr  IS NOT NULL;

  SELECT
  COUNT(*) AS rows,
  SUM(readmit_30d) AS readmit_30d_yes,
  ROUND(AVG(readmit_30d), 4) AS readmit_rate
FROM diabetes_base;

-- diabetes_features:
-- adds a handful of explainable “risk signals” that are easy to talk about in interviews

CREATE OR REPLACE VIEW diabetes_features AS
SELECT
  encounter_id,
  patient_nbr,
  readmitted,
  readmit_30d,

  -- numeric features
  time_in_hospital,
  num_lab_procedures,
  num_procedures,
  num_medications,
  number_outpatient,
  number_emergency,
  number_inpatient,
  number_diagnoses,

  -- ratios (normalize by length of stay)
  CASE WHEN time_in_hospital > 0 THEN num_lab_procedures * 1.0 / time_in_hospital END AS labs_per_day,
  CASE WHEN time_in_hospital > 0 THEN num_medications  * 1.0 / time_in_hospital END AS meds_per_day,

  -- utilization proxy (simple but strong signal)
  (number_inpatient + number_emergency) AS acute_visit_count,

  -- categories
  race,
  gender,
  age_band,

  -- “clinical-ish” flags (these come directly from text categories in the dataset)
  CASE WHEN A1Cresult IN ('>7', '>8') THEN 1 ELSE 0 END AS a1c_high_flag,
  CASE WHEN max_glu_serum IN ('>200', '>300') THEN 1 ELSE 0 END AS glucose_high_flag,

  -- treatment complexity flags
  CASE WHEN insulin IN ('Up', 'Down', 'Steady') THEN 1 ELSE 0 END AS on_insulin_flag,
  CASE WHEN diabetesMed = 'Yes' THEN 1 ELSE 0 END AS diabetes_med_flag,
  CASE WHEN change = 'Ch' THEN 1 ELSE 0 END AS med_changed_flag

FROM diabetes_base;
SELECT * FROM diabetes_features LIMIT 10;

-- modeling view:
-- keep only ML features + label, in one place, so the notebook stays clean

CREATE OR REPLACE VIEW diabetes_model_table AS
SELECT
  encounter_id,
  patient_nbr,
  readmit_30d,

  time_in_hospital,
  num_lab_procedures,
  num_procedures,
  num_medications,
  number_outpatient,
  number_emergency,
  number_inpatient,
  number_diagnoses,
  labs_per_day,
  meds_per_day,
  acute_visit_count,

  race,
  gender,
  age_band,

  a1c_high_flag,
  glucose_high_flag,
  on_insulin_flag,
  diabetes_med_flag,
  med_changed_flag

FROM diabetes_features;
SELECT
  COUNT(*) AS rows,
  SUM(readmit_30d) AS readmit_30d_yes,
  ROUND(AVG(readmit_30d), 4) AS readmit_rate
FROM diabetes_model_table;
