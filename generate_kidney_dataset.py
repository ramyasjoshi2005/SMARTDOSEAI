# =========================================================
# generate_kidney_dataset.py
# =========================================================

import pandas as pd
import numpy as np
import random

NUM_SAMPLES = 5000

data = []

for i in range(NUM_SAMPLES):

    age = random.randint(18, 90)

    gender = random.choice(["Male", "Female"])

    disease = random.choice([

        "Acute Kidney Injury",
        "Chronic Kidney Disease",
        "Kidney Stones",
        "Glomerulonephritis",
        "Diabetic Nephropathy"

    ])

    # ----------------------------------------
    # DISEASE-SPECIFIC PATTERNS
    # ----------------------------------------

    if disease == "Chronic Kidney Disease":

        creatinine = round(np.random.normal(6, 1), 2)
        egfr = round(np.random.normal(15, 5), 2)
        blood_urea = round(np.random.normal(140, 25), 2)
        potassium = round(np.random.normal(6, 0.5), 2)
        uric_acid = round(np.random.normal(8, 1), 2)

    elif disease == "Acute Kidney Injury":

        creatinine = round(np.random.normal(5, 1), 2)
        egfr = round(np.random.normal(30, 8), 2)
        blood_urea = round(np.random.normal(110, 20), 2)
        potassium = round(np.random.normal(5.8, 0.5), 2)
        uric_acid = round(np.random.normal(7, 1), 2)

    elif disease == "Kidney Stones":

        creatinine = round(np.random.normal(1.8, 0.5), 2)
        egfr = round(np.random.normal(75, 10), 2)
        blood_urea = round(np.random.normal(45, 10), 2)
        potassium = round(np.random.normal(4.5, 0.3), 2)
        uric_acid = round(np.random.normal(9, 1), 2)

    elif disease == "Glomerulonephritis":

        creatinine = round(np.random.normal(4, 1), 2)
        egfr = round(np.random.normal(40, 8), 2)
        blood_urea = round(np.random.normal(90, 15), 2)
        potassium = round(np.random.normal(5.2, 0.4), 2)
        uric_acid = round(np.random.normal(6.5, 1), 2)

    else:

        creatinine = round(np.random.normal(5.5, 1), 2)
        egfr = round(np.random.normal(25, 5), 2)
        blood_urea = round(np.random.normal(130, 20), 2)
        potassium = round(np.random.normal(5.8, 0.4), 2)
        uric_acid = round(np.random.normal(8.5, 1), 2)

    sodium = round(np.random.normal(138, 4), 2)

    calcium = round(np.random.normal(9.2, 0.6), 2)

    hemoglobin = round(np.random.normal(11, 2), 2)

    urine_protein = random.choice([
        "Negative",
        "Trace",
        "+1",
        "+2",
        "+3"
    ])

    diabetes = random.choice([0, 1])

    hypertension = random.choice([0, 1])

    swelling = random.choice([0, 1])

    fatigue = random.choice([0, 1])

    appetite_loss = random.choice([0, 1])

    # ----------------------------------------
    # DIALYSIS
    # ----------------------------------------

    if creatinine > 5 and egfr < 20:

        dialysis_risk = "High"

    elif creatinine > 3:

        dialysis_risk = "Medium"

    else:

        dialysis_risk = "Low"

    # ----------------------------------------

    if random.random() < 0.03:
        potassium = np.nan

    data.append([

        age,
        gender,
        creatinine,
        blood_urea,
        egfr,
        potassium,
        sodium,
        calcium,
        uric_acid,
        hemoglobin,
        urine_protein,
        diabetes,
        hypertension,
        swelling,
        fatigue,
        appetite_loss,
        disease,
        dialysis_risk

    ])

df = pd.DataFrame(data, columns=[

    "Age",
    "Gender",
    "Creatinine",
    "Blood_Urea",
    "eGFR",
    "Potassium",
    "Sodium",
    "Calcium",
    "Uric_Acid",
    "Hemoglobin",
    "Urine_Protein",
    "Diabetes",
    "Hypertension",
    "Swelling",
    "Fatigue",
    "Appetite_Loss",
    "Kidney_Disease_Type",
    "Dialysis_Risk"

])

df = df.sample(frac=1).reset_index(drop=True)

df.to_csv("datasets/kidney_dataset.csv", index=False)