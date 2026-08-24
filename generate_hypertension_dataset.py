# =========================================================
# generate_hypertension_dataset.py
# =========================================================

import pandas as pd
import numpy as np
import random

NUM_SAMPLES = 5000

data = []

for i in range(NUM_SAMPLES):

    age = random.randint(20, 90)

    gender = random.choice([
        "Male",
        "Female"
    ])

    disease = random.choice([

        "Primary Hypertension",
        "Secondary Hypertension",
        "Pulmonary Hypertension",
        "Resistant Hypertension"

    ])

    # ----------------------------------------

    if disease == "Primary Hypertension":

        systolic_bp = random.randint(140, 170)
        diastolic_bp = random.randint(90, 110)
        oxygen_level = random.randint(94, 99)

    elif disease == "Secondary Hypertension":

        systolic_bp = random.randint(160, 190)
        diastolic_bp = random.randint(100, 120)
        oxygen_level = random.randint(92, 98)

    elif disease == "Pulmonary Hypertension":

        systolic_bp = random.randint(150, 180)
        diastolic_bp = random.randint(95, 115)
        oxygen_level = random.randint(82, 90)

    else:

        systolic_bp = random.randint(180, 220)
        diastolic_bp = random.randint(110, 140)
        oxygen_level = random.randint(90, 96)

    bmi = round(np.random.normal(30, 4), 2)

    smoking = random.choice([0, 1])

    alcohol = random.choice([0, 1])

    family_history = random.choice([0, 1])

    stress_level = random.choice([
        "Low",
        "Moderate",
        "High"
    ])

    cholesterol = round(np.random.normal(220, 25), 2)

    heart_rate = random.randint(60, 140)

    salt_intake = random.choice([
        "Low",
        "Moderate",
        "High"
    ])

    exercise = random.choice([
        "Low",
        "Moderate",
        "High"
    ])

    data.append([

        age,
        gender,
        systolic_bp,
        diastolic_bp,
        bmi,
        smoking,
        alcohol,
        family_history,
        stress_level,
        cholesterol,
        oxygen_level,
        heart_rate,
        salt_intake,
        exercise,
        disease

    ])

df = pd.DataFrame(data, columns=[

    "Age",
    "Gender",
    "Systolic_BP",
    "Diastolic_BP",
    "BMI",
    "Smoking",
    "Alcohol",
    "Family_History",
    "Stress_Level",
    "Cholesterol",
    "Oxygen_Level",
    "Heart_Rate",
    "Salt_Intake",
    "Exercise",
    "Hypertension_Type"

])

df = df.sample(frac=1).reset_index(drop=True)

df.to_csv("datasets/hypertension_dataset.csv", index=False)