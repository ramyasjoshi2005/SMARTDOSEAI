# =========================================================
# generate_heart_dataset.py
# =========================================================

import pandas as pd
import numpy as np
import random

NUM_SAMPLES = 5000

data = []

for i in range(NUM_SAMPLES):

    age = random.randint(25, 90)

    gender = random.choice(["Male", "Female"])

    disease = random.choice([

        "Heart Failure",
        "Coronary Artery Disease",
        "Arrhythmia",
        "Hypertension",
        "Valve Disease"

    ])

    # ----------------------------------------

    if disease == "Heart Failure":

        troponin = round(np.random.normal(4, 1), 2)
        ldl = random.randint(170, 250)
        ecg = "Abnormal"
        chest_pain = 1
        breathlessness = 1
        dizziness = 1

    elif disease == "Coronary Artery Disease":

        troponin = round(np.random.normal(2.5, 0.8), 2)
        ldl = random.randint(180, 260)
        ecg = "Abnormal"
        chest_pain = 1
        breathlessness = random.choice([0, 1])
        dizziness = 0

    elif disease == "Arrhythmia":

        troponin = round(np.random.normal(1.5, 0.5), 2)
        ldl = random.randint(120, 200)
        ecg = "Abnormal"
        chest_pain = 0
        breathlessness = 0
        dizziness = 1

    elif disease == "Valve Disease":

        troponin = round(np.random.normal(2, 0.6), 2)
        ldl = random.randint(130, 220)
        ecg = "Abnormal"
        chest_pain = 1
        breathlessness = 1
        dizziness = 0

    else:

        troponin = round(np.random.normal(1.2, 0.4), 2)
        ldl = random.randint(130, 220)
        ecg = "Normal"
        chest_pain = 0
        breathlessness = 0
        dizziness = 0

    systolic_bp = random.randint(110, 220)

    diastolic_bp = random.randint(70, 140)

    heart_rate = random.randint(60, 150)

    cholesterol = random.randint(140, 350)

    hdl = random.randint(25, 90)

    triglycerides = random.randint(100, 350)

    smoking = random.choice([0, 1])

    obesity = random.choice([0, 1])

    family_history = random.choice([0, 1])

    if random.random() < 0.03:
        troponin = np.nan

    data.append([

        age,
        gender,
        systolic_bp,
        diastolic_bp,
        heart_rate,
        troponin,
        cholesterol,
        ldl,
        hdl,
        triglycerides,
        ecg,
        smoking,
        obesity,
        chest_pain,
        breathlessness,
        dizziness,
        family_history,
        disease

    ])

df = pd.DataFrame(data, columns=[

    "Age",
    "Gender",
    "Systolic_BP",
    "Diastolic_BP",
    "Heart_Rate",
    "Troponin",
    "Cholesterol",
    "LDL",
    "HDL",
    "Triglycerides",
    "ECG",
    "Smoking",
    "Obesity",
    "Chest_Pain",
    "Breathlessness",
    "Dizziness",
    "Family_History",
    "Heart_Disease_Type"

])

df = df.sample(frac=1).reset_index(drop=True)

df.to_csv("datasets/heart_dataset.csv", index=False)