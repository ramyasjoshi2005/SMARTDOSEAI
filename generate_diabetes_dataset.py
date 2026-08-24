# =========================================================
# SMARTDOSEAI - DIABETES DATASET GENERATOR
# Improved realistic synthetic dataset
# =========================================================

import os
import random
import numpy as np
import pandas as pd


NUM_SAMPLES = 5000
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

os.makedirs(
    "datasets",
    exist_ok=True
)


DISEASES = [
    "Type 1 Diabetes",
    "Type 2 Diabetes",
    "Gestational Diabetes",
    "Prediabetes"
]


# =========================================================
# HELPER
# =========================================================

def clipped_normal(
    mean,
    std,
    low,
    high
):

    value = np.random.normal(
        mean,
        std
    )

    return round(
        np.clip(
            value,
            low,
            high
        ),
        2
    )


def probability_event(
    probability
):

    return (
        1
        if random.random() < probability
        else 0
    )


# =========================================================
# DATA
# =========================================================

data = []


for _ in range(NUM_SAMPLES):

    age = random.randint(
        18,
        85
    )

    gender = random.choice(
        [
            "Male",
            "Female"
        ]
    )

    disease = random.choice(
        DISEASES
    )


    # =====================================================
    # CLINICAL PROFILES
    #
    # These intentionally overlap.
    # =====================================================

    profiles = {

        "Type 1 Diabetes": {

            "fasting": (
                205,
                38
            ),

            "random": (
                285,
                55
            ),

            "post_meal": (
                305,
                55
            ),

            "hba1c": (
                8.8,
                1.4
            ),

            "insulin": (
                38,
                22
            ),

            "bmi": (
                24,
                4
            )

        },


        "Type 2 Diabetes": {

            "fasting": (
                180,
                38
            ),

            "random": (
                255,
                55
            ),

            "post_meal": (
                280,
                55
            ),

            "hba1c": (
                8.0,
                1.3
            ),

            "insulin": (
                100,
                42
            ),

            "bmi": (
                30,
                5
            )

        },


        "Gestational Diabetes": {

            "fasting": (
                155,
                32
            ),

            "random": (
                225,
                48
            ),

            "post_meal": (
                250,
                50
            ),

            "hba1c": (
                7.1,
                1.0
            ),

            "insulin": (
                92,
                38
            ),

            "bmi": (
                29,
                5
            )

        },


        "Prediabetes": {

            "fasting": (
                125,
                25
            ),

            "random": (
                175,
                40
            ),

            "post_meal": (
                195,
                40
            ),

            "hba1c": (
                6.1,
                0.65
            ),

            "insulin": (
                82,
                35
            ),

            "bmi": (
                28,
                5
            )

        }

    }


    profile = profiles[
        disease
    ]


    # =====================================================
    # GLUCOSE / METABOLIC FEATURES
    # =====================================================

    fasting_glucose = clipped_normal(
        *profile["fasting"],
        80,
        330
    )


    random_glucose = clipped_normal(
        *profile["random"],
        100,
        420
    )


    post_meal_glucose = clipped_normal(
        *profile["post_meal"],
        110,
        450
    )


    hba1c = clipped_normal(
        *profile["hba1c"],
        4.5,
        13
    )


    insulin = clipped_normal(
        *profile["insulin"],
        5,
        200
    )


    bmi = clipped_normal(
        *profile["bmi"],
        15,
        45
    )


    # =====================================================
    # OTHER FEATURES
    # =====================================================

    cholesterol = clipped_normal(
        220,
        30,
        120,
        330
    )


    triglycerides = clipped_normal(
        180,
        45,
        60,
        350
    )


    systolic_bp = random.randint(
        105,
        195
    )


    diastolic_bp = random.randint(
        65,
        125
    )


    # =====================================================
    # PREGNANCY
    #
    # Gestational diabetes has a strong association,
    # but pregnancy is not made perfectly deterministic.
    # =====================================================

    if disease == "Gestational Diabetes":

        pregnancy = probability_event(
            0.90
        )

    else:

        pregnancy = probability_event(
            0.04
        )


    # =====================================================
    # FAMILY HISTORY
    # =====================================================

    family_history = probability_event(
        0.50
    )


    # =====================================================
    # LIFESTYLE
    # =====================================================

    physical_activity = random.choice(
        [
            "Low",
            "Moderate",
            "High"
        ]
    )


    # =====================================================
    # SYMPTOMS
    #
    # Symptoms correlate somewhat with glucose severity,
    # but are deliberately noisy.
    # =====================================================

    glucose_severity = np.clip(

        (

            (
                fasting_glucose - 90
            )
            / 240

            +

            (
                hba1c - 5
            )
            / 8

        )
        / 2,

        0,
        1

    )


    excessive_thirst = probability_event(

        0.20
        +
        0.55 *
        glucose_severity

    )


    frequent_urination = probability_event(

        0.20
        +
        0.55 *
        glucose_severity

    )


    blurred_vision = probability_event(

        0.10
        +
        0.40 *
        glucose_severity

    )


    water_intake = clipped_normal(

        2.5
        +
        1.0 *
        glucose_severity,

        0.9,

        0.5,

        6

    )


    # =====================================================
    # ADD SMALL MEASUREMENT NOISE
    # =====================================================

    fasting_glucose += np.random.normal(
        0,
        7
    )


    random_glucose += np.random.normal(
        0,
        10
    )


    post_meal_glucose += np.random.normal(
        0,
        10
    )


    hba1c += np.random.normal(
        0,
        0.12
    )


    insulin += np.random.normal(
        0,
        5
    )


    bmi += np.random.normal(
        0,
        0.5
    )


    # =====================================================
    # CLIP AFTER NOISE
    # =====================================================

    fasting_glucose = round(
        np.clip(
            fasting_glucose,
            80,
            330
        ),
        2
    )


    random_glucose = round(
        np.clip(
            random_glucose,
            100,
            420
        ),
        2
    )


    post_meal_glucose = round(
        np.clip(
            post_meal_glucose,
            110,
            450
        ),
        2
    )


    hba1c = round(
        np.clip(
            hba1c,
            4.5,
            13
        ),
        2
    )


    insulin = round(
        np.clip(
            insulin,
            5,
            200
        ),
        2
    )


    bmi = round(
        np.clip(
            bmi,
            15,
            45
        ),
        2
    )


    # =====================================================
    # MISSING VALUES
    # =====================================================

    if random.random() < 0.03:

        insulin = np.nan


    if random.random() < 0.03:

        cholesterol = np.nan


    # =====================================================
    # STORE
    # =====================================================

    data.append([

        age,
        gender,
        bmi,
        fasting_glucose,
        random_glucose,
        post_meal_glucose,
        hba1c,
        insulin,
        cholesterol,
        triglycerides,
        systolic_bp,
        diastolic_bp,
        pregnancy,
        family_history,
        physical_activity,
        excessive_thirst,
        frequent_urination,
        blurred_vision,
        water_intake,
        disease

    ])


# =========================================================
# DATAFRAME
# =========================================================

df = pd.DataFrame(

    data,

    columns=[

        "Age",
        "Gender",
        "BMI",
        "Fasting_Glucose",
        "Random_Glucose",
        "Post_Meal_Glucose",
        "HbA1c",
        "Insulin",
        "Cholesterol",
        "Triglycerides",
        "Systolic_BP",
        "Diastolic_BP",
        "Pregnancy",
        "Family_History",
        "Physical_Activity",
        "Excessive_Thirst",
        "Frequent_Urination",
        "Blurred_Vision",
        "Water_Intake",
        "Diabetes_Type"

    ]

)


# =========================================================
# SHUFFLE
# =========================================================

df = df.sample(
    frac=1,
    random_state=SEED
).reset_index(
    drop=True
)


# =========================================================
# SAVE
# =========================================================

output_path = (
    "datasets/diabetes_dataset.csv"
)


df.to_csv(
    output_path,
    index=False
)


# =========================================================
# OUTPUT
# =========================================================

print(
    "\nDiabetes Dataset Generated"
)

print(
    f"Shape: {df.shape}"
)

print(
    "\nDiabetes distribution:"
)

print(
    df[
        "Diabetes_Type"
    ].value_counts()
)

print(
    "\nSaved to:"
)

print(
    output_path
)