# =========================================================
# SMARTDOSEAI
# REALISTIC DIALYSIS RISK DATASET GENERATOR
# =========================================================
#
# IMPORTANT:
# - Loads the EXISTING kidney_dataset.csv
# - Does NOT regenerate kidney features
# - Does NOT change Kidney_Disease_Type
# - Changes ONLY Dialysis_Risk
#
# =========================================================

import os
import random
import numpy as np
import pandas as pd


# =========================================================
# SETTINGS
# =========================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

DATASET_PATH = "datasets/kidney_dataset.csv"


# =========================================================
# LOAD EXISTING DATASET
# =========================================================

if not os.path.exists(DATASET_PATH):

    raise FileNotFoundError(
        f"Dataset not found: {DATASET_PATH}"
    )


df = pd.read_csv(DATASET_PATH)

print("\n" + "=" * 70)
print("DIALYSIS RISK GENERATION")
print("=" * 70)

print(
    f"\nExisting dataset shape: {df.shape}"
)


# =========================================================
# REQUIRED FEATURES
# =========================================================

required_columns = [
    "eGFR",
    "Creatinine",
    "Blood_Urea",
    "Potassium",
    "Sodium",
    "Hemoglobin",
    "Uric_Acid",
    "Age"
]


missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]


if missing_columns:

    raise ValueError(
        "Missing required columns: "
        + ", ".join(missing_columns)
    )


# =========================================================
# HANDLE MISSING VALUES
#
# This is important.
#
# We use median values ONLY for calculating dialysis risk.
# We do NOT modify the original clinical columns.
# =========================================================

risk_features = {}

for column in required_columns:

    risk_features[column] = (
        pd.to_numeric(
            df[column],
            errors="coerce"
        )
        .fillna(
            df[column].median()
        )
    )


# =========================================================
# EXTRACT FEATURES
# =========================================================

egfr = risk_features["eGFR"]

creatinine = risk_features["Creatinine"]

blood_urea = risk_features["Blood_Urea"]

potassium = risk_features["Potassium"]

sodium = risk_features["Sodium"]

hemoglobin = risk_features["Hemoglobin"]

uric_acid = risk_features["Uric_Acid"]

age = risk_features["Age"]


# =========================================================
# NORMALIZED CLINICAL RISK FEATURES
# =========================================================

# ---------------------------------------------------------
# eGFR
# Lower eGFR = higher risk
# ---------------------------------------------------------

egfr_risk = 1 - np.clip(

    (egfr - 15) / 95,

    0,

    1

)


# ---------------------------------------------------------
# Creatinine
# Higher = higher risk
# ---------------------------------------------------------

creatinine_risk = np.clip(

    (creatinine - 0.5) / 5.5,

    0,

    1

)


# ---------------------------------------------------------
# Blood Urea
# ---------------------------------------------------------

urea_risk = np.clip(

    (blood_urea - 15) / 165,

    0,

    1

)


# ---------------------------------------------------------
# Potassium
# ---------------------------------------------------------

potassium_risk = np.clip(

    (potassium - 3.0) / 4.0,

    0,

    1

)


# ---------------------------------------------------------
# Sodium abnormality
# ---------------------------------------------------------

sodium_risk = np.clip(

    np.abs(sodium - 140) / 30,

    0,

    1

)


# ---------------------------------------------------------
# Hemoglobin
# Lower hemoglobin = greater risk
# ---------------------------------------------------------

hemoglobin_risk = 1 - np.clip(

    (hemoglobin - 6) / 14,

    0,

    1

)


# ---------------------------------------------------------
# Uric acid
# ---------------------------------------------------------

uric_acid_risk = np.clip(

    (uric_acid - 2) / 10,

    0,

    1

)


# ---------------------------------------------------------
# Age
# Secondary factor
# ---------------------------------------------------------

age_risk = np.clip(

    (age - 18) / 65,

    0,

    1

)


# =========================================================
# COMBINED CLINICAL RISK SCORE
# =========================================================

risk_score = (

    0.32 * egfr_risk

    +

    0.24 * creatinine_risk

    +

    0.16 * urea_risk

    +

    0.08 * potassium_risk

    +

    0.05 * sodium_risk

    +

    0.06 * hemoglobin_risk

    +

    0.05 * uric_acid_risk

    +

    0.04 * age_risk

)


# =========================================================
# ADD MODERATE CLINICAL VARIABILITY
# =========================================================

risk_score = (

    risk_score

    +

    np.random.normal(
        0,
        0.045,
        size=len(df)
    )

)


# =========================================================
# FORCE NUMERIC CLEANUP
# =========================================================

risk_score = pd.Series(
    risk_score
).replace(
    [np.inf, -np.inf],
    np.nan
).fillna(
    0.5
).to_numpy()


# =========================================================
# QUANTILE THRESHOLDS
#
# We explicitly create three meaningful groups.
#
# Approximate distribution:
#
# Low       ~40%
# Medium    ~35%
# High      ~25%
#
# =========================================================

low_medium_threshold = np.quantile(

    risk_score,

    0.40

)

medium_high_threshold = np.quantile(

    risk_score,

    0.75

)


# =========================================================
# INITIAL CLASSIFICATION
# =========================================================

dialysis_risk = np.where(

    risk_score >= medium_high_threshold,

    "High",

    np.where(

        risk_score >= low_medium_threshold,

        "Medium",

        "Low"

    )

)


# =========================================================
# ADD SMALL LABEL UNCERTAINTY
#
# Only 2% of observations are moved to a neighboring
# category.
#
# This keeps the relationship learnable but not perfect.
# =========================================================

for i in range(
    len(dialysis_risk)
):

    if random.random() < 0.02:

        current = dialysis_risk[i]

        if current == "High":

            dialysis_risk[i] = random.choice(
                [
                    "High",
                    "Medium"
                ]
            )

        elif current == "Medium":

            dialysis_risk[i] = random.choice(
                [
                    "Low",
                    "Medium",
                    "High"
                ]
            )

        else:

            dialysis_risk[i] = random.choice(
                [
                    "Low",
                    "Medium"
                ]
            )


# =========================================================
# REPLACE ONLY DIALYSIS_RISK
# =========================================================

df["Dialysis_Risk"] = dialysis_risk


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

df.to_csv(

    DATASET_PATH,

    index=False

)


# =========================================================
# VERIFY
# =========================================================

print(
    "\n" + "=" * 70
)

print(
    "DIALYSIS DATASET UPDATED SUCCESSFULLY"
)

print(
    "=" * 70
)

print(
    f"\nDataset shape: {df.shape}"
)


print(
    "\nKidney disease distribution:"
)

print(
    df[
        "Kidney_Disease_Type"
    ].value_counts()
)


print(
    "\nDialysis risk distribution:"
)

dialysis_counts = (
    df[
        "Dialysis_Risk"
    ]
    .value_counts()
)

print(
    dialysis_counts
)


print(
    "\nDialysis risk percentages:"
)

print(

    (
        df[
            "Dialysis_Risk"
        ]
        .value_counts(
            normalize=True
        )
        .mul(100)
        .round(2)
    )

)


# =========================================================
# SAFETY CHECK
# =========================================================

classes = set(
    df["Dialysis_Risk"].unique()
)

expected_classes = {
    "Low",
    "Medium",
    "High"
}


if classes != expected_classes:

    raise RuntimeError(

        "ERROR: Dialysis_Risk does not contain "
        "all three classes. Dataset was NOT "
        "considered valid."

    )


print(
    "\nAll three dialysis classes are present."
)

print(
    "\nKidney_Disease_Type was NOT changed."
)

print(
    "All kidney clinical features were NOT changed."
)

print(
    "Only Dialysis_Risk was regenerated."
)

print(
    "\nSaved to:"
)

print(
    DATASET_PATH
)

print(
    "=" * 70
)