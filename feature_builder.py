# =========================================================
# SMARTDOSEAI - FEATURE BUILDER
# =========================================================

import pandas as pd


# =========================================================
# HELPER
# =========================================================

def create_dataframe(data):

    return pd.DataFrame(
        [data]
    )


# =========================================================
# HEART DISEASE FEATURES
# =========================================================

def build_heart_features(parameters):

    data = {

        "Age":
            parameters.get(
                "Age"
            ),

        "Gender":
            parameters.get(
                "Gender"
            ),

        "Systolic_BP":
            parameters.get(
                "Systolic_BP"
            ),

        "Diastolic_BP":
            parameters.get(
                "Diastolic_BP"
            ),

        "Heart_Rate":
            parameters.get(
                "Heart_Rate"
            ),

        "Troponin":
            parameters.get(
                "Troponin"
            ),

        "Cholesterol":
            parameters.get(
                "Cholesterol"
            ),

        "LDL":
            parameters.get(
                "LDL"
            ),

        "HDL":
            parameters.get(
                "HDL"
            ),

        "Triglycerides":
            parameters.get(
                "Triglycerides"
            ),

        "ECG":
            parameters.get(
                "ECG"
            ),

        "Smoking":
            parameters.get(
                "Smoking"
            ),

        "Obesity":
            parameters.get(
                "Obesity"
            ),

        "Chest_Pain":
            parameters.get(
                "Chest_Pain"
            ),

        "Breathlessness":
            parameters.get(
                "Breathlessness"
            ),

        "Dizziness":
            parameters.get(
                "Dizziness"
            ),

        "Family_History":
            parameters.get(
                "Family_History"
            )

    }

    return create_dataframe(
        data
    )


# =========================================================
# DIABETES FEATURES
# =========================================================

def build_diabetes_features(parameters):

    data = {

        "Age":
            parameters.get(
                "Age"
            ),

        "Gender":
            parameters.get(
                "Gender"
            ),

        "BMI":
            parameters.get(
                "BMI"
            ),

        "Fasting_Glucose":
            parameters.get(
                "Fasting_Glucose"
            ),

        "Random_Glucose":
            parameters.get(
                "Random_Glucose",
                parameters.get(
                    "Postprandial_Glucose"
                )
            ),

        "Post_Meal_Glucose":
            parameters.get(
                "Post_Meal_Glucose",
                parameters.get(
                    "Postprandial_Glucose"
                )
            ),

        "HbA1c":
            parameters.get(
                "HbA1c"
            ),

        "Insulin":
            parameters.get(
                "Insulin"
            ),

        "Cholesterol":
            parameters.get(
                "Cholesterol"
            ),

        "Triglycerides":
            parameters.get(
                "Triglycerides"
            ),

        "Systolic_BP":
            parameters.get(
                "Systolic_BP"
            ),

        "Diastolic_BP":
            parameters.get(
                "Diastolic_BP"
            ),

        "Pregnancy":
            parameters.get(
                "Pregnancy"
            ),

        "Family_History":
            parameters.get(
                "Family_History"
            ),

        "Physical_Activity":
            parameters.get(
                "Physical_Activity"
            ),

        "Excessive_Thirst":
            parameters.get(
                "Excessive_Thirst"
            ),

        "Frequent_Urination":
            parameters.get(
                "Frequent_Urination"
            ),

        "Blurred_Vision":
            parameters.get(
                "Blurred_Vision"
            ),

        "Water_Intake":
            parameters.get(
                "Water_Intake"
            )

    }

    return create_dataframe(
        data
    )


# =========================================================
# KIDNEY FEATURES
# =========================================================
#
# IMPORTANT:
#
# Dialysis_Risk is NOT included.
#
# Kidney disease model:
#
#     Inputs → Kidney_Disease_Type
#
# Dialysis model:
#
#     Inputs → Dialysis_Risk
#
# Both models therefore receive the same patient
# clinical features, without using one target to
# predict the other.
# =========================================================

def build_kidney_features(parameters):

    data = {

        "Age":
            parameters.get(
                "Age"
            ),

        "Gender":
            parameters.get(
                "Gender"
            ),

        "Creatinine":
            parameters.get(
                "Creatinine"
            ),

        "Blood_Urea":
            parameters.get(
                "Blood_Urea"
            ),

        "eGFR":
            parameters.get(
                "eGFR"
            ),

        "Potassium":
            parameters.get(
                "Potassium"
            ),

        "Sodium":
            parameters.get(
                "Sodium"
            ),

        "Calcium":
            parameters.get(
                "Calcium"
            ),

        "Uric_Acid":
            parameters.get(
                "Uric_Acid"
            ),

        "Hemoglobin":
            parameters.get(
                "Hemoglobin"
            ),

        "Urine_Protein":
            parameters.get(
                "Urine_Protein"
            ),

        "Diabetes":
            parameters.get(
                "Diabetes"
            ),

        "Hypertension":
            parameters.get(
                "Hypertension"
            ),

        "Swelling":
            parameters.get(
                "Swelling"
            ),

        "Fatigue":
            parameters.get(
                "Fatigue"
            ),

        "Appetite_Loss":
            parameters.get(
                "Appetite_Loss"
            )

    }

    return create_dataframe(
        data
    )


# =========================================================
# HYPERTENSION FEATURES
# =========================================================

def build_hypertension_features(parameters):

    data = {

        "Age":
            parameters.get(
                "Age"
            ),

        "Gender":
            parameters.get(
                "Gender"
            ),

        "Systolic_BP":
            parameters.get(
                "Systolic_BP"
            ),

        "Diastolic_BP":
            parameters.get(
                "Diastolic_BP"
            ),

        "BMI":
            parameters.get(
                "BMI"
            ),

        "Smoking":
            parameters.get(
                "Smoking"
            ),

        "Alcohol":
            parameters.get(
                "Alcohol"
            ),

        "Family_History":
            parameters.get(
                "Family_History"
            ),

        "Stress_Level":
            parameters.get(
                "Stress_Level"
            ),

        "Cholesterol":
            parameters.get(
                "Cholesterol"
            ),

        "Oxygen_Level":
            parameters.get(
                "Oxygen_Level"
            ),

        "Heart_Rate":
            parameters.get(
                "Heart_Rate"
            ),

        "Salt_Intake":
            parameters.get(
                "Salt_Intake"
            ),

        "Exercise":
            parameters.get(
                "Exercise"
            )

    }

    return create_dataframe(
        data
    )