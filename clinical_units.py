"""Map model / OCR keys to patient-facing clinical labels and units."""

FEATURE_UNITS = {
    "HbA1c": "%",
    "Fasting_Glucose": "mg/dL",
    "Random_Glucose": "mg/dL",
    "Post_Meal_Glucose": "mg/dL",
    "Postprandial_Glucose": "mg/dL",
    "Insulin": "µU/mL",
    "BMI": "kg/m²",
    "Cholesterol": "mg/dL",
    "LDL": "mg/dL",
    "HDL": "mg/dL",
    "Triglycerides": "mg/dL",
    "Systolic_BP": "mmHg",
    "Diastolic_BP": "mmHg",
    "Heart_Rate": "bpm",
    "Troponin": "ng/L",
    "Creatinine": "mg/dL",
    "Blood_Urea": "mg/dL",
    "BUN": "mg/dL",
    "eGFR": "mL/min/1.73m²",
    "Potassium": "mEq/L",
    "Sodium": "mEq/L",
    "Calcium": "mg/dL",
    "Uric_Acid": "mg/dL",
    "Hemoglobin": "g/dL",
    "Age": "years",
    "Water_Intake": "L/day",
}

DISPLAY_NAMES = {
    "HbA1c": "HbA1c",
    "Fasting_Glucose": "Fasting Glucose",
    "Random_Glucose": "Random Glucose",
    "Post_Meal_Glucose": "Post-meal Glucose",
    "Postprandial_Glucose": "Postprandial Glucose",
    "Systolic_BP": "Systolic BP",
    "Diastolic_BP": "Diastolic BP",
    "Heart_Rate": "Heart Rate",
    "Blood_Urea": "Blood Urea / BUN",
    "Urine_Protein": "Urine Protein",
    "Family_History": "Family History",
    "Physical_Activity": "Physical Activity",
    "Chest_Pain": "Chest Pain",
    "ECG": "ECG",
}


def strip_transformer_prefix(feature_name):
    name = str(feature_name)
    for prefix in ("numerical__", "categorical__", "num__", "cat__"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    if "=" in name:
        name = name.split("=", 1)[0]
    # one-hot: Physical_Activity_Low -> Physical_Activity
    for cat in ("Physical_Activity", "Gender", "ECG", "Urine_Protein", "Stress_Level", "Salt_Intake", "Exercise"):
        if name.startswith(cat + "_"):
            name = cat
            break
    return name


def readable_feature_name(feature_name):
    base = strip_transformer_prefix(feature_name)
    if base in DISPLAY_NAMES:
        return DISPLAY_NAMES[base]
    return base.replace("_", " ")


def format_clinical_value(feature_name, value):
    base = strip_transformer_prefix(feature_name)
    unit = FEATURE_UNITS.get(base, "")
    if value is None or value == "":
        return "Not recorded"
    try:
        numeric = float(value)
        if numeric.is_integer():
            shown = str(int(numeric))
        else:
            shown = f"{numeric:.2f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        shown = str(value)
        numeric = None
    if unit and numeric is not None:
        return f"{shown} {unit}" if unit != "%" else f"{shown}%"
    return shown
