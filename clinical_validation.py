"""Check OCR / form values before they enter the ML models."""

from __future__ import annotations

KEY_ALIASES = {
    "fasting glucose": "Fasting_Glucose",
    "fasting_glucose": "Fasting_Glucose",
    "fasting plasma glucose": "Fasting_Glucose",
    "fasting_plasma_glucose": "Fasting_Glucose",
    "hba1c": "HbA1c",
    "hba1c ngsp": "HbA1c",
    "hba1c_ngsp": "HbA1c",
    "systolic bp": "Systolic_BP",
    "systolic_bp": "Systolic_BP",
    "diastolic bp": "Diastolic_BP",
    "diastolic_bp": "Diastolic_BP",
    "heart rate": "Heart_Rate",
    "heart_rate": "Heart_Rate",
    "bun": "Blood_Urea",
    "blood urea": "Blood_Urea",
    "blood_urea": "Blood_Urea",
    "egfr": "eGFR",
    "creatinine": "Creatinine",
    "cholesterol": "Cholesterol",
    "insulin": "Insulin",
    "bmi": "BMI",
    "potassium": "Potassium",
    "2 hr postprandial glucose": "Post_Meal_Glucose",
    "2_hr_postprandial_glucose": "Post_Meal_Glucose",
    "postprandial glucose": "Post_Meal_Glucose",
    "postprandial_glucose": "Post_Meal_Glucose",
    "post meal glucose": "Post_Meal_Glucose",
    "post_meal_glucose": "Post_Meal_Glucose",
    "random glucose": "Random_Glucose",
    "random_glucose": "Random_Glucose",
}

RANGES = {
    "HbA1c": (3.0, 20.0),
    "Fasting_Glucose": (30.0, 800.0),
    "Insulin": (1.0, 300.0),
    "BMI": (10.0, 80.0),
    "Systolic_BP": (60.0, 250.0),
    "Diastolic_BP": (30.0, 150.0),
    "Cholesterol": (50.0, 600.0),
    "Heart_Rate": (30.0, 250.0),
    "Creatinine": (0.1, 15.0),
    "Blood_Urea": (1.0, 150.0),
    "eGFR": (1.0, 150.0),
    "Potassium": (1.5, 9.0),
    "LDL": (20.0, 400.0),
    "HDL": (10.0, 120.0),
    "Triglycerides": (20.0, 2000.0),
}

REQUIRED = {
    "diabetes": ["HbA1c", "Fasting_Glucose"],
    "heart": ["Systolic_BP"],
    "kidney": ["Creatinine"],
}

CRITICAL_ANY = {
    "diabetes": ["HbA1c", "Fasting_Glucose"],
    "heart": ["Systolic_BP", "Cholesterol"],
    "kidney": ["Creatinine", "Blood_Urea"],
}


def normalize_parameter_key(key):
    text = str(key or "").strip()
    alias = KEY_ALIASES.get(text.lower().replace("-", "_"))
    if alias:
        return alias
    return text.replace(" ", "_")


def normalize_parameters(parameters):
    out = {}
    for key, value in (parameters or {}).items():
        out[normalize_parameter_key(key)] = value
    return out


def _present(parameters, key):
    value = parameters.get(key)
    return value not in (None, "", "N/A", "n/a")


def validate_clinical_data(disease_type, parameters):
    """
    Returns (is_valid, errors).
    Missing required labs block prediction; impossible ranges also block.
    """
    errors = []
    params = normalize_parameters(parameters)
    disease = (disease_type or "").lower()

    required = REQUIRED.get(disease, [])
    missing_required = [name.replace("_", " ") for name in required if not _present(params, name)]
    if missing_required:
        errors.append(
            "Missing required information. Please provide: "
            + ", ".join(f"• {item}" for item in missing_required)
        )
    else:
        any_keys = CRITICAL_ANY.get(disease, [])
        if any_keys and not any(_present(params, key) for key in any_keys):
            readable = ", ".join(key.replace("_", " ") for key in any_keys)
            errors.append(f"Missing critical lab parameters for {disease.title()} analysis ({readable}).")

    for name, (min_val, max_val) in RANGES.items():
        if not _present(params, name):
            continue
        raw = params[name]
        try:
            numeric_val = float(raw)
        except (TypeError, ValueError):
            errors.append(f"{name.replace('_', ' ')} contains non-numeric characters: '{raw}'")
            continue
        if numeric_val < min_val or numeric_val > max_val:
            errors.append(
                f"{name.replace('_', ' ')} value {numeric_val} is outside biologically probable range "
                f"({min_val} - {max_val})."
            )

    allowed_categorical = {
        "Gender": {"male", "female", "other", "m", "f"},
        "ECG": {"normal", "abnormal", "st-t abnormality", "lv hypertrophy"},
        "Urine_Protein": {"negative", "trace", "1+", "2+", "3+", "positive"},
        "Physical_Activity": {"low", "moderate", "high"},
    }
    for name, allowed in allowed_categorical.items():
        if not _present(params, name):
            continue
        if str(params[name]).strip().lower() not in allowed:
            errors.append(
                f"{name.replace('_', ' ')} has an unexpected value: '{params[name]}'."
            )

    return len(errors) == 0, errors


def parameters_from_form(form):
    """Read confirmed OCR fields from a Flask form (keys prefixed with lab_)."""
    out = {}
    for key, value in form.items():
        if not key.startswith("lab_"):
            continue
        name = key[4:]
        text = str(value or "").strip()
        if text == "":
            continue
        try:
            out[name] = float(text)
        except ValueError:
            out[name] = text
    return normalize_parameters(out)
