from ocr.extract_parameters import (
    extract_diabetes_parameters,
    extract_heart_parameters,
    extract_kidney_parameters,
)
from ocr.report_sections import (
    canonicalize_lab_parameter_keys,
    extract_laboratory_investigations,
    extract_stated_diagnosis_from_text,
    harmonize_heart_aliases,
)
from ocr.extract_text import extract_report_text
from clinical_validation import normalize_parameters


def extract_labs_from_file(disease_type, file_path):
    text = extract_report_text(file_path)
    embedded_dx = extract_stated_diagnosis_from_text(text)
    embedded_labs = extract_laboratory_investigations(text)
    disease = (disease_type or "").lower()

    if disease == "heart":
        extracted = extract_heart_parameters(text)
        extracted.update(embedded_labs)
        harmonize_heart_aliases(extracted)
    elif disease == "diabetes":
        extracted = extract_diabetes_parameters(text)
        extracted.update(embedded_labs)
    elif disease == "kidney":
        extracted = extract_kidney_parameters(text)
        extracted.update(embedded_labs)
    else:
        extracted = dict(embedded_labs)

    extracted = canonicalize_lab_parameter_keys(normalize_parameters(extracted))
    return {
        "text": text,
        "embedded_diagnosis": embedded_dx,
        "parameters": extracted,
    }


def suggested_review_fields(disease_type, parameters):
    extras = {
        "diabetes": ["HbA1c", "Fasting_Glucose", "Insulin", "BMI", "Post_Meal_Glucose"],
        "heart": ["Systolic_BP", "Diastolic_BP", "Heart_Rate", "Cholesterol", "LDL", "HDL", "Troponin"],
        "kidney": ["Creatinine", "Blood_Urea", "eGFR", "Potassium", "Hemoglobin"],
    }
    keys = list(parameters.keys())
    for name in extras.get((disease_type or "").lower(), []):
        if name not in keys:
            keys.append(name)
    return keys
