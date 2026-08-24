"""Runtime ML inference: raw clinical values -> preprocessor -> model."""

from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd

from feature_builder import (
    build_diabetes_features,
    build_heart_features,
    build_hypertension_features,
    build_kidney_features,
)

LABELS = {
    "heart": {
        0: "Arrhythmia",
        1: "Coronary Artery Disease",
        2: "Heart Failure",
        3: "Hypertension",
        4: "Valve Disease",
    },
    "diabetes": {
        0: "Gestational Diabetes",
        1: "Prediabetes",
        2: "Type 1 Diabetes",
        3: "Type 2 Diabetes",
    },
    "kidney": {
        0: "Acute Kidney Injury",
        1: "Chronic Kidney Disease",
        2: "Diabetic Nephropathy",
        3: "Glomerulonephritis",
        4: "Kidney Stones",
    },
    "dialysis": {
        0: "High",
        1: "Low",
        2: "Medium",
    },
    "hypertension": {
        0: "Primary Hypertension",
        1: "Pulmonary Hypertension",
        2: "Resistant Hypertension",
        3: "Secondary Hypertension",
    },
}

FEATURE_BUILDERS = {
    "heart": build_heart_features,
    "diabetes": build_diabetes_features,
    "kidney": build_kidney_features,
    "dialysis": build_kidney_features,
    "hypertension": build_hypertension_features,
}

_PACKAGES = {}
_PREPROCESSORS = {}


def _project_path(*parts):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), *parts)


def load_model_package(disease_type):
    disease_type = (disease_type or "").lower()
    if disease_type not in _PACKAGES:
        path = _project_path("trained_models", f"{disease_type}_best_model.pkl")
        _PACKAGES[disease_type] = joblib.load(path)
    return _PACKAGES[disease_type]


def load_preprocessor(disease_type):
    disease_type = (disease_type or "").lower()
    if disease_type not in _PREPROCESSORS:
        path = _project_path("preprocessing", f"{disease_type}_preprocessor.pkl")
        _PREPROCESSORS[disease_type] = joblib.load(path)
    return _PREPROCESSORS[disease_type]


def merge_patient_context(parameters, patient=None):
    """Copy OCR/edited labs and overlay registration demographics when present."""
    merged = dict(parameters or {})
    patient = patient or {}
    if patient.get("age") not in (None, ""):
        merged.setdefault("Age", patient.get("age"))
    if patient.get("gender") not in (None, ""):
        merged.setdefault("Gender", patient.get("gender"))
    if patient.get("bmi") not in (None, ""):
        merged.setdefault("BMI", patient.get("bmi"))
    smoking = str(patient.get("smoking") or "").strip().lower()
    if smoking in ("yes", "1", "true", "smoker"):
        merged.setdefault("Smoking", 1)
    elif smoking in ("no", "0", "false"):
        merged.setdefault("Smoking", 0)
    return merged


def build_raw_features(disease_type, parameters):
    builder = FEATURE_BUILDERS.get((disease_type or "").lower())
    if builder is None:
        raise ValueError(f"No feature builder for {disease_type}")
    return builder(parameters)


def transform_features(disease_type, raw_df):
    preprocessor = load_preprocessor(disease_type)
    package = load_model_package(disease_type)
    transformed = preprocessor.transform(raw_df)
    names = list(package.get("feature_names") or preprocessor.get_feature_names_out())
    return pd.DataFrame(transformed, columns=names)


def decode_label(disease_type, class_index):
    mapping = LABELS.get((disease_type or "").lower(), {})
    try:
        return mapping[int(class_index)]
    except (KeyError, TypeError, ValueError):
        return str(class_index)


def probability_to_confidence_band(probability):
    """Presentation band. This is not medical certainty."""
    p = float(probability or 0)
    if p >= 0.85:
        return "High"
    if p >= 0.65:
        return "Moderate"
    return "Low"


def severity_from_probability(probability):
    p = float(probability or 0) * 100
    if p >= 90:
        return "High"
    if p >= 75:
        return "Medium"
    return "Low"


def predict_disease(disease_type, parameters, patient=None):
    """
    Returns a JSON-serializable prediction dict.
    Uses the saved sklearn preprocessor so inference matches training.
    """
    disease_type = (disease_type or "").lower()
    merged = merge_patient_context(parameters, patient)
    raw_df = build_raw_features(disease_type, merged)
    model_input = transform_features(disease_type, raw_df)
    package = load_model_package(disease_type)
    model = package["model"]
    class_index = int(model.predict(model_input)[0])
    probabilities = model.predict_proba(model_input)[0]
    probabilities = [float(x) for x in np.asarray(probabilities).ravel()]
    predicted_probability = probabilities[class_index] if class_index < len(probabilities) else 0.0
    class_probabilities = []
    for idx, prob in enumerate(probabilities):
        class_probabilities.append(
            {
                "class_index": idx,
                "label": decode_label(disease_type, idx),
                "probability": round(prob * 100, 1),
            }
        )
    class_probabilities.sort(key=lambda row: row["probability"], reverse=True)
    return {
        "disease_type": disease_type,
        "class_index": class_index,
        "prediction": decode_label(disease_type, class_index),
        "model_probability": round(predicted_probability * 100, 1),
        "confidence_band": probability_to_confidence_band(predicted_probability),
        "severity": severity_from_probability(predicted_probability),
        "class_probabilities": class_probabilities,
        "raw_df": raw_df,
        "model_input": model_input,
        "merged_parameters": merged,
        "model_package": package,
    }


def predict_related_models(primary_disease, parameters, patient=None):
    """Re-run related disease models for Digital Twin what-if scenarios."""
    primary_disease = (primary_disease or "").lower()
    related = {
        "diabetes": ["diabetes", "kidney", "hypertension", "heart"],
        "heart": ["heart", "hypertension", "kidney", "dialysis"],
        "kidney": ["kidney", "dialysis", "hypertension", "heart"],
        "hypertension": ["hypertension", "heart", "kidney"],
    }.get(primary_disease, [primary_disease])

    outputs = []
    for name in related:
        try:
            result = predict_disease(name, parameters, patient)
        except Exception as exc:
            outputs.append(
                {
                    "disease_type": name,
                    "prediction": "Unavailable",
                    "model_probability": None,
                    "confidence_band": "Low",
                    "error": str(exc),
                }
            )
            continue
        outputs.append(
            {
                "disease_type": name,
                "prediction": result["prediction"],
                "model_probability": result["model_probability"],
                "confidence_band": result["confidence_band"],
                "severity": result["severity"],
            }
        )
    return outputs
