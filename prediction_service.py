from explainability import generate_patient_explanation_realtime
from inference import predict_disease, predict_related_models
from patient_plain_language import doctor_view_summary, patient_view_from_shap


def _shap_rows(rows):
    safe = []
    for item in rows or []:
        safe.append(
            {
                "feature": item.get("feature"),
                "readable_feature": item.get("readable_feature"),
                "display_value": item.get("display_value") or "Not recorded",
                "clinical_value": item.get("clinical_value"),
                "strength": item.get("strength", "Mild"),
                "direction": item.get("direction"),
                "shap_value": float(item.get("shap_value") or 0),
                "absolute_shap": float(item.get("absolute_shap") or 0),
            }
        )
    return safe


def run_explained_prediction(disease_type, parameters, patient=None):
    result = predict_disease(disease_type, parameters, patient)
    shap_explanation = generate_patient_explanation_realtime(
        result["model_package"],
        result["model_input"],
        result["class_index"],
        result["model_input"].shape[1],
        original_parameters=result["merged_parameters"],
        raw_feature_df=result["raw_df"],
        disease_type=disease_type,
    )
    top = _shap_rows(shap_explanation.get("top_features"))
    dialysis_result = ""
    dialysis_probability = None
    if disease_type == "kidney":
        dialysis = predict_disease("dialysis", parameters, patient)
        dialysis_result = dialysis["prediction"]
        dialysis_probability = dialysis["model_probability"]
    return {
        "prediction": result["prediction"],
        "class_index": result["class_index"],
        "model_probability": result["model_probability"],
        "confidence_band": result["confidence_band"],
        "severity": result["severity"],
        "class_probabilities": result["class_probabilities"],
        "shap_explanation": {
            "top_features": top,
            "contributing_features": _shap_rows(shap_explanation.get("contributing_features")),
            "not_recorded_features": _shap_rows(shap_explanation.get("not_recorded_features")),
            "global_features": _shap_rows(shap_explanation.get("global_features")),
            "probabilities": [float(x) for x in shap_explanation.get("probabilities") or []],
            "predicted_class": shap_explanation.get("predicted_class"),
        },
        "patient_view": patient_view_from_shap(
            result["prediction"],
            top,
            any_missing=len(shap_explanation.get("not_recorded_features") or []) > 0
        ),
        "doctor_view": doctor_view_summary(
            result["prediction"],
            result["model_probability"],
            result["class_index"],
            result["confidence_band"],
        ),
        "related_models": predict_related_models(disease_type, parameters, patient),
        "dialysis_result": dialysis_result,
        "dialysis_probability": dialysis_probability,
        "disease_type": disease_type,
    }
