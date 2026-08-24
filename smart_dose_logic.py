import json

from clinical_safety import extra_safety_checks
from paths import MEDICINES_FILE

_ORGAN_SAFE_HINTS = ("safe", "protective", "kidney protective")

def evaluate_contraindications(medicine_item, patient_parameters):
    if not patient_parameters:
        return True, ""
        
    contraindications = medicine_item.get("contraindications", [])
    for rule in contraindications:
        param = rule.get("parameter")
        cond = rule.get("condition")
        val = rule.get("value")
        
        # Check if the param exists in patient_parameters
        # In a real app we'd map standard lab names to OCR keys
        patient_val = None
        for key, p_val in patient_parameters.items():
            if param.lower() in key.lower():
                patient_val = p_val
                break
                
        if patient_val is not None:
            try:
                numeric_val = float(patient_val)
                numeric_rule = float(val)
                if cond == ">" and numeric_val > numeric_rule:
                    return False, f"{param} is {numeric_val}, which is {cond} {numeric_rule}."
                if cond == "<" and numeric_val < numeric_rule:
                    return False, f"{param} is {numeric_val}, which is {cond} {numeric_rule}."
                if cond == "==" and numeric_val == numeric_rule:
                    return False, f"{param} is exactly {numeric_rule}."
            except ValueError:
                pass
                
    return True, ""

def _pick_recommendation(
    meds,
    disease,
    severity_normalized,
    ai_prediction,
    patient_parameters,
    current_medications="",
):
    """severity_normalized is mild | moderate | severe."""
    disease = (disease or "diabetes").lower()
    
    candidates = []

    if disease == "heart" and ai_prediction:
        block = meds.get("heart_by_diagnosis", {}).get(ai_prediction, {})
        if block:
            candidates = (
                block.get(severity_normalized)
                or block.get("moderate")
                or block.get("mild")
                or block.get("severe")
                or []
            )

    if not candidates and disease == "diabetes" and ai_prediction:
        block = meds.get("diabetes_by_diagnosis", {}).get(ai_prediction, {})
        if block:
            candidates = (
                block.get(severity_normalized)
                or block.get("moderate")
                or block.get("mild")
                or block.get("severe")
                or []
            )

    if not candidates:
        if disease not in meds:
            disease = "diabetes"
        tier = meds[disease]
        candidates = tier.get(severity_normalized) or tier.get("moderate") or tier.get("mild") or []

    rejected_explanations = []

    for item in candidates:
        is_safe, reason = evaluate_contraindications(item, patient_parameters)
        chosen = dict(item)
        extra = extra_safety_checks(
            chosen.get("name"),
            patient_parameters,
            current_medications,
        )
        hard_rejects = [f for f in extra if f["status"] == "reject"]
        flags = [f for f in extra if f["status"] == "flag"]
        if is_safe and not hard_rejects:
            chosen["rejected_explanations"] = list(rejected_explanations)
            chosen["safety_flags"] = flags
            chosen["selection_reason"] = (
                f"{chosen.get('name')} was selected for {disease} "
                f"({severity_normalized} severity) after contraindication, "
                "kidney, heart, and interaction checks in this demo engine."
            )
            if flags:
                chosen["review_required"] = True
            return chosen
        why = reason or "; ".join(f["reason"] for f in hard_rejects) or "safety rule"
        rejected_explanations.append(f"{item['name']} was rejected because {why}")

    # If all candidates rejected
    if candidates:
        fallback = dict(candidates[0])
        fallback["rejected_explanations"] = rejected_explanations
        fallback["name"] = "No safe medication found (Review Required)"
        fallback["safe_dosage"] = "N/A"
        fallback["organ_risks"] = ["All candidates contraindicated"]
        fallback["review_required"] = True
        fallback["safety_flags"] = []
        fallback["selection_reason"] = (
            "Every candidate regimen failed a demo safety check. A clinician must choose therapy."
        )
        return fallback
        
    return None


def check_safety_state(recommendation, patient_parameters):
    kidney_safety = recommendation.get("kidney_safety", "Standard monitoring")
    heart_safety = recommendation.get("heart_safety", "Standard monitoring")
    status = "Approved"
    
    contraindications = recommendation.get("contraindications", [])
    
    # Check Creatinine & eGFR availability
    creatinine_val = None
    egfr_val = None
    if patient_parameters:
        for k, v in patient_parameters.items():
            if "creatinine" in k.lower():
                if v not in (None, "", "Not recorded", "N/A", "n/a"):
                    try:
                        creatinine_val = float(v)
                    except ValueError:
                        pass
            elif "egfr" in k.lower():
                if v not in (None, "", "Not recorded", "N/A", "n/a"):
                    try:
                        egfr_val = float(v)
                    except ValueError:
                        pass
                
    if creatinine_val is None or egfr_val is None:
        kidney_safety = "⚠ Unable to fully assess kidney safety. Required kidney-function data were not available. Doctor review required."
        status = "Requires Doctor Review"
    else:
        # Both are available, evaluate contraindications
        kidney_rules = [r for r in contraindications if r.get("parameter", "").lower() in ("creatinine", "egfr")]
        
        triggered = False
        for rule in kidney_rules:
            param = rule.get("parameter", "").lower()
            val_to_check = creatinine_val if param == "creatinine" else egfr_val
            cond = rule.get("condition", ">")
            try:
                rule_val = float(rule.get("value", 1.5))
                if (cond == ">" and val_to_check > rule_val) or (cond == "<" and val_to_check < rule_val):
                    triggered = True
            except ValueError:
                pass
        
        if triggered:
            kidney_safety = "⚠ Contraindication triggered"
            status = "Requires Doctor Review"
        else:
            kidney_safety = "✓ Safety check passed"

    # Check ALT / Heart Safety (or liver rules)
    alt_rules = [r for r in contraindications if r.get("parameter", "").lower() == "alt"]
    alt_val = None
    if patient_parameters:
        for k, v in patient_parameters.items():
            if k.lower() == "alt":
                if v not in (None, "", "Not recorded", "N/A", "n/a"):
                    try:
                        alt_val = float(v)
                    except ValueError:
                        pass
                break
                
    if alt_rules:
        rule = alt_rules[0]
        cond = rule.get("condition", ">")
        rule_val = float(rule.get("value", 150))
        
        if alt_val is None:
            heart_safety = "⚠ Unable to fully assess. ALT was not available in the confirmed laboratory data. Doctor review required."
            status = "Requires Doctor Review"
        else:
            if (cond == ">" and alt_val > rule_val) or (cond == "<" and alt_val < rule_val):
                heart_safety = "⚠ Contraindication triggered"
                status = "Requires Doctor Review"
            else:
                heart_safety = "✓ Safety check passed"

    return kidney_safety, heart_safety, status


def _format_rec(recommendation, patient_parameters=None):
    patient_parameters = patient_parameters or {}
    risks = recommendation.get("organ_risks", [])
    risk_text = ", ".join(risks) if isinstance(risks, list) else str(risks)
    lowered = risk_text.lower()
    organ_safety = "Safe"
    if not any(h in lowered for h in _ORGAN_SAFE_HINTS):
        organ_safety = "Moderate Risk"
    if any(x in lowered for x in ("high organ", "bleeding", "hypoglycemia")):
        organ_safety = "Elevated Risk"
        
    kidney_safety, heart_safety, computed_status = check_safety_state(recommendation, patient_parameters)
    
    status = computed_status
    if recommendation.get("review_required") or "No safe medication" in recommendation["name"]:
        status = "Requires Doctor Review"
        organ_safety = "High Risk"

    flags = recommendation.get("safety_flags") or []
    if flags and status != "Requires Doctor Review":
        status = "Requires Doctor Review"

    return {
        "medicine": recommendation["name"],
        "dose": recommendation.get("safe_dosage", "N/A"),
        "max_dose": recommendation.get("max_dosage", "N/A"),
        "organ_safety": organ_safety,
        "kidney_safety": kidney_safety,
        "heart_safety": heart_safety,
        "interactions": recommendation.get("interactions", "None highlighted"),
        "side_effects": risk_text,
        "rejected_explanations": recommendation.get("rejected_explanations", []),
        "selection_reason": recommendation.get("selection_reason", ""),
        "safety_flags": flags,
        "status": status,
        "disclaimer": (
            "Educational decision-support demo only. This is not autonomous prescribing "
            "and does not replace a licensed clinician."
        ),
    }


def get_smart_dose(
    disease_type,
    severity,
    ai_prediction=None,
    patient_parameters=None,
    current_medications="",
):
    if patient_parameters is None:
        patient_parameters = {}
        
    try:
        with open(MEDICINES_FILE, encoding="utf-8") as f:
            meds = json.load(f)
    except FileNotFoundError:
        return {
            "medicine": "Unknown",
            "dose": "N/A",
            "max_dose": "N/A",
            "organ_safety": "Unknown",
            "kidney_safety": "Unknown",
            "heart_safety": "Unknown",
            "interactions": "Unknown",
            "side_effects": "None",
            "rejected_explanations": [],
            "selection_reason": "",
            "safety_flags": [],
            "status": "Error",
            "disclaimer": "Educational decision-support demo only.",
        }

    sev_map = {"high": "severe", "medium": "moderate", "low": "mild"}
    raw = (severity or "moderate").strip().lower()
    sev = sev_map.get(raw, raw)
    if sev not in ("mild", "moderate", "severe"):
        sev = "moderate"

    recommendation = _pick_recommendation(
        meds,
        disease_type,
        sev,
        (ai_prediction or "").strip() or None,
        patient_parameters,
        current_medications,
    )

    if not recommendation:
        return {
            "medicine": "Unknown",
            "dose": "N/A",
            "max_dose": "N/A",
            "organ_safety": "Unknown",
            "kidney_safety": "Unknown",
            "heart_safety": "Unknown",
            "interactions": "Unknown",
            "side_effects": "No matching regimen in medicines.json",
            "rejected_explanations": [],
            "selection_reason": "",
            "safety_flags": [],
            "status": "Unknown",
            "disclaimer": "Educational decision-support demo only.",
        }

    return _format_rec(recommendation, patient_parameters)
