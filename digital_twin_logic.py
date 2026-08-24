from ocr.report_sections import canonicalize_lab_parameter_keys
from clinical_units import format_clinical_value, readable_feature_name


def format_medicine_recommendation(smart_dose):
    if not smart_dose or not isinstance(smart_dose, dict):
        return "No Smart Dose plan saved yet."
    med = smart_dose.get("medicine")
    dose = smart_dose.get("dose")
    if not med or med in (0, "0", "Unknown", "N/A"):
        return "No Smart Dose plan saved yet."
    if dose and dose not in ("N/A", "0", 0, ""):
        return f"{med} ({dose})"
    return str(med)


def simulate_twin(appt, smart_dose):
    disease = (appt.get("disease_type") or "unknown").lower()
    pred = appt.get("ai_prediction") or ""
    params_norm = canonicalize_lab_parameter_keys(appt.get("parameters") or {})
    safety = (smart_dose or {}).get("organ_safety", "") or ""
    medicine_line = format_medicine_recommendation(smart_dose)

    preview_keys = list(params_norm.keys())[:12]
    lab_bits = [f"{k}: {params_norm[k]}" for k in preview_keys]
    labs_summary = (
        "; ".join(lab_bits) if lab_bits else "No lab numbers were saved for this visit."
    )

    dialysis_prob = "Low"
    heart_attack_risk = "Low"
    kidney_status = "Looks steady in this simulation"
    heart_status = "Looks steady in this simulation"
    kidney_note = (
        "In this demo, kidneys stay in the background unless kidney disease "
        "or higher-risk meds are in play."
    )
    overall = (
        "This picture blends your lab summary, AI label, "
        "and Smart Dose idea for teaching—not a bedside monitor."
    )

    if disease == "kidney":
        kidney_status = "Kidneys are the focus—fluid and salts matter here."
        kidney_note = "The simulator watches fluids, potassium, and blood pressure together."
        dialysis_prob = str(appt.get("dialysis_risk") or "Moderate")
        heart_status = "Heart can feel extra strain if fluid swings are large."

    if disease == "heart":
        kidney_note = (
            "Some heart drugs affect kidney blood flow and salts—labs may need repeats."
        )
        if (
            "PSVT" in pred
            or "Supraventricular Tachycardia" in pred
            or pred == "Arrhythmia"
        ):
            heart_status = "Heartbeat pattern suggests a rhythm issue such as PSVT/SVT—not blocked arteries by itself."
            heart_attack_risk = "Between low and medium until ER/clinic checks rule out a heart attack."
            dialysis_prob = "Low"
            overall = (
                "This view stresses steady heart rate, watching blood pressure dips, "
                "and tying that to your Smart Dose plan plus the vitals/labs pulled from the file."
            )
        elif pred == "Coronary Artery Disease":
            heart_status = "Higher concern for strained heart muscle from narrowed arteries."
            heart_attack_risk = "Elevated until your team finishes their tests."
            overall = (
                "Cholesterol and clot-prevention plans are only examples here—real treatment "
                "follows your cardiologist."
            )
        elif pred == "Heart Failure":
            heart_status = "Heart may be sensitive to extra fluid or swelling."
            heart_attack_risk = "Medium"
            dialysis_prob = "Medium if heart and kidney problems stack together"
            overall = "Balancing fluid (too much vs too little) is the main story in this demo."
        elif pred == "Hypertension":
            heart_status = "Heart working against higher blood pressure."
            heart_attack_risk = "Medium if pressure stays high without treatment"
            overall = "Focus is on safe blood-pressure goals and dizzy spells when standing."
        else:
            heart_status = "General heart follow-up is appropriate."
            heart_attack_risk = "Medium"

    kidney_params = ["Creatinine", "BUN", "Blood_Urea", "eGFR", "Potassium", "Hemoglobin"]
    has_kidney_data = any(k in params_norm for k in kidney_params)

    if disease == "diabetes":
        if has_kidney_data:
            kidney_status = "Kidneys need extra watch when blood sugar runs high for years."
            kidney_note = "Medicine choices in the demo favor kidney-safe options when possible."
        else:
            kidney_status = "Glucose-related kidney risk indicator: Elevated\nNote: This is a model/rule-based simulation, not a kidney-function measurement."
            kidney_note = "Actual kidney-function laboratory parameters were not recorded."
        overall = "Combines sugar goals with kidney and general safety reminders."

    if "Elevated" in safety:
        kidney_status += " — plan closer kidney lab checks in real care."
        kidney_note += " This regimen is tagged higher-risk; kidney numbers should be watched."

    # Organ colors: primary at-risk system in red for the active disease module (demo UI).
    kidney_color = "#10b981"
    heart_color = "#38bdf8"
    liver_color = "rgba(59, 130, 246, 0.55)"

    if disease == "kidney":
        kidney_color = "#ef4444"
        heart_color = "#38bdf8"
        liver_color = "rgba(59, 130, 246, 0.35)"

    elif disease == "heart":
        heart_color = "#ef4444"
        if dialysis_prob == "High" or "Elevated" in safety:
            kidney_color = "#f59e0b"
        else:
            kidney_color = "#10b981"
        liver_color = "rgba(59, 130, 246, 0.35)"

    elif disease == "diabetes":
        liver_color = "#ef4444"
        kidney_color = "#f59e0b"
        heart_color = "#38bdf8"

    elif "Elevated" in safety:
        kidney_color = "#f59e0b"

    if not has_kidney_data:
        kidney_color = "#f59e0b"
        if disease != "diabetes":
            kidney_status = "⚠ Unable to fully assess\nKidney-function data not recorded."
            kidney_note = "Actual kidney-function laboratory parameters (Creatinine, BUN, eGFR, etc.) were not recorded."

    return {
        "kidney_color": kidney_color,
        "heart_color": heart_color,
        "liver_color": liver_color,
        "kidney_status": kidney_status,
        "heart_status": heart_status,
        "overall_result": overall,
        "kidney_note": kidney_note,
        "dialysis_prob": dialysis_prob,
        "heart_attack_risk": heart_attack_risk,
        "labs_summary": labs_summary,
        "smart_dose_summary": medicine_line,
        "diagnosis_line": pred or "Pending",
        "highlight_heart": disease == "heart",
        "highlight_kidney": disease == "kidney",
        "highlight_liver": disease == "diabetes",
        "simulation_disclaimer": (
            "Simulated scenario only. These numbers are what-if model outputs, "
            "not a forecast of the patient's real medical outcome."
        ),
    }


def lab_comparison_rows(current_params, simulated_params):
    current_params = current_params or {}
    simulated_params = simulated_params or {}
    keys = []
    for key in list(current_params.keys()) + list(simulated_params.keys()):
        if key not in keys:
            keys.append(key)
    rows = []
    for key in keys:
        before = current_params.get(key)
        after = simulated_params.get(key)
        changed = str(before) != str(after)
        rows.append(
            {
                "key": key,
                "label": readable_feature_name(key),
                "current": format_clinical_value(key, before),
                "simulated": format_clinical_value(key, after),
                "changed": changed,
            }
        )
    return rows
