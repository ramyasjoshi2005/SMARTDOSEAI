from fpdf import FPDF
import os
from flask import session

from paths import REPORTS_FOLDER

from ocr.report_sections import canonicalize_lab_parameter_keys
from clinical_units import readable_feature_name, format_clinical_value


def _latest_ai_encounter(patient):
    appts = patient.get("appointments") or []
    if not appts:
        return {}
    for i in range(len(appts) - 1, -1, -1):
        if appts[i].get("ai_prediction"):
            return appts[i]
    return appts[-1]


def _pdf_safe(text) -> str:
    """
    Core PDF fonts (Helvetica/Arial) accept Latin-1 only. Strip / replace common
    Unicode punctuation that causes FPDFUnicodeEncodingException.
    """
    if text is None:
        return ""
    s = str(text)
    for a, b in (
        ("\u2014", "-"),  # em dash
        ("\u2013", "-"),  # en dash
        ("\u2212", "-"),  # minus
        ("\u2018", "'"),
        ("\u2019", "'"),
        ("\u201c", '"'),
        ("\u201d", '"'),
        ("\u2026", "..."),
        ("\u00a0", " "),
        ("\u2713", "✓"),
        ("\u26a0", "⚠"),
    ):
        s = s.replace(a, b)
    return s.encode("latin-1", "replace").decode("latin-1")


def generate_patient_pdf(patient):
    appt = _latest_ai_encounter(patient)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)

    pdf.cell(0, 10, txt=_pdf_safe("SmartDoseAI Clinical Report"), ln=True, align="C")
    pdf.ln(10)

    # 1. Patient Details
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, txt=_pdf_safe("1. Patient Details"), ln=True)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, txt=_pdf_safe(f"Patient Name: {patient.get('name', 'N/A')}"), ln=True)
    pdf.cell(
        0,
        8,
        txt=_pdf_safe(
            f"Age: {patient.get('age', 'N/A')} | Gender: {patient.get('gender', 'N/A')} "
            f"| Blood Group: {patient.get('blood_group', 'N/A')}"
        ),
        ln=True,
    )
    pdf.cell(
        0,
        8,
        txt=_pdf_safe(f"Assigned Doctor: {patient.get('assigned_doctor', 'N/A')}"),
        ln=True,
    )
    pdf.cell(
        0,
        8,
        txt=_pdf_safe(f"Appointment Date: {appt.get('appointment_date', 'N/A')}"),
        ln=True,
    )
    pdf.ln(5)

    # 2. Original Lab Report Information
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, txt=_pdf_safe("2. Original Lab Report Information"), ln=True)
    pdf.set_font("Helvetica", "", 12)
    lab_reports = appt.get("lab_reports") or []
    if isinstance(lab_reports, str):
        lab_reports = [lab_reports]
    
    reports_text = ""
    if lab_reports:
        for r in lab_reports:
            if isinstance(r, dict):
                fname = r.get("filename") or r.get("pdf", "Unknown.pdf")
                by = r.get("uploaded_by") or "Reception"
                reports_text += f"{fname} (Uploaded by {by}); "
            else:
                reports_text += f"{r} (Uploaded by Reception); "
        reports_text = reports_text.rstrip("; ")
    else:
        reports_text = "No original lab reports attached."
    
    pdf.multi_cell(0, 8, txt=_pdf_safe(reports_text))
    pdf.ln(5)

    # 3. Confirmed Laboratory Values
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, txt=_pdf_safe("3. Confirmed Laboratory Values"), ln=True)
    pdf.set_font("Helvetica", "", 12)
    
    key_params_list = [
        "Fasting_Glucose", "HbA1c", "Insulin", "BMI",
        "Creatinine", "BUN", "eGFR", "Potassium", "Hemoglobin",
        "Blood_Pressure", "Cholesterol", "Heart_Rate"
    ]
    params_norm = canonicalize_lab_parameter_keys(appt.get("parameters") or {})
    for p_key in key_params_list:
        found_key = None
        for k in params_norm:
            if k.lower() == p_key.lower():
                found_key = k
                break
        
        r_name = readable_feature_name(p_key)
        if found_key and params_norm[found_key] not in (None, "", "Not recorded", "N/A"):
            val_formatted = format_clinical_value(p_key, params_norm[found_key])
            pdf.cell(0, 7, txt=_pdf_safe(f"- {r_name}: {val_formatted}"), ln=True)
        else:
            pdf.cell(0, 7, txt=_pdf_safe(f"- {r_name}: Not recorded"), ln=True)
    pdf.ln(5)

    # 4. AI Prediction
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, txt=_pdf_safe("4. AI Prediction"), ln=True)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, txt=_pdf_safe(f"Prediction: {appt.get('ai_prediction', 'N/A')}"), ln=True)
    pdf.cell(0, 8, txt=_pdf_safe(f"Model Probability: {appt.get('confidence_score', 'N/A')}%"), ln=True)
    pdf.cell(0, 8, txt=_pdf_safe(f"Confidence Band: {appt.get('confidence_band', 'N/A')}"), ln=True)
    pdf.ln(5)

    # 5. Explainability (SHAP Factors)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, txt=_pdf_safe("5. Explainability (Model Insights)"), ln=True)
    pdf.set_font("Helvetica", "", 12)
    
    shap_data = appt.get("shap_explanation", {})
    contributing = shap_data.get("contributing_features", [])
    if not contributing:
        top_features = shap_data.get("top_features", [])
        contributing = [f for f in top_features if f.get("clinical_value") is not None]
        
    if contributing:
        pdf.set_font("Helvetica", "I", 11)
        pdf.cell(0, 6, txt=_pdf_safe("Patient Explanation:"), ln=True)
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, txt=_pdf_safe(appt.get("patient_view", "N/A")))
        pdf.ln(3)
        
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 6, txt=_pdf_safe("Main Contributing Factors:"), ln=True)
        pdf.set_font("Helvetica", "", 11)
        for f in contributing:
            feat_name = f.get("readable_feature") or f.get("feature") or ""
            val = f.get("display_value") or f.get("clinical_value") or ""
            strength = f.get("strength", "Mild")
            pdf.cell(0, 6, txt=_pdf_safe(f"  - {feat_name} ({val}) -- Influence: {strength}"), ln=True)
    else:
        pdf.cell(0, 6, txt=_pdf_safe("No contributing SHAP factors were computed (or all were missing)."), ln=True)
    pdf.ln(5)

    # 6. Smart Dose Recommendation
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, txt=_pdf_safe("6. Smart Dose Recommendation"), ln=True)
    pdf.set_font("Helvetica", "", 12)
    
    dose = appt.get("smart_dose", {})
    pdf.cell(0, 8, txt=_pdf_safe(f"Recommended Regimen: {dose.get('medicine', 'N/A')}"), ln=True)
    pdf.cell(0, 8, txt=_pdf_safe(f"Dose: {dose.get('dose', 'N/A')} (Max Ceiling: {dose.get('max_dose', 'N/A')})"), ln=True)
    pdf.cell(0, 8, txt=_pdf_safe(f"Organ Safety Assessment: {dose.get('organ_safety', 'N/A')}"), ln=True)
    pdf.cell(0, 8, txt=_pdf_safe(f"Kidney Check: {dose.get('kidney_safety', 'N/A')}"), ln=True)
    pdf.cell(0, 8, txt=_pdf_safe(f"Heart Check: {dose.get('heart_safety', 'N/A')}"), ln=True)
    pdf.cell(0, 8, txt=_pdf_safe(f"Interactions: {dose.get('interactions', 'N/A')}"), ln=True)
    pdf.cell(0, 8, txt=_pdf_safe(f"Clinician Review Status: {dose.get('status', 'N/A')}"), ln=True)
    
    rejected = dose.get("rejected_explanations") or []
    if rejected:
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 6, txt=_pdf_safe("Rejected Alternative Regimens:"), ln=True)
        pdf.set_font("Helvetica", "", 10)
        for r in rejected:
            pdf.multi_cell(0, 5, txt=_pdf_safe(f"  - {r}"))
    pdf.ln(5)

    # 7. What-If Digital Twin Simulation
    sim_data = session.get(f"sim_{patient.get('name')}")
    if sim_data:
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, txt=_pdf_safe("7. What-If Digital Twin Simulation"), ln=True)
        pdf.set_font("Helvetica", "", 12)
        
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 6, txt=_pdf_safe("WARNING: What-if simulation - not a forecast of the patient's actual medical outcome."), ln=True)
        pdf.set_font("Helvetica", "", 12)
        
        # Display comparison of changed parameters
        sim_params = sim_data.get("simulated_params") or {}
        orig_params = appt.get("parameters") or {}
        
        pdf.cell(0, 8, txt=_pdf_safe("Parameter Shifts:"), ln=True)
        for k, v in sim_params.items():
            orig_v = orig_params.get(k, "Not recorded")
            if str(orig_v) != str(v):
                r_name = readable_feature_name(k)
                orig_disp = format_clinical_value(k, orig_v) if orig_v != "Not recorded" else "Not recorded"
                sim_disp = format_clinical_value(k, v)
                pdf.cell(0, 6, txt=_pdf_safe(f"  - {r_name}: {orig_disp} -> {sim_disp} (Simulated)"), ln=True)
                
        # Comparison of outputs
        twin = appt.get("digital_twin") or {}
        sim_twin = sim_data.get("simulated_twin") or {}
        
        pdf.ln(2)
        pdf.cell(0, 8, txt=_pdf_safe(f"Actual Prediction: {twin.get('diagnosis_line', 'N/A')} | Simulated: {sim_twin.get('diagnosis_line', 'N/A')}"), ln=True)
        pdf.cell(0, 8, txt=_pdf_safe(f"Actual Dose: {twin.get('smart_dose_summary', 'N/A')} | Simulated: {sim_twin.get('smart_dose_summary', 'N/A')}"), ln=True)
        pdf.cell(0, 8, txt=_pdf_safe(f"Actual Kidney: {twin.get('kidney_status', 'N/A')} | Simulated: {sim_twin.get('kidney_status', 'N/A')}"), ln=True)
        pdf.cell(0, 8, txt=_pdf_safe(f"Actual Heart: {twin.get('heart_status', 'N/A')} | Simulated: {sim_twin.get('heart_status', 'N/A')}"), ln=True)
        pdf.ln(5)

    # 8. Clinical Guidance & Disclaimer
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, txt=_pdf_safe("8. Clinical Guidance & Disclaimer"), ln=True)
    pdf.set_font("Helvetica", "I", 11)
    pdf.multi_cell(
        0,
        6,
        txt=_pdf_safe(
            "This report is generated by SmartDoseAI as an educational decision-support demo. "
            "It does not constitute autonomous medical advice or prescribing authority. "
            "All recommendations must be evaluated and approved by a licensed clinician prior to care implementation."
        ),
    )

    raw_name = patient.get("name", "unknown") or "unknown"
    filename = f"patient_report_{str(raw_name).lower().replace(' ', '_')}.pdf"
    os.makedirs(REPORTS_FOLDER, exist_ok=True)
    filepath = os.path.join(REPORTS_FOLDER, filename)
    pdf.output(filepath)
    return filename
