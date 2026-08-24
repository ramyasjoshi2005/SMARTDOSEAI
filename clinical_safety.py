"""Medication safety flags for the educational Smart Dose engine."""

from __future__ import annotations


def _num(parameters, *keys):
    params = parameters or {}
    for key in keys:
        if key in params and params[key] not in (None, ""):
            try:
                return float(params[key])
            except (TypeError, ValueError):
                continue
        for existing, value in params.items():
            if str(existing).lower().replace(" ", "_") == key.lower().replace(" ", "_"):
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
    return None


def extra_safety_checks(medicine_name, parameters, current_medications=""):
    """
    Returns a list of {status, reason} dicts.
    status: flag | reject
    """
    findings = []
    name = (medicine_name or "").lower()
    meds = (current_medications or "").lower()
    creatinine = _num(parameters, "Creatinine")
    egfr = _num(parameters, "eGFR")
    potassium = _num(parameters, "Potassium")
    heart_rate = _num(parameters, "Heart_Rate", "Heart Rate")
    sbp = _num(parameters, "Systolic_BP", "Systolic BP")
    alt = _num(parameters, "ALT")

    if "metformin" in name:
        if creatinine is not None and creatinine > 1.5:
            findings.append(
                {
                    "status": "reject",
                    "check": "Kidney Safety Check",
                    "reason": f"Potential risk based on current kidney-function indicators (creatinine {creatinine} mg/dL).",
                }
            )
        elif egfr is not None and egfr < 45:
            findings.append(
                {
                    "status": "flag",
                    "check": "Kidney Safety Check",
                    "reason": f"eGFR {egfr} is reduced; metformin dosing usually needs clinician review.",
                }
            )

    if any(token in name for token in ("metoprolol", "atenolol", "bisoprolol")):
        if heart_rate is not None and heart_rate < 55:
            findings.append(
                {
                    "status": "flag",
                    "check": "Heart Safety Check",
                    "reason": f"Resting heart rate {heart_rate} bpm is low for a beta-blocker start in this demo.",
                }
            )
        if sbp is not None and sbp < 90:
            findings.append(
                {
                    "status": "flag",
                    "check": "Heart Safety Check",
                    "reason": f"Systolic BP {sbp} mmHg is low; hypotensive risk should be reviewed.",
                }
            )

    if any(token in name for token in ("telmisartan", "losartan", " ramipril", "enalapril", "lisinopril")):
        if potassium is not None and potassium >= 5.5:
            findings.append(
                {
                    "status": "flag",
                    "check": "Kidney Safety Check",
                    "reason": f"Potassium {potassium} mEq/L is high for an ACE/ARB-type medicine in this demo.",
                }
            )

    if "atorvastatin" in name and alt is not None and alt > 150:
        findings.append(
            {
                "status": "flag",
                "check": "Lab Safety Check",
                "reason": f"ALT {alt} is above the demo threshold used for statin caution.",
            }
        )

    if "aspirin" in name and any(x in meds for x in ("warfarin", "apixaban", "rivaroxaban", "dabigatran")):
        findings.append(
            {
                "status": "flag",
                "check": "Drug Interaction Check",
                "reason": "Aspirin plus an oral anticoagulant increases bleeding risk and needs clinician review.",
            }
        )

    if "insulin" in name:
        glucose = _num(parameters, "Fasting_Glucose", "Fasting Glucose")
        if glucose is not None and glucose < 70:
            findings.append(
                {
                    "status": "reject",
                    "check": "Lab Safety Check",
                    "reason": f"Fasting glucose {glucose} mg/dL is already low; insulin would need urgent clinical review.",
                }
            )

    return findings
