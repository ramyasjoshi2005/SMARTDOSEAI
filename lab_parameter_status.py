"""
Educational/demo reference bands for coloring lab OCR chips (not clinical advice).
"""


def lab_value_status_tuple(param_key: str, value):
    """
    Map a parameter key/value to (css_suffix, label).
    css_suffix: normal | warn | danger | risk — maps to badge styles in templates.
    """
    if value is None or value == "—":
        return ("normal", "Normal")

    key = (param_key or "").replace("-", "_").strip()

    try:
        v = float(value)
    except (TypeError, ValueError):
        return ("normal", "Normal")

    # --- Heart rhythm / BP ---
    if key in ("Heart_Rate", "Resting_Heart_Rate", "HR"):
        if 60 <= v <= 100:
            return ("normal", "Normal")
        if (100 < v <= 110) or (55 <= v < 60):
            return ("warn", "Near border")
        if (110 < v <= 120) or (45 <= v < 55):
            return ("danger", "Danger zone")
        return ("risk", "Risky")

    if key == "Systolic_BP":
        if v < 120:
            return ("normal", "Normal")
        if v < 130:
            return ("warn", "Near border")
        if v < 160:
            return ("danger", "Danger zone")
        return ("risk", "Risky")

    if key == "Diastolic_BP":
        if v < 80:
            return ("normal", "Normal")
        if v < 85:
            return ("warn", "Near border")
        if v < 100:
            return ("danger", "Danger zone")
        return ("risk", "Risky")

    # --- Lipids (mg/dL demo cutoffs) ---
    if key == "LDL":
        if v < 100:
            return ("normal", "Normal")
        if v < 130:
            return ("warn", "Near border")
        if v < 160:
            return ("danger", "Danger zone")
        return ("risk", "Risky")

    if key == "HDL":
        if v >= 50:
            return ("normal", "Normal")
        if v >= 40:
            return ("warn", "Near border")
        if v >= 30:
            return ("danger", "Danger zone")
        return ("risk", "Risky")

    if key == "Cholesterol":
        if v < 200:
            return ("normal", "Normal")
        if v < 240:
            return ("warn", "Near border")
        if v < 280:
            return ("danger", "Danger zone")
        return ("risk", "Risky")

    # --- Troponin (demo heuristic: tiny values resemble mg/mL-scale; teens–tens resemble ng/L) ---
    if key in ("Troponin", "High_sensitivity_Troponin_I", "Troponin_I"):
        if v <= 0.1:
            if v <= 0.04:
                return ("normal", "Normal")
            return ("warn", "Near border")
        if v < 40:
            return ("normal", "Normal")
        if v < 100:
            return ("warn", "Near border")
        if v < 300:
            return ("danger", "Danger zone")
        return ("risk", "Risky")

    # --- NP peptides (pg/mL style demo) ---
    if key in ("NT_proBNP", "NTproBNP", "BNP"):
        nk = key
        threshold_high = 900 if nk != "BNP" else 400
        threshold_mid = 300 if nk != "BNP" else 100
        threshold_low = 150 if nk != "BNP" else 50
        if v < threshold_low:
            return ("normal", "Normal")
        if v < threshold_mid:
            return ("warn", "Near border")
        if v < threshold_high:
            return ("danger", "Danger zone")
        return ("risk", "Risky")

    # --- Electrolytes (mmol/L style) ---
    if key in ("Potassium", "Serum_Potassium"):
        if 3.5 <= v <= 5.0:
            return ("normal", "Normal")
        if (3.2 <= v < 3.5) or (5.0 < v <= 5.5):
            return ("warn", "Near border")
        if (3.0 <= v < 3.2) or (5.5 < v <= 6.0):
            return ("danger", "Danger zone")
        return ("risk", "Risky")

    if key in ("Magnesium", "Serum_Magnesium"):
        if 1.8 <= v <= 2.4:
            return ("normal", "Normal")
        if (1.6 <= v < 1.8) or (2.4 < v <= 2.7):
            return ("warn", "Near border")
        if (1.4 <= v < 1.6) or (2.7 < v <= 2.9):
            return ("danger", "Danger zone")
        return ("risk", "Risky")

    if key == "Calcium":
        if 8.5 <= v <= 10.3:
            return ("normal", "Normal")
        if (8.0 <= v < 8.5) or (10.3 < v <= 11.0):
            return ("warn", "Near border")
        return ("risk", "Risky")

    # --- Renal ---
    if key == "Creatinine":
        if v < 1.2:
            return ("normal", "Normal")
        if v < 1.6:
            return ("warn", "Near border")
        if v < 2.5:
            return ("danger", "Danger zone")
        return ("risk", "Risky")

    if key in ("Blood_Urea", "BUN"):
        if v < 20:
            return ("normal", "Normal")
        if v < 40:
            return ("warn", "Near border")
        if v < 60:
            return ("danger", "Danger zone")
        return ("risk", "Risky")

    if key == "eGFR":
        if v >= 60:
            return ("normal", "Normal")
        if v >= 45:
            return ("warn", "Near border")
        if v >= 30:
            return ("danger", "Danger zone")
        return ("risk", "Risky")

    if key == "Uric_Acid":
        if v < 6:
            return ("normal", "Normal")
        if v < 7.5:
            return ("warn", "Near border")
        if v < 9:
            return ("danger", "Danger zone")
        return ("risk", "Risky")

    # --- Diabetes ---
    if key == "Fasting_Glucose":
        if v < 100:
            return ("normal", "Normal")
        if v < 126:
            return ("warn", "Near border")
        if v < 200:
            return ("danger", "Danger zone")
        return ("risk", "Risky")

    if key == "Postprandial_Glucose":
        if v < 140:
            return ("normal", "Normal")
        if v < 180:
            return ("warn", "Near border")
        if v < 250:
            return ("danger", "Danger zone")
        return ("risk", "Risky")

    if key == "HbA1c":
        if v < 5.7:
            return ("normal", "Normal")
        if v < 6.5:
            return ("warn", "Near border")
        if v < 8:
            return ("danger", "Danger zone")
        return ("risk", "Risky")

    # --- Thyroid ---
    if key == "TSH":
        if 0.4 <= v <= 4.0:
            return ("normal", "Normal")
        if (0.2 <= v < 0.4) or (4.0 < v <= 6.0):
            return ("warn", "Near border")
        return ("risk", "Risky")

    return ("normal", "Normal")
