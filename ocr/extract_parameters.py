import re

# ------------------------------------------------
# HEART PARAMETERS
# ------------------------------------------------

def extract_heart_parameters(text):

    parameters = {}

    patterns = {

        "Systolic_BP":
        r"Systolic BP\s+(\d+)",

        "Diastolic_BP":
        r"Diastolic BP\s+(\d+)",

        "Cholesterol":
        r"Total Cholesterol\s+(\d+)",

        "LDL":
        r"LDL\s+(\d+)",

        "HDL":
        r"HDL\s+(\d+)",

        "Heart_Rate":
        r"Heart Rate\s+(\d+)"

    }

    for key, pattern in patterns.items():

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:

            parameters[key] = float(
                match.group(1)
            )

    # Typical PDF / lab line format: "Label: value"
    report_style = {
        "Troponin":
        r"(?:High[- ]sensitivity\s+)?(?:hs[- ])?Troponin(?:[- ]I)?\s*:\s*(?:<\s*)?([\d.]+)",
        "Heart_Rate":
        r"(?:Resting\s+)?Heart\s+Rate\s*:\s*(\d+)",
        "NT-proBNP":
        r"NT[- ]?proBNP\s*:\s*([\d.]+)",
        "BNP":
        r"(?<![\w/])BNP\s*:\s*([\d.]+)",
        "Systolic_BP":
        r"Systolic\s+(?:BP|Blood\s+Pressure)\s*:\s*(\d+)",
        "Diastolic_BP":
        r"Diastolic\s+(?:BP|Blood\s+Pressure)\s*:\s*(\d+)",
        "Cholesterol":
        r"Total\s+Cholesterol\s*:\s*(\d+)",
        "LDL":
        r"LDL(?:-C)?(?:\s*\([^)]*\))?\s*:\s*(\d+)",
        "HDL":
        r"HDL(?:-C)?(?:\s*\([^)]*\))?\s*:\s*(\d+)",
        "Potassium":
        r"(?:Serum\s+)?Potassium\s*:\s*([\d.]+)",
        "Magnesium":
        r"(?:Serum\s+)?Magnesium\s*:\s*([\d.]+)",
        "TSH":
        r"\bTSH\s*:\s*([\d.]+)",
    }

    for key, pattern in report_style.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            parameters.setdefault(key, float(match.group(1)))

    amb = re.search(
        r"Ambulatory\s+BP[^\d]{0,52}(\d+)\s*/\s*(\d+)",
        text,
        re.IGNORECASE,
    )
    if amb:
        parameters.setdefault("Systolic_BP", float(amb.group(1)))
        parameters.setdefault("Diastolic_BP", float(amb.group(2)))

    dual_bp = re.search(
        r"(?:Average\s+)?(?:Blood\s+Pressure|BP)\s*:\s*(\d+)\s*/\s*(\d+)",
        text,
        re.IGNORECASE,
    )
    if dual_bp:
        parameters.setdefault("Systolic_BP", float(dual_bp.group(1)))
        parameters.setdefault("Diastolic_BP", float(dual_bp.group(2)))

    return parameters

# ------------------------------------------------
# DIABETES PARAMETERS
# ------------------------------------------------

def extract_diabetes_parameters(text):

    parameters = {}

    patterns = {

        "Fasting_Glucose":
        r"Fasting Blood Sugar\s+(\d+)",

        "Postprandial_Glucose":
        r"Postprandial Blood Sugar\s+(\d+)",

        "HbA1c":
        r"HbA1c\s+([\d.]+)",

        "Insulin":
        r"Insulin\s+([\d.]+)",

        "C_Peptide":
        r"C-Peptide\s+([\d.]+)"

    }

    for key, pattern in patterns.items():

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:

            parameters[key] = float(
                match.group(1)
            )

    report_style = {
        "Fasting_Glucose":
        r"Fasting\s+(?:Plasma\s+)?Glucose\s*:\s*(\d+)",
        "Postprandial_Glucose":
        r"(?:2[- ]?\s*h(?:our|r)?\s*)?(?:Postprandial|PP)\s+(?:Plasma\s+)?Glucose\s*:\s*(\d+)",
        "HbA1c":
        r"HbA1c(?:\s*\([^)]*\))?\s*:\s*([\d.]+)",
        "Insulin":
        r"(?<![\w])Insulin\s*:\s*([\d.]+)",
        "C_Peptide":
        r"C[- ]?Peptide\s*:\s*([\d.]+)",
    }

    for key, pattern in report_style.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            parameters.setdefault(key, float(match.group(1)))

    rand_g = re.search(
        r"Random\s+(?:Plasma\s+)?Glucose\s*:\s*(\d+)",
        text,
        re.IGNORECASE,
    )
    if rand_g:
        parameters.setdefault("Postprandial_Glucose", float(rand_g.group(1)))

    return parameters

# ------------------------------------------------
# KIDNEY PARAMETERS
# ------------------------------------------------

def extract_kidney_parameters(text):

    parameters = {}

    patterns = {

        "Creatinine":
        r"Creatinine\s+([\d.]+)",

        "Blood_Urea":
        r"Blood Urea\s+([\d.]+)",

        "eGFR":
        r"eGFR\s+([\d.]+)",

        "Uric_Acid":
        r"Uric Acid\s+([\d.]+)",

        "Potassium":
        r"Potassium\s+([\d.]+)",

        "Calcium":
        r"Calcium\s+([\d.]+)"

    }

    for key, pattern in patterns.items():

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:

            parameters[key] = float(
                match.group(1)
            )

    report_style = {
        "Creatinine":
        r"(?:Serum\s+)?Creatinine(?:\s*\([^)]*\))?\s*:\s*([\d.]+)",
        "Blood_Urea":
        r"(?:Blood\s+Urea\s+Nitrogen(?:\s*\(BUN\))?|BUN)\s*:\s*([\d.]+)",
        "eGFR":
        r"eGFR(?:\s*\([^)]*\))?\s*:\s*([\d.]+)",
        "Uric_Acid":
        r"(?:Serum\s+)?Uric\s+Acid\s*:\s*([\d.]+)",
        "Potassium":
        r"(?:Serum\s+)?Potassium\s*:\s*([\d.]+)",
        "Calcium":
        r"(?:Serum\s+)?Calcium\s*:\s*([\d.]+)",
    }

    for key, pattern in report_style.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            parameters.setdefault(key, float(match.group(1)))

    return parameters
