import os
import joblib
import pandas as pd

from ocr.extract_text import extract_report_text

from ocr.extract_parameters import (

    extract_heart_parameters,

    extract_diabetes_parameters,

    extract_kidney_parameters

)

from feature_builder import (

    build_heart_features,

    build_diabetes_features,

    build_kidney_features

)

# ------------------------------------------------
# LOAD TRAINED MODELS
# ------------------------------------------------

heart_model = joblib.load(
    "trained_models/heart_best_model.pkl"
)

diabetes_model = joblib.load(
    "trained_models/diabetes_best_model.pkl"
)

kidney_model = joblib.load(
    "trained_models/kidney_best_model.pkl"
)

dialysis_model = joblib.load(
    "trained_models/dialysis_best_model.pkl"
)

# ------------------------------------------------
# LABEL MAPS
# ------------------------------------------------

heart_labels = {

    0: "Arrhythmia",
    1: "Coronary Artery Disease",
    2: "Heart Failure",
    3: "Hypertension",
    4: "Valve Disease"

}

diabetes_labels = {

    0: "Gestational Diabetes",
    1: "Prediabetes",
    2: "Type 1 Diabetes",
    3: "Type 2 Diabetes"

}

kidney_labels = {

    0: "Acute Kidney Injury",
    1: "Chronic Kidney Disease",
    2: "Diabetic Nephropathy",
    3: "Glomerulonephritis",
    4: "Kidney Stones"

}

dialysis_labels = {

    0: "High",
    1: "Low",
    2: "Medium"

}

# ------------------------------------------------
# UPLOAD FOLDER
# ------------------------------------------------

UPLOAD_FOLDER = "uploads"

# ------------------------------------------------
# PROCESS REPORTS
# ------------------------------------------------

for file_name in os.listdir(UPLOAD_FOLDER):

    if file_name.endswith(".pdf"):

        print("\n===================================")

        print(f"PROCESSING: {file_name}")

        print("===================================\n")

        file_path = os.path.join(
            UPLOAD_FOLDER,
            file_name
        )

        # ----------------------------------------
        # OCR TEXT EXTRACTION
        # ----------------------------------------

        text = extract_report_text(file_path)

        print("OCR TEXT:\n")

        print(text)

        # ========================================
        # HEART DISEASE
        # ========================================

        if "Heart Disease" in text:

            print("\nDETECTED MODULE: HEART DISEASE")

            parameters = extract_heart_parameters(text)

            print("\nEXTRACTED PARAMETERS:")

            print(parameters)

            # BUILD COMPLETE FEATURES

            df = build_heart_features(parameters)

            # PREDICTION

            prediction = heart_model.predict(df)[0]

            result = heart_labels[prediction]

            print("\nAI PREDICTION:")

            print(result)

        # ========================================
        # DIABETES
        # ========================================

        elif "Diabetes" in text:

            print("\nDETECTED MODULE: DIABETES")

            parameters = extract_diabetes_parameters(text)

            print("\nEXTRACTED PARAMETERS:")

            print(parameters)

            # BUILD COMPLETE FEATURES

            df = build_diabetes_features(parameters)

            # PREDICTION

            prediction = diabetes_model.predict(df)[0]

            result = diabetes_labels[prediction]

            print("\nAI PREDICTION:")

            print(result)

        # ========================================
        # KIDNEY DISEASE
        # ========================================

        elif "Kidney Disease" in text:

            print("\nDETECTED MODULE: KIDNEY DISEASE")

            parameters = extract_kidney_parameters(text)

            print("\nEXTRACTED PARAMETERS:")

            print(parameters)

            # BUILD COMPLETE FEATURES

            df = build_kidney_features(parameters)

            # KIDNEY PREDICTION

            prediction = kidney_model.predict(df)[0]

            result = kidney_labels[prediction]

            print("\nAI PREDICTION:")

            print(result)

            # ------------------------------------
            # DIALYSIS PREDICTION
            # ------------------------------------

            dialysis_prediction = dialysis_model.predict(df)[0]

            dialysis_result = dialysis_labels[
                dialysis_prediction
            ]

            print("\nDIALYSIS RISK:")

            print(dialysis_result)

        # ========================================
        # UNKNOWN
        # ========================================

        else:

            print("\nUNKNOWN REPORT TYPE")

        print("\n\n")