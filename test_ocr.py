import os

from ocr.extract_text import extract_report_text

from ocr.extract_parameters import (

    extract_heart_parameters,

    extract_diabetes_parameters,

    extract_kidney_parameters

)

UPLOAD_FOLDER = "uploads"

# ------------------------------------------------

for file_name in os.listdir(UPLOAD_FOLDER):

    if file_name.endswith(".pdf"):

        file_path = os.path.join(
            UPLOAD_FOLDER,
            file_name
        )

        print("\n===================================")

        print(f"PROCESSING: {file_name}")

        print("===================================\n")

        # OCR

        text = extract_report_text(file_path)

        print("EXTRACTED TEXT:\n")

        print(text)

        # ------------------------------------------------
        # DISEASE DETECTION
        # ------------------------------------------------

        if "Heart Disease" in text:

            parameters = extract_heart_parameters(text)

            disease_type = "Heart Disease"

        elif "Diabetes" in text:

            parameters = extract_diabetes_parameters(text)

            disease_type = "Diabetes"

        elif "Kidney Disease" in text:

            parameters = extract_kidney_parameters(text)

            disease_type = "Kidney Disease"

        else:

            parameters = {}

            disease_type = "Unknown"

        # ------------------------------------------------

        print(f"\nDETECTED MODULE: {disease_type}")

        print("\nEXTRACTED PARAMETERS:\n")

        print(parameters)

        print("\n\n")