"""
Short, plain-English strings for patient-facing screens and PDF summaries.
"""


def simple_ai_explanation(prediction, disease_type, diagnosis_matched_pdf=False):
    """
    High-level explanation for AI dashboard / stored ai_explanation / PDF section 4.
    """
    pred = (prediction or "").strip() or "something that needs a clinician to review"

    dt = (disease_type or "").lower()
    sentences = []

    sentences.append(
        "The tool read lab values from your uploaded report and noted your symptoms. "
        f"It thinks the overall picture fits best with: {pred}."
    )

    if diagnosis_matched_pdf:
        sentences.append(
            "That wording was also taken from the diagnosis section of your uploaded document, "
            "when that part could be read clearly."
        )

    if dt == "heart":
        sentences.append(
            "This is only a guide—not a firm diagnosis. A doctor usually confirms with "
            "an exam and tests such as an ECG or heart tracing."
        )
    elif dt == "kidney":
        sentences.append(
            "Kidney problems need confirmation with repeated blood tests and your doctor's review."
        )
    elif dt == "diabetes":
        sentences.append(
            "Sugar results should always be confirmed with standard blood tests (for example fasting "
            "glucose or A1c) and your clinician."
        )
    else:
        sentences.append(
            "Discuss this summary with your doctor before changing any medicines or plans."
        )

    return " ".join(sentences)


def patient_view_from_shap(prediction, top_features, any_missing=False):
    names = []
    for row in top_features or []:
        feat_name = row.get("readable_feature") or row.get("feature") or ""
        feat_name_lower = feat_name.lower()
        if "glucose" in feat_name_lower:
            if "post" in feat_name_lower:
                feat_name = "post-meal glucose"
            elif "fasting" in feat_name_lower:
                feat_name = "fasting glucose"
            elif "random" in feat_name_lower:
                feat_name = "random glucose"
        else:
            feat_name = feat_name.lower()
        if feat_name and feat_name not in names:
            names.append(feat_name)
            
    names = [n for n in names if n]
    if not names:
        text = "The AI prediction is based on the available laboratory information."
    elif len(names) == 1:
        text = f"Your {names[0]} measurement was among the factors influencing the AI prediction."
    elif len(names) == 2:
        text = f"Your {names[0]} and {names[1]} measurements were among the factors influencing the AI prediction."
    else:
        text = f"Your {', '.join(names[:-1])}, and {names[-1]} measurements were among the factors influencing the AI prediction."
        
    if any_missing:
        text += "\n\nSome measurements were not recorded, so the explanation is based only on the available information."
        
    return text


def doctor_view_summary(prediction, probability, class_index, confidence_band):
    return (
        f"Prediction: {prediction or 'N/A'} (class {class_index}). "
        f"Model probability: {probability}%. Presentation confidence band: {confidence_band}. "
        "Model probability is not the same as medical certainty."
    )
