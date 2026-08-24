"""
Rule-based refinement on top of ML labels using OCR parameters + symptoms (demo / educational).
"""


def refine_heart_diagnosis(ml_label, params, symptoms):
    """
    Adjust obviously mismatched heart-module labels when labs + symptoms fit better.
    Returns (final_label, refinement_note or None).
    """
    params = params or {}
    symptoms_l = (symptoms or "").lower()

    hr = params.get("Heart_Rate")
    trop = params.get("Troponin")
    ntp = params.get("NT-proBNP") or params.get("BNP")
    sbp = params.get("Systolic_BP")
    dbp = params.get("Diastolic_BP")
    ldl = params.get("LDL")
    chol = params.get("Cholesterol")

    palp_words = (
        "palpitation", "palpitations", "arrhythmia", "psvt", "svt",
        "supraventricular",
        "racing heart", "skipped beat", "skipped beats", "extra beats",
    )
    palp = any(w in symptoms_l for w in palp_words)

    rhythm_symsp = palp or any(
        w in symptoms_l for w in ("breathlessness", "dizziness", "lightheaded", "near-syncope", "syncope")
    )

    psvt_label = "Paroxysmal Supraventricular Tachycardia (PSVT)"

    # PSVT / SVT pattern (prioritize before LDL-driven CAD mislabels when HR data exists)
    if hr is not None and hr >= 100:
        trop_low = trop is None or trop <= 15
        hf_unlikely = ntp is None or ntp < 450
        if trop_low and hf_unlikely and (rhythm_symsp or hr >= 110):
            note = (
                "Fast heart rate with troponin and BNP that do not look like a classic "
                "major heart attack pattern point more toward an extra-fast rhythm (SVT/PSVT) "
                "until an ECG proves otherwise."
            )
            return psvt_label, note

    # Data-light fallback when HR missing but symptoms scream rhythm and trop/BNP are reassuring
    if hr is None and rhythm_symsp and (trop is None or trop <= 15) and (ntp is None or ntp < 450):
        note = (
            "Symptoms suggest spells of a racing or irregular heartbeat—an ECG during "
            "symptoms is needed before calling it blocked arteries (CAD)."
        )
        return psvt_label, note

    # Decompensated HF pattern (rough screen using NT-proBNP)
    if ntp is not None and ntp >= 450:
        note = (
            "A high BNP-type marker often means the heart is under strain or there is "
            "extra fluid—the care team should check for heart failure."
        )
        return "Heart Failure", note

    # Hypertensive crisis / sustained hypertension on extracted BP
    if sbp is not None and dbp is not None and sbp >= 140 and dbp >= 90:
        note = ("The blood-pressure numbers on the report are in the high range.")
        return "Hypertension", note

    # ACS-oriented pattern when troponin is clearly elevated on this demo scale
    if trop is not None and trop > 15:
        note = (
            "Higher troponin on the report deserves an urgent heart-attack care-pathway "
            "check by a clinician (this app cannot rule that in or out)."
        )
        return "Coronary Artery Disease", note

    # LDL elevation flags ASCVD risk but must NOT alone flip a rhythm presentation to CAD
    if ldl is not None and ldl >= 160:
        note = (
            "Very high LDL cholesterol raises long-term clogged-artery risk; treat the "
            "urgent rhythm question first, then plan cholesterol care with your doctor."
        )
        return ml_label, note

    return ml_label, None


def refine_diabetes_diagnosis(ml_label, params, symptoms):
    params = params or {}
    symptoms_l = (symptoms or "").lower()
    fbg = params.get("Fasting_Glucose")
    hba1c = params.get("HbA1c")
    c_pep = params.get("C_Peptide")

    pregnant = "pregnan" in symptoms_l or "gestation" in symptoms_l

    if pregnant and hba1c is not None and hba1c >= 5.7:
        return "Gestational Diabetes", "Results fit a pregnancy-related blood-sugar pattern."

    if fbg is not None and fbg >= 200 and (c_pep is None or c_pep < 1.0):
        if "weight loss" in symptoms_l or "polyuria" in symptoms_l or "polydipsia" in symptoms_l:
            return (
                "Type 1 Diabetes",
                "Very high sugar with a low C-peptide suggests the body is not making enough insulin.",
            )

    if hba1c is not None and hba1c >= 6.5 and (c_pep is None or c_pep >= 1.0):
        return (
            "Type 2 Diabetes",
            "High sugar with OK C-peptide fits the common insulin-resistance type of diabetes.",
        )

    if hba1c is not None and 5.7 <= hba1c < 6.5:
        return "Prediabetes", "A1c is a little high but not fully in the diabetes range."

    return ml_label, None


def refine_kidney_diagnosis(ml_label, params, symptoms):
    params = params or {}
    symptoms_l = (symptoms or "").lower()
    cr = params.get("Creatinine")

    if any(k in symptoms_l for k in ("stone", "stones", "renal colic", "calculus", "urolithiasis")):
        return (
            "Kidney Stones",
            "Symptoms fit kidney stones—pain control, fluids, and imaging as your doctor advises.",
        )

    if any(k in symptoms_l for k in ("pyelonephritis", "uti fever", "febrile uti")):
        if cr is not None and cr >= 1.5:
            return (
                "Acute Kidney Injury",
                "Infection plus a jump in creatinine can mean sudden kidney strain—needs urgent review.",
            )

    return ml_label, None


def refine_prediction(disease_type, ml_label, extracted_parameters, symptoms):
    note = None
    if disease_type == "heart":
        ml_label, note = refine_heart_diagnosis(ml_label, extracted_parameters, symptoms)
    elif disease_type == "diabetes":
        ml_label, dnote = refine_diabetes_diagnosis(ml_label, extracted_parameters, symptoms)
        note = dnote
    elif disease_type == "kidney":
        ml_label, knote = refine_kidney_diagnosis(ml_label, extracted_parameters, symptoms)
        note = knote
    return ml_label, note


def next_steps_for_diagnosis(disease_type, prediction):
    """Simple checklist wording for the AI dashboard (education only—not medical orders)."""
    dt = (disease_type or "").lower()
    pred = prediction or ""

    if dt == "heart":
        if "PSVT" in pred or "Supraventricular Tachycardia" in pred:
            return [
                "Get an ECG (heart tracing), especially during a racing-heart episode.",
                "Ask your doctor about simple tricks that can sometimes slow the heart safely.",
                "If spells keep coming back, your doctor may refer you to a rhythm specialist.",
                "Blood tests can check thyroid levels and medicines or caffeine that speed the heart.",
            ]
        if pred == "Arrhythmia":
            return [
                "Get an ECG; a wearable heart monitor helps if symptoms come and go.",
                "Review caffeine, some cold medicines, and thyroid issues with your clinician.",
                "Seek cardiology follow-up for passing out, nonstop fast rate, or known heart disease.",
            ]
        if pred == "Coronary Artery Disease":
            return [
                "If chest pain is active, use the emergency pathway your clinician recommends.",
                "Follow-up tests (like a stress test) are decided by your cardiology team.",
                "Work on blood pressure, cholesterol, blood sugar, and quitting smoking.",
                "Aspirin, statins, and similar drugs only if your doctor prescribes them.",
            ]
        if pred == "Heart Failure":
            return [
                "Track swelling and daily weight; diuretics (“water pills”) need clinician guidance.",
                "An ultrasound of the heart (echo) shows pumping strength and valves.",
                "Kidney labs and salts in the blood are watched when heart medicines change.",
                "Heart-failure nurses or cardiology clinics help adjust medicines safely.",
            ]
        if pred == "Hypertension":
            return [
                "Repeat blood-pressure checks at home or with a 24-hour cuff when advised.",
                "Ask about rare causes if BP is very hard to control or starts very young.",
                "Cut salt, move more, lose extra weight, and limit alcohol.",
                "Blood-pressure pills are chosen by your doctor for your health profile.",
            ]
        if pred == "Valve Disease":
            return [
                "An echo test shows how tight or leaky a valve is.",
                "Tell your team about fevers or dental work if a valve problem is known.",
                "Severe symptoms often need cardiology and sometimes surgery together.",
            ]

    if dt == "diabetes":
        return [
            "Confirm with fasting blood sugar and A1c as your clinician advises.",
            "Food choices, activity, and medicines are adjusted with your care team.",
            "Schedule eye and foot checks; kidney urine tests on the plan your doctor sets.",
            "Learn warning signs of low blood sugar if you use insulin or certain pills.",
        ]

    if dt == "kidney":
        return [
            "Repeat kidney blood tests to see the trend; avoid kidney-harming drugs unless needed.",
            "Urine tests for protein help guide blood-pressure and sugar control.",
            "A kidney specialist helps with fast decline, tough BP, or advanced disease.",
            "If kidneys fail, your team talks through dialysis or transplant in advance.",
        ]

    return [
        "Go over this screen with your doctor and a full exam.",
        "Update your medicine and allergy list before any changes.",
        "Only change treatment after your clinician agrees.",
    ]
