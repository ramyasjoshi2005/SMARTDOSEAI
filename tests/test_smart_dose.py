from smart_dose_logic import get_smart_dose

def test_get_smart_dose_diabetes_mild_no_contra():
    # Metformin 500mg should be recommended if Creatinine is normal
    res = get_smart_dose("diabetes", "mild", "Type 2 Diabetes", {"Creatinine": "1.0", "eGFR": "90"})
    assert "Metformin" in res["medicine"]
    assert res["status"] == "Approved"

def test_get_smart_dose_diabetes_mild_contra():
    # Metformin should be rejected if Creatinine > 1.5
    res = get_smart_dose("diabetes", "mild", "Type 2 Diabetes", {"Creatinine": "1.8", "eGFR": "90"})
    # Since Metformin is the only mild option, it will fallback to Requires Doctor Review
    assert "Requires Doctor Review" in res["status"]
    assert any("Creatinine" in r for r in res["rejected_explanations"])

def test_get_smart_dose_heart_mild_no_contra():
    res = get_smart_dose("heart", "mild", "Coronary Artery Disease", {"ALT": "30", "Creatinine": "1.0", "eGFR": "90"})
    assert "Atorvastatin" in res["medicine"]
    assert res["status"] == "Approved"

def test_get_smart_dose_heart_mild_contra():
    # Atorvastatin should be rejected if ALT > 150
    res = get_smart_dose("heart", "mild", "Coronary Artery Disease", {"ALT": "160", "Creatinine": "1.0", "eGFR": "90"})
    assert "Requires Doctor Review" in res["status"]
    assert any("ALT" in r for r in res["rejected_explanations"])

def test_get_smart_dose_missing_creatinine_unknown_status():
    # If creatinine is missing (Not recorded / N/A), status should be Requires Doctor Review
    res = get_smart_dose("diabetes", "mild", "Type 2 Diabetes", {"Creatinine": "Not recorded", "eGFR": "90"})
    assert res["status"] == "Requires Doctor Review"
    assert "Unable to fully assess" in res["kidney_safety"]
