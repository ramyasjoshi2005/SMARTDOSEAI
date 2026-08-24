from clinical_validation import validate_clinical_data

def test_validation_diabetes_missing_critical():
    is_valid, errors = validate_clinical_data("diabetes", {"BMI": "25"})
    assert not is_valid
    assert any("Missing critical" in e or "Missing required" in e for e in errors)

def test_validation_diabetes_out_of_range():
    is_valid, errors = validate_clinical_data("diabetes", {"HbA1c": "25", "Fasting Glucose": "100"})
    assert not is_valid
    assert any("outside biologically probable range" in e for e in errors)

def test_validation_diabetes_valid():
    is_valid, errors = validate_clinical_data("diabetes", {"HbA1c": "7.5", "Fasting Glucose": "140", "BMI": "30"})
    assert is_valid
    assert len(errors) == 0

def test_validation_kidney_missing_critical():
    is_valid, errors = validate_clinical_data("kidney", {"Potassium": "4.0"})
    assert not is_valid
    assert any("Missing critical" in e or "Missing required" in e for e in errors)

def test_validation_kidney_valid():
    is_valid, errors = validate_clinical_data("kidney", {"Creatinine": "1.2", "BUN": "20"})
    assert is_valid
    assert len(errors) == 0


def test_validation_aliases():
    from clinical_validation import KEY_ALIASES
    # Check that our aliases are present and normalized correctly
    assert KEY_ALIASES.get("fasting plasma glucose") == "Fasting_Glucose"
    assert KEY_ALIASES.get("2 hr postprandial glucose") == "Post_Meal_Glucose"
    assert KEY_ALIASES.get("hba1c ngsp") == "HbA1c"
    
    # Check that validation runs correctly with these keys
    is_valid, errors = validate_clinical_data("diabetes", {
        "HbA1c NGSP": "7.5",
        "Fasting Plasma Glucose": "140",
        "2 hr Postprandial Glucose": "180"
    })
    assert is_valid
    assert len(errors) == 0
