import json

RECORD_FILE = "medical_records/patient_records.json"

# ------------------------------------------------
# LOAD RECORDS
# ------------------------------------------------

def load_records():

    with open(RECORD_FILE, "r") as file:

        return json.load(file)

# ------------------------------------------------
# SAVE RECORDS
# ------------------------------------------------

def save_records(records):

    with open(RECORD_FILE, "w") as file:

        json.dump(
            records,
            file,
            indent=4
        )

# ------------------------------------------------
# ADD PATIENT
# ------------------------------------------------

def add_patient_record(

    patient_id,

    patient_name,

    age,

    gender,

    assigned_doctor

):

    records = load_records()

    new_patient = {

        "patient_id": patient_id,

        "patient_name": patient_name,

        "age": age,

        "gender": gender,

        "assigned_doctor": assigned_doctor,

        "reports": []

    }

    records.append(new_patient)

    save_records(records)

# ------------------------------------------------
# UPLOAD REPORT
# ------------------------------------------------

def upload_lab_report(

    patient_id,

    report_file

):

    records = load_records()

    for patient in records:

        if patient["patient_id"] == patient_id:

            patient["reports"].append(
                report_file
            )

    save_records(records)

# ------------------------------------------------
# GET PATIENT RECORD
# ------------------------------------------------

def get_patient_record(patient_id):

    records = load_records()

    for patient in records:

        if patient["patient_id"] == patient_id:

            return patient

    return None