from flask import Flask, render_template, request, redirect, send_from_directory, send_file, url_for, session
import json
import os
import shutil
from datetime import datetime
from urllib.parse import quote

from werkzeug.utils import secure_filename

from ocr.report_sections import canonicalize_lab_parameter_keys
from smart_dose_logic import get_smart_dose
from digital_twin_logic import lab_comparison_rows, simulate_twin, format_medicine_recommendation
from generate_report import generate_patient_pdf
from clinical_refinement import refine_prediction, next_steps_for_diagnosis
from lab_parameter_status import lab_value_status_tuple
from patient_plain_language import simple_ai_explanation
from clinical_validation import parameters_from_form, validate_clinical_data
from ocr_pipeline import extract_labs_from_file, suggested_review_fields
from prediction_service import run_explained_prediction
from ml_results import load_model_performance, load_shap_importance
from paths import BASE_DIR, DB_FILE, UPLOAD_FOLDER, REPORTS_FOLDER, DATABASE_DIR

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "super_secret_key_for_smartdose_ai")


def _env_or_default(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value


RECEPTION_USERNAME = _env_or_default("RECEPTION_USERNAME", "reception")
RECEPTION_PASSWORD = _env_or_default("RECEPTION_PASSWORD", "123")
DOCTOR_LOGINS = {
    _env_or_default("DOCTOR_SHARMA_USERNAME", "drsharma"): (
        _env_or_default("DOCTOR_SHARMA_PASSWORD", "123"),
        "Dr. Sharma",
    ),
    _env_or_default("DOCTOR_PRIYA_USERNAME", "drpriya"): (
        _env_or_default("DOCTOR_PRIYA_PASSWORD", "123"),
        "Dr. Priya",
    ),
    _env_or_default("DOCTOR_MEHTA_USERNAME", "drmehta"): (
        _env_or_default("DOCTOR_MEHTA_PASSWORD", "123"),
        "Dr. Mehta",
    ),
    _env_or_default("DOCTOR_REDDY_USERNAME", "drreddy"): (
        _env_or_default("DOCTOR_REDDY_PASSWORD", "123"),
        "Dr. Reddy",
    ),
}

# Assigned doctor → AI disease module (server-side; diagnosis UI does not choose).
DOCTOR_DISEASE_MODULE = {
    "Dr. Sharma": ("kidney", "Kidney disease"),
    "Dr. Priya": ("diabetes", "Diabetes mellitus"),
    "Dr. Mehta": ("heart", "Heart disease"),
    "Dr. Reddy": ("heart", "Heart disease"),
}


def lab_report_filename(entry):
    if isinstance(entry, dict):
        fn = entry.get("filename")
        return str(fn).strip() if fn else ""
    if entry is None:
        return ""
    return str(entry).strip()


def patient_all_lab_filenames(patient):
    names = set()
    for appt in patient.get("appointments", []):
        for r in appt.get("lab_reports", []):
            fn = lab_report_filename(r)
            if fn:
                names.add(fn)
    return names


def collect_uploaded_reports_for_diagnosis(patient):
    rows = []
    for appt in patient.get("appointments", []):
        adate = appt.get("appointment_date", "")
        atime = appt.get("appointment_time", "")
        for r in appt.get("lab_reports", []):
            fn = lab_report_filename(r)
            if not fn:
                continue
            uploaded_at = ""
            if isinstance(r, dict):
                uploaded_at = r.get("uploaded_at") or ""
            rows.append({
                "filename": fn,
                "uploaded_at": uploaded_at or "—",
                "appointment_date": adate,
                "appointment_time": atime,
            })
    return rows


def appointment_index_for_lab_file(patient, report_filename):
    """Index of the appointment that lists this uploaded lab file (not always the last row)."""
    if not report_filename:
        return None
    target = secure_filename(report_filename)
    for i, appt in enumerate(patient.get("appointments", [])):
        for r in appt.get("lab_reports", []):
            if lab_report_filename(r) == target:
                return i
    return None


def latest_appointment_with_ai(patient):
    """Most recent appointment row that already has an AI prediction (Smart Dose / Twin targets)."""
    appts = patient.get("appointments") or []
    if not appts:
        return -1, {}
    for i in range(len(appts) - 1, -1, -1):
        if appts[i].get("ai_prediction"):
            return i, appts[i]
    return len(appts) - 1, appts[-1]


@app.template_filter("lab_filename")
def lab_filename_filter(entry):
    return lab_report_filename(entry)


@app.template_filter("lab_uploaded_at")
def lab_uploaded_at_filter(entry):
    if isinstance(entry, dict):
        return entry.get("uploaded_at") or ""
    return ""


@app.template_filter("lab_param_badge")
def lab_param_badge_filter(param_key, value):
    tier, label = lab_value_status_tuple(param_key, value)
    return {"tier": tier, "label": label}


@app.template_filter("canonical_lab_params")
def canonical_lab_params_filter(raw):
    return canonicalize_lab_parameter_keys(raw or {})


def _lab_analysis_record_index_rows(patients):
    rows = []
    for p in patients:
        for i, appt in enumerate(p.get("appointments") or []):
            if (
                appt.get("parameters")
                or appt.get("lab_reports")
                or appt.get("lab_analysis_record")
            ):
                report_hint = next(
                    (
                        lab_report_filename(r)
                        for r in appt.get("lab_reports") or []
                        if lab_report_filename(r)
                    ),
                    "",
                ) or "OCR / parameters on file"
                rows.append(
                    {
                        "patient_name": p["name"],
                        "appt_index": i,
                        "appointment_date": appt.get("appointment_date", "—"),
                        "report_hint": report_hint,
                    }
                )
    return rows


# =========================================================
# DATABASE
# =========================================================

os.makedirs(DATABASE_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)

_seed_db = os.path.join(BASE_DIR, "database", "patients.json")
if not os.path.exists(DB_FILE):
    if os.path.isfile(_seed_db) and os.path.abspath(_seed_db) != os.path.abspath(DB_FILE):
        shutil.copy(_seed_db, DB_FILE)
    else:
        with open(DB_FILE, "w") as file:
            json.dump([], file)

_seed_uploads = os.path.join(BASE_DIR, "uploads")
if os.path.isdir(_seed_uploads) and os.path.abspath(_seed_uploads) != os.path.abspath(UPLOAD_FOLDER):
    for name in os.listdir(_seed_uploads):
        src = os.path.join(_seed_uploads, name)
        dest = os.path.join(UPLOAD_FOLDER, name)
        if os.path.isfile(src) and not os.path.exists(dest):
            shutil.copy(src, dest)

_seed_reports = os.path.join(BASE_DIR, "reports")
if os.path.isdir(_seed_reports) and os.path.abspath(_seed_reports) != os.path.abspath(REPORTS_FOLDER):
    for name in os.listdir(_seed_reports):
        src = os.path.join(_seed_reports, name)
        dest = os.path.join(REPORTS_FOLDER, name)
        if os.path.isfile(src) and not os.path.exists(dest):
            shutil.copy(src, dest)

# Models are loaded lazily in inference.py (preprocessor + classifier).

# =========================================================
# LOAD DATABASE
# =========================================================

def load_patients():

    with open(DB_FILE, "r") as file:

        return json.load(file)

# =========================================================
# SAVE DATABASE
# =========================================================

def save_patients(data):

    with open(DB_FILE, "w") as file:

        json.dump(data, file, indent=4)

# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    return render_template("login.html")

# =========================================================
# LOGIN
# =========================================================

@app.route("/login", methods=["POST"])
def login():

    role = request.form.get("role")

    username = request.form.get("username")

    password = request.form.get("password")

    # =====================================================
    # RECEPTION LOGIN
    # =====================================================

    if role == "reception":

        if username == RECEPTION_USERNAME and password == RECEPTION_PASSWORD:

            return render_template(
                "reception_dashboard.html"
            )

    # =====================================================
    # DOCTOR LOGIN
    # =====================================================

    elif role == "doctor":

        doctor = DOCTOR_LOGINS.get(username)
        if doctor and password == doctor[0]:
            return render_template(
                "doctor_dashboard.html",
                doctor_name=doctor[1]
            )

    return "<h1>Invalid Login</h1>"

# =========================================================
# RECEPTION DASHBOARD
# =========================================================

@app.route("/reception_dashboard")
def reception_dashboard():

    return render_template(
        "reception_dashboard.html"
    )

# =========================================================
# DOCTOR DASHBOARD
# =========================================================

@app.route("/doctor_dashboard")
def doctor_dashboard_generic():
    # Fallback to a default doctor or login screen if accessed generically
    return redirect("/doctor_records")

@app.route("/doctor_dashboard/<doctor_name>")
def doctor_dashboard(doctor_name):

    return render_template(
        "doctor_dashboard.html",
        doctor_name=doctor_name
    )

# =========================================================
# PATIENT FORM
# =========================================================

@app.route("/patient")
def patient():

    return render_template(
        "patient_form.html"
    )

# =========================================================
# SAVE PATIENT
# =========================================================

@app.route("/save_patient", methods=["POST"])
def save_patient():

    patients = load_patients()

    selected_symptoms = request.form.getlist(
        "symptoms"
    )

    other_symptoms = request.form.get(
        "other_symptoms"
    )

    all_symptoms = ", ".join(
        selected_symptoms
    )

    if other_symptoms:

        if all_symptoms != "":

            all_symptoms += ", "

        all_symptoms += other_symptoms

    patient_data = {

        "name":
        request.form.get("name"),

        "age":
        request.form.get("age"),

        "gender":
        request.form.get("gender"),

        "phone":
        request.form.get("phone"),

        "email":
        request.form.get("email"),

        "emergency_contact":
        request.form.get("emergency_contact"),

        "address":
        request.form.get("address"),

        "blood_group":
        request.form.get("blood_group"),

        "height":
        request.form.get("height"),

        "weight":
        request.form.get("weight"),

        "bmi":
        request.form.get("bmi"),

        "appointment_date":
        request.form.get("appointment_date"),

        "appointment_time":
        request.form.get("appointment_time"),

        "reason_visit":
        request.form.get("reason_visit"),

        "assigned_doctor":
        request.form.get("assigned_doctor"),

        "symptoms":
        all_symptoms,

        "smoking":
        request.form.get("smoking"),

        "tobacco":
        request.form.get("tobacco"),

        "alcohol":
        request.form.get("alcohol"),

        "water_intake":
        request.form.get("water_intake"),

        "activity":
        request.form.get("activity"),

        "surgeries":
        request.form.get("surgeries"),

        "allergies":
        request.form.get("allergies"),

        "medications":
        request.form.get("medications"),

        "diseases":
        request.form.get("diseases"),

        "family_history":
        request.form.get("family_history"),

        "notes":
        request.form.get("notes"),

        "diagnosis_status":
        "Pending",

        "appointments": [

            {

                "appointment_date":
                request.form.get("appointment_date"),

                "appointment_time":
                request.form.get("appointment_time"),

                "doctor_review": "",

                "next_appointment_date": "",

                "next_appointment_time": "",

                "lab_reports": [],

                "status": "Completed",

                "ai_prediction": "",

                "parameters": {},

                "dialysis_risk": ""

            }

        ]

    }

    patients.append(patient_data)

    save_patients(patients)

    return render_template(
        "success.html"
    )

# =========================================================
# MEDICAL HISTORY
# =========================================================

@app.route("/history")
def history():

    patients = load_patients()

    return render_template(
        "medical_history.html",
        patients=patients
    )

# =========================================================
# DELETE PATIENT
# =========================================================

@app.route("/delete_patient/<patient_name>")
def delete_patient(patient_name):
    patients = load_patients()
    patients = [p for p in patients if p["name"] != patient_name]
    save_patients(patients)
    return redirect("/history")

# =========================================================
# PATIENT DETAILS
# =========================================================

@app.route("/patient_details/<patient_name>")
def patient_details(patient_name):

    patients = load_patients()

    for patient in patients:

        if patient["name"] == patient_name:

            return render_template(
                "patient_details.html",
                patient=patient
            )

    return "Patient Not Found"

# =========================================================
# DOCTOR RECORDS
# =========================================================

@app.route("/doctor_records/<doctor_name>")
def doctor_records(doctor_name):

    patients = load_patients()

    filtered_patients = []

    for patient in patients:

        if patient["assigned_doctor"] == doctor_name:

            filtered_patients.append(patient)

    return render_template(
        "doctor_records.html",
        patients=filtered_patients,
        doctor_name=doctor_name
    )

# =========================================================
# DOCTOR PATIENT
# =========================================================

@app.route("/doctor_patient/<patient_name>")
def doctor_patient(patient_name):

    patients = load_patients()

    for patient in patients:

        if patient["name"] == patient_name:

            return render_template(
                "doctor_patient_details.html",
                patient=patient
            )

    return "Patient Not Found"

# =========================================================
# DIAGNOSIS
# =========================================================

@app.route("/diagnosis/<patient_name>")
def diagnosis(patient_name):

    patients = load_patients()

    for patient in patients:

        if patient["name"] == patient_name:

            patient["diagnosis_status"] = \
            "Diagnosis Started"

            save_patients(patients)

            doctor_name = (patient.get("assigned_doctor") or "").strip()
            module_key, module_label = DOCTOR_DISEASE_MODULE.get(
                doctor_name,
                (None, None),
            )
            uploaded_reports = collect_uploaded_reports_for_diagnosis(patient)

            return render_template(
                "diagnosis.html",
                patient=patient,
                disease_module_key=module_key,
                disease_module_label=module_label,
                assigned_doctor_name=doctor_name,
                uploaded_reports=uploaded_reports,
                diagnosis_error=None,
            )

    return "Patient Not Found"

# =========================================================
# AI ANALYZE REPORT
# =========================================================

@app.route(
    "/analyze_report/<patient_name>",
    methods=["POST"]
)
@app.route(
    "/upload_and_analyze/<patient_name>",
    methods=["GET", "POST"]
)

def analyze_report(patient_name):

    if request.method == "GET":
        return redirect(f"/diagnosis/{patient_name}")

    patients = load_patients()
    selected_patient = None
    for p in patients:
        if p["name"] == patient_name:
            selected_patient = p
            break

    if not selected_patient:
        return "Patient Not Found"

    doctor_name = (selected_patient.get("assigned_doctor") or "").strip()
    disease_type, _mod_label = DOCTOR_DISEASE_MODULE.get(
        doctor_name,
        (None, None),
    )

    if disease_type not in ("heart", "diabetes", "kidney"):
        return render_template(
            "diagnosis.html",
            patient=selected_patient,
            disease_module_key=None,
            disease_module_label=None,
            assigned_doctor_name=doctor_name,
            uploaded_reports=collect_uploaded_reports_for_diagnosis(selected_patient),
            diagnosis_error=(
                "No AI disease module is configured for this patient's assigned doctor. "
                "Assign Dr. Sharma (kidney), Dr. Priya (diabetes), or Dr. Mehta / Dr. Reddy (heart)."
            ),
        ), 400

    report_filename = secure_filename(request.form.get("report_filename", "") or "")
    if not report_filename:
        return redirect(f"/diagnosis/{patient_name}")

    allowed = patient_all_lab_filenames(selected_patient)
    if report_filename not in allowed:
        return (
            "Selected lab report is not on file for this patient. "
            "Ask reception to upload it from Medical History.",
            400,
        )

    file_path = os.path.join(UPLOAD_FOLDER, report_filename)
    if not os.path.isfile(file_path):
        return "Lab report file is missing from the server uploads folder.", 404

    ocr = extract_labs_from_file(disease_type, file_path)
    appt_idx = appointment_index_for_lab_file(selected_patient, report_filename)
    if appt_idx is None:
        appt_idx = len(selected_patient["appointments"]) - 1
    selected_patient["appointments"][appt_idx]["pending_ocr"] = {
        "report_filename": report_filename,
        "parameters": ocr["parameters"],
        "embedded_diagnosis": ocr.get("embedded_diagnosis"),
        "disease_type": disease_type,
    }
    save_patients(patients)
    return redirect(f"/confirm_labs/{quote(patient_name)}?report={quote(report_filename)}")


def _store_explained_prediction(patient, filename, disease_type, parameters, explained, embedded_dx):
    prediction_result = explained["prediction"]
    refine_note = None
    if embedded_dx:
        refine_note = (
            "The uploaded report also contained a diagnosis line "
            f"({embedded_dx}). The AI label below is still from the model, not a silent copy."
        )
    else:
        prediction_result, refine_note = refine_prediction(
            disease_type,
            explained["prediction"],
            parameters,
            patient.get("symptoms") or "",
        )

    patient["diagnosis_status"] = "Diagnosis Completed"
    patient.setdefault("lab_analysis", []).append(
        {
            "report_name": filename,
            "parameters": parameters,
            "analyzed_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "diagnosis_label": prediction_result,
            "structured_pdf_diagnosis": embedded_dx,
        }
    )
    ai_result = {
        "disease_type": disease_type,
        "predicted_disease": prediction_result,
        "severity": explained["severity"],
        "risk_percentage": explained["model_probability"],
        "model_probability": explained["model_probability"],
        "confidence_band": explained["confidence_band"],
    }
    if refine_note:
        ai_result["refinement_note"] = refine_note
    if disease_type == "kidney":
        ai_result["dialysis_risk"] = explained.get("dialysis_result")
    patient.setdefault("ai_diagnosis", []).append(ai_result)

    appt_idx = appointment_index_for_lab_file(patient, filename)
    if appt_idx is None:
        appt_idx = len(patient["appointments"]) - 1
    appt = patient["appointments"][appt_idx]
    appt["ai_prediction"] = prediction_result
    appt["parameters"] = parameters
    appt["dialysis_risk"] = explained.get("dialysis_result") or ""
    appt["confidence_score"] = explained["model_probability"]
    appt["confidence_band"] = explained["confidence_band"]
    appt["class_index"] = explained["class_index"]
    appt["class_probabilities"] = explained["class_probabilities"]
    appt["severity"] = explained["severity"]
    appt["organ_risk"] = "High Risk" if explained["model_probability"] >= 90 else "Moderate Risk"
    appt["ai_explanation"] = simple_ai_explanation(
        prediction_result, disease_type, diagnosis_matched_pdf=bool(embedded_dx)
    )
    appt["patient_view"] = explained.get("patient_view")
    appt["doctor_view"] = explained.get("doctor_view")
    appt["shap_explanation"] = explained.get("shap_explanation") or {}
    appt["related_models"] = explained.get("related_models") or []
    appt["disease_type"] = disease_type
    appt.pop("pending_ocr", None)
    appt["lab_analysis_record"] = {
        "report_name": filename,
        "analyzed_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "parameters": parameters,
        "ai_diagnosis": prediction_result,
        "disease_module": disease_type,
        "structured_pdf_diagnosis": embedded_dx,
    }
    return appt_idx


@app.route("/confirm_labs/<patient_name>", methods=["GET", "POST"])
def confirm_labs(patient_name):
    patients = load_patients()
    patient = next((p for p in patients if p["name"] == patient_name), None)
    if not patient:
        return "Patient Not Found", 404
    doctor_name = (patient.get("assigned_doctor") or "").strip()
    disease_type, _ = DOCTOR_DISEASE_MODULE.get(doctor_name, (None, None))
    report_filename = secure_filename(
        request.values.get("report_filename") or request.args.get("report") or ""
    )
    appt_idx = appointment_index_for_lab_file(patient, report_filename)
    if appt_idx is None:
        appt_idx = len(patient.get("appointments") or []) - 1
    pending = (patient["appointments"][appt_idx].get("pending_ocr") or {})
    parameters = pending.get("parameters") or {}
    errors = []

    if request.method == "POST":
        parameters = parameters_from_form(request.form)
        ok, errors = validate_clinical_data(disease_type, parameters)
        if ok:
            try:
                explained = run_explained_prediction(disease_type, parameters, patient)
            except Exception as exc:
                errors = [f"Prediction failed: {exc}"]
            else:
                appt_idx = _store_explained_prediction(
                    patient,
                    report_filename,
                    disease_type,
                    parameters,
                    explained,
                    pending.get("embedded_diagnosis"),
                )
                save_patients(patients)
                return redirect(
                    f"/clinical_dashboard/{quote(patient_name)}?appt={appt_idx}"
                )

    fields = suggested_review_fields(disease_type, parameters)
    return render_template(
        "confirm_labs.html",
        patient=patient,
        report_filename=report_filename,
        parameters=parameters,
        field_keys=fields,
        errors=errors,
        embedded_dx=pending.get("embedded_diagnosis"),
    )

# =========================================================
# DOCTOR REVIEW
# =========================================================

@app.route(
    "/doctor_review/<patient_name>",
    methods=["POST"]
)
def doctor_review(patient_name):

    patients = load_patients()

    for patient in patients:

        if patient["name"] == patient_name:

            latest = patient["appointments"][-1]

            latest["doctor_review"] = request.form.get(
                "doctor_review"
            )

            next_date = request.form.get(
                "next_appointment_date"
            )

            next_time = request.form.get(
                "next_appointment_time"
            )

            latest["next_appointment_date"] = next_date

            latest["next_appointment_time"] = next_time

            if next_date != "":

                patient["appointments"].append(

                    {

                        "appointment_date":
                        next_date,

                        "appointment_time":
                        next_time,

                        "doctor_review": "",

                        "next_appointment_date": "",

                        "next_appointment_time": "",

                        "lab_reports": [],

                        "status": "Upcoming",

                        "ai_prediction": "",

                        "parameters": {},

                        "dialysis_risk": ""

                    }

                )

    save_patients(patients)

    return redirect(
        f"/patient_details/{patient_name}"
    )

# =========================================================
# UPLOAD LAB REPORT
# =========================================================

@app.route(
    "/upload_lab_report/<patient_name>/<int:index>",
    methods=["POST"]
)
def upload_lab_report(patient_name, index):

    patients = load_patients()
    file = request.files.get("lab_report")

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        entry = {
            "filename": filename,
            "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }

        for patient in patients:
            if patient["name"] == patient_name:
                patient["appointments"][index]["lab_reports"].append(entry)

        save_patients(patients)

    return redirect(f"/patient_details/{patient_name}")

# =========================================================
# COMPLETE APPOINTMENT
# =========================================================

@app.route(
    "/complete_appointment/<patient_name>/<int:index>"
)
def complete_appointment(
    patient_name,
    index
):

    patients = load_patients()

    for patient in patients:

        if patient["name"] == patient_name:

            patient["appointments"][index][
                "status"
            ] = "Completed"

    save_patients(patients)

    return redirect(
        f"/patient_details/{patient_name}"
    )

# =========================================================
# ADD APPOINTMENT
# =========================================================

@app.route(
    "/add_appointment/<patient_name>",
    methods=["POST"]
)
def add_appointment(patient_name):
    patients = load_patients()
    appointment_date = request.form.get("appointment_date")
    appointment_time = request.form.get("appointment_time")
    
    for patient in patients:
        if patient["name"] == patient_name:
            patient["appointments"].append({
                "appointment_date": appointment_date,
                "appointment_time": appointment_time,
                "status": "Upcoming",
                "lab_reports": []
            })
            break
            
    save_patients(patients)
    return redirect(f"/patient_details/{patient_name}")

# =========================================================
# LAB ANALYSIS DASHBOARD
# =========================================================

@app.route("/lab_analysis_dashboard")
def lab_analysis_dashboard():

    patients = load_patients()
    focus = (request.args.get("patient") or "").strip()
    analyzed_flag = request.args.get("analyzed") == "1"
    appt_raw = (request.args.get("appt") or "").strip()
    appt_idx = int(appt_raw) if appt_raw.isdigit() else None

    lab_records = _lab_analysis_record_index_rows(patients)

    if not focus:
        return render_template(
            "lab_analysis_dashboard.html",
            patients=[],
            lab_records=lab_records,
            record_picker_mode=True,
            focus_patient="",
            analyzed_complete=analyzed_flag,
            focus_appt_index=None,
            appt_error=None,
        )

    matched = [p for p in patients if p.get("name") == focus]
    if not matched:
        return render_template(
            "lab_analysis_dashboard.html",
            patients=[],
            lab_records=lab_records,
            record_picker_mode=False,
            focus_patient=focus,
            analyzed_complete=analyzed_flag,
            focus_appt_index=appt_idx,
            appt_error="patient_not_found",
        )

    patient = matched[0]
    appointments = patient.get("appointments") or []

    if appt_idx is not None:
        if appt_idx < 0 or appt_idx >= len(appointments):
            return render_template(
                "lab_analysis_dashboard.html",
                patients=[],
                lab_records=lab_records,
                record_picker_mode=False,
                focus_patient=focus,
                analyzed_complete=analyzed_flag,
                focus_appt_index=appt_idx,
                appt_error="bad_appt_index",
            )
        patient_view = dict(patient)
        patient_view["appointments"] = [appointments[appt_idx]]
        patients_view = [patient_view]
    else:
        patients_view = [patient]

    return render_template(
        "lab_analysis_dashboard.html",
        patients=patients_view,
        lab_records=lab_records,
        record_picker_mode=False,
        focus_patient=focus,
        analyzed_complete=analyzed_flag,
        focus_appt_index=appt_idx,
        appt_error=None,
    )

# =========================================================
# AI DIAGNOSIS DASHBOARD
# =========================================================

@app.route("/ai_diagnosis_dashboard")
def ai_diagnosis_dashboard():

    patients = load_patients()
    focus = (request.args.get("patient") or "").strip()
    appt_raw = (request.args.get("appt") or "").strip()
    appt_idx = int(appt_raw) if appt_raw.isdigit() else None

    dashboard_patients = []
    ai_records = []

    for p in patients:
        for appt_index, appt in enumerate(p.get("appointments", [])):
            if appt.get("ai_prediction"):
                d_type = appt.get("disease_type", "Unknown")
                prediction = appt.get("ai_prediction", "")
                params_norm = canonicalize_lab_parameter_keys(
                    appt.get("parameters") or {}
                )
                from_pdf = bool(
                    (appt.get("lab_analysis_record") or {}).get(
                        "structured_pdf_diagnosis"
                    )
                )
                row = {
                    "name": p["name"],
                    "appt_index": appt_index,
                    "age": p.get("age", ""),
                    "symptoms": p.get("symptoms", ""),
                    "doctor": p.get("assigned_doctor", ""),
                    "date": appt.get("appointment_date", ""),
                    "disease_type": d_type,
                    "prediction": prediction,
                    "dialysis": appt.get("dialysis_risk", ""),
                    "confidence": appt.get("confidence_score", "95"),
                    "confidence_band": appt.get("confidence_band", ""),
                    "class_index": appt.get("class_index"),
                    "class_probabilities": appt.get("class_probabilities") or [],
                    "severity": appt.get("severity", "Medium"),
                    "organ_risk": appt.get("organ_risk", "Moderate"),
                    "explanation": appt.get("patient_view") or simple_ai_explanation(
                        prediction,
                        d_type,
                        diagnosis_matched_pdf=from_pdf,
                    ),
                    "doctor_view": appt.get("doctor_view") or "",
                    "shap_explanation": appt.get("shap_explanation", {}),
                    "reports": appt.get("lab_reports", []),
                    "parameters": params_norm,
                    "next_steps": next_steps_for_diagnosis(d_type, prediction),
                }
                dashboard_patients.append(row)
                ai_records.append(
                    {
                        "patient_name": p["name"],
                        "appt_index": appt_index,
                        "appointment_date": appt.get("appointment_date", "—"),
                        "prediction": prediction,
                    }
                )

    if not focus:
        return render_template(
            "ai_diagnosis_dashboard.html",
            patients=[],
            ai_records=ai_records,
            record_picker_mode=True,
            focus_patient="",
            focus_appt_index=None,
        )

    if appt_idx is not None:
        dashboard_patients = [
            row
            for row in dashboard_patients
            if row["name"] == focus and row["appt_index"] == appt_idx
        ]
    else:
        dashboard_patients = [row for row in dashboard_patients if row["name"] == focus]

    return render_template(
        "ai_diagnosis_dashboard.html",
        patients=dashboard_patients,
        ai_records=ai_records,
        record_picker_mode=False,
        focus_patient=focus,
        focus_appt_index=appt_idx,
    )

@app.route("/clinical_dashboard/<patient_name>")
def clinical_dashboard(patient_name):
    patients = load_patients()
    patient = next((p for p in patients if p["name"] == patient_name), None)
    if not patient:
        return "Patient not found", 404
    appt_raw = (request.args.get("appt") or "").strip()
    if appt_raw.isdigit():
        appt = patient["appointments"][int(appt_raw)]
    else:
        _idx, appt = latest_appointment_with_ai(patient)
    dose = get_smart_dose(
        appt.get("disease_type", "diabetes"),
        appt.get("severity", "Moderate"),
        appt.get("ai_prediction"),
        appt.get("parameters", {}),
        patient.get("current_medications", ""),
    )
    appt["smart_dose"] = dose
    twin = simulate_twin(appt, dose)
    appt["digital_twin"] = twin
    save_patients(patients)
    
    return render_template(
        "clinical_dashboard.html",
        patient=patient,
        appt=appt,
        dose=dose,
        twin=twin,
    )

@app.route("/ml_results")
def ml_results():
    return render_template(
        "ml_results.html",
        performance=load_model_performance(),
        shap_tables=load_shap_importance(),
    )

# =========================================================
# SMART DOSE
# =========================================================

@app.route("/smart_dose")
def smart_dose_generic():
    patients = load_patients()
    return render_template("select_patient_for_module.html", patients=patients, module_name="Smart Dose Engine", module_url="smart_dose")

@app.route("/smart_dose/<patient_name>")
def smart_dose(patient_name):
    patients = load_patients()
    for patient in patients:
        if patient["name"] == patient_name:
            _idx, appt = latest_appointment_with_ai(patient)
            dose = get_smart_dose(
                appt.get("disease_type", "diabetes"),
                appt.get("severity", "Moderate"),
                appt.get("ai_prediction"),
                appt.get("parameters", {}),
                patient.get("current_medications", ""),
            )
            appt["smart_dose"] = dose
            save_patients(patients)
            return render_template(
                "smart_dose.html",
                patient=patient,
                dose=dose,
                ai_prediction=appt.get("ai_prediction", ""),
            )
    return "Patient not found", 404

@app.route("/approve_and_digital_twin/<patient_name>", methods=["POST"])
def approve_and_digital_twin(patient_name):
    patients = load_patients()
    for patient in patients:
        if patient["name"] == patient_name:
            notes = request.form.get(
                "notes",
                "Approved regimen — pending Digital Twin simulation.",
            )
            _idx, appt = latest_appointment_with_ai(patient)
            appt["doctor_approval_notes"] = notes
            appt["dose_physician_approved"] = True
            save_patients(patients)
            return redirect("/digital_twin")
    return "Error", 400

# =========================================================
# DIGITAL TWIN
# =========================================================

@app.route("/digital_twin")
def digital_twin_generic():
    patients = load_patients()
    return render_template("select_patient_for_module.html", patients=patients, module_name="Digital Twin Simulation", module_url="digital_twin")

@app.route("/digital_twin/<patient_name>", methods=["GET", "POST"])
def digital_twin(patient_name):
    patients = load_patients()
    for patient in patients:
        if patient["name"] == patient_name:
            _idx, appt = latest_appointment_with_ai(patient)
            dose = get_smart_dose(
                appt.get("disease_type", "diabetes"),
                appt.get("severity", "Moderate"),
                appt.get("ai_prediction"),
                appt.get("parameters", {}),
                patient.get("current_medications", ""),
            )
            appt["smart_dose"] = dose
            twin = simulate_twin(appt, dose)
            appt["digital_twin"] = twin
            save_patients(patients)
            
            simulated_twin = None
            simulated_params = None

            if request.method == "POST":
                # Create simulated parameters
                simulated_params = appt.get("parameters", {}).copy()
                for key, val in request.form.items():
                    if key.startswith("sim_"):
                        param_name = key[4:]
                        if val.strip():
                            simulated_params[param_name] = val.strip()

                # Run ML Simulation
                disease_type = appt.get("disease_type")
                sim_results = simulate_prediction(disease_type, simulated_params)
                
                if sim_results:
                    sim_appt = appt.copy()
                    sim_appt["parameters"] = simulated_params
                    sim_appt["ai_prediction"] = sim_results["prediction"]
                    sim_appt["severity"] = sim_results["severity"]
                    if sim_results.get("dialysis_result"):
                        sim_appt["dialysis_risk"] = sim_results["dialysis_result"]
                    
                    sim_smart_dose = get_smart_dose(
                        disease_type, 
                        sim_results["severity"], 
                        sim_results["prediction"],
                        simulated_params
                    )
                    
                    simulated_twin = simulate_twin(sim_appt, sim_smart_dose)
                    simulated_twin["status"] = sim_smart_dose.get("status", "")

                    # Save simulation in session
                    session[f"sim_{patient_name}"] = {
                        "simulated_params": simulated_params,
                        "simulated_twin": simulated_twin,
                        "sim_results": sim_results,
                        "sim_smart_dose": sim_smart_dose
                    }

            if request.method == "GET":
                session.pop(f"sim_{patient_name}", None)
                save_patients(patients)

            patient_reports = []
            for a in patient.get("appointments") or []:
                rpdf = a.get("report_pdf")
                if rpdf:
                    patient_reports.append(
                        {
                            "pdf": rpdf,
                            "date": a.get("appointment_date", "N/A"),
                        }
                    )
            return render_template(
                "digital_twin.html",
                patient=patient,
                twin=twin,
                simulated_twin=simulated_twin,
                original_params=appt.get("parameters", {}),
                simulated_params=simulated_params,
                patient_reports=patient_reports,
            )
    return "Patient not found", 404

# =========================================================
# MEDICAL REPORTS
# =========================================================

@app.route("/generate_report/<patient_name>")
def generate_report(patient_name):
    patients = load_patients()
    for patient in patients:
        if patient["name"] == patient_name:
            _idx, appt = latest_appointment_with_ai(patient)
            if not appt.get("ai_prediction"):
                return (
                    "Run AI diagnosis for this patient before generating a PDF report.",
                    400,
                )
            filename = generate_patient_pdf(patient)
            appt["report_pdf"] = filename
            save_patients(patients)
            return redirect(url_for("download_report", filename=filename))
    return "Patient not found", 404

@app.route("/medical_reports")
def medical_reports():
    patients = load_patients()
    reports = []
    for p in patients:
        for appt in p.get("appointments", []):
            if "report_pdf" in appt:
                reports.append({
                    "patient_name": p["name"],
                    "date": appt.get("appointment_date", "N/A"),
                    "pdf": appt["report_pdf"]
                })
    return render_template("medical_reports.html", reports=reports)

# =========================================================
# VIEW REPORT (PDF)
# =========================================================

@app.route("/reports/<filename>")
def download_report(filename):
    safe_name = secure_filename(filename)
    filepath = os.path.join(REPORTS_FOLDER, safe_name)
    if not os.path.isfile(filepath):
        return "Report not found", 404
    response = send_file(
        filepath,
        mimetype="application/pdf",
        as_attachment=False,
        download_name=safe_name,
        max_age=0,
    )
    response.headers["Content-Disposition"] = f'inline; filename="{safe_name}"'
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response

# =========================================================
# VIEW UPLOADED FILES
# =========================================================

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, secure_filename(filename))

# =========================================================
# RUN APP
# =========================================================

# =========================================================
# HELPER: RUN SIMULATION (DIGITAL TWIN)
# =========================================================

def simulate_prediction(disease_type, parameters):
    """Runs the ML model for temporary digital twin simulations."""
    try:
        res = run_explained_prediction(disease_type, parameters)
        return {
            "prediction": res["prediction"],
            "risk_percentage": res["model_probability"],
            "severity": res["severity"],
            "dialysis_result": res.get("dialysis_result", ""),
            "shap_explanation": res["shap_explanation"]
        }
    except Exception as e:
        print(f"Simulation error: {e}")
        return None

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)