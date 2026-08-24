"""Project and writable data locations for local runs and Render."""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.environ.get("DATA_DIR") or BASE_DIR)


def project_path(*parts):
    return os.path.join(BASE_DIR, *parts)


def data_path(*parts):
    return os.path.join(DATA_DIR, *parts)


UPLOAD_FOLDER = data_path("uploads")
REPORTS_FOLDER = data_path("reports")
DATABASE_DIR = data_path("database")
DB_FILE = data_path("database", "patients.json")
MEDICINES_FILE = project_path("database", "medicines.json")
