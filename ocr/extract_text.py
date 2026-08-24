import pdfplumber
import pytesseract
import cv2
import os
import shutil

_tesseract_env = (os.environ.get("TESSERACT_CMD") or "").strip()
_windows_tesseract = r"C:\Users\ramya\OneDrive\Desktop\tesseract.exe"
if _tesseract_env:
    pytesseract.pytesseract.tesseract_cmd = _tesseract_env
elif os.path.isfile(_windows_tesseract):
    pytesseract.pytesseract.tesseract_cmd = _windows_tesseract
else:
    _on_path = shutil.which("tesseract")
    if _on_path:
        pytesseract.pytesseract.tesseract_cmd = _on_path

# ------------------------------------

def extract_text_from_pdf(pdf_path):

    text = ""

    with pdfplumber.open(pdf_path) as pdf:

        for page in pdf.pages:

            extracted = page.extract_text()

            if extracted:

                text += extracted + "\n"

    return text

# ------------------------------------

def extract_text_from_image(image_path):

    image = cv2.imread(image_path)

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    text = pytesseract.image_to_string(gray)

    return text

# ------------------------------------

def extract_report_text(file_path):

    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".pdf":

        return extract_text_from_pdf(file_path)

    elif extension in [

        ".png",
        ".jpg",
        ".jpeg"

    ]:

        return extract_text_from_image(file_path)

    else:

        return "Unsupported File"