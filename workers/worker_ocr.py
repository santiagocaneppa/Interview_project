import pytesseract
from pdf2image import convert_from_path

def extract_text_ocr(pdf_path):
    extracted_text = ""
    images = convert_from_path(pdf_path)
    for img in images:
        text = pytesseract.image_to_string(img, lang='por')
        extracted_text += text + "\n"
    return extracted_text
