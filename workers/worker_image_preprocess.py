import pytesseract
import cv2
import numpy as np
from pdf2image import convert_from_path


def extract_text_with_positions(pdf_path):
    images = convert_from_path(pdf_path)
    all_data = []

    for img in images:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)[1]

        # Extração com coordenadas
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, lang='por')
        all_data.append(data)

    return all_data

