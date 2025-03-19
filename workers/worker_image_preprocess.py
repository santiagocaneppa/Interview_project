import cv2
import numpy as np
from pdf2image import convert_from_path


def preprocess_image(pdf_path):
    images = convert_from_path(pdf_path)
    processed_images = []

    for img in images:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)  # Escala de cinza
        img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)[1]  # Remove ruído
        processed_images.append(img)

    return processed_images
