import pdfplumber

def extract_text_tables(pdf_path):
    extracted_data = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            table = page.extract_table()
            if table:
                extracted_data.extend(table)
    return extracted_data
