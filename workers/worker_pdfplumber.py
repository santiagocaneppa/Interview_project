import pdfplumber
import re


def clean_text(text):
    """ Remove espaços extras e caracteres indesejados. """
    return text.replace("\n", " ").strip() if text else ""


def extract_text_tables(pdf_path):
    extracted_data = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            table = page.extract_table()
            if table:
                # Limpar os cabeçalhos e garantir que todas as colunas estão alinhadas
                header = [clean_text(col) for col in table[0] if col]  # Remove valores None

                structured_rows = []
                for row in table[1:]:  # Ignora o cabeçalho
                    cleaned_row = [clean_text(cell) for cell in row]
                    # Se a linha estiver vazia, ignore
                    if any(cleaned_row):
                        structured_rows.append(cleaned_row)

                # Reestruturando o CSV no formato correto
                for row in structured_rows:
                    if len(row) == len(header):  # Garantir alinhamento correto
                        extracted_data.append(dict(zip(header, row)))

    return extracted_data
