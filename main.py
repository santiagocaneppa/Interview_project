import os
import json
import openai
import pandas as pd
from dotenv import load_dotenv
from workers.worker_pdfplumber import extract_text_tables
from workers.worker_ocr import extract_text_ocr
from workers.worker_image_preprocess import preprocess_image
from workers.worker_ai_refiner import process_with_openai

# Carregar variáveis do arquivo .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "media", "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "media", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def identify_pdf_type(pdf_path):
    """
    Usa IA para identificar o tipo de PDF e escolher o worker adequado.
    """
    prompt = f"""
    O arquivo PDF contém dados imobiliários. Determine qual categoria ele pertence:
    - 'TABELA' se for um PDF com tabelas estruturadas
    - 'IMAGEM' se for um PDF sem texto direto, apenas imagens
    - 'TEXTO' se for um PDF com texto estruturado.

    Retorne apenas a palavra correspondente sem explicações.

    Arquivo: {os.path.basename(pdf_path)}
    """

    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response["choices"][0]["message"]["content"].strip()


def process_pdf(pdf_path):
    """
    Encaminha o PDF para o worker correto e gera um CSV estruturado.
    """
    doc_type = identify_pdf_type(pdf_path)

    if doc_type == "TABELA":
        extracted_data = extract_text_tables(pdf_path)
    elif doc_type == "IMAGEM":
        processed_images = preprocess_image(pdf_path)
        extracted_data = extract_text_ocr(processed_images)
    elif doc_type == "TEXTO":
        extracted_data = extract_text_ocr(pdf_path)
    else:
        print(f"Erro ao identificar o tipo do PDF: {pdf_path}")
        return

    structured_data = process_with_openai(json.dumps(extracted_data))
    structured_data = json.loads(structured_data)  # Converter resposta JSON da IA

    # Criar DataFrame com colunas necessárias
    df = pd.DataFrame(structured_data,
                      columns=["nome_empreendimento", "unidade", "disponibilidade", "valor", "observações"])

    output_csv = os.path.join(OUTPUT_DIR, "resultado_imoveis.csv")
    df.to_csv(output_csv, sep=";", index=False)

    print(f"Arquivo processado e salvo em {output_csv}")


# Processa todos os PDFs da pasta input
for pdf in os.listdir(INPUT_DIR):
    if pdf.endswith(".pdf"):
        process_pdf(os.path.join(INPUT_DIR, pdf))
