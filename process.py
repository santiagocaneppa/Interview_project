import os
import json
import logging
import shutil
import unicodedata
import pdfplumber
import pdf2image
import openai
import pandas as pd
from dotenv import load_dotenv
from workers.worker_pdfplumber import extract_tables_from_pdf
from workers.worker_image_preprocess import process_ocr_with_langchain, extract_text_ocr
from workers.worker_pdf_mix import process_pdf_combined

# Carregar variáveis do .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Configuração do logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Caminho temporário para processamento
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(BASE_DIR, "media", "temp")
os.makedirs(TEMP_DIR, exist_ok=True)


def check_pdf_content(pdf_path):
    has_text = False
    has_images = False
    min_text_length = 50

    try:
        extracted_text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    extracted_text += text

        images = pdf2image.convert_from_path(pdf_path)
        if images:
            has_images = True

        if len(extracted_text.strip()) >= min_text_length:
            has_text = True

    except Exception as e:
        logging.error(f"Erro ao processar PDF {pdf_path}: {e}")
        return None

    if has_text and has_images:
        return "MIX"
    elif has_text:
        return "TABELA"
    elif has_images:
        return "IMAGEM"
    else:
        return None


def identify_pdf_type(pdf_path):
    logging.info(f"Identificando tipo de PDF: {pdf_path}")
    detected_type = check_pdf_content(pdf_path)

    if detected_type:
        logging.info(f"Tipo de PDF identificado automaticamente: {detected_type}")
        return detected_type

    logging.info("Usando IA para classificar o tipo do PDF...")

    prompt = f"""
    O arquivo PDF contém **informações imobiliárias** e pode conter diferentes formatos.
    Sua tarefa é **analisar cuidadosamente** o conteúdo e classificar corretamente o **tipo** do documento entre **três categorias**.

    **Critérios obrigatórios:**
    - **TABELA** → O PDF contém **tabelas estruturadas reais**, com colunas bem definidas e dados extraíveis por sistemas de processamento.
      - As tabelas devem ser reconhecidas **diretamente como texto** no documento (não em imagens).
      - **Se houver apenas imagens de tabelas (digitalizadas ou escaneadas), marque como 'IMAGEM'.**

    - **IMAGEM** → O PDF contém **apenas imagens**, sem tabelas extraíveis diretamente (OCR necessário).
      - Se todas as tabelas estiverem embutidas em imagens, marque **'IMAGEM'**.

    - **MIX** → O PDF contém **tabelas extraíveis e imagens ao mesmo tempo**.
      - **Somente selecione 'MIX' se houver tabelas extraíveis reais e imagens significativas ao mesmo tempo.**

    ❗ **IMPORTANTE:**
    - Responda exclusivamente com uma das seguintes palavras: `TABELA`, `IMAGEM`, `MIX`
    - **Não inclua explicações ou qualquer outro texto na resposta.**

    Nome do arquivo PDF analisado: {os.path.basename(pdf_path)}
    """

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    doc_type = response.choices[0].message.content.strip()
    logging.info(f"PDF identificado pela IA como: {doc_type}")
    return doc_type


def process_pdfs(input_dir: str, output_csv_path: str):
    logging.info(f"Processando PDFs do diretório: {input_dir}")
    logging.info(f"CSV consolidado será salvo em: {output_csv_path}")

    all_extracted_data = []

    for file in os.listdir(input_dir):
        if file.endswith(".pdf"):
            temp_pdf_path = os.path.join(TEMP_DIR, file)
            original_pdf_path = os.path.join(input_dir, file)
            shutil.copy2(original_pdf_path, temp_pdf_path)

            logging.info(f"Arquivo movido para temp: {file}")

            extracted_data = process_pdf(temp_pdf_path)

            # Obtenção nome do arquivo.pdf para nomear empreendimento
            file_name = os.path.splitext(file)[0]
            safe_name = unicodedata.normalize('NFKD', file_name).encode('ASCII', 'ignore').decode('utf-8').replace(" ", "_")

            if extracted_data:
                for row in extracted_data:
                    row["nome_empreendimento"] = safe_name
                all_extracted_data.extend(extracted_data)

            os.remove(temp_pdf_path)
            logging.info(f"Arquivo removido: {file}")

    if all_extracted_data:
        df = pd.DataFrame(all_extracted_data, columns=["nome_empreendimento", "unidade", "disponibilidade", "valor"])
        df["valor"] = df["valor"].astype(str).str.replace(".", "", regex=False)
        df["valor"] = df["valor"].str.replace(",", ".", regex=False)
        df.to_csv(output_csv_path, sep=";", index=False, encoding="utf-8")
        logging.info(f"CSV consolidado salvo em: {output_csv_path}")
    else:
        logging.warning(f"Nenhum dado extraído dos PDFs no diretório {input_dir}")


def process_pdf(pdf_path):
    logging.info(f"Processando PDF: {pdf_path}")

    file_name = os.path.splitext(os.path.basename(pdf_path))[0]
    doc_type = identify_pdf_type(pdf_path)

    extracted_data = None

    if doc_type == "TABELA":
        extracted_data = extract_tables_from_pdf(pdf_path)
    elif doc_type == "IMAGEM":
        extracted_data = extract_text_ocr(pdf_path)
        if extracted_data and "ocr_text" in extracted_data:
            extracted_data = process_ocr_with_langchain(extracted_data)
    elif doc_type == "MIX":
        extracted_data = process_pdf_combined(pdf_path)
    else:
        logging.error(f"Tipo de PDF desconhecido: {pdf_path}")
        return None

    if not extracted_data:
        logging.warning(f"Nenhum dado extraído do PDF {pdf_path}.")
        return None

    json_output_path = os.path.join(TEMP_DIR, f"{file_name}.json")
    with open(json_output_path, "w", encoding="utf-8") as json_file:
        json.dump(extracted_data, json_file, indent=4, ensure_ascii=False)

    return extracted_data
