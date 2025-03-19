import os
import json
import openai
import pandas as pd
import logging
from dotenv import load_dotenv
from workers.worker_pdfplumber import extract_text_tables
from workers.worker_ocr import extract_text_ocr
from workers.worker_image_preprocess import extract_text_with_positions
from workers.worker_ai_refiner import process_with_openai

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

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
    logging.info(f"Identificando tipo de PDF: {pdf_path}")

    prompt = f"""
    O arquivo PDF contém dados imobiliários e pode conter diferentes formatos de apresentação.
    Sua tarefa é classificar corretamente o tipo de documento entre três categorias:

    - 'TABELA': Se houver tabelas organizadas, com colunas bem definidas (Exemplo: lista de unidades, preços, tamanhos).
    - 'IMAGEM': Se for um documento apenas com imagens e sem texto legível.
    - 'TEXTO': Se for um documento apenas textual, sem formatação tabular.

    **Exemplo de uma tabela que deve ser classificada como 'TABELA':**
    Unidade | Metragem | Entrada R$ | Parcelas | Balões | Financiamento | Preço Total
    ------- | -------- | ---------- | -------- | ------ | ------------- | ------------
    204-205 | 67m²    | 48.810,00  | 23x2360  | 4x7000 | 360.940,00    | 492.030,00
    504-505 | 67m²    | 49.390,00  | 23x2400  | 4x7000 | 367.000,00    | 499.590,00

    **Exemplo de um documento textual que deve ser classificado como 'TEXTO':**
    "As unidades do Residencial Vila Montreal possuem metragem privativa de 67m² e incluem entrada de R$ 48.810,00.
    O financiamento pode ser realizado em 23 parcelas de R$ 2.360,00 com balões semestrais de R$ 7.000,00."

    **Exemplo de um documento que deve ser classificado como 'IMAGEM':**
    - PDF sem texto selecionável, contendo apenas imagens escaneadas.

    **Agora, classifique o seguinte arquivo:** {os.path.basename(pdf_path)}
    Responda apenas com uma das palavras: 'TABELA', 'IMAGEM' ou 'TEXTO'.
    """

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    doc_type = response.choices[0].message.content.strip()
    logging.info(f"PDF identificado como: {doc_type}")
    return doc_type



def process_pdf(pdf_path):
    """
    Encaminha o PDF para o worker correto e gera um CSV estruturado.
    """
    logging.info(f"Processando PDF: {pdf_path}")
    doc_type = identify_pdf_type(pdf_path)

    extracted_data = None

    if doc_type == "TABELA":
        logging.info("Usando worker de tabelas estruturadas (pdfplumber)")
        extracted_data = extract_text_tables(pdf_path)
    elif doc_type == "IMAGEM":
        logging.info("Usando worker de pré-processamento de imagem e OCR")
        processed_images = extract_text_with_positions(pdf_path)
        extracted_data = extract_text_ocr(processed_images)
    elif doc_type == "TEXTO":
        logging.info("Usando worker de OCR")
        extracted_data = extract_text_ocr(pdf_path)
    else:
        logging.error(f"Erro ao identificar o tipo do PDF: {pdf_path}")
        return

    logging.info("Enviando dados extraídos para IA para refinamento")
    logging.info(f"Dados extraídos antes do refinamento: {extracted_data}")

    if not extracted_data:
        logging.warning(f"Nenhum dado extraído do PDF {pdf_path}")
        return

    structured_data = process_with_openai(json.dumps(extracted_data))
    if isinstance(structured_data, str):
        structured_data = json.loads(structured_data)  # Converter resposta JSON da IA

    # Criar DataFrame com colunas necessárias
    df = pd.DataFrame(structured_data,
                      columns=["nome_empreendimento", "unidade", "disponibilidade", "valor", "observações"])

    output_csv = os.path.join(OUTPUT_DIR, "resultado_imoveis.csv")
    df.to_csv(output_csv, sep=";", index=False)

    logging.info(f"Arquivo processado e salvo em {output_csv}")


# Processa todos os PDFs da pasta input
for pdf in os.listdir(INPUT_DIR):
    if pdf.endswith(".pdf"):
        process_pdf(os.path.join(INPUT_DIR, pdf))
