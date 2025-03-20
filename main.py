import os
import json
import openai
import pandas as pd
import logging
from dotenv import load_dotenv
from workers.worker_pdfplumber import extract_tables_from_pdf
from workers.worker_image_preprocess import process_ocr_with_langchain, extract_text_ocr

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

# Carregar variáveis do .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "media", "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "media", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Caminho do CSV consolidado
output_csv = os.path.join(OUTPUT_DIR, "resultado_imoveis.csv")


def identify_pdf_type(pdf_path):
    """
    Usa IA para identificar o tipo de PDF e sugerir qual worker deve ser usado.
    """
    logging.info(f"Identificando tipo de PDF: {pdf_path}")

    prompt = f"""
    O arquivo PDF contém dados imobiliários e pode conter diferentes formatos de apresentação.
    Sua tarefa é classificar corretamente o tipo do documento entre três categorias:

    - 'TABELA': Se houver tabelas organizadas, com colunas bem definidas (Exemplo: lista de unidades, preços, tamanhos).
    - 'IMAGEM': Se for um documento apenas com imagens e sem texto legível.
    - 'TEXTO': Se for um documento apenas textual, sem formatação tabular.

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
    Determina qual worker usar e processa o PDF de maneira adequada.
    """
    logging.info(f"Processando PDF: {pdf_path}")

    # Nome do arquivo PDF sem extensão
    file_name = os.path.splitext(os.path.basename(pdf_path))[0]

    # **1️⃣ Identificação Automática do Tipo de PDF**
    #doc_type = identify_pdf_type(pdf_path)

    # **2️⃣ Supervisor pode sobrescrever o tipo, se necessário**
    logging.info(f"Supervisor pode alterar a escolha automática do worker.")

    # **3️⃣ Escolhe o worker correto com base no tipo identificado**
    extracted_data = None

    doc_type = "IMAGEM"

    if doc_type == "TABELA":
        logging.info("📊 Usando worker de tabelas estruturadas (pdfplumber)")
        extracted_data = extract_tables_from_pdf(pdf_path)

    elif doc_type == "IMAGEM":
        logging.info("🖼️ Usando worker de pré-processamento de imagem e OCR")
        extracted_data = extract_text_ocr(pdf_path)

        # **Processamento adicional com IA via LangChain**
        if extracted_data and "ocr_text" in extracted_data:
            extracted_data = process_ocr_with_langchain(extracted_data)

    elif doc_type == "TEXTO":
        logging.info("📄 Usando worker de extração de texto")
        extracted_data = extract_text_ocr(pdf_path)

    else:
        logging.error(f"⚠️ Erro: Tipo de PDF desconhecido ({pdf_path})")
        return

    # **4️⃣ Verifica se houve extração válida**
    if not extracted_data:
        logging.warning(f"⚠️ Nenhum dado extraído do PDF {pdf_path}. Pulando para o próximo arquivo.")
        return

    # **5️⃣ Adiciona nome do arquivo ao JSON extraído**
    try:
        extracted_data = json.loads(extracted_data) if isinstance(extracted_data, str) else extracted_data
        if isinstance(extracted_data, list):  # Garante que é uma lista de dicionários
            for item in extracted_data:
                item["arquivo_origem"] = file_name  # Adiciona o nome do PDF como referência
        else:
            logging.error(f"⚠️ Erro: Dados extraídos não estão no formato esperado.")
            return
    except json.JSONDecodeError:
        logging.error("⚠️ Erro ao interpretar resposta da IA. Nenhum dado extraído.")
        return

    # **6️⃣ Salva o JSON estruturado**
    json_output_path = os.path.join(OUTPUT_DIR, f"{file_name}.json")
    with open(json_output_path, "w", encoding="utf-8") as json_file:
        json.dump(extracted_data, json_file, indent=4, ensure_ascii=False)
    logging.info(f"✅ JSON salvo: {json_output_path}")

    # **7️⃣ Criação ou atualização do CSV Consolidado**
    df = pd.DataFrame(extracted_data,
                      columns=["arquivo_origem", "nome_empreendimento", "unidade", "disponibilidade", "valor",
                               "observações"])

    if os.path.exists(output_csv):
        df.to_csv(output_csv, sep=";", index=False, mode="a", header=False,
                  encoding="utf-8")  # Adiciona ao CSV existente
    else:
        df.to_csv(output_csv, sep=";", index=False, encoding="utf-8")  # Cria um novo CSV com cabeçalho

    logging.info(f"📂 Dados do arquivo {file_name} adicionados ao CSV consolidado: {output_csv}")


# **Executa o processo para cada PDF individualmente**
for pdf in os.listdir(INPUT_DIR):
    if pdf.endswith(".pdf"):
        process_pdf(os.path.join(INPUT_DIR, pdf))