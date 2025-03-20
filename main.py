import os
import json
import logging
import pdfplumber
import pdf2image
import openai
import pandas as pd
from dotenv import load_dotenv
from workers.worker_pdfplumber import extract_tables_from_pdf
from workers.worker_image_preprocess import process_ocr_with_langchain, extract_text_ocr
from workers.worker_pdf_mix import process_pdf_combined

# 🔹 Carregar variáveis do .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# 🔹 Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

# 🔹 Caminhos da aplicação
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "media", "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "media", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 🔹 Caminho do CSV consolidado
output_csv = os.path.join(OUTPUT_DIR, "resultado_imoveis.csv")


def check_pdf_content(pdf_path):
    """
    Verifica se o PDF contém texto extraível e/ou imagens incorporadas.
    Retorna:
    - "TABELA" se houver texto extraível significativo.
    - "IMAGEM" se não houver texto, mas houver imagens.
    - "MIX" se houver texto e imagens simultaneamente.
    - None se não conseguir determinar.
    """
    has_text = False
    has_images = False
    min_text_length = 50  # Definir um mínimo de caracteres reais para ser "TABELA"

    try:
        extracted_text = ""

        # 🔍 Teste de texto extraível
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    extracted_text += text

        # 🖼️ Teste de imagens embutidas
        images = pdf2image.convert_from_path(pdf_path)
        if images:
            has_images = True

        # Se o texto extraído for muito pequeno, tratar como IMAGEM
        if len(extracted_text.strip()) < min_text_length:
            has_text = False
        else:
            has_text = True

    except Exception as e:
        logging.error(f"⚠️ Erro ao processar PDF {pdf_path}: {e}")
        return None

    # 🔹 Classificação final com base nas verificações
    if has_text and has_images:
        return "MIX"
    elif has_text:
        return "TABELA"
    elif has_images:
        return "IMAGEM"
    else:
        return None


def identify_pdf_type(pdf_path):
    """
    Identifica o tipo do PDF primeiro com testes diretos, depois complementa com IA se necessário.
    """
    logging.info(f"📌 Identificando tipo de PDF: {pdf_path}")

    # 🔹 Primeiro, verifica se há texto ou imagens diretamente no documento
    detected_type = check_pdf_content(pdf_path)

    # Se já determinamos que é IMAGEM, TABELA ou MIX, não precisa de IA
    if detected_type:
        logging.info(f"✅ Tipo de PDF identificado automaticamente: {detected_type}")
        return detected_type

    # Se não conseguimos identificar, chamamos a IA para decidir
    logging.info(f"🤖 IA ajudará a identificar o tipo do PDF...")

    prompt = f"""
    O arquivo PDF contém **informações imobiliárias** e pode conter diferentes formatos.  
    Sua tarefa é **analisar cuidadosamente** o conteúdo e classificar corretamente o **tipo** do documento entre **três categorias**.

    ### **Critérios obrigatórios:**
    - **'TABELA'** → O PDF contém **tabelas estruturadas reais**, com colunas bem definidas e dados extraíveis por sistemas de processamento.  
      - As tabelas devem ser reconhecidas **diretamente como texto** no documento (não em imagens).  
      - **Se houver apenas imagens de tabelas (digitalizadas ou escaneadas), marque como 'IMAGEM'.**  

    - **'IMAGEM'** → O PDF contém **apenas imagens**, sem tabelas extraíveis diretamente (OCR necessário).  
      - Se todas as tabelas estiverem embutidas em imagens, marque **'IMAGEM'**.  

    - **'MIX'** → O PDF contém **tabelas extraíveis e imagens ao mesmo tempo**.  
      - **Somente selecione 'MIX' se houver tabelas extraíveis reais e imagens significativas ao mesmo tempo.**  

    ❗ **IMPORTANTE:**  
    🔹 **Responda exclusivamente com uma das seguintes palavras:**  
      - TABELA  
      - IMAGEM  
      - MIX  
    🔹 **Não inclua explicações ou qualquer outro texto na resposta.**

    🔍 **Nome do arquivo PDF analisado**: {os.path.basename(pdf_path)}
    """

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    doc_type = response.choices[0].message.content.strip()
    logging.info(f"✅ PDF identificado pela IA como: {doc_type}")
    return doc_type


def process_pdf(pdf_path):
    """
    Determina qual worker usar e processa o PDF de maneira adequada.
    """
    logging.info(f"📂 Processando PDF: {pdf_path}")

    # Nome do arquivo PDF sem extensão
    file_name = os.path.splitext(os.path.basename(pdf_path))[0]

    # **1️⃣ Identificação Automática do Tipo de PDF**
    doc_type = identify_pdf_type(pdf_path)

    # **2️⃣ Supervisor pode sobrescrever o tipo, se necessário**
    logging.info(f"⚙️ Supervisor pode alterar a escolha automática do worker.")

    # **3️⃣ Escolhe o worker correto com base no tipo identificado**
    extracted_data = None

    if doc_type == "TABELA":
        logging.info("📊 Usando worker de tabelas estruturadas (pdfplumber)")
        extracted_data = extract_tables_from_pdf(pdf_path)

    elif doc_type == "IMAGEM":
        logging.info("🖼️ Usando worker de pré-processamento de imagem e OCR")
        extracted_data = extract_text_ocr(pdf_path)

        # **Processamento adicional com IA via LangChain**
        if extracted_data and "ocr_text" in extracted_data:
            extracted_data = process_ocr_with_langchain(extracted_data)

    elif doc_type == "MIX":
        logging.info("🔀 Usando worker com dados mistos de PDF")
        extracted_data = process_pdf_combined(pdf_path)

    else:
        logging.error(f"⚠️ Erro: Tipo de PDF desconhecido ({pdf_path})")
        return

    # **4️⃣ Verifica se houve extração válida**
    if not extracted_data:
        logging.warning(f"⚠️ Nenhum dado extraído do PDF {pdf_path}. Pulando para o próximo arquivo.")
        return

    # **6️⃣ Salva o JSON estruturado**
    json_output_path = os.path.join(OUTPUT_DIR, f"{file_name}.json")
    with open(json_output_path, "w", encoding="utf-8") as json_file:
        json.dump(extracted_data, json_file, indent=4, ensure_ascii=False)
    logging.info(f"✅ JSON salvo: {json_output_path}")

    # **7️⃣ Criação ou atualização do CSV Consolidado**
    df = pd.DataFrame(extracted_data)

    # 🔹 Remove colunas extras, mantendo apenas as necessárias
    df = df.loc[:, ["nome_empreendimento", "unidade", "disponibilidade", "valor"]]

    # 🔹 Substitui ',' em valores para garantir que o separador decimal não quebre o CSV
    df["valor"] = df["valor"].astype(str).str.replace(".", "", regex=False)  # Remove milhar
    df["valor"] = df["valor"].str.replace(",", ".", regex=False)  # Transforma decimal

    if os.path.exists(output_csv):
        df.to_csv(output_csv, sep=";", index=False, mode="a", header=False, encoding="utf-8")
    else:
        df.to_csv(output_csv, sep=";", index=False, encoding="utf-8")

    logging.info(f"📂 Dados do arquivo {file_name} adicionados ao CSV consolidado: {output_csv}")


# **Executa o processo para cada PDF individualmente**
for pdf in os.listdir(INPUT_DIR):
    if pdf.endswith(".pdf"):
        process_pdf(os.path.join(INPUT_DIR, pdf))
