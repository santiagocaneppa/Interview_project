import os
import json
import logging
import cv2
import pdfplumber
import pdf2image
import pytesseract
import numpy as np
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

# Carregar variáveis do .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)


def preprocess_image(image):
    """
    Aplica pré-processamento na imagem para melhorar a extração OCR.
    """
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)  # Converter para escala de cinza
    enhanced = cv2.convertScaleAbs(gray, alpha=1.2, beta=10)  # Melhorar contraste
    return enhanced


def extract_text_ocr(pdf_path):
    """
    Converte PDF para imagens e extrai texto usando OCR.
    """
    logging.info(f"📄 Convertendo PDF para imagens e extraindo texto OCR: {pdf_path}")

    extracted_text = []
    images = pdf2image.convert_from_path(pdf_path, dpi=300)  # Aumenta DPI para melhor precisão

    for img in images:
        processed_img = preprocess_image(img)
        text = pytesseract.image_to_string(processed_img, lang="por", config="--psm 6")
        extracted_text.append(text.strip())

    if not extracted_text:
        logging.warning("⚠️ Nenhum texto extraído do PDF via OCR.")
        return None

    return {"ocr_text": extracted_text}


def extract_tables_from_pdf(pdf_path):
    """
    Extrai tabelas e contexto do PDF usando pdfplumber.
    """
    logging.info(f"📊 Extraindo tabelas do PDF: {pdf_path}")
    extracted_tables = []
    context_info = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            table = page.extract_table()

            if text:
                context_info.append(text.strip())

            if table:
                cleaned_table = [[cell.strip() if cell else "" for cell in row] for row in table]
                extracted_tables.append(cleaned_table)

    if not extracted_tables and not context_info:
        logging.warning("⚠️ Nenhuma tabela ou texto extraído do PDF.")
        return None

    return {"tables": extracted_tables, "context": context_info}


def process_pdf_combined(pdf_path):
    """
    Processa PDFs que contêm **TABELAS, IMAGENS e TEXTOS** misturados.
    """
    logging.info(f"📂 Processando PDF (Misto): {pdf_path}")

    # Nome do arquivo PDF sem extensão
    file_name = os.path.splitext(os.path.basename(pdf_path))[0]

    # **Extração de todos os formatos possíveis**
    table_data = extract_tables_from_pdf(pdf_path)
    ocr_data = extract_text_ocr(pdf_path)

    # Unificação dos dados extraídos
    extracted_data = {
        "tables": table_data["tables"] if table_data else [],
        "context": table_data["context"] if table_data else [],
        "ocr_text": ocr_data["ocr_text"] if ocr_data else []
    }

    if not any([extracted_data["tables"], extracted_data["context"], extracted_data["ocr_text"]]):
        logging.warning(f"⚠️ Nenhum dado relevante extraído do PDF {pdf_path}.")
        return None

    logging.info("🧠 Enviando dados extraídos para IA via LangChain...")

    # Modelo de IA usando OpenAI via LangChain
    llm = ChatOpenAI(model="gpt-4-turbo", openai_api_key=openai_api_key, temperature=0)

    # Define o modelo de saída (JSON)
    parser = JsonOutputParser()

    # Template do Prompt **MELHORADO PARA MISTO**
    prompt = PromptTemplate(
        template="""
        O seguinte conjunto de informações foi extraído de um PDF imobiliário contendo múltiplos formatos de dados (Tabelas, Imagens e Textos). 
        Sua tarefa é interpretar, organizar e reestruturar esses dados corretamente.

        ⚠ **ATENÇÃO**:
        - Algumas tabelas podem estar desalinhadas ou incompletas.
        - Algumas informações podem estar no OCR em vez da tabela.
        - Todos os campos devem seguir um padrão fixo e, caso algum valor esteja ausente, deve ser preenchido como `"Indeterminado"` ou `null`.

        ### **Dados extraídos do PDF**
        **Tabelas extraídas:**
        ```json
        {tables}
        ```

        **Texto complementar extraído do PDF:**
        ```json
        {context}
        ```

        **Texto extraído via OCR do PDF:**
        ```json
        {ocr_text}
        ```

        ### **Tarefa**
        - Identifique e organize corretamente os dados de cada unidade imobiliária.
        - Combine informações da tabela, contexto e OCR para garantir a extração correta.
        - Caso algum campo esteja ausente, use `"Indeterminado"`, exceto onde especificado que deve ser `null`.
        - Formate os valores corretamente.

        ### **Formato de saída esperado (JSON)**
        ```json
        [
            {{
                "nome_empreendimento": "Nome do Empreendimento",
                "unidade": "204-205",
                "disponibilidade": "Disponível",
                "valor": "492.030,00"
            }},
            {{
                "nome_empreendimento": "Nome do Empreendimento",
                "unidade": "304-305",
                "disponibilidade": "Reservado",
                "valor": "499.590,00"
            }}
        ]
        ```

        ### **Regras IMPORTANTES:**
        - O JSON **DEVE** seguir o formato exato acima, sem campos extras.
        - **Colunas e seus formatos obrigatórios**:
            - `"nome_empreendimento"`: Nome do empreendimento imobiliário. **(string)**
            - `"unidade"`: Número da unidade. **(string)**
            - `"disponibilidade"`: Estado da unidade (`"Disponível"`, `"Reservado"`, `"Permuta"`, ou `"Indeterminado"` caso não especificado). **(string)**
            - `"valor"`: Valor do imóvel, **sempre no formato `000.000,00`** (exemplo: `"492.030,00"`). Se não existir, preencha com `"Indeterminado"`. **(string)**
        - O JSON deve ser **100% válido** e **sem explicações adicionais**.
        """,
        input_variables=["tables", "context", "ocr_text"]
    )

    # Criar a cadeia do LangChain
    chain = prompt | llm | parser

    # Gerar resposta da IA
    response = chain.invoke({
        "tables": json.dumps(extracted_data["tables"], indent=2, ensure_ascii=False),
        "context": json.dumps(extracted_data["context"], indent=2, ensure_ascii=False),
        "ocr_text": json.dumps(extracted_data["ocr_text"], indent=2, ensure_ascii=False)
    })

    logging.info("✅ Dados processados com sucesso pela IA.")
    return response
