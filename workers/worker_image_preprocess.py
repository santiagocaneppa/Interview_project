import os
import json
import logging
import cv2
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
    """ Aplica pré-processamento na imagem para melhorar a extração OCR. """
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)  # Converter para escala de cinza
    thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]  # Binarização
    return thresh


def extract_text_ocr(pdf_path):
    """ Converte PDF para imagens e extrai texto usando OCR. """
    logging.info(f"Convertendo PDF para imagens e extraindo texto OCR: {pdf_path}")

    extracted_text = []
    images = pdf2image.convert_from_path(pdf_path)

    for img in images:
        # Pré-processa a imagem antes da extração
        processed_img = preprocess_image(img)

        # Executa OCR
        text = pytesseract.image_to_string(processed_img, lang="por")
        extracted_text.append(text.strip())

    if not extracted_text:
        logging.warning("Nenhum texto extraído do PDF via OCR.")
        return None

    # Retorna os textos extraídos organizados em um dicionário
    return {"ocr_text": extracted_text}


def process_ocr_with_langchain(ocr_data):
    """ Usa LangChain + OpenAI para organizar os dados extraídos corretamente. """
    if not ocr_data or "ocr_text" not in ocr_data:
        logging.error("Nenhum dado foi extraído para processar via OCR.")
        return None

    logging.info("Enviando dados extraídos via OCR para a IA via LangChain...")

    # Modelo de IA usando OpenAI via LangChain (com temperatura 0 para máxima precisão)
    llm = ChatOpenAI(model="gpt-4-turbo", openai_api_key=openai_api_key, temperature=0)

    # Define o modelo de saída (JSON)
    parser = JsonOutputParser()

    # Template do Prompt **MELHORADO PARA OCR**
    prompt = PromptTemplate(
        template="""
        O seguinte conteúdo textual foi extraído de um PDF imobiliário usando OCR. 
        Sua tarefa é interpretar, organizar e reestruturar os dados corretamente.

        ⚠ **ATENÇÃO**:
        - Algumas informações podem estar desorganizadas ou desalinhadas devido ao OCR.
        - Todos os campos devem seguir um padrão fixo e, caso algum valor esteja ausente, deve ser preenchido como `"Indeterminado"` ou `null`.

        ### **Texto extraído via OCR do PDF**
        ```json
        {ocr_text}
        ```

        ### **Tarefa**
        - Identifique e organize corretamente os dados de cada unidade imobiliária.
        - Caso algum campo esteja ausente, use `"Indeterminado"`, exceto onde especificado que deve ser `null`.
        - Combine informações fragmentadas e formate os valores corretamente.

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
        input_variables=["ocr_text"]
    )

    chain = prompt | llm | parser

    # Gerar resposta da IA
    response = chain.invoke({
        "ocr_text": json.dumps(ocr_data["ocr_text"], indent=2, ensure_ascii=False)
    })

    logging.info("✅ Dados processados com sucesso pela IA para OCR.")
    return response
