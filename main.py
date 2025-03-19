import pdfplumber
import pytesseract
from pdf2image import convert_from_path
import pandas as pd
import openai
import json
import os

# Configuração da API da OpenAI
openai.api_key = "SUA_OPENAI_API_KEY"

# Definir caminho base do projeto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "media", "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "media", "output")

# Garantir que a pasta de saída existe
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Listar arquivos na pasta de entrada
pdf_files = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) if f.endswith(".pdf")]


# Função para extrair tabelas de PDFs estruturados
def extract_text_tables(pdf_path):
    extracted_data = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            table = page.extract_table()
            if table:
                extracted_data.extend(table)
    return extracted_data


# Função para extrair texto de PDFs com imagens (OCR)
def extract_text_ocr(pdf_path):
    extracted_text = ""
    images = convert_from_path(pdf_path)
    for img in images:
        text = pytesseract.image_to_string(img, lang='por')
        extracted_text += text + "\n"
    return extracted_text


# Função para processar texto extraído usando a OpenAI
def process_with_openai(text, empreendimento):
    prompt = f"""
    Extraia as seguintes informações do texto abaixo sobre um empreendimento imobiliário:
    - Nome do empreendimento
    - Número da unidade
    - Status da unidade (Disponível, Reservado, Permuta, Vendido, etc.)
    - Valor da unidade (se disponível, caso contrário, \"Indisponível\")

    Texto extraído:
    {text}

    Responda no formato JSON:
    [
        {{"nome_empreendimento": "{empreendimento}", "unidade": "101", "disponibilidade": "Disponível", "valor": "R$ 350.000,00"}},
        {{"nome_empreendimento": "{empreendimento}", "unidade": "102", "disponibilidade": "Reservado", "valor": "Indisponível"}}
    ]
    """

    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "Você é um especialista em extração de dados de documentos imobiliários."},
            {"role": "user", "content": prompt}],
        temperature=0
    )

    try:
        return json.loads(response["choices"][0]["message"]["content"])
    except:
        return []


# Processamento dos PDFs
all_data = []
for pdf in pdf_files:
    empreendimento = os.path.splitext(os.path.basename(pdf))[0]

    try:
        extracted_data = extract_text_tables(pdf)
        extracted_text = extract_text_ocr(pdf) if not extracted_data else ""

        # Formatar dados para envio à OpenAI
        formatted_text = json.dumps(extracted_data) if extracted_data else extracted_text
        structured_data = process_with_openai(formatted_text, empreendimento)

        all_data.extend(structured_data)
    except Exception as e:
        print(f"Erro ao processar {pdf}: {e}")

# Criar DataFrame e salvar CSV
df = pd.DataFrame(all_data, columns=["nome_empreendimento", "unidade", "disponibilidade", "valor"])
csv_path = os.path.join(OUTPUT_DIR, "resultado_imoveis.csv")
df.to_csv(csv_path, sep=";", index=False)

print(f"Processo concluído! Arquivo salvo em {csv_path}")
