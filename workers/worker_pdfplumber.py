import os
import json
import pdfplumber
import logging
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

def extract_tables_from_pdf(pdf_path):
    """ Extrai tabelas e contexto do PDF usando pdfplumber. """
    logging.info(f"Extraindo tabelas do PDF: {pdf_path}")
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

    if not extracted_tables:
        logging.warning("Nenhuma tabela extraída do PDF.")
        return None

    return {"tables": extracted_tables, "context": context_info}

def process_with_langchain(pdf_data):
    """ Usa LangChain + OpenAI para organizar os dados extraídos corretamente. """
    if not pdf_data:
        logging.error("Nenhum dado foi extraído para processar.")
        return None

    logging.info("Enviando dados extraídos para a IA via LangChain...")

    # Modelo de IA usando OpenAI via LangChain (com temperatura 0 para máxima precisão)
    llm = ChatOpenAI(model="gpt-4-turbo", openai_api_key=openai_api_key, temperature=0)

    # Define o modelo de saída (JSON)
    parser = JsonOutputParser()

    # Template do Prompt **MELHORADO**
    prompt = PromptTemplate(
        template="""
        O seguinte conjunto de tabelas foi extraído de um PDF imobiliário. 
        Sua tarefa é interpretar, organizar e reestruturar esses dados no formato correto.

        ⚠ **ATENÇÃO**:
        - Algumas tabelas podem estar desalinhadas ou com colunas incompletas.
        - Algumas informações podem estar no texto do contexto ao invés da tabela.
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

        ### **Tarefa**
        - Você deve organizar os dados para que cada **linha** represente uma unidade imobiliária corretamente preenchida.
        - Caso algum campo esteja ausente, use `"Indeterminado"`, exceto onde especificado que deve ser `null`.
        - Se um campo estiver quebrado em várias linhas, combine as informações corretamente.

        ### **Formato de saída esperado (JSON)**
        ```json
        [
            {{
                "nome_empreendimento": "Nome do Empreendimento",
                "unidade": "204-205",
                "disponibilidade": "Disponível",
                "valor": "492.030,00",
                "observações": "Garden consultar"
            }},
            {{
                "nome_empreendimento": "Nome do Empreendimento",
                "unidade": "304-305",
                "disponibilidade": "Reservado",
                "valor": "499.590,00",
                "observações": null
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
            - `"observações"`: Informações adicionais. Se não houver, retorne `null`. **(string ou null)**
        - O JSON deve ser **100% válido** e **sem explicações adicionais**.
        """,
        input_variables=["tables", "context"]
    )

    # Criar a cadeia do LangChain
    chain = prompt | llm | parser

    # Gerar resposta da IA
    response = chain.invoke({
        "tables": json.dumps(pdf_data["tables"], indent=2, ensure_ascii=False),
        "context": json.dumps(pdf_data["context"], indent=2, ensure_ascii=False)
    })

    logging.info("Dados processados com sucesso pela IA.")
    return response
