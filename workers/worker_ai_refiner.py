import openai
import json
import os
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def process_with_openai(text):
    """
    Usa IA para estruturar e organizar tabelas extraídas via OCR.
    """
    prompt = f"""
    O texto abaixo foi extraído de um PDF contendo tabelas imobiliárias. O formato original da tabela pode ter sido perdido devido ao OCR.
    Sua tarefa é reconstruir e organizar os dados no formato correto, garantindo que cada coluna seja extraída corretamente.

    Estruture os dados no formato JSON com as seguintes colunas:
    - "nome_empreendimento": Nome do empreendimento
    - "unidade": Número da unidade
    - "disponibilidade": Status da unidade (Disponível, Reservado, Vendido, Permuta)
    - "valor": Valor do imóvel (se não existir, retornar "Indisponível")
    - "observações": Informações adicionais (se houver)

    Aqui está o texto extraído do PDF:
    {text}

    Retorne **apenas** um JSON estruturado sem explicações adicionais.
    """

    response = openai.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return json.loads(response.choices[0].message.content)
