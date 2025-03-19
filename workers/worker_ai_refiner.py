import openai
import json

openai.api_key = "SUA_OPENAI_API_KEY"

def process_with_openai(text):
    """
    Usa IA para estruturar os dados extraídos.
    """
    prompt = f"""
    O texto abaixo foi extraído de um PDF de informações imobiliárias.
    Estruture os dados no formato JSON com:
    - nome_empreendimento
    - unidade
    - disponibilidade (Disponível, Reservado, Vendido, Permuta)
    - valor (se não existir, retornar "Indisponível").

    Texto:
    {text}

    Responda apenas em JSON:
    """

    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response["choices"][0]["message"]["content"]
