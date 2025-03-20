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

    O formato JSON de saída **deve ser exatamente o seguinte**:
    ```json
    [
        {
    "nome_empreendimento": "Residencial Vila Montreal",
            "unidade": "204-205",
            "disponibilidade": "Disponível",
            "valor": "492.030,00",
            "observações": "Garden consultar"
        },
        {
    "nome_empreendimento": "Residencial Vila Montreal",
            "unidade": "304-305",
            "disponibilidade": "Reservado",
            "valor": "499.590,00",
            "observações": Null
        }
    ]
    ```

    **Regras de preenchimento:**
    - Todos os objetos da lista devem ter **exatamente** as mesmas chaves.
    - Se uma informação estiver ausente, deve ser retornado `Null`.
    - O número da unidade deve ser formatado corretamente.
    - O valor deve seguir o formato `000.000,00` para valores monetários.

    Aqui está o texto extraído do PDF:
    {text}

    Retorne **exclusivamente** um JSON válido sem explicações adicionais.
    """

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        return []  # Retorna lista vazia caso a IA falhe na conversão para JSON