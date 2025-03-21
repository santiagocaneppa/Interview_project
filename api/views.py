import os
import logging
from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse
from process import process_pdfs

router = APIRouter()


@router.post("/process/", summary="Processar PDFs e salvar CSV", tags=["PDF Processing"])
async def process_pdf_api(
        pdf_path: str = Form(...),
        output_dir: str = Form(...)
):
    """
    **Processamento de PDF**

    - Processa **todos os arquivos PDF** dentro do diretório especificado e gera um CSV consolidado.
    - O usuário fornece:
        - O **caminho da pasta** onde estão os arquivos PDF (`pdf_path`).
        - O **caminho do diretório de saída** onde será salvo o CSV (`output_dir`).

    **Parâmetros de Entrada:**

    - `pdf_path`: Caminho do diretório contendo os arquivos PDF. *(Obrigatório)*
    - `output_dir`: Caminho do diretório onde o CSV será salvo. *(Obrigatório)*

    **Saída:**

    - `message`: Confirmação de sucesso.
    - `output_csv`: Caminho do CSV gerado.

    **Observações:**

    - O diretório de entrada (`pdf_path`) **precisa existir** e conter arquivos PDF.
    - O diretório de saída (`output_dir`) **precisa ser acessível**.

    """

    # Validar se o diretório de entrada existe
    if not os.path.exists(pdf_path) or not os.path.isdir(pdf_path):
        return JSONResponse(
            content={"error": "O diretório de PDFs não foi encontrado ou não é válido."},
            status_code=400,
        )

    # Validar se o diretório de saída existe
    if not os.path.exists(output_dir):
        return JSONResponse(
            content={"error": "O diretório de saída não existe. Verifique o caminho informado."},
            status_code=400,
        )

    # Definir caminho do CSV final
    output_csv_path = os.path.join(output_dir, "resultado_imoveis.csv")

    # Processar todos os PDFs dentro do diretório
    process_pdfs(pdf_path, output_csv_path)

    return JSONResponse(
        content={"message": "Processamento concluído!", "output_csv": output_csv_path},
        status_code=200,
    )
