import logging
from fastapi import FastAPI
from api.urls import api_router

# 🔹 Configuração do logging para depuração detalhada
logging.basicConfig(
    level=logging.DEBUG,  # Ativa modo DEBUG
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# 🔹 Configuração da API FastAPI com MODO DEBUG ativado
app = FastAPI(
    title="PDF Processor API",
    description="API para processamento de PDFs com extração de tabelas e OCR",
    version="1.0.0",
    debug=True  # Modo depuração ativado
)

# 🔹 Inclui as rotas da API
app.include_router(api_router)

# 🔹 Rota inicial para testar se a API está rodando
@app.get("/")
async def root():
    return {"message": "API de processamento de PDFs rodando. Acesse /docs para testar o Swagger."}

# 🔹 Executa o servidor com Uvicorn ao rodar `main.py`
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8989, reload=True)
