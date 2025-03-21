# PDF Processing API - Extração Inteligente de Dados Imobiliários

## Visão Geral
Este projeto oferece uma API para extração automatizada de dados imobiliários a partir de documentos PDF. Desenvolvido com **FastAPI**, **LangChain** e **OpenAI**, o sistema é capaz de interpretar diversos formatos de arquivos (PDFs com texto, imagens ou ambos), permitindo alta flexibilidade e precisão na extração dos dados. A interface é documentada com Swagger, facilitando a integração com sistemas já existentes e frontend.

## Como Rodar o Projeto

### 1. Requisitos
- Python 3.10+
- pip
- Virtualenv (recomendado)
- Chave de API válida da OpenAI
  
### 2. Instalação do Ambiente

```bash
# Crie e ative o ambiente virtual
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate    # Windows

# Instale as dependências
pip install -r requirements.txt
```

### 3. Configuração do Ambiente

Crie um arquivo `.env` na raiz do projeto com o seguinte conteúdo:

```env
OPENAI_API_KEY=sua-chave-do-openai-aqui
```

Essa chave é obrigatória para permitir o uso dos modelos GPT via LangChain.

### 4. Execução do requirements.txt

No nível da raiz do projeto plique no seu terminal a função

```bash
pip install -r requirements.txt
```
O processo irá realizar a instalação das dependências presentes no projeto

### 5. Execução do Servidor FastAPI

```bash
uvicorn main:app --host 0.0.0.0 --port 8989 --reload
```

Acesse a documentação Swagger em: [http://localhost:8989/docs](http://localhost:8989/docs)

## Uso da API

### Endpoint: `POST /api/process/`
Este endpoint processa todos os arquivos PDF localizados em uma pasta especificada e gera um arquivo `.csv` consolidado com os dados extraídos.

#### Parâmetros (form-data):
- `pdf_path`: caminho absoluto para a pasta onde estão os arquivos PDF (pode conter 1 ou múltiplos arquivos).
- `output_csv_path`: caminho absoluto para a pasta onde o arquivo CSV resultante será salvo.

O processo funciona lendo os arquivos PDF a partir da pasta informada e, ao final, salvando o CSV diretamente no local indicado. O Swagger permite testar isso facilmente e pode ser usado também 
por sistemas externos para integração com frontends personalizados.

### Finalidade do Swagger
O processo identifica cada PDF individualmente e o classifica pelo tipo, para assiim direciona-lo para o modelo de extração de dados do PDF mais adequado.

### Finalidade do Swagger
A interface via Swagger foi escolhida para facilitar testes, integração com outras APIs e facilitar o uso por sistemas frontend, permitindo que usuários selecionem arquivos e caminhos diretamente.

## Estrutura do Projeto
- `process.py`: núcleo de detecção de tipo do PDF e orquestração da extração.
- `workers/`: implementações específicas de cada tipo de processamento:
  - `worker_pdfplumber.py`: para PDFs com tabelas textuais.
  - `worker_image_preprocess.py`: para PDFs com imagens (OCR).
  - `worker_pdf_mix.py`: para PDFs com múltiplos formatos.
- `api/views.py`: view principal com endpoint de upload e processamento.
- `api/urls.py`: roteador de endpoints.
- `main.py`: inicialização da aplicação FastAPI.

## Decisões Técnicas

### Uso da OpenAI com LangChain
A escolha pela OpenAI (modelo GPT-4-turbo) foi motivada pela:
- Eficiência na interpretação de linguagem natural.
- Redução de tempo de desenvolvimento.
- Capacidade de lidar com instruções complexas e variados layouts de documentos.

Modelos locais como **LLaMA** ou **Ollama** foram descartados devido à curva de configuração mais complexa, necessidade de recursos computacionais maiores e o prazo de entrega curto do projeto.

### Alternativas Consideradas

Inicialmente foi considerada a utilização do **Amazon Textract**, uma solução rápida porém de custo elevado para o volume e variação dos documentos envolvidos. Em seguida, foi cogitada a criação de **regex customizadas**, mas a alta diversidade de layouts inviabilizou essa abordagem dentro do prazo estimado.

### Arquitetura Modular com 3 Fluxos
O sistema é capaz de identificar e tratar três principais categorias de PDFs:
1. **TABELA**: documentos com tabelas estruturadas em texto.
2. **IMAGEM**: documentos escaneados com necessidade de OCR.
3. **MIX**: documentos com uma combinação de texto e imagem.

Caso a detecção automatizada não seja conclusiva, a IA é acionada para classificar corretamente, melhorando a assertividade do processo.

### OCR para PDFs Escaneados
Documentos que continham apenas imagens apresentaram desafio inicial. A solução foi implementar OCR (via `pdf2image` + `pytesseract`) para converter imagens em texto. Esse texto é tratado posteriormente pela IA, reduzindo o número de tokens necessários e melhorando o desempenho.

### Principais Desafios e Soluções
- **Diversidade de Layouts de PDF**: resolvido com detecção inteligente e fluxo adaptativo a depender do formato do PDF.
- **Informações desalinhadas ou incompletas**: resolvido com uso de extração de dados via OCR, pdfplumber e processamento via LangChain com validação de JSON.
- **Diminuição de contexto (tokens) da OpenAI**: mitigado com resumos ou divisão do conteúdo quando necessário.
- **Interpretação de imagens e textos**: extração facilitada via OCR integrado juntamente a pdfplumber para extração dos textos.
- **Integração facilitada em outros contextos**: Swagger documentado e suporte via FastAPI.

---


