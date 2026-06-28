# Sales Sequence Intelligence Copilot

A production-ready RAG (Retrieval-Augmented Generation) application designed to analyze B2B sales sequence metrics, diagnose performance drop-offs, rewrite poorly performing email steps, and answer factual queries with ground-truth citations.

---

## Key Features
- **Strict Single-Document Lifecycle**: Enforces uploading and analyzing one CSV or PDF document at a time. The database is cleared when changing files to prevent data mixing (crucial for free tier storage safety).
- **Hybrid Search & Reranking**: Combines semantic embeddings (Sentence-Transformers `all-MiniLM-L6-v2`) and keyword matching (BM25) with a Cross-Encoder Reranker (`ms-marco-MiniLM-L-6-v2`) for premium retrieval precision.
- **Aggregation-Aware Prompting**: Automatically detects query keywords like "highest", "lowest", "across all", or "compare" to construct a global database summary table alongside top-K candidate chunks so that the LLM has complete context.
- **Dual Query Output Engine**:
  - **Factual**: Renders direct, clean Markdown responses for questions about rates, subjects, or target audience details.
  - **Diagnostic**: Renders rich analytical components outlining Step Drop-offs, Root Causes, Actionable Fixes, and Copy Improvements.
- **Prompt Injection Defense**: Passive context chunk isolation and input blacklists block malicious overrides at the API entry point.

---

## Project Directory Structure

```
├── main.py                   # FastAPI service entrypoint & Lifespan checks
├── config/
│   ├── config.yaml           # App configuration (retrieval k, embeddings, type)
│   └── config.py             # Pydantic Configuration Model & Environment Overrides
├── api/
│   ├── routes.py             # FastAPI REST endpoints & Input validation
│   └── security.py           # Prompt injection detection filters
├── core/
│   ├── adapters/
│   │   ├── base.py           # Unified SalesEmail/Chunk data models
│   │   ├── csv_adapter.py    # CSV Parsing with fuzzy header mapping
│   │   └── pdf_adapter.py    # PDF Extraction supporting inline & newline layouts
│   ├── chunking/             # Sentence, sliding-window & semantic chunking strategies
│   ├── vectorstore/
│   │   ├── base.py           # Abstract database interface
│   │   ├── local_store.py    # Local file-based JSON vector store
│   │   ├── qdrant_store.py   # Qdrant Cloud client integrations
│   │   └── azure_store.py    # Azure AI Search vector store integration
│   ├── semantic_retriever.py # Vector distance similarity retriever
│   ├── bm25_retriever.py     # Okapi BM25 keyword search retriever
│   ├── hybrid_retriever.py   # Reciprocal Rank Fusion (RRF) combiner
│   └── orchestrator.py       # RAG flow coordinator (Retrieve -> Augment -> Generate)
├── reranker/                 # Cross-encoder rank optimizer
├── embeddings/               # Factory loader for local/cloud embedding models
├── static/                   # Front-end dashboard HTML, CSS, and JS (glassmorphic UI)
└── verify_api.py             # Core API integration & regression tests
```

---

## Local Setup & Run

### 1. Requirements
- Python 3.11+
- Virtual environment manager (optional but recommended)

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a `.env` file in the project root (see `.env.example` for details):
```env
# System Settings
LOG_LEVEL=INFO

# Vector Store Settings ('local', 'qdrant', or 'azure')
VECTORSTORE_TYPE=local

# Azure AI Search (Required if VECTORSTORE_TYPE is 'azure')
AZURE_AI_SEARCH_URL=https://<your-service-name>.search.windows.net
AZURE_AI_SEARCH_KEY=<your-primary-admin-key>

# Gemini API Key (Required for live LLM responses)
GEMINI_API_KEY=your_gemini_api_key_here
LLM_PROVIDER=gemini
LLM_MODEL_NAME=gemini-2.5-flash
```

### 4. Run the API and Dashboard Server
```bash
py main.py
```
Open your browser and navigate to `http://localhost:8000` to access the glassmorphic dashboard!

### 5. Run Integration Tests
```bash
py verify_api.py
```

---

## Azure AI Search Integration

When setting `VECTORSTORE_TYPE=azure`, the application utilizes **Azure AI Search** to manage indexes, vector collections, and hybrid queries:
- **Vector Space Dimension**: Aligned to `384` to match the local `all-MiniLM-L6-v2` transformer.
- **Index Lifecycle**: Implements a delete-and-recreate `clear()` method, meaning removing a document from the UI completely resets index storage to `0 KB` (highly recommended for the 50MB free tier).
- **HNSW Profiler**: Configured to build vector graphs automatically using Hierarchical Navigable Small World algorithm configurations (`my-hnsw-config`).

---

## Deployment to Azure App Service (Step-by-Step)

FastAPI runs on **Azure Web App for Linux** (Python 3.11 stack). Here is how to deploy it using the Azure CLI:

### 1. Login and Set Context
Open your terminal, login, and verify your subscription:
```bash
az login
az account show
```

### 2. Create Azure Resources
If you don't have existing resources, create a Resource Group and an App Service Plan:
```bash
# Create Resource Group
az group create --name sales-copilot-rg --location eastus

# Create App Service Plan (B1 basic plan for Linux)
az appservice plan create --name sales-copilot-plan --resource-group sales-copilot-rg --sku B1 --is-linux
```

### 3. Create Web App
Create the Python 3.11 web app:
```bash
az webapp create --name <your-app-name> --resource-group sales-copilot-rg --plan sales-copilot-plan --runtime "PYTHON|3.11"
```

### 4. Configure Application Settings (Environment Variables)
Set your environment variables so the app reads them in production via `os.environ`:
```bash
az webapp config appsettings set --name <your-app-name> --resource-group sales-copilot-rg --settings \
    VECTORSTORE_TYPE="azure" \
    AZURE_AI_SEARCH_URL="https://<your-service-name>.search.windows.net" \
    AZURE_AI_SEARCH_KEY="<your-primary-admin-key>" \
    GEMINI_API_KEY="<your-gemini-api-key>" \
    LLM_PROVIDER="gemini" \
    LLM_MODEL_NAME="gemini-2.5-flash" \
    EMBEDDINGS_PROVIDER="sentence-transformers" \
    EMBEDDINGS_DIMENSION="384" \
    RERANKER_ENABLED="true" \
    RERANKER_TOP_N="10"
```

### 5. Set Startup Command
FastAPI requires Uvicorn to run. Configure the startup command on Azure:
```bash
az webapp config set --name <your-app-name> --resource-group sales-copilot-rg --startup-file "python -m uvicorn main:app --host 0.0.0.0 --port 8000"
```

### 6. Deploy Code
Deploy the codebase using the local git deployment method or zip deployment:

**Local Git Deployment Method:**
```bash
# Get deployment credentials
az webapp deployment user set --user-name <deploy-username> --password <deploy-password>

# Enable local Git repository on Web App
az webapp deployment source config-local-git --name <your-app-name> --resource-group sales-copilot-rg --query url --output tsv
# Copy the returned URL (looks like https://<username>@<app>.scm.azurewebsites.net/<app>.git)

# In your local project repository root:
git init
git add .
git commit -m "Initialize Sales Intelligence Copilot deployment"
git remote add azure <URL-you-copied-above>
git push azure master
```

---

## Security Practices
- **Passive Context Construction**: Chunk inputs are formatted inside the prompt with markers separating data from instructions. The model is specifically instructed never to treat context chunk text as instructions.
- **Input Blacklisting**: Detects common prompt injection jailbreak keywords (e.g. `ignore previous`, `system prompt`, `override instructions`) and rejects the request with a `400 Bad Request` before calling embeddings or LLM APIs.
