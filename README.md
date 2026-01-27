# Enterprise Multimodal RAG & Monitoring Pipeline

A production-ready **Retrieval-Augmented Generation (RAG)** system designed for UK enterprise environments. This project automates the process of ingesting raw documents (PDFs, images, text), converting them into searchable vectors, and using AI to answer questions with full source transparency.

Built with an **Azure-first** mindset while keeping the core open-source for versatility.

## Features

- **Hybrid Search**: Combines semantic (BGE-M3) and keyword (BM25) search with Reciprocal Rank Fusion
- **Multimodal Support**: Search images using text queries via CLIP embeddings
- **Verification Loop**: AI validates its answers against retrieved context before responding
- **Source Transparency**: Every answer includes citations with page numbers and relevance scores
- **MLOps Observability**: Full audit trails via MLflow for enterprise compliance
- **Azure Ready**: Deployable to AKS with one command

---

## 🚀 How to Run Locally

### 1. Prerequisites

| Requirement | Details |
|-------------|---------|
| **Docker & Docker Compose** | v2.0+ installed and running |
| **Ollama** | Download from [ollama.com](https://ollama.com/) |
| **Python** | 3.11+ (for running scripts outside Docker) |
| **RAM** | 16GB minimum recommended |

After installing Ollama, pull the Llama 3 model:

```bash
ollama pull llama3.1:8b
```

### 2. Initial Setup

Run the setup script to prepare environment variables and folders:

```bash
./scripts/setup.sh
```

This will:
- Create `.env` from `.env.example`
- Set up required directories
- Optionally start Docker services

### 3. Launch the Stack

Start all services in one command:

```bash
docker compose up -d
```

### 4. Initialize the Vector Database

Before the first run, create the Qdrant collections:

```bash
# Install dependencies first (if running outside Docker)
pip install -r requirements.txt

# Initialize and verify
python scripts/initialize_qdrant.py --verify
```

### 5. Access the Services

| Service | URL | Credentials |
|---------|-----|-------------|
| **Streamlit UI** | http://localhost:8501 | - |
| **API Docs** | http://localhost:8000/docs | - |
| **Airflow** | http://localhost:8080 | admin / admin |
| **MinIO** | http://localhost:9001 | minioadmin / minioadmin |
| **Qdrant** | http://localhost:6333 | - |
| **MLflow** | http://localhost:5000 | - |

---

## 🧪 Testing the Pipeline (Step-by-Step)

### Step 1: Ingest Data

1. Open **MinIO UI** at `http://localhost:9001`
2. Login with `minioadmin / minioadmin`
3. Upload a PDF or text file to the `raw-documents` bucket
4. Open **Airflow** at `http://localhost:8080`
5. Enable and trigger the `enterprise_rag_ingestion` DAG
6. The `enterprise_rag_embedding` DAG will run automatically after ingestion completes

### Step 2: Test Search from Terminal

Verify the system understands your documents:

```bash
python src/retriever.py "What is the main topic of this document?"
```

Expected output:
```
Searching: 'What is the main topic of this document?'
Mode: hybrid, Top-K: 5
==================================================

[1] Score: 0.0234
    Text: The document discusses quarterly revenue...
    RRF: 0.0234, Dense: 0.89, Sparse: 0.76
```

### Step 3: Use the AI Interface

1. Open **Streamlit UI** at `http://localhost:8501`
2. Type a question in the chatbox
3. **What to verify:**
   - ✅ **Citations**: Does it show which pages were used?
   - ✅ **Scores**: Check the hybrid score breakdown (semantic vs keyword)
   - ✅ **Verification**: Ask something NOT in the document - it should say "Information not found"
   - ✅ **Feedback**: Test the thumbs up/down buttons

### Step 4: Check MLflow Tracking

1. Open **MLflow** at `http://localhost:5000`
2. View the `enterprise-rag-api` experiment
3. Each query is logged with:
   - Search time, generation time, total time
   - Faithfulness score
   - Number of citations used

---

## 📖 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface                           │
│                    (Streamlit @ :8501)                          │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                         FastAPI Backend                         │
│                         (API @ :8000)                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   Search    │  │  RAG Query  │  │   Document Upload       │  │
│  │  /search    │  │   /query    │  │      /upload            │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
          │                  │                      │
          ▼                  ▼                      ▼
┌──────────────┐    ┌──────────────┐       ┌──────────────┐
│   Qdrant     │    │   Ollama     │       │    MinIO     │
│  (Vectors)   │    │   (LLM)      │       │  (Storage)   │
│    :6333     │    │   :11434     │       │    :9000     │
└──────────────┘    └──────────────┘       └──────────────┘
                                                  │
                                                  ▼
                                          ┌──────────────┐
                                          │   Airflow    │
                                          │ (Orchestration)
                                          │    :8080     │
                                          └──────────────┘
```

### Data Flow

1. **Storage (MinIO)** → Raw documents land here
2. **Orchestration (Airflow)** → Triggers chunking and embedding pipelines
3. **Vector DB (Qdrant)** → Stores dense + sparse vectors for hybrid search
4. **API (FastAPI)** → Bridges retrieval and LLM generation
5. **Observability (MLflow)** → Tracks all queries for audit compliance

---

## 🔑 Key AI Features

### Semantic Chunking
Unlike simple character-based splitters, this system:
- Detects natural boundaries (headers, paragraphs, sentences)
- Preserves contextual integrity for better retrieval
- Enriches each chunk with metadata (importance score, semantic type)

### Hybrid Search with RRF
Combines two search strategies:
- **Dense Search (BGE-M3)**: Understands meaning and context
- **Sparse Search (BM25)**: Finds exact keyword matches

**Reciprocal Rank Fusion** merges results to get the best of both worlds.

### Verification Loop
Before showing an answer, the AI:
1. Generates a response based on retrieved context
2. Reviews its own answer against the sources
3. Assigns a **faithfulness score** (0.0 - 1.0)
4. Flags answers that may not be fully grounded

---

## 🏥 Health Check

Quick command to verify all services are running:

```bash
docker compose ps
```

All containers should show `Up (healthy)`.

For detailed health status:
```bash
curl http://localhost:8000/health | jq
```

Expected response:
```json
{
  "status": "healthy",
  "services": {
    "api": "healthy",
    "qdrant": "healthy",
    "minio": "healthy",
    "llm": "healthy",
    "mlflow": "healthy"
  }
}
```

---

## ☁️ Azure Deployment

Deploy to Azure Kubernetes Service with one command:

```bash
cd deploy/azure
./deploy.sh --resource-group rg-rag-pipeline --location uksouth
```

This creates:
- Azure Container Registry (ACR)
- Azure Kubernetes Service (AKS) with 3 nodes
- Azure Blob Storage
- All necessary secrets and config maps

See [deploy/azure/README.md](deploy/azure/README.md) for detailed options.

---

## 📁 Project Structure

```
.
├── docker-compose.yml          # Full local stack
├── requirements.txt            # Python dependencies
├── .env.example               # Configuration template
│
├── src/
│   ├── chunker.py             # Semantic text chunking
│   ├── document_processor.py  # PDF/DOCX extraction
│   ├── embedding_engine.py    # BGE-M3 + CLIP embeddings
│   ├── retriever.py           # Hybrid search with RRF
│   ├── llm_service.py         # LLM backends + verification
│   └── api/main.py            # FastAPI backend
│
├── dags/
│   ├── ingestion_dag.py       # Document processing DAG
│   └── embedding_dag.py       # Vector embedding DAG
│
├── ui/
│   └── app.py                 # Streamlit interface
│
├── scripts/
│   ├── setup.sh               # Local setup script
│   └── initialize_qdrant.py   # Vector DB initialization
│
└── deploy/
    ├── azure/deploy.sh        # Azure deployment
    └── k8s/                   # Kubernetes manifests
```

---

## 🛠️ Configuration

Key environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | ollama | LLM backend (ollama/openai/azure_openai) |
| `LLM_MODEL` | llama3.1:8b | Model to use |
| `EMBEDDING_MODEL` | BAAI/bge-m3 | Embedding model |
| `QDRANT_COLLECTION` | enterprise_docs | Default collection |
| `CHUNK_SIZE` | 400 | Tokens per chunk |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest tests/`
5. Submit a pull request

---

## 📚 Resources

- [Building Production RAG Pipelines (Video)](https://www.youtube.com/watch?v=ciqWMIf7Pz0)
- [BGE-M3 Paper](https://arxiv.org/abs/2402.03216)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

**Built for the UK AI Engineer job market 2026** 🇬🇧
