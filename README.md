# Enterprise RAG Pipeline

A document question-answering system that lets you upload PDFs and ask questions about them. Uses vector search to find relevant passages and an LLM to generate answers with citations.

## What it does

1. You upload a document (PDF, DOCX, or TXT)
2. The system extracts text, splits it into chunks, and creates embeddings
3. When you ask a question, it searches for relevant chunks using hybrid search (semantic + keyword)
4. An LLM reads those chunks and writes an answer, citing which pages it used

## Tech stack

- **FastAPI** - REST API backend
- **Qdrant** - Vector database for storing embeddings
- **Ollama** - Runs the LLM locally (llama3.2:1b)
- **Sentence Transformers** - BGE-small-en for embeddings (384 dimensions)
- **Streamlit** - Web interface
- **MinIO** - S3-compatible storage for uploaded files
- **Airflow** - Pipeline orchestration (for batch processing)
- **MLflow** - Tracks queries and metrics
- **PostgreSQL** - Metadata storage

## Requirements

- Docker and Docker Compose v2.0+
- 8GB RAM minimum (16GB recommended)

## Setup

```bash
# Start all services
docker compose up -d

# Wait about 2 minutes for everything to initialize

# Pull the LLM model (run once)
docker exec -it $(docker ps -qf "name=ollama") ollama pull llama3.2:1b
```

## Services

| Service | URL | Login |
|---------|-----|-------|
| Web UI | http://localhost:8501 | - |
| API docs | http://localhost:8000/docs | - |
| Airflow | http://localhost:8080 | admin / admin |
| MinIO | http://localhost:9001 | minioadmin / minioadmin |
| Qdrant | http://localhost:6333/dashboard | - |
| MLflow | http://localhost:5050 | - |

## How to use

### Through the web UI

1. Go to http://localhost:8501
2. Click the **Documents** tab
3. Upload a file and click **Upload & Process**
4. Wait for processing to complete (you'll see "X chunks indexed")
5. Go to **Chat** tab and ask a question

### Through the API

```bash
# Check if API is running
curl http://localhost:8000/health

# Upload a document
curl -X POST http://localhost:8000/upload \
  -F "file=@your-document.pdf"

# Process it (use the file_id from upload response)
curl -X POST http://localhost:8000/process/YOUR_FILE_ID

# Search without LLM (fast)
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "what is the main topic?", "top_k": 5}'

# Ask a question with LLM answer (slow on CPU)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "what is the main topic?", "top_k": 5}'
```

## Project structure

```
├── src/
│   ├── api/main.py         # FastAPI endpoints
│   ├── chunker.py          # Splits text into chunks
│   ├── embedding_engine.py # Generates embeddings
│   ├── retriever.py        # Hybrid search logic
│   └── llm_service.py      # Ollama integration
│
├── ui/app.py               # Streamlit frontend
├── dags/                   # Airflow pipelines
├── docker-compose.yml      # Local development stack
├── Dockerfile.api          # API container
└── Dockerfile.streamlit    # UI container
```

## How it works

### Document processing

1. PDF text extraction using PyMuPDF
2. Text split into ~500 character chunks with overlap
3. Each chunk embedded using BGE-small-en (384-dim vectors)
4. Vectors stored in Qdrant with metadata (source file, page number)

### Search

Uses hybrid search combining:
- **Dense search**: Semantic similarity using embeddings
- **Sparse search**: Keyword matching (BM25-style)
- Results merged using Reciprocal Rank Fusion (RRF)

### Answer generation

1. Search returns top-k relevant chunks
2. Chunks sent to LLM as context
3. LLM generates answer based only on provided context
4. Response includes citations pointing to source chunks

## Known limitations

- LLM runs on CPU inside Docker, so responses take 1-2 minutes
- If LLM times out, the UI shows search results instead
- Only tested with English documents
- Large PDFs (100+ pages) may take a while to process

## Troubleshooting

**API not responding:**
```bash
docker compose logs rag-api --tail=50
```

**No search results after upload:**
- Make sure you clicked "Upload & Process", not just "Upload"
- Check if Qdrant has data: `curl http://localhost:6333/collections/enterprise_docs`

**LLM keeps timing out:**
- This is expected on CPU. The search still works, just without the generated answer.
- For faster responses, run Ollama natively on your machine with GPU support.

**Out of memory:**
- Reduce Docker memory usage by stopping unused services
- Or increase Docker's memory limit in Docker Desktop settings

## License

MIT
