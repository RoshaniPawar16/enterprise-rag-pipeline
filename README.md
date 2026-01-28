# Enterprise RAG Pipeline

A document question-answering system that lets you upload files and ask questions about them. Uses vector search to find relevant passages and an LLM to generate answers with citations.

![Demo Screenshot](docs/screenshots/demo.png)
*Screenshot: Upload a document, ask questions, get answers with sources*

## What it does

1. Upload a document (PDF, DOCX, or TXT)
2. System extracts text, splits into chunks, creates embeddings
3. Ask a question → finds relevant chunks using hybrid search
4. LLM reads chunks and writes an answer with citations

## Quick Start

### 1. Clone and configure

```bash
git clone https://github.com/YOUR_USERNAME/enterprise-rag-pipeline.git
cd enterprise-rag-pipeline

# Copy environment file
cp .env.example .env
```

### 2. Add your OpenAI key (recommended)

Edit `.env` and add your OpenAI API key:
```
OPENAI_API_KEY=sk-your-key-here
```

This gives you fast responses. Without it, you can use local Ollama but it's slow on CPU.

### 3. Start the services

```bash
docker compose up -d
```

Wait about 1 minute for everything to start.

### 4. Open the UI

Go to **http://localhost:8501**

### 5. Try it out

1. Go to **Documents** tab
2. Upload the sample file: `sample_data/company_handbook.txt`
3. Click **Upload & Process**
4. Go to **Chat** tab
5. Ask: "What is the annual leave policy?"

## Sample Questions to Try

Using the included `sample_data/company_handbook.txt`:

- "What is the annual leave policy?"
- "How does the pension scheme work?"
- "What are the password requirements?"
- "Where are the offices located?"

## Screenshots

| Upload Documents | Ask Questions |
|------------------|---------------|
| ![Upload](docs/screenshots/upload.png) | ![Chat](docs/screenshots/chat.png) |

## Architecture

```
┌─────────────────────────────────────────────┐
│           Streamlit UI (:8501)              │
└─────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│           FastAPI Backend (:8000)           │
│     /upload  /process  /search  /query      │
└─────────────────────────────────────────────┘
        │              │              │
        ▼              ▼              ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│   Qdrant    │ │   OpenAI    │ │    MinIO    │
│  (vectors)  │ │   (LLM)     │ │  (storage)  │
└─────────────┘ └─────────────┘ └─────────────┘
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Vector DB | Qdrant |
| Embeddings | sentence-transformers (BGE-small) |
| LLM | OpenAI GPT-3.5 / GPT-4 (or Ollama) |
| Backend | FastAPI |
| Frontend | Streamlit |
| Storage | MinIO |

## Project Structure

```
├── src/
│   ├── api/main.py         # API endpoints
│   ├── chunker.py          # Text chunking
│   ├── embedding_engine.py # Embeddings
│   ├── retriever.py        # Hybrid search
│   └── llm_service.py      # LLM integration
├── ui/app.py               # Web interface
├── tests/                  # Unit tests
├── sample_data/            # Sample documents
├── docker-compose.yml
└── requirements.txt
```

## Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run tests
pytest tests/ -v

# Skip slow tests (require model download)
pytest tests/ -v -m "not slow"
```

## Configuration Options

### Using OpenAI (Recommended)

```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-3.5-turbo
OPENAI_API_KEY=sk-your-key
```

### Using Local Ollama (Free but slow)

```env
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2:1b
```

Then start with the Ollama profile:
```bash
docker compose --profile local-llm up -d
docker exec -it $(docker ps -qf "name=ollama") ollama pull llama3.2:1b
```

## API Endpoints

```bash
# Health check
curl http://localhost:8000/health

# Upload document
curl -X POST http://localhost:8000/upload -F "file=@document.pdf"

# Process document (returns file_id from upload)
curl -X POST http://localhost:8000/process/{file_id}

# Search (no LLM, fast)
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "annual leave policy", "top_k": 5}'

# Query with LLM answer
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the annual leave policy?"}'
```

## Troubleshooting

**"Connection refused" errors**
- Wait 1-2 minutes after `docker compose up` for services to initialize

**Slow responses with Ollama**
- Expected on CPU. Use OpenAI for faster responses.

**No search results**
- Make sure you clicked "Upload & Process", not just upload
- Check collection: `curl http://localhost:6333/collections/enterprise_docs`

## License

MIT
