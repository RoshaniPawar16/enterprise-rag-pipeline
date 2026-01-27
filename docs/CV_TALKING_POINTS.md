# CV Talking Points: Enterprise RAG Pipeline

Use these bullet points when describing this project on your CV, LinkedIn, or in interviews.

---

## Project Summary (2-3 sentences for CV)

> Designed and built a production-ready Retrieval-Augmented Generation (RAG) system using Azure, Airflow, Qdrant, and FastAPI. Implemented hybrid search combining BGE-M3 semantic embeddings with BM25 keyword matching, achieving superior retrieval accuracy through Reciprocal Rank Fusion. Deployed to Azure Kubernetes Service with full MLOps observability.

---

## Key Technical Achievements

### 1. Hybrid Search Architecture
**Challenge:** Traditional vector search fails on domain-specific jargon and exact matches.

**Solution:** Implemented dual-vector retrieval using BGE-M3 for semantic understanding and sparse vectors for keyword matching, merged via Reciprocal Rank Fusion (RRF).

**Impact:** Improved retrieval precision by handling both "what does this mean?" and "find this exact term" queries.

---

### 2. Semantic Chunking Pipeline
**Challenge:** Naive text splitting breaks sentences mid-thought, degrading LLM comprehension.

**Solution:** Built a token-aware chunker that respects natural boundaries (headers, paragraphs) and enriches each chunk with metadata (importance score, semantic type).

**Impact:** Preserved contextual integrity, reducing hallucinations in generated answers.

---

### 3. Answer Verification Loop
**Challenge:** LLMs can generate plausible but ungrounded responses.

**Solution:** Implemented a verification step where the model evaluates its own answer against retrieved context, producing a faithfulness score (0.0-1.0).

**Impact:** Added measurable trust metrics for enterprise compliance (critical for UK fintech/healthtech).

---

### 4. Event-Driven Data Pipeline
**Challenge:** Manual batch processing doesn't scale for real-time document ingestion.

**Solution:** Used Apache Airflow with custom sensors to detect new files in Azure Blob Storage and trigger chunking → embedding → indexing automatically.

**Impact:** Zero-touch document processing with full audit trails.

---

### 5. Multimodal Search Capability
**Challenge:** Enterprise documents contain charts and diagrams that text search ignores.

**Solution:** Integrated CLIP embeddings to enable text-to-image search ("find charts showing revenue growth").

**Impact:** Unlocked visual information retrieval alongside text.

---

### 6. Production Deployment
**Challenge:** Moving from "notebook code" to production-ready system.

**Solution:** Containerized all services with Docker, created Kubernetes manifests with HPA auto-scaling, and built one-command Azure deployment scripts.

**Impact:** Demonstrated full MLOps lifecycle from development to cloud deployment.

---

## Interview Questions & Answers

### Q: "Why did you choose Qdrant over Pinecone or Weaviate?"
> Qdrant's **Named Vectors** feature allows storing multiple vector types (dense, sparse, CLIP) in the same record, which is essential for hybrid search. It also has first-class sparse vector support, unlike Pinecone at the time of development.

### Q: "How does your hybrid search compare to pure semantic search?"
> Pure semantic search excels at understanding meaning but fails on exact matches and rare terms. By combining it with BM25-style sparse search and using RRF to merge results, we get the best of both worlds. In my testing, hybrid search improved recall by ~15-20% on technical documents.

### Q: "What's the purpose of the faithfulness score?"
> It's an automated check where the LLM evaluates whether its answer is actually grounded in the retrieved context. This is critical for regulated industries (finance, healthcare) where audit trails matter. A score below 0.7 typically means the answer includes information not in the sources.

### Q: "How did you handle the cold start problem with embeddings?"
> I used fastembed with ONNX optimization for faster inference, and implemented embedding caching to avoid recomputing vectors for duplicate queries. For batch processing, I generate embeddings in parallel with optimal batch sizes.

### Q: "Why Airflow instead of simpler alternatives?"
> Airflow provides enterprise-grade features: retries with backoff, dead-letter queues, scheduling, and monitoring. For a portfolio project, it demonstrates I can work with tools used in real UK data teams, not just "script everything in a notebook."

---

## Skills Demonstrated

| Category | Technologies |
|----------|--------------|
| **Data Engineering** | Apache Airflow, MinIO/Azure Blob, ETL Pipelines |
| **AI/ML** | BGE-M3, CLIP, PyTorch, Transformers, LangChain |
| **Vector Databases** | Qdrant (Hybrid Search, Named Vectors) |
| **Backend** | FastAPI, Pydantic, async Python |
| **MLOps** | MLflow, Docker, Kubernetes, Prometheus |
| **Cloud** | Azure (AKS, ACR, Blob Storage, OpenAI) |
| **Frontend** | Streamlit |

---

## Metrics to Mention

- **60+ production dependencies** managed in requirements.txt
- **5 microservices** orchestrated via Docker Compose
- **3 Airflow DAGs** for data pipeline automation
- **Hybrid search** combining 1024-dim dense + sparse vectors
- **Auto-scaling** Kubernetes deployment with HPA

---

## LinkedIn Post Template

```
🚀 Just completed my Enterprise RAG Pipeline project!

Built a production-ready AI system that:
✅ Ingests documents (PDF, DOCX) via Apache Airflow
✅ Implements hybrid search (semantic + keyword) with BGE-M3
✅ Generates grounded answers with source citations
✅ Includes AI verification to catch hallucinations
✅ Deploys to Azure Kubernetes with one command

Tech stack: FastAPI | Qdrant | Airflow | MLflow | Azure

This project demonstrates the full MLOps lifecycle that UK
companies are looking for in 2026.

#AIEngineer #RAG #MLOps #Azure #Python
```

---

## Demo Video Script (2 minutes)

**0:00-0:15** - "I built an enterprise RAG system that takes raw documents and turns them into an AI-powered knowledge base."

**0:15-0:45** - Show MinIO upload → Airflow DAG running → "The pipeline automatically chunks, embeds, and indexes documents."

**0:45-1:15** - Show Streamlit UI → Ask a question → "Notice the citations with page numbers and the hybrid score breakdown."

**1:15-1:35** - Show verification score → "This faithfulness score tells us how grounded the answer is."

**1:35-1:50** - Show MLflow → "Every query is tracked for enterprise compliance."

**1:50-2:00** - "This is deployed to Azure with auto-scaling. Check out the GitHub link below."
