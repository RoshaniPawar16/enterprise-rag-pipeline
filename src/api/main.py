"""
Enterprise RAG Pipeline - FastAPI Backend

Production-ready API providing:
- Hybrid search (dense + sparse + multimodal)
- RAG query with LLM integration and verification
- Document upload and processing
- MLflow experiment tracking
- Health monitoring and metrics

Supports multiple LLM backends: Ollama, OpenAI, Azure OpenAI
"""

import os
import sys
import time
import uuid
import hashlib
from contextlib import asynccontextmanager
from datetime import datetime
from io import BytesIO
from typing import Annotated

import structlog
import fitz  # PyMuPDF for PDF extraction
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from qdrant_client.models import PointStruct
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = structlog.get_logger(__name__)

# ============================================
# Configuration
# ============================================

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "enterprise_docs")

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET_RAW = "raw-documents"

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


# ============================================
# Request/Response Models
# ============================================

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    services: dict[str, str]


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results")
    search_type: str = Field(default="hybrid", pattern="^(hybrid|dense|sparse|multimodal)$")
    collection: str = Field(default=COLLECTION_NAME)
    filters: dict | None = Field(default=None, description="Metadata filters")


class SearchResultItem(BaseModel):
    chunk_id: str
    text: str
    score: float
    source_file: str
    page_number: int | None
    semantic_type: str | None
    importance_score: float | None
    dense_score: float | None = None
    sparse_score: float | None = None
    rrf_score: float | None = None


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResultItem]
    search_type: str
    total_results: int
    search_time_ms: float


class RAGRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="Question to answer")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of sources to retrieve")
    model: str = Field(default=LLM_MODEL, description="LLM model to use")
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    collection: str = Field(default=COLLECTION_NAME)
    verify: bool = Field(default=True, description="Enable answer verification")
    stream: bool = Field(default=False, description="Stream the response")


class CitationItem(BaseModel):
    index: int
    source_file: str
    page_number: int | None
    text_snippet: str
    relevance_score: float


class RAGResponse(BaseModel):
    query: str
    answer: str
    citations: list[CitationItem]
    sources: list[SearchResultItem]
    model: str
    provider: str
    faithfulness_score: float | None
    verified: bool
    generation_time_ms: float
    search_time_ms: float
    total_time_ms: float


class DocumentUploadResponse(BaseModel):
    filename: str
    file_id: str
    status: str
    message: str
    size_bytes: int


class FeedbackRequest(BaseModel):
    query_id: str
    feedback: str = Field(..., pattern="^(positive|negative)$")
    comment: str | None = None


class FeedbackResponse(BaseModel):
    status: str
    query_id: str
    feedback: str


class CollectionInfo(BaseModel):
    name: str
    vectors_count: int
    points_count: int
    status: str


class ProcessResponse(BaseModel):
    file_id: str
    filename: str
    status: str
    chunks_created: int
    message: str


class MetricsResponse(BaseModel):
    total_documents: int
    total_chunks: int
    queries_today: int
    avg_response_time_ms: float
    avg_faithfulness_score: float
    collections: list[CollectionInfo]
    llm_provider: str
    embedding_model: str


# ============================================
# Application State
# ============================================

class AppState:
    """Application state container."""
    retriever = None
    rag_service = None
    minio_client = None
    mlflow_client = None
    query_count = 0
    total_response_time = 0.0
    total_faithfulness = 0.0
    faithfulness_count = 0


state = AppState()


# ============================================
# Lifecycle Management
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    logger.info("Starting Enterprise RAG API...")

    # Initialize retriever
    try:
        from retriever import HybridRetriever
        state.retriever = HybridRetriever(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            collection_name=COLLECTION_NAME,
        )
        logger.info("Retriever initialized", collection=COLLECTION_NAME)
    except Exception as e:
        logger.error("Failed to initialize retriever", error=str(e))

    # Initialize RAG service
    try:
        from llm_service import create_rag_service
        state.rag_service = create_rag_service(
            provider=LLM_PROVIDER,
            model=LLM_MODEL,
            enable_verification=True,
        )
        logger.info("RAG service initialized", provider=LLM_PROVIDER, model=LLM_MODEL)
    except Exception as e:
        logger.error("Failed to initialize RAG service", error=str(e))

    # Initialize MinIO client
    try:
        from minio import Minio
        state.minio_client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=False,
        )
        logger.info("MinIO client initialized")
    except Exception as e:
        logger.error("Failed to initialize MinIO", error=str(e))

    # Initialize MLflow
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("enterprise-rag-api")
        state.mlflow_client = mlflow
        logger.info("MLflow initialized", uri=MLFLOW_TRACKING_URI)
    except Exception as e:
        logger.warning("MLflow not available", error=str(e))

    logger.info("Enterprise RAG API started successfully")
    yield

    # Shutdown
    logger.info("Shutting down Enterprise RAG API...")


# ============================================
# FastAPI Application
# ============================================

app = FastAPI(
    title="Enterprise RAG Pipeline API",
    description="""
Production-ready Retrieval-Augmented Generation API with:
- **Hybrid Search**: Combined semantic and keyword search with RRF reranking
- **Multimodal Support**: Search images with text queries via CLIP
- **LLM Integration**: Grounded answers with automatic citations
- **Verification Loop**: AI validates its answers against retrieved context
- **MLflow Tracking**: Full observability for enterprise compliance
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# Health & Metrics Endpoints
# ============================================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API and all dependent services health."""
    services = {"api": "healthy"}

    # Check Qdrant
    try:
        if state.retriever:
            state.retriever.client.get_collections()
            services["qdrant"] = "healthy"
        else:
            services["qdrant"] = "not_initialized"
    except Exception:
        services["qdrant"] = "unhealthy"

    # Check MinIO
    try:
        if state.minio_client:
            state.minio_client.bucket_exists(MINIO_BUCKET_RAW)
            services["minio"] = "healthy"
        else:
            services["minio"] = "not_initialized"
    except Exception:
        services["minio"] = "unhealthy"

    # Check LLM
    services["llm"] = "healthy" if state.rag_service else "not_initialized"

    # Check MLflow
    services["mlflow"] = "healthy" if state.mlflow_client else "not_available"

    overall_status = "healthy" if all(
        s in ("healthy", "not_available") for s in services.values()
    ) else "degraded"

    return HealthResponse(
        status=overall_status,
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0",
        services=services,
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["Health"])
async def get_metrics():
    """Get pipeline metrics and statistics."""
    collections = []

    if state.retriever:
        try:
            collection_list = state.retriever.client.get_collections()
            for col in collection_list.collections:
                info = state.retriever.client.get_collection(col.name)
                # Handle different qdrant client versions
                vectors = getattr(info, 'vectors_count', None) or getattr(info, 'indexed_vectors_count', 0) or 0
                points = getattr(info, 'points_count', 0) or 0
                status_val = info.status.value if hasattr(info.status, 'value') else str(info.status)
                collections.append(CollectionInfo(
                    name=col.name,
                    vectors_count=vectors,
                    points_count=points,
                    status=status_val,
                ))
        except Exception as e:
            logger.warning("Failed to get collection info", error=str(e))

    total_chunks = sum(c.points_count for c in collections)
    avg_response = state.total_response_time / state.query_count if state.query_count > 0 else 0.0
    avg_faithfulness = state.total_faithfulness / state.faithfulness_count if state.faithfulness_count > 0 else 0.0

    return MetricsResponse(
        total_documents=len(collections),
        total_chunks=total_chunks,
        queries_today=state.query_count,
        avg_response_time_ms=avg_response,
        avg_faithfulness_score=avg_faithfulness,
        collections=collections,
        llm_provider=LLM_PROVIDER,
        embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
    )


# ============================================
# Search Endpoints
# ============================================

@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search(request: SearchRequest):
    """
    Execute hybrid search across the document collection.

    Supports:
    - **dense**: Semantic similarity search
    - **sparse**: Keyword/BM25-style search
    - **hybrid**: Combined with Reciprocal Rank Fusion (recommended)
    - **multimodal**: Image search via CLIP embeddings
    """
    if not state.retriever:
        raise HTTPException(status_code=503, detail="Retriever not initialized")

    start_time = time.time()

    try:
        from retriever import SearchMode

        mode_map = {
            "dense": SearchMode.DENSE,
            "sparse": SearchMode.SPARSE,
            "hybrid": SearchMode.HYBRID,
            "multimodal": SearchMode.MULTIMODAL,
        }

        response = state.retriever.search(
            query=request.query,
            top_k=request.top_k,
            mode=mode_map[request.search_type],
            collection=request.collection,
            filters=request.filters,
        )

        results = [
            SearchResultItem(
                chunk_id=r.id,
                text=r.text,
                score=r.score,
                source_file=r.metadata.get("source_file", "unknown"),
                page_number=r.metadata.get("page_number"),
                semantic_type=r.metadata.get("semantic_type"),
                importance_score=r.metadata.get("importance_score"),
                dense_score=r.dense_score,
                sparse_score=r.sparse_score,
                rrf_score=r.rrf_score,
            )
            for r in response.results
        ]

        search_time = (time.time() - start_time) * 1000

        return SearchResponse(
            query=request.query,
            results=results,
            search_type=request.search_type,
            total_results=response.total_results,
            search_time_ms=search_time,
        )

    except Exception as e:
        logger.error("Search failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# RAG Query Endpoints
# ============================================

@app.post("/query", response_model=RAGResponse, tags=["RAG"])
async def rag_query(request: RAGRequest):
    """
    Execute a RAG query: retrieve context and generate a grounded answer.

    The pipeline:
    1. **Retrieve**: Hybrid search for relevant document chunks
    2. **Augment**: Build context-aware prompt with citations
    3. **Generate**: LLM produces grounded answer
    4. **Verify**: (Optional) Validate faithfulness to context

    Returns structured response with citations and confidence scores.
    """
    if not state.retriever or not state.rag_service:
        raise HTTPException(status_code=503, detail="Services not initialized")

    start_time = time.time()
    query_id = str(uuid.uuid4())

    try:
        from retriever import SearchMode

        # 1. Retrieve context
        search_start = time.time()
        search_response = state.retriever.search(
            query=request.query,
            top_k=request.top_k,
            mode=SearchMode.HYBRID,
            collection=request.collection,
        )
        search_time = (time.time() - search_start) * 1000

        # Convert to source format
        sources = [
            {"text": r.text, "score": r.score, "metadata": r.metadata}
            for r in search_response.results
        ]

        # 2. Generate answer
        llm_response = await state.rag_service.agenerate(
            query=request.query,
            sources=sources,
            temperature=request.temperature,
            verify=request.verify,
        )

        total_time = (time.time() - start_time) * 1000

        # Update metrics
        state.query_count += 1
        state.total_response_time += total_time
        if llm_response.faithfulness_score is not None:
            state.total_faithfulness += llm_response.faithfulness_score
            state.faithfulness_count += 1

        # Log to MLflow
        if state.mlflow_client:
            try:
                with state.mlflow_client.start_run(run_name=f"query_{query_id[:8]}"):
                    state.mlflow_client.log_params({
                        "query": request.query[:100],
                        "model": request.model,
                        "top_k": request.top_k,
                    })
                    state.mlflow_client.log_metrics({
                        "search_time_ms": search_time,
                        "generation_time_ms": llm_response.generation_time_ms,
                        "total_time_ms": total_time,
                        "faithfulness_score": llm_response.faithfulness_score or 0,
                        "num_citations": len(llm_response.citations),
                    })
            except Exception as e:
                logger.warning("MLflow logging failed", error=str(e))

        # Build response
        search_results = [
            SearchResultItem(
                chunk_id=r.id,
                text=r.text,
                score=r.score,
                source_file=r.metadata.get("source_file", "unknown"),
                page_number=r.metadata.get("page_number"),
                semantic_type=r.metadata.get("semantic_type"),
                importance_score=r.metadata.get("importance_score"),
                dense_score=r.dense_score,
                sparse_score=r.sparse_score,
                rrf_score=r.rrf_score,
            )
            for r in search_response.results
        ]

        citations = [
            CitationItem(
                index=c.index,
                source_file=c.source_file,
                page_number=c.page_number,
                text_snippet=c.text_snippet,
                relevance_score=c.relevance_score,
            )
            for c in llm_response.citations
        ]

        return RAGResponse(
            query=request.query,
            answer=llm_response.answer,
            citations=citations,
            sources=search_results,
            model=llm_response.model,
            provider=llm_response.provider,
            faithfulness_score=llm_response.faithfulness_score,
            verified=llm_response.verified,
            generation_time_ms=llm_response.generation_time_ms,
            search_time_ms=search_time,
            total_time_ms=total_time,
        )

    except Exception as e:
        logger.error("RAG query failed", query_id=query_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream", tags=["RAG"])
async def rag_query_stream(request: RAGRequest):
    """Stream a RAG response token by token."""
    if not state.retriever or not state.rag_service:
        raise HTTPException(status_code=503, detail="Services not initialized")

    try:
        from retriever import SearchMode

        search_response = state.retriever.search(
            query=request.query,
            top_k=request.top_k,
            mode=SearchMode.HYBRID,
            collection=request.collection,
        )

        sources = [
            {"text": r.text, "score": r.score, "metadata": r.metadata}
            for r in search_response.results
        ]

        async def generate():
            for token in state.rag_service.stream_generate(
                query=request.query,
                sources=sources,
                temperature=request.temperature,
            ):
                yield token

        return StreamingResponse(generate(), media_type="text/plain")

    except Exception as e:
        logger.error("Stream query failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# Document Management Endpoints
# ============================================

@app.post("/upload", response_model=DocumentUploadResponse, tags=["Documents"])
async def upload_document(
    file: Annotated[UploadFile, File(description="Document to upload")],
    background_tasks: BackgroundTasks,
):
    """
    Upload a document for processing.

    Supported formats: PDF, DOCX, TXT, MD
    """
    if not state.minio_client:
        raise HTTPException(status_code=503, detail="Storage not initialized")

    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    allowed_extensions = {".pdf", ".docx", ".txt", ".md"}
    ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""

    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}",
        )

    try:
        file_id = str(uuid.uuid4())
        object_name = f"{file_id}_{file.filename}"

        content = await file.read()
        size = len(content)

        state.minio_client.put_object(
            MINIO_BUCKET_RAW,
            object_name,
            BytesIO(content),
            size,
            content_type=file.content_type or "application/octet-stream",
        )

        logger.info("Document uploaded", file_id=file_id, filename=file.filename, size=size)

        return DocumentUploadResponse(
            filename=file.filename,
            file_id=file_id,
            status="uploaded",
            message="Document uploaded successfully. Processing will begin shortly.",
            size_bytes=size,
        )

    except Exception as e:
        logger.error("Upload failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process/{file_id}", response_model=ProcessResponse, tags=["Documents"])
async def process_document(file_id: str):
    """
    Process an uploaded document: extract text, chunk, embed, and index.

    This endpoint takes a file_id from the upload response and processes the document
    into searchable vector embeddings.
    """
    if not state.minio_client or not state.retriever:
        raise HTTPException(status_code=503, detail="Services not initialized")

    try:
        # Find the file in MinIO by file_id prefix
        objects = list(state.minio_client.list_objects(MINIO_BUCKET_RAW, prefix=file_id))
        if not objects:
            raise HTTPException(status_code=404, detail=f"File not found: {file_id}")

        obj = objects[0]
        object_name = obj.object_name
        filename = object_name.split("_", 1)[1] if "_" in object_name else object_name

        logger.info("Processing document", file_id=file_id, filename=filename)

        # Download file from MinIO
        response = state.minio_client.get_object(MINIO_BUCKET_RAW, object_name)
        file_content = response.read()
        response.close()
        response.release_conn()

        # Extract text based on file type
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

        if ext == "pdf":
            # Extract text from PDF
            doc = fitz.open(stream=file_content, filetype="pdf")
            pages_text = []
            for page_num, page in enumerate(doc, 1):
                text = page.get_text()
                if text.strip():
                    pages_text.append({"page": page_num, "text": text})
            doc.close()
        elif ext in ("txt", "md"):
            pages_text = [{"page": 1, "text": file_content.decode("utf-8")}]
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

        if not pages_text:
            raise HTTPException(status_code=400, detail="No text extracted from document")

        # Simple chunking - split by paragraphs with overlap
        chunks = []
        chunk_size = 500  # characters
        overlap = 100

        for page_data in pages_text:
            page_num = page_data["page"]
            text = page_data["text"]

            # Split into paragraphs first
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

            current_chunk = ""
            for para in paragraphs:
                if len(current_chunk) + len(para) < chunk_size:
                    current_chunk += para + "\n\n"
                else:
                    if current_chunk:
                        chunks.append({
                            "text": current_chunk.strip(),
                            "page": page_num,
                            "source": filename,
                        })
                    # Start new chunk with overlap
                    if len(current_chunk) > overlap:
                        current_chunk = current_chunk[-overlap:] + para + "\n\n"
                    else:
                        current_chunk = para + "\n\n"

            # Add remaining text
            if current_chunk.strip():
                chunks.append({
                    "text": current_chunk.strip(),
                    "page": page_num,
                    "source": filename,
                })

        if not chunks:
            raise HTTPException(status_code=400, detail="No chunks created from document")

        logger.info("Chunks created", count=len(chunks))

        # Generate embeddings for all chunks
        texts = [c["text"] for c in chunks]
        embeddings = list(state.retriever.embedder.embed_batch(texts))

        # Prepare points for Qdrant
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = hashlib.md5(f"{file_id}_{i}_{chunk['text'][:100]}".encode()).hexdigest()

            point = PointStruct(
                id=chunk_id,
                vector={"dense": embedding.dense.to_list()},
                payload={
                    "text": chunk["text"],
                    "source_file": chunk["source"],
                    "page_number": chunk["page"],
                    "file_id": file_id,
                    "chunk_index": i,
                    "semantic_type": "text",
                    "importance_score": 0.5,
                },
            )
            points.append(point)

        # Upsert to Qdrant
        state.retriever.client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
        )

        logger.info("Document indexed", file_id=file_id, chunks=len(points))

        return ProcessResponse(
            file_id=file_id,
            filename=filename,
            status="processed",
            chunks_created=len(points),
            message=f"Successfully processed {filename}: {len(points)} chunks indexed",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Processing failed", file_id=file_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/collections", tags=["Documents"])
async def list_collections():
    """List all available vector collections."""
    if not state.retriever:
        raise HTTPException(status_code=503, detail="Retriever not initialized")

    try:
        collections = state.retriever.client.get_collections()
        return {"collections": [{"name": col.name} for col in collections.collections]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/collections/{name}", response_model=CollectionInfo, tags=["Documents"])
async def get_collection(name: str):
    """Get information about a specific collection."""
    if not state.retriever:
        raise HTTPException(status_code=503, detail="Retriever not initialized")

    try:
        info = state.retriever.client.get_collection(name)
        # Handle different qdrant client versions
        vectors = getattr(info, 'vectors_count', None) or getattr(info, 'indexed_vectors_count', 0) or 0
        points = getattr(info, 'points_count', 0) or 0
        status_val = info.status.value if hasattr(info.status, 'value') else str(info.status)
        return CollectionInfo(
            name=name,
            vectors_count=vectors,
            points_count=points,
            status=status_val,
        )
    except Exception as e:
        logger.warning(f"Collection lookup failed: {e}")
        raise HTTPException(status_code=404, detail=f"Collection not found: {name}")


# ============================================
# Feedback Endpoints
# ============================================

@app.post("/feedback", response_model=FeedbackResponse, tags=["Feedback"])
async def submit_feedback(request: FeedbackRequest):
    """Submit feedback for a query response (logged to MLflow)."""
    logger.info("Feedback received", query_id=request.query_id, feedback=request.feedback)

    if state.mlflow_client:
        try:
            with state.mlflow_client.start_run(run_name=f"feedback_{request.query_id[:8]}"):
                state.mlflow_client.log_params({
                    "query_id": request.query_id,
                    "feedback": request.feedback,
                    "comment": request.comment or "",
                })
                state.mlflow_client.log_metrics({
                    "positive_feedback": 1 if request.feedback == "positive" else 0,
                    "negative_feedback": 1 if request.feedback == "negative" else 0,
                })
        except Exception as e:
            logger.warning("MLflow feedback logging failed", error=str(e))

    return FeedbackResponse(
        status="recorded",
        query_id=request.query_id,
        feedback=request.feedback,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, workers=1)
