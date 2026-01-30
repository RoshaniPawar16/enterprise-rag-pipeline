"""
Enterprise RAG Pipeline - FastAPI Backend

Production-ready patterns:
- Async/await throughout
- Dependency injection
- Connection pooling with httpx
- Retry logic with tenacity
- Structured error handling
- Pydantic settings for config
"""

import asyncio
import hashlib
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from functools import lru_cache
from io import BytesIO
from typing import Annotated, AsyncIterator

import httpx
import structlog
from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile, File, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from tenacity import retry, stop_after_attempt, wait_exponential

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = structlog.get_logger(__name__)


# ============================================
# Configuration (Pydantic Settings)
# ============================================

class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "enterprise_docs"

    # MinIO
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket_raw: str = "raw-documents"

    # LLM
    llm_provider: str = "openai"
    llm_model: str = "gpt-3.5-turbo"
    openai_api_key: str = ""
    ollama_host: str = "http://localhost:11434"

    # Timeouts
    llm_timeout: int = 60
    search_timeout: int = 30

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance."""
    return Settings()


# ============================================
# Custom Exceptions
# ============================================

class RAGException(Exception):
    """Base exception for RAG pipeline."""
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class ServiceUnavailable(RAGException):
    """Service not ready."""
    def __init__(self, service: str):
        super().__init__(f"{service} not initialized", status_code=503)


class DocumentNotFound(RAGException):
    """Document not found."""
    def __init__(self, file_id: str):
        super().__init__(f"Document not found: {file_id}", status_code=404)


# ============================================
# Request/Response Models
# ============================================

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=50)
    collection: str | None = None


class SearchResult(BaseModel):
    chunk_id: str
    text: str
    score: float
    source_file: str
    page_number: int | None = None


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    total_results: int
    search_time_ms: float


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    collection: str | None = None


class Citation(BaseModel):
    index: int
    source_file: str
    page_number: int | None
    text_snippet: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: list[Citation]
    sources: list[SearchResult]
    model: str
    search_time_ms: float
    generation_time_ms: float
    total_time_ms: float


class UploadResponse(BaseModel):
    file_id: str
    filename: str
    size_bytes: int
    status: str


class ProcessResponse(BaseModel):
    file_id: str
    filename: str
    chunks_created: int
    status: str


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    services: dict[str, str]


# ============================================
# Async Service Clients
# ============================================

class AsyncServiceManager:
    """Manages async connections to external services."""

    def __init__(self):
        self.http_client: httpx.AsyncClient | None = None
        self.retriever = None
        self.rag_service = None
        self.minio_client = None
        self._initialized = False

    async def initialize(self, settings: Settings):
        """Initialize all service connections."""
        if self._initialized:
            return

        # HTTP client with connection pooling
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
        )

        # Initialize retriever
        try:
            from retriever import HybridRetriever
            self.retriever = HybridRetriever(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
                collection_name=settings.qdrant_collection,
            )
            logger.info("Retriever initialized")
        except Exception as e:
            logger.error("Failed to initialize retriever", error=str(e))

        # Initialize RAG service
        try:
            from llm_service import create_rag_service
            self.rag_service = create_rag_service(
                provider=settings.llm_provider,
                model=settings.llm_model,
            )
            logger.info("RAG service initialized", provider=settings.llm_provider)
        except Exception as e:
            logger.warning("RAG service not available", error=str(e))

        # Initialize MinIO
        try:
            from minio import Minio
            self.minio_client = Minio(
                settings.minio_endpoint,
                access_key=settings.minio_access_key,
                secret_key=settings.minio_secret_key,
                secure=False,
            )
            # Ensure bucket exists
            if not self.minio_client.bucket_exists(settings.minio_bucket_raw):
                self.minio_client.make_bucket(settings.minio_bucket_raw)
            logger.info("MinIO initialized")
        except Exception as e:
            logger.error("Failed to initialize MinIO", error=str(e))

        self._initialized = True

    async def close(self):
        """Close all connections."""
        if self.http_client:
            await self.http_client.aclose()


# Global service manager
services = AsyncServiceManager()


# ============================================
# Dependencies
# ============================================

async def get_retriever():
    """Dependency: Get retriever or raise error."""
    if not services.retriever:
        raise ServiceUnavailable("Retriever")
    return services.retriever


async def get_rag_service():
    """Dependency: Get RAG service or raise error."""
    if not services.rag_service:
        raise ServiceUnavailable("LLM service")
    return services.rag_service


async def get_minio():
    """Dependency: Get MinIO client or raise error."""
    if not services.minio_client:
        raise ServiceUnavailable("Storage")
    return services.minio_client


# ============================================
# Retry Logic
# ============================================

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True
)
async def search_with_retry(retriever, query: str, top_k: int, collection: str):
    """Search with automatic retry on failure."""
    from retriever import SearchMode
    return retriever.search(
        query=query,
        top_k=top_k,
        mode=SearchMode.HYBRID,
        collection=collection,
    )


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)
async def generate_with_retry(rag_service, query: str, sources: list, temperature: float):
    """Generate answer with retry."""
    return await rag_service.agenerate(
        query=query,
        sources=sources,
        temperature=temperature,
        verify=False,  # Skip verification for speed
    )


# ============================================
# Lifespan Management
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown."""
    logger.info("Starting RAG API...")
    settings = get_settings()
    await services.initialize(settings)
    logger.info("RAG API ready")

    yield

    logger.info("Shutting down RAG API...")
    await services.close()


# ============================================
# FastAPI Application
# ============================================

app = FastAPI(
    title="Enterprise RAG API",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# Exception Handlers
# ============================================

@app.exception_handler(RAGException)
async def rag_exception_handler(request: Request, exc: RAGException):
    """Handle custom RAG exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.message, "type": type(exc).__name__}
    )


# ============================================
# Health Endpoint
# ============================================

@app.get("/health", response_model=HealthResponse)
async def health():
    """Check service health."""
    checks = {
        "api": "healthy",
        "qdrant": "healthy" if services.retriever else "unavailable",
        "storage": "healthy" if services.minio_client else "unavailable",
        "llm": "healthy" if services.rag_service else "unavailable",
    }

    overall = "healthy" if all(v == "healthy" for v in checks.values()) else "degraded"

    return HealthResponse(
        status=overall,
        timestamp=datetime.utcnow().isoformat(),
        services=checks,
    )


# ============================================
# Search Endpoint
# ============================================

@app.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    retriever=Depends(get_retriever),
    settings: Settings = Depends(get_settings),
):
    """
    Search documents using hybrid search.

    Combines semantic (dense) and keyword (sparse) search
    with Reciprocal Rank Fusion for best results.
    """
    start = time.time()
    collection = request.collection or settings.qdrant_collection

    # Search with retry
    response = await search_with_retry(
        retriever,
        request.query,
        request.top_k,
        collection
    )

    results = [
        SearchResult(
            chunk_id=r.id,
            text=r.text,
            score=r.score,
            source_file=r.metadata.get("source_file", "unknown"),
            page_number=r.metadata.get("page_number"),
        )
        for r in response.results
    ]

    return SearchResponse(
        query=request.query,
        results=results,
        total_results=len(results),
        search_time_ms=(time.time() - start) * 1000,
    )


# ============================================
# Query Endpoint (RAG)
# ============================================

@app.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    retriever=Depends(get_retriever),
    rag_service=Depends(get_rag_service),
    settings: Settings = Depends(get_settings),
):
    """
    Answer a question using RAG.

    1. Search for relevant document chunks
    2. Send chunks + question to LLM
    3. Return answer with citations
    """
    start = time.time()
    collection = request.collection or settings.qdrant_collection

    # Step 1: Search
    search_start = time.time()
    search_response = await search_with_retry(
        retriever,
        request.query,
        request.top_k,
        collection,
    )
    search_time = (time.time() - search_start) * 1000

    sources = [
        {"text": r.text, "score": r.score, "metadata": r.metadata}
        for r in search_response.results
    ]

    # Step 2: Generate answer
    gen_start = time.time()
    llm_response = await generate_with_retry(
        rag_service,
        request.query,
        sources,
        request.temperature,
    )
    gen_time = (time.time() - gen_start) * 1000

    # Build response
    results = [
        SearchResult(
            chunk_id=r.id,
            text=r.text,
            score=r.score,
            source_file=r.metadata.get("source_file", "unknown"),
            page_number=r.metadata.get("page_number"),
        )
        for r in search_response.results
    ]

    citations = [
        Citation(
            index=c.index,
            source_file=c.source_file,
            page_number=c.page_number,
            text_snippet=c.text_snippet[:200],
        )
        for c in llm_response.citations
    ]

    return QueryResponse(
        query=request.query,
        answer=llm_response.answer,
        citations=citations,
        sources=results,
        model=llm_response.model,
        search_time_ms=search_time,
        generation_time_ms=gen_time,
        total_time_ms=(time.time() - start) * 1000,
    )


# ============================================
# Upload Endpoint
# ============================================

@app.post("/upload", response_model=UploadResponse)
async def upload(
    file: Annotated[UploadFile, File(description="Document to upload")],
    minio=Depends(get_minio),
    settings: Settings = Depends(get_settings),
):
    """Upload a document for processing."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename")

    # Validate extension
    allowed = {".pdf", ".txt", ".md", ".docx"}
    ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Allowed: {allowed}")

    # Upload to MinIO
    file_id = str(uuid.uuid4())
    object_name = f"{file_id}_{file.filename}"
    content = await file.read()

    minio.put_object(
        settings.minio_bucket_raw,
        object_name,
        BytesIO(content),
        len(content),
    )

    logger.info("File uploaded", file_id=file_id, size=len(content))

    return UploadResponse(
        file_id=file_id,
        filename=file.filename,
        size_bytes=len(content),
        status="uploaded",
    )


# ============================================
# Process Endpoint
# ============================================

@app.post("/process/{file_id}", response_model=ProcessResponse)
async def process(
    file_id: str,
    minio=Depends(get_minio),
    retriever=Depends(get_retriever),
    settings: Settings = Depends(get_settings),
):
    """Process an uploaded document: extract, chunk, embed, index."""
    import fitz
    from qdrant_client.models import PointStruct

    # Find file
    objects = list(minio.list_objects(settings.minio_bucket_raw, prefix=file_id))
    if not objects:
        raise DocumentNotFound(file_id)

    obj = objects[0]
    filename = obj.object_name.split("_", 1)[1] if "_" in obj.object_name else obj.object_name

    logger.info("Processing", file_id=file_id, filename=filename)

    # Download
    response = minio.get_object(settings.minio_bucket_raw, obj.object_name)
    content = response.read()
    response.close()
    response.release_conn()

    # Extract text
    ext = filename.rsplit(".", 1)[-1].lower()

    if ext == "pdf":
        doc = fitz.open(stream=content, filetype="pdf")
        pages = [{"page": i+1, "text": p.get_text()} for i, p in enumerate(doc) if p.get_text().strip()]
        doc.close()
    else:
        pages = [{"page": 1, "text": content.decode("utf-8")}]

    if not pages:
        raise HTTPException(status_code=400, detail="No text extracted")

    # Chunk
    chunks = []
    for page in pages:
        paragraphs = [p.strip() for p in page["text"].split("\n\n") if p.strip()]
        current = ""
        for para in paragraphs:
            if len(current) + len(para) < 500:
                current += para + "\n\n"
            else:
                if current:
                    chunks.append({"text": current.strip(), "page": page["page"], "source": filename})
                current = para + "\n\n"
        if current.strip():
            chunks.append({"text": current.strip(), "page": page["page"], "source": filename})

    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks created")

    # Embed
    texts = [c["text"] for c in chunks]
    embeddings = list(retriever.embedder.embed_batch(texts))

    # Index
    points = [
        PointStruct(
            id=hashlib.md5(f"{file_id}_{i}".encode()).hexdigest(),
            vector={"dense": emb.dense.to_list()},
            payload={
                "text": chunk["text"],
                "source_file": chunk["source"],
                "page_number": chunk["page"],
                "file_id": file_id,
            },
        )
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
    ]

    retriever.client.upsert(
        collection_name=settings.qdrant_collection,
        points=points,
    )

    logger.info("Indexed", file_id=file_id, chunks=len(points))

    return ProcessResponse(
        file_id=file_id,
        filename=filename,
        chunks_created=len(points),
        status="processed",
    )


# ============================================
# Collections Endpoint
# ============================================

@app.get("/collections")
async def list_collections(retriever=Depends(get_retriever)):
    """List vector collections."""
    cols = retriever.client.get_collections()
    return {"collections": [{"name": c.name} for c in cols.collections]}


@app.get("/collections/{name}")
async def get_collection(name: str, retriever=Depends(get_retriever)):
    """Get collection info."""
    try:
        info = retriever.client.get_collection(name)
        return {
            "name": name,
            "points_count": getattr(info, "points_count", 0),
            "status": str(info.status),
        }
    except Exception:
        raise HTTPException(status_code=404, detail=f"Collection not found: {name}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
