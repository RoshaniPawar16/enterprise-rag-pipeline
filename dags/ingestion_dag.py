"""
Data Ingestion DAG for Enterprise RAG Pipeline

This DAG orchestrates the end-to-end data ingestion process:
1. Monitors Azure Blob Storage (or MinIO locally) for new documents
2. Extracts text from PDFs, DOCX, and other formats
3. Applies semantic chunking with metadata enrichment
4. Uploads processed chunks for downstream embedding

Author: Enterprise RAG Pipeline
Version: 1.0.0
"""

import json
import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.decorators import task
from airflow.operators.python import PythonOperator
from airflow.sensors.base import PokeReturnValue
from airflow.models import Variable
from airflow.utils.trigger_rule import TriggerRule

# Add src to path for local imports
sys.path.insert(0, "/opt/airflow/src")

# ============================================
# DAG Configuration
# ============================================

default_args = {
    "owner": "rag-pipeline",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=1),
}

# Environment-specific configuration
ENVIRONMENT = os.getenv("ENVIRONMENT", "local")
USE_AZURE = os.getenv("USE_AZURE", "false").lower() == "true"

# Storage configuration
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")

# Azure configuration (for production)
AZURE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
AZURE_CONTAINER_RAW = os.getenv("AZURE_CONTAINER_RAW", "raw-documents")
AZURE_CONTAINER_PROCESSED = os.getenv("AZURE_CONTAINER_PROCESSED", "processed-chunks")

# MinIO buckets (for local development)
MINIO_BUCKET_RAW = "raw-documents"
MINIO_BUCKET_PROCESSED = "processed-chunks"


# ============================================
# Helper Functions
# ============================================

def get_storage_client():
    """Get the appropriate storage client based on environment."""
    if USE_AZURE and AZURE_CONNECTION_STRING:
        from azure.storage.blob import BlobServiceClient
        return BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
    else:
        from minio import Minio
        return Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=False,
        )


def list_new_files(client, last_processed_marker: str | None = None) -> list[str]:
    """List files in raw storage that haven't been processed."""
    files = []

    if USE_AZURE and AZURE_CONNECTION_STRING:
        container = client.get_container_client(AZURE_CONTAINER_RAW)
        blobs = container.list_blobs()
        for blob in blobs:
            if blob.name.endswith(('.pdf', '.docx', '.txt', '.md')):
                files.append(blob.name)
    else:
        objects = client.list_objects(MINIO_BUCKET_RAW, recursive=True)
        for obj in objects:
            if obj.object_name.endswith(('.pdf', '.docx', '.txt', '.md')):
                files.append(obj.object_name)

    return files


def download_file(client, file_name: str, local_path: str):
    """Download a file from storage to local path."""
    if USE_AZURE and AZURE_CONNECTION_STRING:
        container = client.get_container_client(AZURE_CONTAINER_RAW)
        blob = container.get_blob_client(file_name)
        with open(local_path, "wb") as f:
            data = blob.download_blob()
            f.write(data.readall())
    else:
        client.fget_object(MINIO_BUCKET_RAW, file_name, local_path)


def upload_chunks(client, chunks: list[dict], source_file: str):
    """Upload processed chunks to storage."""
    output_name = f"{Path(source_file).stem}_chunks.json"
    chunks_json = json.dumps(chunks, indent=2, ensure_ascii=False)

    if USE_AZURE and AZURE_CONNECTION_STRING:
        container = client.get_container_client(AZURE_CONTAINER_PROCESSED)
        blob = container.get_blob_client(output_name)
        blob.upload_blob(chunks_json.encode(), overwrite=True)
    else:
        from io import BytesIO
        data = BytesIO(chunks_json.encode())
        client.put_object(
            MINIO_BUCKET_PROCESSED,
            output_name,
            data,
            len(chunks_json.encode()),
            content_type="application/json",
        )

    return output_name


# ============================================
# DAG Definition
# ============================================

with DAG(
    dag_id="enterprise_rag_ingestion",
    default_args=default_args,
    description="Automated data ingestion pipeline for RAG system",
    schedule_interval="*/15 * * * *",  # Every 15 minutes
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["rag", "ingestion", "data-engineering"],
    doc_md=__doc__,
) as dag:

    @task.sensor(poke_interval=60, timeout=3600, mode="poke")
    def wait_for_new_files() -> PokeReturnValue:
        """
        Sensor that waits for new files in the raw documents bucket.

        This replaces the traditional AzureBlobStorageSensor with a
        more flexible custom sensor that works with both Azure and MinIO.
        """
        import structlog
        logger = structlog.get_logger(__name__)

        try:
            client = get_storage_client()
            files = list_new_files(client)

            if files:
                logger.info("New files detected", count=len(files), files=files[:5])
                return PokeReturnValue(is_done=True, xcom_value=files)
            else:
                logger.debug("No new files found, continuing to poll")
                return PokeReturnValue(is_done=False)

        except Exception as e:
            logger.error("Error checking for files", error=str(e))
            return PokeReturnValue(is_done=False)

    @task
    def extract_text(files: list[str]) -> list[dict]:
        """
        Extract text content from documents.

        Supports:
        - PDF (using PyMuPDF)
        - DOCX (using python-docx)
        - TXT/MD (plain text)
        """
        import fitz  # PyMuPDF
        from docx import Document
        import structlog

        logger = structlog.get_logger(__name__)
        client = get_storage_client()
        extracted = []

        for file_name in files:
            logger.info("Processing file", file=file_name)

            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file_name).suffix) as tmp:
                try:
                    download_file(client, file_name, tmp.name)
                    suffix = Path(file_name).suffix.lower()

                    if suffix == ".pdf":
                        text = extract_pdf_text(tmp.name)
                    elif suffix == ".docx":
                        text = extract_docx_text(tmp.name)
                    elif suffix in (".txt", ".md"):
                        with open(tmp.name, "r", encoding="utf-8") as f:
                            text = f.read()
                    else:
                        logger.warning("Unsupported file type", file=file_name, suffix=suffix)
                        continue

                    extracted.append({
                        "file_name": file_name,
                        "text": text,
                        "char_count": len(text),
                        "extracted_at": datetime.utcnow().isoformat(),
                    })

                    logger.info(
                        "Text extracted successfully",
                        file=file_name,
                        chars=len(text),
                    )

                except Exception as e:
                    logger.error("Failed to extract text", file=file_name, error=str(e))

                finally:
                    os.unlink(tmp.name)

        return extracted


    def extract_pdf_text(pdf_path: str) -> str:
        """Extract text from PDF with page tracking."""
        import fitz

        doc = fitz.open(pdf_path)
        text_parts = []

        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")
            if text.strip():
                text_parts.append(f"[Page {page_num}]\n{text}")

        doc.close()
        return "\n\n".join(text_parts)


    def extract_docx_text(docx_path: str) -> str:
        """Extract text from DOCX."""
        from docx import Document

        doc = Document(docx_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs)


    @task
    def apply_semantic_chunking(documents: list[dict]) -> list[dict]:
        """
        Apply semantic chunking to extracted documents.

        Uses the advanced_semantic_chunking function from the chunker module.
        """
        from chunker import advanced_semantic_chunking
        import structlog

        logger = structlog.get_logger(__name__)
        all_chunks = []

        for doc in documents:
            logger.info(
                "Chunking document",
                file=doc["file_name"],
                chars=doc["char_count"],
            )

            try:
                chunks = advanced_semantic_chunking(
                    text=doc["text"],
                    source_file=doc["file_name"],
                    preset="default",
                )

                # Add extraction metadata to each chunk
                for chunk in chunks:
                    chunk["metadata"]["extracted_at"] = doc["extracted_at"]

                all_chunks.extend(chunks)

                logger.info(
                    "Chunking complete",
                    file=doc["file_name"],
                    num_chunks=len(chunks),
                )

            except Exception as e:
                logger.error(
                    "Chunking failed",
                    file=doc["file_name"],
                    error=str(e),
                )

        return all_chunks


    @task
    def enrich_metadata(chunks: list[dict]) -> list[dict]:
        """
        Enrich chunks with additional metadata.

        This task adds:
        - Processing timestamp
        - Pipeline version
        - Quality indicators
        """
        import structlog
        logger = structlog.get_logger(__name__)

        enriched = []
        pipeline_version = "1.0.0"

        for chunk in chunks:
            # Add pipeline metadata
            chunk["metadata"]["pipeline_version"] = pipeline_version
            chunk["metadata"]["processed_at"] = datetime.utcnow().isoformat()

            # Add quality indicators
            text = chunk["text"]
            chunk["metadata"]["quality"] = {
                "has_numbers": bool(any(c.isdigit() for c in text)),
                "has_proper_nouns": bool(any(w[0].isupper() for w in text.split() if w)),
                "avg_word_length": sum(len(w) for w in text.split()) / max(len(text.split()), 1),
                "sentence_count": text.count('.') + text.count('!') + text.count('?'),
            }

            enriched.append(chunk)

        logger.info("Metadata enrichment complete", total_chunks=len(enriched))
        return enriched


    @task
    def upload_processed_chunks(chunks: list[dict]) -> dict:
        """
        Upload processed chunks to the processed storage bucket.

        Returns upload statistics.
        """
        import structlog
        logger = structlog.get_logger(__name__)

        client = get_storage_client()

        # Group chunks by source file
        by_source = {}
        for chunk in chunks:
            source = chunk["metadata"]["source_file"]
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(chunk)

        uploaded_files = []
        for source_file, file_chunks in by_source.items():
            output_name = upload_chunks(client, file_chunks, source_file)
            uploaded_files.append(output_name)
            logger.info(
                "Chunks uploaded",
                source=source_file,
                output=output_name,
                num_chunks=len(file_chunks),
            )

        return {
            "status": "success",
            "files_processed": len(by_source),
            "total_chunks": len(chunks),
            "uploaded_files": uploaded_files,
            "completed_at": datetime.utcnow().isoformat(),
        }


    @task(trigger_rule=TriggerRule.ALL_DONE)
    def log_pipeline_metrics(upload_result: dict):
        """
        Log pipeline metrics for monitoring.

        In production, this would push metrics to Prometheus/Azure Monitor.
        """
        import structlog
        logger = structlog.get_logger(__name__)

        logger.info(
            "Pipeline completed",
            **upload_result,
        )

        # TODO: Push metrics to MLflow/Prometheus
        # mlflow.log_metrics({
        #     "files_processed": upload_result["files_processed"],
        #     "total_chunks": upload_result["total_chunks"],
        # })


    # ============================================
    # DAG Task Dependencies
    # ============================================

    files = wait_for_new_files()
    documents = extract_text(files)
    chunks = apply_semantic_chunking(documents)
    enriched_chunks = enrich_metadata(chunks)
    upload_result = upload_processed_chunks(enriched_chunks)
    log_pipeline_metrics(upload_result)


# ============================================
# Supplementary DAGs
# ============================================

# DAG for manual reprocessing of specific files
with DAG(
    dag_id="enterprise_rag_reprocess",
    default_args=default_args,
    description="Manually reprocess specific documents",
    schedule_interval=None,  # Manual trigger only
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["rag", "ingestion", "manual"],
    params={
        "files": {
            "type": "array",
            "description": "List of file names to reprocess",
            "default": [],
        }
    },
) as reprocess_dag:

    @task
    def get_files_to_reprocess(**context) -> list[str]:
        """Get files from DAG params."""
        params = context["params"]
        return params.get("files", [])

    @task
    def reprocess_files(files: list[str]) -> list[dict]:
        """Reprocess specified files through the pipeline."""
        # Reuse the same extraction logic
        return extract_text.function(files)

    files_to_process = get_files_to_reprocess()
    reprocessed = reprocess_files(files_to_process)
    chunks = apply_semantic_chunking(reprocessed)
    enriched = enrich_metadata(chunks)
    result = upload_processed_chunks(enriched)
