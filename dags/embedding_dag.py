"""
Embedding Pipeline DAG for Enterprise RAG

This DAG handles the embedding generation and vector storage:
1. Reads processed chunks from storage
2. Generates hybrid embeddings (dense + sparse)
3. Upserts vectors into Qdrant
4. Tracks metrics in MLflow

Can be triggered:
- Automatically after ingestion_dag completes
- Manually for batch reprocessing
- On a schedule for incremental updates
"""

import json
import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.decorators import task
from airflow.models import Variable
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.utils.trigger_rule import TriggerRule

# Add src to path
sys.path.insert(0, "/opt/airflow/src")

# ============================================
# Configuration
# ============================================

default_args = {
    "owner": "rag-pipeline",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}

# Environment configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "enterprise_docs")

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET_PROCESSED = "processed-chunks"

EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", 32))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")


# ============================================
# Helper Functions
# ============================================

def get_minio_client():
    """Get MinIO client."""
    from minio import Minio
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False,
    )


def get_qdrant_client():
    """Get Qdrant client."""
    from qdrant_client import QdrantClient
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


# ============================================
# DAG Definition
# ============================================

with DAG(
    dag_id="enterprise_rag_embedding",
    default_args=default_args,
    description="Generate embeddings and upsert to Qdrant",
    schedule_interval=None,  # Triggered by ingestion DAG or manually
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["rag", "embedding", "ai-engineering"],
    doc_md=__doc__,
    params={
        "collection": {
            "type": "string",
            "default": COLLECTION_NAME,
            "description": "Qdrant collection to upsert into",
        },
        "reprocess_all": {
            "type": "boolean",
            "default": False,
            "description": "Reprocess all chunks, not just new ones",
        },
        "files": {
            "type": "array",
            "default": [],
            "description": "Specific chunk files to process (empty = all)",
        },
    },
) as dag:

    @task
    def list_chunk_files(**context) -> list[str]:
        """
        List processed chunk files ready for embedding.

        Checks for new files that haven't been embedded yet.
        """
        import structlog
        logger = structlog.get_logger(__name__)

        params = context["params"]
        specific_files = params.get("files", [])
        reprocess_all = params.get("reprocess_all", False)

        client = get_minio_client()

        # List all chunk files
        objects = client.list_objects(MINIO_BUCKET_PROCESSED, recursive=True)
        chunk_files = [
            obj.object_name
            for obj in objects
            if obj.object_name.endswith("_chunks.json")
        ]

        # Filter to specific files if provided
        if specific_files:
            chunk_files = [f for f in chunk_files if f in specific_files]

        # Filter to only new files (unless reprocessing all)
        if not reprocess_all:
            # Check what's already in Qdrant
            qdrant = get_qdrant_client()
            try:
                # Get list of already indexed files
                indexed_files = set()
                scroll_result = qdrant.scroll(
                    collection_name=params.get("collection", COLLECTION_NAME),
                    scroll_filter=None,
                    limit=10000,
                    with_payload=["source_file"],
                )

                for point in scroll_result[0]:
                    if point.payload:
                        indexed_files.add(point.payload.get("source_file"))

                # Filter to only new files
                new_files = []
                for chunk_file in chunk_files:
                    # Extract original source file from chunk filename
                    source_file = chunk_file.replace("_chunks.json", "")
                    if source_file not in indexed_files:
                        new_files.append(chunk_file)

                chunk_files = new_files
                logger.info(
                    "Filtered to new files",
                    total=len(chunk_files),
                    already_indexed=len(indexed_files),
                )

            except Exception as e:
                logger.warning("Could not check indexed files", error=str(e))

        logger.info("Found chunk files", count=len(chunk_files))
        return chunk_files

    @task
    def load_chunks(chunk_files: list[str]) -> list[dict]:
        """
        Load chunk data from storage.

        Returns all chunks ready for embedding.
        """
        import structlog
        import tempfile

        logger = structlog.get_logger(__name__)
        client = get_minio_client()

        all_chunks = []

        for chunk_file in chunk_files:
            try:
                # Download chunk file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
                    client.fget_object(MINIO_BUCKET_PROCESSED, chunk_file, tmp.name)

                    with open(tmp.name, "r") as f:
                        chunks = json.load(f)

                    all_chunks.extend(chunks)
                    logger.info(
                        "Loaded chunks",
                        file=chunk_file,
                        count=len(chunks),
                    )

                    os.unlink(tmp.name)

            except Exception as e:
                logger.error("Failed to load chunks", file=chunk_file, error=str(e))

        logger.info("Total chunks loaded", count=len(all_chunks))
        return all_chunks

    @task
    def generate_embeddings(chunks: list[dict], **context) -> list[dict]:
        """
        Generate hybrid embeddings for all chunks.

        Uses BGE-M3 for both dense and sparse embeddings.
        """
        import structlog
        from embedding_engine import HybridEmbedder

        logger = structlog.get_logger(__name__)

        if not chunks:
            logger.warning("No chunks to embed")
            return []

        logger.info(
            "Generating embeddings",
            chunks=len(chunks),
            model=EMBEDDING_MODEL,
            batch_size=EMBEDDING_BATCH_SIZE,
        )

        # Initialize embedder
        embedder = HybridEmbedder(
            model_name=EMBEDDING_MODEL,
            batch_size=EMBEDDING_BATCH_SIZE,
        )

        # Extract texts
        texts = [chunk["text"] for chunk in chunks]

        # Generate embeddings in batches
        embedded_chunks = []
        embeddings = list(embedder.embed_batch(texts, show_progress=True))

        for chunk, embedding in zip(chunks, embeddings):
            embedded_chunk = {
                **chunk,
                "dense_vector": embedding.dense.to_list(),
                "sparse_vector": embedding.sparse.to_dict(),
                "embedding_model": EMBEDDING_MODEL,
                "embedding_dim": embedding.dense.dimension,
            }
            embedded_chunks.append(embedded_chunk)

        logger.info(
            "Embedding generation complete",
            chunks=len(embedded_chunks),
            dense_dim=embeddings[0].dense.dimension if embeddings else 0,
        )

        return embedded_chunks

    @task
    def upsert_to_qdrant(chunks: list[dict], **context) -> dict:
        """
        Upsert embedded chunks to Qdrant.

        Creates points with both dense and sparse vectors.
        """
        import structlog
        from qdrant_client import models

        logger = structlog.get_logger(__name__)

        params = context["params"]
        collection = params.get("collection", COLLECTION_NAME)

        if not chunks:
            logger.warning("No chunks to upsert")
            return {"status": "skipped", "reason": "no chunks"}

        client = get_qdrant_client()

        logger.info(
            "Upserting to Qdrant",
            chunks=len(chunks),
            collection=collection,
        )

        # Build points
        points = []
        for chunk in chunks:
            # Build sparse vector if available
            sparse_vectors = None
            if chunk.get("sparse_vector") and chunk["sparse_vector"].get("indices"):
                sparse_vectors = {
                    "sparse": models.SparseVector(
                        indices=chunk["sparse_vector"]["indices"],
                        values=chunk["sparse_vector"]["values"],
                    )
                }

            point = models.PointStruct(
                id=chunk["metadata"]["chunk_id"],
                vector={
                    "dense": chunk["dense_vector"],
                },
                sparse_vectors=sparse_vectors,
                payload={
                    "text": chunk["text"],
                    "source_file": chunk["metadata"]["source_file"],
                    "page_number": chunk["metadata"].get("page_number"),
                    "chunk_index": chunk["metadata"]["chunk_index"],
                    "total_chunks": chunk["metadata"]["total_chunks"],
                    "importance_score": chunk["metadata"]["importance_score"],
                    "semantic_type": chunk["metadata"]["semantic_type"],
                    "headers": chunk["metadata"].get("headers", []),
                    "created_at": chunk["metadata"]["created_at"],
                    "embedding_model": chunk.get("embedding_model"),
                },
            )
            points.append(point)

        # Batch upsert
        batch_size = 100
        total_upserted = 0

        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            client.upsert(
                collection_name=collection,
                points=batch,
            )
            total_upserted += len(batch)
            logger.debug(
                "Batch upserted",
                batch=i // batch_size + 1,
                count=len(batch),
            )

        logger.info(
            "Upsert complete",
            total=total_upserted,
            collection=collection,
        )

        return {
            "status": "success",
            "documents_upserted": total_upserted,
            "collection": collection,
        }

    @task
    def log_metrics(upsert_result: dict, chunks: list[dict], **context) -> dict:
        """
        Log embedding metrics to MLflow.

        Tracks:
        - Documents processed
        - Embedding generation time
        - Vector storage stats
        """
        import structlog

        logger = structlog.get_logger(__name__)

        metrics = {
            "documents_upserted": upsert_result.get("documents_upserted", 0),
            "total_chunks": len(chunks),
            "collection": upsert_result.get("collection"),
            "embedding_model": EMBEDDING_MODEL,
            "completed_at": datetime.utcnow().isoformat(),
        }

        logger.info("Embedding pipeline metrics", **metrics)

        # TODO: Push to MLflow
        # import mlflow
        # with mlflow.start_run(run_name="embedding_pipeline"):
        #     mlflow.log_metrics({
        #         "documents_upserted": metrics["documents_upserted"],
        #         "total_chunks": metrics["total_chunks"],
        #     })

        return metrics

    @task(trigger_rule=TriggerRule.ALL_DONE)
    def verify_collection(**context) -> dict:
        """
        Verify the collection state after upsert.

        Checks document counts and collection health.
        """
        import structlog

        logger = structlog.get_logger(__name__)

        params = context["params"]
        collection = params.get("collection", COLLECTION_NAME)

        client = get_qdrant_client()

        try:
            info = client.get_collection(collection)

            result = {
                "collection": collection,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status.value,
                "verified_at": datetime.utcnow().isoformat(),
            }

            logger.info("Collection verified", **result)
            return result

        except Exception as e:
            logger.error("Collection verification failed", error=str(e))
            return {"error": str(e)}

    # ============================================
    # Task Dependencies
    # ============================================

    chunk_files = list_chunk_files()
    chunks = load_chunks(chunk_files)
    embedded_chunks = generate_embeddings(chunks)
    upsert_result = upsert_to_qdrant(embedded_chunks)
    metrics = log_metrics(upsert_result, embedded_chunks)
    verification = verify_collection()

    # Set dependencies
    upsert_result >> metrics >> verification


# ============================================
# Scheduled Embedding DAG
# ============================================

with DAG(
    dag_id="enterprise_rag_embedding_scheduled",
    default_args=default_args,
    description="Scheduled embedding pipeline (runs after ingestion)",
    schedule_interval="*/30 * * * *",  # Every 30 minutes
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["rag", "embedding", "scheduled"],
) as scheduled_dag:

    # Wait for ingestion to complete
    wait_for_ingestion = ExternalTaskSensor(
        task_id="wait_for_ingestion",
        external_dag_id="enterprise_rag_ingestion",
        external_task_id=None,  # Wait for entire DAG
        mode="reschedule",
        timeout=3600,
        poke_interval=60,
        allowed_states=["success"],
        failed_states=["failed", "skipped"],
    )

    # Trigger embedding pipeline
    trigger_embedding = TriggerDagRunOperator(
        task_id="trigger_embedding",
        trigger_dag_id="enterprise_rag_embedding",
        wait_for_completion=False,
        conf={
            "collection": COLLECTION_NAME,
            "reprocess_all": False,
        },
    )

    wait_for_ingestion >> trigger_embedding
