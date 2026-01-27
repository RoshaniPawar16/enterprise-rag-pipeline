#!/usr/bin/env python3
"""
Qdrant Collection Initialization Script

Sets up the vector collections with hybrid search capabilities:
- Dense vectors (BGE-M3, 1024-dim) for semantic search
- Sparse vectors for keyword/BM25-style search
- CLIP vectors for multimodal image search
- Proper indexing and optimization settings

Run this script before starting the ingestion pipeline.
"""

import os
import sys
from typing import Any

import structlog
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

logger = structlog.get_logger(__name__)

# Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

# Collection configurations
COLLECTIONS = {
    "enterprise_docs": {
        "description": "Main document collection with hybrid search",
        "dense_dim": 1024,  # BGE-M3
        "enable_sparse": True,
        "enable_multimodal": True,
        "multimodal_dim": 512,  # CLIP ViT-B-32
    },
    "financial_reports": {
        "description": "Financial documents and reports",
        "dense_dim": 1024,
        "enable_sparse": True,
        "enable_multimodal": True,
        "multimodal_dim": 512,
    },
    "clinical_trials": {
        "description": "Clinical trial documents and research",
        "dense_dim": 1024,
        "enable_sparse": True,
        "enable_multimodal": False,
    },
}


def create_hybrid_collection(
    client: QdrantClient,
    collection_name: str,
    config: dict[str, Any],
    recreate: bool = False,
) -> bool:
    """
    Create a collection with hybrid search support.

    Args:
        client: Qdrant client
        collection_name: Name of the collection
        config: Collection configuration
        recreate: If True, delete and recreate existing collection

    Returns:
        True if collection was created/updated successfully
    """
    logger.info(
        "Creating collection",
        name=collection_name,
        config=config,
    )

    # Check if collection exists
    try:
        existing = client.get_collection(collection_name)
        if existing and not recreate:
            logger.info("Collection already exists", name=collection_name)
            return True
        elif existing and recreate:
            logger.warning("Deleting existing collection", name=collection_name)
            client.delete_collection(collection_name)
    except UnexpectedResponse:
        pass  # Collection doesn't exist

    # Build vectors configuration
    vectors_config = {
        # Dense vector for semantic search
        "dense": models.VectorParams(
            size=config["dense_dim"],
            distance=models.Distance.COSINE,
            on_disk=True,  # Optimize for large collections
            hnsw_config=models.HnswConfigDiff(
                m=16,  # Number of connections per node
                ef_construct=100,  # Build-time accuracy
                full_scan_threshold=10000,  # Switch to brute force for small collections
            ),
            quantization_config=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True,  # Keep quantized vectors in RAM
                ),
            ),
        ),
    }

    # Add multimodal vector if enabled
    if config.get("enable_multimodal", False):
        vectors_config["clip"] = models.VectorParams(
            size=config.get("multimodal_dim", 512),
            distance=models.Distance.COSINE,
            on_disk=True,
        )

    # Sparse vectors configuration
    sparse_vectors_config = None
    if config.get("enable_sparse", True):
        sparse_vectors_config = {
            "sparse": models.SparseVectorParams(
                index=models.SparseIndexParams(
                    on_disk=True,  # Store sparse index on disk
                ),
            ),
        }

    # Create collection
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config,
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=20000,  # Start indexing after 20k points
                memmap_threshold=50000,  # Use mmap for collections > 50k
            ),
            # Payload schema for filtering
            on_disk_payload=True,  # Store payloads on disk
        )

        logger.info("Collection created successfully", name=collection_name)

        # Create payload indexes for common filter fields
        _create_payload_indexes(client, collection_name)

        return True

    except Exception as e:
        logger.error("Failed to create collection", name=collection_name, error=str(e))
        return False


def _create_payload_indexes(client: QdrantClient, collection_name: str):
    """Create indexes on frequently filtered payload fields."""
    index_fields = [
        ("source_file", models.PayloadSchemaType.KEYWORD),
        ("page_number", models.PayloadSchemaType.INTEGER),
        ("semantic_type", models.PayloadSchemaType.KEYWORD),
        ("importance_score", models.PayloadSchemaType.FLOAT),
        ("created_at", models.PayloadSchemaType.DATETIME),
    ]

    for field_name, field_type in index_fields:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=field_type,
            )
            logger.debug("Created index", collection=collection_name, field=field_name)
        except Exception as e:
            logger.debug("Index may already exist", field=field_name, error=str(e))


def get_collection_info(client: QdrantClient, collection_name: str) -> dict:
    """Get information about a collection."""
    try:
        info = client.get_collection(collection_name)
        return {
            "name": collection_name,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": info.status.value,
            "optimizer_status": info.optimizer_status.status.value,
            "config": {
                "vectors": {k: v.size for k, v in info.config.params.vectors.items()}
                if hasattr(info.config.params, 'vectors') else {},
            },
        }
    except Exception as e:
        return {"name": collection_name, "error": str(e)}


def initialize_all_collections(
    host: str = QDRANT_HOST,
    port: int = QDRANT_PORT,
    recreate: bool = False,
) -> dict[str, bool]:
    """
    Initialize all configured collections.

    Args:
        host: Qdrant host
        port: Qdrant port
        recreate: If True, recreate existing collections

    Returns:
        Dict mapping collection names to success status
    """
    logger.info("Connecting to Qdrant", host=host, port=port)

    client = QdrantClient(host=host, port=port)

    # Verify connection
    try:
        client.get_collections()
        logger.info("Connected to Qdrant successfully")
    except Exception as e:
        logger.error("Failed to connect to Qdrant", error=str(e))
        return {}

    results = {}
    for name, config in COLLECTIONS.items():
        results[name] = create_hybrid_collection(
            client=client,
            collection_name=name,
            config=config,
            recreate=recreate,
        )

    # Print summary
    print("\n" + "=" * 50)
    print("Qdrant Collection Initialization Summary")
    print("=" * 50)

    for name, success in results.items():
        status = "✅ Created" if success else "❌ Failed"
        print(f"  {name}: {status}")

        if success:
            info = get_collection_info(client, name)
            print(f"    Points: {info.get('points_count', 0)}")
            print(f"    Status: {info.get('status', 'unknown')}")

    print("=" * 50 + "\n")

    return results


def verify_hybrid_search(
    client: QdrantClient,
    collection_name: str = "enterprise_docs",
) -> bool:
    """
    Verify that hybrid search is working correctly.

    Inserts a test point and performs both dense and sparse searches.
    """
    import numpy as np

    logger.info("Verifying hybrid search", collection=collection_name)

    # Generate test vectors
    test_dense = np.random.rand(1024).tolist()
    test_sparse_indices = [1, 5, 10, 50, 100]
    test_sparse_values = [0.5, 0.3, 0.8, 0.2, 0.9]

    test_point_id = "test-hybrid-verification"

    try:
        # Upsert test point
        client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=test_point_id,
                    vector={
                        "dense": test_dense,
                    },
                    sparse_vectors={
                        "sparse": models.SparseVector(
                            indices=test_sparse_indices,
                            values=test_sparse_values,
                        ),
                    },
                    payload={
                        "text": "Test document for hybrid search verification",
                        "source_file": "test.txt",
                        "is_test": True,
                    },
                ),
            ],
        )

        # Test dense search
        dense_results = client.search(
            collection_name=collection_name,
            query_vector=models.NamedVector(
                name="dense",
                vector=test_dense,
            ),
            limit=1,
        )

        # Test sparse search
        sparse_results = client.search(
            collection_name=collection_name,
            query_vector=models.NamedSparseVector(
                name="sparse",
                vector=models.SparseVector(
                    indices=test_sparse_indices,
                    values=test_sparse_values,
                ),
            ),
            limit=1,
        )

        # Clean up test point
        client.delete(
            collection_name=collection_name,
            points_selector=models.PointIdsList(
                points=[test_point_id],
            ),
        )

        if dense_results and sparse_results:
            logger.info("Hybrid search verification passed")
            return True
        else:
            logger.error("Hybrid search verification failed")
            return False

    except Exception as e:
        logger.error("Hybrid search verification error", error=str(e))
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Initialize Qdrant collections")
    parser.add_argument("--host", default=QDRANT_HOST, help="Qdrant host")
    parser.add_argument("--port", type=int, default=QDRANT_PORT, help="Qdrant port")
    parser.add_argument("--recreate", action="store_true", help="Recreate existing collections")
    parser.add_argument("--verify", action="store_true", help="Verify hybrid search after init")

    args = parser.parse_args()

    results = initialize_all_collections(
        host=args.host,
        port=args.port,
        recreate=args.recreate,
    )

    if args.verify and results.get("enterprise_docs", False):
        client = QdrantClient(host=args.host, port=args.port)
        verify_hybrid_search(client, "enterprise_docs")

    # Exit with error if any collection failed
    if not all(results.values()):
        sys.exit(1)
