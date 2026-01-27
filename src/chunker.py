"""
Semantic Chunking Module for Enterprise RAG Pipeline

This module implements production-grade semantic chunking that preserves
contextual integrity - critical for downstream LLM performance.

Key Features:
- Token-aware chunking (respects model context windows)
- Semantic boundary detection (headers, paragraphs, sentences)
- Metadata enrichment with importance scoring
- Support for multiple chunking strategies
"""

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable

import semchunk
import structlog
from transformers import AutoTokenizer

logger = structlog.get_logger(__name__)


class ChunkingStrategy(Enum):
    """Available chunking strategies."""
    SEMANTIC = "semantic"
    RECURSIVE = "recursive"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"


@dataclass
class ChunkMetadata:
    """Metadata attached to each text chunk."""
    chunk_id: str
    source_file: str
    page_number: int | None
    chunk_index: int
    total_chunks: int
    token_count: int
    char_count: int
    importance_score: float
    created_at: str
    headers: list[str] = field(default_factory=list)
    semantic_type: str = "text"

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "source_file": self.source_file,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "token_count": self.token_count,
            "char_count": self.char_count,
            "importance_score": self.importance_score,
            "created_at": self.created_at,
            "headers": self.headers,
            "semantic_type": self.semantic_type,
        }


@dataclass
class Chunk:
    """A text chunk with its metadata."""
    text: str
    metadata: ChunkMetadata

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "metadata": self.metadata.to_dict(),
        }


class SemanticChunker:
    """
    Production-grade semantic chunker for RAG pipelines.

    Uses semchunk library for high-performance chunking with
    token-aware boundaries that respect model context windows.
    """

    # Header patterns for semantic boundary detection
    HEADER_PATTERNS = [
        r'^#{1,6}\s+.+$',  # Markdown headers
        r'^[A-Z][A-Z\s]{2,}$',  # ALL CAPS headers
        r'^\d+\.\s+[A-Z]',  # Numbered sections
        r'^(?:Chapter|Section|Part)\s+\d+',  # Document sections
        r'^(?:Abstract|Introduction|Conclusion|Summary|References)',  # Academic sections
    ]

    # Patterns indicating important content
    IMPORTANCE_PATTERNS = [
        (r'\b(?:key|important|critical|essential|significant)\b', 0.2),
        (r'\b(?:conclusion|summary|result|finding)\b', 0.15),
        (r'\b(?:however|therefore|consequently|thus)\b', 0.1),
        (r'\b(?:figure|table|chart|graph)\s+\d+', 0.1),
        (r'\b\d+(?:\.\d+)?%', 0.05),  # Percentages
        (r'£\d+|€\d+|\$\d+', 0.05),  # Currency
    ]

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 400,
        chunk_overlap: int = 50,
        strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC,
    ):
        """
        Initialize the semantic chunker.

        Args:
            model_name: HuggingFace model for tokenization
            chunk_size: Target chunk size in tokens
            chunk_overlap: Number of overlapping tokens between chunks
            strategy: Chunking strategy to use
        """
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy

        logger.info(
            "Initializing SemanticChunker",
            model=model_name,
            chunk_size=chunk_size,
            overlap=chunk_overlap,
            strategy=strategy.value,
        )

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Create semchunk chunker
        self._chunker = semchunk.chunkerify(
            self.tokenizer,
            chunk_size=chunk_size,
        )

        # Compile regex patterns
        self._header_regex = [re.compile(p, re.MULTILINE) for p in self.HEADER_PATTERNS]
        self._importance_regex = [(re.compile(p, re.IGNORECASE), w) for p, w in self.IMPORTANCE_PATTERNS]

    def chunk(
        self,
        text: str,
        source_file: str = "unknown",
        page_number: int | None = None,
        custom_metadata: dict | None = None,
    ) -> list[Chunk]:
        """
        Chunk text into semantically coherent segments.

        Args:
            text: The text to chunk
            source_file: Source filename for metadata
            page_number: Page number if from PDF
            custom_metadata: Additional metadata to include

        Returns:
            List of Chunk objects with metadata
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for chunking")
            return []

        # Extract headers for context
        headers = self._extract_headers(text)

        # Perform chunking based on strategy
        if self.strategy == ChunkingStrategy.SEMANTIC:
            raw_chunks = self._semantic_chunk(text)
        elif self.strategy == ChunkingStrategy.RECURSIVE:
            raw_chunks = self._recursive_chunk(text)
        elif self.strategy == ChunkingStrategy.SENTENCE:
            raw_chunks = self._sentence_chunk(text)
        else:
            raw_chunks = self._paragraph_chunk(text)

        # Build chunks with metadata
        chunks = []
        total_chunks = len(raw_chunks)

        for idx, chunk_text in enumerate(raw_chunks):
            chunk_id = self._generate_chunk_id(source_file, idx, chunk_text)
            token_count = len(self.tokenizer.encode(chunk_text))
            importance = self._calculate_importance(chunk_text)

            # Find relevant headers for this chunk
            chunk_headers = self._find_relevant_headers(text, chunk_text, headers)

            metadata = ChunkMetadata(
                chunk_id=chunk_id,
                source_file=source_file,
                page_number=page_number,
                chunk_index=idx,
                total_chunks=total_chunks,
                token_count=token_count,
                char_count=len(chunk_text),
                importance_score=importance,
                created_at=datetime.utcnow().isoformat(),
                headers=chunk_headers,
                semantic_type=self._detect_semantic_type(chunk_text),
            )

            chunks.append(Chunk(text=chunk_text, metadata=metadata))

        logger.info(
            "Chunking complete",
            source=source_file,
            total_chunks=total_chunks,
            avg_tokens=sum(c.metadata.token_count for c in chunks) / max(len(chunks), 1),
        )

        return chunks

    def _semantic_chunk(self, text: str) -> list[str]:
        """Use semchunk for semantic boundary detection."""
        return self._chunker(text)

    def _recursive_chunk(self, text: str) -> list[str]:
        """Recursive chunking with multiple separators."""
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size * 4,  # Approximate chars per token
            chunk_overlap=self.chunk_overlap * 4,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=lambda x: len(self.tokenizer.encode(x)),
        )
        return splitter.split_text(text)

    def _sentence_chunk(self, text: str) -> list[str]:
        """Chunk by sentences, grouping to target size."""
        import nltk
        try:
            sentences = nltk.sent_tokenize(text)
        except LookupError:
            nltk.download('punkt', quiet=True)
            sentences = nltk.sent_tokenize(text)

        chunks = []
        current_chunk = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = len(self.tokenizer.encode(sentence))

            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                # Keep overlap
                overlap_text = " ".join(current_chunk[-2:]) if len(current_chunk) > 1 else ""
                current_chunk = [overlap_text] if overlap_text else []
                current_tokens = len(self.tokenizer.encode(overlap_text)) if overlap_text else 0

            current_chunk.append(sentence)
            current_tokens += sentence_tokens

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _paragraph_chunk(self, text: str) -> list[str]:
        """Chunk by paragraphs."""
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]

    def _extract_headers(self, text: str) -> list[tuple[int, str]]:
        """Extract headers with their positions."""
        headers = []
        for regex in self._header_regex:
            for match in regex.finditer(text):
                headers.append((match.start(), match.group().strip()))
        return sorted(headers, key=lambda x: x[0])

    def _find_relevant_headers(
        self,
        full_text: str,
        chunk_text: str,
        headers: list[tuple[int, str]],
    ) -> list[str]:
        """Find headers that apply to a given chunk."""
        if not headers:
            return []

        # Find chunk position in original text
        chunk_start = full_text.find(chunk_text)
        if chunk_start == -1:
            return []

        # Get all headers before this chunk
        relevant = [h[1] for h in headers if h[0] < chunk_start]
        return relevant[-3:] if relevant else []  # Keep last 3 headers for context

    def _calculate_importance(self, text: str) -> float:
        """Calculate importance score based on content patterns."""
        base_score = 0.5

        for regex, weight in self._importance_regex:
            if regex.search(text):
                base_score += weight

        # Normalize to 0-1 range
        return min(1.0, base_score)

    def _detect_semantic_type(self, text: str) -> str:
        """Detect the semantic type of content."""
        text_lower = text.lower()

        if re.search(r'(?:table|figure|chart)\s+\d+', text_lower):
            return "visual_reference"
        elif re.search(r'^\s*\d+\.\s+', text, re.MULTILINE):
            return "list"
        elif re.search(r'```|def\s+\w+|class\s+\w+', text):
            return "code"
        elif re.search(r'\b(?:conclusion|summary)\b', text_lower):
            return "conclusion"
        elif re.search(r'\b(?:introduction|abstract)\b', text_lower):
            return "introduction"
        else:
            return "text"

    def _generate_chunk_id(self, source: str, index: int, text: str) -> str:
        """Generate a unique, deterministic chunk ID."""
        content = f"{source}:{index}:{text[:100]}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class ChunkerFactory:
    """Factory for creating chunkers with different configurations."""

    PRESETS = {
        "default": {
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "chunk_size": 400,
            "chunk_overlap": 50,
            "strategy": ChunkingStrategy.SEMANTIC,
        },
        "bge-m3": {
            "model_name": "BAAI/bge-m3",
            "chunk_size": 512,
            "chunk_overlap": 64,
            "strategy": ChunkingStrategy.SEMANTIC,
        },
        "openai": {
            "model_name": "Xenova/gpt-4",
            "chunk_size": 500,
            "chunk_overlap": 50,
            "strategy": ChunkingStrategy.RECURSIVE,
        },
        "long-context": {
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "chunk_size": 1024,
            "chunk_overlap": 128,
            "strategy": ChunkingStrategy.SEMANTIC,
        },
    }

    @classmethod
    def create(cls, preset: str = "default", **overrides) -> SemanticChunker:
        """Create a chunker from a preset configuration."""
        if preset not in cls.PRESETS:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(cls.PRESETS.keys())}")

        config = {**cls.PRESETS[preset], **overrides}
        return SemanticChunker(**config)


def advanced_semantic_chunking(
    text: str,
    source_file: str = "unknown",
    preset: str = "default",
) -> list[dict]:
    """
    High-level function for semantic chunking.

    This is the main entry point for the Airflow DAG.

    Args:
        text: Text to chunk
        source_file: Source filename
        preset: Chunker preset to use

    Returns:
        List of chunk dictionaries ready for storage/embedding
    """
    chunker = ChunkerFactory.create(preset)
    chunks = chunker.chunk(text, source_file=source_file)
    return [chunk.to_dict() for chunk in chunks]


if __name__ == "__main__":
    # Example usage
    sample_text = """
    # Introduction

    This document provides an overview of our financial performance in Q4 2025.
    The results show a significant 15% increase in revenue compared to the previous quarter.

    # Key Findings

    Our analysis reveals several important trends:

    1. Customer acquisition costs decreased by £2.3M
    2. The conversion rate improved to 4.2%
    3. Annual recurring revenue reached £45M

    However, we observed challenges in the European market segment.

    # Conclusion

    In summary, the quarter demonstrated strong growth despite market headwinds.
    The key recommendation is to focus on operational efficiency in Q1 2026.
    """

    chunks = advanced_semantic_chunking(sample_text, source_file="q4_report.pdf")

    for chunk in chunks:
        print(f"\n--- Chunk {chunk['metadata']['chunk_index']} ---")
        print(f"Importance: {chunk['metadata']['importance_score']:.2f}")
        print(f"Type: {chunk['metadata']['semantic_type']}")
        print(f"Text: {chunk['text'][:100]}...")
