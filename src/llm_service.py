"""
LLM Service for Enterprise RAG Pipeline

This module provides a unified interface for LLM inference with:
- Multiple backend support (Ollama, OpenAI, Azure OpenAI)
- Answer verification and faithfulness scoring
- Structured output with citations
- Streaming support for real-time responses

Key Features:
- Verification Loop: AI checks its answer against retrieved context
- Citation Generation: Automatic source attribution
- Faithfulness Scoring: Measures groundedness of responses
"""

import json
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Iterator

import structlog

logger = structlog.get_logger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers."""
    OLLAMA = "ollama"
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    ANTHROPIC = "anthropic"


@dataclass
class Citation:
    """A citation reference to a source document."""
    index: int
    source_file: str
    page_number: int | None
    text_snippet: str
    relevance_score: float


@dataclass
class LLMResponse:
    """Complete LLM response with metadata."""
    answer: str
    citations: list[Citation] = field(default_factory=list)
    model: str = ""
    provider: str = ""
    tokens_used: int = 0
    generation_time_ms: float = 0.0
    faithfulness_score: float | None = None
    verified: bool = False

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "citations": [
                {
                    "index": c.index,
                    "source_file": c.source_file,
                    "page_number": c.page_number,
                    "text_snippet": c.text_snippet[:200],
                    "relevance_score": c.relevance_score,
                }
                for c in self.citations
            ],
            "model": self.model,
            "provider": self.provider,
            "tokens_used": self.tokens_used,
            "generation_time_ms": self.generation_time_ms,
            "faithfulness_score": self.faithfulness_score,
            "verified": self.verified,
        }


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response from the LLM."""
        pass

    @abstractmethod
    async def agenerate(self, prompt: str, **kwargs) -> str:
        """Async generate a response from the LLM."""
        pass

    @abstractmethod
    def stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """Stream a response from the LLM."""
        pass


class OllamaBackend(LLMBackend):
    """Ollama backend for local LLM inference."""

    def __init__(
        self,
        model: str = "llama3.1:8b",
        host: str | None = None,
    ):
        self.model = model
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")

        logger.info("Initializing Ollama backend", model=model, host=self.host)

    def generate(self, prompt: str, **kwargs) -> str:
        import ollama

        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            options={
                "temperature": kwargs.get("temperature", 0.1),
                "num_predict": kwargs.get("max_tokens", 1024),
            },
        )
        return response["response"]

    async def agenerate(self, prompt: str, **kwargs) -> str:
        import ollama

        response = await ollama.AsyncClient(host=self.host).generate(
            model=self.model,
            prompt=prompt,
            options={
                "temperature": kwargs.get("temperature", 0.1),
                "num_predict": kwargs.get("max_tokens", 1024),
            },
        )
        return response["response"]

    def stream(self, prompt: str, **kwargs) -> Iterator[str]:
        import ollama

        stream = ollama.generate(
            model=self.model,
            prompt=prompt,
            stream=True,
            options={
                "temperature": kwargs.get("temperature", 0.1),
                "num_predict": kwargs.get("max_tokens", 1024),
            },
        )
        for chunk in stream:
            yield chunk["response"]


class OpenAIBackend(LLMBackend):
    """OpenAI backend for cloud LLM inference."""

    def __init__(
        self,
        model: str = "gpt-4-turbo",
        api_key: str | None = None,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError("OpenAI API key required")

        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key)

        logger.info("Initializing OpenAI backend", model=model)

    def generate(self, prompt: str, **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", 0.1),
            max_tokens=kwargs.get("max_tokens", 1024),
        )
        return response.choices[0].message.content

    async def agenerate(self, prompt: str, **kwargs) -> str:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=self.api_key)

        response = await client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", 0.1),
            max_tokens=kwargs.get("max_tokens", 1024),
        )
        return response.choices[0].message.content

    def stream(self, prompt: str, **kwargs) -> Iterator[str]:
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", 0.1),
            max_tokens=kwargs.get("max_tokens", 1024),
            stream=True,
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class AzureOpenAIBackend(LLMBackend):
    """Azure OpenAI backend for enterprise cloud inference."""

    def __init__(
        self,
        deployment: str | None = None,
        api_key: str | None = None,
        endpoint: str | None = None,
    ):
        self.deployment = deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")
        self.api_key = api_key or os.getenv("AZURE_OPENAI_KEY")
        self.endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")

        if not all([self.api_key, self.endpoint]):
            raise ValueError("Azure OpenAI credentials required")

        from openai import AzureOpenAI
        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version="2024-02-01",
            azure_endpoint=self.endpoint,
        )

        logger.info("Initializing Azure OpenAI backend", deployment=self.deployment)

    def generate(self, prompt: str, **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", 0.1),
            max_tokens=kwargs.get("max_tokens", 1024),
        )
        return response.choices[0].message.content

    async def agenerate(self, prompt: str, **kwargs) -> str:
        from openai import AsyncAzureOpenAI
        client = AsyncAzureOpenAI(
            api_key=self.api_key,
            api_version="2024-02-01",
            azure_endpoint=self.endpoint,
        )

        response = await client.chat.completions.create(
            model=self.deployment,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", 0.1),
            max_tokens=kwargs.get("max_tokens", 1024),
        )
        return response.choices[0].message.content

    def stream(self, prompt: str, **kwargs) -> Iterator[str]:
        stream = self.client.chat.completions.create(
            model=self.deployment,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", 0.1),
            max_tokens=kwargs.get("max_tokens", 1024),
            stream=True,
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class RAGService:
    """
    Production RAG service with verification loop.

    Implements the complete RAG pipeline:
    1. Retrieve relevant context
    2. Generate answer with citations
    3. Verify answer faithfulness
    4. Return structured response
    """

    # System prompt for grounded generation
    SYSTEM_PROMPT = """You are a helpful assistant that answers questions based ONLY on the provided context.

IMPORTANT RULES:
1. Answer ONLY using information from the context below
2. If the answer is not in the context, say "I cannot find this information in the provided documents"
3. Always cite your sources using [Source N] format where N is the source number
4. Be concise and direct in your answers
5. If multiple sources support your answer, cite all of them

CONTEXT:
{context}

USER QUESTION: {question}

Provide a clear, grounded answer with citations:"""

    # Verification prompt
    VERIFICATION_PROMPT = """Evaluate if the following answer is fully supported by the given context.

CONTEXT:
{context}

ANSWER:
{answer}

Score the answer from 0.0 to 1.0 based on:
- 1.0: Completely supported by context, all claims have citations
- 0.7-0.9: Mostly supported, minor unsupported details
- 0.4-0.6: Partially supported, some claims lack evidence
- 0.1-0.3: Poorly supported, most claims are not in context
- 0.0: Not supported or contradicts context

Return ONLY a JSON object: {{"score": <float>, "reasoning": "<brief explanation>"}}"""

    def __init__(
        self,
        provider: LLMProvider = LLMProvider.OLLAMA,
        model: str | None = None,
        enable_verification: bool = True,
        **kwargs,
    ):
        """
        Initialize the RAG service.

        Args:
            provider: LLM provider to use
            model: Model name (provider-specific)
            enable_verification: Enable answer verification loop
            **kwargs: Additional provider-specific arguments
        """
        self.provider = provider
        self.enable_verification = enable_verification

        # Initialize backend
        if provider == LLMProvider.OLLAMA:
            self.backend = OllamaBackend(
                model=model or "llama3.1:8b",
                **kwargs,
            )
        elif provider == LLMProvider.OPENAI:
            self.backend = OpenAIBackend(
                model=model or "gpt-4-turbo",
                **kwargs,
            )
        elif provider == LLMProvider.AZURE_OPENAI:
            self.backend = AzureOpenAIBackend(
                deployment=model,
                **kwargs,
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        self.model = model or "default"
        logger.info(
            "RAG Service initialized",
            provider=provider.value,
            model=self.model,
            verification=enable_verification,
        )

    def _format_context(self, sources: list[dict]) -> tuple[str, list[Citation]]:
        """Format retrieved sources into context string and citations."""
        context_parts = []
        citations = []

        for i, source in enumerate(sources, 1):
            source_file = source.get("metadata", {}).get("source_file", "Unknown")
            page = source.get("metadata", {}).get("page_number")
            text = source.get("text", "")
            score = source.get("score", 0.0)

            # Format context block
            page_str = f", Page {page}" if page else ""
            context_parts.append(f"[Source {i}] ({source_file}{page_str}):\n{text}\n")

            # Create citation
            citations.append(Citation(
                index=i,
                source_file=source_file,
                page_number=page,
                text_snippet=text[:500],
                relevance_score=score,
            ))

        return "\n".join(context_parts), citations

    def _extract_used_citations(self, answer: str, all_citations: list[Citation]) -> list[Citation]:
        """Extract only the citations actually used in the answer."""
        used = []
        for citation in all_citations:
            pattern = rf'\[Source\s*{citation.index}\]'
            if re.search(pattern, answer, re.IGNORECASE):
                used.append(citation)
        return used if used else all_citations[:3]  # Fallback to top 3

    def generate(
        self,
        query: str,
        sources: list[dict],
        temperature: float = 0.1,
        verify: bool | None = None,
    ) -> LLMResponse:
        """
        Generate a grounded answer with citations.

        Args:
            query: User question
            sources: Retrieved source documents
            temperature: Generation temperature
            verify: Override verification setting

        Returns:
            LLMResponse with answer and metadata
        """
        start_time = time.time()

        # Format context
        context, all_citations = self._format_context(sources)

        # Generate answer
        prompt = self.SYSTEM_PROMPT.format(context=context, question=query)
        answer = self.backend.generate(prompt, temperature=temperature)

        # Extract used citations
        used_citations = self._extract_used_citations(answer, all_citations)

        generation_time = (time.time() - start_time) * 1000

        # Build response
        response = LLMResponse(
            answer=answer,
            citations=used_citations,
            model=self.model,
            provider=self.provider.value,
            generation_time_ms=generation_time,
        )

        # Verify if enabled
        should_verify = verify if verify is not None else self.enable_verification
        if should_verify and sources:
            response.faithfulness_score = self._verify_answer(context, answer)
            response.verified = True

        logger.info(
            "Answer generated",
            query=query[:50],
            citations=len(used_citations),
            time_ms=f"{generation_time:.1f}",
            faithfulness=response.faithfulness_score,
        )

        return response

    async def agenerate(
        self,
        query: str,
        sources: list[dict],
        temperature: float = 0.1,
        verify: bool | None = None,
    ) -> LLMResponse:
        """Async version of generate."""
        start_time = time.time()

        context, all_citations = self._format_context(sources)
        prompt = self.SYSTEM_PROMPT.format(context=context, question=query)
        answer = await self.backend.agenerate(prompt, temperature=temperature)

        used_citations = self._extract_used_citations(answer, all_citations)
        generation_time = (time.time() - start_time) * 1000

        response = LLMResponse(
            answer=answer,
            citations=used_citations,
            model=self.model,
            provider=self.provider.value,
            generation_time_ms=generation_time,
        )

        should_verify = verify if verify is not None else self.enable_verification
        if should_verify and sources:
            response.faithfulness_score = self._verify_answer(context, answer)
            response.verified = True

        return response

    def _verify_answer(self, context: str, answer: str) -> float:
        """
        Verify answer faithfulness against context.

        Uses a separate LLM call to evaluate groundedness.
        """
        try:
            prompt = self.VERIFICATION_PROMPT.format(context=context, answer=answer)
            result = self.backend.generate(prompt, temperature=0.0)

            # Parse JSON response
            json_match = re.search(r'\{[^}]+\}', result)
            if json_match:
                parsed = json.loads(json_match.group())
                score = float(parsed.get("score", 0.5))
                logger.debug(
                    "Verification complete",
                    score=score,
                    reasoning=parsed.get("reasoning", "")[:100],
                )
                return min(1.0, max(0.0, score))

        except Exception as e:
            logger.warning("Verification failed", error=str(e))

        return 0.5  # Default score on failure

    def stream_generate(
        self,
        query: str,
        sources: list[dict],
        temperature: float = 0.1,
    ) -> Iterator[str]:
        """
        Stream generate an answer token by token.

        Note: Streaming doesn't support verification.
        """
        context, _ = self._format_context(sources)
        prompt = self.SYSTEM_PROMPT.format(context=context, question=query)

        for token in self.backend.stream(prompt, temperature=temperature):
            yield token


def create_rag_service(
    provider: str = "ollama",
    model: str | None = None,
    **kwargs,
) -> RAGService:
    """
    Factory function to create a RAG service.

    Args:
        provider: "ollama", "openai", or "azure_openai"
        model: Model name
        **kwargs: Provider-specific arguments

    Returns:
        Configured RAGService instance
    """
    provider_enum = LLMProvider(provider.lower())
    return RAGService(provider=provider_enum, model=model, **kwargs)


if __name__ == "__main__":
    # Example usage
    service = create_rag_service(provider="ollama", model="llama3.1:8b")

    # Mock sources
    sources = [
        {
            "text": "The company reported revenue of £45M in Q4 2025, representing a 15% increase from Q3.",
            "score": 0.95,
            "metadata": {"source_file": "quarterly_report.pdf", "page_number": 3},
        },
        {
            "text": "Operating expenses decreased by 8% due to cost optimization initiatives.",
            "score": 0.87,
            "metadata": {"source_file": "quarterly_report.pdf", "page_number": 5},
        },
    ]

    response = service.generate(
        query="What was the Q4 2025 revenue and how did it compare to Q3?",
        sources=sources,
    )

    print("\n=== RAG Response ===")
    print(f"Answer: {response.answer}")
    print(f"\nCitations used: {len(response.citations)}")
    for c in response.citations:
        print(f"  [{c.index}] {c.source_file} (Page {c.page_number})")
    print(f"\nFaithfulness Score: {response.faithfulness_score}")
    print(f"Generation Time: {response.generation_time_ms:.1f}ms")
