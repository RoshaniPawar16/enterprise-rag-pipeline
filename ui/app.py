"""
Enterprise RAG Pipeline - Streamlit UI

Production-ready interface featuring:
- Chat-based Q&A with streaming responses
- Source transparency with relevance scores
- Document upload and management
- Analytics dashboard with MLflow metrics
- Feedback collection for model improvement
"""

import os
import time
import uuid
from datetime import datetime

import httpx
import streamlit as st

# ============================================
# Configuration
# ============================================

RAG_API_URL = os.getenv("RAG_API_URL", "http://localhost:8000")
API_TIMEOUT = 300  # seconds (LLM can be slow on CPU)

# Page configuration
st.set_page_config(
    page_title="Enterprise RAG Pipeline",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================
# Custom CSS
# ============================================

st.markdown("""
<style>
    /* Chat message styling - dark mode compatible */
    .user-message {
        background-color: rgba(30, 136, 229, 0.2);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #1976d2;
        color: inherit;
    }
    .assistant-message {
        background-color: rgba(76, 175, 80, 0.15);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #4caf50;
        color: inherit;
    }

    /* Source card styling */
    .source-card {
        background-color: #fafafa;
        padding: 12px;
        border-radius: 8px;
        margin: 8px 0;
        border: 1px solid #e0e0e0;
    }
    .source-card:hover {
        border-color: #1976d2;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    /* Score badge styling */
    .score-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
    }
    .score-high { background-color: #c8e6c9; color: #2e7d32; }
    .score-medium { background-color: #fff3e0; color: #e65100; }
    .score-low { background-color: #ffebee; color: #c62828; }

    /* Metric card styling */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }

    /* Citation styling */
    .citation {
        background-color: #e8f5e9;
        padding: 3px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 13px;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ============================================
# Session State Initialization
# ============================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_query_id" not in st.session_state:
    st.session_state.current_query_id = None

if "api_status" not in st.session_state:
    st.session_state.api_status = "unknown"


# ============================================
# API Helper Functions
# ============================================

def check_api_health() -> dict:
    """Check the health of the RAG API."""
    try:
        response = httpx.get(f"{RAG_API_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        return {"status": "degraded", "services": {}}
    except Exception:
        return {"status": "offline", "services": {}}


def get_metrics() -> dict:
    """Get pipeline metrics from the API."""
    try:
        response = httpx.get(f"{RAG_API_URL}/metrics", timeout=5)
        if response.status_code == 200:
            return response.json()
        return {}
    except Exception:
        return {}


def search_documents(query: str, top_k: int, search_type: str, collection: str) -> dict:
    """Execute a search query."""
    try:
        response = httpx.post(
            f"{RAG_API_URL}/search",
            json={
                "query": query,
                "top_k": top_k,
                "search_type": search_type,
                "collection": collection,
            },
            timeout=API_TIMEOUT,
        )
        if response.status_code == 200:
            return response.json()
        return {"error": f"API error: {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}


def rag_query(query: str, top_k: int, model: str, temperature: float, collection: str, verify: bool) -> dict:
    """Execute a RAG query. Falls back to search-only if LLM times out."""
    try:
        response = httpx.post(
            f"{RAG_API_URL}/query",
            json={
                "query": query,
                "top_k": top_k,
                "model": model,
                "temperature": temperature,
                "collection": collection,
                "verify": verify,
            },
            timeout=API_TIMEOUT,
        )
        if response.status_code == 200:
            return response.json()
        # If LLM fails, fallback to search-only
        if response.status_code == 500:
            return search_fallback(query, top_k, collection)
        return {"error": f"API error: {response.status_code}"}
    except httpx.TimeoutException:
        # LLM timeout - fallback to search results
        return search_fallback(query, top_k, collection)
    except Exception as e:
        return {"error": str(e)}


def search_fallback(query: str, top_k: int, collection: str) -> dict:
    """Fallback to search-only when LLM is unavailable."""
    try:
        response = httpx.post(
            f"{RAG_API_URL}/search",
            json={"query": query, "top_k": top_k, "collection": collection},
            timeout=60,
        )
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            if results:
                # Build answer from search results
                answer = "**LLM timed out. Here are the relevant search results:**\n\n"
                for i, r in enumerate(results[:3], 1):
                    text = r.get("text", "")[:200]
                    source = r.get("source_file", "Unknown")
                    answer += f"**[{i}] {source}:** {text}...\n\n"
                return {
                    "answer": answer,
                    "sources": results,
                    "citations": [],
                    "model": "search-only",
                    "provider": "fallback",
                    "faithfulness_score": None,
                    "verified": False,
                    "generation_time_ms": 0,
                    "search_time_ms": data.get("search_time_ms", 0),
                    "total_time_ms": data.get("search_time_ms", 0),
                }
        return {"error": "Search also failed"}
    except Exception as e:
        return {"error": f"Fallback failed: {e}"}


def upload_document(file) -> dict:
    """Upload a document to the pipeline."""
    try:
        files = {"file": (file.name, file.getvalue(), file.type)}
        response = httpx.post(
            f"{RAG_API_URL}/upload",
            files=files,
            timeout=60,
        )
        if response.status_code == 200:
            return response.json()
        return {"error": f"Upload failed: {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}


def process_document(file_id: str) -> dict:
    """Process an uploaded document (extract, chunk, embed, index)."""
    try:
        response = httpx.post(
            f"{RAG_API_URL}/process/{file_id}",
            timeout=180,  # Processing can take time
        )
        if response.status_code == 200:
            return response.json()
        return {"error": f"Processing failed: {response.status_code}"}
    except httpx.TimeoutException:
        return {"error": "Processing timed out - document may still be processing in background"}
    except Exception as e:
        return {"error": str(e)}


def submit_feedback(query_id: str, feedback: str, comment: str = None) -> dict:
    """Submit feedback for a query."""
    try:
        response = httpx.post(
            f"{RAG_API_URL}/feedback",
            json={
                "query_id": query_id,
                "feedback": feedback,
                "comment": comment,
            },
            timeout=10,
        )
        if response.status_code == 200:
            return response.json()
        return {"error": "Feedback submission failed"}
    except Exception:
        return {"error": "Could not connect to API"}


def get_collections() -> list:
    """Get available collections."""
    try:
        response = httpx.get(f"{RAG_API_URL}/collections", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return [c["name"] for c in data.get("collections", [])]
        return ["enterprise_docs"]
    except Exception:
        return ["enterprise_docs"]


# ============================================
# UI Components
# ============================================

def render_score_badge(score: float) -> str:
    """Render a colored score badge."""
    if score >= 0.8:
        return f'<span class="score-badge score-high">{score:.2f}</span>'
    elif score >= 0.5:
        return f'<span class="score-badge score-medium">{score:.2f}</span>'
    else:
        return f'<span class="score-badge score-low">{score:.2f}</span>'


def render_source_card(source: dict, index: int):
    """Render a source document card."""
    score = source.get("score", 0)
    source_file = source.get("source_file", "Unknown")
    page = source.get("page_number")
    text = source.get("text", "")[:300]
    semantic_type = source.get("semantic_type", "text")

    page_str = f" (Page {page})" if page else ""

    with st.expander(f"📄 Source {index}: {source_file}{page_str} - Score: {score:.3f}"):
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown(f"**File:** `{source_file}`")
        with col2:
            st.markdown(f"**Type:** `{semantic_type}`")
        with col3:
            if source.get("rrf_score"):
                st.markdown(f"**RRF:** `{source['rrf_score']:.4f}`")

        st.markdown("---")
        st.markdown(f"*{text}...*")

        # Show additional scores if available
        if source.get("dense_score") or source.get("sparse_score"):
            st.markdown("**Score Breakdown:**")
            cols = st.columns(2)
            if source.get("dense_score"):
                cols[0].metric("Semantic", f"{source['dense_score']:.3f}")
            if source.get("sparse_score"):
                cols[1].metric("Keyword", f"{source['sparse_score']:.3f}")


def render_chat_message(role: str, content: str, metadata: dict = None):
    """Render a chat message with metadata."""
    if role == "user":
        st.markdown(f'<div class="user-message">🧑 **You:** {content}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="assistant-message">🤖 **Assistant:** {content}</div>', unsafe_allow_html=True)

        if metadata:
            cols = st.columns(4)
            if metadata.get("faithfulness_score") is not None:
                cols[0].metric("Faithfulness", f"{metadata['faithfulness_score']:.2f}")
            if metadata.get("search_time_ms"):
                cols[1].metric("Search", f"{metadata['search_time_ms']:.0f}ms")
            if metadata.get("generation_time_ms"):
                cols[2].metric("Generation", f"{metadata['generation_time_ms']:.0f}ms")
            if metadata.get("total_time_ms"):
                cols[3].metric("Total", f"{metadata['total_time_ms']:.0f}ms")


# ============================================
# Main Application
# ============================================

def main():
    # Header
    st.title("🔍 Enterprise RAG Pipeline")
    st.markdown("*Intelligent document search and question answering with source transparency*")

    # ============================================
    # Sidebar
    # ============================================

    with st.sidebar:
        st.header("⚙️ Configuration")

        # Get available collections
        collections = get_collections()

        # Collection selector
        collection = st.selectbox(
            "Vector Collection",
            collections,
            index=0,
        )

        st.divider()

        # Search settings
        st.subheader("🔎 Search Settings")
        top_k = st.slider("Number of sources", 1, 20, 5)
        search_type = st.selectbox(
            "Search Type",
            ["hybrid", "dense", "sparse", "multimodal"],
            index=0,
            format_func=lambda x: {
                "hybrid": "Hybrid (Recommended)",
                "dense": "Semantic Only",
                "sparse": "Keyword Only",
                "multimodal": "Multimodal (Images)",
            }[x]
        )

        st.divider()

        # LLM settings
        st.subheader("🤖 LLM Settings")
        model = st.selectbox(
            "Model",
            ["llama3.1:8b", "llama3.1:70b", "gpt-4-turbo", "gpt-4o"],
            index=0,
        )
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05)
        verify_answer = st.checkbox("Verify answer faithfulness", value=True)

        st.divider()

        # System status
        st.subheader("📊 System Status")
        health = check_api_health()

        if health.get("status") == "healthy":
            st.success("✅ All Systems Operational")
        elif health.get("status") == "degraded":
            st.warning("⚠️ System Degraded")
        else:
            st.error("❌ API Offline")

        # Service status details
        services = health.get("services", {})
        if services:
            for service, status in services.items():
                icon = "✅" if status == "healthy" else "⚠️" if status == "not_available" else "❌"
                st.text(f"{icon} {service.capitalize()}")

        st.divider()

        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

    # ============================================
    # Main Content Tabs
    # ============================================

    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📄 Documents", "📈 Analytics"])

    # ============================================
    # Chat Tab
    # ============================================

    with tab1:
        # Display chat history
        for message in st.session_state.messages:
            render_chat_message(
                message["role"],
                message["content"],
                message.get("metadata"),
            )

            # Show sources for assistant messages
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("📚 View Sources", expanded=False):
                    for i, source in enumerate(message["sources"], 1):
                        render_source_card(source, i)

                # Feedback buttons
                if message.get("query_id"):
                    col1, col2, col3 = st.columns([1, 1, 4])
                    with col1:
                        if st.button("👍", key=f"pos_{message['query_id']}"):
                            submit_feedback(message["query_id"], "positive")
                            st.success("Thanks for your feedback!")
                    with col2:
                        if st.button("👎", key=f"neg_{message['query_id']}"):
                            submit_feedback(message["query_id"], "negative")
                            st.info("Feedback recorded. We'll improve!")

        # Chat input
        query = st.chat_input("Ask a question about your documents...")

        if query:
            # Add user message
            st.session_state.messages.append({
                "role": "user",
                "content": query,
            })

            # Generate response
            with st.spinner("🔍 Searching and generating answer..."):
                query_id = str(uuid.uuid4())

                response = rag_query(
                    query=query,
                    top_k=top_k,
                    model=model,
                    temperature=temperature,
                    collection=collection,
                    verify=verify_answer,
                )

                if "error" in response:
                    st.error(f"Error: {response['error']}")
                else:
                    # Add assistant message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response.get("answer", "No answer generated."),
                        "query_id": query_id,
                        "sources": response.get("sources", []),
                        "citations": response.get("citations", []),
                        "metadata": {
                            "faithfulness_score": response.get("faithfulness_score"),
                            "search_time_ms": response.get("search_time_ms"),
                            "generation_time_ms": response.get("generation_time_ms"),
                            "total_time_ms": response.get("total_time_ms"),
                            "model": response.get("model"),
                            "verified": response.get("verified"),
                        },
                    })

            st.rerun()

    # ============================================
    # Documents Tab
    # ============================================

    with tab2:
        st.subheader("📄 Document Management")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### Upload New Documents")
            uploaded_files = st.file_uploader(
                "Drop files here or click to browse",
                type=["pdf", "docx", "txt", "md"],
                accept_multiple_files=True,
                help="Supported formats: PDF, DOCX, TXT, MD",
            )

            if uploaded_files:
                st.markdown("**Selected files:**")
                for file in uploaded_files:
                    size_kb = file.size / 1024
                    st.text(f"📎 {file.name} ({size_kb:.1f} KB)")

                if st.button("📤 Upload & Process", type="primary"):
                    progress = st.progress(0)
                    total_steps = len(uploaded_files) * 2  # Upload + Process
                    step = 0

                    for file in uploaded_files:
                        # Step 1: Upload
                        with st.spinner(f"Uploading {file.name}..."):
                            result = upload_document(file)

                        step += 1
                        progress.progress(step / total_steps)

                        if "error" in result:
                            st.error(f"Failed to upload {file.name}: {result['error']}")
                            continue

                        st.info(f"✅ {file.name} uploaded, now processing...")
                        file_id = result.get("file_id")

                        # Step 2: Process (extract, chunk, embed, index)
                        if file_id:
                            with st.spinner(f"Processing {file.name} (this may take a minute)..."):
                                process_result = process_document(file_id)

                            if "error" in process_result:
                                st.warning(f"⚠️ {file.name}: {process_result['error']}")
                            else:
                                chunks = process_result.get("chunks_created", 0)
                                st.success(f"✅ {file.name} processed: {chunks} chunks indexed!")

                        step += 1
                        progress.progress(step / total_steps)

                    st.balloons()

        with col2:
            st.markdown("### Collection Info")
            try:
                response = httpx.get(f"{RAG_API_URL}/collections/{collection}", timeout=5)
                if response.status_code == 200:
                    info = response.json()
                    st.metric("Total Vectors", f"{info.get('vectors_count', 0):,}")
                    st.metric("Total Points", f"{info.get('points_count', 0):,}")
                    st.metric("Status", info.get("status", "Unknown"))
                else:
                    st.info("Collection info not available")
            except Exception:
                st.info("Could not fetch collection info")

    # ============================================
    # Analytics Tab
    # ============================================

    with tab3:
        st.subheader("📈 Pipeline Analytics")

        metrics = get_metrics()

        if metrics:
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Collections",
                    len(metrics.get("collections", [])),
                )
            with col2:
                st.metric(
                    "Total Chunks",
                    f"{metrics.get('total_chunks', 0):,}",
                )
            with col3:
                st.metric(
                    "Queries Today",
                    metrics.get("queries_today", 0),
                )
            with col4:
                avg_time = metrics.get("avg_response_time_ms", 0)
                st.metric(
                    "Avg Response Time",
                    f"{avg_time:.0f}ms" if avg_time else "N/A",
                )

            st.divider()

            # Additional metrics
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### System Configuration")
                st.json({
                    "LLM Provider": metrics.get("llm_provider", "Unknown"),
                    "Embedding Model": metrics.get("embedding_model", "Unknown"),
                    "Avg Faithfulness": f"{metrics.get('avg_faithfulness_score', 0):.2f}",
                })

            with col2:
                st.markdown("### Collections")
                collections_data = metrics.get("collections", [])
                if collections_data:
                    for coll in collections_data:
                        st.markdown(f"""
                        **{coll['name']}**
                        - Vectors: {coll.get('vectors_count', 0):,}
                        - Points: {coll.get('points_count', 0):,}
                        - Status: {coll.get('status', 'Unknown')}
                        """)
                else:
                    st.info("No collections found")
        else:
            st.warning("Could not fetch metrics from API")

            # Show placeholder metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Collections", "—")
            col2.metric("Total Chunks", "—")
            col3.metric("Queries Today", "—")
            col4.metric("Avg Response", "—")


if __name__ == "__main__":
    main()
