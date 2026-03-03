import os
import tempfile
import streamlit as st
from pathlib import Path

from src.config import PAGE_TITLE, PAGE_ICON, LLM_PROVIDER
from src.document_processor import (
    get_vectorstore,
    ingest_pdf,
    get_collection_stats,
    delete_paper,
    get_file_hash,
    get_indexed_files,
)
from src.rag_chain import build_streaming_chain, format_docs_with_metadata
from src.llm_provider import get_provider_info


# --- Page Config 
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon="book",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS 
st.markdown("""
<style>
    .main-header { font-size: 2rem; font-weight: 700; margin-bottom: 0; }
    .sub-header { color: #666; margin-top: 0; margin-bottom: 1.5rem; }
    .source-card {
        background: #1e2530;
        border-left: 4px solid #4da6ff;
        padding: 0.75rem 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
        font-size: 0.85rem;
        color: #e0e0e0;
    }
    .source-card strong {
        color: #4da6ff;
    }
    .source-card small {
        color: #aaaaaa;
    }
    .stat-box {
        background: #e8f4fd;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        text-align: center;
    }
    .provider-badge {
        background: #2a3441;
        border: 1px solid #4da6ff;
        border-radius: 20px;
        padding: 0.2rem 0.8rem;
        font-size: 0.8rem;
        color: #4da6ff;
        display: inline-block;
    }
    .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# --- Session State 
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chain" not in st.session_state:
    st.session_state.chain = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None


@st.cache_resource
def load_vectorstore():
    return get_vectorstore()


def initialize_chain(vs):
    chain, retriever = build_streaming_chain(vs)
    st.session_state.chain = chain
    st.session_state.retriever = retriever


# --- Sidebar 
with st.sidebar:
    st.markdown("## Academic Paper Assistant")

    # Provider info
    info = get_provider_info()
    st.markdown(
        f'<span class="provider-badge">{info["provider"]}  {info["model"]}</span>',
        unsafe_allow_html=True,
    )
    st.divider()

    # Load vectorstore
    if st.session_state.vectorstore is None:
        with st.spinner("Loading vector store..."):
            try:
                st.session_state.vectorstore = load_vectorstore()
                initialize_chain(st.session_state.vectorstore)
            except Exception as e:
                st.error(f"Failed to load vector store: {e}")

    vs = st.session_state.vectorstore

    # Stats
    if vs:
        stats = get_collection_stats(vs)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Papers", stats["indexed_papers"])
        with col2:
            st.metric("Chunks", stats["total_chunks"])
        st.divider()

    # Upload PDFs
    st.markdown("### Upload Papers")
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files and vs:
        if st.button("Index Papers", use_container_width=True, type="primary"):
            progress = st.progress(0)
            status_placeholder = st.empty()
            for i, uploaded_file in enumerate(uploaded_files):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                try:
                    status_placeholder.info(f"Processing: {uploaded_file.name}")
                    result = ingest_pdf(tmp_path, vs, original_name=uploaded_file.name)
                    if result["status"] == "success":
                        st.success(f"{uploaded_file.name}: {result['chunks']} chunks indexed")
                    elif result["status"] == "skipped":
                        st.warning(f"{uploaded_file.name}: Already indexed")
                finally:
                    os.unlink(tmp_path)
                progress.progress((i + 1) / len(uploaded_files))

            status_placeholder.empty()
            initialize_chain(vs)
            st.rerun()

    st.divider()

    # Indexed papers list
    if vs:
        stats = get_collection_stats(vs)
        if stats["paper_names"]:
            st.markdown("### Indexed Papers")
            for paper in stats["paper_names"]:
                col_p, col_d = st.columns([4, 1])
                with col_p:
                    st.markdown(f"`{paper}`")
                with col_d:
                    if st.button("x", key=f"del_{paper}", help=f"Remove {paper}"):
                        delete_paper(paper, vs)
                        st.success(f"Removed {paper}")
                        st.rerun()
        else:
            st.info("No papers indexed yet. Upload PDFs above.")

    st.divider()

    # Clear chat
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# --- Main Chat Interface 
st.markdown('<p class="main-header">Academic Paper Assistant</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Ask questions about your indexed research papers and get cited answers.</p>',
    unsafe_allow_html=True,
)

vs = st.session_state.vectorstore
stats = get_collection_stats(vs) if vs else {"indexed_papers": 0}

if stats["indexed_papers"] == 0:
    st.info(
        "Welcome! Upload PDF papers using the sidebar to get started. "
        "Once indexed, you can ask questions and get answers with citations."
    )

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources", expanded=False):
                for src in message["sources"]:
                    st.markdown(
                        f'<div class="source-card">'
                        f'<strong>{src["source"]}</strong>  Page {src["page"]}<br>'
                        f'<small>{src["content"][:300]}...</small>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

# Chat input
if prompt := st.chat_input(
    "Ask anything about your papers...",
    disabled=(stats["indexed_papers"] == 0 or st.session_state.chain is None),
):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get AI response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        try:
            # Stream the response
            for chunk in st.session_state.chain.stream(prompt):
                full_response += chunk
                response_placeholder.markdown(full_response + "")
            response_placeholder.markdown(full_response)

            # Get sources for this query
            sources = []
            if st.session_state.retriever:
                source_docs = st.session_state.retriever.invoke(prompt)
                sources = format_docs_with_metadata(source_docs)
                if sources:
                    with st.expander("Sources", expanded=False):
                        for src in sources:
                            st.markdown(
                                f'<div class="source-card">'
                                f'<strong>{src["source"]}</strong>  Page {src["page"]}<br>'
                                f'<small>{src["content"][:300]}...</small>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

        except Exception as e:
            full_response = f"Error generating response: {str(e)}"
            response_placeholder.error(full_response)
            sources = []

    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "sources": sources,
    })