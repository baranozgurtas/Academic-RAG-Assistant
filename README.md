# Academic Paper Assistant

> An end-to-end RAG (Retrieval-Augmented Generation) system that transforms academic PDFs into an interactive, citation-aware research assistant.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.2-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?logo=streamlit)

## Overview

Inspired by managing hundreds of research papers during graduate studies at UZH/ETH Zurich, this tool:

- Uploads multiple academic PDFs at once
- Asks natural language questions across all papers
- Returns cited answers with exact paper name and page references
- Switches between OpenAI GPT-4o or local Llama 3 via Ollama

## Architecture
```
PDF Upload → Text Chunking → ChromaDB Vector Store
                ↓
    Streamlit UI ← RAG Chain ← MMR Retrieval
                ↓
        OpenAI GPT-4o OR Ollama Llama 3
```

**Key Features:**
- MMR retrieval for diverse, non-redundant context
- MD5 hashing prevents duplicate indexing
- Streaming responses for real-time UX
- Provider abstraction — swap LLMs with one env variable

## Quick Start
```bash
# Install
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Configure
cp .env.example .env
# Add OPENAI_API_KEY or set LLM_PROVIDER=ollama

# Run
streamlit run app.py
```

## Using Ollama (Free & Local)
```bash
# Install Ollama from https://ollama.ai
ollama pull llama3.2
ollama pull nomic-embed-text

# Set in .env:
LLM_PROVIDER=ollama
```

## Project Structure
```
academic-paper-assistant/
├── app.py                     # Streamlit UI & chat
├── evaluate.py                # Auto-evaluation script
├── requirements.txt
├── .env
├── src/
│   ├── config.py              # Centralized config
│   ├── llm_provider.py        # OpenAI/Ollama abstraction
│   ├── document_processor.py  # PDF processing + ChromaDB
│   └── rag_chain.py           # RAG pipeline
└── .streamlit/
    └── config.toml            # UI theme
```

## Evaluation & Performance

### Chunking Strategy

After testing chunk sizes of 500, 1000, and 1500 tokens:
- **Chosen: 1000 tokens with 200 overlap**
- Rationale: Best balance between retrieval recall and context size
- 500 tokens: Higher precision but missed context
- 1500 tokens: Lower retrieval quality due to mixed topics

### Retrieval Quality

- **Recall@8**: ~75-80% (relevant papers appear in top-8 chunks)
- **Citation Rate**: ~70-90% (answers include source references)
- **Cross-paper queries**: Successfully retrieves from multiple papers
- **MMR benefit**: Improved paper diversity by ~20% vs pure similarity search

Run `python evaluate.py` to measure performance on your own paper collection.

### Performance

| Metric | OpenAI GPT-4o | Ollama Llama 3.2 |
|--------|---------------|------------------|
| Answer Quality | High | Medium |
| Latency (avg) | ~2-3s | ~6-10s |
| Cost | $0.01/query | Free |

### Known Limitations

1. **Source-Answer Mismatch**: Sources fetched post-streaming may occasionally differ from context used in answer
2. **Single-language**: Optimized for English academic papers
3. **No re-ranking**: Uses MMR only; hybrid retrieval could improve precision
4. **Context window**: Limited to top-K chunks; very long papers may miss sections

### Scaling Notes

- **Storage**: ~1-2MB per indexed paper
- **Memory**: ChromaDB loads full index; 100+ papers may need optimization
- **Recommended**: 5-20 papers per index for optimal performance

## Tech Stack

- LLM: OpenAI GPT-4o / Llama 3 (Ollama)
- Embeddings: text-embedding-3-small / nomic-embed-text
- Vector DB: ChromaDB
- Framework: LangChain
- UI: Streamlit

## License

MIT
