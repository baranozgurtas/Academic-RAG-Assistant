import os
from dotenv import load_dotenv

load_dotenv()

# LLM Provider
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()

# OpenAI settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# Ollama settings
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ChromaDB settings
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./vectorstore")
COLLECTION_NAME = "academic_papers"

# Chunking settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

# Retrieval settings
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", 5))

# Streamlit page config
PAGE_TITLE = "Academic Paper Assistant"
PAGE_ICON = "book"
