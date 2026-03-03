from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from src.config import (
    LLM_PROVIDER, OPENAI_API_KEY, OPENAI_MODEL,
    OLLAMA_MODEL, OLLAMA_BASE_URL
)


def get_llm():
    """Returns the configured LLM based on LLM_PROVIDER env variable."""
    if LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set. Add it to your .env file.")
        return ChatOpenAI(
            model=OPENAI_MODEL,
            api_key=OPENAI_API_KEY,
            temperature=0.2,
            streaming=True,
        )
    elif LLM_PROVIDER == "ollama":
        return ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.2,
        )
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: '{LLM_PROVIDER}'. Use 'openai' or 'ollama'.")


def get_embeddings():
    """Returns the embedding model based on LLM_PROVIDER env variable."""
    if LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set. Add it to your .env file.")
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=OPENAI_API_KEY,
        )
    elif LLM_PROVIDER == "ollama":
        return OllamaEmbeddings(
            model="nomic-embed-text",
            base_url=OLLAMA_BASE_URL,
        )
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: '{LLM_PROVIDER}'. Use 'openai' or 'ollama'.")


def get_provider_info() -> dict:
    """Returns a dict with current provider info for display in UI."""
    if LLM_PROVIDER == "openai":
        return {"provider": "OpenAI", "model": OPENAI_MODEL}
    else:
        return {"provider": "Ollama (Local)", "model": OLLAMA_MODEL}
