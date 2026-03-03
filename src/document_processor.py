import os
import hashlib
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.schema import Document

from src.config import CHUNK_SIZE, CHUNK_OVERLAP, CHROMA_PERSIST_DIR, COLLECTION_NAME, RETRIEVAL_K
from src.llm_provider import get_embeddings


def get_file_hash(file_path: str) -> str:
    """Generate MD5 hash of a file to detect duplicates."""
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def load_and_split_pdf(file_path: str, original_name: str = None) -> List[Document]:
    """Load a PDF and split it into chunks."""
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)

    # Use original filename if provided, otherwise fall back to path stem
    file_name = Path(original_name).stem if original_name else Path(file_path).stem
    for chunk in chunks:
        chunk.metadata["source_file"] = file_name
        chunk.metadata["file_hash"] = get_file_hash(file_path)

    return chunks


def get_vectorstore() -> Chroma:
    """Load or create the ChromaDB vector store."""
    embeddings = get_embeddings()
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    return vectorstore


def get_indexed_files(vectorstore: Chroma) -> set:
    """Return set of already indexed file hashes to avoid re-indexing."""
    try:
        collection = vectorstore._collection
        results = collection.get(include=["metadatas"])
        hashes = {m.get("file_hash") for m in results["metadatas"] if m.get("file_hash")}
        return hashes
    except Exception:
        return set()


def ingest_pdf(file_path: str, vectorstore: Chroma, original_name: str = None) -> dict:
    """
    Ingest a single PDF into the vector store.
    Returns a status dict with info about the operation.
    """
    file_hash = get_file_hash(file_path)
    indexed_hashes = get_indexed_files(vectorstore)

    if file_hash in indexed_hashes:
        return {
            "status": "skipped",
            "message": f"Already indexed.",
            "chunks": 0,
        }

    chunks = load_and_split_pdf(file_path, original_name=original_name)
    vectorstore.add_documents(chunks)

    return {
        "status": "success",
        "message": f"Successfully indexed {len(chunks)} chunks.",
        "chunks": len(chunks),
    }


def ingest_multiple_pdfs(file_paths: List[str], vectorstore: Chroma) -> List[dict]:
    """Ingest multiple PDFs and return status for each."""
    results = []
    for path in file_paths:
        file_name = Path(path).name
        result = ingest_pdf(path, vectorstore)
        result["file"] = file_name
        results.append(result)
    return results


def get_retriever(vectorstore: Chroma):
    """Return a retriever from the vector store."""
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": RETRIEVAL_K, "fetch_k": RETRIEVAL_K * 5, "lambda_mult": 0.7},
    )


def get_collection_stats(vectorstore: Chroma) -> dict:
    """Return stats about the current collection."""
    try:
        collection = vectorstore._collection
        results = collection.get(include=["metadatas"])
        total_chunks = len(results["metadatas"])
        unique_files = {m.get("source_file") for m in results["metadatas"] if m.get("source_file")}
        return {
            "total_chunks": total_chunks,
            "indexed_papers": len(unique_files),
            "paper_names": sorted(unique_files),
        }
    except Exception:
        return {"total_chunks": 0, "indexed_papers": 0, "paper_names": []}


def delete_paper(source_file: str, vectorstore: Chroma) -> bool:
    """Delete all chunks belonging to a specific paper."""
    try:
        collection = vectorstore._collection
        results = collection.get(include=["metadatas"])
        ids_to_delete = [
            results["ids"][i]
            for i, m in enumerate(results["metadatas"])
            if m.get("source_file") == source_file
        ]
        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
            return True
        return False
    except Exception:
        return False
