from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel
from langchain.schema.output_parser import StrOutputParser
from langchain_chroma import Chroma

from src.llm_provider import get_llm
from src.document_processor import get_retriever


SYSTEM_PROMPT = """You are an expert academic research assistant. Your role is to help researchers \
understand and analyze academic papers.

Use ONLY the provided context from the papers to answer questions. If the answer cannot be found \
in the context, clearly state: "I couldn't find relevant information about this in the indexed papers."

Always cite your sources by mentioning the paper name (source_file from metadata) when referencing \
specific information. Format citations as [Paper: <paper_name>].

Be precise, academic in tone, and highlight key findings, methodologies, and conclusions when relevant.

Context from papers:
{context}
"""

HUMAN_PROMPT = "{question}"


def format_docs(docs) -> str:
    """Format retrieved documents into a single context string with citations."""
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source_file", "Unknown")
        page = doc.metadata.get("page", "?")
        formatted.append(
            f"[Source: {source} | Page: {page}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(formatted)


def format_docs_with_metadata(docs) -> list:
    """Return list of dicts with content and metadata for UI display."""
    return [
        {
            "content": doc.page_content,
            "source": doc.metadata.get("source_file", "Unknown"),
            "page": doc.metadata.get("page", "?"),
        }
        for doc in docs
    ]


def build_rag_chain(vectorstore: Chroma):
    """Build and return the RAG chain."""
    retriever = get_retriever(vectorstore)
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT),
    ])

    # Build chain with source documents passthrough
    rag_chain_with_sources = RunnableParallel(
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
            "source_documents": retriever,
        }
    )

    # Final chain: retrieve  format  prompt  LLM  parse
    chain = (
        rag_chain_with_sources
        | {
            "answer": (
                lambda x: {"context": x["context"], "question": x["question"]}
            )
            | prompt
            | llm
            | StrOutputParser(),
            "source_documents": lambda x: format_docs_with_metadata(x["source_documents"]),
        }
    )

    return chain


def build_streaming_chain(vectorstore: Chroma):
    """Build a simpler chain for streaming responses."""
    retriever = get_retriever(vectorstore)
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT),
    ])

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever
