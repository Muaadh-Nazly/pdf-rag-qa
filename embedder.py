import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()


def build_vector_store(chunks: list) -> FAISS:
    """
    Convert document chunks into embeddings and store in FAISS index.

    Args:
        chunks: List of document chunks from ingestor

    Returns:
        FAISS vector store
    """
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    vector_store = FAISS.from_documents(chunks, embeddings)
    print(f"Vector store built with {len(chunks)} chunks")

    return vector_store


def save_vector_store(vector_store: FAISS, path: str = "faiss_index") -> None:
    """Save FAISS index to disk."""
    vector_store.save_local(path)
    print(f"Vector store saved to {path}/")


def load_vector_store(path: str = "faiss_index") -> FAISS:
    """Load FAISS index from disk."""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    vector_store = FAISS.load_local(
        path, embeddings, allow_dangerous_deserialization=True
    )
    print(f"Vector store loaded from {path}/")

    return vector_store
