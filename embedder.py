import os
import numpy as np
import faiss
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
EMBED_MODEL = "gemini-embedding-001"


def embed_texts(texts: list[str]) -> np.ndarray:
    """Convert a list of text chunks into embeddings."""
    response = client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
    )
    return np.array([e.values for e in response.embeddings], dtype="float32")


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """Build a FAISS index from embeddings."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    print(f"FAISS index built with {index.ntotal} vectors")
    return index


def build_vector_store(chunks: list[str]) -> tuple[faiss.IndexFlatL2, list[str]]:
    """Full pipeline - embed chunks and return FAISS index + chunks."""
    embeddings = embed_texts(chunks)
    print(f"Embeddings shape: {embeddings.shape}")
    index = build_index(embeddings)
    return index, chunks
