import os
import numpy as np
import faiss
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
EMBED_MODEL = "gemini-embedding-001"


def retrieve_relevant_chunks(
    query: str, index: faiss.IndexFlatL2, chunks: list[str], k: int = 4
) -> list[str]:
    """
    Embed the query and retrieve the k most relevant chunks from FAISS index.

    Args:
        query: User question
        index: FAISS index built from document chunks
        chunks: Original text chunks matching the index
        k: Number of chunks to retrieve

    Returns:
        List of most relevant text chunks
    """
    response = client.models.embed_content(
        model=EMBED_MODEL,
        contents=[query],
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    query_vec = np.array([response.embeddings[0].values], dtype="float32")
    _, indices = index.search(query_vec, k)
    relevant = [chunks[i] for i in indices[0] if i < len(chunks)]
    print(f"Retrieved {len(relevant)} relevant chunks")
    return relevant
