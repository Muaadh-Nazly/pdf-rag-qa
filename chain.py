import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
GEN_MODEL = "llama-3.3-70b-versatile"


def generate_answer(query: str, context_chunks: list[str]) -> dict:
    """
    Generate an answer using Groq LLM given retrieved context chunks.

    Args:
        query: User question
        context_chunks: Relevant chunks retrieved from FAISS

    Returns:
        Dictionary with answer and source chunks
    """
    context = "\n\n".join(context_chunks)
    prompt = f"""Use the following context extracted from the document to answer the question.
If the answer is not found in the context, say "I could not find an answer in the provided document."
Do not make up answers.

Context:
{context}

Question: {query}

Answer:"""

    response = groq_client.chat.completions.create(
        model=GEN_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.3
    )

    return {"answer": response.choices[0].message.content, "sources": context_chunks}
