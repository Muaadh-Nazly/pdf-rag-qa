from langchain_community.vectorstores import FAISS


def retrieve_relevant_chunks(vector_store: FAISS, query: str, k: int = 4) -> list:
    """
    Retrieve the most relevant document chunks for a given query.

    Args:
        vector_store: FAISS vector store
        query: User's question
        k: Number of chunks to retrieve

    Returns:
        List of relevant document chunks
    """
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": k}
    )

    relevant_chunks = retriever.invoke(query)
    print(f"Retrieved {len(relevant_chunks)} relevant chunks for query")

    return relevant_chunks
