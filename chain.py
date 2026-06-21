import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()


def build_qa_chain(vector_store: FAISS) -> RetrievalQA:
    """
    Build a RetrievalQA chain using Gemini and FAISS retriever.

    Args:
        vector_store: FAISS vector store

    Returns:
        RetrievalQA chain
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.3,
    )

    prompt_template = """
    Use the following context extracted from the document to answer the question.
    If the answer is not found in the context, say "I could not find an answer in the provided document."
    Do not make up answers.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )

    return qa_chain


def ask_question(qa_chain: RetrievalQA, question: str) -> dict:
    """
    Run a question through the QA chain.

    Args:
        qa_chain: RetrievalQA chain
        question: User's question

    Returns:
        Dictionary with answer and source documents
    """
    result = qa_chain.invoke({"query": question})

    return {"answer": result["result"], "sources": result["source_documents"]}
