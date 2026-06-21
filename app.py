import streamlit as st
import tempfile
import os
from ingestor import load_and_chunk_pdf
from embedder import build_vector_store
from retriever import retrieve_relevant_chunks
from chain import generate_answer

st.set_page_config(page_title="PDF RAG QA System", page_icon="📄", layout="centered")

st.title("📄 PDF Question Answering System")
st.markdown("Upload a PDF and ask questions about its content.")

# Session state
if "index" not in st.session_state:
    st.session_state.index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# PDF Upload
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file is not None and st.session_state.index is None:
    with st.spinner("Processing PDF - please wait..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        chunks = load_and_chunk_pdf(tmp_path)
        index, chunks = build_vector_store(chunks)

        st.session_state.index = index
        st.session_state.chunks = chunks
        os.unlink(tmp_path)

    st.success(
        f"PDF processed - {len(chunks)} chunks indexed. Ask your questions below."
    )

# Chat interface
if st.session_state.index is not None:
    question = st.chat_input("Ask a question about your document...")

    if question:
        with st.spinner("Thinking..."):
            relevant = retrieve_relevant_chunks(
                question, st.session_state.index, st.session_state.chunks
            )
            response = generate_answer(question, relevant)

        st.session_state.chat_history.append(
            {"question": question, "answer": response["answer"]}
        )

    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["question"])
        with st.chat_message("assistant"):
            st.write(chat["answer"])

# Reset button
if st.session_state.index is not None:
    if st.button("Upload a different PDF"):
        st.session_state.index = None
        st.session_state.chunks = None
        st.session_state.chat_history = []
        st.rerun()
