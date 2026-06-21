import streamlit as st
import tempfile
import os
from ingestor import load_and_chunk_pdf
from embedder import build_vector_store
from chain import build_qa_chain, ask_question

st.set_page_config(page_title="PDF RAG QA System", page_icon="📄", layout="centered")

st.title("📄 PDF Question Answering System")
st.markdown("Upload a PDF and ask questions about its content.")

# Session state to persist vector store and chain across reruns
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# PDF Upload
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file is not None and st.session_state.qa_chain is None:
    with st.spinner("Processing PDF - please wait..."):
        # Save uploaded file to a temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        # Run pipeline
        chunks = load_and_chunk_pdf(tmp_path)
        vector_store = build_vector_store(chunks)
        st.session_state.qa_chain = build_qa_chain(vector_store)

        # Clean up temp file
        os.unlink(tmp_path)

    st.success(
        f"PDF processed - {len(chunks)} chunks indexed. Ask your questions below."
    )

# Chat interface
if st.session_state.qa_chain is not None:
    question = st.chat_input("Ask a question about your document...")

    if question:
        with st.spinner("Thinking..."):
            response = ask_question(st.session_state.qa_chain, question)

        st.session_state.chat_history.append(
            {"question": question, "answer": response["answer"]}
        )

    # Display chat history
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["question"])
        with st.chat_message("assistant"):
            st.write(chat["answer"])

# Reset button
if st.session_state.qa_chain is not None:
    if st.button("Upload a different PDF"):
        st.session_state.qa_chain = None
        st.session_state.chat_history = []
        st.rerun()
