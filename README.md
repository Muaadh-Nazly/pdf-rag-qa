# 📄 PDF RAG QA System

A Retrieval-Augmented Generation (RAG) system that lets you upload any PDF and ask questions about its content. Built with Google Gemini embeddings, FAISS vector search, Groq LLaMA 3.3 generation, and a clean Streamlit UI.

---

## 🏗️ System Architecture

```
PDF Upload
    ↓
Text Extraction (pypdf)
    ↓
Text Chunking (1000 chars, 200 overlap)
    ↓
Embedding (Google Gemini - gemini-embedding-001)
    ↓
Vector Storage (FAISS IndexFlatL2)
    ↓
User Query → Query Embedding → FAISS Similarity Search
    ↓
Top-k Relevant Chunks → Groq LLaMA 3.3 → Answer
```

---

## 🎬 Demo

> Upload a PDF, ask questions, get grounded answers with no hallucinations.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| PDF Extraction | `pypdf` |
| Embeddings | Google Gemini `gemini-embedding-001` (3072 dims) |
| Vector Store | `faiss-cpu` - IndexFlatL2 |
| Generation | Groq `llama-3.3-70b-versatile` |
| UI | Streamlit |

---

## 📁 Project Structure

```
pdf-rag-qa/
├── ingestor.py       # PDF text extraction and chunking
├── embedder.py       # Gemini embeddings + FAISS index builder
├── retriever.py      # Similarity search over FAISS index
├── chain.py          # Groq LLM answer generation
├── app.py            # Streamlit UI - wires all modules together
├── requirements.txt  # Pinned dependencies
├── .env.example      # Environment variable template
└── .gitignore
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Muaadh-Nazly/pdf-rag-qa.git
cd pdf-rag-qa
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```
GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key
```

- Get your Google API key at [aistudio.google.com](https://aistudio.google.com)
- Get your Groq API key at [console.groq.com](https://console.groq.com)

### 5. Run the app

```bash
python -m streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 💡 How It Works

1. **Upload** - Upload any PDF via the Streamlit UI
2. **Extract** - `pypdf` extracts raw text from all pages
3. **Chunk** - Text is split into 1000-character chunks with 200-character overlap to preserve context across boundaries
4. **Embed** - Each chunk is embedded using Google Gemini `gemini-embedding-001` producing 3072-dimensional vectors
5. **Index** - Embeddings are stored in a FAISS `IndexFlatL2` for fast similarity search
6. **Query** - Your question is embedded with the same model using `RETRIEVAL_QUERY` task type
7. **Retrieve** - FAISS returns the top 4 most semantically similar chunks
8. **Generate** - Retrieved context is passed to Groq `llama-3.3-70b-versatile` which generates a grounded answer

---

## ⚙️ Configuration

Key settings in `ingestor.py`:

```python
CHUNK_SIZE = 1000     # Characters per chunk
CHUNK_OVERLAP = 200   # Overlap between chunks
```

Key settings in `retriever.py`:

```python
k = 4    # Number of chunks to retrieve per query
```

---

## 📦 Requirements

```
google-genai==1.68.0
faiss-cpu==1.7.4
pypdf==6.12.1
groq==1.4.0
numpy<2
streamlit
python-dotenv
```

> Note: `numpy<2` is required for `faiss-cpu==1.7.4` compatibility on Python 3.10.

---

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.
