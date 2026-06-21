from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_and_chunk_pdf(file_path: str) -> list:
    """
    Load a PDF file and split it into chunks for embedding.

    Args:
        file_path: Path to the PDF file

    Returns:
        List of document chunks
    """
    # Load PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = splitter.split_documents(documents)
    print(f"Loaded {len(documents)} page(s) - split into {len(chunks)} chunks")

    return chunks