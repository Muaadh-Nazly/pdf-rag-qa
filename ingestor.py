from pypdf import PdfReader

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def extract_text(pdf_path: str) -> str:
    """Extract all text from a PDF file."""
    reader = PdfReader(pdf_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks for embedding."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c.strip() for c in chunks if c.strip()]


def load_and_chunk_pdf(pdf_path: str) -> list[str]:
    """Full pipeline - extract text from PDF and return chunks."""
    text = extract_text(pdf_path)
    chunks = chunk_text(text)
    print(f"Extracted and split into {len(chunks)} chunks")
    return chunks
