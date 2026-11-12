"""Step 1-4 of the RAG pipeline: read the PDF (keeping page numbers) and split it into chunks."""
from pathlib import Path

import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE
from utils.helpers import PDFProcessingError


def extract_pages(pdf_path: Path, source_name: str) -> list:
    """Return one LangChain Document per page that has text. Metadata: source filename + page number."""
    try:
        pdf = fitz.open(str(pdf_path))
    except Exception as exc:
        raise PDFProcessingError("This file could not be opened. It is not a valid PDF or it is corrupted.") from exc

    with pdf:
        if pdf.needs_pass:
            raise PDFProcessingError("This PDF is password-protected. Please upload an unlocked copy.")
        if pdf.page_count == 0:
            raise PDFProcessingError("This PDF has no pages.")
        pages = []
        for number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append(Document(page_content=text, metadata={"source": source_name, "page": number}))

    if not pages:
        raise PDFProcessingError(
            "No text could be extracted. The PDF may be a scanned image; please upload a text-based PDF."
        )
    return pages


def split_pages(pages: list) -> list:
    """Split page documents into overlapping chunks. Each chunk keeps its page number and gets a chunk_id."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(pages)
    if not chunks:
        raise PDFProcessingError("The PDF text was too short to split into chunks.")
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index
    return chunks
