"""Small shared helpers + the project's custom exceptions."""
import hashlib
import re
from pathlib import Path

from config import UPLOAD_DIR


# ---------- Exceptions (each one carries a message that is safe to show to the user) ----------
class PaperPilotError(Exception):
    """Base class for all expected errors."""


class PDFProcessingError(PaperPilotError):
    """Invalid, empty, encrypted or scanned PDF."""


class EmbeddingError(PaperPilotError):
    """Embedding model could not be loaded or used."""


class VectorStoreError(PaperPilotError):
    """ChromaDB problem."""


class LLMServiceError(PaperPilotError):
    """Ollama not running, model missing, or bad model output."""


class GenerationError(PaperPilotError):
    """An agent could not produce a usable result (e.g. no context, PPT failure)."""


# ---------- Files ----------
def file_hash(data: bytes) -> str:
    """Short SHA-256 fingerprint used to avoid indexing the same PDF twice."""
    return hashlib.sha256(data).hexdigest()[:16]


def safe_filename(name: str) -> str:
    name = Path(name).name
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name) or "paper.pdf"


def save_upload(name: str, data: bytes) -> Path:
    path = UPLOAD_DIR / safe_filename(name)
    path.write_bytes(data)
    return path


# ---------- Text / document helpers ----------
def unique_pages(docs) -> list:
    """Sorted list of distinct page numbers from retrieved chunks."""
    return sorted({d.metadata.get("page") for d in docs if d.metadata.get("page") is not None})


def format_pages(pages) -> str:
    if not pages:
        return ""
    return ("p. " if len(pages) == 1 else "pp. ") + ", ".join(str(p) for p in pages)


def build_context(docs, max_chars: int) -> str:
    """Join chunks as '[Page N] text' blocks, stopping at max_chars so small models are not overloaded."""
    parts, used = [], 0
    for d in docs:
        block = f"[Page {d.metadata.get('page', '?')}]\n{d.page_content.strip()}"
        if used + len(block) > max_chars and parts:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)


def split_markdown_sections(text: str) -> dict:
    """Turn '## Heading\\n body ...' markdown into {heading: body} so the UI can use expanders."""
    sections, current, buffer = {}, None, []
    for line in text.splitlines():
        match = re.match(r"^\s*#{1,4}\s*(.+?)\s*$", line)
        if match:
            if current is not None:
                sections[current] = "\n".join(buffer).strip()
            current = re.sub(r"^[\d.\s]+|[*:]+", "", match.group(1)).strip() or "Section"
            buffer = []
        else:
            buffer.append(line)
    if current is not None:
        sections[current] = "\n".join(buffer).strip()
    if not sections:
        return {"Response": text.strip()}
    return sections
