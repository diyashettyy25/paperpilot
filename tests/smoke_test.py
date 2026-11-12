"""Smoke test. Run from the project folder:  python tests/smoke_test.py

Tests 1-6 need no Ollama (they use a temporary ChromaDB folder, so your real data is untouched).
Test 7 runs the real agents and is skipped automatically if Ollama / the model is not ready.
"""
import os
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
TMP = tempfile.mkdtemp()
os.environ["CHROMA_DIR"] = str(Path(TMP) / "chroma")  # must be set before importing config

import fitz  # noqa: E402

SAMPLE_TEXT = [
    "Leaf Disease Detection using Convolutional Neural Networks\nA. Rao, 2024\nAbstract: We classify leaf images "
    "with a five layer CNN. The problem is that farmers cannot identify plant diseases early.",
    "Methodology: images from the PlantVillage dataset are resized, normalised and passed to a CNN trained with "
    "the Adam optimiser. Results: the model reaches 94 percent accuracy. Future work: field images.",
]


def make_pdf(path, pages):
    doc = fitz.open()
    for text in pages:
        doc.new_page().insert_textbox(fitz.Rect(50, 50, 550, 780), text, fontsize=11)
    doc.save(str(path))
    doc.close()


def test_invalid_pdf():
    from rag.pdf_processor import extract_pages
    from utils.helpers import PDFProcessingError
    bad = Path(TMP) / "bad.pdf"
    bad.write_bytes(b"this is not a pdf")
    try:
        extract_pages(bad, "bad.pdf")
    except PDFProcessingError:
        return
    raise AssertionError("invalid PDF was accepted")


def test_no_text_pdf():
    from rag.pdf_processor import extract_pages
    from utils.helpers import PDFProcessingError
    blank = Path(TMP) / "blank.pdf"
    make_pdf(blank, [""])
    try:
        extract_pages(blank, "blank.pdf")
    except PDFProcessingError:
        return
    raise AssertionError("PDF without text was accepted")


def test_pipeline_and_chroma():
    from rag.pdf_processor import extract_pages, split_pages
    from rag.retriever import get_front_matter, retrieve
    from rag.vector_store import clear_store, get_chunk_count, index_chunks, is_indexed
    pdf = Path(TMP) / "paper.pdf"
    make_pdf(pdf, SAMPLE_TEXT)
    pages = extract_pages(pdf, "paper.pdf")
    assert [p.metadata["page"] for p in pages] == [1, 2]
    chunks = split_pages(pages)
    assert index_chunks(chunks, "hash1") is True
    assert index_chunks(chunks, "hash1") is False, "duplicate indexing should be skipped"
    assert is_indexed("hash1") and get_chunk_count() == len(chunks)
    top = retrieve("What accuracy was reached?", k=1)[0]
    assert top.metadata["page"] == 2 and top.metadata["source"] == "paper.pdf"
    assert "Leaf Disease" in get_front_matter()
    index_chunks(chunks, "hash2")  # new paper replaces the old one
    assert not is_indexed("hash1") and is_indexed("hash2")
    clear_store()
    assert retrieve("anything") == []


def test_ppt():
    from services.ppt_service import build_presentation
    slides = [{"title": "Title", "bullets": ["A. Rao"], "diagram_steps": []}]
    for name in ["Introduction", "Problem Statement", "Objectives", "Research Gap", "Methodology",
                 "System Architecture / Workflow", "Results and Discussion", "Conclusion", "Future Scope and References"]:
        slides.append({"title": name, "bullets": ["Point one", "Point two"],
                       "diagram_steps": ["Input", "Model", "Output"] if "Workflow" in name else []})
    data = build_presentation("Test Paper", slides)
    assert data[:2] == b"PK" and len(data) > 10_000  # a .pptx is a zip file


def test_slide_normalisation():
    from agents.slide_generator import normalize_slides
    from utils.helpers import GenerationError
    title, slides = normalize_slides({"paper_title": "T", "slides": [{"title": f"S{i}", "bullets": ["a"] * 8} for i in range(10)]})
    assert len(slides) == 10 and all(len(s["bullets"]) <= 5 for s in slides)
    try:
        normalize_slides({"slides": []})
    except GenerationError:
        return
    raise AssertionError("empty slides accepted")


def test_intent_rules():
    from utils.intent_router import detect_intent
    cases = {"Summarize this paper": "summary", "Make a PPT": "slides", "Generate viva questions": "viva_questions",
             "quiz me": "viva_practice", "extract the key concepts": "concepts"}
    for text, expected in cases.items():
        assert detect_intent(text)["intent"] == expected, text


def test_live_llm():
    from services.llm_service import check_ollama
    status = check_ollama()
    if not status["model_ready"]:
        print("   (skipped: " + status["message"] + ")")
        return
    from agents.qa_agent import answer_question
    from agents.summarizer import summarize_paper
    from agents.viva_agent import generate_viva_questions
    from rag.pdf_processor import extract_pages, split_pages
    from rag.vector_store import index_chunks
    pdf = Path(TMP) / "paper.pdf"
    index_chunks(split_pages(extract_pages(pdf, "paper.pdf")), "live")
    assert summarize_paper()["sections"]
    result = answer_question("What accuracy did the model reach according to the paper?")
    assert result["used_retrieval"] and result["answer"]
    assert len(generate_viva_questions(8)) >= 5


if __name__ == "__main__":
    failures = 0
    tests = [test_invalid_pdf, test_no_text_pdf, test_pipeline_and_chroma, test_ppt, test_slide_normalisation,
             test_intent_rules, test_live_llm]
    for number, test in enumerate(tests, start=1):
        try:
            test()
            print(f"PASS {number}. {test.__name__}")
        except Exception:
            failures += 1
            print(f"FAIL {number}. {test.__name__}")
            traceback.print_exc()
    sys.exit(1 if failures else 0)
