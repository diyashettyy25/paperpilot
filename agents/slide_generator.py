"""Agent 3 - Slide Generator. The LLM writes slide JSON (few-shot prompt); ppt_service turns it into a .pptx."""
from langchain_core.output_parsers import JsonOutputParser

from config import MAX_CONTEXT_CHARS
from prompts.prompt_templates import SLIDE_PROMPT
from rag.retriever import get_front_matter, retrieve_multi
from services.llm_service import get_llm, run_chain
from services.ppt_service import build_presentation
from utils.helpers import GenerationError, build_context, unique_pages

SLIDE_QUERIES = [
    "title, authors and abstract",
    "introduction and background",
    "problem statement",
    "objectives and contributions",
    "research gap and related work",
    "methodology and proposed approach",
    "system architecture, framework and workflow",
    "results and discussion",
    "conclusion and future scope",
    "references",
]


def _as_list(value) -> list:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def normalize_slides(data, fallback_title: str = "Research Paper") -> tuple:
    """Validate the model's JSON and tidy it (max 5 bullets, 5 diagram steps, max 10 slides)."""
    if not isinstance(data, dict) or not isinstance(data.get("slides"), list):
        raise GenerationError("The model did not return slide content in the expected format. Please try again.")
    slides = []
    for raw in data["slides"][:10]:
        if not isinstance(raw, dict):
            continue
        title = str(raw.get("title", "")).strip()
        bullets = _as_list(raw.get("bullets"))[:5]
        if title and bullets:
            slides.append({"title": title, "bullets": bullets, "diagram_steps": _as_list(raw.get("diagram_steps"))[:5]})
    if len(slides) < 6:
        raise GenerationError("The model produced too few usable slides. Please try again or use a larger model.")
    paper_title = str(data.get("paper_title") or "").strip() or fallback_title
    return paper_title, slides


def generate_presentation() -> dict:
    docs = retrieve_multi(SLIDE_QUERIES, k_each=2)
    if not docs:
        raise GenerationError("No content could be retrieved from the paper. Try re-uploading the PDF.")
    chain = SLIDE_PROMPT | get_llm(temperature=0.2, json_mode=True) | JsonOutputParser()
    data = run_chain(chain, {"front_matter": get_front_matter(), "context": build_context(docs, MAX_CONTEXT_CHARS)})
    paper_title, slides = normalize_slides(data)
    pptx_bytes = build_presentation(paper_title, slides)   # real .pptx built with python-pptx
    return {"paper_title": paper_title, "slides": slides, "pptx_bytes": pptx_bytes, "pages": unique_pages(docs)}
