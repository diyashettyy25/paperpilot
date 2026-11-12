"""Agent 2 - Concept Extractor. Pulls problem, objectives, gap, concepts, methods, contributions and results."""
from langchain_core.output_parsers import StrOutputParser

from config import MAX_CONTEXT_CHARS
from prompts.prompt_templates import CONCEPT_PROMPT
from rag.retriever import get_front_matter, retrieve_multi
from services.llm_service import get_llm, run_chain
from utils.helpers import GenerationError, build_context, split_markdown_sections, unique_pages

CONCEPT_QUERIES = [
    "research problem and motivation",
    "research objectives and contributions",
    "related work and research gap",
    "key concepts, algorithms, techniques and technologies used",
    "proposed methodology and architecture",
    "results, evaluation and findings",
]


def extract_concepts() -> dict:
    docs = retrieve_multi(CONCEPT_QUERIES, k_each=2)
    if not docs:
        raise GenerationError("No content could be retrieved from the paper. Try re-uploading the PDF.")
    chain = CONCEPT_PROMPT | get_llm(temperature=0.1) | StrOutputParser()
    text = run_chain(chain, {"front_matter": get_front_matter(), "context": build_context(docs, MAX_CONTEXT_CHARS)})
    if not text.strip():
        raise GenerationError("The model returned an empty result. Please try again.")
    return {"sections": split_markdown_sections(text), "pages": unique_pages(docs)}
