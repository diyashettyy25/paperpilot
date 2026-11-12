"""Agent 1 - Summarizer. Retrieves one or two chunks per paper section, then writes a structured summary."""
from langchain_core.output_parsers import StrOutputParser

from config import MAX_CONTEXT_CHARS
from prompts.prompt_templates import SUMMARY_PROMPT
from rag.retriever import get_front_matter, retrieve_multi
from services.llm_service import get_llm, run_chain
from utils.helpers import GenerationError, build_context, split_markdown_sections, unique_pages

SUMMARY_QUERIES = [
    "abstract, introduction and background",
    "problem statement and motivation",
    "objectives, aims and goals of the study",
    "proposed methodology, approach and model",
    "experimental results and findings",
    "conclusion",
    "limitations and future work",
]


def summarize_paper() -> dict:
    docs = retrieve_multi(SUMMARY_QUERIES, k_each=2)
    if not docs:
        raise GenerationError("No content could be retrieved from the paper. Try re-uploading the PDF.")
    chain = SUMMARY_PROMPT | get_llm(temperature=0.2) | StrOutputParser()   # LCEL: prompt | model | parser
    text = run_chain(chain, {"front_matter": get_front_matter(), "context": build_context(docs, MAX_CONTEXT_CHARS)})
    if not text.strip():
        raise GenerationError("The model returned an empty summary. Please try again.")
    return {"sections": split_markdown_sections(text), "pages": unique_pages(docs)}
