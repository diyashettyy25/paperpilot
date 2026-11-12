"""Q&A Agent (agentic RAG). Controller decides retrieval -> retrieve -> grounded answer -> reflection check."""
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

from config import MAX_CONTEXT_CHARS, REFLECTION_ENABLED, TOP_K
from prompts.prompt_templates import GENERAL_PROMPT, NOT_FOUND, QA_PROMPT, REFLECTION_PROMPT
from rag.retriever import get_front_matter, retrieve
from services.llm_service import get_llm, run_chain
from utils.helpers import PaperPilotError, build_context, unique_pages
from utils.intent_router import decide_retrieval


def _reflect(question: str, context: str, answer: str):
    """Second LLM pass: is the answer supported by the excerpts? Returns None if the check itself fails."""
    try:
        chain = REFLECTION_PROMPT | get_llm(temperature=0, json_mode=True) | JsonOutputParser()
        result = run_chain(chain, {"question": question, "context": context, "answer": answer}, retries=0)
        if isinstance(result, dict) and "supported" in result:
            return {"supported": bool(result["supported"]), "issue": str(result.get("issue", "")).strip()}
    except PaperPilotError:
        pass
    return None


def answer_question(question: str) -> dict:
    decision = decide_retrieval(question, get_front_matter(300))

    if decision["action"] == "answer_directly":
        chain = GENERAL_PROMPT | get_llm(temperature=0.3) | StrOutputParser()
        answer = run_chain(chain, {"question": question})
        return {"answer": answer.strip(), "used_retrieval": False, "reason": decision["reason"],
                "pages": [], "sources": [], "reflection": None}

    docs = retrieve(decision["query"] or question, k=TOP_K)
    if not docs:   # empty retrieval -> say so honestly, no invented answer
        return {"answer": NOT_FOUND, "used_retrieval": True, "reason": decision["reason"],
                "pages": [], "sources": [], "reflection": None}

    context = build_context(docs, MAX_CONTEXT_CHARS)
    chain = QA_PROMPT | get_llm(temperature=0.1) | StrOutputParser()
    answer = run_chain(chain, {"context": context, "question": question}).strip()
    reflection = _reflect(question, context, answer) if REFLECTION_ENABLED and answer != NOT_FOUND else None
    return {
        "answer": answer,
        "used_retrieval": True,
        "reason": decision["reason"],
        "pages": unique_pages(docs),
        "sources": [{"page": d.metadata.get("page"), "snippet": d.page_content[:220].replace("\n", " ")} for d in docs],
        "reflection": reflection,
    }
