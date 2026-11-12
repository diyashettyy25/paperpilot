"""Intent detection + the RAG controller (decides whether the paper must be searched).

Two small routing functions:
  detect_intent(message)  -> which agent should handle the request
  decide_retrieval(...)   -> for free questions: search the paper or answer directly (ReAct-style tool choice)
"""
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

from prompts.prompt_templates import INTENT_PROMPT, RAG_CONTROLLER_PROMPT
from services.llm_service import get_llm, run_chain
from utils.helpers import PaperPilotError

# intent -> (agent name, how retrieval is decided)
INTENT_INFO = {
    "summary": ("Summarizer Agent", "always retrieves paper sections"),
    "concepts": ("Concept Extractor Agent", "always retrieves paper sections"),
    "slides": ("Slide Generator Agent", "always retrieves paper sections"),
    "viva_questions": ("Viva Agent (question bank)", "always retrieves paper sections"),
    "viva_practice": ("Viva Agent (practice mode)", "always retrieves paper sections"),
    "question_answering": ("Q&A Agent", "decided by the RAG controller"),
}

# Keyword rules are checked in this order (most specific first).
_PRACTICE_WORDS = ("practice", "practise", "rehearse", "mock", "quiz me", "test me", "grade my")
_RULES = [
    ("viva_practice", lambda t: ("viva" in t and any(w in t for w in _PRACTICE_WORDS))
        or any(w in t for w in ("quiz me", "test me", "mock interview"))),
    ("viva_questions", lambda t: any(w in t for w in ("viva", "examiner", "interview question", "exam question"))),
    ("slides", lambda t: any(w in t for w in ("slide", "ppt", "powerpoint", "presentation", " deck"))),
    ("summary", lambda t: any(w in t for w in ("summar", "tl;dr", "tldr", "overview of the paper", "abstract"))),
    ("concepts", lambda t: any(w in t for w in (
        "key concept", "important concept", "main concept", "extract", "concept extraction", "list the concepts"))),
]
_PAPER_WORDS = ("the paper", "this paper", "the authors", "the author", "this study", "their method",
                "the proposed", "the model", "the dataset", "the algorithm", "the results", "the experiment")


def _decision(intent: str, method: str, reason: str) -> dict:
    agent, retrieval = INTENT_INFO[intent]
    return {"intent": intent, "agent": agent, "retrieval": retrieval, "method": method, "reason": reason}


def detect_intent(message: str) -> dict:
    """1) fast keyword rules  2) few-shot LLM classifier  3) default to question answering."""
    text = f" {message.lower().strip()} "
    for intent, matches in _RULES:
        if matches(text):
            return _decision(intent, "keyword rules", f"The message contains words typical of '{intent}'.")

    try:
        raw = run_chain(INTENT_PROMPT | get_llm(temperature=0) | StrOutputParser(), {"message": message}, retries=0)
        label = raw.strip().lower()
        for intent in INTENT_INFO:
            if intent in label:
                return _decision(intent, "LLM classifier (few-shot)", "The language model matched the request to this intent.")
    except PaperPilotError:
        pass  # if Ollama is down the Q&A step will show the real error message
    return _decision("question_answering", "default", "No special request detected, so it is treated as a question.")


def decide_retrieval(question: str, paper_hint: str = "") -> dict:
    """RAG controller. Returns {"action": "search_paper"|"answer_directly", "query": str, "reason": str}."""
    lowered = question.lower()
    if any(word in lowered for word in _PAPER_WORDS):
        return {"action": "search_paper", "query": question, "reason": "The question refers to the paper directly."}

    chain = RAG_CONTROLLER_PROMPT | get_llm(temperature=0, json_mode=True) | JsonOutputParser()
    result = run_chain(chain, {"question": question, "paper_hint": paper_hint[:300] or "unknown"})
    if not isinstance(result, dict):
        result = {}
    action = "answer_directly" if result.get("action") == "answer_directly" else "search_paper"
    return {
        "action": action,
        "query": (result.get("query") or question) if action == "search_paper" else "",
        "reason": str(result.get("thought") or "Checking the paper to stay grounded."),
    }
