"""Agent 4 - Viva Agent. Generates questions + model answers, and evaluates a student's typed answer."""
from langchain_core.output_parsers import JsonOutputParser

from config import MAX_CONTEXT_CHARS
from prompts.prompt_templates import EVALUATOR_PROMPT, VIVA_PROMPT
from rag.retriever import get_front_matter, retrieve, retrieve_multi
from services.llm_service import get_llm, run_chain
from utils.helpers import GenerationError, build_context

VIVA_QUERIES = [
    "proposed methodology and approach",
    "algorithms and techniques used",
    "system architecture and workflow",
    "dataset, experiments and implementation",
    "results and evaluation",
    "limitations and challenges",
    "conclusion and future scope",
]
LEVELS = {"basic": "Basic", "intermediate": "Intermediate", "advanced": "Advanced"}


def generate_viva_questions(num_questions: int = 18) -> list:
    docs = retrieve_multi(VIVA_QUERIES, k_each=2)
    if not docs:
        raise GenerationError("No content could be retrieved from the paper. Try re-uploading the PDF.")
    chain = VIVA_PROMPT | get_llm(temperature=0.3, json_mode=True) | JsonOutputParser()
    data = run_chain(chain, {
        "front_matter": get_front_matter(),
        "context": build_context(docs, MAX_CONTEXT_CHARS),
        "num_questions": num_questions,
    })
    raw_list = data.get("questions") if isinstance(data, dict) else None
    questions = []
    for item in raw_list or []:
        if not isinstance(item, dict):
            continue
        question, answer = str(item.get("question", "")).strip(), str(item.get("answer", "")).strip()
        if question and answer:
            questions.append({
                "level": LEVELS.get(str(item.get("level", "")).strip().lower(), "Intermediate"),
                "topic": str(item.get("topic", "General")).strip() or "General",
                "question": question,
                "answer": answer,
                "page": item.get("page") if isinstance(item.get("page"), int) else None,
            })
    if len(questions) < 5:
        raise GenerationError("The model produced too few viva questions. Please try again or use a larger model.")
    return questions


def evaluate_answer(question: str, model_answer: str, user_answer: str) -> dict:
    if not user_answer.strip():
        raise GenerationError("Please type an answer before evaluating.")
    context = build_context(retrieve(question, k=3), 4000)
    chain = EVALUATOR_PROMPT | get_llm(temperature=0.1, json_mode=True) | JsonOutputParser()
    data = run_chain(chain, {
        "question": question, "model_answer": model_answer, "context": context, "user_answer": user_answer,
    })
    if not isinstance(data, dict):
        raise GenerationError("The evaluation result was not in the expected format. Please try again.")
    try:
        score = max(0, min(10, int(round(float(data.get("score", 0))))))
    except (TypeError, ValueError):
        raise GenerationError("The model returned an invalid score. Please try again.")
    return {
        "score": score,
        "verdict": str(data.get("verdict", "")).strip(),
        "strengths": [str(s) for s in data.get("strengths", []) if str(s).strip()] if isinstance(data.get("strengths"), list) else [],
        "improvements": [str(s) for s in data.get("improvements", []) if str(s).strip()] if isinstance(data.get("improvements"), list) else [],
        "feedback": str(data.get("feedback", "")).strip(),
    }
