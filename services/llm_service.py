"""Provider-neutral LangChain LLM helpers for Groq API and local Ollama."""
import json
import urllib.request

from langchain_core.exceptions import OutputParserException
from langchain_ollama import ChatOllama

from config import (
    LLM_PROVIDER, GROQ_API_KEY, GROQ_MODEL,
    OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_NUM_CTX,
)
from utils.helpers import LLMServiceError


def get_llm(temperature: float = 0.2, json_mode: bool = False):
    """Create the configured chat model. Groq is the default; Ollama is the local alternative."""
    if LLM_PROVIDER == "groq":
        if not GROQ_API_KEY:
            raise LLMServiceError(
                "GROQ_API_KEY is missing. Add your Groq key to the project's .env file, "
                "or set LLM_PROVIDER=ollama to use your local model."
            )
        try:
            from langchain_groq import ChatGroq
        except ImportError as exc:
            raise LLMServiceError(
                "The Groq integration is not installed. Run: python -m pip install langchain-groq"
            ) from exc

        options = {
            "model": GROQ_MODEL,
            "api_key": GROQ_API_KEY,
            "temperature": temperature,
        }
        # Groq JSON Object Mode produces valid JSON for the existing JsonOutputParser chains.
        if json_mode:
            options["model_kwargs"] = {"response_format": {"type": "json_object"}}
        return ChatGroq(**options)

    if LLM_PROVIDER == "ollama":
        options = {
            "model": OLLAMA_MODEL,
            "base_url": OLLAMA_BASE_URL,
            "temperature": temperature,
            "num_ctx": OLLAMA_NUM_CTX,
        }
        if json_mode:
            options["format"] = "json"
        return ChatOllama(**options)

    raise LLMServiceError(
        f"Unsupported LLM_PROVIDER '{LLM_PROVIDER}'. Choose 'groq' or 'ollama' in .env."
    )


def check_ollama() -> dict:
    """Backward-compatible Ollama status check (also used by the optional Ollama smoke test)."""
    url = OLLAMA_BASE_URL.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            names = [m.get("name", "") for m in json.load(response).get("models", [])]
    except Exception:
        return {
            "running": False,
            "model_ready": False,
            "message": f"Ollama is not running at {OLLAMA_BASE_URL}. Start the Ollama app or run `ollama serve`.",
        }
    ready = any(n == OLLAMA_MODEL or n == f"{OLLAMA_MODEL}:latest" for n in names)
    if not ready:
        return {
            "running": True,
            "model_ready": False,
            "message": f"Ollama is running but model '{OLLAMA_MODEL}' is missing. Run `ollama pull {OLLAMA_MODEL}`.",
        }
    return {"running": True, "model_ready": True, "message": f"Ollama ready · model: {OLLAMA_MODEL}"}


def check_llm() -> dict:
    """Return a quick configuration status for the selected provider without making a billable call."""
    if LLM_PROVIDER == "groq":
        ready = bool(GROQ_API_KEY)
        return {
            "running": ready,
            "model_ready": ready,
            "provider": "Groq API",
            "message": (
                f"Groq API configured · model: {GROQ_MODEL}"
                if ready else
                "Groq API key missing. Add GROQ_API_KEY to .env and restart Streamlit."
            ),
        }
    if LLM_PROVIDER == "ollama":
        status = check_ollama()
        status["provider"] = "Ollama (local)"
        return status
    return {
        "running": False,
        "model_ready": False,
        "provider": LLM_PROVIDER,
        "message": f"Unsupported LLM_PROVIDER '{LLM_PROVIDER}'. Choose 'groq' or 'ollama' in .env.",
    }


def _translate_error(exc: Exception) -> LLMServiceError:
    text = str(exc).lower()
    if LLM_PROVIDER == "groq":
        if "api key" in text or "authentication" in text or "unauthorized" in text or "401" in text:
            return LLMServiceError("Groq authentication failed. Check GROQ_API_KEY in .env.")
        if "rate limit" in text or "429" in text:
            return LLMServiceError("Groq rate limit reached. Wait a little and try again.")
        if "model" in text and ("not found" in text or "decommission" in text):
            return LLMServiceError(f"Groq model '{GROQ_MODEL}' is unavailable. Update GROQ_MODEL in .env.")
        return LLMServiceError(f"Groq API request failed: {exc}")
    if "not found" in text and "model" in text:
        return LLMServiceError(f"Model '{OLLAMA_MODEL}' is not installed. Run `ollama pull {OLLAMA_MODEL}`.")
    if "connect" in text or "refused" in text or "timed out" in text:
        return LLMServiceError(
            f"Cannot reach Ollama at {OLLAMA_BASE_URL}. Start the Ollama app or run `ollama serve`."
        )
    return LLMServiceError(f"The language model call failed: {exc}")


def run_chain(chain, inputs: dict, retries: int = 1):
    """Invoke an LCEL chain. Retries once if structured output could not be parsed."""
    last_error = None
    for _ in range(retries + 1):
        try:
            return chain.invoke(inputs)
        except OutputParserException as exc:
            last_error = exc
        except Exception as exc:
            raise _translate_error(exc) from exc
    raise LLMServiceError(
        "The model did not return valid structured output. Try again or choose another model."
    ) from last_error
