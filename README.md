# PaperPilot AI – Paper-to-Presentation & Viva Agent

An Agentic-RAG application. Upload a research paper (PDF) and get a structured summary, key concepts, a
downloadable PowerPoint, viva questions with model answers, a graded viva practice mode, and a chat to ask
questions about the paper. The default LLM is **Groq API** (API key required); you can switch to **Ollama** for local inference. Embeddings and ChromaDB remain local.

## 1. Problem statement
Students must read long research papers, present them and defend them in a viva. Doing this by hand is slow.
Generic chatbots may invent facts that are not in the paper.

## 2. Objectives
- Use the uploaded PDF as the **only** knowledge source (grounded answers with page references).
- Generate summary, concepts, slides and viva material with four separate agents.
- Demonstrate LLM integration, prompt engineering, LangChain/LCEL, agentic RAG, intent routing and a vector DB.

## 3. Features
| Section | What it does |
|---|---|
| Dashboard | Upload/replace the PDF, processing status, feature cards |
| Paper Summary | Introduction → future scope, in expandable sections |
| Key Concepts | Problem, objectives, gap, concepts, algorithms, methodology, contributions, findings |
| Generate Slides | 10-slide blue/white/grey PPTX with a workflow diagram, downloadable |
| Viva Preparation | 15–20 Basic/Intermediate/Advanced questions with answers + practice mode with scoring |
| Ask Your Paper | Chat. Shows detected intent, selected agent, whether retrieval was used, and source pages |

## 4. Technology stack (what each part does – useful for your viva)
| Technology | Role in this project |
|---|---|
| **Python** | Backend language |
| **Streamlit** | Builds the web UI; `st.session_state` keeps results while you switch pages |
| **Groq** | Cloud LLM inference through `ChatGroq` and a securely stored `GROQ_API_KEY` |
| **Ollama** | Optional local LLM alternative (default local model `llama3.2`) |
| **LangChain** | Framework to build LLM apps: prompt templates, model wrappers, parsers, text splitters |
| **ChatPromptTemplate** | Reusable prompt with a system message + user message and `{variables}` |
| **LCEL** (LangChain Expression Language) | Connects steps with `|`, e.g. `prompt | llm | parser` |
| **Output parsers** | `StrOutputParser` (plain text) and `JsonOutputParser` (JSON → Python dict) |
| **Embeddings** | Numbers that capture the meaning of text; similar meaning → nearby vectors |
| **HuggingFace Sentence-Transformers** | The embedding model (`all-MiniLM-L6-v2`), runs locally |
| **ChromaDB** | Vector database that stores chunk embeddings + metadata and finds the most similar chunks |
| **PyMuPDF** | Extracts text from each PDF page (so we can keep page numbers) |
| **python-pptx** | Creates the `.pptx` file |
| **RAG** (Retrieval-Augmented Generation) | Retrieve relevant chunks first, then let the LLM answer using only them |
| **Agentic RAG** | The system *decides* whether to retrieve and which agent to use |

## 5. Agent architecture
```
 User request ──► Intent Router ──► Agent ──► (RAG controller / retrieval) ──► Groq or Ollama LLM ──► UI
                  utils/intent_router.py
```
| Agent (file) | Responsibility | Prompt technique |
|---|---|---|
| Summarizer (`agents/summarizer.py`) | Structured summary | Zero-shot |
| Concept Extractor (`agents/concept_extractor.py`) | Problem, gap, concepts, contributions | Zero-shot |
| Slide Generator (`agents/slide_generator.py`) | Slide JSON → PPTX | Few-shot (JSON example) |
| Viva Agent (`agents/viva_agent.py`) | Questions + model answers; scores student answers | Few-shot (questions), zero-shot rubric (evaluation) |
| Q&A Agent (`agents/qa_agent.py`) *(extra, needed for the chat)* | Grounded answers with pages | ReAct-style tool choice + reflection |

Each agent has its own prompt in `prompts/prompt_templates.py` and its own retrieval queries.

## 6. RAG workflow
1. Upload PDF in Streamlit → saved to `data/uploads/`.
2. **PyMuPDF** extracts text page by page (page number kept as metadata).
3. **RecursiveCharacterTextSplitter** makes ~1000-character chunks (150 overlap); each chunk keeps `source`, `page`, `chunk_id`.
4. **Sentence-Transformers** turn chunks into embeddings.
5. **ChromaDB** (persistent, `chroma_db/`) stores embeddings + metadata. The PDF's SHA-256 hash prevents duplicate indexing; a new paper deletes the old collection (one active paper).
6. At question time, the agent runs **similarity search** and passes the top chunks (as `[Page N] text`) to the prompt.
7. **Groq or Ollama** generates the answer; the UI shows the pages used. If nothing relevant exists the app says so instead of guessing.

## 7. How intent routing works
`detect_intent()` in `utils/intent_router.py` uses three steps:
1. **Keyword rules** (fast, free): "slides/ppt" → slides, "viva + practice/quiz me" → viva_practice, "viva/examiner" → viva_questions, "summar…" → summary, "key concepts/extract" → concepts.
2. If no rule matches, a **few-shot LLM classifier** picks one of the six intents.
3. Otherwise it defaults to `question_answering`.

**RAG controller** (`decide_retrieval`): summary, concepts, slides and viva always retrieve. For free questions a
**ReAct-style** prompt chooses a tool – `search_paper` or `answer_directly` – and gives a one-sentence reason
(no hidden chain-of-thought). "Hi" or "Who won the 2018 World Cup?" does **not** trigger retrieval.
After answering, a **reflection** step asks the model whether the answer is supported by the retrieved text and
shows a warning if not (`REFLECTION_ENABLED=false` in `.env` turns it off).

## 8. Folder structure
```
paperpilot-ai/
├── app.py                  Streamlit UI (no AI logic)
├── config.py               Settings from .env
├── requirements.txt  .env.example  .gitignore  README.md
├── agents/                 summarizer, concept_extractor, slide_generator, viva_agent, qa_agent
├── rag/                    pdf_processor, embeddings, vector_store, retriever
├── prompts/                prompt_templates.py (all prompts)
├── services/               llm_service.py (Groq/Ollama), ppt_service.py (python-pptx)
├── utils/                  intent_router.py, helpers.py (helpers + custom errors)
├── tests/smoke_test.py
├── .streamlit/config.toml  light blue theme
├── data/uploads/           uploaded PDFs
└── chroma_db/              vector database (created automatically)
```

## 9. Installation
Requires **Python 3.10 – 3.12**.
```bash
cd paperpilot-ai
python -m venv .venv
# Windows:  .venv\Scripts\activate        macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # Windows: copy .env.example .env   (optional: change model name)
```

### Configure Groq API (default)
1. Create an API key at https://console.groq.com/keys.
2. In the project folder, copy `.env.example` to a new file named `.env`.
3. Open `.env` and replace `put_your_groq_api_key_here` with your key. Do not share the key or commit `.env` to Git.
4. Keep `LLM_PROVIDER=groq` and `GROQ_MODEL=llama-3.3-70b-versatile` (or set a currently available model from your Groq console).
5. Install dependencies with `python -m pip install -r requirements.txt` and run `python -m streamlit run app.py`.

### Switch to local Ollama instead
1. Install Ollama from https://ollama.com/download and run `ollama pull llama3.2`.
2. In `.env`, set `LLM_PROVIDER=ollama`.
3. Run `python -m streamlit run app.py`.

The sidebar reports whether the selected provider is configured. Groq status checks only confirm that a key is present; the first actual model request verifies API connectivity and credentials.

## 10. Run
```bash
streamlit run app.py
```
Open http://localhost:8501. The sidebar shows whether the selected LLM provider is configured.
The first PDF upload downloads the embedding model (~90 MB, needs internet once).

## 11. Testing
```bash
python tests/smoke_test.py
```
Checks invalid/empty PDFs, extraction with page numbers, ChromaDB indexing (no duplicates, replacement), retrieval,
PPT creation, intent rules, and – when the selected Groq/Ollama provider is configured – the real agents.

Manual test: upload a text-based paper → Dashboard shows pages/chunks → generate summary, concepts, PPT
(open the downloaded file), viva questions + practice → in *Ask Your Paper* try: "What dataset was used?",
"Hi there", "Who won the 2018 World Cup?" (no retrieval), "make slides", "quiz me".

## 12. Seven acceptance criteria
| # | Requirement | Where it is implemented |
|---|---|---|
| 1 | LLM integration | `services/llm_service.py` – `ChatGroq` or `ChatOllama`, selected by `LLM_PROVIDER`; `check_llm()` checks provider configuration |
| 2 | Interactive GUI | `app.py` – upload, six sections, PPT download, viva practice, chat |
| 3 | Prompt engineering | `prompts/prompt_templates.py` – zero-shot, few-shot, ReAct-style controller, reflection; separate system + user prompts |
| 4 | LangChain core | `ChatPromptTemplate`, LCEL `prompt \| llm \| parser`, `StrOutputParser`, `JsonOutputParser` in every agent |
| 5 | Agentic RAG | `decide_retrieval()` + `agents/qa_agent.py`: conditional retrieval from ChromaDB, grounded answers |
| 6 | Intent detection & routing | `detect_intent()`; intent and agent displayed in chat and sidebar |
| 7 | Vector database | `rag/vector_store.py` + `rag/retriever.py` – ChromaDB, similarity search, `source`/`page` metadata |

## 13. Troubleshooting
| Message | Fix |
|---|---|
| "Groq API key missing" | Add `GROQ_API_KEY` to `.env`, then restart Streamlit |
| "Groq authentication failed" | Check the key in `.env`; never paste it into source code |
| "Ollama is not running" | Set `LLM_PROVIDER=ollama`, then start the Ollama app or `ollama serve` |
| "Model … is missing" | `ollama pull <model>` (name must match `OLLAMA_MODEL`) |
| "No text could be extracted" | PDF is a scan; use a text-based PDF |
| "Could not load the embedding model" | Connect to the internet for the first run |
| "did not return valid structured output" | Retry, or use a larger model |
| Slow generation | Normal on CPU; use a smaller model or reduce `MAX_CONTEXT_CHARS` |
