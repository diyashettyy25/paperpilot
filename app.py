"""PaperPilot AI - Streamlit front end. All AI logic lives in agents/, rag/, services/ and utils/."""
import streamlit as st

import config
from agents.concept_extractor import extract_concepts
from agents.qa_agent import answer_question
from agents.slide_generator import generate_presentation
from agents.summarizer import summarize_paper
from agents.viva_agent import evaluate_answer, generate_viva_questions
from rag.pdf_processor import extract_pages, split_pages
from rag.vector_store import clear_store, index_chunks
from services.llm_service import check_llm
from utils.helpers import PaperPilotError, file_hash, format_pages, safe_filename, save_upload
from utils.intent_router import detect_intent

st.set_page_config(page_title="PaperPilot AI", page_icon="📄", layout="wide")

PAGES = ["Dashboard", "Paper Summary", "Key Concepts", "Generate Slides", "Viva Preparation", "Ask Your Paper"]

st.markdown("""
<style>
h1, h2, h3 { color: #0B2545; }
.pp-card { background:#fff; border:1px solid #E1E8F0; border-radius:14px; padding:1.1rem 1.3rem; margin-bottom:.8rem; }
.pp-card h4 { margin:0 0 .3rem 0; color:#0B2545; }
.pp-card p { margin:0; color:#5B677A; font-size:.92rem; }
.pp-hero { background:#0B2545; border-radius:16px; padding:1.6rem 1.8rem; margin-bottom:1rem; }
.pp-hero h1 { color:#fff; margin:0; font-size:2rem; }
.pp-hero p { color:#DCEBFB; margin:.4rem 0 0 0; }
.pp-pill { display:inline-block; background:#DCEBFB; color:#0B2545; border-radius:999px; padding:.15rem .7rem;
           font-size:.8rem; margin-right:.4rem; }
.stButton > button { border-radius:10px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------- session state
for key, default in {"doc": None, "summary": None, "concepts": None, "slides": None, "viva": None,
                     "practice": None, "chat": [], "page": "Dashboard", "uploader_key": 0,
                     "failed_hash": None, "upload_error": None, "last_route": None}.items():
    st.session_state.setdefault(key, default)

if "nav_target" in st.session_state:          # programmatic navigation (must happen before the radio is drawn)
    st.session_state.page = st.session_state.pop("nav_target")


def go_to(page: str):
    st.session_state.nav_target = page
    st.rerun()


def reset_results():
    for key in ("summary", "concepts", "slides", "viva", "practice"):
        st.session_state[key] = None
    st.session_state.chat = []


@st.cache_data(ttl=15, show_spinner=False)
def llm_status():
    return check_llm()


# ---------------------------------------------------------------- document processing
def process_pdf(uploaded):
    data = uploaded.getvalue()
    doc_hash = file_hash(data)
    doc = st.session_state.doc
    if (doc and doc["hash"] == doc_hash) or st.session_state.failed_hash == doc_hash:
        return
    st.session_state.upload_error = None
    with st.status("Processing paper...", expanded=True) as status:
        try:
            st.write("Reading PDF and keeping page numbers...")
            pages = extract_pages(save_upload(uploaded.name, data), uploaded.name)
            st.write("Splitting text into chunks...")
            chunks = split_pages(pages)
            st.write("Creating embeddings and storing them in ChromaDB (first run downloads the model)...")
            newly_indexed = index_chunks(chunks, doc_hash)
            reset_results()
            st.session_state.doc = {"name": uploaded.name, "hash": doc_hash, "pages": len(pages),
                                    "chunks": len(chunks), "reused": not newly_indexed}
            status.update(label="Paper ready", state="complete", expanded=False)
        except PaperPilotError as exc:
            st.session_state.doc = None
            st.session_state.failed_hash = doc_hash
            st.session_state.upload_error = str(exc)
            status.update(label="Processing failed", state="error")


def remove_paper():
    try:
        clear_store()
    except PaperPilotError as exc:
        st.error(str(exc))
        return
    st.session_state.doc = None
    reset_results()
    st.session_state.uploader_key += 1
    st.session_state.failed_hash = None
    st.rerun()


def require_doc() -> bool:
    if st.session_state.doc is None:
        st.info("Upload a research paper on the Dashboard first.")
        return False
    return True


def ensure(key: str, fn, label: str, force: bool = False):
    """Return the cached result from session state, or run the agent (with a spinner) and cache it."""
    if st.session_state[key] is None or force:
        with st.spinner(label):
            st.session_state[key] = fn()
    return st.session_state[key]


# ---------------------------------------------------------------- shared renderers
def render_sections(result: dict, expanded_first: bool = True, all_open: bool = False):
    for i, (title, body) in enumerate(result["sections"].items()):
        with st.expander(title, expanded=all_open or (expanded_first and i == 0)):
            st.markdown(body or "_Not mentioned in the paper._")
    if result.get("pages"):
        st.caption("Retrieved from " + format_pages(result["pages"]))


def render_viva_bank(questions: list, key_prefix: str):
    level = st.radio("Show", ["All", "Basic", "Intermediate", "Advanced"], horizontal=True, key=f"{key_prefix}_level")
    for i, q in enumerate(questions, start=1):
        if level != "All" and q["level"] != level:
            continue
        with st.expander(f"Q{i} · [{q['level']}] {q['question']}"):
            st.markdown(f"**Model answer:** {q['answer']}")
            st.caption(f"Topic: {q['topic']}" + (f" · Page {q['page']}" if q["page"] else ""))


def render_answer(result: dict):
    st.markdown(result["answer"])
    if result["reflection"] and not result["reflection"]["supported"]:
        st.warning("Self-check: this answer may not be fully supported by the paper. "
                   + result["reflection"]["issue"])
    if result["pages"]:
        st.caption("Sources: " + format_pages(result["pages"]))
        with st.expander("Retrieved passages"):
            for src in result["sources"]:
                st.markdown(f"**Page {src['page']}** — {src['snippet']}…")
    elif not result["used_retrieval"]:
        st.caption("Answered without searching the paper.")


def download_name(title: str) -> str:
    return safe_filename(title.replace(" ", "_")[:50]) + ".pptx"


# ---------------------------------------------------------------- pages
def page_dashboard():
    st.markdown("""<div class="pp-hero"><h1>📄 PaperPilot AI</h1>
    <p>Paper-to-Presentation &amp; Viva Agent — upload a research paper and get a summary, key concepts,
    a PowerPoint deck and viva practice, powered by local agents (Ollama + LangChain + ChromaDB).</p></div>""",
                unsafe_allow_html=True)

    st.subheader("Upload your paper")
    uploaded = st.file_uploader("Research paper (PDF)", type=["pdf"], key=f"uploader_{st.session_state.uploader_key}")
    if uploaded is not None:
        process_pdf(uploaded)
    if st.session_state.upload_error:
        st.error(st.session_state.upload_error)

    doc = st.session_state.doc
    if doc:
        c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
        c1.metric("Active paper", doc["name"])
        c2.metric("Pages", doc["pages"])
        c3.metric("Chunks", doc["chunks"])
        with c4:
            st.write("")
            if st.button("Remove paper"):
                remove_paper()
        st.success("Ready. " + ("This paper was already indexed, so it was reused." if doc["reused"]
                                else "Indexed in ChromaDB."))
    else:
        st.caption("Status: no paper loaded. Uploading a new PDF replaces the previous one.")

    st.subheader("What you can do")
    cards = [("📝", "Paper Summary", "Structured summary: introduction to future scope."),
             ("🔍", "Key Concepts", "Objectives, research gap, methods, contributions, findings."),
             ("📊", "Generate Slides", "A downloadable 10-slide PPTX in a blue theme."),
             ("🎓", "Viva Preparation", "Likely questions, model answers and graded practice.")]
    for col, (icon, name, text) in zip(st.columns(4), cards):
        with col:
            st.markdown(f'<div class="pp-card"><h4>{icon} {name}</h4><p>{text}</p></div>', unsafe_allow_html=True)
            if st.button("Open", key=f"open_{name}"):
                go_to(name)
    st.caption("Tip: you can also just type a request on the **Ask Your Paper** page, e.g. "
               "“make slides” or “quiz me”. The router picks the right agent.")


def page_summary():
    st.header("📝 Paper Summary")
    if not require_doc():
        return
    if st.button("Generate summary", type="primary"):
        try:
            ensure("summary", summarize_paper, "Summarizer Agent is reading the paper...", force=True)
        except PaperPilotError as exc:
            st.error(str(exc))
    if st.session_state.summary:
        render_sections(st.session_state.summary)
    else:
        st.caption("Click the button to generate the summary.")


def page_concepts():
    st.header("🔍 Key Concepts")
    if not require_doc():
        return
    if st.button("Extract key concepts", type="primary"):
        try:
            ensure("concepts", extract_concepts, "Concept Extractor Agent is working...", force=True)
        except PaperPilotError as exc:
            st.error(str(exc))
    if st.session_state.concepts:
        render_sections(st.session_state.concepts, all_open=True)
    else:
        st.caption("Click the button to extract the concepts.")


def page_slides():
    st.header("📊 Generate Slides")
    if not require_doc():
        return
    if st.button("Generate PPT", type="primary"):
        with st.status("Slide Generator Agent is working...", expanded=True) as status:
            try:
                st.write("Retrieving paper sections, writing slide content, building the PPTX...")
                st.session_state.slides = generate_presentation()
                status.update(label="Slides ready", state="complete", expanded=False)
            except PaperPilotError as exc:
                status.update(label="Slide generation failed", state="error")
                st.error(str(exc))
    result = st.session_state.slides
    if result:
        st.download_button("⬇️ Download PPTX", data=result["pptx_bytes"], file_name=download_name(result["paper_title"]),
                           mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                           type="primary")
        st.subheader("Slide preview")
        for i, slide in enumerate(result["slides"], start=1):
            with st.expander(f"Slide {i}: {slide['title']}"):
                st.markdown("\n".join(f"- {b}" for b in slide["bullets"]))
                if slide["diagram_steps"]:
                    st.caption("Diagram: " + " → ".join(slide["diagram_steps"]))


def start_practice():
    st.session_state.practice = {"index": 0, "results": {}}


def next_question():
    st.session_state.practice["index"] += 1


def page_viva():
    st.header("🎓 Viva Preparation")
    if not require_doc():
        return
    if st.button("Generate viva questions", type="primary"):
        try:
            ensure("viva", generate_viva_questions, "Viva Agent is preparing questions...", force=True)
            st.session_state.practice = None
        except PaperPilotError as exc:
            st.error(str(exc))
    questions = st.session_state.viva
    if not questions:
        st.caption("Generate questions first, then browse them or start practice mode.")
        return
    bank_tab, practice_tab = st.tabs(["Question bank", "Practice mode"])
    with bank_tab:
        render_viva_bank(questions, "bank")
    with practice_tab:
        render_practice(questions)


def render_practice(questions: list):
    practice = st.session_state.practice
    if practice is None:
        st.write("Answer one question at a time and get a score with feedback.")
        st.button("Start practice", on_click=start_practice, type="primary")
        return
    i, total = practice["index"], len(questions)
    if i >= total:
        scores = [r["score"] for r in practice["results"].values()]
        st.subheader("Practice complete")
        st.metric("Average score", f"{sum(scores) / len(scores):.1f} / 10" if scores else "n/a")
        st.button("Restart practice", on_click=start_practice)
        return

    q = questions[i]
    st.progress(i / total, text=f"Question {i + 1} of {total}")
    st.markdown(f'<span class="pp-pill">{q["level"]}</span><span class="pp-pill">{q["topic"]}</span>',
                unsafe_allow_html=True)
    st.markdown(f"### {q['question']}")
    answer = st.text_area("Your answer", key=f"practice_answer_{i}", height=150)
    if st.button("Evaluate answer", type="primary", key=f"eval_{i}"):
        try:
            with st.spinner("Viva Agent is evaluating your answer..."):
                practice["results"][i] = {**evaluate_answer(q["question"], q["answer"], answer), "user_answer": answer}
        except PaperPilotError as exc:
            st.error(str(exc))
    result = practice["results"].get(i)
    if result:
        st.metric("Score", f"{result['score']} / 10")
        if result["verdict"]:
            st.info(result["verdict"])
        if result["feedback"]:
            st.markdown(f"**Feedback:** {result['feedback']}")
        left, right = st.columns(2)
        with left:
            st.markdown("**Strengths**\n" + "\n".join(f"- {s}" for s in result["strengths"]))
        with right:
            st.markdown("**To improve**\n" + "\n".join(f"- {s}" for s in result["improvements"]))
        st.success(f"**Suggested model answer:** {q['answer']}")
        st.button("Next question →" if i < total - 1 else "Finish", on_click=next_question, key=f"next_{i}")


# ---------------------------------------------------------------- chat (intent routing)
def build_assistant_message(decision: dict, prompt: str) -> dict:
    """Run the agent chosen by the router and wrap its output in a chat message dict."""
    intent = decision["intent"]
    base = {"role": "assistant", "decision": decision}
    try:
        if intent == "summary":
            result = ensure("summary", summarize_paper, "Summarizer Agent is reading the paper...")
            return {**base, "kind": "sections", "content": result}
        if intent == "concepts":
            result = ensure("concepts", extract_concepts, "Concept Extractor Agent is working...")
            return {**base, "kind": "sections", "content": result}
        if intent == "slides":
            result = ensure("slides", generate_presentation, "Slide Generator Agent is building your deck...")
            return {**base, "kind": "slides", "content": result}
        if intent == "viva_questions":
            result = ensure("viva", generate_viva_questions, "Viva Agent is preparing questions...")
            return {**base, "kind": "viva", "content": result}
        if intent == "viva_practice":
            ensure("viva", generate_viva_questions, "Viva Agent is preparing questions...")
            start_practice()
            return {**base, "kind": "text", "content": "Practice mode is ready. Opening **Viva Preparation → Practice mode**…",
                    "navigate": "Viva Preparation"}
        with st.spinner("Q&A Agent is thinking..."):
            return {**base, "kind": "answer", "content": answer_question(prompt)}
    except PaperPilotError as exc:
        return {**base, "kind": "error", "content": str(exc)}


def render_message(msg: dict, index: int):
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
            return
        d = msg["decision"]
        st.markdown(f'<span class="pp-pill">Intent: {d["intent"]}</span><span class="pp-pill">Agent: {d["agent"]}</span>'
                    f'<span class="pp-pill">Retrieval: {d["retrieval"]}</span>', unsafe_allow_html=True)
        st.caption(f"Routing: {d['method']} — {d['reason']}")
        kind, content = msg["kind"], msg["content"]
        if kind == "error":
            st.error(content)
        elif kind == "sections":
            render_sections(content, expanded_first=False)
        elif kind == "slides":
            st.success(f"Presentation ready: {len(content['slides'])} slides.")
            st.download_button("⬇️ Download PPTX", data=content["pptx_bytes"], file_name=download_name(content["paper_title"]),
                               mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                               key=f"chat_dl_{index}")
        elif kind == "viva":
            render_viva_bank(content, f"chat_{index}")
        elif kind == "answer":
            if content["used_retrieval"] is False:
                st.caption(f"Controller: {content['reason']}")
            render_answer(content)
        else:
            st.markdown(content)


def page_chat():
    st.header("💬 Ask Your Paper")
    if not require_doc():
        return
    st.caption("Ask a question, or request a summary, slides or viva practice. The router shows which agent handled it.")
    for i, msg in enumerate(st.session_state.chat):
        render_message(msg, i)

    prompt = st.chat_input("Ask something about your paper...")
    if not prompt:
        return
    user_msg = {"role": "user", "content": prompt}
    render_message(user_msg, len(st.session_state.chat))
    with st.spinner("Detecting intent..."):
        decision = detect_intent(prompt)
    st.session_state.last_route = decision
    reply = build_assistant_message(decision, prompt)
    st.session_state.chat += [user_msg, reply]
    if reply.get("navigate"):
        go_to(reply["navigate"])
    render_message(reply, len(st.session_state.chat) - 1)


# ---------------------------------------------------------------- sidebar + routing of pages
with st.sidebar:
    st.markdown("## 📄 PaperPilot AI")
    st.radio("Navigate", PAGES, key="page", label_visibility="collapsed")
    st.divider()
    status = llm_status()
    (st.success if status["model_ready"] else st.error)(status["message"])
    if st.button("Re-check LLM"):
        llm_status.clear()
        st.rerun()
    if st.session_state.doc:
        st.caption(f"Active paper: {st.session_state.doc['name']}")
    if st.session_state.last_route:
        route = st.session_state.last_route
        st.caption(f"Last intent: **{route['intent']}** → {route['agent']}")

{"Dashboard": page_dashboard, "Paper Summary": page_summary, "Key Concepts": page_concepts,
 "Generate Slides": page_slides, "Viva Preparation": page_viva, "Ask Your Paper": page_chat}[st.session_state.page]()
