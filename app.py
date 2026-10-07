import hashlib
import html
import json
import os
import re
import secrets
import sqlite3
import tempfile
from datetime import datetime, timezone
from urllib.parse import quote

import streamlit as st

from rag_pipeline import (
    ProductionRAGPipeline,
    create_gemini_client,
    load_embedding_model,
)

# ============================================================
# Page config
# ============================================================

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Database (shared conversations)
# NOTE: Streamlit Cloud's disk is ephemeral. Links stop working after a
# restart/redeploy. For permanent links use Supabase/Postgres/Firestore.
# ============================================================

DB_FILE = "documind.db"
SHARE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


def get_db():
    connection = sqlite3.connect(DB_FILE, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    return connection


def init_database():
    connection = get_db()
    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS shared_conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                share_id TEXT UNIQUE NOT NULL,
                document_name TEXT,
                messages_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.commit()
    finally:
        connection.close()


init_database()

# ============================================================
# Cached heavy resources (shared safely: they hold no document state)
# ============================================================


@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return load_embedding_model()


@st.cache_resource(show_spinner=False)
def get_client():
    return create_gemini_client()


# ============================================================
# Styling
# ============================================================

LOGO_SVG = """
<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M6 2.8C6 2.358 6.358 2 6.8 2H14L19 7V21.2C19 21.642 18.642 22 18.2 22H6.8C6.358 22 6 21.642 6 21.2V2.8Z" fill="white" opacity="0.96"/>
  <path d="M14 2V7H19" stroke="#6366F1" stroke-width="1.5" stroke-linejoin="round"/>
  <path d="M9 12H16M9 15.5H16M9 8.5H11" stroke="#6366F1" stroke-width="1.5" stroke-linecap="round"/>
</svg>
"""


def logo_html(size=42):
    return (
        f'<div class="logo" style="width:{size}px;height:{size}px;min-width:{size}px;">'
        + LOGO_SVG.format(s=int(size * 0.55))
        + "</div>"
    )


st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; }

.stApp {
    background:
        radial-gradient(circle at 8% -5%, rgba(99,102,241,.22), transparent 32%),
        radial-gradient(circle at 95% 5%, rgba(236,72,153,.12), transparent 30%),
        #0b0e14;
}

.block-container { padding-top: 2rem; max-width: 1150px; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #11151f, #0d1119);
    border-right: 1px solid #232a38;
}

.logo {
    border-radius: 14px; display: flex; align-items: center; justify-content: center;
    background: linear-gradient(135deg, #6366f1, #8b5cf6 55%, #ec4899);
    box-shadow: 0 10px 30px rgba(99,102,241,.35);
}

/* Hero */
.hero {
    display: flex; align-items: center; gap: 18px; padding: 22px 26px; margin-bottom: 22px;
    background: linear-gradient(135deg, rgba(99,102,241,.16), rgba(139,92,246,.07));
    border: 1px solid rgba(139,92,246,.28); border-radius: 22px;
    backdrop-filter: blur(8px);
}
.hero-title {
    font-size: 36px; font-weight: 800; letter-spacing: -1.2px; line-height: 1.1;
    background: linear-gradient(90deg, #fff, #c4b5fd 60%, #f9a8d4);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-sub { color: #9aa3b5; font-size: 15px; margin-top: 6px; }

/* Metric cards */
.metric-card {
    background: linear-gradient(145deg, rgba(30,35,48,.92), rgba(19,23,32,.92));
    border: 1px solid #293140; border-radius: 16px; padding: 16px 18px; min-height: 92px;
    transition: transform .2s ease, border-color .2s ease;
}
.metric-card:hover { transform: translateY(-3px); border-color: #6366f1; }
.metric-label { color: #8992a3; font-size: 11px; letter-spacing: .08em; font-weight: 600; margin-bottom: 8px; }
.metric-value { font-size: 17px; font-weight: 650; color: #f1f3f7; word-break: break-word; }

/* Chat */
[data-testid="stChatMessage"] {
    border-radius: 18px; border: 1px solid #232a38;
    background: rgba(21,26,36,.65); padding: 14px 16px; margin-bottom: 10px;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    background: #151a24; border-radius: 10px; padding: 8px 18px; border: 1px solid #252c3a;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6); color: #fff !important; border-color: transparent;
}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

/* Buttons */
.stButton > button, .stDownloadButton > button, .stLinkButton > a {
    border-radius: 12px; font-weight: 600; transition: all .2s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover { transform: translateY(-2px); }
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6); border: none;
    box-shadow: 0 8px 22px rgba(99,102,241,.35);
}

/* Upload */
[data-testid="stFileUploader"] section {
    background: #151a24; border: 1.5px dashed #4a5370; border-radius: 14px;
}

/* Source snippets */
.source-pill {
    display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: 12px; font-weight: 600;
    background: rgba(99,102,241,.15); border: 1px solid rgba(99,102,241,.4); color: #c7d2fe;
}
.snippet {
    background: #121722; border-left: 3px solid #6366f1; border-radius: 8px;
    padding: 10px 12px; margin: 8px 0; color: #b6bfd0; font-size: 13px; line-height: 1.55;
}

/* Empty state */
.empty-card {
    text-align: center; padding: 60px 25px; margin-top: 10px;
    background: rgba(20,24,34,.7); border: 1px solid #282f3d; border-radius: 24px;
}
.empty-title { font-size: 28px; font-weight: 800; margin-top: 16px; }
.empty-description { color: #9aa3b5; font-size: 15px; max-width: 560px; margin: 10px auto 0; }
.feature-card {
    background: #131822; border: 1px solid #262d3b; border-radius: 16px; padding: 20px; height: 100%;
}
.feature-card h4 { margin: 0 0 6px 0; }
.feature-card p { color: #9aa3b5; font-size: 14px; margin: 0; }

.share-card {
    background: linear-gradient(145deg, #151a24, #121722);
    border: 1px solid #293140; border-radius: 16px; padding: 16px 18px; margin: 12px 0 4px;
}

footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# Session state
# ============================================================

defaults = {
    "pipeline": None,
    "pdf_hash": None,
    "pdf_processed": False,
    "pdf_name": None,
    "chat_history": [],
    "summary": "",
    "summary_pages": [],
    "share_id": None,
    "pending_question": None,
}
for key, value in defaults.items():
    st.session_state.setdefault(key, value)

# ============================================================
# Helpers
# ============================================================


def pages_label(pages):
    return ", ".join(str(p) for p in pages)


def render_sources(sources):
    """Show retrieved snippets in an expander."""
    if not sources:
        return
    with st.expander(f"🔎 View {len(sources)} source passages"):
        for s in sources:
            snippet = html.escape(s["text"][:420]) + ("…" if len(s["text"]) > 420 else "")
            st.markdown(
                f'<span class="source-pill">Page {s["page"]}</span> '
                f'<span style="color:#7c859a;font-size:12px;">relevance {s["score"]:.2f}</span>'
                f'<div class="snippet">{snippet}</div>',
                unsafe_allow_html=True,
            )


def create_chat_text(history=None, doc_name=None):
    history = st.session_state.chat_history if history is None else history
    doc_name = st.session_state.pdf_name if doc_name is None else doc_name

    if not history:
        return "No conversation available."

    lines = ["DocuMind AI", "=" * 45, ""]
    if doc_name:
        lines += [f"Document: {doc_name}", ""]

    for m in history:
        if m["role"] == "user":
            lines += ["USER", "-" * 20, m["content"]]
        else:
            lines += ["ASSISTANT", "-" * 20, m["content"]]
            if m.get("pages"):
                lines.append("Sources: Pages " + pages_label(m["pages"]))
        lines.append("")
    return "\n".join(lines)


def create_chat_markdown():
    history = st.session_state.chat_history
    lines = ["# DocuMind AI", ""]
    if st.session_state.pdf_name:
        lines += [f"**Document:** {st.session_state.pdf_name}", ""]

    for m in history:
        if m["role"] == "user":
            lines += ["### 👤 User", "", m["content"]]
        else:
            lines += ["### 🤖 Assistant", "", m["content"]]
            if m.get("pages"):
                lines += ["", "**Sources:** Pages " + pages_label(m["pages"])]
        lines.append("")
    return "\n".join(lines)


def save_shared_chat():
    if not st.session_state.chat_history:
        return None

    share_id = secrets.token_urlsafe(12)
    # Store only what the shared view needs (no source passages: they are
    # raw document text and shouldn't leak through a public link).
    messages = [
        {"role": m["role"], "content": m["content"], "pages": m.get("pages", [])}
        for m in st.session_state.chat_history
    ]

    connection = get_db()
    try:
        connection.execute(
            "INSERT INTO shared_conversations "
            "(share_id, document_name, messages_json, created_at) VALUES (?, ?, ?, ?)",
            (
                share_id,
                st.session_state.pdf_name,
                json.dumps(messages),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        connection.commit()
    finally:
        connection.close()

    st.session_state.share_id = share_id
    return share_id


def get_shared_chat(share_id):
    if not SHARE_ID_PATTERN.match(share_id or ""):
        return None
    connection = get_db()
    try:
        return connection.execute(
            "SELECT * FROM shared_conversations WHERE share_id = ?", (share_id,)
        ).fetchone()
    finally:
        connection.close()


def get_share_url(share_id):
    base_url = ""
    try:
        base_url = getattr(st.context, "url", "") or ""
    except Exception:
        pass
    base_url = base_url.split("?")[0]
    return f"{base_url}?share={share_id}"


# ============================================================
# Shared conversation view
# ============================================================

share_parameter = st.query_params.get("share")

if share_parameter:
    shared = get_shared_chat(share_parameter)

    st.markdown(
        f"""
        <div class="hero">
            {logo_html(52)}
            <div>
                <div class="hero-title">DocuMind AI</div>
                <div class="hero-sub">Shared PDF conversation</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not shared:
        st.warning("This shared conversation was not found. The link may be wrong or expired.")
        st.stop()

    messages = json.loads(shared["messages_json"])
    st.caption(f"📄 Document: {shared['document_name']}")

    for m in messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant" and m.get("pages"):
                st.caption("📌 Sources: Pages " + pages_label(m["pages"]))

    st.download_button(
        "📥 Download Conversation",
        data=create_chat_text(messages, shared["document_name"]),
        file_name="shared_pdf_chat.txt",
        mime="text/plain",
        use_container_width=True,
    )
    st.caption("This conversation was shared from DocuMind AI.")
    st.stop()

# ============================================================
# Pipeline (one per session; heavy models are cached and shared)
# ============================================================

if st.session_state.pipeline is None:
    with st.spinner("Loading document intelligence..."):
        try:
            st.session_state.pipeline = ProductionRAGPipeline(
                embedding_model=get_embedding_model(),
                client=get_client(),
            )
        except Exception as e:
            st.error("Unable to load the AI pipeline.")
            st.code(str(e))
            st.stop()

pipeline = st.session_state.pipeline

# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:22px;">
            {logo_html(44)}
            <div>
                <div style="font-size:20px;font-weight:800;">DocuMind</div>
                <div style="color:#858e9f;font-size:12px;">PDF Intelligence</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 📂 Document")
    uploaded_file = st.file_uploader(
        "Upload PDF", type=["pdf"], help="Upload the PDF you want to analyse."
    )

    st.divider()
    st.markdown("### ⚙️ Retrieval")
    top_k = st.slider("Relevant sections", min_value=3, max_value=10, value=5)

    st.divider()
    st.markdown(
        """
**How it works**

📄 PDF → ✂️ Chunks → 🧠 Embeddings → 🔎 FAISS → 🤖 Gemini
"""
    )
    st.divider()
    st.caption("DocuMind AI • RAG Document Assistant")

# ============================================================
# Hero
# ============================================================

st.markdown(
    f"""
    <div class="hero">
        {logo_html(60)}
        <div>
            <div class="hero-title">DocuMind AI</div>
            <div class="hero-sub">Ask questions. Understand documents. Get answers grounded in your PDF.</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Process PDF
# ============================================================

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    current_hash = hashlib.sha256(file_bytes).hexdigest()

    if current_hash != st.session_state.pdf_hash:
        # New document: start from a clean slate.
        st.session_state.pdf_processed = False
        st.session_state.chat_history = []
        st.session_state.summary = ""
        st.session_state.summary_pages = []
        st.session_state.share_id = None

        temp_pdf_path = None
        with st.spinner("Reading PDF and building search index..."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(file_bytes)
                    temp_pdf_path = tmp.name

                pipeline.extract_and_chunk_pdf(temp_pdf_path)
                pipeline.build_vector_index()
                pipeline.pdf_name = uploaded_file.name

                st.session_state.pdf_name = uploaded_file.name
                st.session_state.pdf_hash = current_hash
                st.session_state.pdf_processed = True
            except Exception as e:
                st.session_state.pdf_hash = None
                st.error("Could not process this PDF.")
                st.code(str(e))
                st.stop()
            finally:
                if temp_pdf_path and os.path.exists(temp_pdf_path):
                    os.remove(temp_pdf_path)

elif st.session_state.pdf_processed:
    # User removed the file from the uploader: reset everything.
    pipeline.chunks, pipeline.index = [], None
    for key in ("pdf_hash", "pdf_name", "share_id"):
        st.session_state[key] = None
    st.session_state.pdf_processed = False
    st.session_state.chat_history = []
    st.session_state.summary = ""
    st.session_state.summary_pages = []

# ============================================================
# Document loaded
# ============================================================

if st.session_state.pdf_processed:
    safe_name = html.escape(st.session_state.pdf_name or "")
    ai_status = "🟢 Gemini Connected" if pipeline.client else "🔴 API Key Missing"

    cards = [
        ("DOCUMENT", f"📄 {safe_name}"),
        ("PAGES", f"📑 {pipeline.page_count}"),
        ("SEARCHABLE SECTIONS", f"🧩 {len(pipeline.chunks)}"),
        ("AI STATUS", ai_status),
    ]
    for col, (label, value) in zip(st.columns(4), cards):
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">{label}</div>'
                f'<div class="metric-value">{value}</div></div>',
                unsafe_allow_html=True,
            )

    st.write("")
    chat_tab, summary_tab = st.tabs(["💬 Chat", "📝 Summary"])

    # --------------------------------------------------------
    # Chat tab
    # --------------------------------------------------------
    with chat_tab:
        st.subheader("Chat with your PDF")
        st.caption("Ask anything about the information in your uploaded document.")

        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant":
                    if message.get("pages"):
                        st.caption("📌 Sources: Pages " + pages_label(message["pages"]))
                    render_sources(message.get("sources"))

        # Suggested prompts for an empty chat.
        if not st.session_state.chat_history:
            st.markdown("**Try asking:**")
            suggestions = [
                "What is this document about?",
                "List the key points and important dates.",
                "What are the main terms or conditions?",
            ]
            for col, text in zip(st.columns(3), suggestions):
                with col:
                    if st.button(text, key=f"sg_{text}", use_container_width=True):
                        st.session_state.pending_question = text
                        st.rerun()

        question = st.chat_input("Ask something about your PDF...")
        if not question and st.session_state.pending_question:
            question = st.session_state.pending_question
        st.session_state.pending_question = None

        if question:
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Searching the document..."):
                    try:
                        answer, pages, sources = pipeline.ask(question, top_k=top_k)
                    except Exception as e:
                        answer, pages, sources = f"⚠️ Something went wrong: {e}", [], []

                st.markdown(answer)
                if pages:
                    st.caption("📌 Sources: Pages " + pages_label(pages))
                render_sources(sources)

            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer, "pages": pages, "sources": sources}
            )

        # ----------------------------------------------------
        # Share & export
        # ----------------------------------------------------
        if st.session_state.chat_history:
            st.divider()
            st.subheader("📤 Share & Export")

            chat_text = create_chat_text()
            chat_markdown = create_chat_markdown()

            d1, d2, d3 = st.columns(3)
            with d1:
                st.download_button(
                    "📥 Download TXT", data=chat_text, file_name="documind_chat.txt",
                    mime="text/plain", use_container_width=True,
                )
            with d2:
                st.download_button(
                    "📥 Download Markdown", data=chat_markdown, file_name="documind_chat.md",
                    mime="text/markdown", use_container_width=True,
                )
            with d3:
                if st.button("🧹 Clear Chat", use_container_width=True):
                    st.session_state.chat_history = []
                    st.session_state.share_id = None
                    st.rerun()

            if st.button("🔗 Create Shareable Link", type="primary", use_container_width=True):
                with st.spinner("Creating share link..."):
                    save_shared_chat()

            if st.session_state.share_id:
                share_url = get_share_url(st.session_state.share_id)

                st.markdown(
                    """
                    <div class="share-card">
                        <b>🔗 Your shared conversation</b><br>
                        <span style="color:#9aa3b5;font-size:13px;">
                        Anyone with this link can read this conversation
                        (questions and answers). Share it only with people you trust.
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.code(share_url, language="text")  # has a built-in copy icon

                wa_url = "https://wa.me/?text=" + quote(
                    "Here is my PDF conversation:\n\n" + share_url
                )
                mail_url = (
                    "mailto:?subject=" + quote("Shared PDF Conversation")
                    + "&body=" + quote("Here is the PDF conversation:\n\n" + share_url)
                )

                s1, s2 = st.columns(2)
                with s1:
                    st.link_button("💬 Share on WhatsApp", wa_url, use_container_width=True)
                with s2:
                    st.link_button("📧 Share by Email", mail_url, use_container_width=True)

            with st.expander("📋 View complete conversation"):
                st.text_area(
                    "Conversation", value=chat_text, height=300, label_visibility="collapsed"
                )

    # --------------------------------------------------------
    # Summary tab
    # --------------------------------------------------------
    with summary_tab:
        st.subheader("📝 Executive Summary")
        st.caption("Generate a professional overview of the uploaded document.")

        if st.button("✨ Generate Summary", type="primary", use_container_width=True):
            with st.spinner("Analysing your document..."):
                summary, pages = pipeline.summarize()
            st.session_state.summary = summary
            st.session_state.summary_pages = pages

        if st.session_state.summary:
            st.markdown(st.session_state.summary)

            if st.session_state.summary_pages:
                st.divider()
                st.caption(
                    "Summary based on pages: " + pages_label(st.session_state.summary_pages)
                )

            st.download_button(
                "📥 Download Summary", data=st.session_state.summary,
                file_name="documind_summary.txt", mime="text/plain",
                use_container_width=True,
            )

# ============================================================
# Empty state
# ============================================================

else:
    st.markdown(
        f"""
        <div class="empty-card">
            <div style="display:flex;justify-content:center;">{logo_html(72)}</div>
            <div class="empty-title">Your documents, understood.</div>
            <div class="empty-description">
                Upload a PDF from the sidebar to ask questions, find important
                details and generate summaries with AI-powered semantic search.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    features = [
        ("🔎 Smart Search", "Semantic search finds meaning, not just keywords."),
        ("💬 Document Chat", "Natural questions, answers grounded in your PDF with page citations."),
        ("🔗 Easy Sharing", "Export a chat or share it with a simple link."),
    ]
    for col, (title, text) in zip(st.columns(3), features):
        with col:
            st.markdown(
                f'<div class="feature-card"><h4>{title}</h4><p>{text}</p></div>',
                unsafe_allow_html=True,
            )
