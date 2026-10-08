"""
PDF Chat - Streamlit UI.

Features: PDF upload (OCR), chat in any language, output-language dropdown,
word-count summary, download chat / summary, share (WhatsApp / Email / copy).

Run:  streamlit run app.py
Pipeline code `rag_pipeline.py` mein hai.
"""
import hashlib
import html
import os
import tempfile
from datetime import datetime
from urllib.parse import quote

import streamlit as st

st.set_page_config(page_title="PDF Chat - Ask your documents", page_icon="📄", layout="wide")

# ---------------------------------------------------------------------------
# Styling (light + dark dono theme mein chalta hai)
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 2rem; max-width: 1100px;}

/* Hero */
.hero {
  padding: 1.5rem 1.7rem; border-radius: 20px; margin-bottom: 1.2rem;
  background: linear-gradient(135deg, rgba(99,102,241,.20), rgba(236,72,153,.15));
  border: 1px solid rgba(128,128,128,.25);
}
.hero h1 {
  margin: 0; font-size: 2.3rem; font-weight: 800; line-height: 1.2;
  background: linear-gradient(90deg, #6366f1, #ec4899);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero p {margin: .4rem 0 .6rem; opacity: .85; font-size: 1.02rem;}

/* Chips */
.chip {
  display: inline-block; padding: .18rem .75rem; margin: .15rem .35rem .15rem 0;
  border-radius: 999px; font-size: .8rem; font-weight: 500;
  border: 1px solid rgba(128,128,128,.35); background: rgba(128,128,128,.12);
}
.chip.accent {border-color: rgba(99,102,241,.6); background: rgba(99,102,241,.18);}

/* Feature cards (landing) */
.feature-grid {
  display: grid; gap: 1rem; margin-top: .5rem;
  grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
}
.feature {
  padding: 1.2rem; border-radius: 18px; transition: all .18s ease;
  border: 1px solid rgba(128,128,128,.25); background: rgba(128,128,128,.07);
}
.feature:hover {transform: translateY(-4px); box-shadow: 0 10px 28px rgba(99,102,241,.28);}
.feature .icon {font-size: 2rem;}
.feature h4 {margin: .45rem 0 .25rem;}
.feature p {margin: 0; font-size: .88rem; opacity: .75;}

/* Sidebar */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(99,102,241,.12), rgba(236,72,153,.07));
}
.brand {font-size: 1.4rem; font-weight: 800; margin-bottom: .2rem;}
.brand-sub {opacity: .7; font-size: .85rem; margin-bottom: .8rem;}

/* Buttons */
div.stButton > button[kind="primary"],
button[data-testid="stBaseButton-primary"] {
  background: linear-gradient(90deg, #6366f1, #ec4899); color: #fff; border: none;
  font-weight: 600; transition: transform .12s ease, box-shadow .12s ease;
}
div.stButton > button[kind="primary"]:hover,
button[data-testid="stBaseButton-primary"]:hover {
  transform: translateY(-1px); box-shadow: 0 6px 18px rgba(236,72,153,.35); color: #fff;
}
div.stButton > button, div.stDownloadButton > button, a[data-testid="stBaseLinkButton-secondary"] {
  border-radius: 12px;
}

/* Chat bubbles */
[data-testid="stChatMessage"] {
  border-radius: 18px; padding: .85rem 1.05rem; margin-bottom: .6rem;
  border: 1px solid rgba(128,128,128,.18); background: rgba(128,128,128,.06);
}
[data-testid="stChatInput"] {border-radius: 16px;}
</style>
""",
    unsafe_allow_html=True,
)


def hero_html(subtitle: str, chips=()) -> str:
    chips_html = "".join(f'<span class="chip">{c}</span>' for c in chips)
    return (
        '<div class="hero"><h1>📄 PDF Chat</h1>'
        f"<p>{subtitle}</p>{chips_html}</div>"
    )


# Pehle UI dikhao, heavy imports (torch etc.) uske baad.
hero_slot = st.empty()
hero_slot.markdown(
    hero_html("Chat with any PDF, in any language."), unsafe_allow_html=True
)

with st.spinner("Loading libraries (first start takes a while)..."):
    from rag_pipeline import (
        AUTO_LANGUAGE,
        OUTPUT_LANGUAGES,
        ProductionRAGPipeline,
        count_words,
        create_gemini_client,
        load_embedding_model,
    )

SUGGESTIONS = [
    "What is this document about?",
    "List the key points",
    "Who are the main people mentioned?",
]


# Heavy objects: sirf ek baar load hote hain (har rerun par nahi).
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedder():
    return load_embedding_model()


@st.cache_resource
def get_client():
    return create_gemini_client()


def load_pdf(uploaded) -> None:
    """PDF sirf tab process hota hai jab file badli ho (rerun par dobara OCR nahi)."""
    data = uploaded.getvalue()
    key = hashlib.md5(data).hexdigest()
    if st.session_state.get("pdf_key") == key:
        return

    rag = ProductionRAGPipeline(embedding_model=get_embedder(), client=get_client())
    rag.pdf_name = uploaded.name

    with st.status("Reading PDF...", expanded=True) as status:
        bar = st.progress(0.0, text="Extracting text...")

        def on_progress(done: int, total: int):
            bar.progress(done / total, text=f"OCR: {done}/{total} pages")

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            rag.extract_and_chunk_pdf(tmp_path, progress_cb=on_progress)
        finally:
            os.remove(tmp_path)

        status.update(label="Building search index...")
        rag.build_vector_index()
        status.update(label=f"Ready: {rag.page_count} pages", state="complete")

    st.session_state.update(pdf_key=key, rag=rag, messages=[], summary=None)


def shown_text(msg: dict, lang: str, rag: ProductionRAGPipeline) -> str:
    """Selected language mein text dikhao; translation cache hoti hai."""
    if lang in (AUTO_LANGUAGE, msg["lang"]):
        return msg["content"]
    cache = msg.setdefault("translations", {})
    if lang not in cache:
        with st.spinner(f"Translating to {lang}..."):
            result = rag.translate_text(msg["content"], lang)
        if result.startswith("⚠️"):
            return result  # failed translation ko cache mat karo
        cache[lang] = result
    return cache[lang]


def cached_text(msg: dict, lang: str) -> str:
    """Export ke liye: translate kiye bina jo text abhi screen par hai wahi do."""
    if lang in (AUTO_LANGUAGE, msg["lang"]):
        return msg["content"]
    return msg.get("translations", {}).get(lang, msg["content"])


def pages_chips(pages) -> str:
    if not pages:
        return ""
    return "".join(f'<span class="chip">📄 p. {p}</span>' for p in pages)


def render_assistant(text: str, pages, sources) -> None:
    st.markdown(text)
    if pages:
        st.markdown(pages_chips(pages), unsafe_allow_html=True)
    if sources:
        with st.expander("🔎 View sources"):
            for src in sources:
                snippet = src["text"].strip().replace("\n", " ")
                if len(snippet) > 350:
                    snippet = snippet[:350].rstrip() + "..."
                st.caption(f"**Page {src['page']}**: {snippet}")


def build_chat_export(rag, msgs, lang) -> str:
    lines = [
        f"PDF Chat - {rag.pdf_name}",
        f"Exported: {datetime.now():%d %b %Y, %H:%M}",
        "=" * 50,
        "",
    ]
    for m in msgs:
        if m["role"] == "user":
            lines.append(f"Q: {m['content']}")
        else:
            lines.append(f"A: {cached_text(m, lang)}")
            if m.get("pages"):
                lines.append("(Pages: " + ", ".join(map(str, m["pages"])) + ")")
        lines.append("")
    return "\n".join(lines)


def share_widget(label: str, text: str, title: str) -> None:
    """WhatsApp / Email link + poora text copy karne ke liye box."""
    limit = 1500
    with st.popover(label, use_container_width=True):
        st.caption("Share via")
        short = text if len(text) <= limit else text[:limit].rstrip() + "..."
        body = quote(f"{title}\n\n{short}")
        c1, c2 = st.columns(2)
        c1.link_button("💬 WhatsApp", f"https://wa.me/?text={body}", use_container_width=True)
        c2.link_button(
            "✉️ Email",
            f"mailto:?subject={quote(title)}&body={body}",
            use_container_width=True,
        )
        if len(text) > limit:
            st.caption(f"Link mein pehle {limit} characters jayenge. Poora text neeche se copy karo.")
        st.caption("Copy full text")
        st.code(text, language=None, wrap_lines=True)


def safe_stem(name: str) -> str:
    stem = os.path.splitext(name or "document")[0].strip()
    return stem.replace(" ", "_") or "document"


@st.dialog("📝 Summarize PDF")
def summary_dialog(lang: str) -> None:
    rag = st.session_state["rag"]
    st.write("How many words should the summary be?")
    words = st.number_input(
        "Number of words", min_value=20, max_value=2000, value=150, step=10
    )
    where = "the document's language" if lang == AUTO_LANGUAGE else lang
    st.caption(f"The summary will be written in {where}, in about {int(words)} words.")
    if st.button("✨ Generate summary", type="primary", use_container_width=True):
        with st.spinner("Writing your summary..."):
            text, pages = rag.summarize(lang, words=int(words))
        if text.startswith("⚠️"):
            st.error(text)
            return
        st.session_state["summary"] = {
            "content": text,
            "lang": lang,
            "pages": pages,
            "requested": int(words),
        }
        st.rerun()  # dialog band ho jata hai


# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown(
        '<div class="brand">📄 PDF Chat</div>'
        '<div class="brand-sub">Ask, summarize and share, in any language</div>',
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader("Upload PDF (any language)", type=["pdf"])
    lang = st.selectbox("Output language", OUTPUT_LANGUAGES, key="out_lang")
    actions = st.container()

if uploaded:
    load_pdf(uploaded)
elif "rag" in st.session_state:
    # File hata di gayi -> sab reset
    for k in ("pdf_key", "rag", "messages", "summary"):
        st.session_state.pop(k, None)

rag = st.session_state.get("rag")

# ---------------- Landing (PDF abhi upload nahi hua) ----------------
if rag is None:
    st.markdown(
        """
<div class="feature-grid">
  <div class="feature"><div class="icon">🌐</div><h4>Any language</h4>
    <p>Upload a PDF in Hindi, Odia, English and more. Ask in your own language.</p></div>
  <div class="feature"><div class="icon">🔍</div><h4>Scanned PDFs too</h4>
    <p>Built-in OCR reads scanned pages and images, not just typed text.</p></div>
  <div class="feature"><div class="icon">📝</div><h4>Summary in N words</h4>
    <p>Choose exactly how long the summary should be.</p></div>
  <div class="feature"><div class="icon">⬇️</div><h4>Download &amp; share</h4>
    <p>Save your chat or summary, or share it on WhatsApp and Email.</p></div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.stop()

# ---------------- Hero (PDF load ho chuka) ----------------
chips = [f"📄 {rag.page_count} pages", f"🧩 {len(rag.chunks)} sections indexed"]
if rag.ocr_pages:
    chips.append(f"🔍 OCR on {len(rag.ocr_pages)} pages")
hero_slot.markdown(
    hero_html(f"<b>{html.escape(rag.pdf_name or 'document')}</b> is ready. Ask anything!", chips),
    unsafe_allow_html=True,
)
if getattr(rag, "garbled_text", False):
    st.warning("Some text in this PDF looks garbled, so answers may be less accurate.")

# ---------------- Sidebar actions ----------------
with actions:
    st.divider()
    if st.button("📝 Summarize PDF", type="primary", use_container_width=True):
        summary_dialog(lang)

# ---------------- Summary card ----------------
summary = st.session_state.get("summary")
if summary:
    summary_text = shown_text(summary, lang, rag)
    with st.container(border=True):
        st.markdown(
            "### 📝 Summary "
            f'<span class="chip accent">{count_words(summary_text)} words</span>'
            f'<span class="chip">requested {summary["requested"]}</span>',
            unsafe_allow_html=True,
        )
        st.markdown(summary_text)
        if summary.get("pages"):
            st.markdown(pages_chips(summary["pages"]), unsafe_allow_html=True)
        b1, b2, b3 = st.columns(3)
        b1.download_button(
            "⬇️ Download summary",
            data=f"Summary - {rag.pdf_name}\n\n{summary_text}\n",
            file_name=f"{safe_stem(rag.pdf_name)}_summary.txt",
            mime="text/plain",
            use_container_width=True,
        )
        with b2:
            share_widget("🔗 Share summary", summary_text, f"Summary of {rag.pdf_name}")
        if b3.button("✖ Close summary", use_container_width=True):
            st.session_state["summary"] = None
            st.rerun()

# ---------------- Chat ----------------
question = st.chat_input("Ask anything about the PDF (any language)")
if not question:
    question = st.session_state.pop("pending_q", None)

messages = st.session_state["messages"]

if not messages and not question:
    st.markdown("##### 💡 Try asking")
    cols = st.columns(len(SUGGESTIONS))
    for col, q in zip(cols, SUGGESTIONS):
        if col.button(q, use_container_width=True):
            st.session_state["pending_q"] = q
            st.rerun()

for msg in messages:
    avatar = "🧑" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            render_assistant(shown_text(msg, lang, rag), msg.get("pages"), msg.get("sources"))

if question:
    messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Searching..."):
            # Dropdown Auto ho to question ki language mein, warna selected language mein
            answer, pages, sources = rag.ask(question, output_language=lang)
        render_assistant(answer, pages, sources)

    messages.append(
        {
            "role": "assistant",
            "content": answer,
            "lang": lang,
            "pages": pages,
            "sources": sources,
        }
    )

# ---------------- Sidebar: chat export (naye message ke baad) ----------------
if messages:
    chat_txt = build_chat_export(rag, messages, lang)
    with actions:
        st.download_button(
            "⬇️ Download chat",
            data=chat_txt,
            file_name=f"{safe_stem(rag.pdf_name)}_chat.txt",
            mime="text/plain",
            use_container_width=True,
        )
        share_widget("🔗 Share chat", chat_txt, f"Chat about {rag.pdf_name}")
        if st.button("🗑 Clear chat", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()
