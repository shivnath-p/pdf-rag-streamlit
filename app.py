"""
Streamlit UI example: PDF upload + chat + language dropdown (auto-translate).

Run:  streamlit run app.py
Pipeline code `rag_pipeline.py` mein hai.
"""
import hashlib
import os
import tempfile

import streamlit as st

st.set_page_config(page_title="PDF Chat", layout="wide")
st.title("PDF Chat")  # pehle UI dikhao, heavy imports (torch etc.) uske baad

with st.spinner("Loading libraries (first start takes a while)..."):
    from rag_pipeline import (
        AUTO_LANGUAGE,
        OUTPUT_LANGUAGES,
        ProductionRAGPipeline,
        create_gemini_client,
        load_embedding_model,
    )


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

    st.session_state.update(pdf_key=key, rag=rag, messages=[])


def shown_text(msg: dict, lang: str, rag: ProductionRAGPipeline) -> str:
    """Selected language mein answer dikhao; translation cache hoti hai."""
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


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Settings")
    uploaded = st.file_uploader("Upload PDF (any language)", type=["pdf"])
    lang = st.selectbox("Output language", OUTPUT_LANGUAGES, key="out_lang")

if uploaded:
    load_pdf(uploaded)

rag = st.session_state.get("rag")
if rag is None:
    st.stop()

# ---------------- Chat ----------------
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            st.markdown(shown_text(msg, lang, rag))
            if msg.get("pages"):
                st.caption("Pages: " + ", ".join(map(str, msg["pages"])))

question = st.chat_input("Ask anything about the PDF (any language)")
if question:
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            # Dropdown Auto ho to question ki language mein, warna selected language mein
            answer, pages, sources = rag.ask(question, output_language=lang)
        st.markdown(answer)
        if pages:
            st.caption("Pages: " + ", ".join(map(str, pages)))

    st.session_state["messages"].append(
        {"role": "assistant", "content": answer, "lang": lang, "pages": pages}
    )
