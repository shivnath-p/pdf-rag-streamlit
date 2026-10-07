import os
import hashlib
import tempfile
from urllib.parse import quote

import streamlit as st

from rag_pipeline import ProductionRAGPipeline


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="PDF Intelligence Workspace",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# Styling
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #0e1117;
    }

    section[data-testid="stSidebar"] {
        background-color: #151922;
        border-right: 1px solid #292f3b;
    }

    .main-title {
        font-size: 42px;
        font-weight: 700;
        letter-spacing: -1px;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #9ca3af;
        font-size: 17px;
        margin-bottom: 30px;
    }

    .stButton > button {
        border-radius: 9px;
        font-weight: 600;
    }

    [data-testid="stFileUploader"] {
        background-color: #171b24;
        border: 1px dashed #414957;
        border-radius: 12px;
        padding: 8px;
    }

    [data-testid="stChatMessage"] {
        border-radius: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# Chat export helpers
# ============================================================

def create_chat_text():

    history = st.session_state.chat_history

    if not history:
        return "No conversation available."

    lines = [
        "PDF Intelligence Workspace",
        "=" * 40,
        ""
    ]

    if st.session_state.pdf_name:
        lines.append(
            f"Document: {st.session_state.pdf_name}"
        )
        lines.append("")

    for message in history:

        if message["role"] == "user":

            lines.append("USER")
            lines.append("-" * 20)
            lines.append(message["content"])

        else:

            lines.append("ASSISTANT")
            lines.append("-" * 20)
            lines.append(message["content"])

            pages = message.get(
                "pages",
                []
            )

            if pages:

                lines.append(
                    "Sources: Pages "
                    + ", ".join(
                        str(page)
                        for page in pages
                    )
                )

        lines.append("")

    return "\n".join(lines)


def create_chat_markdown():

    history = st.session_state.chat_history

    if not history:
        return (
            "# PDF Intelligence Workspace\n\n"
            "No conversation available."
        )

    lines = [
        "# PDF Intelligence Workspace",
        ""
    ]

    if st.session_state.pdf_name:

        lines.append(
            f"**Document:** {st.session_state.pdf_name}"
        )

        lines.append("")

    for message in history:

        if message["role"] == "user":

            lines.append("### 👤 User")
            lines.append("")
            lines.append(message["content"])

        else:

            lines.append("### 🤖 Assistant")
            lines.append("")
            lines.append(message["content"])

            pages = message.get(
                "pages",
                []
            )

            if pages:

                lines.append("")
                lines.append(
                    "**Sources:** Pages "
                    + ", ".join(
                        str(page)
                        for page in pages
                    )
                )

        lines.append("")

    return "\n".join(lines)


# ============================================================
# Session state
# ============================================================

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

if "pdf_hash" not in st.session_state:
    st.session_state.pdf_hash = None

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "summary" not in st.session_state:
    st.session_state.summary = ""

if "summary_pages" not in st.session_state:
    st.session_state.summary_pages = []


# ============================================================
# Header
# ============================================================

st.markdown(
    '<div class="main-title">'
    '📄 PDF Intelligence Workspace'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload a document, ask questions, and get answers '
    'grounded in your PDF.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.markdown("## 📂 Document")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        help="Upload the document you want to analyse."
    )

    st.divider()

    st.markdown("### ⚙️ Settings")

    top_k = st.slider(
        "Relevant sections",
        min_value=3,
        max_value=10,
        value=5
    )

    st.divider()

    st.markdown(
        """
        ### How it works

        📄 Upload PDF

        ↓

        ✂️ Chunk document

        ↓

        🧠 Create embeddings

        ↓

        🔎 FAISS retrieval

        ↓

        🤖 Gemini answer
        """
    )

    st.divider()

    st.caption(
        "PDF Intelligence Workspace"
    )

    st.caption(
        "RAG-powered document analysis"
    )


# ============================================================
# Load RAG pipeline
# ============================================================

if st.session_state.pipeline is None:

    with st.spinner(
        "Loading document intelligence model..."
    ):

        try:

            st.session_state.pipeline = (
                load_pipeline()
            )

        except Exception as e:

            st.error(
                "Unable to load the AI pipeline."
            )

            st.code(
                str(e)
            )

            st.stop()


pipeline = st.session_state.pipeline


# ============================================================
# Process uploaded PDF
# ============================================================

if uploaded_file is not None:

    file_bytes = uploaded_file.getvalue()

    current_hash = hashlib.md5(
        file_bytes
    ).hexdigest()

    # Don't rebuild the entire vector index every time
    # Streamlit reruns the application.
    if current_hash != st.session_state.pdf_hash:

        st.session_state.pdf_processed = False
        st.session_state.chat_history = []
        st.session_state.summary = ""
        st.session_state.summary_pages = []

        with st.spinner(
            "Reading PDF and building search index..."
        ):

            temp_pdf_path = None

            try:

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as tmp_file:

                    tmp_file.write(
                        file_bytes
                    )

                    temp_pdf_path = (
                        tmp_file.name
                    )

                pipeline.extract_and_chunk_pdf(
                    temp_pdf_path
                )

                pipeline.build_vector_index()

                pipeline.pdf_name = (
                    uploaded_file.name
                )

                st.session_state.pdf_name = (
                    uploaded_file.name
                )

                st.session_state.pdf_hash = (
                    current_hash
                )

                st.session_state.pdf_processed = (
                    True
                )

            except Exception as e:

                st.error(
                    "Could not process this PDF."
                )

                st.code(
                    str(e)
                )

                st.stop()

            finally:

                if (
                    temp_pdf_path
                    and os.path.exists(temp_pdf_path)
                ):

                    os.remove(
                        temp_pdf_path
                    )


# ============================================================
# Main UI
# ============================================================

if st.session_state.pdf_processed:

    # --------------------------------------------------------
    # Document information
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown("### 📄 Document")

        st.caption(
            st.session_state.pdf_name
        )

    with col2:

        st.markdown("### 🧩 Chunks")

        st.caption(
            f"{len(pipeline.chunks)} searchable sections"
        )

    with col3:

        st.markdown("### 🤖 AI Status")

        if pipeline.client:

            st.success(
                "Gemini Connected"
            )

        else:

            st.error(
                "API Key Missing"
            )

    st.divider()


    # --------------------------------------------------------
    # Tabs
    # --------------------------------------------------------

    chat_tab, summary_tab = st.tabs(
        [
            "💬 Chat with PDF",
            "📝 Document Summary"
        ]
    )


    # ========================================================
    # CHAT
    # ========================================================

    with chat_tab:

        st.subheader(
            "Ask your document"
        )

        st.caption(
            "Ask questions based on the content "
            "of your uploaded PDF."
        )

        # Display previous conversation.
        for message in st.session_state.chat_history:

            with st.chat_message(
                message["role"]
            ):

                st.markdown(
                    message["content"]
                )

                if (
                    message["role"] == "assistant"
                    and message.get("pages")
                ):

                    st.caption(
                        "📌 Sources: Pages "
                        + ", ".join(
                            str(page)
                            for page in message["pages"]
                        )
                    )


        # New question
        question = st.chat_input(
            "Ask something about your PDF..."
        )

        if question:

            st.session_state.chat_history.append(
                {
                    "role": "user",
                    "content": question
                }
            )

            with st.chat_message("user"):

                st.markdown(
                    question
                )


            with st.chat_message("assistant"):

                with st.spinner(
                    "Searching the document..."
                ):

                    answer, pages = pipeline.ask(
                        question,
                        top_k=top_k
                    )

                st.markdown(
                    answer
                )

                if pages:

                    st.caption(
                        "📌 Sources: Pages "
                        + ", ".join(
                            str(page)
                            for page in pages
                        )
                    )

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "pages": pages
                }
            )


        # ----------------------------------------------------
        # Download and share
        # ----------------------------------------------------

        if st.session_state.chat_history:

            st.divider()

            st.subheader(
                "📤 Share & Export"
            )

            chat_text = create_chat_text()
            chat_markdown = create_chat_markdown()

            col1, col2 = st.columns(2)

            with col1:

                st.download_button(
                    "📥 Download TXT",
                    data=chat_text,
                    file_name="pdf_chat.txt",
                    mime="text/plain",
                    use_container_width=True
                )

            with col2:

                st.download_button(
                    "📥 Download Markdown",
                    data=chat_markdown,
                    file_name="pdf_chat.md",
                    mime="text/markdown",
                    use_container_width=True
                )


            st.markdown(
                "#### Share conversation"
            )

            # URL encoding keeps spaces and special characters
            # safe when opening WhatsApp or an email client.
            encoded_chat = quote(
                chat_text
            )

            whatsapp_url = (
                "https://wa.me/?text="
                + encoded_chat
            )

            email_subject = quote(
                "PDF Intelligence Chat"
            )

            email_body = quote(
                chat_text
            )

            email_url = (
                "mailto:?subject="
                + email_subject
                + "&body="
                + email_body
            )

            share1, share2, share3 = st.columns(3)

            with share1:

                st.link_button(
                    "💬 WhatsApp",
                    whatsapp_url,
                    use_container_width=True
                )

            with share2:

                st.link_button(
                    "📧 Email",
                    email_url,
                    use_container_width=True
                )

            with share3:

                if st.button(
                    "🧹 Clear Chat",
                    use_container_width=True
                ):

                    st.session_state.chat_history = []

                    st.rerun()


            # Handy when the user wants to paste the conversation
            # somewhere else without downloading a file.
            with st.expander(
                "📋 View / Copy complete chat"
            ):

                st.text_area(
                    "Conversation",
                    value=chat_text,
                    height=300,
                    label_visibility="collapsed"
                )


    # ========================================================
    # SUMMARY
    # ========================================================

    with summary_tab:

        st.subheader(
            "📝 Executive Summary"
        )

        st.caption(
            "Generate a professional overview "
            "of your uploaded document."
        )

        if st.button(
            "✨ Generate Summary",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "Analysing your document..."
            ):

                summary, pages = (
                    pipeline.summarize()
                )

            st.session_state.summary = summary
            st.session_state.summary_pages = pages


        if st.session_state.summary:

            st.markdown(
                st.session_state.summary
            )

            if st.session_state.summary_pages:

                st.divider()

                st.caption(
                    "Summary based on pages: "
                    + ", ".join(
                        str(page)
                        for page in (
                            st.session_state.summary_pages
                        )
                    )
                )

            st.download_button(
                "📥 Download Summary",
                data=st.session_state.summary,
                file_name="pdf_summary.txt",
                mime="text/plain",
                use_container_width=True
            )


# ============================================================
# Empty state
# ============================================================

else:

    st.markdown(
        "## 📄 Upload a PDF to get started"
    )

    st.caption(
        "Choose a PDF from the sidebar. "
        "Once uploaded, you can ask questions "
        "or generate an executive summary."
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            "### 🔎 Ask Questions"
        )

        st.caption(
            "Ask natural-language questions "
            "about your document."
        )

    with col2:

        st.markdown(
            "### 🧠 Semantic Search"
        )

        st.caption(
            "FAISS retrieves relevant sections "
            "from your PDF."
        )

    with col3:

        st.markdown(
            "### 📝 Summary"
        )

        st.caption(
            "Generate a clear professional "
            "document summary."
        )
