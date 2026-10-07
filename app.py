import os
import sqlite3
import hashlib
import secrets
import tempfile
from datetime import datetime, timezone
from urllib.parse import quote

import streamlit as st

from rag_pipeline import ProductionRAGPipeline


# ============================================================
# App configuration
# ============================================================

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# Database
# ============================================================

DB_FILE = "documind.db"


def get_db():

    connection = sqlite3.connect(
        DB_FILE,
        check_same_thread=False
    )

    connection.row_factory = sqlite3.Row

    return connection


def init_database():

    connection = get_db()

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS shared_chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            share_id TEXT UNIQUE NOT NULL,
            document_name TEXT,
            chat_text TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    connection.commit()
    connection.close()


init_database()


# ============================================================
# Logo
# ============================================================

def logo_html(size=42):

    return f"""
    <div style="
        width:{size}px;
        height:{size}px;
        min-width:{size}px;
        border-radius:12px;
        display:flex;
        align-items:center;
        justify-content:center;
        background:linear-gradient(135deg,#6366f1,#8b5cf6);
        box-shadow:0 8px 25px rgba(99,102,241,.25);
    ">
        <svg
            width="{int(size * 0.55)}"
            height="{int(size * 0.55)}"
            viewBox="0 0 24 24"
            fill="none"
            xmlns="http://www.w3.org/2000/svg">

            <path
                d="M6 2.8C6 2.35817 6.35817 2 6.8 2H14L19 7V21.2C19 21.6418 18.6418 22 18.2 22H6.8C6.35817 22 6 21.6418 6 21.2V2.8Z"
                fill="white"
                opacity="0.96"/>

            <path
                d="M14 2V7H19"
                stroke="#6366F1"
                stroke-width="1.5"
                stroke-linejoin="round"/>

            <path
                d="M9 12H16M9 15.5H16M9 8.5H11"
                stroke="#6366F1"
                stroke-width="1.5"
                stroke-linecap="round"/>
        </svg>
    </div>
    """


# ============================================================
# Global styling
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- Global ---------- */

    .stApp {
        background:
            radial-gradient(
                circle at 10% 0%,
                rgba(99,102,241,.10),
                transparent 28%
            ),
            #0b0e14;
    }

    /* ---------- Sidebar ---------- */

    section[data-testid="stSidebar"] {
        background: #10141d;
        border-right: 1px solid #252b38;
    }

    /* ---------- Main title ---------- */

    .brand-title {
        font-size: 39px;
        font-weight: 750;
        letter-spacing: -1.5px;
        margin: 0;
    }

    .brand-subtitle {
        color: #9299a8;
        font-size: 16px;
        margin-top: 6px;
    }

    /* ---------- Cards ---------- */

    .metric-card {
        background:
            linear-gradient(
                145deg,
                rgba(30,35,48,.95),
                rgba(19,23,32,.95)
            );
        border: 1px solid #293140;
        border-radius: 16px;
        padding: 18px;
        min-height: 105px;
    }

    .metric-label {
        color: #8992a3;
        font-size: 13px;
        margin-bottom: 7px;
    }

    .metric-value {
        font-size: 17px;
        font-weight: 650;
        color: #f1f3f7;
    }

    /* ---------- Chat ---------- */

    [data-testid="stChatMessage"] {
        border-radius: 16px;
    }

    /* ---------- Buttons ---------- */

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 10px;
        font-weight: 600;
    }

    /* ---------- Upload ---------- */

    [data-testid="stFileUploader"] {
        background: #151a24;
        border: 1px dashed #394252;
        border-radius: 14px;
    }

    /* ---------- Source ---------- */

    .source-pill {
        display:inline-block;
        padding:5px 9px;
        margin:2px;
        border-radius:8px;
        background:#1b2230;
        border:1px solid #30394a;
        color:#aeb8ca;
        font-size:12px;
    }

    /* ---------- Empty state ---------- */

    .empty-card {
        text-align:center;
        padding:65px 25px;
        margin-top:30px;
        background:rgba(20,24,34,.75);
        border:1px solid #282f3d;
        border-radius:22px;
    }

    .empty-icon {
        font-size:54px;
    }

    .empty-title {
        font-size:26px;
        font-weight:700;
        margin-top:12px;
    }

    .empty-description {
        color:#9299a8;
        font-size:15px;
        max-width:600px;
        margin:10px auto 0;
    }

    /* ---------- Share card ---------- */

    .share-card {
        background:#151a24;
        border:1px solid #293140;
        border-radius:16px;
        padding:18px;
        margin-top:12px;
    }

    /* ---------- Hide unnecessary decoration ---------- */

    footer {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True
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
    "share_id": None
}

for key, value in defaults.items():

    if key not in st.session_state:

        st.session_state[key] = value


# ============================================================
# Helper functions
# ============================================================

def load_pipeline():

    return ProductionRAGPipeline()


def create_chat_text():

    history = st.session_state.chat_history

    if not history:

        return "No conversation available."

    lines = [
        "DocuMind AI",
        "=" * 45,
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
            lines.append(
                message["content"]
            )

        else:

            lines.append("ASSISTANT")
            lines.append("-" * 20)
            lines.append(
                message["content"]
            )

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

    lines = [
        "# DocuMind AI",
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
            lines.append(
                message["content"]
            )

        else:

            lines.append("### 🤖 Assistant")
            lines.append("")
            lines.append(
                message["content"]
            )

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


def create_share_id():

    # A random URL-safe ID prevents users from guessing
    # another conversation's share URL.
    return secrets.token_urlsafe(12)


def save_shared_chat():

    if not st.session_state.chat_history:

        return None

    share_id = create_share_id()

    chat_text = create_chat_text()

    connection = get_db()

    connection.execute(
        """
        INSERT INTO shared_chats
        (
            share_id,
            document_name,
            chat_text,
            created_at
        )
        VALUES (?, ?, ?, ?)
        """,
        (
            share_id,
            st.session_state.pdf_name,
            chat_text,
            datetime.now(
                timezone.utc
            ).isoformat()
        )
    )

    connection.commit()
    connection.close()

    st.session_state.share_id = share_id

    return share_id


def get_shared_chat(share_id):

    connection = get_db()

    result = connection.execute(
        """
        SELECT *
        FROM shared_chats
        WHERE share_id = ?
        """,
        (share_id,)
    ).fetchone()

    connection.close()

    return result


def get_share_url(share_id):

    # Streamlit uses the query parameter to identify the
    # shared conversation.
    base_url = (
        st.context.url
        if hasattr(st, "context")
        else ""
    )

    if base_url:

        separator = "&" if "?" in base_url else "?"

        return (
            base_url
            + separator
            + "share="
            + share_id
        )

    return (
        "?share="
        + share_id
    )


# ============================================================
# Check whether this is a shared conversation
# ============================================================

try:

    share_parameter = st.query_params.get(
        "share"
    )

except Exception:

    share_parameter = None


if share_parameter:

    shared_chat = get_shared_chat(
        share_parameter
    )

    if shared_chat:

        # Shared view gets a deliberately simple layout.
        st.markdown(
            f"""
            <div style="
                display:flex;
                align-items:center;
                gap:12px;
                margin-bottom:25px;
            ">
                {logo_html(46)}

                <div>
                    <div style="
                        font-size:25px;
                        font-weight:700;
                    ">
                        DocuMind AI
                    </div>

                    <div style="
                        color:#8f98a8;
                        font-size:13px;
                    ">
                        Shared PDF conversation
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            "## 🔗 Shared Conversation"
        )

        st.caption(
            f"Document: {shared_chat['document_name']}"
        )

        st.text_area(
            "Conversation",
            value=shared_chat["chat_text"],
            height=600,
            disabled=True,
            label_visibility="collapsed"
        )

        st.download_button(
            "📥 Download Conversation",
            data=shared_chat["chat_text"],
            file_name="shared_pdf_chat.txt",
            mime="text/plain",
            use_container_width=True
        )

        st.divider()

        st.caption(
            "This conversation was shared from DocuMind AI."
        )

        st.stop()


# ============================================================
# Load pipeline
# ============================================================

if st.session_state.pipeline is None:

    with st.spinner(
        "Loading document intelligence..."
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
# Sidebar branding
# ============================================================

with st.sidebar:

    st.markdown(
        f"""
        <div style="
            display:flex;
            align-items:center;
            gap:11px;
            margin-bottom:25px;
        ">

            {logo_html(42)}

            <div>
                <div style="
                    font-size:20px;
                    font-weight:700;
                ">
                    DocuMind
                </div>

                <div style="
                    color:#858e9f;
                    font-size:12px;
                ">
                    PDF Intelligence
                </div>
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "### 📂 Document"
    )

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        help="Upload the PDF you want to analyse."
    )

    st.divider()

    st.markdown(
        "### ⚙️ Retrieval"
    )

    top_k = st.slider(
        "Relevant sections",
        min_value=3,
        max_value=10,
        value=5
    )

    st.divider()

    st.markdown(
        """
        **Pipeline**

        📄 PDF

        ↓

        ✂️ Chunking

        ↓

        🧠 Embeddings

        ↓

        🔎 FAISS

        ↓

        🤖 Gemini
        """
    )

    st.divider()

    st.caption(
        "DocuMind AI • RAG Document Assistant"
    )


# ============================================================
# Main branding
# ============================================================

brand_col1, brand_col2 = st.columns(
    [0.08, 0.92]
)

with brand_col1:

    st.markdown(
        logo_html(52),
        unsafe_allow_html=True
    )

with brand_col2:

    st.markdown(
        '<div class="brand-title">'
        'DocuMind AI'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="brand-subtitle">'
        'Ask questions. Understand documents. '
        'Get answers grounded in your PDF.'
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# Process PDF
# ============================================================

if uploaded_file is not None:

    file_bytes = uploaded_file.getvalue()

    current_hash = hashlib.md5(
        file_bytes
    ).hexdigest()

    # Only rebuild the index if the user uploads a different file.
    if current_hash != st.session_state.pdf_hash:

        st.session_state.pdf_processed = False
        st.session_state.chat_history = []
        st.session_state.summary = ""
        st.session_state.summary_pages = []
        st.session_state.share_id = None

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

                    temp_pdf_path = tmp_file.name

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
                    and os.path.exists(
                        temp_pdf_path
                    )
                ):

                    os.remove(
                        temp_pdf_path
                    )


# ============================================================
# Document loaded
# ============================================================

if st.session_state.pdf_processed:

    # --------------------------------------------------------
    # Document metrics
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    DOCUMENT
                </div>

                <div class="metric-value">
                    📄 {st.session_state.pdf_name}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    SEARCHABLE CONTENT
                </div>

                <div class="metric-value">
                    🧩 {len(pipeline.chunks)} sections
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        status = (
            "🟢 Gemini Connected"
            if pipeline.client
            else "🔴 API Key Missing"
        )

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    AI STATUS
                </div>

                <div class="metric-value">
                    {status}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()


    # --------------------------------------------------------
    # Main tabs
    # --------------------------------------------------------

    chat_tab, summary_tab = st.tabs(
        [
            "💬 Chat",
            "📝 Summary"
        ]
    )


    # ========================================================
    # CHAT TAB
    # ========================================================

    with chat_tab:

        st.subheader(
            "Chat with your PDF"
        )

        st.caption(
            "Ask anything about the information contained "
            "in your uploaded document."
        )

        # Render existing conversation.
        for message in (
            st.session_state.chat_history
        ):

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

                    answer, pages = (
                        pipeline.ask(
                            question,
                            top_k=top_k
                        )
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
        # Export and sharing
        # ----------------------------------------------------

        if st.session_state.chat_history:

            st.divider()

            st.subheader(
                "📤 Share & Export"
            )

            chat_text = create_chat_text()

            chat_markdown = create_chat_markdown()


            download1, download2 = st.columns(2)

            with download1:

                st.download_button(
                    "📥 Download TXT",
                    data=chat_text,
                    file_name="documind_chat.txt",
                    mime="text/plain",
                    use_container_width=True
                )

            with download2:

                st.download_button(
                    "📥 Download Markdown",
                    data=chat_markdown,
                    file_name="documind_chat.md",
                    mime="text/markdown",
                    use_container_width=True
                )


            # ------------------------------------------------
            # Create public share link
            # ------------------------------------------------

            if st.button(
                "🔗 Create Shareable Link",
                type="primary",
                use_container_width=True
            ):

                with st.spinner(
                    "Creating secure share link..."
                ):

                    share_id = save_shared_chat()

                if share_id:

                    st.success(
                        "Share link created successfully."
                    )

                    share_url = get_share_url(
                        share_id
                    )

                    st.code(
                        share_url,
                        language="text"
                    )

                    st.info(
                        "Copy the link above and send it "
                        "to anyone you want to share this "
                        "conversation with."
                    )


            # Show existing link again after reruns.
            if st.session_state.share_id:

                share_url = get_share_url(
                    st.session_state.share_id
                )

                st.markdown(
                    """
                    <div class="share-card">
                        <b>🔗 Your shared conversation</b>
                        <br>
                        <span style="color:#9299a8;">
                        Anyone with this link can view
                        the saved conversation.
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.code(
                    share_url,
                    language="text"
                )

                encoded_url = quote(
                    share_url
                )

                whatsapp_share = (
                    "https://wa.me/?text="
                    + quote(
                        "Here is my PDF conversation:\n\n"
                        + share_url
                    )
                )

                email_share = (
                    "mailto:?subject="
                    + quote(
                        "Shared PDF Conversation"
                    )
                    + "&body="
                    + quote(
                        "Here is the PDF conversation:\n\n"
                        + share_url
                    )
                )

                share1, share2, share3 = st.columns(3)

                with share1:

                    st.link_button(
                        "💬 WhatsApp",
                        whatsapp_share,
                        use_container_width=True
                    )

                with share2:

                    st.link_button(
                        "📧 Email",
                        email_share,
                        use_container_width=True
                    )

                with share3:

                    if st.button(
                        "📋 Copy Link",
                        use_container_width=True
                    ):

                        st.info(
                            "Copy the link from the box above."
                        )


            st.divider()

            # Direct sharing without creating a permanent link.
            st.markdown(
                "#### Quick Share"
            )

            whatsapp_text = quote(
                chat_text
            )

            whatsapp_url = (
                "https://wa.me/?text="
                + whatsapp_text
            )

            email_url = (
                "mailto:?subject="
                + quote("DocuMind AI Chat")
                + "&body="
                + quote(chat_text)
            )

            quick1, quick2, quick3 = st.columns(3)

            with quick1:

                st.link_button(
                    "💬 WhatsApp Chat",
                    whatsapp_url,
                    use_container_width=True
                )

            with quick2:

                st.link_button(
                    "📧 Email Chat",
                    email_url,
                    use_container_width=True
                )

            with quick3:

                if st.button(
                    "🧹 Clear Chat",
                    use_container_width=True
                ):

                    st.session_state.chat_history = []
                    st.session_state.share_id = None

                    st.rerun()


            with st.expander(
                "📋 View complete conversation"
            ):

                st.text_area(
                    "Conversation",
                    value=chat_text,
                    height=300,
                    label_visibility="collapsed"
                )


    # ========================================================
    # SUMMARY TAB
    # ========================================================

    with summary_tab:

        st.subheader(
            "📝 Executive Summary"
        )

        st.caption(
            "Generate a professional overview of the "
            "uploaded document."
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
                file_name="documind_summary.txt",
                mime="text/plain",
                use_container_width=True
            )


# ============================================================
# Empty state
# ============================================================

else:

    st.markdown(
        f"""
        <div class="empty-card">

            <div style="
                display:flex;
                justify-content:center;
            ">
                {logo_html(64)}
            </div>

            <div class="empty-title">
                Your documents, understood.
            </div>

            <div class="empty-description">
                Upload a PDF and use AI-powered semantic
                search to ask questions, understand
                important details and generate summaries.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    feature1, feature2, feature3 = st.columns(3)

    with feature1:

        st.markdown(
            "### 🔎 Smart Search"
        )

        st.caption(
            "Find relevant information using "
            "semantic search instead of simple keywords."
        )

    with feature2:

        st.markdown(
            "### 💬 Document Chat"
        )

        st.caption(
            "Ask natural questions and receive "
            "answers grounded in your PDF."
        )

    with feature3:

        st.markdown(
            "### 🔗 Easy Sharing"
        )

        st.caption(
            "Save a conversation and share it "
            "using a simple link."
        )
