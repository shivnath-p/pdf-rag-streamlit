import os
import hashlib
import tempfile
from typing import List, Dict, Tuple

import streamlit as st
import numpy as np
import faiss
import pymupdf

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Gemini
from google import genai


# PAGE CONFIG

st.set_page_config(
    page_title="PDF Intelligence",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)


# CUSTOM CSS

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background: #0e1117;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #151922;
        border-right: 1px solid #262b36;
    }

    /* Main title */
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

    /* Cards */
    .info-card {
        background: #171b24;
        border: 1px solid #292f3b;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 15px;
    }

    .card-title {
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 8px;
    }

    .card-text {
        color: #aeb6c4;
        font-size: 14px;
        line-height: 1.6;
    }

    /* Source badge */
    .source-badge {
        display: inline-block;
        background: #202633;
        border: 1px solid #343b49;
        padding: 6px 10px;
        margin: 3px;
        border-radius: 8px;
        font-size: 13px;
        color: #d5d9e0;
    }

    /* Status */
    .status-success {
        background: #13271e;
        border: 1px solid #245c43;
        color: #73d5a5;
        padding: 10px 14px;
        border-radius: 10px;
        margin: 10px 0;
    }

    /* Upload box */
    [data-testid="stFileUploader"] {
        background: #171b24;
        border: 1px dashed #414957;
        border-radius: 12px;
        padding: 8px;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 9px;
        font-weight: 600;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        border-radius: 12px;
    }

    /* Divider */
    hr {
        border-color: #292f3b;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# GEMINI CONFIG

def get_gemini_client():

    api_key = None

    # Streamlit secrets
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        pass

    # Environment variable fallback
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return None

    return genai.Client(api_key=api_key)


# RAG PIPELINE

class ProductionRAGPipeline:

    def __init__(self):

        # Embedding model

        self.embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        # Text splitter

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=120,
            separators=[
                "\n\n",
                "\n",
                ". ",
                "? ",
                "! ",
                " ",
                ""
            ]
        )

        # Storage

        self.chunks: List[Dict] = []

        self.index = None

        self.pdf_name = None

        # Gemini

        self.client = get_gemini_client()

        self.gemini_model = os.getenv(
            "GEMINI_MODEL",
            "gemini-2.5-flash"
        )


    # PDF EXTRACTION

    def extract_and_chunk_pdf(self, pdf_path: str):

        self.chunks = []

        document = pymupdf.open(pdf_path)

        try:

            for page_number, page in enumerate(document, start=1):

                text = page.get_text("text")

                if not text:
                    continue

                text = text.strip()

                if not text:
                    continue

                page_chunks = self.text_splitter.split_text(text)

                for chunk in page_chunks:

                    chunk = chunk.strip()

                    if len(chunk) < 30:
                        continue

                    self.chunks.append(
                        {
                            "text": chunk,
                            "page": page_number
                        }
                    )

        finally:
            document.close()

        if not self.chunks:
            raise ValueError(
                "No readable text was found in this PDF."
            )

        return self.chunks


    # VECTOR INDEX

    def build_vector_index(self):

        if not self.chunks:
            raise ValueError(
                "No chunks available. Process a PDF first."
            )

        texts = [
            item["text"]
            for item in self.chunks
        ]

        embeddings = self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        embeddings = embeddings.astype("float32")

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)

        self.index.add(embeddings)

        return self.index


    # VECTOR SEARCH

    def retrieve(
        self,
        question: str,
        top_k: int = 5
    ) -> List[Dict]:

        if self.index is None:
            raise ValueError(
                "Vector index has not been created."
            )

        question_embedding = self.embedding_model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        scores, indices = self.index.search(
            question_embedding,
            min(top_k, len(self.chunks))
        )

        results = []

        for score, index_id in zip(
            scores[0],
            indices[0]
        ):

            if index_id < 0:
                continue

            item = self.chunks[index_id].copy()

            item["score"] = float(score)

            results.append(item)

        return results


    # GEMINI ANSWER GENERATION

    def generate_answer(
        self,
        question: str,
        retrieved_chunks: List[Dict]
    ) -> str:

        # If Gemini API is not configured

        if self.client is None:

            context = "\n\n".join(
                [
                    f"[Page {item['page']}]\n{item['text']}"
                    for item in retrieved_chunks
                ]
            )

            return (
                "Gemini API key is not configured yet.\n\n"
                "Here are the most relevant sections I found:\n\n"
                + context
            )

        # ----------------------------------------------------
        # Build context
        # ----------------------------------------------------

        context_parts = []

        for item in retrieved_chunks:

            context_parts.append(
                f"""
[Page {item['page']}]

{item['text']}
"""
            )

        context = "\n".join(context_parts)

        # Prompt

        prompt = f"""
You are an intelligent PDF document assistant.

Answer the user's question using ONLY the information
contained in the provided document context.

Rules:

1. Do not invent information.
2. If the answer is not present in the context,
   clearly say that the information was not found.
3. Give a direct and useful answer.
4. Use simple professional language.
5. When useful, use bullet points.
6. Do not mention that you are an AI unless necessary.
7. Do not make unsupported assumptions.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""

        try:

            response = self.client.models.generate_content(
                model=self.gemini_model,
                contents=prompt
            )

            if response.text:
                return response.text.strip()

            return "I couldn't generate an answer from the document."

        except Exception as e:

            return (
                f"Unable to generate the answer right now.\n\n"
                f"Error: {str(e)}"
            )


    # ========================================================
    # ASK QUESTION
    # ========================================================

    def ask(
        self,
        question: str,
        top_k: int = 5
    ) -> Tuple[str, List[int]]:

        if not question.strip():
            return "Please enter a question.", []

        retrieved = self.retrieve(
            question,
            top_k=top_k
        )

        if not retrieved:
            return (
                "I couldn't find relevant information "
                "in the document."
            ), []

        answer = self.generate_answer(
            question,
            retrieved
        )

        pages = sorted(
            list(
                set(
                    item["page"]
                    for item in retrieved
                )
            )
        )

        return answer, pages


    # SUMMARY

    def summarize(self) -> Tuple[str, List[int]]:

        if not self.chunks:
            return "No document loaded.", []

        # Use representative chunks from the document.
        # This avoids sending the complete PDF to Gemini.
        total = len(self.chunks)

        sample_count = min(15, total)

        if total <= sample_count:

            selected = self.chunks

        else:

            positions = np.linspace(
                0,
                total - 1,
                sample_count,
                dtype=int
            )

            selected = [
                self.chunks[i]
                for i in positions
            ]

        context = "\n\n".join(
            [
                f"[Page {item['page']}]\n{item['text']}"
                for item in selected
            ]
        )

        if self.client is None:

            return (
                "Gemini API key is not configured.\n\n"
                "Relevant document sections:\n\n"
                + context
            ), sorted(
                list(
                    set(
                        item["page"]
                        for item in selected
                    )
                )
            )

        prompt = f"""
Create a professional executive summary of the
following PDF document.

The summary should include:

- Main purpose of the document
- Important topics
- Key terms or conditions
- Important obligations, if present
- Important dates or numbers, if present
- Major conclusions

Use clear headings and bullet points.

Do not invent information.

DOCUMENT:

{context}
"""

        try:

            response = self.client.models.generate_content(
                model=self.gemini_model,
                contents=prompt
            )

            summary = response.text.strip()

        except Exception as e:

            summary = (
                f"Unable to generate summary.\n\n"
                f"Error: {str(e)}"
            )

        pages = sorted(
            list(
                set(
                    item["page"]
                    for item in selected
                )
            )
        )

        return summary, pages


# LOAD PIPELINE

@st.cache_resource
def load_pipeline():

    return ProductionRAGPipeline()


# SESSION STATE

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

if "pdf_hash" not in st.session_state:
    st.session_state.pdf_hash = None

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# HEADER


st.markdown(
    '<div class="main-title">📄 PDF Intelligence Workspace</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Upload a document, ask questions, and get answers grounded
    in your PDF.
    </div>
    """,
    unsafe_allow_html=True
)


# SIDEBAR

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
        value=5,
        help="Number of document sections used to answer a question."
    )

    st.divider()

    st.markdown(
        """
        **How it works**

        1. 📄 Upload PDF
        2. ✂️ Split document
        3. 🧠 Generate embeddings
        4. 🔎 Search relevant sections
        5. 🤖 Generate answer
        """
    )

    st.divider()

    st.caption("PDF Intelligence Workspace")
    st.caption("RAG-powered document analysis")


# PIPELINE INITIALIZATION

if st.session_state.pipeline is None:

    with st.spinner("Loading document intelligence model..."):

        try:

            st.session_state.pipeline = load_pipeline()

        except Exception as e:

            st.error(
                f"Unable to load the AI pipeline: {str(e)}"
            )

            st.stop()


pipeline = st.session_state.pipeline


# PROCESS PDF

if uploaded_file is not None:

    file_bytes = uploaded_file.getvalue()

    current_hash = hashlib.md5(
        file_bytes
    ).hexdigest()

    # --------------------------------------------------------
    # Only process if PDF changed
    # --------------------------------------------------------

    if current_hash != st.session_state.pdf_hash:

        st.session_state.pdf_processed = False
        st.session_state.chat_history = []

        with st.spinner(
            "Reading PDF and building search index..."
        ):

            try:

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as tmp_file:

                    tmp_file.write(file_bytes)

                    temp_pdf_path = tmp_file.name

                # Extract
                pipeline.extract_and_chunk_pdf(
                    temp_pdf_path
                )

                # Build FAISS index
                pipeline.build_vector_index()

                pipeline.pdf_name = uploaded_file.name

                st.session_state.pdf_hash = current_hash
                st.session_state.pdf_processed = True

                # Cleanup
                if os.path.exists(temp_pdf_path):
                    os.remove(temp_pdf_path)

            except Exception as e:

                st.error(
                    f"Could not process this PDF: {str(e)}"
                )

                st.stop()


# ============================================================
# DOCUMENT STATUS
# ============================================================

if st.session_state.pdf_processed:

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            f"""
            <div class="info-card">
                <div class="card-title">📄 Document</div>
                <div class="card-text">
                    {pipeline.pdf_name}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            f"""
            <div class="info-card">
                <div class="card-title">🧩 Chunks</div>
                <div class="card-text">
                    {len(pipeline.chunks)} searchable sections
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        status = (
            "Connected"
            if pipeline.client
            else "API key required"
        )

        st.markdown(
            f"""
            <div class="info-card">
                <div class="card-title">🤖 AI Status</div>
                <div class="card-text">
                    {status}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()


    # ========================================================
    # TABS
    # ========================================================

    chat_tab, summary_tab = st.tabs(
        [
            "💬 Chat with PDF",
            "📝 Document Summary"
        ]
    )


    # ========================================================
    # CHAT TAB
    # ========================================================

    with chat_tab:

        st.subheader("Ask your document")

        st.caption(
            "Ask questions based on the content of your uploaded PDF."
        )

        # Display previous messages
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

                    pages_text = ", ".join(
                        str(page)
                        for page in message["pages"]
                    )

                    st.caption(
                        f"📌 Sources: Pages {pages_text}"
                    )


        question = st.chat_input(
            "Ask something about your PDF..."
        )

        if question:

            # User message
            st.session_state.chat_history.append(
                {
                    "role": "user",
                    "content": question
                }
            )

            with st.chat_message("user"):

                st.markdown(question)

            # AI answer
            with st.chat_message("assistant"):

                with st.spinner("Searching the document..."):

                    answer, pages = pipeline.ask(
                        question,
                        top_k=top_k
                    )

                st.markdown(answer)

                if pages:

                    page_badges = "".join(
                        [
                            f'<span class="source-badge">Page {page}</span>'
                            for page in pages
                        ]
                    )

                    st.markdown(
                        f"""
                        <div style="margin-top:12px;">
                            <b>Sources</b><br>
                            {page_badges}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "pages": pages
                }
            )


    # ========================================================
    # SUMMARY TAB
    # ========================================================

    with summary_tab:

        st.subheader("Executive Summary")

        st.caption(
            "Generate a concise professional overview of the document."
        )

        if st.button(
            "✨ Generate Summary",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "Analysing your document..."
            ):

                summary, pages = pipeline.summarize()

            st.markdown(summary)

            if pages:

                st.divider()

                st.caption(
                    "Summary generated from document pages: "
                    + ", ".join(
                        str(page)
                        for page in pages
                    )
                )


else:

    # ========================================================
    # EMPTY STATE
    # ========================================================

    st.markdown(
        """
        <div style="
            text-align:center;
            padding:70px 20px;
            background:#151922;
            border:1px solid #292f3b;
            border-radius:18px;
            margin-top:30px;
        ">

            <div style="font-size:60px;">📄</div>

            <h2>Upload a PDF to get started</h2>

            <p style="
                color:#9ca3af;
                font-size:16px;
            ">
                Search your document, ask questions,
                and generate an executive summary.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )
