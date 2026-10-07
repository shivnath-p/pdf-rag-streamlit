import os
from typing import List, Dict, Tuple

import faiss
import numpy as np
import pymupdf

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from google import genai


class ProductionRAGPipeline:
    """
    Handles the complete RAG pipeline.

    Flow:
        PDF
         ↓
        Text extraction
         ↓
        Chunking
         ↓
        Embeddings
         ↓
        FAISS
         ↓
        Relevant chunks
         ↓
        Gemini
         ↓
        Final answer
    """

    def __init__(self):

        # Small embedding model keeps the application reasonably
        # fast and memory-friendly on Streamlit Cloud.
        self.embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        # These settings give us reasonably sized chunks while
        # keeping some context between neighbouring chunks.
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

        self.chunks: List[Dict] = []
        self.index = None
        self.pdf_name = None

        self.client = self._create_gemini_client()

        # You can change this through Streamlit Secrets/environment
        # without changing the Python code.
        self.gemini_model = os.getenv(
            "GEMINI_MODEL",
            "gemini-2.5-flash"
        )

    # ---------------------------------------------------------
    # Gemini client
    # ---------------------------------------------------------

    def _create_gemini_client(self):

        api_key = None

        # Streamlit Cloud
        try:
            import streamlit as st
            api_key = st.secrets.get("GEMINI_API_KEY")
        except Exception:
            pass

        # Local development
        if not api_key:
            api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            return None

        return genai.Client(
            api_key=api_key
        )

    # ---------------------------------------------------------
    # Extract text from PDF and create chunks
    # ---------------------------------------------------------

    def extract_and_chunk_pdf(
        self,
        pdf_path: str
    ) -> List[Dict]:

        self.chunks = []

        document = pymupdf.open(pdf_path)

        try:

            for page_number, page in enumerate(
                document,
                start=1
            ):

                text = page.get_text("text")

                if not text:
                    continue

                text = text.strip()

                if not text:
                    continue

                page_chunks = self.text_splitter.split_text(
                    text
                )

                for chunk in page_chunks:

                    chunk = chunk.strip()

                    # Ignore tiny fragments that aren't useful
                    # for semantic search.
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
                "No readable text was found in this PDF. "
                "The PDF may be scanned/image-based."
            )

        return self.chunks

    # ---------------------------------------------------------
    # Create FAISS vector index
    # ---------------------------------------------------------

    def build_vector_index(self):

        if not self.chunks:
            raise ValueError(
                "No chunks found. Process a PDF first."
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

        embeddings = embeddings.astype(
            "float32"
        )

        dimension = embeddings.shape[1]

        # Inner-product search works well because the embeddings
        # are normalized above.
        self.index = faiss.IndexFlatIP(
            dimension
        )

        self.index.add(
            embeddings
        )

        return self.index

    # ---------------------------------------------------------
    # Retrieve relevant document sections
    # ---------------------------------------------------------

    def retrieve(
        self,
        question: str,
        top_k: int = 5
    ) -> List[Dict]:

        if self.index is None:
            raise ValueError(
                "Vector index is not ready."
            )

        question_embedding = self.embedding_model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        number_to_retrieve = min(
            top_k,
            len(self.chunks)
        )

        scores, indices = self.index.search(
            question_embedding,
            number_to_retrieve
        )

        results = []

        for score, index_id in zip(
            scores[0],
            indices[0]
        ):

            if index_id < 0:
                continue

            result = self.chunks[
                index_id
            ].copy()

            result["score"] = float(
                score
            )

            results.append(
                result
            )

        return results

    # ---------------------------------------------------------
    # Generate answer with Gemini
    # ---------------------------------------------------------

    def generate_answer(
        self,
        question: str,
        retrieved_chunks: List[Dict]
    ) -> str:

        if self.client is None:

            return (
                "⚠️ Gemini API key is not configured.\n\n"
                "Please add GEMINI_API_KEY to "
                "Streamlit Secrets."
            )

        context_parts = []

        for item in retrieved_chunks:

            context_parts.append(
                f"""
[Page {item['page']}]

{item['text']}
"""
            )

        context = "\n".join(
            context_parts
        )

        prompt = f"""
You are a professional PDF document assistant.

Answer the user's question using ONLY the information
contained in the document context below.

Rules:

1. Do not invent information.
2. Do not use outside information.
3. If the answer is not available in the document,
   clearly say that it was not found.
4. Give a direct and useful answer.
5. Use simple professional language.
6. Use bullet points when appropriate.
7. Preserve dates, numbers and names accurately.
8. Do not unnecessarily repeat the question.

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

            return (
                "I could not generate an answer "
                "from the document."
            )

        except Exception as e:

            return (
                "⚠️ Gemini request failed.\n\n"
                f"Error: {str(e)}"
            )

    # ---------------------------------------------------------
    # Complete question-answer flow
    # ---------------------------------------------------------

    def ask(
        self,
        question: str,
        top_k: int = 5
    ) -> Tuple[str, List[int]]:

        if not question.strip():

            return (
                "Please enter a question.",
                []
            )

        retrieved_chunks = self.retrieve(
            question,
            top_k
        )

        if not retrieved_chunks:

            return (
                "I could not find relevant information "
                "in the document.",
                []
            )

        answer = self.generate_answer(
            question,
            retrieved_chunks
        )

        pages = sorted(
            set(
                item["page"]
                for item in retrieved_chunks
            )
        )

        return answer, pages

    # ---------------------------------------------------------
    # Generate executive summary
    # ---------------------------------------------------------

    def summarize(
        self
    ) -> Tuple[str, List[int]]:

        if not self.chunks:

            return (
                "No document loaded.",
                []
            )

        total_chunks = len(
            self.chunks
        )

        # Instead of sending a huge PDF to Gemini, take
        # representative sections from throughout the document.
        sample_count = min(
            15,
            total_chunks
        )

        if total_chunks <= sample_count:

            selected_chunks = self.chunks

        else:

            positions = np.linspace(
                0,
                total_chunks - 1,
                sample_count,
                dtype=int
            )

            selected_chunks = [
                self.chunks[i]
                for i in positions
            ]

        context = "\n\n".join(
            [
                f"[Page {item['page']}]\n{item['text']}"
                for item in selected_chunks
            ]
        )

        if self.client is None:

            return (
                "⚠️ Gemini API key is not configured.",
                []
            )

        prompt = f"""
Create a professional executive summary of this PDF.

Use the following structure:

## Main Purpose
Explain what the document is mainly about.

## Key Topics
List the most important topics.

## Important Terms / Conditions
Mention important terms, conditions or requirements.

## Important Details
Include important dates, numbers, obligations,
responsibilities or limitations if present.

## Conclusion
Give the main takeaway from the document.

Rules:

- Use ONLY the provided document context.
- Do not invent information.
- Keep the summary clear and professional.
- Use bullet points where useful.

DOCUMENT:

{context}
"""

        try:

            response = self.client.models.generate_content(
                model=self.gemini_model,
                contents=prompt
            )

            if response.text:

                summary = response.text.strip()

            else:

                summary = (
                    "Unable to generate the summary."
                )

        except Exception as e:

            summary = (
                "⚠️ Summary generation failed.\n\n"
                f"Error: {str(e)}"
            )

        pages = sorted(
            set(
                item["page"]
                for item in selected_chunks
            )
        )

        return summary, pages
