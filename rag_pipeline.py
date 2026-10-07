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
    Handles document processing and question answering.

    PDF
      ↓
    Text extraction
      ↓
    Chunking
      ↓
    Embeddings
      ↓
    FAISS retrieval
      ↓
    Gemini
      ↓
    Answer
    """

    def __init__(self):

        # A lightweight embedding model keeps the application
        # practical for a cloud deployment.
        self.embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        # Overlap helps preserve context between neighbouring chunks.
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

        self.gemini_model = os.getenv(
            "GEMINI_MODEL",
            "gemini-2.5-flash"
        )

    # ---------------------------------------------------------
    # Gemini client
    # ---------------------------------------------------------

    def _create_gemini_client(self):

        api_key = None

        # Streamlit Cloud secrets
        try:
            import streamlit as st
            api_key = st.secrets.get("GEMINI_API_KEY")
        except Exception:
            pass

        # Local environment fallback
        if not api_key:
            api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            return None

        return genai.Client(
            api_key=api_key
        )

    # ---------------------------------------------------------
    # Extract PDF text and create chunks
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
                "The PDF may be scanned or image-based."
            )

        return self.chunks

    # ---------------------------------------------------------
    # Build FAISS index
    # ---------------------------------------------------------

    def build_vector_index(self):

        if not self.chunks:

            raise ValueError(
                "No document chunks found."
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

        self.index = faiss.IndexFlatIP(
            dimension
        )

        self.index.add(
            embeddings
        )

        return self.index

    # ---------------------------------------------------------
    # Search relevant chunks
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

            item = self.chunks[
                index_id
            ].copy()

            item["score"] = float(
                score
            )

            results.append(item)

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
                "Add GEMINI_API_KEY in Streamlit Secrets."
            )

        context = "\n\n".join(
            [
                f"[Page {item['page']}]\n{item['text']}"
                for item in retrieved_chunks
            ]
        )

        prompt = f"""
You are a professional document assistant.

Answer the user's question using ONLY the document
context provided below.

Rules:

- Do not invent facts.
- Do not use outside information.
- If the answer is not available, say so clearly.
- Give a direct and useful answer.
- Use simple professional language.
- Use bullet points when useful.
- Preserve dates, numbers, names and conditions accurately.

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
    # Ask a question
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

        retrieved = self.retrieve(
            question,
            top_k
        )

        if not retrieved:

            return (
                "I could not find relevant information "
                "in the document.",
                []
            )

        answer = self.generate_answer(
            question,
            retrieved
        )

        pages = sorted(
            set(
                item["page"]
                for item in retrieved
            )
        )

        return answer, pages

    # ---------------------------------------------------------
    # Generate document summary
    # ---------------------------------------------------------

    def summarize(self) -> Tuple[str, List[int]]:

        if not self.chunks:

            return (
                "No document loaded.",
                []
            )

        total_chunks = len(
            self.chunks
        )

        # We sample sections from the entire document so that
        # the summary isn't based only on the first few pages.
        sample_count = min(
            15,
            total_chunks
        )

        if total_chunks <= sample_count:

            selected = self.chunks

        else:

            positions = np.linspace(
                0,
                total_chunks - 1,
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
                "⚠️ Gemini API key is not configured.",
                []
            )

        prompt = f"""
Create a professional executive summary of this document.

Use these sections:

## Main Purpose
Explain the main purpose of the document.

## Key Topics
List the major topics.

## Important Terms / Conditions
Mention important terms and conditions.

## Important Details
Mention important dates, numbers, obligations,
responsibilities or limitations when present.

## Conclusion
Give the main takeaway.

Rules:

- Use ONLY the provided document context.
- Do not invent information.
- Keep the summary professional and easy to read.
- Use bullet points where appropriate.

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

                summary = "Unable to generate the summary."

        except Exception as e:

            summary = (
                "⚠️ Summary generation failed.\n\n"
                f"Error: {str(e)}"
            )

        pages = sorted(
            set(
                item["page"]
                for item in selected
            )
        )

        return summary, pages
