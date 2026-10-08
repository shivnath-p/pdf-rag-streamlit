# 📄 PDF Chat — Multilingual RAG-Based PDF Question Answering

> An intelligent document assistant that lets you upload any PDF, ask questions in your preferred language, generate summaries with a custom word count, and share or download results.

🔗 **Live Demo:  https://pdf-rag-app-9uulezdrpqcix893umcnen.streamlit.app/
---

## 🚀 Overview

**PDF Chat** is a multilingual **Retrieval-Augmented Generation (RAG)** application that allows users to interact with PDF documents using natural language.

The application supports both **text-based and scanned PDFs**, automatically extracts relevant information, retrieves the most relevant document sections, and generates grounded answers using **Google Gemini**.

Users can also generate summaries, select their preferred output language, download conversations, and share results.

---

## ✨ Key Features

### 🌐 Multilingual Support
- Ask questions in multiple languages.
- Supports documents containing Hindi, Odia, English, and other languages.
- Choose from **19 output languages**.
- Change the output language at any time.

### 📄 PDF Processing
- Supports standard text-based PDFs.
- Supports scanned/image-based PDFs.
- Automatic OCR fallback for pages with insufficient readable text.
- OCR powered by **Google Gemini**.

### 💬 Chat with Your PDF
- Ask natural-language questions about your document.
- Answers are generated using retrieved document content.
- Page numbers are provided with answers.
- Includes a **View Sources** panel for transparency.

### 📝 Custom Word-Count Summarization
- Generate a summary of the uploaded PDF.
- Specify the desired number of words.
- Useful for reports, research papers, notes, documentation, and other long-form documents.

### ⬇️ Download & Share
- Download chat conversations as `.txt`.
- Download generated summaries.
- Share results through:
  - WhatsApp
  - Email
  - Copy to clipboard

### 🎨 Modern User Interface
- Clean and responsive Streamlit interface.
- Dark and light themes.
- Suggestion chips for quick questions.
- Progress indicators for long-running operations.

---

## 🧠 RAG Architecture

The application follows a Retrieval-Augmented Generation pipeline:

```text
                    ┌─────────────────────┐
                    │      PDF Upload     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   PDF Text Extract  │
                    │      PyMuPDF        │
                    └──────────┬──────────┘
                               │
                     No readable text?
                               │
                               ▼
                    ┌─────────────────────┐
                    │    OCR Fallback     │
                    │   Google Gemini     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      Chunking       │
                    │  800 chars / 120    │
                    │      overlap        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Embeddings       │
                    │ multilingual-e5-    │
                    │       small         │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   FAISS Vector DB   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Relevant Chunks    │
                    │      Top-K          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Google Gemini     │
                    │  Answer Generation  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Answer + Sources    │
                    │    + Page Numbers   │
                    └─────────────────────┘
