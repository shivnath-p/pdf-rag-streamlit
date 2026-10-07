import streamlit as st
import os
import tempfile
# Import the pipeline we created earlier
# Note: In production, ensure the ProductionRAGPipeline class is defined in this app.py file
# We will paste the self-contained pipeline code here for the Streamlit app.

import pymupdf
import faiss
import numpy as np
import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM

@st.cache_resource
def load_pipeline():
    # Using streamlit resource cache so models are loaded only once
    return ProductionRAGPipeline()

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Intelligent PDF Workspace", page_icon="📄", layout="wide")

st.title("📄 Production-Grade PDF Intelligent Workspace")
st.markdown("### Upload your PDF to instantly search details conversational-style or get an immediate summary!")

try:
    pipeline = load_pipeline()
    st.success("AI Models loaded successfully!")
except Exception as e:
    st.error(f"Error loading AI models: {e}")

uploaded_file = st.sidebar.file_uploader("📂 Step 1: Upload PDF File", type=["pdf"])

if uploaded_file is not None:
    # Save uploaded file to a temporary file path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_pdf_path = tmp_file.name

    # Process PDF and build index
    with st.spinner("Processing PDF and generating vector index..."):
        try:
            pipeline.extract_and_chunk_pdf(temp_pdf_path)
            pipeline.build_vector_index()
            st.sidebar.success("PDF processed and indexed successfully!")
        except Exception as e:
            st.sidebar.error(f"Error processing PDF: {e}")

    # Create two interactive tabs in Streamlit
    tab1, tab2 = st.tabs(["💬 Chat with PDF", "📝 Summarize Your Uploaded PDF"])

    with tab1:
        st.header("Chat with PDF")
        question = st.text_input("Ask any question about the PDF", placeholder="e.g., What are the terms of termination?")
        if st.button("Ask AI", type="primary"):
            if question:
                with st.spinner("Thinking..."):
                    answer, pages_used = pipeline.ask(question)
                    st.markdown("#### AI Response (Human-Like)")
                    st.write(answer)
                    st.info(f"Sources Reference Page(s): {', '.join(map(str, pages_used)) if pages_used else 'N/A'}")
            else:
                st.warning("Please enter a question first!")

    with tab2:
        st.header("Document Executive Summary")
        st.markdown("Click the button below to get an instant professional summary of the document.")
        if st.button("Generate Summary ✨"):
            with st.spinner("Summarizing entire document..."):
                summary_query = "Give a comprehensive, clear, and professional executive summary of this entire document, highlighting its main purpose and key terms in bullet points."
                answer, _ = pipeline.ask(summary_query)
                st.markdown("#### Summary Overview")
                st.write(answer)

    # Clean up temporary file
    if os.path.exists(temp_pdf_path):
        os.remove(temp_pdf_path)
else:
    st.info("Please upload a PDF file in the sidebar to get started!")
