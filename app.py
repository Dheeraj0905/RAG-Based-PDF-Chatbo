"""
Simple RAG PDF Chatbot
"""
import streamlit as st
import os
from pipeline import rag_pipeline_with_context, summarize_document, OLLAMA_BASE_URL, OLLAMA_MODEL

st.set_page_config(page_title="RAG PDF Chatbot")

st.title("RAG Based PDF Chatbot")
st.caption(f"Powered by Ollama ({OLLAMA_MODEL})")

# File upload
uploaded_files = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    # Create a temporary directory for uploaded files
    temp_dir = "temp_docs"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # Save uploaded files
    saved_file_paths = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_file_paths.append(file_path)
    
    st.success(f"{len(saved_file_paths)} PDF(s) uploaded successfully!")

    # ----- Settings sidebar -----
    with st.sidebar:
        st.header("Settings")
        extract_media = st.toggle(
            "Extract images & tables",
            value=False,
            help="Enable multimedia extraction to include tables and image descriptions. "
                 "Requires pdfplumber, PyMuPDF, and a vision model (llava) in Ollama."
        )

    # Question input
    question = st.text_input("Ask a question about the documents:")

    col1, col2 = st.columns(2)
    get_answer_clicked = col1.button("Get Answer")
    summarize_clicked = col2.button("Summarize Document(s)")

    if get_answer_clicked and question:
        with st.spinner("Processing..."):
            try:
                answer, used_context = rag_pipeline_with_context(
                    saved_file_paths, question, extract_media=extract_media
                )
                st.write("**Answer:**")
                st.write(answer)

                if used_context:
                    with st.expander("Show context used for this answer"):
                        for chunk in used_context:
                            st.info(f"From **{chunk['document']}** (Page {chunk['page']}):")
                            st.write(chunk['text'])
            except Exception as e:
                st.error(f"An error occurred: {e}")

    if summarize_clicked:
        with st.spinner("Summarizing..."):
            try:
                for file_path in saved_file_paths:
                    st.write(f"---")
                    st.write(f"**Summary for {os.path.basename(file_path)}:**")
                    summary = summarize_document(file_path, extract_media=extract_media)
                    st.write(summary)
            except Exception as e:
                st.error(f"An error occurred during summarization: {e}")
else:
    st.info("Upload one or more PDF files to get started.")
