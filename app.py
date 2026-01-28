"""
Simple Streamlit App for RAG Pipeline
"""
import streamlit as st
import os
from pipeline import rag_pipeline, OLLAMA_BASE_URL, OLLAMA_MODEL

st.set_page_config(page_title="Simple RAG Chatbot", page_icon="📚")
st.title("📚 Simple RAG PDF Chatbot")
st.write("Upload PDF → Ask Questions → Get Answers")
st.caption(f"🦙 Powered by Ollama ({OLLAMA_MODEL})")

# File upload
uploaded_file = st.file_uploader("Upload PDF", type=['pdf'])

if uploaded_file:
    # Save uploaded file
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success("PDF uploaded successfully!")
    
    # Question input
    question = st.text_input("Ask a question about the PDF:")
    
    if st.button("Get Answer") and question:
        with st.spinner("Processing... (may take a moment)"):
            try:
                answer = rag_pipeline("temp.pdf", question)
                st.write("### Answer:")
                st.write(answer)
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info(f"Make sure Ollama is running at {OLLAMA_BASE_URL}")
    
    # Clean up
    if os.path.exists("temp.pdf"):
        os.remove("temp.pdf")

else:
    st.info("👆 Upload a PDF to get started")
    st.markdown(f"""
    **Setup:**
    1. Make sure Ollama is running: `{OLLAMA_BASE_URL}`
    2. Model loaded: `{OLLAMA_MODEL}`
    3. Upload your PDF and ask questions!
    """)
