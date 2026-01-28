"""
Simple RAG Pipeline: PDF → Text → Chunks → Embeddings → FAISS → Retriever → LLM → Answer
"""
import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")


def extract_text_from_pdf(pdf_path):
    """Step 1: PDF → Text Extraction"""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    print(f"Extracted text from {len(reader.pages)} pages")
    return text


def chunk_text(text, chunk_size=500):
    """Step 2: Text → Chunking"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    
    print(f"Created {len(chunks)} chunks")
    return chunks


def create_embeddings(chunks):
    """Step 3: Chunks → Embeddings"""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks)
    print(f"Created embeddings: {embeddings.shape}")
    return embeddings, model


def create_vector_database(embeddings):
    """Step 4: Embeddings → Vector Database (FAISS)"""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    print(f"FAISS index created with {index.ntotal} vectors")
    return index


def retrieve_relevant_chunks(query, chunks, embeddings_model, faiss_index, top_k=3):
    """Step 5: Query → Retriever"""
    query_embedding = embeddings_model.encode([query])
    distances, indices = faiss_index.search(query_embedding.astype('float32'), top_k)
    
    relevant_chunks = [chunks[i] for i in indices[0]]
    print(f"Retrieved {len(relevant_chunks)} relevant chunks")
    return relevant_chunks


def generate_answer(query, relevant_chunks):
    """Step 6: Context + Query → LLM (Ollama) → Answer"""
    context = "\n\n".join(relevant_chunks)
    
    prompt = f"""Context:
{context}

Question: {query}

Answer based on the context above:"""
    
    # Call Ollama API
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        }
    )
    
    if response.status_code == 200:
        return response.json()["response"]
    else:
        return f"Error: {response.status_code} - {response.text}"


def rag_pipeline(pdf_path, query):
    """Complete RAG Pipeline"""
    print("🚀 Starting RAG Pipeline...")
    
    # Step 1: PDF → Text
    text = extract_text_from_pdf(pdf_path)
    
    # Step 2: Text → Chunks
    chunks = chunk_text(text)
    
    # Step 3: Chunks → Embeddings
    embeddings, model = create_embeddings(chunks)
    
    # Step 4: Embeddings → Vector Database
    faiss_index = create_vector_database(embeddings)
    
    # Step 5: Query → Relevant Chunks
    relevant_chunks = retrieve_relevant_chunks(query, chunks, model, faiss_index)
    
    # Step 6: LLM → Answer
    answer = generate_answer(query, relevant_chunks)
    
    return answer


if __name__ == "__main__":
    # Example usage
    pdf_path = input("Enter PDF path: ")
    question = input("Enter your question: ")
    
    answer = rag_pipeline(pdf_path, question)
    print(f"\n📄 Answer: {answer}")
