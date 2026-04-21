"""
Enhanced RAG Pipeline: PDF → Text/Tables/Images → Chunks → Embeddings → FAISS → Retriever → LLM → Answer
Supports multimedia extraction (tables, images) and multilingual documents.
"""
import os
import base64
import tempfile
from pypdf import PdfReader
import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "llava")

# Minimum word count to consider a document as having meaningful content
MIN_CONTENT_WORDS = 3


def call_ollama(prompt, model=None):
    """Call Ollama API with a prompt and return model response text."""
    model = model or OLLAMA_MODEL
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }
    )

    if response.status_code == 200:
        return response.json()["response"]
    return f"Error: {response.status_code} - {response.text}"


def call_ollama_vision(prompt, image_path, model=None):
    """Call Ollama API with a prompt and an image for multimodal analysis."""
    model = model or OLLAMA_VISION_MODEL
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "images": [image_b64],
                "stream": False
            }
        )
        
        if response.status_code == 200:
            return response.json()["response"]
        elif response.status_code == 404:
            return f"(Vision model '{model}' is not installed. Run 'ollama run {model}' to enable image analysis.)"
            
        return f"(Error describing image: {response.text})"
    except Exception as e:
        return f"(Connection error to Ollama: {e})"


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def _extract_tables_from_page(pdf_path, page_number):
    """Extract tables from a specific page using pdfplumber and return as markdown."""
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            if page_number - 1 >= len(pdf.pages):
                return ""
            page = pdf.pages[page_number - 1]
            tables = page.extract_tables()
            if not tables:
                return ""

            md_parts = []
            for table in tables:
                if not table or not table[0]:
                    continue
                # Build a markdown table
                headers = table[0]
                md = "| " + " | ".join(str(h) if h else "" for h in headers) + " |\n"
                md += "| " + " | ".join("---" for _ in headers) + " |\n"
                for row in table[1:]:
                    md += "| " + " | ".join(str(c) if c else "" for c in row) + " |\n"
                md_parts.append(md)
            return "\n".join(md_parts)
    except ImportError:
        return ""
    except Exception as e:
        print(f"Table extraction error on page {page_number}: {e}")
        return ""


def _extract_images_from_page(pdf_path, page_number, has_text=False):
    """
    Extract the page as an image using PyMuPDF and get a text description.
    By rendering the whole page to a pixmap, it ensures we capture images even
    if they are complex, layered, or scanned graphics.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return "[Error: PyMuPDF is not installed. Run 'pip install pymupdf' to enable image extraction]"

    try:
        doc = fitz.open(pdf_path)
        if page_number - 1 >= len(doc):
            doc.close()
            return ""
            
        page = doc[page_number - 1]
        
        images = page.get_images(full=True)
        descriptions = []

        if has_text:
            # The page already has high-quality embedded text. 
            # Only extract isolated explicit embedded images so the vision model doesn't hallucinate OCR on the full page.
            if not images:
                doc.close()
                return ""
                
            for img_index, img_info in enumerate(images):
                xref = img_info[0]
                try:
                    base_image = doc.extract_image(xref)
                    if not base_image:
                        continue
                    image_bytes = base_image["image"]
                    
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                        tmp.write(image_bytes)
                        tmp_path = tmp.name
                        
                    desc = call_ollama_vision(
                        "Describe this image briefly. Focus on data, charts, or visual diagrams. Do not guess or hallucinate any text.",
                        tmp_path
                    )
                    descriptions.append(f"[Image {img_index + 1} on page {page_number}]: {desc}")
                    
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
                except Exception:
                    continue
                    
            doc.close()
            return "\n".join(descriptions)

        else:
            # The page lacks selectable text (likely a full-page scan or flattened image).
            # Render the entire page to a pixmap so the Vision model can execute OCR and visual analysis.
            pix = page.get_pixmap(dpi=150)
            image_bytes = pix.tobytes("png")
            
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp.write(image_bytes)
                tmp_path = tmp.name

            doc.close()

            desc = call_ollama_vision(
                "Extract all readable text from this image exactly as written without hallucinating or paraphrasing. Next, briefly describe any important non-textual visual elements, diagrams, or charts.",
                tmp_path
            )
            
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
                
            return f"[Visual Content of page {page_number}]: {desc}"

    except Exception as e:
        return f"[Error: PyMuPDF image extraction failed on page {page_number}: {e}]"


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def extract_text_from_pdfs(pdf_paths, extract_media=False):
    """Step 1: PDF → Text Extraction with Page Numbers (+ optional tables/images)."""
    docs_with_pages = []
    for pdf_path in pdf_paths:
        try:
            reader = PdfReader(pdf_path)
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                text = text.strip()

                page_num = i + 1
                doc_name = os.path.basename(pdf_path)

                # ---------- Multimedia enrichment ----------
                extra_parts = []
                if extract_media:
                    table_md = _extract_tables_from_page(pdf_path, page_num)
                    if table_md:
                        extra_parts.append(f"[Tables on page {page_num}]:\n{table_md}")

                    has_text = len(text.split()) > 20
                    image_desc = _extract_images_from_page(pdf_path, page_num, has_text=has_text)
                    if image_desc:
                        extra_parts.append(image_desc)

                combined = text
                if extra_parts:
                    combined = text + "\n\n" + "\n\n".join(extra_parts)

                # Skip completely empty pages
                if not combined.strip():
                    continue

                docs_with_pages.append({
                    "text": combined,
                    "page_number": page_num,
                    "document": doc_name
                })
        except Exception as e:
            print(f"Error reading {pdf_path}: {e}")

    print(f"Extracted text from {len(docs_with_pages)} pages across {len(pdf_paths)} documents")
    return docs_with_pages


def _total_word_count(docs_with_pages):
    """Return total word count across all extracted pages."""
    return sum(len(doc["text"].split()) for doc in docs_with_pages)


def chunk_text(docs_with_pages, chunk_size=500):
    """Step 2: Text → Chunking (with deduplication)."""
    chunks_with_metadata = []
    seen_texts = set()

    for doc in docs_with_pages:
        words = doc["text"].split()
        if not words:
            continue

        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            # Deduplicate identical chunks
            if chunk in seen_texts:
                continue
            seen_texts.add(chunk)

            chunks_with_metadata.append({
                "text": chunk,
                "page_number": doc["page_number"],
                "document": doc["document"]
            })

    print(f"Created {len(chunks_with_metadata)} unique chunks")
    return chunks_with_metadata


def create_embeddings(chunks_with_metadata):
    """Step 3: Chunks → Embeddings (multilingual model for non-English support)."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    chunks_text = [chunk['text'] for chunk in chunks_with_metadata]
    embeddings = model.encode(chunks_text)
    print(f"Created embeddings: {embeddings.shape}")
    return embeddings, model


def create_vector_database(embeddings):
    """Step 4: Embeddings → Vector Database (FAISS)."""
    import faiss
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    print(f"FAISS index created with {index.ntotal} vectors")
    return index


def retrieve_relevant_chunks(query, chunks_with_metadata, embeddings_model, faiss_index, top_k=5):
    """Step 5: Query → Retriever (with document diversity enforcement)."""
    # Fetch a larger pool to allow for diversity picking
    pool_size = min(top_k * 4, len(chunks_with_metadata))
    if pool_size == 0:
        return []

    query_embedding = embeddings_model.encode([query])
    distances, indices = faiss_index.search(query_embedding.astype('float32'), pool_size)

    # Parse and deduplicate
    candidate_chunks = []
    seen = set()
    for idx in indices[0]:
        if idx < 0:
            continue
        if idx not in seen:
            seen.add(idx)
            candidate_chunks.append(chunks_with_metadata[idx])

    # Enforce document diversity: Round-robin chunk selection from each document present in the candidates
    # This prevents one document from hoarding all the context slots if distances are tied/close.
    docs_to_chunks = {}
    for chunk in candidate_chunks:
        docs_to_chunks.setdefault(chunk["document"], []).append(chunk)

    diverse_chunks = []
    while len(diverse_chunks) < top_k and docs_to_chunks:
        # Loop through each document and take its best remaining chunk
        for doc in list(docs_to_chunks.keys()):
            diverse_chunks.append(docs_to_chunks[doc].pop(0))
            if not docs_to_chunks[doc]:
                del docs_to_chunks[doc]
            if len(diverse_chunks) >= top_k:
                break

    print(f"Retrieved {len(diverse_chunks)} diverse chunks across {len(set(c['document'] for c in diverse_chunks))} documents")
    return diverse_chunks


def generate_answer(query, relevant_chunks):
    """Step 6: Context + Query → LLM (Ollama) → Answer (always in English)."""
    context = "\n\n".join([
        f"From {chunk['document']} (Page {chunk['page_number']}):\n{chunk['text']}"
        for chunk in relevant_chunks
    ])

    prompt = f"""Context:
{context}

Question: {query}

Instructions:
- Answer ONLY based on the context provided above.
- If the context does not contain enough information to answer the question, say so clearly.
- Always respond in English, even if the context is in another language.
- Keep your response under 200 words."""

    return call_ollama(prompt)


def rag_pipeline(pdf_paths, query, extract_media=False):
    """Complete RAG Pipeline for multiple documents."""
    print("Starting RAG Pipeline...")

    docs_with_pages = extract_text_from_pdfs(pdf_paths, extract_media=extract_media)

    # Validate content
    if not docs_with_pages:
        return "The uploaded document(s) appear to be blank or contain no extractable text."

    word_count = _total_word_count(docs_with_pages)
    if word_count < MIN_CONTENT_WORDS:
        return (
            f"The document(s) contain very little content ({word_count} word(s)). "
            "There is not enough information to answer questions from."
        )

    chunks_with_metadata = chunk_text(docs_with_pages)

    if not chunks_with_metadata:
        return "Could not create meaningful text chunks from the document(s)."

    embeddings, model = create_embeddings(chunks_with_metadata)
    faiss_index = create_vector_database(embeddings)
    relevant_chunks = retrieve_relevant_chunks(query, chunks_with_metadata, model, faiss_index)
    answer = generate_answer(query, relevant_chunks)

    return answer


def rag_pipeline_with_context(pdf_paths, query, top_k=6, extract_media=False):
    """Run RAG pipeline and return both answer and retrieved context chunks."""
    docs_with_pages = extract_text_from_pdfs(pdf_paths, extract_media=extract_media)

    # ---- Blank / insufficient content guard ----
    if not docs_with_pages:
        return (
            "The uploaded document(s) appear to be blank or contain no extractable text.",
            []
        )

    word_count = _total_word_count(docs_with_pages)
    if word_count < MIN_CONTENT_WORDS:
        return (
            f"The document(s) contain very little content ({word_count} word(s)). "
            "There is not enough information to answer questions from.",
            []
        )

    chunks_with_metadata = chunk_text(docs_with_pages)

    if not chunks_with_metadata:
        return "Could not create meaningful text chunks from the document(s).", []

    embeddings, model = create_embeddings(chunks_with_metadata)
    faiss_index = create_vector_database(embeddings)
    relevant_chunks = retrieve_relevant_chunks(
        query, chunks_with_metadata, model, faiss_index, top_k=top_k
    )
    answer = generate_answer(query, relevant_chunks)

    context_for_display = [
        {
            "text": chunk["text"],
            "page": chunk["page_number"],
            "document": chunk["document"]
        } for chunk in relevant_chunks
    ]

    return answer, context_for_display


def summarize_document(pdf_path, max_words=3000, extract_media=False):
    """Generate a concise summary of a single document (always in English)."""
    print(f"Summarizing {os.path.basename(pdf_path)}...")
    docs_with_pages = extract_text_from_pdfs([pdf_path], extract_media=extract_media)

    if not docs_with_pages:
        return "Could not extract any content from the document. The document may be blank."

    full_text = " ".join([doc['text'] for doc in docs_with_pages])
    words = full_text.split()

    if len(words) < MIN_CONTENT_WORDS:
        return (
            f"The document contains very little content ({len(words)} word(s)): "
            f'"{full_text.strip()}". There is not enough information to generate a meaningful summary.'
        )

    trimmed_text = " ".join(words[:max_words])

    prompt = f"""Summarize the following document in simple language.
Keep it concise and cover the key points.
IMPORTANT: Always provide the summary in English, even if the document is written in another language.
If the document contains tables or image descriptions, include their key information in the summary.

Document:
{trimmed_text}

Summary:"""

    return call_ollama(prompt)


if __name__ == "__main__":
    pdf_path = input("Enter PDF path: ")
    question = input("Enter your question: ")

    answer = rag_pipeline([pdf_path], question)
    print(f"\n📄 Answer: {answer}")
