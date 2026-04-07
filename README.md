# 📚 RAG-Based PDF Chatbot

> Intelligent Document Question-Answering System using Retrieval Augmented Generation

Ask questions about your PDF documents and get accurate, context-aware answers powered by AI.

---

## 🎯 What is RAG?

**RAG (Retrieval Augmented Generation)** combines document retrieval with AI language models to provide accurate answers based on your specific documents.

**Pipeline Flow:**
```
PDF → Text Extraction → Chunking → Embeddings → Vector DB (FAISS) → Retriever → LLM → Answer
```

---

## ✨ Features

- 📄 **PDF Processing** - Extract and analyze text from PDF documents
- 🔍 **Semantic Search** - Find relevant information using AI embeddings
- 💬 **Natural Language Q&A** - Ask questions conversationally
- ⚡ **Fast Responses** - Vector similarity search in milliseconds
- 🆓 **100% Free** - Powered by Ollama (runs locally)
- 🔒 **Privacy First** - All processing happens on your machine

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| **Programming** | Python 3.13 |
| **LLM** | Ollama (Llama 3.2) |
| **Embeddings** | Sentence Transformers |
| **Vector DB** | FAISS |
| **PDF Processing** | PyPDF |
| **Web Interface** | Streamlit |

---

## 📦 Installation

### Step 1: Install Ollama

Download and install from: **https://ollama.ai/download**

Then pull the model:
```bash
ollama pull llama3.2
```

Verify installation:
```bash
ollama list
```

### Step 2: Clone Repository

```bash
git clone https://github.com/Dheeraj0905/RAG-Based-PDF-Chatbot.git
cd "RAG-Based-PDF-Chatbot"
```

### Step 3: Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Option 1: Web Interface (Recommended)

1. **Start Streamlit:**
   ```bash
   streamlit run app.py
   ```

2. **Open Browser:**
   - Navigate to `http://localhost:8501`

3. **Upload & Ask:**
   - Upload your PDF
   - Type your question
   - Get instant answers!

### Option 2: Command Line

```bash
python pipeline.py
```

Then follow the prompts to enter PDF path and ask questions.

---

## 📂 Project Structure

```
RAG-Based-PDF-Chatbot/
│
├── pipeline.py           # Core RAG implementation
├── app.py               # Streamlit web interface
├── requirements.txt     # Python dependencies
├── .env                # Configuration (Ollama settings)
├── .env.example        # Configuration template
├── README.md           # This file
├── .gitignore          # Git ignore rules
│
└── data/               # Store your PDF files here
    └── AWS.pdf
```

---

## 🔧 Configuration

Edit `.env` file to customize settings:

```env
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
```

**Available Models:**
- `llama3.2` - Default, balanced performance
- `llama3.1` - Latest version
- `mistral` - Faster, lighter
- `codellama` - Better for technical docs

Change model:
```bash
ollama pull mistral
```
Then update `.env` with `OLLAMA_MODEL=mistral`

---

## 📖 How It Works

### 1. **Document Processing**
```python
# Extract text from PDF
text = extract_text_from_pdf("document.pdf")
```

### 2. **Text Chunking**
```python
# Split into manageable pieces
chunks = chunk_text(text, chunk_size=500)
```

### 3. **Create Embeddings**
```python
# Convert text to vectors
embeddings = create_embeddings(chunks)
```

### 4. **Vector Database**
```python
# Store in FAISS for fast retrieval
vector_db = create_vector_database(embeddings)
```

### 5. **Semantic Retrieval**
```python
# Find most relevant chunks
relevant = retrieve_relevant_chunks(query, vector_db)
```

### 6. **Generate Answer**
```python
# Use LLM to create response
answer = generate_answer(query, relevant)
```

---

## 💡 Example Questions

**General:**
- "What is the main topic of this document?"
- "Summarize the key points"

**Specific:**
- "What are the benefits mentioned in section 3?"
- "Explain the methodology used"

**Comparative:**
- "What's the difference between X and Y?"
- "Compare the approaches discussed"

---

## 🐛 Troubleshooting

### Ollama Not Running
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama (Windows)
ollama serve
```

### Model Not Found
```bash
# List installed models
ollama list

# Pull required model
ollama pull llama3.2
```

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

### Streamlit Issues
```bash
# Clear cache
streamlit cache clear

# Restart app
streamlit run app.py
```

---

## 🎓 Use Cases

| Industry | Application |
|----------|-------------|
| 📚 **Education** | Research papers, textbook Q&A |
| 🏢 **Business** | Report analysis, documentation |
| ⚖️ **Legal** | Contract review, case research |
| 🏥 **Healthcare** | Medical literature search |
| 🔬 **Research** | Academic paper analysis |

---

## 🚀 Future Enhancements

- [ ] Multi-file chat (upload multiple PDFs)
- [ ] Chat history persistence
- [ ] Export Q&A to PDF/Word
- [ ] Support for Word, Excel, PowerPoint
- [ ] Advanced RAG techniques (HyDE, ReAct)
- [ ] Web scraping integration
- [ ] Multi-language support
- [ ] Mobile app

---

## 📝 Requirements

**System:**
- Python 3.13+
- 4GB RAM minimum
- Ollama installed

**Python Packages:**
```
pypdf
faiss-cpu
sentence-transformers
streamlit
python-dotenv
requests
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 👨‍💻 Author

**Dheeraj**  
- GitHub: [@Dheeraj0905](https://github.com/Dheeraj0905)
- Project: [RAG-Based-PDF-Chatbot](https://github.com/Dheeraj0905/RAG-Based-PDF-Chatbot)

---

## 🙏 Acknowledgments

- [Ollama](https://ollama.ai/) - Local LLM runtime
- [FAISS](https://github.com/facebookresearch/faiss) - Vector similarity search
- [Sentence Transformers](https://www.sbert.net/) - Text embeddings
- [Streamlit](https://streamlit.io/) - Web framework

---

## ⭐ Star History

If you find this project helpful, please give it a ⭐ on GitHub!

---

**Made with ❤️ using Python and AI**

## Files

- `config.py` - Settings
- `document_processor.py` - Load PDFs
- `vector_store.py` - Store embeddings (free HuggingFace)
- `chatbot.py` - Answer questions (Ollama)
- `app.py` - Web UI
- `main.py` - CLI

That's it!
