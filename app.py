import streamlit as st
import os
import base64
import tempfile
from pipeline import rag_pipeline_with_context, summarize_document, OLLAMA_BASE_URL, OLLAMA_MODEL

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocuMind AI",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Professional CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"],
p, h1, h2, h3, h4, h5, h6, div, span, input, textarea, label, li, td, th, a, button {
    font-family: 'Inter', sans-serif !important;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Hide Deploy button and toolbar */
[data-testid="stToolbar"] {visibility: hidden;}
[data-testid="stDecoration"] {display: none;}
[data-testid="stHeader"] {display: none !important;}

/* Reduce default padding */
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 0 !important;
    max-width: 100% !important;
}

/* ── Sidebar — always visible, toggle hidden ─────────────── */
section[data-testid="stSidebar"] {
    background: #f8fafc;
    border-right: 1px solid #e2e8f0;
    min-width: 18rem !important;
    max-width: 18rem !important;
    transform: none !important;
    transition: none !important;
}

section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #1e293b !important;
    font-size: 0.95rem !important;
}

button[data-testid="stBaseButton-headerNoPadding"] {
    display: none !important;
}

/* ── Top bar ─────────────────────────────────────────────── */
.top-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    padding: 0.7rem 1.5rem;
    border-radius: 10px;
    margin-bottom: 0.8rem;
    border: 1px solid #334155;
}

.top-bar-left {
    display: flex;
    align-items: center;
    gap: 12px;
}

.top-bar-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #f1f5f9;
    letter-spacing: -0.3px;
}

.top-bar-model {
    display: flex;
    align-items: center;
    gap: 6px;
    color: #94a3b8;
    font-size: 0.76rem;
}

.top-bar-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #22c55e;
    display: inline-block;
    animation: pulse-dot 2s ease-in-out infinite;
}

@keyframes pulse-dot {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* ── Panel headers ───────────────────────────────────────── */
.panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.55rem 0.8rem;
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-radius: 10px 10px 0 0;
    margin-bottom: 0;
}

.panel-header-title {
    font-size: 0.8rem;
    font-weight: 600;
    color: #334155;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

.panel-header-subtitle {
    font-size: 0.72rem;
    color: #94a3b8;
    font-weight: 400;
}

/* ── Highlight legend pill ───────────────────────────────── */
.highlight-legend {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #fefce8;
    border: 1px solid #fde68a;
    border-radius: 6px;
    padding: 0.25rem 0.6rem;
    font-size: 0.72rem;
    color: #92400e;
    font-weight: 500;
}

.highlight-swatch {
    width: 12px;
    height: 12px;
    background: #fde047;
    border-radius: 2px;
    display: inline-block;
    border: 1px solid #ca8a04;
}

/* ── PDF Viewer ──────────────────────────────────────────── */
.pdf-viewer-frame {
    border: 1px solid #e2e8f0;
    border-top: none;
    border-radius: 0 0 10px 10px;
    overflow: hidden;
    background: #525659;
}

.pdf-viewer-frame iframe {
    width: 100%;
    height: 72vh;
    border: none;
    display: block;
}

/* ── Welcome card ────────────────────────────────────────── */
.welcome-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 50vh;
    text-align: center;
    padding: 2rem;
}

.welcome-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: #1e293b;
    margin-bottom: 0.4rem;
}

.welcome-desc {
    font-size: 0.92rem;
    color: #64748b;
    max-width: 480px;
    line-height: 1.6;
    margin-bottom: 1.5rem;
}

.welcome-features {
    display: flex;
    gap: 1.2rem;
    flex-wrap: wrap;
    justify-content: center;
    margin-top: 0.8rem;
}

.welcome-feature {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem 1.3rem;
    width: 160px;
    text-align: center;
    transition: all 0.2s ease;
}

.welcome-feature:hover {
    border-color: #93c5fd;
    box-shadow: 0 4px 12px rgba(59,130,246,0.08);
    transform: translateY(-2px);
}

.welcome-feature-text {
    font-size: 0.8rem;
    font-weight: 600;
    color: #334155;
}

.welcome-feature-sub {
    font-size: 0.7rem;
    color: #94a3b8;
    margin-top: 2px;
}

/* ── Buttons ─────────────────────────────────────────────── */
.stButton > button {
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 0.82rem !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important;
}

/* ── Input fields ────────────────────────────────────────── */
.stTextInput > div > div > input {
    border-radius: 10px !important;
    border: 1.5px solid #e2e8f0 !important;
    font-size: 0.88rem !important;
    padding: 0.6rem 0.9rem !important;
}

.stTextInput > div > div > input:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.1) !important;
}

/* ── File uploader ───────────────────────────────────────── */
.stFileUploader > div {
    border-radius: 10px !important;
}

/* ── Scrollbar ───────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #94a3b8; }
</style>
""", unsafe_allow_html=True)

# ─── Session State ───────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "saved_file_paths" not in st.session_state:
    st.session_state.saved_file_paths = []
if "active_pdf_index" not in st.session_state:
    st.session_state.active_pdf_index = 0
if "highlight_pdf_path" not in st.session_state:
    st.session_state.highlight_pdf_path = None   # path to annotated temp PDF
if "highlight_page" not in st.session_state:
    st.session_state.highlight_page = 1          # page to jump to in iframe
if "highlight_count" not in st.session_state:
    st.session_state.highlight_count = 0         # number of highlights applied

# ─── Helper: Render PDF iframe ───────────────────────────────────────────────
def render_pdf_viewer(file_path: str, page: int = 1):
    """Render a PDF file in an embedded iframe, jumping to the given page."""
    with open(file_path, "rb") as f:
        pdf_bytes = f.read()
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    st.markdown(f"""
    <div class="pdf-viewer-frame">
        <iframe src="data:application/pdf;base64,{b64}#page={page}&toolbar=1&navpanes=0"
                title="PDF Viewer"></iframe>
    </div>
    """, unsafe_allow_html=True)


# ─── Sidebar — Settings & Upload ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### RAG based PDF Chatbot")
    st.caption(f"Powered by Ollama ({OLLAMA_MODEL})")
    st.divider()

    st.markdown("### Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        temp_dir = "temp_docs"
        os.makedirs(temp_dir, exist_ok=True)

        saved_paths = []
        for uf in uploaded_files:
            fp = os.path.join(temp_dir, uf.name)
            with open(fp, "wb") as f:
                f.write(uf.getbuffer())
            saved_paths.append(fp)

        st.session_state.saved_file_paths = saved_paths
        # Reset highlights when new files are uploaded
        st.session_state.highlight_pdf_path = None
        st.session_state.highlight_page = 1
        st.session_state.highlight_count = 0

        st.success(f"{len(saved_paths)} PDF(s) loaded successfully.")

    st.divider()



    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        # Also clear highlights
        st.session_state.highlight_pdf_path = None
        st.session_state.highlight_page = 1
        st.session_state.highlight_count = 0
        st.rerun()

    st.divider()


# ─── Top Bar ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="top-bar">
    <div class="top-bar-left">
        <span class="top-bar-title">DocuMind AI</span>
    </div>
    <div class="top-bar-model">
        <span class="top-bar-dot"></span>
        {OLLAMA_MODEL} &nbsp;•&nbsp; Ollama
    </div>
</div>
""", unsafe_allow_html=True)


# ─── No files — Welcome Screen ───────────────────────────────────────────────
if not st.session_state.saved_file_paths:
    st.markdown("""
    <div class="welcome-container">
        <div class="welcome-title">Upload a PDF to get started</div>
        <div class="welcome-desc">
            Upload one or more PDF documents from the sidebar, view them inline,
            and ask questions. Relevant passages will be displayed directly
            inside the document viewer.
        </div>
        <div class="welcome-features">
            <div class="welcome-feature">
                <div class="welcome-feature-text">Contextual Q&amp;A</div>
                <div class="welcome-feature-sub">Retrieval-augmented answers from your documents</div>
            </div>
            <div class="welcome-feature">
                <div class="welcome-feature-text">Multimedia Extraction</div>
                <div class="welcome-feature-sub">Extracts and analyzes tables and images</div>
            </div>
            <div class="welcome-feature">
                <div class="welcome-feature-text">Multilingual</div>
                <div class="welcome-feature-sub">Supports non-English documents</div>
            </div>
            <div class="welcome-feature">
                <div class="welcome-feature-text">100% Local</div>
                <div class="welcome-feature-sub">All processing on-device via Ollama</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── Files loaded — Split Panel Layout ───────────────────────────────────────
else:
    saved_file_paths = st.session_state.saved_file_paths
    active_idx = min(st.session_state.active_pdf_index, len(saved_file_paths) - 1)
    active_pdf = saved_file_paths[active_idx]
    active_name = os.path.basename(active_pdf)

    pdf_col, chat_col = st.columns([3, 2], gap="medium")

    # ── Left: PDF Viewer ─────────────────────────────────────────────────────
    with pdf_col:
        # Header — build HTML string first to avoid f-string injection issues
        highlight_count = st.session_state.highlight_count
        import html as _html
        safe_name = _html.escape(active_name)

        if highlight_count > 0:
            legend_part = (
                f'<span class="highlight-legend">'
                f'<span class="highlight-swatch"></span>'
                f'{highlight_count} passage{"s" if highlight_count != 1 else ""} highlighted'
                f'</span>'
            )
        else:
            legend_part = ""

        header_html = (
            '<div class="panel-header">'
            '<span class="panel-header-title">Document Viewer</span>'
            '<span style="display:flex;align-items:center;gap:8px;">'
            + legend_part +
            f'<span class="panel-header-subtitle">{safe_name}</span>'
            '</span></div>'
        )
        st.markdown(header_html, unsafe_allow_html=True)

        # Decide which PDF to render: highlighted copy or original
        display_pdf = st.session_state.highlight_pdf_path or active_pdf
        jump_page   = st.session_state.highlight_page

        # Verify the highlighted file still exists (temp files can be cleaned up)
        if display_pdf != active_pdf and not os.path.exists(display_pdf):
            display_pdf = active_pdf
            jump_page   = 1

        render_pdf_viewer(display_pdf, page=jump_page)

        if len(saved_file_paths) > 1:
            st.markdown("<br>", unsafe_allow_html=True)
            doc_names = [os.path.basename(p) for p in saved_file_paths]
            selected_doc = st.pills(
                "Select document to view",
                options=doc_names,
                default=doc_names[active_idx],
                selection_mode="single",
                label_visibility="collapsed"
            )
            
            if selected_doc and selected_doc != doc_names[active_idx]:
                st.session_state.active_pdf_index = doc_names.index(selected_doc)
                st.rerun()

    # ── Right: Chat Panel ────────────────────────────────────────────────────
    with chat_col:
        msg_count = len(st.session_state.chat_history)
        st.markdown(f"""
        <div class="panel-header">
            <span class="panel-header-title">Chat</span>
            <span class="panel-header-subtitle">{msg_count} messages</span>
        </div>
        """, unsafe_allow_html=True)

        chat_area = st.container(height=530)
        with chat_area:
            if not st.session_state.chat_history:
                st.markdown("""
                <div style="display:flex; flex-direction:column; align-items:center;
                            justify-content:center; height:300px;
                            text-align:center; padding:2rem;">
                    <div style="font-size:0.9rem; font-weight:500; color:#64748b;">
                        No messages yet
                    </div>
                    <div style="font-size:0.78rem; margin-top:0.4rem; color:#94a3b8;">
                        Type a question below or click Summarize
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                for msg in st.session_state.chat_history:
                    if msg["role"] == "user":
                        with st.chat_message("user"):
                            st.markdown(msg["content"])
                    else:
                        with st.chat_message("assistant"):
                            st.markdown(msg["content"])
                            # Show page reference if context was found (no raw text dump)
                            if "context" in msg and msg["context"]:
                                doc_to_pages = {}
                                for c in msg["context"]:
                                    doc_to_pages.setdefault(c["document"], set()).add(c["page"])
                                
                                source_parts = []
                                for doc_name, pages in doc_to_pages.items():
                                    page_str = ", ".join(f"p.{p}" for p in sorted(pages))
                                    source_parts.append(f"**{doc_name}** ({page_str})")
                                    
                                st.caption(f"Sources: {' • '.join(source_parts)}")

        # ── Input area ───────────────────────────────────────────────────────
        question = st.text_input(
            "Ask a question",
            placeholder="Ask anything about your document...",
            label_visibility="collapsed",
            key="question_input",
        )

        btn_col1, btn_col2, btn_col3 = st.columns([2, 2, 1])
        with btn_col1:
            ask_clicked = st.button("Ask", use_container_width=True, type="primary")
        with btn_col2:
            summarize_clicked = st.button("Summarize", use_container_width=True)
        with btn_col3:
            clear_clicked = st.button("Clear", use_container_width=True, help="Clear chat history")
            if clear_clicked:
                st.session_state.chat_history = []
                st.session_state.highlight_pdf_path = None
                st.session_state.highlight_page = 1
                st.session_state.highlight_count = 0
                st.rerun()

        # ── Handle Ask ───────────────────────────────────────────────────────
        if ask_clicked and question:
            st.session_state.chat_history.append({
                "role": "user",
                "content": question,
            })

            # Handle Ask with spinner
            with st.spinner("Searching documents and generating answer..."):
                try:
                    answer, ctx = rag_pipeline_with_context(
                        saved_file_paths,
                        question,
                        extract_media=True,
                    )
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "context": ctx if ctx else [],
                    })

                    # ── Navigate to context in the active PDF ───────────────
                    if ctx:
                        # Only use chunks that belong to the currently viewed document
                        active_chunks = [c for c in ctx if c["document"] == active_name]
                        if not active_chunks:
                            active_chunks = ctx  # fall back to all chunks

                        # Navigate to the first relevant page
                        st.session_state.highlight_pdf_path = None
                        pages = sorted({c["page"] for c in active_chunks})
                        st.session_state.highlight_page = pages[0] if pages else 1
                        st.session_state.highlight_count = 0

                except Exception as e:
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"An error occurred: {e}",
                    })

            st.rerun()

        # ── Handle Summarize ─────────────────────────────────────────────────
        if summarize_clicked:
            st.session_state.chat_history.append({
                "role": "user",
                "content": "Summarize the uploaded document(s)",
            })

            with st.spinner("Generating document summary..."):
                try:
                    all_summaries = []
                    for fp in saved_file_paths:
                        doc_name = os.path.basename(fp)
                        summary = summarize_document(
                            fp, extract_media=True,
                        )
                        all_summaries.append(f"**{doc_name}**\n\n{summary}")

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": "\n\n---\n\n".join(all_summaries),
                    })
                except Exception as e:
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"Summarization error: {e}",
                    })

            st.rerun()