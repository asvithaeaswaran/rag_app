"""
========================================================================================
📂 My Documents — Simple RAG Q&A System (No LangChain, Pure Python)
========================================================================================
Built with:
1. Text Extraction: PyPDF & python-docx
2. Text Chunking: Pure Python Recursive Splitter
3. Embedding Model: sentence-transformers/all-MiniLM-L6-v2
4. Vector Database: FAISS (Facebook AI Similarity Search)
5. LLMs: Google Gemini, Ollama Local, Groq, OpenAI (via direct API requests)
========================================================================================
"""

import os
import json
import shutil
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string
from dotenv import load_dotenv

# Optional direct imports
import pypdf
import docx
import faiss
from sentence_transformers import SentenceTransformer
import requests

# Load .env variables
load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'my_documents_files')
app.config['INDEX_FOLDER'] = os.path.join(os.path.dirname(__file__), 'my_documents_faiss')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['INDEX_FOLDER'], exist_ok=True)

# -----------------------------------------------------------------------------
# 1. INITIALIZE SIMPLE EMBEDDING MODEL (all-MiniLM-L6-v2)
# -----------------------------------------------------------------------------
print("[My Documents] Loading embedding model: sentence-transformers/all-MiniLM-L6-v2...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
EMBEDDING_DIM = 384  # Dimension of all-MiniLM-L6-v2

# In-memory document & chunk registry
# Schema: [{'doc_id': ..., 'filename': ..., 'chunk_index': ..., 'page': ..., 'text': ...}]
chunks_registry = []
uploaded_documents = []


# -----------------------------------------------------------------------------
# 2. DOCUMENT PARSERS & TEXT EXTRACTORS (Pure Python)
# -----------------------------------------------------------------------------

def extract_text(file_path: str, filename: str) -> list[dict]:
    """
    Extracts text page by page from PDF, DOCX, or TXT.
    Returns: list of {'text': page_text, 'page': page_number}
    """
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else 'txt'
    pages = []

    # A) PDF Files
    if ext == 'pdf':
        try:
            reader = pypdf.PdfReader(file_path)
            for i, page in enumerate(reader.pages):
                txt = (page.extract_text() or "").strip()
                if txt:
                    pages.append({'text': txt, 'page': i + 1})
        except Exception as e:
            print(f"Error reading PDF {filename}: {e}")

    # B) Word Documents (.docx)
    elif ext in ['docx', 'doc']:
        try:
            doc = docx.Document(file_path)
            paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            for table in doc.tables:
                for row in table.rows:
                    row_txt = " | ".join(c.text.strip() for c in row.cells if c.text.strip())
                    if row_txt:
                        paras.append(row_txt)
            full_text = "\n\n".join(paras)
            if full_text.strip():
                pages.append({'text': full_text, 'page': 1})
        except Exception as e:
            print(f"Error reading DOCX {filename}: {e}")

    # C) Plain Text / Markdown / CSV
    else:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read().strip()
                if content:
                    pages.append({'text': content, 'page': 1})
        except Exception as e:
            print(f"Error reading Text file {filename}: {e}")

    return pages


# -----------------------------------------------------------------------------
# 3. SIMPLE TEXT CHUNKER (Pure Python without LangChain)
# -----------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 80) -> list[str]:
    """
    Splits text into overlapping chunks cleanly on sentence/paragraph boundaries.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        
        # Try to find a natural break near the end (period, newline, or space)
        if end < len(text):
            break_pos = text.rfind('\n', start, end)
            if break_pos == -1 or break_pos < start + (chunk_size // 2):
                break_pos = text.rfind('. ', start, end)
            if break_pos == -1 or break_pos < start + (chunk_size // 2):
                break_pos = text.rfind(' ', start, end)
            if break_pos != -1 and break_pos > start:
                end = break_pos + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap
        if start >= len(text) - overlap:
            break

    return chunks


# -----------------------------------------------------------------------------
# 4. FAISS VECTOR DATABASE STORAGE (Direct FAISS Index)
# -----------------------------------------------------------------------------

faiss_index_path = os.path.join(app.config['INDEX_FOLDER'], 'my_documents.index')
meta_path = os.path.join(app.config['INDEX_FOLDER'], 'metadata.json')


def init_or_load_faiss_index():
    """Load existing FAISS index from disk or create a fresh L2 Index."""
    global chunks_registry, uploaded_documents
    if os.path.exists(faiss_index_path) and os.path.exists(meta_path):
        try:
            index = faiss.read_index(faiss_index_path)
            with open(meta_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                chunks_registry = data.get('chunks', [])
                uploaded_documents = data.get('documents', [])
            return index
        except Exception as e:
            print(f"Failed to load FAISS index: {e}")

    # Create empty Flat L2 Index
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    return index


# Load index at startup
faiss_index = init_or_load_faiss_index()


def save_faiss_index():
    """Save FAISS index and chunk metadata to disk."""
    faiss.write_index(faiss_index, faiss_index_path)
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump({
            'chunks': chunks_registry,
            'documents': uploaded_documents
        }, f, indent=2)


def add_document_to_faiss(file_path: str, filename: str) -> int:
    """
    Parses document, chunks it, computes embeddings, and stores in FAISS.
    Returns: number of chunks added
    """
    global faiss_index, chunks_registry, uploaded_documents

    pages = extract_text(file_path, filename)
    if not pages:
        return 0

    new_chunks = []
    for p in pages:
        text_splits = chunk_text(p['text'], chunk_size=500, overlap=80)
        for idx, chunk_str in enumerate(text_splits):
            new_chunks.append({
                'filename': filename,
                'page': p['page'],
                'chunk_index': idx,
                'text': chunk_str
            })

    if not new_chunks:
        return 0

    # 1. Compute Embeddings with all-MiniLM-L6-v2
    texts_to_embed = [c['text'] for c in new_chunks]
    embeddings = embedding_model.encode(texts_to_embed, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    # 2. Add to FAISS Vector Index
    faiss_index.add(embeddings)
    chunks_registry.extend(new_chunks)

    # Record in document list
    doc_info = {
        'filename': filename,
        'chunks_count': len(new_chunks),
        'file_size': os.path.getsize(file_path)
    }
    # Update if already exists, else append
    uploaded_documents = [d for d in uploaded_documents if d['filename'] != filename]
    uploaded_documents.append(doc_info)

    save_faiss_index()
    return len(new_chunks)


def search_faiss(query: str, top_k: int = 3) -> list[dict]:
    """
    Embeds user question and retrieves the top-k most relevant chunks from FAISS.
    """
    if faiss_index.ntotal == 0 or len(chunks_registry) == 0:
        return []

    # 1. Embed query
    query_vec = embedding_model.encode([query], normalize_embeddings=True)
    query_vec = np.array(query_vec, dtype=np.float32)

    # 2. Search FAISS
    k = min(top_k, faiss_index.ntotal)
    distances, indices = faiss_index.search(query_vec, k)

    results = []
    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), 1):
        if idx < len(chunks_registry):
            chunk = chunks_registry[idx]
            results.append({
                'rank': rank,
                'filename': chunk['filename'],
                'page': chunk['page'],
                'text': chunk['text'],
                'score': round(float(dist), 4),
                'snippet': chunk['text'][:250] + ('...' if len(chunk['text']) > 250 else '')
            })

    return results


# -----------------------------------------------------------------------------
# 5. DIRECT LLM CALLS (Google Gemini, Ollama, Groq, OpenAI)
# -----------------------------------------------------------------------------

def sanitize_key(val):
    """Clean API keys removing all whitespace, newlines, and carriage returns."""
    if not val:
        return None
    cleaned = str(val).strip().replace('\r', '').replace('\n', '').replace('\t', '').strip()
    return cleaned if cleaned else None


def call_llm(user_question: str, context_text: str, provider: str = "auto", api_key: str = None) -> str:
    """
    Calls the LLM directly with the RAG prompt without external framework wrappers.
    """
    system_instruction = (
        "You are 'My Documents AI', a helpful and precise assistant. "
        "Answer the user's question accurately using ONLY the provided document excerpts below. "
        "If the information is not contained in the excerpts, clearly say that the uploaded documents do not have this information. "
        "Format your answer cleanly with bullet points and bold key terms."
    )

    full_prompt = (
        f"DOCUMENT EXCERPTS:\n{context_text}\n\n"
        f"USER QUESTION: {user_question}\n\n"
        "Please provide a complete and accurate answer based on the document excerpts above."
    )

    # Sanitize and resolve keys
    cleaned_user_key = sanitize_key(api_key)
    gemini_key = sanitize_key(cleaned_user_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    groq_openai_key = sanitize_key(cleaned_user_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("GROQ_API_KEY"))
    ollama_url = sanitize_key(os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")) or "http://localhost:11434"

    # A) Google Gemini Direct API
    if (provider == "gemini" or (not provider and gemini_key and gemini_key.startswith("AIza"))) and gemini_key:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={gemini_key}"
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": f"System: {system_instruction}\n\n{full_prompt}"}
                        ]
                    }
                ],
                "generationConfig": {"temperature": 0.3, "maxOutputTokens": 2048}
            }
            res = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=30)
            if res.status_code == 200:
                data = res.json()
                if 'candidates' in data and data['candidates']:
                    return data['candidates'][0]['content']['parts'][0]['text']
            else:
                print(f"Gemini API error: {res.status_code} - {res.text}")
        except Exception as e:
            print(f"Gemini request exception: {e}")

    # B) Ollama Local Model Direct API
    if provider == "ollama" or (not provider and not gemini_key and not groq_openai_key):
        try:
            url = f"{ollama_url.rstrip('/')}/api/generate"
            payload = {
                "model": os.environ.get("OLLAMA_MODEL", "llama3"),
                "prompt": f"{system_instruction}\n\n{full_prompt}",
                "stream": False,
                "options": {"temperature": 0.3}
            }
            res = requests.post(url, json=payload, timeout=30)
            if res.status_code == 200:
                return res.json().get('response', '')
        except Exception as e:
            print(f"Ollama local request error: {e}")

    # C) Groq / OpenAI Direct API
    if groq_openai_key:
        try:
            base_url = "https://api.groq.com/openai/v1/chat/completions" if groq_openai_key.startswith("gsk_") else "https://api.openai.com/v1/chat/completions"
            model = os.environ.get("OPENAI_MODEL", "openai/gpt-oss-120b" if groq_openai_key.startswith("gsk_") else "gpt-4o-mini")
            
            clean_auth_header = f"Bearer {groq_openai_key}".strip()
            headers = {
                "Authorization": clean_auth_header,
                "Content-Type": "application/json"
            }
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": full_prompt}
                ],
                "temperature": 0.3
            }
            res = requests.post(base_url, json=payload, headers=headers, timeout=30)
            if res.status_code == 200:
                return res.json()['choices'][0]['message']['content']
            else:
                print(f"Groq/OpenAI API error: {res.status_code} - {res.text}")
        except Exception as e:
            print(f"Groq/OpenAI error: {e}")

    # D) Intelligent Pure-Python Fallback (Zero external dependencies)
    return None



# -----------------------------------------------------------------------------
# 6. FLASK WEB ROUTES & BEAUTIFUL UI
# -----------------------------------------------------------------------------

UI_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>My Documents — Simple RAG System</title>
    <!-- Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- Marked.js Markdown Renderer -->
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <!-- FontAwesome -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">

    <style>
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #334155; border-radius: 9999px; }
        .markdown-body p { margin-bottom: 0.75rem; }
        .markdown-body ul { list-style-type: disc; padding-left: 1.25rem; margin-bottom: 0.75rem; }
        .markdown-body pre { background: #0f172a; padding: 0.75rem; border-radius: 0.5rem; overflow-x: auto; margin: 0.5rem 0; }
        .markdown-body code { background: rgba(255,255,255,0.1); padding: 0.1rem 0.3rem; border-radius: 0.2rem; font-size: 0.9em; }
    </style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen flex flex-col font-sans">
    
    <!-- Top Header -->
    <header class="bg-slate-900 border-b border-slate-800 px-6 py-4 flex items-center justify-between sticky top-0 z-20 shadow-md">
        <div class="flex items-center gap-3">
            <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-blue-600 via-indigo-600 to-violet-600 flex items-center justify-center text-white text-lg shadow-lg">
                <i class="fa-solid fa-folder-tree"></i>
            </div>
            <div>
                <h1 class="text-lg font-bold text-white tracking-wide flex items-center gap-2">
                    <span>My Documents</span>
                    <span class="text-[10px] px-2 py-0.5 rounded-full bg-blue-500/10 text-blue-400 border border-blue-500/20 font-medium">RAG Q&A</span>
                </h1>
                <p class="text-xs text-slate-400">Pure Python RAG &bull; all-MiniLM-L6-v2 &bull; FAISS Vector Search</p>
            </div>
        </div>
        <div class="flex items-center gap-3">
            <button id="settingsBtn" class="text-xs px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-300 transition flex items-center gap-1.5">
                <i class="fa-solid fa-sliders text-indigo-400"></i>
                <span>LLM Settings</span>
            </button>
        </div>
    </header>

    <!-- Main Container -->
    <div class="flex-1 max-w-7xl w-full mx-auto p-4 md:p-6 grid grid-cols-1 md:grid-cols-3 gap-6">
        
        <!-- Left Panel: My Documents (Upload & Knowledge Base) -->
        <div class="bg-slate-900 rounded-2xl border border-slate-800 p-5 flex flex-col h-[78vh] shadow-xl">
            <div class="flex items-center justify-between mb-3">
                <h2 class="text-sm font-bold text-white flex items-center gap-2">
                    <i class="fa-solid fa-book-bookmark text-blue-400"></i>
                    <span>Uploaded Documents</span>
                </h2>
                <span id="docBadge" class="text-[11px] px-2 py-0.5 rounded-full bg-blue-950 text-blue-300 font-semibold border border-blue-800/50">0 files</span>
            </div>

            <!-- Upload Dropzone -->
            <div id="dropzone" class="border-2 border-dashed border-slate-700 hover:border-blue-500 bg-slate-950/60 rounded-xl p-5 text-center cursor-pointer transition mb-4">
                <input type="file" id="fileInput" multiple accept=".pdf,.docx,.txt,.csv,.md" class="hidden">
                <div class="w-10 h-10 rounded-full bg-blue-500/10 text-blue-400 flex items-center justify-center mx-auto mb-2 text-lg">
                    <i class="fa-solid fa-cloud-arrow-up"></i>
                </div>
                <div class="text-xs font-semibold text-white">Upload Documents</div>
                <p class="text-[10px] text-slate-400 mt-0.5">Click or drag PDF, Word, or TXT files</p>
            </div>

            <!-- Indexing progress indicator -->
            <div id="uploadingBox" class="hidden p-3 rounded-xl bg-blue-950/60 border border-blue-500/30 text-blue-300 text-xs text-center mb-3">
                <i class="fa-solid fa-circle-notch fa-spin mr-1.5"></i> Extracting chunks & computing FAISS embeddings...
            </div>

            <!-- List of Documents -->
            <div class="flex items-center justify-between text-xs text-slate-400 mb-2">
                <span class="font-medium">Indexed in Vector DB</span>
                <button id="clearAllBtn" class="text-rose-400 hover:text-rose-300 text-[11px] transition">Clear All</button>
            </div>
            <div id="documentsList" class="flex-1 overflow-y-auto space-y-2 pr-1 text-xs">
                <div class="text-center py-12 text-slate-500">No documents in knowledge base yet.</div>
            </div>
        </div>

        <!-- Right Panel: Q&A Chat Area -->
        <div class="md:col-span-2 bg-slate-900 rounded-2xl border border-slate-800 p-5 flex flex-col h-[78vh] shadow-xl">
            <!-- Messages Stream -->
            <div id="chatMessages" class="flex-1 overflow-y-auto space-y-4 pr-2 mb-4">
                <div id="welcomeMessage" class="text-center py-20 px-4">
                    <div class="w-14 h-14 rounded-2xl bg-gradient-to-tr from-blue-500 to-indigo-600 text-white flex items-center justify-center text-2xl mx-auto mb-3 shadow-lg">
                        <i class="fa-solid fa-magnifying-glass-chart"></i>
                    </div>
                    <h3 class="text-lg font-bold text-white mb-1">Ask questions on your documents</h3>
                    <p class="text-xs text-slate-400 max-w-md mx-auto">Upload any PDF or document to the left. The system will retrieve relevant chunks using FAISS and answer your questions with precise citations.</p>
                </div>
            </div>

            <!-- Searching indicator -->
            <div id="searchingIndicator" class="hidden text-xs text-blue-400 mb-2 flex items-center gap-2">
                <i class="fa-solid fa-circle-notch fa-spin"></i>
                <span>Retrieving relevant chunks from FAISS vector store...</span>
            </div>

            <!-- Input Form -->
            <form id="questionForm" class="flex gap-2">
                <input
                    type="text"
                    id="questionInput"
                    placeholder="Ask a question based on your uploaded documents..."
                    class="flex-1 bg-slate-950 border border-slate-800 rounded-xl px-4 py-3 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-blue-500 transition"
                    required
                />
                <button
                    type="submit"
                    id="sendBtn"
                    class="px-5 py-3 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-semibold text-xs shadow-md transition flex items-center gap-2"
                >
                    <span>Ask AI</span>
                    <i class="fa-solid fa-arrow-up text-xs"></i>
                </button>
            </form>
        </div>
    </div>

    <!-- Settings Modal -->
    <div id="settingsModal" class="hidden fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50 p-4">
        <div class="bg-slate-900 border border-slate-800 rounded-2xl w-full max-w-md p-5 shadow-2xl space-y-4 text-xs">
            <div class="flex items-center justify-between border-b border-slate-800 pb-3">
                <h3 class="text-sm font-bold text-white flex items-center gap-2">
                    <i class="fa-solid fa-sliders text-blue-400"></i>
                    <span>Model Settings</span>
                </h3>
                <button id="closeSettingsBtn" class="text-slate-400 hover:text-white">
                    <i class="fa-solid fa-xmark text-base"></i>
                </button>
            </div>

            <div>
                <label class="block font-medium text-slate-300 mb-1">LLM Provider</label>
                <select id="providerSelect" class="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-white">
                    <option value="gemini">Google Gemini (Free API Key)</option>
                    <option value="ollama">Ollama Local (http://localhost:11434)</option>
                    <option value="groq">Groq Live AI (Ultra Fast)</option>
                    <option value="openai">OpenAI (GPT-4o-mini)</option>
                </select>
            </div>

            <div>
                <label class="block font-medium text-slate-300 mb-1">API Key (Gemini / Groq / OpenAI)</label>
                <input type="password" id="apiKeyInput" placeholder="Paste your API key (e.g. AIza... or gsk_...)" class="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-white">
            </div>

            <div class="p-3 rounded-lg bg-slate-950/80 border border-slate-800 text-[11px] text-slate-400 space-y-1">
                <div><strong>Embedding Model:</strong> all-MiniLM-L6-v2 (384d)</div>
                <div><strong>Vector Database:</strong> FAISS FlatL2 (Local)</div>
            </div>

            <div class="flex justify-end gap-2 pt-2 border-t border-slate-800">
                <button id="saveSettingsBtn" class="px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-white font-semibold">Save Settings</button>
            </div>
        </div>
    </div>

    <!-- Frontend Script -->
    <script>
        const dropzone = document.getElementById('dropzone');
        const fileInput = document.getElementById('fileInput');
        const uploadingBox = document.getElementById('uploadingBox');
        const documentsList = document.getElementById('documentsList');
        const docBadge = document.getElementById('docBadge');
        const clearAllBtn = document.getElementById('clearAllBtn');
        const chatMessages = document.getElementById('chatMessages');
        const welcomeMessage = document.getElementById('welcomeMessage');
        const questionForm = document.getElementById('questionForm');
        const questionInput = document.getElementById('questionInput');
        const searchingIndicator = document.getElementById('searchingIndicator');
        const sendBtn = document.getElementById('sendBtn');
        const settingsModal = document.getElementById('settingsModal');
        const settingsBtn = document.getElementById('settingsBtn');
        const closeSettingsBtn = document.getElementById('closeSettingsBtn');
        const saveSettingsBtn = document.getElementById('saveSettingsBtn');
        const providerSelect = document.getElementById('providerSelect');
        const apiKeyInput = document.getElementById('apiKeyInput');

        let currentApiKey = localStorage.getItem('my_doc_api_key') || '';
        let currentProvider = localStorage.getItem('my_doc_provider') || 'gemini';

        providerSelect.value = currentProvider;
        apiKeyInput.value = currentApiKey;

        // Settings Modal
        settingsBtn.addEventListener('click', () => settingsModal.classList.remove('hidden'));
        closeSettingsBtn.addEventListener('click', () => settingsModal.classList.add('hidden'));
        saveSettingsBtn.addEventListener('click', () => {
            currentApiKey = apiKeyInput.value.trim();
            currentProvider = providerSelect.value;
            localStorage.setItem('my_doc_api_key', currentApiKey);
            localStorage.setItem('my_doc_provider', currentProvider);
            settingsModal.classList.add('hidden');
            alert('Settings saved!');
        });

        // Dropzone
        dropzone.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', (e) => uploadFiles(e.target.files));

        dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('border-blue-500'); });
        dropzone.addEventListener('dragleave', () => dropzone.classList.remove('border-blue-500'));
        dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropzone.classList.remove('border-blue-500');
            uploadFiles(e.dataTransfer.files);
        });

        async function uploadFiles(files) {
            if (!files || files.length === 0) return;
            uploadingBox.classList.remove('hidden');

            const formData = new FormData();
            for (let i = 0; i < files.length; i++) {
                formData.append('files', files[i]);
            }

            try {
                const res = await fetch('/api/upload', { method: 'POST', body: formData });
                const data = await res.json();
                if (data.status === 'success') {
                    fetchDocuments();
                } else {
                    alert('Upload failed: ' + data.error);
                }
            } catch (err) {
                alert('Upload error: ' + err.message);
            } finally {
                uploadingBox.classList.add('hidden');
                fileInput.value = '';
            }
        }

        async function fetchDocuments() {
            try {
                const res = await fetch('/api/documents');
                const data = await res.json();
                renderDocuments(data.documents || []);
            } catch (err) {
                console.error(err);
            }
        }

        function renderDocuments(docs) {
            docBadge.textContent = `${docs.length} file${docs.length === 1 ? '' : 's'}`;
            documentsList.innerHTML = '';
            if (docs.length === 0) {
                documentsList.innerHTML = '<div class="text-center py-12 text-slate-500">No documents in knowledge base yet.</div>';
                return;
            }

            docs.forEach(doc => {
                const item = document.createElement('div');
                item.className = 'p-2.5 rounded-xl bg-slate-950/80 border border-slate-800 flex items-center justify-between text-xs group';
                item.innerHTML = `
                    <div class="flex items-center gap-2 truncate flex-1">
                        <i class="fa-solid fa-file-lines text-blue-400 text-sm flex-shrink-0"></i>
                        <div class="truncate">
                            <div class="font-semibold text-white truncate">${doc.filename}</div>
                            <div class="text-[10px] text-slate-400">${doc.chunks_count} chunks in FAISS</div>
                        </div>
                    </div>
                    <div class="flex items-center gap-1.5 flex-shrink-0">
                        <span class="text-[10px] px-2 py-0.5 rounded-full bg-emerald-950/80 text-emerald-400 border border-emerald-500/20 font-medium">Ready</span>
                        <button class="delete-single-btn text-slate-500 hover:text-rose-400 p-1 transition" title="Delete document">
                            <i class="fa-solid fa-trash text-xs"></i>
                        </button>
                    </div>
                `;

                item.querySelector('.delete-single-btn').addEventListener('click', async (e) => {
                    e.stopPropagation();
                    if (confirm(`Remove "${doc.filename}" from FAISS index?`)) {
                        await fetch('/api/delete', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ filename: doc.filename })
                        });
                        fetchDocuments();
                    }
                });

                documentsList.appendChild(item);
            });
        }

        clearAllBtn.addEventListener('click', async () => {
            if (confirm('Delete all documents and reset FAISS vector index?')) {
                await fetch('/api/clear', { method: 'POST' });
                fetchDocuments();
                chatMessages.innerHTML = '';
                chatMessages.appendChild(welcomeMessage);
            }
        });

        questionForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const text = questionInput.value.trim();
            if (!text) return;

            if (welcomeMessage) welcomeMessage.classList.add('hidden');
            questionInput.value = '';
            appendMessageBubble('user', text);
            searchingIndicator.classList.remove('hidden');
            sendBtn.disabled = true;

            try {
                const res = await fetch('/api/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        query: text,
                        provider: currentProvider,
                        api_key: currentApiKey
                    })
                });
                const data = await res.json();
                appendMessageBubble('assistant', data.answer, data.sources);
            } catch (err) {
                appendMessageBubble('assistant', '⚠️ Error: ' + err.message);
            } finally {
                searchingIndicator.classList.add('hidden');
                sendBtn.disabled = false;
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
        });

        function appendMessageBubble(role, text, sources = []) {
            const isUser = role === 'user';
            const msgEl = document.createElement('div');
            msgEl.className = `flex flex-col ${isUser ? 'items-end' : 'items-start'} text-xs space-y-1`;

            let sourcesHtml = '';
            if (!isUser && sources && sources.length > 0) {
                const items = sources.map(s => `
                    <div class="p-2.5 rounded-lg bg-slate-950 border border-slate-800 text-[11px] mt-1 space-y-1">
                        <div class="font-semibold text-blue-300 flex justify-between">
                            <span>📄 ${s.filename} (Page ${s.page})</span>
                            <span class="text-[10px] text-emerald-400">FAISS Score: ${s.score}</span>
                        </div>
                        <p class="text-slate-400 italic bg-slate-900/60 p-1.5 rounded">"${s.snippet}"</p>
                    </div>
                `).join('');

                sourcesHtml = `
                    <details class="mt-2.5 pt-2 border-t border-slate-800 w-full text-slate-400">
                        <summary class="cursor-pointer font-semibold text-blue-400 hover:text-blue-300 text-[11px]">
                            📚 View ${sources.length} Referenced FAISS Chunks
                        </summary>
                        <div class="mt-2 space-y-1.5">${items}</div>
                    </details>
                `;
            }

            msgEl.innerHTML = `
                <div class="text-[10px] font-semibold text-slate-400 px-1">${isUser ? 'You' : 'My Documents AI'}</div>
                <div class="p-4 rounded-2xl max-w-xl ${isUser ? 'bg-blue-600 text-white' : 'bg-slate-900 text-slate-200 border border-slate-800'} markdown-body shadow-md">
                    ${isUser ? text : marked.parse(text)}
                    ${sourcesHtml}
                </div>
            `;

            chatMessages.appendChild(msgEl);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        fetchDocuments();
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(UI_TEMPLATE)


@app.route('/api/upload', methods=['POST'])
def api_upload():
    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': 'No file attached'}), 400

    total_chunks = 0
    for f in files:
        if not f.filename:
            continue
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        f.save(save_path)
        chunks_added = add_document_to_faiss(save_path, f.filename)
        total_chunks += chunks_added

    return jsonify({
        'status': 'success',
        'total_chunks': total_chunks,
        'message': f'Embedded {total_chunks} chunks into FAISS vector database.'
    })


@app.route('/api/documents', methods=['GET'])
def api_documents():
    return jsonify({'documents': uploaded_documents})


@app.route('/api/clear', methods=['POST'])
def api_clear():
    global faiss_index, chunks_registry, uploaded_documents
    chunks_registry = []
    uploaded_documents = []
    faiss_index = faiss.IndexFlatL2(EMBEDDING_DIM)

    if os.path.exists(app.config['INDEX_FOLDER']):
        shutil.rmtree(app.config['INDEX_FOLDER'])
    os.makedirs(app.config['INDEX_FOLDER'], exist_ok=True)

    if os.path.exists(app.config['UPLOAD_FOLDER']):
        shutil.rmtree(app.config['UPLOAD_FOLDER'])
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    save_faiss_index()
    return jsonify({'status': 'success', 'message': 'All documents and FAISS index cleared.'})


@app.route('/api/delete', methods=['POST'])
def api_delete_doc():
    global faiss_index, chunks_registry, uploaded_documents
    data = request.get_json() or {}
    filename = data.get('filename', '').strip()
    if not filename:
        return jsonify({'error': 'Filename required'}), 400

    # Filter out chunks and document record
    chunks_registry = [c for c in chunks_registry if c['filename'] != filename]
    uploaded_documents = [d for d in uploaded_documents if d['filename'] != filename]

    # Rebuild FAISS index from remaining chunks
    faiss_index = faiss.IndexFlatL2(EMBEDDING_DIM)
    if chunks_registry:
        texts = [c['text'] for c in chunks_registry]
        embeddings = embedding_model.encode(texts, normalize_embeddings=True)
        faiss_index.add(np.array(embeddings, dtype=np.float32))

    # Delete physical file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception:
            pass

    save_faiss_index()
    return jsonify({'status': 'success', 'message': f'Document {filename} removed from index.'})



@app.route('/api/query', methods=['POST'])
def api_query():
    data = request.get_json() or {}
    query_text = data.get('query', '').strip()
    provider = data.get('provider', 'gemini')
    api_key = data.get('api_key', None)

    if not query_text:
        return jsonify({'error': 'Question cannot be empty'}), 400

    # 1. Search FAISS
    retrieved_chunks = search_faiss(query_text, top_k=3)
    if not retrieved_chunks:
        return jsonify({
            'answer': 'No documents found in the knowledge base. Please upload documents first.',
            'sources': []
        })

    # 2. Build Context String
    context_parts = []
    for c in retrieved_chunks:
        context_parts.append(f"[Document: {c['filename']}, Page: {c['page']}]\n{c['text']}")
    context_str = "\n\n".join(context_parts)

    # 3. Call LLM
    answer = call_llm(query_text, context_str, provider=provider, api_key=api_key)

    # 4. Pure Python Fallback Answer if LLM API is unavailable
    if not answer:
        answer = f"Based on your documents, here are the most relevant findings for **\"{query_text}\"**:\n\n"
        for c in retrieved_chunks:
            answer += f"- **From {c['filename']} (Page {c['page']}):**\n  > \"{c['text']}\"\n\n"
        answer += "\n---\n💡 *Configure a Google Gemini API Key or run Ollama to synthesize natural conversational responses.*"

    return jsonify({
        'answer': answer,
        'sources': retrieved_chunks
    })


if __name__ == '__main__':
    print("\n" + "="*70)
    print(">> 'My Documents' Simple RAG Application is running!")
    print(">> Open in browser: http://127.0.0.1:5000")
    print("="*70 + "\n")
    app.run(host='127.0.0.1', port=5000, debug=True)

