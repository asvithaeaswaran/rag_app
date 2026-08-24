import os
import json
import shutil
import requests
import numpy as np
import faiss
import pypdf
import docx

from flask import Flask, request, jsonify, render_template_string
from dotenv import load_dotenv

load_dotenv()

# -----------------------------------------------------------------------------
# 1. FLASK APPLICATION
# -----------------------------------------------------------------------------

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "my_documents_files")
INDEX_FOLDER = os.path.join(BASE_DIR, "my_documents_faiss")

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["INDEX_FOLDER"] = INDEX_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(INDEX_FOLDER, exist_ok=True)


# -----------------------------------------------------------------------------
# 2. OPENAI EMBEDDING MODEL
# -----------------------------------------------------------------------------

# OpenAI text-embedding-3-small creates 1536-dimensional embeddings.
EMBEDDING_DIM = 1536

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

print("[My Documents] Using OpenAI text-embedding-3-small for embeddings.")


def create_embeddings(texts):
    """
    Create embeddings using OpenAI API.
    This avoids loading PyTorch / sentence-transformers into Render memory.
    """

    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        raise Exception("OPENAI_API_KEY is not configured.")

    url = "https://api.openai.com/v1/embeddings"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "text-embedding-3-small",
        "input": texts
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=60
    )

    if response.status_code != 200:
        raise Exception(
            f"OpenAI Embedding API error: {response.status_code} - {response.text}"
        )

    data = response.json()

    embeddings = [
        item["embedding"]
        for item in data["data"]
    ]

    return np.array(embeddings, dtype=np.float32)


# -----------------------------------------------------------------------------
# 3. IN-MEMORY DOCUMENT REGISTRY
# -----------------------------------------------------------------------------

chunks_registry = []
uploaded_documents = []


# -----------------------------------------------------------------------------
# 4. DOCUMENT PARSERS & TEXT EXTRACTORS
# -----------------------------------------------------------------------------

def extract_text(file_path: str, filename: str) -> list[dict]:
    """
    Extract text from PDF, DOCX, TXT, CSV and Markdown.
    Returns:
        [
            {
                "text": "...",
                "page": 1
            }
        ]
    """

    ext = (
        filename.rsplit(".", 1)[-1].lower()
        if "." in filename
        else "txt"
    )

    pages = []

    # -------------------------------------------------------------------------
    # PDF
    # -------------------------------------------------------------------------

    if ext == "pdf":

        try:
            reader = pypdf.PdfReader(file_path)

            for i, page in enumerate(reader.pages):

                txt = (page.extract_text() or "").strip()

                if txt:
                    pages.append({
                        "text": txt,
                        "page": i + 1
                    })

        except Exception as e:
            print(f"Error reading PDF {filename}: {e}")

    # -------------------------------------------------------------------------
    # DOCX
    # -------------------------------------------------------------------------

    elif ext in ["docx", "doc"]:

        try:
            document = docx.Document(file_path)

            paragraphs = [
                p.text.strip()
                for p in document.paragraphs
                if p.text.strip()
            ]

            for table in document.tables:

                for row in table.rows:

                    row_text = " | ".join(
                        cell.text.strip()
                        for cell in row.cells
                        if cell.text.strip()
                    )

                    if row_text:
                        paragraphs.append(row_text)

            full_text = "\n\n".join(paragraphs)

            if full_text.strip():

                pages.append({
                    "text": full_text,
                    "page": 1
                })

        except Exception as e:
            print(f"Error reading DOCX {filename}: {e}")

    # -------------------------------------------------------------------------
    # TXT / CSV / MD
    # -------------------------------------------------------------------------

    else:

        try:

            with open(
                file_path,
                "r",
                encoding="utf-8",
                errors="replace"
            ) as f:

                content = f.read().strip()

                if content:

                    pages.append({
                        "text": content,
                        "page": 1
                    })

        except Exception as e:
            print(f"Error reading text file {filename}: {e}")

    return pages


# -----------------------------------------------------------------------------
# 5. SIMPLE TEXT CHUNKER
# -----------------------------------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 80
) -> list[str]:

    if len(text) <= chunk_size:
        return [text]

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        if end < len(text):

            break_pos = text.rfind(
                "\n",
                start,
                end
            )

            if (
                break_pos == -1
                or break_pos < start + (chunk_size // 2)
            ):
                break_pos = text.rfind(
                    ". ",
                    start,
                    end
                )

            if (
                break_pos == -1
                or break_pos < start + (chunk_size // 2)
            ):
                break_pos = text.rfind(
                    " ",
                    start,
                    end
                )

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
# 6. FAISS STORAGE
# -----------------------------------------------------------------------------

faiss_index_path = os.path.join(
    INDEX_FOLDER,
    "my_documents.index"
)

meta_path = os.path.join(
    INDEX_FOLDER,
    "metadata.json"
)


def init_or_load_faiss_index():

    global chunks_registry
    global uploaded_documents

    if (
        os.path.exists(faiss_index_path)
        and os.path.exists(meta_path)
    ):

        try:

            index = faiss.read_index(
                faiss_index_path
            )

            with open(
                meta_path,
                "r",
                encoding="utf-8"
            ) as f:

                data = json.load(f)

                chunks_registry = data.get(
                    "chunks",
                    []
                )

                uploaded_documents = data.get(
                    "documents",
                    []
                )

            return index

        except Exception as e:

            print(
                f"Failed to load FAISS index: {e}"
            )

    return faiss.IndexFlatL2(
        EMBEDDING_DIM
    )


faiss_index = init_or_load_faiss_index()


def save_faiss_index():

    os.makedirs(
        INDEX_FOLDER,
        exist_ok=True
    )

    faiss.write_index(
        faiss_index,
        faiss_index_path
    )

    with open(
        meta_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "chunks": chunks_registry,
                "documents": uploaded_documents
            },
            f,
            indent=2
        )


# -----------------------------------------------------------------------------
# 7. ADD DOCUMENT TO FAISS
# -----------------------------------------------------------------------------

def add_document_to_faiss(
    file_path: str,
    filename: str
) -> int:

    global faiss_index
    global chunks_registry
    global uploaded_documents

    pages = extract_text(
        file_path,
        filename
    )

    if not pages:
        return 0

    new_chunks = []

    for page_data in pages:

        text_splits = chunk_text(
            page_data["text"],
            chunk_size=500,
            overlap=80
        )

        for idx, chunk_str in enumerate(
            text_splits
        ):

            new_chunks.append(
                {
                    "filename": filename,
                    "page": page_data["page"],
                    "chunk_index": idx,
                    "text": chunk_str
                }
            )

    if not new_chunks:
        return 0

    # -------------------------------------------------------------------------
    # Create embeddings using OpenAI
    # -------------------------------------------------------------------------

    texts_to_embed = [
        chunk["text"]
        for chunk in new_chunks
    ]

    embeddings = create_embeddings(
        texts_to_embed
    )

    # -------------------------------------------------------------------------
    # Add embeddings to FAISS
    # -------------------------------------------------------------------------

    faiss_index.add(
        embeddings
    )

    chunks_registry.extend(
        new_chunks
    )

    doc_info = {
        "filename": filename,
        "chunks_count": len(new_chunks),
        "file_size": os.path.getsize(file_path)
    }

    uploaded_documents = [
        d
        for d in uploaded_documents
        if d["filename"] != filename
    ]

    uploaded_documents.append(
        doc_info
    )

    save_faiss_index()

    return len(new_chunks)


# -----------------------------------------------------------------------------
# 8. SEARCH FAISS
# -----------------------------------------------------------------------------

def search_faiss(
    query: str,
    top_k: int = 3
) -> list[dict]:

    if (
        faiss_index.ntotal == 0
        or len(chunks_registry) == 0
    ):
        return []

    query_embedding = create_embeddings(
        [query]
    )

    k = min(
        top_k,
        faiss_index.ntotal
    )

    distances, indices = faiss_index.search(
        query_embedding,
        k
    )

    results = []

    for rank, (idx, dist) in enumerate(
        zip(
            indices[0],
            distances[0]
        ),
        1
    ):

        if idx < len(chunks_registry):

            chunk = chunks_registry[idx]

            results.append(
                {
                    "rank": rank,
                    "filename": chunk["filename"],
                    "page": chunk["page"],
                    "text": chunk["text"],
                    "score": round(
                        float(dist),
                        4
                    ),
                    "snippet": (
                        chunk["text"][:250]
                        + (
                            "..."
                            if len(chunk["text"]) > 250
                            else ""
                        )
                    )
                }
            )

    return results


# -----------------------------------------------------------------------------
# 9. DIRECT LLM CALL
# -----------------------------------------------------------------------------

def call_llm(
    user_question: str,
    context_text: str,
    provider: str = "gemini",
    api_key: str = None
) -> str:

    system_instruction = (
        "You are 'My Documents AI', a helpful and precise assistant. "
        "Answer the user's question accurately using ONLY the provided "
        "document excerpts below. "
        "If the information is not contained in the excerpts, clearly say "
        "that the uploaded documents do not have this information. "
        "Format your answer cleanly with bullet points and bold key terms."
    )

    full_prompt = (
        f"DOCUMENT EXCERPTS:\n"
        f"{context_text}\n\n"
        f"USER QUESTION: {user_question}\n\n"
        "Please provide a complete and accurate answer "
        "based on the document excerpts above."
    )

    openai_key = (
        api_key
        or os.environ.get("OPENAI_API_KEY")
    )

    if not openai_key:
        return None

    # -------------------------------------------------------------------------
    # OpenAI
    # -------------------------------------------------------------------------

    try:

        url = (
            "https://api.openai.com/v1/chat/completions"
        )

        headers = {
            "Authorization": f"Bearer {openai_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": os.environ.get(
                "OPENAI_MODEL",
                "gpt-4o-mini"
            ),
            "messages": [
                {
                    "role": "system",
                    "content": system_instruction
                },
                {
                    "role": "user",
                    "content": full_prompt
                }
            ],
            "temperature": 0.3,
            "max_tokens": 2048
        }

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=60
        )

        if response.status_code == 200:

            data = response.json()

            return (
                data["choices"][0]["message"]["content"]
            )

        print(
            "OpenAI API error:",
            response.status_code,
            response.text
        )

    except Exception as e:

        print(
            "OpenAI request error:",
            e
        )

    return None


# -----------------------------------------------------------------------------
# 10. HTML USER INTERFACE
# -----------------------------------------------------------------------------

UI_TEMPLATE = """
<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta
name="viewport"
content="width=device-width, initial-scale=1.0"
>

<title>My Documents — Simple RAG System</title>

<script src="https://cdn.tailwindcss.com"></script>

<script
src="https://cdn.jsdelivr.net/npm/marked/marked.min.js">
</script>

<link
rel="stylesheet"
href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css"
>

<style>

::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}

::-webkit-scrollbar-track {
    background: transparent;
}

::-webkit-scrollbar-thumb {
    background: #334155;
    border-radius: 9999px;
}

.markdown-body p {
    margin-bottom: 0.75rem;
}

.markdown-body ul {
    list-style-type: disc;
    padding-left: 1.25rem;
    margin-bottom: 0.75rem;
}

.markdown-body pre {
    background: #0f172a;
    padding: 0.75rem;
    border-radius: 0.5rem;
    overflow-x: auto;
}

.markdown-body code {
    background: rgba(255,255,255,0.1);
    padding: 0.1rem 0.3rem;
    border-radius: 0.2rem;
}

</style>

</head>

<body
class="bg-slate-950 text-slate-100 min-h-screen flex flex-col font-sans"
>

<header
class="bg-slate-900 border-b border-slate-800 px-6 py-4 flex items-center justify-between sticky top-0 z-20"
>

<div class="flex items-center gap-3">

<div
class="w-10 h-10 rounded-xl bg-blue-600 flex items-center justify-center text-white"
>

<i class="fa-solid fa-folder-tree"></i>

</div>

<div>

<h1 class="text-lg font-bold text-white">

My Documents

<span
class="text-[10px] px-2 py-0.5 rounded-full bg-blue-500/10 text-blue-400 border border-blue-500/20"
>
RAG Q&A
</span>

</h1>

<p class="text-xs text-slate-400">
OpenAI Embeddings • FAISS Vector Search
</p>

</div>

</div>

</header>


<div
class="flex-1 max-w-7xl w-full mx-auto p-4 md:p-6 grid grid-cols-1 md:grid-cols-3 gap-6"
>


<!-- LEFT -->

<div
class="bg-slate-900 rounded-2xl border border-slate-800 p-5 flex flex-col h-[78vh]"
>

<div
class="flex items-center justify-between mb-3"
>

<h2 class="text-sm font-bold text-white">

<i class="fa-solid fa-book-bookmark text-blue-400"></i>

Uploaded Documents

</h2>

<span
id="docBadge"
class="text-[11px] px-2 py-0.5 rounded-full bg-blue-950 text-blue-300"
>
0 files
</span>

</div>


<div
id="dropzone"
class="border-2 border-dashed border-slate-700 hover:border-blue-500 bg-slate-950/60 rounded-xl p-5 text-center cursor-pointer transition mb-4"
>

<input
type="file"
id="fileInput"
multiple
accept=".pdf,.docx,.txt,.csv,.md"
class="hidden"
>

<div
class="w-10 h-10 rounded-full bg-blue-500/10 text-blue-400 flex items-center justify-center mx-auto mb-2"
>

<i class="fa-solid fa-cloud-arrow-up"></i>

</div>

<div class="text-xs font-semibold text-white">
Upload Documents
</div>

<p class="text-[10px] text-slate-400">
Click or drag PDF, Word, or TXT files
</p>

</div>


<div
id="uploadingBox"
class="hidden p-3 rounded-xl bg-blue-950/60 border border-blue-500/30 text-blue-300 text-xs text-center mb-3"
>

<i class="fa-solid fa-circle-notch fa-spin"></i>

Extracting text, creating embeddings and indexing...

</div>


<div
class="flex items-center justify-between text-xs text-slate-400 mb-2"
>

<span>
Indexed in Vector DB
</span>

<button
id="clearAllBtn"
class="text-rose-400"
>
Clear All
</button>

</div>


<div
id="documentsList"
class="flex-1 overflow-y-auto space-y-2 pr-1 text-xs"
>

<div
class="text-center py-12 text-slate-500"
>
No documents in knowledge base yet.
</div>

</div>

</div>


<!-- RIGHT -->

<div
class="md:col-span-2 bg-slate-900 rounded-2xl border border-slate-800 p-5 flex flex-col h-[78vh]"
>

<div
id="chatMessages"
class="flex-1 overflow-y-auto space-y-4 pr-2 mb-4"
>

<div
id="welcomeMessage"
class="text-center py-20 px-4"
>

<div
class="w-14 h-14 rounded-2xl bg-blue-600 text-white flex items-center justify-center text-2xl mx-auto mb-3"
>

<i class="fa-solid fa-magnifying-glass-chart"></i>

</div>

<h3 class="text-lg font-bold text-white mb-1">

Ask questions on your documents

</h3>

<p
class="text-xs text-slate-400 max-w-md mx-auto"
>

Upload a document to the left.

The system will retrieve relevant chunks
using FAISS and answer your question.

</p>

</div>

</div>


<div
id="searchingIndicator"
class="hidden text-xs text-blue-400 mb-2"
>

<i class="fa-solid fa-circle-notch fa-spin"></i>

Retrieving relevant chunks...

</div>


<form
id="questionForm"
class="flex gap-2"
>

<input
type="text"
id="questionInput"
placeholder="Ask a question based on your uploaded documents..."
class="flex-1 bg-slate-950 border border-slate-800 rounded-xl px-4 py-3 text-xs text-white"
required
>

<button
type="submit"
id="sendBtn"
class="px-5 py-3 rounded-xl bg-blue-600 text-white font-semibold text-xs"
>

Ask AI

<i class="fa-solid fa-arrow-up"></i>

</button>

</form>

</div>

</div>


<script>

const dropzone =
document.getElementById('dropzone');

const fileInput =
document.getElementById('fileInput');

const uploadingBox =
document.getElementById('uploadingBox');

const documentsList =
document.getElementById('documentsList');

const docBadge =
document.getElementById('docBadge');

const clearAllBtn =
document.getElementById('clearAllBtn');

const chatMessages =
document.getElementById('chatMessages');

const welcomeMessage =
document.getElementById('welcomeMessage');

const questionForm =
document.getElementById('questionForm');

const questionInput =
document.getElementById('questionInput');

const searchingIndicator =
document.getElementById('searchingIndicator');

const sendBtn =
document.getElementById('sendBtn');


dropzone.addEventListener(
'click',
() => fileInput.click()
);


fileInput.addEventListener(
'change',
(e) => uploadFiles(e.target.files)
);


dropzone.addEventListener(
'dragover',
(e) => {
    e.preventDefault();
}
);


dropzone.addEventListener(
'drop',
(e) => {

    e.preventDefault();

    uploadFiles(
        e.dataTransfer.files
    );

}
);


async function uploadFiles(files) {

    if (!files || files.length === 0)
        return;

    uploadingBox.classList.remove(
        'hidden'
    );

    const formData =
        new FormData();

    for (
        let i = 0;
        i < files.length;
        i++
    ) {

        formData.append(
            'files',
            files[i]
        );

    }

    try {

        const res =
            await fetch(
                '/api/upload',
                {
                    method: 'POST',
                    body: formData
                }
            );

        const data =
            await res.json();

        if (
            data.status === 'success'
        ) {

            fetchDocuments();

            alert(
                data.message
            );

        } else {

            alert(
                'Upload failed: ' +
                data.error
            );

        }

    } catch (err) {

        alert(
            'Upload error: ' +
            err.message
        );

    } finally {

        uploadingBox.classList.add(
            'hidden'
        );

        fileInput.value = '';

    }

}


async function fetchDocuments() {

    try {

        const res =
            await fetch(
                '/api/documents'
            );

        const data =
            await res.json();

        renderDocuments(
            data.documents || []
        );

    } catch (err) {

        console.error(err);

    }

}


function renderDocuments(docs) {

    docBadge.textContent =
        `${docs.length} file${docs.length === 1 ? '' : 's'}`;

    documentsList.innerHTML = '';

    if (docs.length === 0) {

        documentsList.innerHTML =
            '<div class="text-center py-12 text-slate-500">No documents in knowledge base yet.</div>';

        return;

    }


    docs.forEach(doc => {

        const item =
            document.createElement(
                'div'
            );

        item.className =
            'p-2.5 rounded-xl bg-slate-950 border border-slate-800 flex items-center justify-between';


        item.innerHTML = `

<div class="flex items-center gap-2 truncate">

<i class="fa-solid fa-file-lines text-blue-400"></i>

<div class="truncate">

<div class="font-semibold text-white truncate">

${doc.filename}

</div>

<div class="text-[10px] text-slate-400">

${doc.chunks_count} chunks in FAISS

</div>

</div>

</div>


<div class="flex items-center gap-2">

<span
class="text-[10px] px-2 py-0.5 rounded-full bg-emerald-950 text-emerald-400"
>
Ready
</span>

<button
class="delete-single-btn text-rose-400"
>
<i class="fa-solid fa-trash"></i>
</button>

</div>

`;


        item
        .querySelector(
            '.delete-single-btn'
        )
        .addEventListener(
            'click',
            async () => {

                if (
                    confirm(
                        `Remove "${doc.filename}"?`
                    )
                ) {

                    await fetch(
                        '/api/delete',
                        {
                            method: 'POST',
                            headers: {
                                'Content-Type':
                                'application/json'
                            },
                            body:
                                JSON.stringify({
                                    filename:
                                        doc.filename
                                })
                        }
                    );

                    fetchDocuments();

                }

            }
        );


        documentsList.appendChild(
            item
        );

    });

}


clearAllBtn.addEventListener(
'click',
async () => {

    if (
        confirm(
            'Delete all documents and reset FAISS?'
        )
    ) {

        await fetch(
            '/api/clear',
            {
                method: 'POST'
            }
        );

        fetchDocuments();

        chatMessages.innerHTML = '';

        chatMessages.appendChild(
            welcomeMessage
        );

    }

});


questionForm.addEventListener(
'submit',
async (e) => {

    e.preventDefault();

    const text =
        questionInput.value.trim();

    if (!text)
        return;


    welcomeMessage.classList.add(
        'hidden'
    );

    questionInput.value = '';

    appendMessageBubble(
        'user',
        text
    );

    searchingIndicator.classList.remove(
        'hidden'
    );

    sendBtn.disabled = true;


    try {

        const res =
            await fetch(
                '/api/query',
                {
                    method: 'POST',
                    headers: {
                        'Content-Type':
                        'application/json'
                    },
                    body:
                        JSON.stringify({
                            query: text
                        })
                }
            );

        const data =
            await res.json();


        appendMessageBubble(
            'assistant',
            data.answer,
            data.sources
        );


    } catch (err) {

        appendMessageBubble(
            'assistant',
            'Error: ' + err.message
        );

    } finally {

        searchingIndicator.classList.add(
            'hidden'
        );

        sendBtn.disabled = false;

        chatMessages.scrollTop =
            chatMessages.scrollHeight;

    }

});


function appendMessageBubble(
    role,
    text,
    sources = []
) {

    const isUser =
        role === 'user';

    const msgEl =
        document.createElement(
            'div'
        );

    msgEl.className =
        `flex flex-col ${
            isUser
            ? 'items-end'
            : 'items-start'
        } text-xs space-y-1`;


    let sourcesHtml = '';


    if (
        !isUser &&
        sources &&
        sources.length > 0
    ) {

        const items =
            sources.map(
                s => `

<div
class="p-2.5 rounded-lg bg-slate-950 border border-slate-800 text-[11px] mt-1"
>

<div
class="font-semibold text-blue-300"
>

📄 ${s.filename}
(Page ${s.page})

</div>

<p
class="text-slate-400 italic mt-1"
>

"${s.snippet}"

</p>

</div>

`
            ).join('');


        sourcesHtml = `

<details
class="mt-2 pt-2 border-t border-slate-800 w-full"
>

<summary
class="cursor-pointer font-semibold text-blue-400"
>

📚 View ${sources.length}
Referenced Chunks

</summary>

<div class="mt-2">

${items}

</div>

</details>

`;

    }


    msgEl.innerHTML = `

<div
class="text-[10px] font-semibold text-slate-400 px-1"
>

${isUser ? 'You' : 'My Documents AI'}

</div>


<div
class="p-4 rounded-2xl max-w-xl ${
    isUser
    ? 'bg-blue-600 text-white'
    : 'bg-slate-900 text-slate-200 border border-slate-800'
} markdown-body shadow-md"
>

${
    isUser
    ? text
    : marked.parse(text || '')
}

${sourcesHtml}

</div>

`;


    chatMessages.appendChild(
        msgEl
    );

    chatMessages.scrollTop =
        chatMessages.scrollHeight;

}


fetchDocuments();

</script>

</body>

</html>
"""


# -----------------------------------------------------------------------------
# 11. HOME PAGE
# -----------------------------------------------------------------------------

@app.route("/")
def index():

    return render_template_string(
        UI_TEMPLATE
    )


# -----------------------------------------------------------------------------
# 12. UPLOAD API
# -----------------------------------------------------------------------------

@app.route(
    "/api/upload",
    methods=["POST"]
)
def api_upload():

    files = request.files.getlist(
        "files"
    )

    if not files:

        return jsonify({
            "error": "No file attached"
        }), 400


    total_chunks = 0

    try:

        for f in files:

            if not f.filename:
                continue

            save_path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                f.filename
            )

            f.save(
                save_path
            )

            chunks_added = (
                add_document_to_faiss(
                    save_path,
                    f.filename
                )
            )

            total_chunks += chunks_added


        return jsonify({
            "status": "success",
            "total_chunks": total_chunks,
            "message":
                f"Embedded {total_chunks} chunks into FAISS vector database."
        })


    except Exception as e:

        print(
            "Upload error:",
            e
        )

        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


# -----------------------------------------------------------------------------
# 13. DOCUMENTS API
# -----------------------------------------------------------------------------

@app.route(
    "/api/documents",
    methods=["GET"]
)
def api_documents():

    return jsonify({
        "documents":
            uploaded_documents
    })


# -----------------------------------------------------------------------------
# 14. CLEAR ALL
# -----------------------------------------------------------------------------

@app.route(
    "/api/clear",
    methods=["POST"]
)
def api_clear():

    global faiss_index
    global chunks_registry
    global uploaded_documents

    chunks_registry = []

    uploaded_documents = []

    faiss_index = faiss.IndexFlatL2(
        EMBEDDING_DIM
    )


    if os.path.exists(
        app.config["INDEX_FOLDER"]
    ):

        shutil.rmtree(
            app.config["INDEX_FOLDER"]
        )


    os.makedirs(
        app.config["INDEX_FOLDER"],
        exist_ok=True
    )


    if os.path.exists(
        app.config["UPLOAD_FOLDER"]
    ):

        shutil.rmtree(
            app.config["UPLOAD_FOLDER"]
        )


    os.makedirs(
        app.config["UPLOAD_FOLDER"],
        exist_ok=True
    )


    save_faiss_index()


    return jsonify({
        "status": "success",
        "message":
            "All documents and FAISS index cleared."
    })


# -----------------------------------------------------------------------------
# 15. DELETE SINGLE DOCUMENT
# -----------------------------------------------------------------------------

@app.route(
    "/api/delete",
    methods=["POST"]
)
def api_delete_doc():

    global faiss_index
    global chunks_registry
    global uploaded_documents

    data = request.get_json() or {}

    filename = (
        data.get(
            "filename",
            ""
        ).strip()
    )


    if not filename:

        return jsonify({
            "error":
                "Filename required"
        }), 400


    chunks_registry = [
        c
        for c in chunks_registry
        if c["filename"] != filename
    ]


    uploaded_documents = [
        d
        for d in uploaded_documents
        if d["filename"] != filename
    ]


    # -------------------------------------------------------------------------
    # Rebuild FAISS index
    # -------------------------------------------------------------------------

    faiss_index = faiss.IndexFlatL2(
        EMBEDDING_DIM
    )


    if chunks_registry:

        texts = [
            c["text"]
            for c in chunks_registry
        ]

        embeddings = create_embeddings(
            texts
        )

        faiss_index.add(
            embeddings
        )


    file_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )


    if os.path.exists(
        file_path
    ):

        try:

            os.remove(
                file_path
            )

        except Exception:
            pass


    save_faiss_index()


    return jsonify({
        "status": "success",
        "message":
            f"Document {filename} removed from index."
    })


# -----------------------------------------------------------------------------
# 16. QUERY API
# -----------------------------------------------------------------------------

@app.route(
    "/api/query",
    methods=["POST"]
)
def api_query():

    data = request.get_json() or {}

    query_text = (
        data.get(
            "query",
            ""
        ).strip()
    )


    if not query_text:

        return jsonify({
            "error":
                "Question cannot be empty"
        }), 400


    # -------------------------------------------------------------------------
    # 1. Search FAISS
    # -------------------------------------------------------------------------

    try:

        retrieved_chunks = search_faiss(
            query_text,
            top_k=3
        )

    except Exception as e:

        print(
            "Search error:",
            e
        )

        return jsonify({
            "error":
                str(e)
        }), 500


    if not retrieved_chunks:

        return jsonify({
            "answer":
                "No documents found in the knowledge base. Please upload documents first.",
            "sources": []
        })


    # -------------------------------------------------------------------------
    # 2. Build context
    # -------------------------------------------------------------------------

    context_parts = []

    for c in retrieved_chunks:

        context_parts.append(
            f"[Document: {c['filename']}, "
            f"Page: {c['page']}]\n"
            f"{c['text']}"
        )


    context_str = "\n\n".join(
        context_parts
    )


    # -------------------------------------------------------------------------
    # 3. Call LLM
    # -------------------------------------------------------------------------

    answer = call_llm(
        query_text,
        context_str,
        provider="gemini"
    )


    # -------------------------------------------------------------------------
    # 4. Fallback
    # -------------------------------------------------------------------------

    if not answer:

        answer = (
            f'Based on your documents, '
            f'here are the most relevant findings '
            f'for **"{query_text}"**:\n\n'
        )


        for c in retrieved_chunks:

            answer += (
                f"- **From {c['filename']} "
                f"(Page {c['page']}):**\n"
                f'  > "{c["text"]}"\n\n'
            )


        answer += (
            "\n---\n"
            "Configure OPENAI_API_KEY to get "
            "natural conversational responses."
        )


    return jsonify({
        "answer": answer,
        "sources": retrieved_chunks
    })


# -----------------------------------------------------------------------------
# 17. RUN APPLICATION
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    print("\n" + "=" * 70)

    print(
        ">> 'My Documents' Simple RAG Application is running!"
    )

    print(
        ">> Open in browser: http://127.0.0.1:5000"
    )

    print("=" * 70 + "\n")

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )