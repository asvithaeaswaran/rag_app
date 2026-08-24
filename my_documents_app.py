import os
import json
import shutil
import numpy as np
import requests
import pypdf
import docx
import faiss

from flask import Flask, request, jsonify, render_template_string
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "my_documents_files")
INDEX_FOLDER = os.path.join(BASE_DIR, "my_documents_faiss")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(INDEX_FOLDER, exist_ok=True)

FAISS_INDEX_PATH = os.path.join(INDEX_FOLDER, "my_documents.index")
METADATA_PATH = os.path.join(INDEX_FOLDER, "metadata.json")

# OpenAI text-embedding-3-small = 1536 dimensions
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

chunks_registry = []
uploaded_documents = []


# ============================================================
# OPENAI EMBEDDINGS
# ============================================================

def get_openai_key():
    key = os.environ.get("OPENAI_API_KEY")

    if not key:
        return None

    return key.strip()


def create_embeddings(texts):
    """
    Create embeddings using OpenAI API.
    No sentence-transformers is used.
    """

    api_key = get_openai_key()

    if not api_key:
        raise Exception("OPENAI_API_KEY is not configured in Render Environment Variables.")

    url = "https://api.openai.com/v1/embeddings"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": EMBEDDING_MODEL,
        "input": texts
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=120
    )

    if response.status_code != 200:
        raise Exception(
            f"OpenAI Embeddings API error {response.status_code}: {response.text}"
        )

    data = response.json()

    embeddings = [
        item["embedding"]
        for item in data["data"]
    ]

    return np.array(embeddings, dtype=np.float32)


# ============================================================
# FAISS
# ============================================================

def create_empty_index():
    return faiss.IndexFlatL2(EMBEDDING_DIM)


def load_faiss_index():

    global chunks_registry
    global uploaded_documents

    if (
        os.path.exists(FAISS_INDEX_PATH)
        and os.path.exists(METADATA_PATH)
    ):
        try:
            index = faiss.read_index(FAISS_INDEX_PATH)

            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)

            chunks_registry = data.get("chunks", [])
            uploaded_documents = data.get("documents", [])

            print(
                f"[My Documents] Loaded FAISS index with "
                f"{index.ntotal} vectors."
            )

            return index

        except Exception as e:
            print("Could not load existing FAISS index:", e)

    print("[My Documents] Creating new FAISS index.")

    return create_empty_index()


faiss_index = load_faiss_index()


def save_faiss_index():

    faiss.write_index(
        faiss_index,
        FAISS_INDEX_PATH
    )

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "chunks": chunks_registry,
                "documents": uploaded_documents
            },
            f,
            indent=2
        )


# ============================================================
# DOCUMENT EXTRACTION
# ============================================================

def extract_text(file_path, filename):

    extension = ""

    if "." in filename:
        extension = filename.rsplit(".", 1)[-1].lower()

    pages = []

    # ---------------- PDF ----------------

    if extension == "pdf":

        try:

            reader = pypdf.PdfReader(file_path)

            for page_number, page in enumerate(
                reader.pages,
                start=1
            ):

                text = page.extract_text() or ""
                text = text.strip()

                if text:
                    pages.append(
                        {
                            "text": text,
                            "page": page_number
                        }
                    )

        except Exception as e:

            raise Exception(
                f"Could not read PDF: {str(e)}"
            )

    # ---------------- DOCX ----------------

    elif extension == "docx":

        try:

            document = docx.Document(file_path)

            paragraphs = []

            for paragraph in document.paragraphs:

                text = paragraph.text.strip()

                if text:
                    paragraphs.append(text)

            for table in document.tables:

                for row in table.rows:

                    cells = []

                    for cell in row.cells:

                        cell_text = cell.text.strip()

                        if cell_text:
                            cells.append(cell_text)

                    if cells:
                        paragraphs.append(
                            " | ".join(cells)
                        )

            full_text = "\n\n".join(paragraphs)

            if full_text.strip():

                pages.append(
                    {
                        "text": full_text,
                        "page": 1
                    }
                )

        except Exception as e:

            raise Exception(
                f"Could not read DOCX: {str(e)}"
            )

    # ---------------- TXT / CSV / MD ----------------

    else:

        try:

            with open(
                file_path,
                "r",
                encoding="utf-8",
                errors="replace"
            ) as f:

                text = f.read().strip()

            if text:

                pages.append(
                    {
                        "text": text,
                        "page": 1
                    }
                )

        except Exception as e:

            raise Exception(
                f"Could not read text file: {str(e)}"
            )

    return pages


# ============================================================
# TEXT CHUNKING
# ============================================================

def chunk_text(
    text,
    chunk_size=800,
    overlap=100
):

    text = text.strip()

    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks = []

    start = 0

    while start < len(text):

        end = min(
            start + chunk_size,
            len(text)
        )

        if end < len(text):

            break_position = text.rfind(
                "\n",
                start,
                end
            )

            if (
                break_position == -1
                or break_position < start + chunk_size // 2
            ):

                break_position = text.rfind(
                    ". ",
                    start,
                    end
                )

            if (
                break_position == -1
                or break_position < start + chunk_size // 2
            ):

                break_position = text.rfind(
                    " ",
                    start,
                    end
                )

            if break_position > start:
                end = break_position + 1

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(
            0,
            end - overlap
        )

    return chunks


# ============================================================
# ADD DOCUMENT
# ============================================================

def add_document_to_faiss(
    file_path,
    filename
):

    global faiss_index
    global chunks_registry
    global uploaded_documents

    pages = extract_text(
        file_path,
        filename
    )

    if not pages:
        raise Exception(
            "No readable text was found in the document."
        )

    new_chunks = []

    for page in pages:

        pieces = chunk_text(
            page["text"],
            chunk_size=800,
            overlap=100
        )

        for chunk_number, text in enumerate(
            pieces
        ):

            new_chunks.append(
                {
                    "filename": filename,
                    "page": page["page"],
                    "chunk_index": chunk_number,
                    "text": text
                }
            )

    if not new_chunks:
        raise Exception(
            "No text chunks were created."
        )

    texts = [
        item["text"]
        for item in new_chunks
    ]

    print(
        f"[My Documents] Creating embeddings "
        f"for {len(texts)} chunks..."
    )

    embeddings = create_embeddings(texts)

    # If same document already exists,
    # rebuild index after removing old version.
    chunks_registry = [
        item
        for item in chunks_registry
        if item["filename"] != filename
    ]

    uploaded_documents = [
        item
        for item in uploaded_documents
        if item["filename"] != filename
    ]

    chunks_registry.extend(new_chunks)

    uploaded_documents.append(
        {
            "filename": filename,
            "chunks_count": len(new_chunks),
            "file_size": os.path.getsize(file_path)
        }
    )

    rebuild_faiss_index()

    save_faiss_index()

    return len(new_chunks)


# ============================================================
# REBUILD FAISS INDEX
# ============================================================

def rebuild_faiss_index():

    global faiss_index

    faiss_index = create_empty_index()

    if not chunks_registry:
        return

    texts = [
        item["text"]
        for item in chunks_registry
    ]

    embeddings = create_embeddings(texts)

    faiss_index.add(embeddings)

    print(
        f"[My Documents] FAISS rebuilt with "
        f"{faiss_index.ntotal} vectors."
    )


# ============================================================
# SEARCH
# ============================================================

def search_faiss(
    query,
    top_k=3
):

    if (
        faiss_index.ntotal == 0
        or not chunks_registry
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

    for rank, (index, distance) in enumerate(
        zip(indices[0], distances[0]),
        start=1
    ):

        if index < 0:
            continue

        if index >= len(chunks_registry):
            continue

        chunk = chunks_registry[index]

        results.append(
            {
                "rank": rank,
                "filename": chunk["filename"],
                "page": chunk["page"],
                "text": chunk["text"],
                "score": round(
                    float(distance),
                    4
                ),
                "snippet": (
                    chunk["text"][:300]
                    + (
                        "..."
                        if len(chunk["text"]) > 300
                        else ""
                    )
                )
            }
        )

    return results


# ============================================================
# LLM
# ============================================================

def call_llm(
    question,
    context,
    provider="gemini",
    api_key=None
):

    system_prompt = (
        "You are My Documents AI. "
        "Answer the user's question using only the "
        "provided document excerpts. "
        "If the answer is not present in the documents, "
        "say that the uploaded documents do not contain "
        "that information. "
        "Give a clear and concise answer."
    )

    user_prompt = (
        "DOCUMENT EXCERPTS:\n\n"
        + context
        + "\n\nUSER QUESTION:\n"
        + question
    )

    # ========================================================
    # GEMINI
    # ========================================================

    if provider == "gemini":

        gemini_key = (
            api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )

        if gemini_key:

            try:

                url = (
                    "https://generativelanguage.googleapis.com/"
                    "v1beta/models/gemini-2.0-flash:"
                    "generateContent"
                )

                url += "?key=" + gemini_key.strip()

                payload = {
                    "contents": [
                        {
                            "parts": [
                                {
                                    "text":
                                    system_prompt
                                    + "\n\n"
                                    + user_prompt
                                }
                            ]
                        }
                    ],
                    "generationConfig": {
                        "temperature": 0.3,
                        "maxOutputTokens": 2048
                    }
                }

                response = requests.post(
                    url,
                    json=payload,
                    timeout=60
                )

                if response.status_code == 200:

                    data = response.json()

                    candidates = data.get(
                        "candidates",
                        []
                    )

                    if candidates:

                        parts = (
                            candidates[0]
                            .get("content", {})
                            .get("parts", [])
                        )

                        if parts:

                            return parts[0].get(
                                "text",
                                ""
                            )

                print(
                    "Gemini error:",
                    response.status_code,
                    response.text
                )

            except Exception as e:

                print(
                    "Gemini request error:",
                    e
                )

    # ========================================================
    # OPENAI
    # ========================================================

    if provider == "openai":

        openai_key = (
            api_key
            or os.environ.get("OPENAI_API_KEY")
        )

        if openai_key:

            try:

                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization":
                        "Bearer " + openai_key.strip(),
                        "Content-Type":
                        "application/json"
                    },
                    json={
                        "model":
                        os.environ.get(
                            "OPENAI_MODEL",
                            "gpt-4o-mini"
                        ),
                        "messages": [
                            {
                                "role": "system",
                                "content":
                                system_prompt
                            },
                            {
                                "role": "user",
                                "content":
                                user_prompt
                            }
                        ],
                        "temperature": 0.3
                    },
                    timeout=60
                )

                if response.status_code == 200:

                    data = response.json()

                    return (
                        data["choices"][0]
                        ["message"]["content"]
                    )

                print(
                    "OpenAI error:",
                    response.status_code,
                    response.text
                )

            except Exception as e:

                print(
                    "OpenAI request error:",
                    e
                )

    return None


# ============================================================
# HTML
# ============================================================

HTML = """
<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<meta name="viewport"
content="width=device-width, initial-scale=1.0">

<title>My Documents AI</title>

<script src="https://cdn.tailwindcss.com"></script>

<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>

<style>

body {
    font-family: Arial, sans-serif;
}

::-webkit-scrollbar {
    width: 6px;
}

::-webkit-scrollbar-thumb {
    background: #475569;
    border-radius: 10px;
}

</style>

</head>

<body class="bg-slate-950 text-white min-h-screen">

<div class="max-w-7xl mx-auto p-5">

<div class="mb-5">

<h1 class="text-2xl font-bold">
My Documents AI
</h1>

<p class="text-slate-400 text-sm">
Simple RAG Q&A using OpenAI Embeddings and FAISS
</p>

</div>

<div class="grid grid-cols-1 md:grid-cols-3 gap-5">

<!-- DOCUMENTS -->

<div class="bg-slate-900 border border-slate-800 rounded-xl p-5">

<h2 class="font-bold mb-4">
Uploaded Documents
</h2>

<div
id="dropzone"
class="border-2 border-dashed border-slate-700 rounded-xl p-8 text-center cursor-pointer hover:border-blue-500"
>

<input
type="file"
id="fileInput"
multiple
accept=".pdf,.docx,.txt,.csv,.md"
style="display:none;"
>

<div class="text-3xl mb-2">
📁
</div>

<div class="font-semibold">
Upload Documents
</div>

<div class="text-xs text-slate-400 mt-2">
PDF, DOCX, TXT, CSV or MD
</div>

</div>

<div
id="uploadingBox"
class="hidden mt-3 p-3 bg-blue-950 rounded-lg text-sm"
>
Uploading and creating embeddings...
</div>

<div class="flex justify-between mt-5 mb-2">

<span class="text-sm text-slate-400">
Documents
</span>

<button
id="clearAllBtn"
class="text-xs text-red-400"
>
Clear All
</button>

</div>

<div
id="documentsList"
class="space-y-2"
>
</div>

</div>


<!-- CHAT -->

<div class="md:col-span-2 bg-slate-900 border border-slate-800 rounded-xl p-5 flex flex-col min-h-[700px]">

<div
id="chatMessages"
class="flex-1 overflow-y-auto space-y-4 mb-4"
>

<div
id="welcome"
class="text-center text-slate-400 py-20"
>

<div class="text-4xl mb-3">
🤖
</div>

<h2 class="text-white font-bold text-lg">
Ask questions about your documents
</h2>

<p class="text-sm mt-2">
Upload a document and then ask a question.
</p>

</div>

</div>


<div class="flex gap-2">

<select
id="providerSelect"
class="bg-slate-950 border border-slate-700 rounded-lg px-3 text-sm"
>

<option value="gemini">
Gemini
</option>

<option value="openai">
OpenAI
</option>

</select>

<input
id="questionInput"
class="flex-1 bg-slate-950 border border-slate-700 rounded-lg px-4 py-3 text-sm"
placeholder="Ask a question..."
>

<button
id="askBtn"
class="bg-blue-600 hover:bg-blue-500 px-5 rounded-lg"
>
Ask
</button>

</div>

</div>

</div>

</div>


<script>

const dropzone =
document.getElementById("dropzone");

const fileInput =
document.getElementById("fileInput");

const uploadingBox =
document.getElementById("uploadingBox");

const documentsList =
document.getElementById("documentsList");

const clearAllBtn =
document.getElementById("clearAllBtn");

const chatMessages =
document.getElementById("chatMessages");

const welcome =
document.getElementById("welcome");

const questionInput =
document.getElementById("questionInput");

const askBtn =
document.getElementById("askBtn");

const providerSelect =
document.getElementById("providerSelect");


/* ==========================================================
   FILE SELECTION
   ========================================================== */

dropzone.onclick = function () {

    fileInput.click();

};


fileInput.onchange = function (event) {

    uploadFiles(event.target.files);

};


dropzone.ondragover = function (event) {

    event.preventDefault();

    dropzone.classList.add(
        "border-blue-500"
    );

};


dropzone.ondragleave = function () {

    dropzone.classList.remove(
        "border-blue-500"
    );

};


dropzone.ondrop = function (event) {

    event.preventDefault();

    dropzone.classList.remove(
        "border-blue-500"
    );

    uploadFiles(
        event.dataTransfer.files
    );

};


/* ==========================================================
   UPLOAD
   ========================================================== */

async function uploadFiles(files) {

    if (!files || files.length === 0) {
        return;
    }

    uploadingBox.classList.remove(
        "hidden"
    );

    const formData =
        new FormData();

    for (
        let i = 0;
        i < files.length;
        i++
    ) {

        formData.append(
            "files",
            files[i]
        );

    }

    try {

        const response =
            await fetch(
                "/api/upload",
                {
                    method: "POST",
                    body: formData
                }
            );

        const data =
            await response.json();

        if (!response.ok) {

            alert(
                "Upload failed: "
                + (
                    data.error
                    || "Unknown server error"
                )
            );

            return;
        }

        if (
            data.status === "success"
        ) {

            alert(
                data.message
            );

            fetchDocuments();

        } else {

            alert(
                "Upload failed: "
                + data.error
            );

        }

    } catch (error) {

        alert(
            "Upload error: "
            + error.message
        );

    } finally {

        uploadingBox.classList.add(
            "hidden"
        );

        fileInput.value = "";

    }

}


/* ==========================================================
   DOCUMENT LIST
   ========================================================== */

async function fetchDocuments() {

    try {

        const response =
            await fetch(
                "/api/documents"
            );

        const data =
            await response.json();

        renderDocuments(
            data.documents || []
        );

    } catch (error) {

        console.error(error);

    }

}


function renderDocuments(docs) {

    documentsList.innerHTML = "";

    if (docs.length === 0) {

        documentsList.innerHTML =
            '<div class="text-sm text-slate-500">' +
            'No documents uploaded yet.' +
            '</div>';

        return;

    }

    docs.forEach(function(doc) {

        const div =
            document.createElement("div");

        div.className =
            "bg-slate-950 border border-slate-800 rounded-lg p-3 flex justify-between items-center";

        div.innerHTML =

            '<div>' +

            '<div class="text-sm font-semibold">' +
            escapeHtml(doc.filename) +
            '</div>' +

            '<div class="text-xs text-slate-500">' +
            doc.chunks_count +
            ' chunks' +
            '</div>' +

            '</div>' +

            '<button class="deleteBtn text-red-400 text-xs">' +
            'Delete' +
            '</button>';

        div
            .querySelector(".deleteBtn")
            .onclick = async function() {

                await fetch(
                    "/api/delete",
                    {
                        method: "POST",
                        headers: {
                            "Content-Type":
                            "application/json"
                        },
                        body: JSON.stringify({
                            filename:
                            doc.filename
                        })
                    }
                );

                fetchDocuments();

            };

        documentsList.appendChild(
            div
        );

    });

}


function escapeHtml(text) {

    const div =
        document.createElement("div");

    div.textContent = text;

    return div.innerHTML;

}


/* ==========================================================
   CLEAR ALL
   ========================================================== */

clearAllBtn.onclick =
async function() {

    if (
        !confirm(
            "Delete all documents?"
        )
    ) {
        return;
    }

    await fetch(
        "/api/clear",
        {
            method: "POST"
        }
    );

    fetchDocuments();

    chatMessages.innerHTML = "";

};


/* ==========================================================
   ASK QUESTION
   ========================================================== */

askBtn.onclick =
async function() {

    const question =
        questionInput.value.trim();

    if (!question) {
        return;
    }

    welcome.classList.add(
        "hidden"
    );

    addMessage(
        "You",
        question,
        true
    );

    questionInput.value = "";

    askBtn.disabled = true;

    try {

        const response =
            await fetch(
                "/api/query",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                        "application/json"
                    },

                    body: JSON.stringify({

                        query:
                        question,

                        provider:
                        providerSelect.value

                    })
                }
            );

        const data =
            await response.json();

        if (!response.ok) {

            addMessage(
                "My Documents AI",
                "Error: "
                + (
                    data.error
                    || "Unknown error"
                ),
                false
            );

            return;
        }

        addMessage(
            "My Documents AI",
            data.answer,
            false
        );

    } catch (error) {

        addMessage(
            "My Documents AI",
            "Error: "
            + error.message,
            false
        );

    } finally {

        askBtn.disabled = false;

    }

};


questionInput.addEventListener(
    "keydown",
    function(event) {

        if (
            event.key === "Enter"
        ) {

            askBtn.click();

        }

    }
);


function addMessage(
    name,
    text,
    user
) {

    const div =
        document.createElement("div");

    div.className =
        user
        ? "flex justify-end"
        : "flex justify-start";

    const bubble =
        document.createElement("div");

    bubble.className =
        user
        ? "bg-blue-600 rounded-xl p-4 max-w-xl text-sm"
        : "bg-slate-950 border border-slate-800 rounded-xl p-4 max-w-xl text-sm";

    const title =
        document.createElement("div");

    title.className =
        "text-xs font-bold mb-2";

    title.textContent = name;

    const content =
        document.createElement("div");

    if (
        !user &&
        typeof marked !== "undefined"
    ) {

        content.innerHTML =
            marked.parse(text);

    } else {

        content.textContent = text;

    }

    bubble.appendChild(title);

    bubble.appendChild(content);

    div.appendChild(bubble);

    chatMessages.appendChild(div);

    chatMessages.scrollTop =
        chatMessages.scrollHeight;

}


fetchDocuments();

</script>

</body>

</html>
"""


# ============================================================
# FLASK ROUTES
# ============================================================

@app.route("/")
def index():

    return render_template_string(
        HTML
    )


@app.route(
    "/api/upload",
    methods=["POST"]
)
def api_upload():

    try:

        files = request.files.getlist(
            "files"
        )

        if not files:

            return jsonify(
                {
                    "error":
                    "No file attached."
                }
            ), 400

        total_chunks = 0

        for file in files:

            if not file.filename:
                continue

            filename = os.path.basename(
                file.filename
            )

            allowed_extensions = {
                "pdf",
                "docx",
                "txt",
                "csv",
                "md"
            }

            extension = ""

            if "." in filename:

                extension = filename.rsplit(
                    ".",
                    1
                )[-1].lower()

            if extension not in allowed_extensions:

                return jsonify(
                    {
                        "error":
                        "Unsupported file type."
                    }
                ), 400

            save_path = os.path.join(
                UPLOAD_FOLDER,
                filename
            )

            file.save(save_path)

            chunks_added = (
                add_document_to_faiss(
                    save_path,
                    filename
                )
            )

            total_chunks += chunks_added

        return jsonify(
            {
                "status":
                "success",

                "total_chunks":
                total_chunks,

                "message":
                f"Successfully uploaded and indexed "
                f"{total_chunks} document chunks."
            }
        )

    except Exception as e:

        print(
            "Upload error:",
            str(e)
        )

        return jsonify(
            {
                "error":
                str(e)
            }
        ), 500


@app.route(
    "/api/documents",
    methods=["GET"]
)
def api_documents():

    return jsonify(
        {
            "documents":
            uploaded_documents
        }
    )


@app.route(
    "/api/delete",
    methods=["POST"]
)
def api_delete():

    global chunks_registry
    global uploaded_documents
    global faiss_index

    try:

        data = request.get_json() or {}

        filename = (
            data.get("filename", "")
            .strip()
        )

        if not filename:

            return jsonify(
                {
                    "error":
                    "Filename is required."
                }
            ), 400

        chunks_registry = [
            chunk
            for chunk in chunks_registry
            if chunk["filename"] != filename
        ]

        uploaded_documents = [
            document
            for document in uploaded_documents
            if document["filename"] != filename
        ]

        rebuild_faiss_index()

        save_faiss_index()

        file_path = os.path.join(
            UPLOAD_FOLDER,
            filename
        )

        if os.path.exists(file_path):

            os.remove(file_path)

        return jsonify(
            {
                "status":
                "success",

                "message":
                "Document deleted."
            }
        )

    except Exception as e:

        print(
            "Delete error:",
            str(e)
        )

        return jsonify(
            {
                "error":
                str(e)
            }
        ), 500


@app.route(
    "/api/clear",
    methods=["POST"]
)
def api_clear():

    global chunks_registry
    global uploaded_documents
    global faiss_index

    chunks_registry = []

    uploaded_documents = []

    faiss_index = create_empty_index()

    try:

        if os.path.exists(
            INDEX_FOLDER
        ):

            shutil.rmtree(
                INDEX_FOLDER
            )

        os.makedirs(
            INDEX_FOLDER,
            exist_ok=True
        )

        if os.path.exists(
            UPLOAD_FOLDER
        ):

            shutil.rmtree(
                UPLOAD_FOLDER
            )

        os.makedirs(
            UPLOAD_FOLDER,
            exist_ok=True
        )

        save_faiss_index()

    except Exception as e:

        print(
            "Clear error:",
            str(e)
        )

        return jsonify(
            {
                "error":
                str(e)
            }
        ), 500

    return jsonify(
        {
            "status":
            "success",

            "message":
            "All documents cleared."
        }
    )


@app.route(
    "/api/query",
    methods=["POST"]
)
def api_query():

    try:

        data = request.get_json() or {}

        question = (
            data.get("query", "")
            .strip()
        )

        provider = (
            data.get(
                "provider",
                "gemini"
            )
            or "gemini"
        )

        if not question:

            return jsonify(
                {
                    "error":
                    "Question cannot be empty."
                }
            ), 400

        if not chunks_registry:

            return jsonify(
                {
                    "answer":
                    "Please upload a document first.",

                    "sources":
                    []
                }
            )

        retrieved_chunks = search_faiss(
            question,
            top_k=3
        )

        if not retrieved_chunks:

            return jsonify(
                {
                    "answer":
                    "No relevant information was found "
                    "in the uploaded documents.",

                    "sources":
                    []
                }
            )

        context_parts = []

        for chunk in retrieved_chunks:

            context_parts.append(
                "[Document: "
                + chunk["filename"]
                + ", Page: "
                + str(chunk["page"])
                + "]\n"
                + chunk["text"]
            )

        context = "\n\n".join(
            context_parts
        )

        answer = call_llm(
            question,
            context,
            provider=provider
        )

        if not answer:

            answer = (
                "I found these relevant sections "
                "in your documents:\n\n"
            )

            for chunk in retrieved_chunks:

                answer += (
                    "**"
                    + chunk["filename"]
                    + " - Page "
                    + str(chunk["page"])
                    + "**\n\n"
                    + chunk["text"]
                    + "\n\n"
                )

        return jsonify(
            {
                "answer":
                answer,

                "sources":
                retrieved_chunks
            }
        )

    except Exception as e:

        print(
            "Query error:",
            str(e)
        )

        return jsonify(
            {
                "error":
                str(e)
            }
        ), 500


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    print(
        "My Documents AI is running..."
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )