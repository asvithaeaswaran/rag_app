import os
import json
import shutil
import numpy as np
import requests
import faiss
import pypdf
import docx

from flask import Flask, request, jsonify, render_template_string
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "my_documents_files")
INDEX_FOLDER = os.path.join(BASE_DIR, "my_documents_faiss")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(INDEX_FOLDER, exist_ok=True)

# ============================================================
# OPENAI EMBEDDINGS
# ============================================================

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536


def get_openai_key():
    key = os.environ.get("OPENAI_API_KEY")

    if not key:
        return None

    return key.strip()


def create_embeddings(texts):
    """
    Create embeddings using OpenAI text-embedding-3-small.
    """

    api_key = get_openai_key()

    if not api_key:
        raise Exception(
            "OPENAI_API_KEY is not configured in Render Environment Variables."
        )

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
        timeout=60
    )

    if response.status_code != 200:
        raise Exception(
            f"OpenAI Embeddings Error: {response.status_code} - {response.text}"
        )

    data = response.json()

    embeddings = [
        item["embedding"]
        for item in data["data"]
    ]

    return np.array(embeddings, dtype=np.float32)


# ============================================================
# FILE PATHS
# ============================================================

FAISS_INDEX_PATH = os.path.join(
    INDEX_FOLDER,
    "my_documents.index"
)

METADATA_PATH = os.path.join(
    INDEX_FOLDER,
    "metadata.json"
)


# ============================================================
# MEMORY REGISTRY
# ============================================================

chunks_registry = []
uploaded_documents = []


# ============================================================
# DOCUMENT EXTRACTION
# ============================================================

def extract_text(file_path, filename):

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

                    pages.append({
                        "text": text,
                        "page": page_number
                    })

        except Exception as e:

            print(
                f"PDF extraction error: {e}"
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

                    row_text = []

                    for cell in row.cells:

                        cell_text = cell.text.strip()

                        if cell_text:

                            row_text.append(cell_text)

                    if row_text:

                        paragraphs.append(
                            " | ".join(row_text)
                        )

            full_text = "\n\n".join(paragraphs)

            if full_text:

                pages.append({
                    "text": full_text,
                    "page": 1
                })

        except Exception as e:

            print(
                f"DOCX extraction error: {e}"
            )

    # ---------------- TXT / CSV / MD ----------------

    else:

        try:

            with open(
                file_path,
                "r",
                encoding="utf-8",
                errors="replace"
            ) as file:

                text = file.read().strip()

            if text:

                pages.append({
                    "text": text,
                    "page": 1
                })

        except Exception as e:

            print(
                f"Text extraction error: {e}"
            )

    return pages


# ============================================================
# TEXT CHUNKING
# ============================================================

def chunk_text(
    text,
    chunk_size=500,
    overlap=80
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
# FAISS
# ============================================================

def create_empty_index():

    return faiss.IndexFlatL2(
        EMBEDDING_DIM
    )


def load_faiss():

    global chunks_registry
    global uploaded_documents

    if (
        os.path.exists(FAISS_INDEX_PATH)
        and os.path.exists(METADATA_PATH)
    ):

        try:

            index = faiss.read_index(
                FAISS_INDEX_PATH
            )

            with open(
                METADATA_PATH,
                "r",
                encoding="utf-8"
            ) as file:

                data = json.load(file)

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
                f"Could not load FAISS index: {e}"
            )

    return create_empty_index()


faiss_index = load_faiss()


def save_faiss():

    faiss.write_index(
        faiss_index,
        FAISS_INDEX_PATH
    )

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            {
                "chunks": chunks_registry,
                "documents": uploaded_documents
            },
            file,
            indent=2
        )


# ============================================================
# ADD DOCUMENT
# ============================================================

def add_document(
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
            "No readable text found in the document."
        )

    new_chunks = []

    for page in pages:

        pieces = chunk_text(
            page["text"],
            500,
            80
        )

        for index, piece in enumerate(pieces):

            new_chunks.append(
                {
                    "filename": filename,
                    "page": page["page"],
                    "chunk_index": index,
                    "text": piece
                }
            )

    if not new_chunks:

        raise Exception(
            "No text chunks were created."
        )

    texts = [
        chunk["text"]
        for chunk in new_chunks
    ]

    print(
        f"[My Documents] Creating {len(texts)} embeddings..."
    )

    embeddings = create_embeddings(
        texts
    )

    faiss_index.add(
        embeddings
    )

    chunks_registry.extend(
        new_chunks
    )

    uploaded_documents = [
        document
        for document in uploaded_documents
        if document["filename"] != filename
    ]

    uploaded_documents.append(
        {
            "filename": filename,
            "chunks_count": len(new_chunks),
            "file_size": os.path.getsize(file_path)
        }
    )

    save_faiss()

    return len(new_chunks)


# ============================================================
# SEARCH
# ============================================================

def search_documents(
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
        zip(
            indices[0],
            distances[0]
        ),
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


# ============================================================
# OPENAI CHAT
# ============================================================

def call_openai(
    question,
    context
):

    api_key = get_openai_key()

    if not api_key:

        raise Exception(
            "OPENAI_API_KEY is not configured."
        )

    model = os.environ.get(
        "OPENAI_MODEL",
        "gpt-4o-mini"
    )

    url = (
        "https://api.openai.com/v1/chat/completions"
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    system_message = """
You are My Documents AI.

Answer the user's question using ONLY the
information contained in the provided document excerpts.

If the answer is not present in the documents,
say clearly that the uploaded documents do not
contain that information.

Do not invent facts.

Give a clear and useful answer.
"""

    user_message = f"""
DOCUMENT EXCERPTS:

{context}

USER QUESTION:

{question}

Answer based only on the document excerpts.
"""

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": user_message
            }
        ],
        "temperature": 0.2
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=60
    )

    if response.status_code != 200:

        raise Exception(
            f"OpenAI API Error: "
            f"{response.status_code} - "
            f"{response.text}"
        )

    data = response.json()

    return (
        data["choices"][0]
        ["message"]
        ["content"]
    )


# ============================================================
# HTML
# ============================================================

HTML = """

<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<meta
name="viewport"
content="width=device-width, initial-scale=1.0"
>

<title>My Documents - RAG Q&A</title>

<script src="https://cdn.tailwindcss.com"></script>

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

<header
class="bg-slate-900 border-b border-slate-800 p-5"
>

<div
class="max-w-7xl mx-auto flex justify-between items-center"
>

<div>

<h1
class="text-2xl font-bold"
>
My Documents
</h1>

<p
class="text-sm text-slate-400"
>
RAG Document Q&A
</p>

</div>

<div
class="text-xs text-green-400"
>
OpenAI + FAISS
</div>

</div>

</header>


<main
class="max-w-7xl mx-auto p-5 grid grid-cols-1 md:grid-cols-3 gap-5"
>


<!-- DOCUMENT PANEL -->

<section
class="bg-slate-900 border border-slate-800 rounded-xl p-5"
>

<h2
class="font-bold mb-4"
>
Uploaded Documents
</h2>


<div
id="dropzone"
class="border-2 border-dashed border-slate-700 rounded-xl p-8 text-center cursor-pointer hover:border-blue-500"
>

<input
id="fileInput"
type="file"
multiple
accept=".pdf,.docx,.txt,.csv,.md"
class="hidden"
>

<div
class="text-3xl mb-3"
>
📁
</div>

<p
class="font-semibold"
>
Upload Documents
</p>

<p
class="text-xs text-slate-400 mt-2"
>
PDF, DOCX, TXT, CSV, MD
</p>

</div>


<div
id="uploadStatus"
class="hidden mt-4 bg-blue-950 text-blue-300 p-3 rounded-lg text-sm"
>
Uploading and creating embeddings...
</div>


<div
class="flex justify-between items-center mt-5 mb-3"
>

<span
class="text-sm text-slate-400"
>
Knowledge Base
</span>

<button
id="clearBtn"
class="text-xs text-red-400"
>
Clear All
</button>

</div>


<div
id="documents"
class="space-y-2 max-h-[55vh] overflow-y-auto"
>

<p
class="text-sm text-slate-500"
>
No documents uploaded.
</p>

</div>

</section>


<!-- CHAT PANEL -->

<section
class="md:col-span-2 bg-slate-900 border border-slate-800 rounded-xl p-5 flex flex-col min-h-[75vh]"
>


<div
id="messages"
class="flex-1 overflow-y-auto space-y-4 mb-4"
>

<div
id="welcome"
class="text-center py-20"
>

<div
class="text-5xl mb-4"
>
🤖
</div>

<h2
class="text-xl font-bold"
>
Ask questions about your documents
</h2>

<p
class="text-sm text-slate-400 mt-2"
>
Upload a document and ask questions about it.
</p>

</div>

</div>


<div
id="loading"
class="hidden text-sm text-blue-400 mb-3"
>
Searching documents and generating answer...
</div>


<form
id="questionForm"
class="flex gap-2"
>

<input
id="question"
type="text"
placeholder="Ask a question..."
required
class="flex-1 bg-slate-950 border border-slate-700 rounded-xl px-4 py-3 outline-none focus:border-blue-500"
>

<button
class="bg-blue-600 hover:bg-blue-500 px-5 rounded-xl font-semibold"
>
Ask
</button>

</form>

</section>

</main>


<script>

const fileInput =
    document.getElementById("fileInput");

const dropzone =
    document.getElementById("dropzone");

const documents =
    document.getElementById("documents");

const uploadStatus =
    document.getElementById("uploadStatus");

const messages =
    document.getElementById("messages");

const welcome =
    document.getElementById("welcome");

const questionForm =
    document.getElementById("questionForm");

const questionInput =
    document.getElementById("question");

const loading =
    document.getElementById("loading");

const clearBtn =
    document.getElementById("clearBtn");


dropzone.addEventListener(
    "click",
    () => fileInput.click()
);


fileInput.addEventListener(
    "change",
    () => uploadFiles(fileInput.files)
);


dropzone.addEventListener(
    "dragover",
    (event) => {
        event.preventDefault();
        dropzone.classList.add(
            "border-blue-500"
        );
    }
);


dropzone.addEventListener(
    "dragleave",
    () => {
        dropzone.classList.remove(
            "border-blue-500"
        );
    }
);


dropzone.addEventListener(
    "drop",
    (event) => {

        event.preventDefault();

        dropzone.classList.remove(
            "border-blue-500"
        );

        uploadFiles(
            event.dataTransfer.files
        );

    }
);


async function uploadFiles(files) {

    if (!files || files.length === 0) {
        return;
    }

    uploadStatus.classList.remove(
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

        if (data.status === "success") {

            alert(
                "Document uploaded successfully."
            );

            loadDocuments();

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

        uploadStatus.classList.add(
            "hidden"
        );

        fileInput.value = "";

    }

}


async function loadDocuments() {

    try {

        const response =
            await fetch(
                "/api/documents"
            );

        const data =
            await response.json();

        documents.innerHTML = "";

        if (
            !data.documents
            || data.documents.length === 0
        ) {

            documents.innerHTML =
                '<p class="text-sm text-slate-500">No documents uploaded.</p>';

            return;

        }


        data.documents.forEach(
            document => {

                const item =
                    document.createElement
                    ? document.createElement("div")
                    : null;

            }
        );

        for (
            const doc of data.documents
        ) {

            const div =
                document.createElement(
                    "div"
                );

            div.className =
                "bg-slate-950 border border-slate-800 rounded-lg p-3 flex justify-between items-center";


            div.innerHTML = `

                <div>

                    <div class="font-semibold text-sm">
                        📄 ${escapeHtml(doc.filename)}
                    </div>

                    <div class="text-xs text-slate-500 mt-1">
                        ${doc.chunks_count} chunks
                    </div>

                </div>

                <button
                    class="text-red-400 text-xs"
                    onclick="deleteDocument('${encodeURIComponent(doc.filename)}')"
                >
                    Delete
                </button>

            `;

            documents.appendChild(div);

        }

    } catch (error) {

        console.error(error);

    }

}


function escapeHtml(text) {

    const div =
        document.createElement("div");

    div.textContent = text;

    return div.innerHTML;

}


async function deleteDocument(filename) {

    if (
        !confirm(
            "Delete this document?"
        )
    ) {
        return;
    }

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
                    decodeURIComponent(filename)
            })
        }
    );

    loadDocuments();

}


clearBtn.addEventListener(
    "click",
    async () => {

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

        loadDocuments();

        messages.innerHTML = "";

        messages.appendChild(
            welcome
        );

    }
);


questionForm.addEventListener(
    "submit",
    async (event) => {

        event.preventDefault();

        const question =
            questionInput.value.trim();

        if (!question) {
            return;
        }

        if (welcome) {

            welcome.classList.add(
                "hidden"
            );

        }

        addMessage(
            "You",
            question,
            "bg-blue-600"
        );

        questionInput.value = "";

        loading.classList.remove(
            "hidden"
        );

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
                            query: question
                        })
                    }
                );

            const data =
                await response.json();

            if (data.error) {

                addMessage(
                    "AI",
                    "Error: " + data.error,
                    "bg-red-900"
                );

            } else {

                let answer =
                    data.answer || "";

                if (
                    data.sources
                    && data.sources.length > 0
                ) {

                    answer +=
                        "\n\nSources:\n";

                    data.sources.forEach(
                        source => {

                            answer +=
                                "\n📄 "
                                + source.filename
                                + " - Page "
                                + source.page;

                        }
                    );

                }

                addMessage(
                    "AI",
                    answer,
                    "bg-slate-950"
                );

            }

        } catch (error) {

            addMessage(
                "AI",
                "Error: " + error.message,
                "bg-red-900"
            );

        } finally {

            loading.classList.add(
                "hidden"
            );

        }

    }
);


function addMessage(
    sender,
    text,
    background
) {

    const div =
        document.createElement(
            "div"
        );

    div.className =
        "flex flex-col gap-1";


    const name =
        document.createElement(
            "div"
        );

    name.className =
        "text-xs text-slate-500";

    name.textContent =
        sender;


    const bubble =
        document.createElement(
            "div"
        );

    bubble.className =
        background
        + " rounded-xl p-4 text-sm whitespace-pre-wrap";


    bubble.textContent =
        text;


    div.appendChild(name);

    div.appendChild(bubble);

    messages.appendChild(div);

    messages.scrollTop =
        messages.scrollHeight;

}


loadDocuments();

</script>

</body>

</html>

"""


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template_string(
        HTML
    )


# ============================================================
# UPLOAD API
# ============================================================

@app.route(
    "/api/upload",
    methods=["POST"]
)
def api_upload():

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

    try:

        for file in files:

            if not file.filename:

                continue

            filename = os.path.basename(
                file.filename
            )

            save_path = os.path.join(
                UPLOAD_FOLDER,
                filename
            )

            file.save(
                save_path
            )

            chunks = add_document(
                save_path,
                filename
            )

            total_chunks += chunks

        return jsonify(
            {
                "status": "success",
                "total_chunks": total_chunks,
                "message":
                    f"Added {total_chunks} chunks to FAISS."
            }
        )

    except Exception as e:

        print(
            f"Upload error: {e}"
        )

        return jsonify(
            {
                "status": "error",
                "error": str(e)
            }
        ), 500


# ============================================================
# DOCUMENT LIST
# ============================================================

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


# ============================================================
# CLEAR ALL
# ============================================================

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

        save_faiss()

    except Exception as e:

        print(
            f"Clear error: {e}"
        )

    return jsonify(
        {
            "status":
                "success"
        }
    )


# ============================================================
# DELETE ONE DOCUMENT
# ============================================================

@app.route(
    "/api/delete",
    methods=["POST"]
)
def api_delete():

    global faiss_index
    global chunks_registry
    global uploaded_documents

    data = request.get_json() or {}

    filename = data.get(
        "filename",
        ""
    ).strip()

    if not filename:

        return jsonify(
            {
                "error":
                "Filename required."
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

    # Rebuild FAISS index

    faiss_index = create_empty_index()

    if chunks_registry:

        texts = [
            chunk["text"]
            for chunk in chunks_registry
        ]

        embeddings = create_embeddings(
            texts
        )

        faiss_index.add(
            embeddings
        )

    file_path = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    if os.path.exists(
        file_path
    ):

        try:

            os.remove(
                file_path
            )

        except Exception as e:

            print(
                f"File deletion error: {e}"
            )

    save_faiss()

    return jsonify(
        {
            "status":
                "success"
        }
    )


# ============================================================
# QUERY API
# ============================================================

@app.route(
    "/api/query",
    methods=["POST"]
)
def api_query():

    data = request.get_json() or {}

    question = data.get(
        "query",
        ""
    ).strip()

    if not question:

        return jsonify(
            {
                "error":
                "Question cannot be empty."
            }
        ), 400

    try:

        # RAG RETRIEVAL

        retrieved_chunks = search_documents(
            question,
            top_k=3
        )

        if not retrieved_chunks:

            return jsonify(
                {
                    "answer":
                    "No documents have been uploaded yet.",
                    "sources": []
                }
            )

        context_parts = []

        for chunk in retrieved_chunks:

            context_parts.append(
                f"""
Document: {chunk['filename']}
Page: {chunk['page']}

{chunk['text']}
"""
            )

        context = "\n\n".join(
            context_parts
        )

        # LLM

        answer = call_openai(
            question,
            context
        )

        return jsonify(
            {
                "answer": answer,
                "sources":
                    retrieved_chunks
            }
        )

    except Exception as e:

        print(
            f"Query error: {e}"
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
        "\n"
        + "=" * 60
    )

    print(
        "My Documents RAG Application"
    )

    print(
        "Embedding: OpenAI text-embedding-3-small"
    )

    print(
        "Vector Database: FAISS"
    )

    print(
        "LLM: OpenAI"
    )

    print(
        "=" * 60
        + "\n"
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )