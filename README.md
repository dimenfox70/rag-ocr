# Document Chat Assistant (RAG)

A clean, modern **"Chat with Documents"** web app built with **Streamlit**. Upload PDF / DOCX / TXT files, and ask questions about them. The app is a complete local **Retrieval-Augmented Generation (RAG)** pipeline:

**Upload → Text Extraction → Chunking → Embeddings → FAISS Indexing → Retrieval → Grounded Answer Generation**

It runs entirely on your machine (CPU) and uses open-source Hugging Face models — no external API keys required.

---

## Features

- **Multi-format upload** — PDF (`.pdf`), Word (`.docx`), and Text (`.txt`), multiple files at once.
- **Text extraction** — `PyPDF2` (with page markers), `python-docx`, and built-in text handling.
- **Chunking** — LangChain `RecursiveCharacterTextSplitter` (`chunk_size=1000`, `chunk_overlap=200`).
- **Embeddings** — `sentence-transformers/all-MiniLM-L6-v2` (normalized for stable similarity).
- **Vector search** — in-memory **FAISS** `IndexFlatL2`.
- **Grounded generation** — `microsoft/Phi-3-mini-4k-instruct` answers strictly from retrieved context, with source citations.
- **Retrieval settings** — adjustable Top-K and number of source previews.
- **Diagnostics & confidence** — per-answer retrieval count, timing (ms), and a High/Medium/Low confidence indicator.
- **Source citations** — every answer lists `filename (Chunk X)` with an expandable "View Sources" preview.
- **Session controls** — example query buttons and a "Clear Session" reset.

---

## Tech Stack

| Layer | Library |
|-------|---------|
| UI | Streamlit |
| Extraction | PyPDF2, python-docx |
| Chunking | langchain-text-splitters |
| Embeddings | sentence-transformers |
| Vector index | faiss-cpu |
| Generation | transformers, torch, accelerate, sentencepiece |

---

## Getting Started

### 1. Prerequisites
- **Python 3.10+**
- ~16 GB RAM recommended (the Phi-3-mini model runs in float32 on CPU)

### 2. Clone the repository
```bash
git clone https://github.com/dimenfox70/rag-ocr.git
cd rag-ocr
```

### 3. Create a virtual environment (recommended)
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the app
```bash
streamlit run app.py
```
The app opens at `http://localhost:8501`.

> **First run note:** The first question triggers a one-time download of the embedding model (~80 MB) and the Phi-3-mini model (~7–8 GB). Both are cached afterward, so subsequent runs are fast.

---

## How to Use

1. Upload one or more documents in the **Documents** sidebar.
2. Click **Process Documents** to extract, chunk, embed, and index the text.
3. Wait for the **Ready to Chat** indicator.
4. Ask a question in the chat box (or use an example query button).
5. The assistant answers using only your documents and cites its sources.

Use **Retrieval Settings** to tune Top-K and source previews, and **Clear Session** to start over.

---

## Project Structure

```
rag-ocr/
├── app.py             # Full Streamlit RAG application
├── requirements.txt   # Python dependencies
├── .gitignore
└── README.md
```

---

## Configuration

Key settings live near the top of `app.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | Sentence-Transformers model |
| `GENERATION_MODEL_NAME` | `Phi-3-mini-4k-instruct` | Hugging Face causal LM |
| `MAX_NEW_TOKENS` | `256` | Max tokens generated per answer |
| `TEMPERATURE` | `0.2` | Generation temperature |
| `TOP_K` | `3` | Chunks retrieved per question |

### Using a lighter model
If you're RAM-constrained, change `GENERATION_MODEL_NAME` (and optionally `torch_dtype`) inside `load_generation_model()` — for example to `Qwen/Qwen2.5-1.5B-Instruct`. The generation layer is isolated, so no other code needs to change.

---

## Notes & Limitations

- CPU inference with Phi-3-mini is functional but slow; expect several seconds per answer.
- Scanned/image-only PDFs have no extractable text layer (no OCR is performed).
- No conversation memory — each question is answered independently.

---

## License

Add a license of your choice (e.g. MIT) if you plan to share this publicly.
