"""
Document Chat Assistant
-----------------------
A clean, modern "Chat with Documents" UI built with Streamlit.

This file implements the frontend, DOCUMENT TEXT EXTRACTION, TEXT CHUNKING, and
EMBEDDINGS + FAISS INDEXING. When the user clicks "Process Documents", uploaded
PDF / DOCX / TXT files are read directly from Streamlit, their text is extracted
into ``st.session_state["documents"]``, split into overlapping chunks in
``st.session_state["chunks"]``, embedded with a Sentence-Transformers model, and
indexed into an in-memory FAISS index.

Chat is RETRIEVAL-AUGMENTED GENERATION (RAG): a question is embedded with the
cached model, the FAISS index is searched for the most similar chunks, the
retrieved chunks are combined into a context string, and a Hugging Face causal
LM (microsoft/Phi-3-mini-4k-instruct) generates a grounded, natural-language
answer constrained to that context. Source citations are appended to the
answer.

The generation layer is isolated: swapping models only requires changing
``load_generation_model``. The retrieval layer (``embed_query``,
``search_index``, ``retrieve_context``) is unchanged.
"""

import io
import time

import numpy as np
import streamlit as st

# Third-party extraction libraries. Imported lazily-friendly at module top so a
# clear, actionable error is shown if a dependency is missing.
from PyPDF2 import PdfReader
from docx import Document as DocxDocument

# LangChain's recursive splitter handles sentence/paragraph-aware chunking.
from langchain_text_splitters import RecursiveCharacterTextSplitter

# FAISS provides the in-memory vector index. ``SentenceTransformer`` is imported
# lazily inside the cached loader so module import stays fast.
import faiss

# ---------------------------------------------------------------------------
# Constants & Configuration
# ---------------------------------------------------------------------------

# File types the uploader will accept (extensions without the leading dot).
SUPPORTED_TYPES = ["pdf", "docx", "txt"]

# Human-friendly label for the supported formats notice.
SUPPORTED_FORMATS_LABEL = "PDF, Word (.docx), and Text (.txt)"

# The assistant's opening greeting shown when the app first loads.
WELCOME_MESSAGE = (
    "Hello! Upload one or more documents and ask me anything about them."
)

# Chunking configuration for the text splitter. These values are a good
# general-purpose default for RAG and are kept here so they're easy to tune.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Number of chunks to preview in the UI after processing.
CHUNK_PREVIEW_COUNT = 5

# Hugging Face Sentence-Transformers model used to embed chunks.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Hugging Face causal LM used to generate grounded answers.
GENERATION_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

# Generation settings.
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.2

# Returned by the model when the answer isn't in the context.
NOT_FOUND_MESSAGE = "I couldn't find this information in the uploaded documents."

# Number of indexed chunks to preview in the indexing results panel.
INDEX_PREVIEW_COUNT = 5

# Default number of chunks to retrieve per question (user-tunable).
TOP_K = 3

# Default number of source previews shown under an answer (user-tunable).
DEFAULT_SOURCE_PREVIEWS = 3

# How many characters of each retrieved chunk to show under "View Sources".
SOURCE_PREVIEW_CHARS = 500

# Confidence thresholds applied to the best normalized similarity score.
HIGH_CONFIDENCE_THRESHOLD = 0.5
MEDIUM_CONFIDENCE_THRESHOLD = 0.3

# Prepended to answers when only weak matches were found.
LOW_CONFIDENCE_WARNING = (
    "The answer may be incomplete because only weak matches were found in the "
    "uploaded documents."
)

# Message shown when the user asks a question before processing documents.
NOT_READY_MESSAGE = (
    "Please upload and process documents before asking questions."
)

# Example queries rendered as buttons below the chat input.
EXAMPLE_QUERIES = [
    "Summarize this document",
    "What are the important points?",
    "What action items are mentioned?",
    "Explain this in simple language",
]

# Quick-action suggestions displayed under the welcome message.
SUGGESTIONS = [
    "Summarize this document",
    "What are the important points?",
    "Give me key takeaways",
    "Explain this in simple language",
    "What action items are mentioned?",
]


# ---------------------------------------------------------------------------
# Page Setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Document Chat Assistant",  # Browser tab title.
    page_icon="📄",
    layout="wide",  # Use the full screen width.
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def format_size(num_bytes: int) -> str:
    """Convert a raw byte count into a readable string (B / KB / MB)."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            # Show whole numbers for bytes, one decimal otherwise.
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


# ---------------------------------------------------------------------------
# Document Text Extraction
# ---------------------------------------------------------------------------
#
# Each extractor receives an in-memory bytes buffer (the raw file contents read
# from the Streamlit UploadedFile) and returns a tuple of:
#     (extracted_text: str, page_count: int)
#
# Keeping each format in its own small, testable function makes the extraction
# layer modular and easy to extend with new file types later.

def extract_pdf_text(file_bytes: bytes) -> tuple[str, int]:
    """Extract text from a PDF, preserving page numbers.

    Each page's text is prefixed with a ``--- Page N ---`` marker so page
    boundaries (and numbering) are retained in the stored text.
    """
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = reader.pages
    page_texts = []
    for index, page in enumerate(pages, start=1):
        # ``extract_text`` can return None for pages with no extractable text
        # (e.g. scanned images); fall back to an empty string in that case.
        page_text = page.extract_text() or ""
        page_texts.append(f"--- Page {index} ---\n{page_text.strip()}")

    full_text = "\n\n".join(page_texts).strip()
    return full_text, len(pages)


def extract_docx_text(file_bytes: bytes) -> tuple[str, int]:
    """Extract text from a Word (.docx) document.

    DOCX has no fixed page concept available without rendering, so the page
    count is reported as 0 (not applicable) for this format.
    """
    document = DocxDocument(io.BytesIO(file_bytes))
    # Join non-empty paragraphs to avoid long runs of blank lines.
    paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
    full_text = "\n".join(paragraphs).strip()
    return full_text, 0


def extract_txt_text(file_bytes: bytes) -> tuple[str, int]:
    """Extract text from a plain-text (.txt) file.

    Decoding is attempted as UTF-8 first, then falls back to a permissive
    decode so unusual encodings don't crash processing.
    """
    try:
        text = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text = file_bytes.decode("utf-8", errors="replace")
    return text.strip(), 0


# Map of file extension -> (extractor function, canonical file_type label).
EXTRACTORS = {
    "pdf": (extract_pdf_text, "pdf"),
    "docx": (extract_docx_text, "docx"),
    "txt": (extract_txt_text, "txt"),
}


def get_file_extension(filename: str) -> str:
    """Return the lowercased extension (without dot) of a filename."""
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def process_documents(uploaded_files) -> tuple[list[dict], list[str]]:
    """Extract text from each uploaded file.

    Returns a tuple of:
        - documents: list of dicts shaped as
          {"filename", "file_type", "pages", "text"}
        - errors: list of human-readable error messages for failed files

    Each file is processed independently so a single failure does not abort
    the rest of the batch.
    """
    documents: list[dict] = []
    errors: list[str] = []

    for file in uploaded_files:
        extension = get_file_extension(file.name)
        extractor_entry = EXTRACTORS.get(extension)

        # Guard against unsupported types slipping through.
        if extractor_entry is None:
            errors.append(f"{file.name}: unsupported file type '.{extension}'.")
            continue

        extractor, file_type = extractor_entry
        try:
            # Read raw bytes directly from the Streamlit UploadedFile.
            file_bytes = file.getvalue()
            text, pages = extractor(file_bytes)

            if not text:
                errors.append(
                    f"{file.name}: no readable text could be extracted."
                )
                continue

            documents.append(
                {
                    "filename": file.name,
                    "file_type": file_type,
                    "pages": pages,
                    "text": text,
                }
            )
        except Exception as exc:  # noqa: BLE001 - surface any parsing failure
            errors.append(f"{file.name}: failed to process ({exc}).")

    return documents, errors


# ---------------------------------------------------------------------------
# Text Chunking
# ---------------------------------------------------------------------------
#
# The chunking layer turns each extracted document into a list of overlapping
# text chunks while preserving its metadata. It is intentionally decoupled from
# the UI so the same functions can be reused directly by the RAG pipeline later.

def get_text_splitter() -> RecursiveCharacterTextSplitter:
    """Build the configured RecursiveCharacterTextSplitter.

    Centralising construction keeps the chunk settings in one place.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )


def split_document_into_chunks(document: dict, splitter=None) -> list[dict]:
    """Split a single extracted document into metadata-rich chunks.

    ``document`` is a dict shaped as
    {"filename", "file_type", "pages", "text"}.

    Returns a list of chunk dicts, each shaped as
    {"chunk_id", "filename", "file_type", "content"}.

    ``chunk_id`` here is local to the document (starts at 1); ``process_chunks``
    reassigns globally-unique ids across the whole batch.
    """
    splitter = splitter or get_text_splitter()
    text = document.get("text", "") or ""

    # Nothing to split if the document has no text.
    if not text.strip():
        return []

    pieces = splitter.split_text(text)
    chunks = []
    for local_id, content in enumerate(pieces, start=1):
        chunks.append(
            {
                "chunk_id": local_id,
                "filename": document.get("filename", "unknown"),
                "file_type": document.get("file_type", "unknown"),
                "content": content,
            }
        )
    return chunks


def process_chunks(documents: list[dict]) -> tuple[list[dict], list[str]]:
    """Chunk every document, preserving metadata and assigning unique ids.

    Returns a tuple of:
        - chunks: list of {"chunk_id", "filename", "file_type", "content"}
          with ``chunk_id`` unique and sequential across all documents.
        - errors: human-readable messages for any document that failed.

    Each document is chunked independently so one failure does not stop the
    rest of the batch from being processed.
    """
    splitter = get_text_splitter()
    chunks: list[dict] = []
    errors: list[str] = []
    next_id = 1

    for document in documents:
        filename = document.get("filename", "unknown")
        try:
            doc_chunks = split_document_into_chunks(document, splitter)
            # Reassign globally-unique, sequential ids across the batch.
            for chunk in doc_chunks:
                chunk["chunk_id"] = next_id
                next_id += 1
                chunks.append(chunk)
        except Exception as exc:  # noqa: BLE001 - surface any chunking failure
            errors.append(f"{filename}: failed to chunk ({exc}).")

    return chunks, errors


# ---------------------------------------------------------------------------
# Embeddings & FAISS Indexing
# ---------------------------------------------------------------------------
#
# This layer turns chunks into vector embeddings and indexes them in FAISS.
# Every function here is independent of the UI so the RAG pipeline can reuse
# them directly later for retrieval.

@st.cache_resource(show_spinner=False)
def load_embedding_model(model_name: str = EMBEDDING_MODEL_NAME):
    """Load (and cache) the Sentence-Transformers embedding model.

    ``st.cache_resource`` ensures the model is downloaded/loaded only once per
    session (and shared across reruns), which avoids slow reloads.
    """
    # Imported here so the (heavy) dependency only loads when actually needed.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def generate_embeddings(chunks: list[dict], model) -> np.ndarray:
    """Embed the ``content`` of every chunk into a float32 matrix.

    Returns a 2D numpy array of shape (num_chunks, embedding_dim), which is the
    format FAISS expects.
    """
    texts = [chunk["content"] for chunk in chunks]
    # Normalize embeddings so squared-L2 distance maps cleanly to cosine
    # similarity, which makes the confidence scoring meaningful and stable.
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    # FAISS requires contiguous float32 arrays.
    return np.asarray(embeddings, dtype="float32")


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """Build an in-memory FAISS ``IndexFlatL2`` from an embedding matrix."""
    dimension = int(embeddings.shape[1])
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


def prepare_metadata(chunks: list[dict]) -> list[dict]:
    """Build the per-chunk metadata list stored alongside the FAISS index.

    The FAISS index stores only vectors; this parallel list maps each vector
    position back to its source chunk. Embeddings are intentionally NOT stored
    here.
    """
    return [
        {
            "chunk_id": chunk["chunk_id"],
            "filename": chunk["filename"],
            "file_type": chunk["file_type"],
            "content": chunk["content"],
        }
        for chunk in chunks
    ]


def build_search_index(chunks: list[dict]):
    """Run the full embed-and-index pipeline for a list of chunks.

    Returns a tuple of (model, faiss_index, metadata, embedding_dim). Kept as a
    thin orchestrator so the UI stays simple and the steps remain individually
    reusable.
    """
    model = load_embedding_model()
    embeddings = generate_embeddings(chunks, model)
    index = build_faiss_index(embeddings)
    metadata = prepare_metadata(chunks)
    return model, index, metadata, int(embeddings.shape[1])


def init_session_state() -> None:
    """Create any session_state keys we rely on if they don't exist yet.

    Using session_state keeps the chat history and other state alive across
    Streamlit reruns (every interaction triggers a full script rerun).
    """
    if "messages" not in st.session_state:
        # Chat history is a list of {"role": ..., "content": ...} dicts.
        # Seed it with the assistant's welcome message.
        st.session_state.messages = [
            {"role": "assistant", "content": WELCOME_MESSAGE}
        ]

    if "uploaded_files" not in st.session_state:
        # Holds the most recent list of uploaded file objects.
        st.session_state.uploaded_files = []

    if "documents_processed" not in st.session_state:
        # Simple flag to drive placeholder "processed" messaging.
        st.session_state.documents_processed = False

    if "documents" not in st.session_state:
        # Holds extracted documents:
        # [{"filename", "file_type", "pages", "text"}, ...]
        st.session_state.documents = []

    if "processing_errors" not in st.session_state:
        # Error messages from the most recent processing run.
        st.session_state.processing_errors = []

    if "chunks" not in st.session_state:
        # Holds chunked text:
        # [{"chunk_id", "filename", "file_type", "content"}, ...]
        st.session_state.chunks = []

    if "chunking_errors" not in st.session_state:
        # Error messages from the most recent chunking run.
        st.session_state.chunking_errors = []

    if "embedding_model" not in st.session_state:
        # The loaded Sentence-Transformers model (set after indexing).
        st.session_state.embedding_model = None

    if "faiss_index" not in st.session_state:
        # The in-memory FAISS index of chunk embeddings.
        st.session_state.faiss_index = None

    if "metadata" not in st.session_state:
        # Parallel metadata list mapping vector positions back to chunks.
        st.session_state.metadata = []

    if "embedding_dim" not in st.session_state:
        # Dimensionality of the embedding vectors.
        st.session_state.embedding_dim = None

    if "is_ready_to_chat" not in st.session_state:
        # True once embeddings are generated and the index is built.
        st.session_state.is_ready_to_chat = False

    if "indexing_error" not in st.session_state:
        # User-friendly message if embedding/indexing failed.
        st.session_state.indexing_error = None

    if "top_k" not in st.session_state:
        # User-tunable number of chunks retrieved per question.
        st.session_state.top_k = TOP_K

    if "num_source_previews" not in st.session_state:
        # User-tunable number of source previews shown under an answer.
        st.session_state.num_source_previews = DEFAULT_SOURCE_PREVIEWS

    if "pending_prompt" not in st.session_state:
        # Buffer for a suggestion click so it can be handled on the next rerun.
        st.session_state.pending_prompt = None


def clear_session() -> None:
    """Reset all document/chat state back to a fresh start.

    Retrieval settings (top_k, source previews) are intentionally preserved.
    The cached embedding model is left loaded so it doesn't need to reload.
    """
    st.session_state.documents = []
    st.session_state.processing_errors = []
    st.session_state.documents_processed = False
    st.session_state.chunks = []
    st.session_state.chunking_errors = []
    st.session_state.embedding_model = None
    st.session_state.faiss_index = None
    st.session_state.metadata = []
    st.session_state.embedding_dim = None
    st.session_state.is_ready_to_chat = False
    st.session_state.indexing_error = None
    # Reset chat history back to just the welcome message.
    st.session_state.messages = [
        {"role": "assistant", "content": WELCOME_MESSAGE}
    ]
    st.session_state.pending_prompt = None


# ---------------------------------------------------------------------------
# Retrieval (question -> embedding -> FAISS search -> chunks)
# ---------------------------------------------------------------------------
#
# Each function does one step of retrieval and is independent of the UI so the
# RAG pipeline can reuse them. To add a generative model later, only
# ``build_answer`` needs to change.

def embed_query(question: str, model) -> np.ndarray:
    """Embed a single question into a float32 row vector for FAISS search.

    Uses the same normalization as the indexed chunks so distances are
    comparable.
    """
    embedding = model.encode(
        [question],
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return np.asarray(embedding, dtype="float32")


def search_index(
    query_embedding: np.ndarray, faiss_index, top_k: int = TOP_K
) -> tuple[list[int], list[float]]:
    """Search the FAISS index and return (indices, distances) of top matches.

    ``top_k`` is clamped to the number of vectors actually in the index so we
    never request more results than exist. Distances are squared-L2.
    """
    k = min(top_k, faiss_index.ntotal)
    if k <= 0:
        return [], []
    distances, indices = faiss_index.search(query_embedding, k)
    # Both arrays are shape (1, k); flatten and drop FAISS's -1 "not found".
    pairs = [
        (int(i), float(d))
        for i, d in zip(indices[0], distances[0])
        if i != -1
    ]
    out_indices = [i for i, _ in pairs]
    out_distances = [d for _, d in pairs]
    return out_indices, out_distances


def retrieve_context(indices: list[int], metadata: list[dict]) -> list[dict]:
    """Map FAISS row indices back to their chunk metadata."""
    return [metadata[i] for i in indices if 0 <= i < len(metadata)]


def distance_to_score(distance: float) -> float:
    """Convert a squared-L2 distance (normalized vectors) to a 0..1 score.

    For unit vectors, squared-L2 = 2 - 2*cosine, so cosine = 1 - distance/2.
    We clamp to [0, 1] since negative cosine means "irrelevant".
    """
    score = 1.0 - (distance / 2.0)
    return max(0.0, min(1.0, score))


def confidence_label(score: float) -> str:
    """Map a normalized similarity score to a human-readable label."""
    if score >= HIGH_CONFIDENCE_THRESHOLD:
        return "High Confidence"
    if score >= MEDIUM_CONFIDENCE_THRESHOLD:
        return "Medium Confidence"
    return "Low Confidence"


def build_context(retrieved: list[dict]) -> str:
    """Combine retrieved chunks into a single context string for the LLM."""
    return "\n\n".join(chunk["content"].strip() for chunk in retrieved)


def build_sources_block(retrieved: list[dict]) -> str:
    """Build the deduplicated, ordered "Sources:" citation block."""
    source_lines = []
    seen = set()
    for chunk in retrieved:
        key = (chunk["filename"], chunk["chunk_id"])
        if key in seen:
            continue
        seen.add(key)
        source_lines.append(f"• {chunk['filename']} (Chunk {chunk['chunk_id']})")
    return "Sources:\n" + "\n".join(source_lines)


# ---------------------------------------------------------------------------
# Answer Generation (grounded LLM)
# ---------------------------------------------------------------------------
#
# This layer is fully isolated from retrieval. To swap in a different model,
# only ``load_generation_model`` needs to change.

# Exact prompt template used to ground the model in the retrieved context.
PROMPT_TEMPLATE = """You are a helpful document assistant.

Answer ONLY using the provided context.

If the answer cannot be found in the context, say:
"I couldn't find this information in the uploaded documents."

Keep answers concise and factual.

Context:
{context}

Question:
{question}

Answer:"""


@st.cache_resource(show_spinner=False)
def load_generation_model(model_name: str = GENERATION_MODEL_NAME):
    """Load (and cache) the tokenizer and causal LM for CPU inference.

    ``st.cache_resource`` ensures the (large) model is loaded only once per
    session. Heavy imports are kept local so module import stays fast.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # CPU inference uses float32.
        trust_remote_code=True,
        attn_implementation="eager",  # Avoid flash-attn on CPU.
    )
    model.to("cpu")
    model.eval()
    return tokenizer, model


def build_prompt(context: str, question: str) -> str:
    """Fill the grounding prompt template with the context and question."""
    return PROMPT_TEMPLATE.format(context=context, question=question)


def generate_answer(context: str, question: str) -> str:
    """Generate a grounded natural-language answer from context + question.

    REPLACEMENT POINT
    -----------------
    This is the only place that calls the generative model. To use a different
    model, change ``load_generation_model`` (and, if needed, this decoding
    logic); the retrieval pipeline stays untouched.
    """
    import torch

    tokenizer, model = load_generation_model()
    prompt = build_prompt(context, question)

    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (exclude the prompt).
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return answer.strip()


def generate_response(user_message: str) -> dict:
    """Produce an assistant reply for a question via retrieval.

    Returns a dict with keys:
        - content: the answer text
        - sources: retrieved chunks, each augmented with ``rank`` and ``score``
        - diagnostics: {retrieved_count, query_time_ms, faiss_time_ms} or None
        - confidence: {label, score} or None

    The RETRIEVAL pipeline (embed_query/search_index/retrieve_context) is
    unchanged. After retrieval the chunks are combined into a context string
    and passed to ``generate_answer`` for grounded natural-language generation.
    """
    # Guard: the index must be built before we can answer.
    if not st.session_state.get("is_ready_to_chat"):
        return {"content": NOT_READY_MESSAGE, "sources": [],
                "diagnostics": None, "confidence": None}

    model = st.session_state.get("embedding_model")
    index = st.session_state.get("faiss_index")
    metadata = st.session_state.get("metadata", [])
    top_k = int(st.session_state.get("top_k", TOP_K))

    if model is None or index is None or not metadata:
        return {"content": NOT_READY_MESSAGE, "sources": [],
                "diagnostics": None, "confidence": None}

    try:
        # --- Timed retrieval (UNCHANGED pipeline) ---
        start = time.perf_counter()
        query_embedding = embed_query(user_message, model)

        faiss_start = time.perf_counter()
        indices, distances = search_index(query_embedding, index, top_k=top_k)
        faiss_ms = (time.perf_counter() - faiss_start) * 1000.0

        retrieved = retrieve_context(indices, metadata)
        total_ms = (time.perf_counter() - start) * 1000.0
    except Exception as exc:  # noqa: BLE001 - retrieval failure
        return {
            "content": (
                "Sorry, something went wrong while searching your documents "
                f"({exc}). Please try again."
            ),
            "sources": [],
            "diagnostics": None,
            "confidence": None,
        }

    # No matches at all: short-circuit before invoking the LLM.
    if not retrieved:
        return {
            "content": NOT_FOUND_MESSAGE,
            "sources": [],
            "diagnostics": {
                "retrieved_count": 0,
                "query_time_ms": total_ms,
                "faiss_time_ms": faiss_ms,
            },
            "confidence": {"label": "Low Confidence", "score": 0.0},
        }

    # Attach rank (1-based) and normalized score to each source.
    scores = [distance_to_score(d) for d in distances]
    sources = []
    for rank, (chunk, score) in enumerate(zip(retrieved, scores), start=1):
        enriched = dict(chunk)  # shallow copy so we don't mutate metadata
        enriched["rank"] = rank
        enriched["score"] = score
        sources.append(enriched)

    # Confidence comes from the best (first) match.
    best_score = scores[0] if scores else 0.0
    label = confidence_label(best_score)

    # --- Grounded generation ---
    context = build_context(retrieved)
    try:
        with st.spinner("Generating answer..."):
            generated = generate_answer(context, user_message)
    except Exception as exc:  # noqa: BLE001 - generation failure (friendly msg)
        return {
            "content": (
                "Sorry, I couldn't generate an answer right now "
                f"({exc}). Please try again."
            ),
            "sources": sources,
            "diagnostics": {
                "retrieved_count": len(retrieved),
                "query_time_ms": total_ms,
                "faiss_time_ms": faiss_ms,
            },
            "confidence": {"label": label, "score": best_score},
        }

    # Compose final answer: generated text + preserved source citations.
    answer = f"{generated}\n\n{build_sources_block(retrieved)}"
    # Prepend a warning if confidence is low.
    if label == "Low Confidence":
        answer = f"{LOW_CONFIDENCE_WARNING}\n\n{answer}"

    return {
        "content": answer,
        "sources": sources,
        "diagnostics": {
            "retrieved_count": len(retrieved),
            "query_time_ms": total_ms,
            "faiss_time_ms": faiss_ms,
        },
        "confidence": {"label": label, "score": best_score},
    }


def handle_user_message(user_message: str) -> None:
    """Append a user message and the assistant's retrieved response to history."""
    st.session_state.messages.append({"role": "user", "content": user_message})
    result = generate_response(user_message)
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["content"],
            "sources": result["sources"],
            "diagnostics": result["diagnostics"],
            "confidence": result["confidence"],
        }
    )


# ---------------------------------------------------------------------------
# Sidebar: Document Upload & Management
# ---------------------------------------------------------------------------

def render_sidebar():
    """Render the document upload sidebar and return the uploaded files."""
    with st.sidebar:
        st.header("📁 Documents")
        st.caption("Upload files to chat with their content.")

        # File uploader accepting multiple PDF / DOCX / TXT files.
        uploaded_files = st.file_uploader(
            label="Upload documents",
            type=SUPPORTED_TYPES,
            accept_multiple_files=True,
            help=f"Supported formats: {SUPPORTED_FORMATS_LABEL}",
        )

        # Keep a reference in session_state so other sections can use it.
        st.session_state.uploaded_files = uploaded_files or []

        st.divider()

        # Show details for each uploaded file.
        if uploaded_files:
            st.subheader("Uploaded files")
            for file in uploaded_files:
                # An expander keeps the sidebar tidy when many files exist.
                with st.expander(file.name, expanded=False):
                    st.write(f"**Name:** {file.name}")
                    st.write(f"**Size:** {format_size(file.size)}")
                    st.write(f"**Type:** {file.type or 'unknown'}")
        else:
            st.info("No documents uploaded yet.")

        st.divider()

        # Process button is disabled until at least one file is uploaded.
        process_clicked = st.button(
            "⚙️ Process Documents",
            use_container_width=True,
            disabled=not uploaded_files,
        )

        # Run text extraction then chunking when the button is clicked.
        if process_clicked:
            with st.spinner("Extracting text from documents..."):
                documents, errors = process_documents(uploaded_files)

            # Persist extraction results so they survive subsequent reruns.
            st.session_state.documents = documents
            st.session_state.processing_errors = errors
            st.session_state.documents_processed = True

            # Chunk the freshly extracted documents.
            with st.spinner("Splitting documents into chunks..."):
                chunks, chunking_errors = process_chunks(documents)

            st.session_state.chunks = chunks
            st.session_state.chunking_errors = chunking_errors

            # Generate embeddings and build the FAISS index from the chunks.
            # Reset any previous "ready" state before re-indexing.
            st.session_state.is_ready_to_chat = False
            st.session_state.indexing_error = None
            if chunks:
                try:
                    with st.spinner(
                        "Generating embeddings and building index..."
                    ):
                        model, index, metadata, dim = build_search_index(chunks)

                    # Persist the model, index and metadata for retrieval later.
                    st.session_state.embedding_model = model
                    st.session_state.faiss_index = index
                    st.session_state.metadata = metadata
                    st.session_state.embedding_dim = dim
                    st.session_state.is_ready_to_chat = True
                except Exception as exc:  # noqa: BLE001 - surface to the user
                    st.session_state.indexing_error = str(exc)

            if documents:
                st.success(f"Processed {len(documents)} document(s).")
            if errors:
                st.warning(f"{len(errors)} file(s) could not be processed.")
            if chunking_errors:
                st.warning(
                    f"{len(chunking_errors)} document(s) could not be chunked."
                )
            if st.session_state.indexing_error:
                st.error("Indexing failed. See details in the main panel.")
        elif st.session_state.documents_processed and st.session_state.documents:
            st.info(
                f"{len(st.session_state.documents)} document(s) ready, "
                f"{len(st.session_state.chunks)} chunk(s)."
            )

        # Persistent readiness indicator in the sidebar.
        if st.session_state.is_ready_to_chat:
            st.success("✅ Ready to Chat")

        st.divider()

        # ------------------------------------------------------------------
        # Retrieval Settings
        # ------------------------------------------------------------------
        st.header("⚙️ Retrieval Settings")
        # Sliders bound to session_state via ``key`` so values persist and are
        # read directly by the retrieval pipeline.
        st.slider(
            "Top K retrieval",
            min_value=1,
            max_value=10,
            key="top_k",
            help="How many chunks to retrieve per question.",
        )
        st.slider(
            "Source previews displayed",
            min_value=1,
            max_value=5,
            key="num_source_previews",
            help="How many retrieved sources to show under each answer.",
        )

        st.divider()

        # ------------------------------------------------------------------
        # Clear Session
        # ------------------------------------------------------------------
        if st.button("🗑️ Clear Session", use_container_width=True):
            clear_session()
            st.rerun()

    return st.session_state.uploaded_files


# ---------------------------------------------------------------------------
# Main Area: Header
# ---------------------------------------------------------------------------

def render_header():
    """Render the top header and subtitle."""
    st.title("📄 Chat with Documents")
    st.markdown(
        "##### Upload your documents and ask questions about them."
    )
    st.divider()


# ---------------------------------------------------------------------------
# Main Area: Document Information Panel
# ---------------------------------------------------------------------------

def render_document_info(uploaded_files):
    """Show a compact summary of the uploaded documents above the chat."""
    if not uploaded_files:
        return

    total_size = sum(file.size for file in uploaded_files)

    # Three metrics laid out side by side for a clean, responsive look.
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Documents", len(uploaded_files))
    with col2:
        st.metric("Total size", format_size(total_size))
    with col3:
        st.metric("Supported formats", "PDF • DOCX • TXT")

    st.caption(f"Supported formats: {SUPPORTED_FORMATS_LABEL}.")
    st.divider()


# ---------------------------------------------------------------------------
# Main Area: Extraction Results Panel
# ---------------------------------------------------------------------------

def render_extraction_results():
    """Show extraction results for documents that have been processed.

    Displays, per document: filename, file type, page count and character
    count, plus any errors raised during processing.
    """
    documents = st.session_state.get("documents", [])
    errors = st.session_state.get("processing_errors", [])

    # Nothing to show until the user has processed at least once.
    if not documents and not errors:
        return

    st.subheader("Processing results")

    # Per-document success summary with key statistics.
    for doc in documents:
        char_count = len(doc["text"])
        # "Pages" is only meaningful for PDFs; show N/A otherwise.
        page_display = doc["pages"] if doc["file_type"] == "pdf" else "N/A"

        st.success(f"Extracted text from **{doc['filename']}**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("File", doc["filename"])
        with col2:
            st.metric("Type", doc["file_type"].upper())
        with col3:
            st.metric("Pages", page_display)
        with col4:
            st.metric("Characters", f"{char_count:,}")

        # Let the user peek at the extracted text without cluttering the UI.
        with st.expander("Preview extracted text", expanded=False):
            preview = doc["text"][:2000]
            st.text(preview + ("..." if len(doc["text"]) > 2000 else ""))

    # Surface any files that failed to process.
    for message in errors:
        st.error(message)

    st.divider()


# ---------------------------------------------------------------------------
# Main Area: Chunking Results Panel
# ---------------------------------------------------------------------------

def render_chunk_results():
    """Show chunk statistics and a preview of the first few chunks.

    Reads from ``st.session_state["chunks"]`` and displays aggregate stats plus
    an expandable preview of the first ``CHUNK_PREVIEW_COUNT`` chunks.
    """
    chunks = st.session_state.get("chunks", [])
    chunking_errors = st.session_state.get("chunking_errors", [])

    # Nothing to show until chunking has run.
    if not chunks and not chunking_errors:
        return

    st.subheader("Chunking results")

    if chunks:
        # Compute aggregate statistics over chunk lengths.
        lengths = [len(chunk["content"]) for chunk in chunks]
        total_chunks = len(chunks)
        # Documents processed = distinct filenames represented in the chunks.
        docs_processed = len({chunk["filename"] for chunk in chunks})
        avg_length = sum(lengths) // total_chunks
        largest = max(lengths)
        smallest = min(lengths)

        # Success message in the requested format.
        st.success(
            f"✅ {total_chunks} chunks created successfully and ready for indexing."
        )

        # Five-column statistics row.
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Documents processed", docs_processed)
        with col2:
            st.metric("Total chunks", total_chunks)
        with col3:
            st.metric("Avg chunk length", f"{avg_length:,}")
        with col4:
            st.metric("Largest chunk", f"{largest:,}")
        with col5:
            st.metric("Smallest chunk", f"{smallest:,}")

        # Expandable preview of the first N chunks.
        with st.expander(
            f"Preview first {min(CHUNK_PREVIEW_COUNT, total_chunks)} chunks",
            expanded=False,
        ):
            for chunk in chunks[:CHUNK_PREVIEW_COUNT]:
                content = chunk["content"]
                st.markdown(
                    f"**Chunk #{chunk['chunk_id']}** — "
                    f"`{chunk['filename']}` — {len(content):,} chars"
                )
                # First 300 characters of the chunk content.
                preview = content[:300]
                st.text(preview + ("..." if len(content) > 300 else ""))
                st.divider()

    # Surface any documents that failed to chunk.
    for message in chunking_errors:
        st.error(message)

    st.divider()


# ---------------------------------------------------------------------------
# Main Area: Indexing Results Panel
# ---------------------------------------------------------------------------

def render_index_results():
    """Show embedding/indexing statistics and a metadata preview.

    Reads the FAISS index and metadata from session_state and displays the
    model name, indexed chunk count, embedding dimension and an estimate of the
    index size, plus a preview of the first few indexed chunks' metadata.
    """
    index = st.session_state.get("faiss_index")
    metadata = st.session_state.get("metadata", [])
    indexing_error = st.session_state.get("indexing_error")

    # Show any error first, even if nothing was indexed.
    if indexing_error:
        st.subheader("Indexing results")
        st.error(f"Embedding/indexing failed: {indexing_error}")
        st.divider()
        return

    # Nothing to show until an index exists.
    if index is None:
        return

    st.subheader("Indexing results")

    # Clear, prominent readiness indicator.
    if st.session_state.get("is_ready_to_chat"):
        st.success("✅ Ready to Chat")

    num_indexed = index.ntotal
    dimension = st.session_state.get("embedding_dim") or index.d
    # Estimate of the raw vector storage: vectors * dim * 4 bytes (float32).
    index_bytes = num_indexed * dimension * 4

    # Four-column statistics row.
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Embedding model", EMBEDDING_MODEL_NAME.split("/")[-1])
    with col2:
        st.metric("Indexed chunks", f"{num_indexed:,}")
    with col3:
        st.metric("Embedding dimension", dimension)
    with col4:
        st.metric("Index size", format_size(index_bytes))

    st.caption(f"Model: `{EMBEDDING_MODEL_NAME}` · FAISS IndexFlatL2")

    # Expandable preview of the first N indexed chunks' metadata.
    if metadata:
        with st.expander(
            f"Preview first {min(INDEX_PREVIEW_COUNT, len(metadata))} "
            "indexed chunks",
            expanded=False,
        ):
            for item in metadata[:INDEX_PREVIEW_COUNT]:
                content = item["content"]
                st.markdown(
                    f"**Chunk #{item['chunk_id']}** — `{item['filename']}` "
                    f"({item['file_type'].upper()}) — {len(content):,} chars"
                )
                preview = content[:300]
                st.text(preview + ("..." if len(content) > 300 else ""))
                st.divider()

    st.divider()


# ---------------------------------------------------------------------------
# Main Area: Suggested Actions
# ---------------------------------------------------------------------------

def render_suggestions():
    """Render clickable suggestion buttons under the welcome message.

    A clicked suggestion is buffered into `pending_prompt` and processed on
    the next rerun, so it flows through the exact same path as typed input.
    """
    st.markdown("**Try asking:**")

    # Distribute suggestion buttons evenly across columns.
    cols = st.columns(len(SUGGESTIONS))
    for col, suggestion in zip(cols, SUGGESTIONS):
        with col:
            if st.button(suggestion, use_container_width=True, key=f"sugg_{suggestion}"):
                st.session_state.pending_prompt = suggestion
                st.rerun()

    st.divider()


# ---------------------------------------------------------------------------
# Main Area: Chat Interface
# ---------------------------------------------------------------------------

# Maps confidence labels to the Streamlit status helper used to render them.
CONFIDENCE_RENDERERS = {
    "High Confidence": lambda msg: st.success(msg),
    "Medium Confidence": lambda msg: st.warning(msg),
    "Low Confidence": lambda msg: st.error(msg),
}


def render_confidence(confidence: dict) -> None:
    """Render the confidence indicator for an answer."""
    label = confidence["label"]
    score = confidence["score"]
    renderer = CONFIDENCE_RENDERERS.get(label, st.info)
    renderer(f"{label} · similarity {score:.2f}")


def render_diagnostics(diagnostics: dict) -> None:
    """Render retrieval diagnostics (counts and timings) for an answer."""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Retrieved chunks", diagnostics["retrieved_count"])
    with col2:
        st.metric("Query time", f"{diagnostics['query_time_ms']:.1f} ms")
    with col3:
        st.metric("FAISS search", f"{diagnostics['faiss_time_ms']:.1f} ms")


def render_sources(sources: list[dict]) -> None:
    """Render the expandable "View Sources" section with improved formatting.

    The number of previews shown is controlled by ``num_source_previews``.
    """
    preview_count = int(
        st.session_state.get("num_source_previews", DEFAULT_SOURCE_PREVIEWS)
    )
    shown = sources[:preview_count]
    with st.expander(f"View Sources ({len(shown)} of {len(sources)})", expanded=False):
        for chunk in shown:
            content = chunk["content"]
            # Improved, multi-line source display.
            st.markdown(f"**{chunk['filename']}**")
            st.markdown(f"Chunk {chunk['chunk_id']}")
            st.markdown(f"Relevance Rank: #{chunk.get('rank', '?')}")
            preview = content[:SOURCE_PREVIEW_CHARS]
            st.text(
                preview + ("..." if len(content) > SOURCE_PREVIEW_CHARS else "")
            )
            st.divider()


def render_chat():
    """Render the full chat history using Streamlit's chat components.

    Assistant messages may also carry a confidence indicator, retrieval
    diagnostics and an expandable "View Sources" section. All of these are
    optional, so a future generative answer can omit them without breaking.
    """
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Confidence indicator (assistant answers with retrieval only).
            confidence = message.get("confidence")
            if confidence:
                render_confidence(confidence)

            # Retrieval diagnostics.
            diagnostics = message.get("diagnostics")
            if diagnostics:
                render_diagnostics(diagnostics)

            # Retrieved sources with improved display.
            sources = message.get("sources")
            if sources:
                render_sources(sources)


# ---------------------------------------------------------------------------
# Application Entry Point
# ---------------------------------------------------------------------------

def main():
    # Ensure all session_state keys exist before anything reads them.
    init_session_state()

    # Sidebar (upload + process) returns the current list of uploaded files.
    uploaded_files = render_sidebar()

    # Header / title block.
    render_header()

    # Optional document info panel (only when files are present).
    render_document_info(uploaded_files)

    # Extraction results (only after the user clicks "Process Documents").
    render_extraction_results()

    # Chunking results / statistics (only after processing).
    render_chunk_results()

    # Embedding / FAISS indexing results (only after indexing).
    render_index_results()

    # Suggested quick actions.
    render_suggestions()

    # Render chat history so far.
    render_chat()

    # Handle a pending suggestion click (set on the previous rerun).
    if st.session_state.pending_prompt:
        prompt = st.session_state.pending_prompt
        st.session_state.pending_prompt = None
        handle_user_message(prompt)
        st.rerun()

    # Chat input pinned to the bottom of the page.
    user_input = st.chat_input("Ask a question about your documents...")
    if user_input:
        handle_user_message(user_input)
        st.rerun()

    # Example query buttons below the chat input. Clicking buffers the query
    # into ``pending_prompt`` so it flows through the same handler on rerun.
    st.caption("Example queries:")
    example_cols = st.columns(len(EXAMPLE_QUERIES))
    for col, query in zip(example_cols, EXAMPLE_QUERIES):
        with col:
            if st.button(query, use_container_width=True, key=f"ex_{query}"):
                st.session_state.pending_prompt = query
                st.rerun()


if __name__ == "__main__":
    main()
