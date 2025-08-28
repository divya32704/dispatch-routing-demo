import os
import io
import pickle
from typing import List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss

# Optional PDF support
try:
    from pypdf import PdfReader
    HAS_PDF = True
except Exception:
    HAS_PDF = False

DATA_DIR = "data"
INDEX_PATH = os.path.join(DATA_DIR, "index.faiss")
META_PATH = os.path.join(DATA_DIR, "metadata.pkl")

# Initialize FastAPI
app = FastAPI(title="Dispatch RAG Demo", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy globals
_embeddings_model: Optional[SentenceTransformer] = None
_qa_pipeline = None
_index: Optional[faiss.IndexFlatIP] = None
_chunks: List[str] = []

# ---------------
# Utilities
# ---------------
def get_embeddings_model() -> SentenceTransformer:
    global _embeddings_model
    if _embeddings_model is None:
        # Small, widely available model
        _embeddings_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _embeddings_model

def embed_texts(texts: List[str]) -> np.ndarray:
    model = get_embeddings_model()
    vecs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return vecs.astype("float32")

def split_into_chunks(text: str, chunk_size: int = 600, overlap: int = 100) -> List[str]:
    # Simple token-agnostic chunking by characters
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
        if start >= len(text):
            break
    return [c.strip() for c in chunks if c.strip()]

def read_txt_or_md(content: bytes) -> str:
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return content.decode("latin-1", errors="ignore")

def read_pdf(content: bytes) -> str:
    if not HAS_PDF:
        return ""
    with io.BytesIO(content) as f:
        reader = PdfReader(f)
        texts = []
        for page in reader.pages:
            try:
                texts.append(page.extract_text() or "")
            except Exception:
                continue
        return "\n".join(texts)

def save_state(index: faiss.IndexFlatIP, chunks: List[str]) -> None:
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump({"chunks": chunks}, f)

def load_state() -> Tuple[Optional[faiss.IndexFlatIP], List[str]]:
    if not (os.path.exists(INDEX_PATH) and os.path.exists(META_PATH)):
        return None, []
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    return index, meta.get("chunks", [])

def get_index() -> Tuple[faiss.IndexFlatIP, List[str]]:
    global _index, _chunks
    if _index is None:
        idx, ch = load_state()
        if idx is None:
            idx = faiss.IndexFlatIP(384)  # all-MiniLM-L6-v2 has 384-dim embeddings
            ch = []
        _index, _chunks = idx, ch
    return _index, _chunks

def get_qa_pipeline():
    global _qa_pipeline
    if _qa_pipeline is None:
        _qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
    return _qa_pipeline

# ---------------
# Schemas
# ---------------
class QueryRequest(BaseModel):
    question: str
    top_k: int = 4

class QueryResponse(BaseModel):
    answer: str
    context_snippets: List[str]

# ---------------
# Routes
# ---------------
@app.post("/ingest")
async def ingest(files: List[UploadFile] = File(...)):
    index, chunks = get_index()
    new_chunks: List[str] = []

    for f in files:
        name = f.filename.lower()
        content = await f.read()

        text = ""
        if name.endswith(".txt") or name.endswith(".md"):
            text = read_txt_or_md(content)
        elif name.endswith(".pdf"):
            text = read_pdf(content)
        else:
            # skip unsupported formats gracefully
            continue

        if not text.strip():
            continue

        doc_chunks = split_into_chunks(text)
        new_chunks.extend(doc_chunks)

    if not new_chunks:
        raise HTTPException(status_code=400, detail="No valid text extracted from uploads.")

    embeddings = embed_texts(new_chunks)
    if index.ntotal == 0:
        # Build new index
        dim = embeddings.shape[1]
        new_index = faiss.IndexFlatIP(dim)
        new_index.add(embeddings)
        _ = new_index.ntotal
        # Replace global index
        globals()["_index"] = new_index
        globals()["_chunks"] = new_chunks
        save_state(new_index, new_chunks)
    else:
        index.add(embeddings)
        chunks.extend(new_chunks)
        save_state(index, chunks)

    return {"ingested_chunks": len(new_chunks), "total_chunks": len(get_index()[1])}

@app.post("/query", response_model=QueryResponse)
def query(body: QueryRequest):
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    index, chunks = get_index()
    if index.ntotal == 0 or not chunks:
        raise HTTPException(status_code=400, detail="Index is empty. Ingest documents first.")

    # Retrieve top_k
    q_emb = embed_texts([question])
    scores, ids = index.search(q_emb, k=min(body.top_k, index.ntotal))
    candidates = [chunks[i] for i in ids[0] if i >= 0]

    # Extractive QA over candidates; pick best
    qa = get_qa_pipeline()
    best = {"score": -1e9, "answer": ""}
    for ctx in candidates:
        try:
            out = qa(question=question, context=ctx)
            if out and out.get("score", 0) > best["score"]:
                best = {"score": out.get("score", 0), "answer": out.get("answer", "")}
        except Exception:
            continue

    answer = best["answer"] or "No answer found in the indexed content."
    return QueryResponse(answer=answer, context_snippets=candidates)

@app.delete("/reset")
def reset():
    globals()["_index"] = None
    globals()["_chunks"] = []
    # remove persisted files
    try:
        if os.path.exists(INDEX_PATH):
            os.remove(INDEX_PATH)
        if os.path.exists(META_PATH):
            os.remove(META_PATH)
    except Exception:
        pass
    return {"status": "reset"}
