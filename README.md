# Dispatch RAG Demo (FastAPI + FAISS + Hugging Face)

A minimal, open-source friendly **Document Q&A (RAG) demo** that shows how to:
- Ingest documents (.txt, .md, optional .pdf)
- Build a vector index with **sentence-transformers** + **FAISS**
- Retrieve top passages and answer questions with **Hugging Face** QA pipeline
- Run a simple **FastAPI** backend, with **Docker** + `requirements.txt` for reproducibility

> This is a **demo** extracted from a larger, private project. It demonstrates coding style, ML integration, and infra hygiene without disclosing proprietary code.

---

## Features
- 🔎 **RAG pipeline**: embeddings (SentenceTransformers) + FAISS retrieval
- ❓ **Question Answering**: extractive QA using `deepset/roberta-base-squad2`
- 📄 **Document ingestion**: .txt, .md (basic), and PDF (optional with `pypdf`)
- 🚀 **FastAPI** server with two routes: `/ingest` and `/query`
- 🐳 **Dockerfile** provided for containerized serving

---

## API
### 1) Ingest documents
`POST /ingest` — form-data with one or more files:

```
curl -X POST "http://localhost:8000/ingest"   -F "files=@sample_data/sample.txt"
```

### 2) Ask a question
`POST /query` — JSON body with `question`:

```
curl -X POST "http://localhost:8000/query"   -H "Content-Type: application/json"   -d '{"question": "What does this repo demonstrate?"}'
```

### 3) Reset index (optional)
`DELETE /reset`

---

## Quickstart (Local)

```bash
# 1) Create a venv and install deps
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2) Run the API
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Open docs at: http://localhost:8000/docs

---

## Docker

```bash
# Build
docker build -t dispatch-rag-demo .

# Run
docker run -p 8000:8000 dispatch-rag-demo
```

---

## Project Structure
```
dispatch-rag-demo/
│── app.py
│── requirements.txt
│── Dockerfile
│── README.md
│── .gitignore
│── data/                 # FAISS index + metadata
└── sample_data/
    └── sample.txt
```

---

## Notes
- This demo uses CPU-friendly defaults. For GPU, install torch with CUDA wheels and set `device=0` where appropriate.
- PDF extraction requires `pypdf`. If not installed, PDF files will be skipped with a warning.
- For a gen-AI answer style, swap extractive QA for a small seq2seq model (e.g., `google/flan-t5-base`) and build a prompt from retrieved chunks.
