# AI Engineering Documentation Assistant (RAG)

A Retrieval-Augmented Generation app that answers questions from uploaded engineering PDFs (manuals, specifications, maintenance guides, technical reports). Answers are grounded in retrieved document excerpts and cite the document name and page number. If the answer isn't in the documents, the assistant says so.

> **Scope:** This is an informational prototype. It has not been validated for safety-critical or airworthiness use. Any use in aerospace or other safety-critical work requires domain validation, security review, and qualified engineering sign-off.

## Architecture

```
            ┌──────────────────────┐
            │  Streamlit frontend  │  upload, document list, chat, citations
            └──────────┬───────────┘
                       │ HTTP (REST)
            ┌──────────▼───────────┐
            │   FastAPI backend    │  /documents  /search  /query  /health
            └──────────┬───────────┘
                       │
   ┌───────────────────▼────────────────────────────┐
   │                 RAG service                    │
   │                                                │
   │  Ingest:  PyMuPDF → page text → chunks         │
   │           → Sentence-Transformers → FAISS      │
   │                                                │
   │  Query:   question → embedding → top-K search  │
   │           → grounded prompt → LLM (LangChain)  │
   │           (Gemini or OpenAI) → answer + sources│
   └────────────────────────────────────────────────┘
```

| Layer | Module | Notes |
|---|---|---|
| Extraction | `core/ingestion.py` | PyMuPDF; each page is chunked separately so page numbers stay exact |
| Chunking | `core/ingestion.py` | ~800 characters with 150 overlap, cut at paragraph, sentence, or word boundaries |
| Embeddings | `core/embeddings.py` | `all-MiniLM-L6-v2`, normalised so inner product equals cosine similarity |
| Vector store | `core/vector_store.py` | FAISS `IndexFlatIP` (exact search), metadata persisted as JSON, per-document filtering and removal |
| Prompting | `core/prompts.py` | Numbered excerpts, citation rules, explicit "not found" behaviour |
| LLM | `core/llm.py` | LangChain chat models; provider switched by `LLM_PROVIDER` |
| Pipeline | `core/rag.py` | Ingest, retrieve, answer; thread-safe writes |
| API | `backend/` | FastAPI with validated schemas, file type and size checks |
| UI | `frontend/app.py` | Streamlit chat with source expanders and document filtering |
| Evaluation | `evaluation/evaluate.py` | Hit@K and MRR on a labelled question set (no LLM calls) |

## Project layout

```
.
├── backend/            FastAPI app and Pydantic schemas
├── core/               RAG pipeline (ingestion, embeddings, vector store, LLM, prompts)
├── frontend/app.py     Streamlit UI
├── evaluation/         Retrieval evaluation script and example eval set
├── tests/              Unit tests for chunking, extraction, and vector search
├── data/index/         Persisted FAISS index and chunk metadata (created at runtime)
├── requirements.txt
├── .env.example
└── pytest.ini
```

## Setup

Requires Python 3.10+. The first run downloads the embedding model (about 90 MB).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # then add your GEMINI_API_KEY (or OPENAI_API_KEY)
```

## Run

Start the backend (terminal 1):

```bash
uvicorn backend.main:app --reload
```

Start the frontend (terminal 2):

```bash
streamlit run frontend/app.py
```

Open http://localhost:8501, upload PDFs in the sidebar, click **Index uploads**, then ask questions. API docs are at http://localhost:8000/docs.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Service status and indexed chunk count |
| GET | `/documents` | List indexed documents with page and chunk counts |
| POST | `/documents` | Upload and index a PDF (multipart field `file`) |
| DELETE | `/documents/{name}` | Remove a document from the index |
| POST | `/search` | Retrieval only: returns the top-K chunks with scores |
| POST | `/query` | Full RAG: returns a grounded answer, sources, and latency |

Example:

```bash
curl -X POST localhost:8000/query -H "Content-Type: application/json" -d '{
  "question": "What torque is specified for the pump mounting bolts?",
  "doc_names": ["hydraulic_pump_manual.pdf"],
  "history": []
}'
```

## Evaluation

Retrieval quality is measured against labelled questions. Each example names the document and page where the answer lives:

1. Copy `evaluation/eval_set.example.json` to `evaluation/eval_set.json`.
2. Replace the examples with 20–50 questions about your own PDFs, with the correct page for each.
3. Run:

```bash
python -m evaluation.evaluate --k 4
```

It reports **Hit@K** (the share of questions where the correct page appears in the top K) and **MRR** (mean reciprocal rank of the correct page), lists the misses, and writes `evaluation/results.json`. Compare settings such as chunk size, overlap, and `TOP_K` by changing the `.env` values and re-running.

## Tests

```bash
pytest
```

The tests cover PDF extraction, page-aware chunking, chunk overlap, FAISS search, document filtering, index persistence, and document removal. They run without API keys and without downloading the embedding model.

## Design decisions

- **Page-aware chunking.** Each page is split on its own, so every chunk carries an exact page number for citations.
- **Cosine similarity via normalised vectors.** Normalising embeddings lets FAISS use a plain inner-product index.
- **Exact search.** `IndexFlatIP` is accurate and fast for up to roughly a few hundred thousand chunks. Switch to an IVF or HNSW index beyond that.
- **Grounded prompt.** The model is told to answer only from numbered excerpts, cite them, and use a fixed phrase when the answer is missing. If retrieval returns nothing, the LLM isn't called at all.
- **Provider abstraction.** `LLM_PROVIDER` switches between Gemini and OpenAI without code changes.
- **Replace on re-upload.** Uploading a file with the same name replaces its old chunks, so the index doesn't accumulate duplicates.

## Known limitations and next steps

- **Scanned PDFs** have no text layer and are rejected. Add OCR (for example Tesseract via `pytesseract`) to handle them.
- **Follow-up questions** are retrieved using only the latest question, so "what about the second one?" retrieves poorly. A query-rewriting step that uses conversation history would fix this.
- **Tables and diagrams** lose structure when extracted as plain text. Table-aware extraction (for example PyMuPDF's `find_tables`) would help with specification sheets.
- **Hybrid search and reranking** are not included yet. Adding BM25 for exact part numbers plus a cross-encoder reranker is the natural next step.
- **Answer quality** is not scored automatically. Adding faithfulness and citation-accuracy checks (for example with RAGAS) would complete the evaluation.
- **Security:** CORS is open and there is no authentication. Restrict both before any shared deployment.
