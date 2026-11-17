"""FastAPI service exposing document management, semantic search, and Q&A."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from urllib.parse import unquote

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import (
    DocumentInfo,
    QueryRequest,
    QueryResponse,
    SearchRequest,
    SearchResponse,
    SourceOut,
    UploadResponse,
)
from core.config import get_settings
from core.rag import RAGService, RetrievedChunk

logger = logging.getLogger("docs_assistant")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rag = RAGService()
    yield


app = FastAPI(
    title="AI Engineering Documentation Assistant API",
    version="1.0.0",
    description="RAG over uploaded engineering PDFs with source-grounded answers.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_rag(request: Request) -> RAGService:
    return request.app.state.rag


def _to_source(c: RetrievedChunk) -> SourceOut:
    return SourceOut(
        chunk_id=c.chunk_id,
        document=c.doc_name,
        page=c.page,
        score=c.score,
        snippet=c.text[:300],
    )


@app.get("/health")
def health(rag: RAGService = Depends(get_rag)):
    return {
        "status": "ok",
        "indexed_chunks": len(rag.store.chunks),
        "llm_provider": get_settings().llm_provider,
    }


@app.get("/documents", response_model=list[DocumentInfo])
def list_documents(rag: RAGService = Depends(get_rag)):
    return rag.list_documents()


@app.post("/documents", response_model=UploadResponse, status_code=201)
def upload_document(file: UploadFile = File(...), rag: RAGService = Depends(get_rag)):
    limit = get_settings().upload_max_mb * 1024 * 1024
    data = file.file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(status_code=413, detail=f"File exceeds {get_settings().upload_max_mb} MB limit")
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if not data.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="File does not look like a PDF")

    try:
        return rag.ingest_pdf(file.filename or "upload.pdf", data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.delete("/documents/{doc_name}")
def delete_document(doc_name: str, rag: RAGService = Depends(get_rag)):
    removed = rag.delete_document(unquote(doc_name))
    if removed == 0:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"document": doc_name, "chunks_removed": removed}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest, rag: RAGService = Depends(get_rag)):
    results = rag.retrieve(
        req.question,
        top_k=req.top_k,
        doc_names=set(req.doc_names) if req.doc_names else None,
    )
    return SearchResponse(results=[_to_source(r) for r in results])


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, rag: RAGService = Depends(get_rag)):
    try:
        result = rag.answer(
            req.question,
            history=[m.model_dump() for m in req.history],
            top_k=req.top_k,
            doc_names=set(req.doc_names) if req.doc_names else None,
        )
    except RuntimeError as exc:  # missing API key or similar configuration issue
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception:
        logger.exception("LLM call failed")
        raise HTTPException(status_code=502, detail="The language model request failed. Check server logs.")

    return QueryResponse(
        answer=result.answer,
        sources=[_to_source(s) for s in result.sources],
        latency_ms=result.latency_ms,
    )
