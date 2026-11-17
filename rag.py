"""RAG pipeline: ingestion, retrieval, and grounded answer generation."""
from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from core.config import Settings, get_settings
from core.embeddings import Embedder
from core.ingestion import build_chunks, extract_pages
from core.llm import build_llm
from core.prompts import SYSTEM_PROMPT, build_user_prompt
from core.vector_store import VectorStore

NO_ANSWER = "I could not find this in the uploaded documents."


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_name: str
    page: int
    text: str
    score: float


@dataclass
class Answer:
    answer: str
    sources: list[RetrievedChunk]
    latency_ms: int


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


class RAGService:
    def __init__(
        self,
        settings: Settings | None = None,
        embedder: Embedder | None = None,
        llm: Any = None,
    ):
        self.settings = settings or get_settings()
        self.embedder = embedder or Embedder(self.settings.embedding_model)
        self.store = VectorStore(self.settings.index_dir, self.embedder.dim)
        self._llm = llm
        self._lock = threading.Lock()  # FastAPI runs sync routes in a thread pool

    @property
    def llm(self):
        if self._llm is None:
            self._llm = build_llm(self.settings)
        return self._llm

    # ---------- documents ----------
    def ingest_pdf(self, filename: str, pdf_bytes: bytes) -> dict:
        doc_name = Path(filename).name
        if not doc_name.lower().endswith(".pdf"):
            raise ValueError("Only PDF files are supported.")

        pages = extract_pages(pdf_bytes)
        chunks = build_chunks(doc_name, pages, self.settings.chunk_size, self.settings.chunk_overlap)
        if not chunks:
            raise ValueError("No extractable text found. Scanned PDFs need OCR before indexing.")

        vectors = self.embedder.embed([c.text for c in chunks])  # slow; done outside the lock
        with self._lock:
            self.store.remove_document(doc_name)  # re-uploading a file replaces the old version
            self.store.add(vectors, [asdict(c) for c in chunks])

        return {"document": doc_name, "pages": len(pages), "chunks": len(chunks)}

    def delete_document(self, doc_name: str) -> int:
        with self._lock:
            return self.store.remove_document(doc_name)

    def list_documents(self) -> list[dict]:
        with self._lock:
            return self.store.list_documents()

    # ---------- retrieval ----------
    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        doc_names: set[str] | None = None,
    ) -> list[RetrievedChunk]:
        k = top_k or self.settings.top_k
        query_vec = self.embedder.embed([question])[0]
        with self._lock:
            hits = self.store.search(query_vec, k, doc_names)
        return [
            RetrievedChunk(
                chunk_id=c["chunk_id"],
                doc_name=c["doc_name"],
                page=c["page"],
                text=c["text"],
                score=round(score, 4),
            )
            for c, score in hits
        ]

    # ---------- generation ----------
    def answer(
        self,
        question: str,
        history: list[dict[str, Any]] | None = None,
        top_k: int | None = None,
        doc_names: set[str] | None = None,
    ) -> Answer:
        started = time.perf_counter()
        sources = self.retrieve(question, top_k=top_k, doc_names=doc_names)
        if not sources:
            return Answer(NO_ANSWER, [], _elapsed_ms(started))

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=build_user_prompt(question, sources, history)),
        ]
        response = self.llm.invoke(messages)
        content = response.content if isinstance(response.content, str) else str(response.content)
        return Answer(content.strip(), sources, _elapsed_ms(started))
