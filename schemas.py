"""Pydantic request and response models for the REST API."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    doc_names: list[str] | None = Field(default=None, description="Limit search to these documents")
    history: list[HistoryMessage] = Field(default_factory=list, max_length=20)
    top_k: int | None = Field(default=None, ge=1, le=20)


class SearchRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    doc_names: list[str] | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)


class SourceOut(BaseModel):
    chunk_id: str
    document: str
    page: int
    score: float
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceOut]
    latency_ms: int


class SearchResponse(BaseModel):
    results: list[SourceOut]


class DocumentInfo(BaseModel):
    name: str
    chunks: int
    pages: int


class UploadResponse(BaseModel):
    document: str
    pages: int
    chunks: int
