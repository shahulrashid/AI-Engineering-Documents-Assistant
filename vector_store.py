"""FAISS-backed vector store with JSON metadata persistence."""
from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np


class VectorStore:
    """Exact cosine-similarity search (IndexFlatIP over normalised vectors).

    Chunk metadata is kept in a list row-aligned with the FAISS index.
    Both are persisted to disk after every mutation.
    """

    def __init__(self, index_dir: Path, dim: int):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.index_dir / "faiss.index"
        self.meta_path = self.index_dir / "chunks.json"
        self.dim = dim
        self.chunks: list[dict] = []
        self.index = faiss.IndexFlatIP(dim)
        self._load()

    # ---------- persistence ----------
    def _load(self) -> None:
        if self.index_path.exists() and self.meta_path.exists():
            index = faiss.read_index(str(self.index_path))
            if index.d != self.dim:
                raise RuntimeError(
                    f"Stored index dimension {index.d} does not match embedding dimension {self.dim}. "
                    "Delete data/index/ and re-upload your documents."
                )
            self.index = index
            self.chunks = json.loads(self.meta_path.read_text(encoding="utf-8"))

    def _save(self) -> None:
        faiss.write_index(self.index, str(self.index_path))
        self.meta_path.write_text(json.dumps(self.chunks, ensure_ascii=False), encoding="utf-8")

    # ---------- mutations ----------
    def add(self, vectors: np.ndarray, chunks: list[dict]) -> None:
        vectors = np.asarray(vectors, dtype="float32")
        if vectors.ndim != 2 or vectors.shape[1] != self.dim or len(vectors) != len(chunks):
            raise ValueError("vectors must be (n, dim) and match the number of chunks")
        self.index.add(vectors)
        self.chunks.extend(chunks)
        self._save()

    def remove_document(self, doc_name: str) -> int:
        """Remove all chunks belonging to doc_name. Returns the number of chunks removed."""
        keep = [i for i, c in enumerate(self.chunks) if c["doc_name"] != doc_name]
        removed = len(self.chunks) - len(keep)
        if removed == 0:
            return 0

        new_index = faiss.IndexFlatIP(self.dim)
        if keep and self.index.ntotal > 0:
            all_vectors = self.index.reconstruct_n(0, self.index.ntotal)
            new_index.add(np.ascontiguousarray(all_vectors[keep], dtype="float32"))
        self.index = new_index
        self.chunks = [self.chunks[i] for i in keep]
        self._save()
        return removed

    # ---------- queries ----------
    def search(
        self,
        query_vec: np.ndarray,
        top_k: int,
        doc_names: set[str] | None = None,
    ) -> list[tuple[dict, float]]:
        if self.index.ntotal == 0:
            return []
        q = np.asarray(query_vec, dtype="float32").reshape(1, -1)
        # Over-fetch when filtering so enough candidates survive the filter.
        fetch = min(self.index.ntotal, top_k * 10 if doc_names else top_k)
        scores, ids = self.index.search(q, fetch)

        results: list[tuple[dict, float]] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            chunk = self.chunks[idx]
            if doc_names and chunk["doc_name"] not in doc_names:
                continue
            results.append((chunk, float(score)))
            if len(results) == top_k:
                break
        return results

    def list_documents(self) -> list[dict]:
        summary: dict[str, dict] = {}
        for c in self.chunks:
            entry = summary.setdefault(c["doc_name"], {"name": c["doc_name"], "chunks": 0, "pages": set()})
            entry["chunks"] += 1
            entry["pages"].add(c["page"])
        return [
            {"name": d["name"], "chunks": d["chunks"], "pages": len(d["pages"])}
            for d in sorted(summary.values(), key=lambda d: d["name"])
        ]
