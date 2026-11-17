"""Retrieval evaluation: Hit@K and MRR against a labeled question set.

Eval file format (JSON list):
    [{"question": "...", "document": "manual.pdf", "page": 12}, ...]

A retrieved chunk counts as relevant when it comes from the labeled document
and page. Only retrieval is scored here, so no LLM API key is needed.

Usage:
    python -m evaluation.evaluate --eval-set evaluation/eval_set.json --k 4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.rag import RAGService

RESULTS_PATH = Path(__file__).resolve().parent / "results.json"


def evaluate(rag: RAGService, examples: list[dict], k: int) -> dict:
    hits = 0
    reciprocal_sum = 0.0
    rows = []
    for ex in examples:
        retrieved = rag.retrieve(ex["question"], top_k=k)
        rank = next(
            (i for i, r in enumerate(retrieved, start=1) if r.doc_name == ex["document"] and r.page == ex["page"]),
            None,
        )
        if rank is not None:
            hits += 1
            reciprocal_sum += 1.0 / rank
        rows.append({"question": ex["question"], "expected": f"{ex['document']} p{ex['page']}", "rank": rank})

    n = len(examples)
    return {
        "k": k,
        "num_questions": n,
        f"hit_rate@{k}": round(hits / n, 4) if n else 0.0,
        f"mrr@{k}": round(reciprocal_sum / n, 4) if n else 0.0,
        "misses": [r for r in rows if r["rank"] is None],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality.")
    parser.add_argument("--eval-set", default="evaluation/eval_set.json")
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args()

    eval_path = Path(args.eval_set)
    if not eval_path.exists():
        raise SystemExit(
            f"{eval_path} not found. Copy evaluation/eval_set.example.json, "
            "then write questions about your own indexed PDFs."
        )

    examples = json.loads(eval_path.read_text(encoding="utf-8"))
    report = evaluate(RAGService(), examples, args.k)

    print(json.dumps({k: v for k, v in report.items() if k != "misses"}, indent=2))
    if report["misses"]:
        print("\nMisses (question not answered from the labeled page within top-k):")
        for m in report["misses"]:
            print(f"  - {m['question']}  [expected {m['expected']}]")

    RESULTS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nFull report written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
