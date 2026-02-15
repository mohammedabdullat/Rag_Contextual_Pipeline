"""
Unified Retriever — orchestrates all three retrieval methods and
optionally fuses results using Reciprocal Rank Fusion (RRF).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Literal

from src.indexing.bm25_index import BM25Index
from src.indexing.embedding_index import EmbeddingIndex
from src.indexing.tfidf_index import TFIDFIndex

logger = logging.getLogger(__name__)

RetrievalMethod = Literal["contextual", "bm25", "tfidf", "hybrid"]


@dataclass
class RetrievalResult:
    """Holds results and timing for all retrieval methods for a single query."""
    query: str
    top_k: int
    contextual: list[dict] = field(default_factory=list)
    bm25: list[dict] = field(default_factory=list)
    tfidf: list[dict] = field(default_factory=list)
    hybrid: list[dict] = field(default_factory=list)
    latency_ms: dict[str, float] = field(default_factory=dict)


class Retriever:
    """
    Unified retriever that queries all three indices and returns structured results.

    Supports:
      - Individual method queries (contextual / bm25 / tfidf)
      - Hybrid query via Reciprocal Rank Fusion
    """

    # RRF constant — controls rank penalty. Typical value: 60
    _RRF_K = 60

    def __init__(
        self,
        embedding_index: EmbeddingIndex,
        bm25_index: BM25Index,
        tfidf_index: TFIDFIndex,
    ) -> None:
        self._embedding = embedding_index
        self._bm25 = bm25_index
        self._tfidf = tfidf_index

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        """
        Run all three retrieval methods and return a unified RetrievalResult.

        Timing is captured per method so latency benchmarks are accurate.
        """
        result = RetrievalResult(query=query, top_k=top_k)

        # --- Contextual Embedding ---
        t0 = time.perf_counter()
        result.contextual = self._embedding.query(query, top_k=top_k)
        result.latency_ms["contextual"] = round((time.perf_counter() - t0) * 1000, 2)

        # --- BM25 ---
        t0 = time.perf_counter()
        result.bm25 = self._bm25.query(query, top_k=top_k)
        result.latency_ms["bm25"] = round((time.perf_counter() - t0) * 1000, 2)

        # --- TF-IDF ---
        t0 = time.perf_counter()
        result.tfidf = self._tfidf.query(query, top_k=top_k)
        result.latency_ms["tfidf"] = round((time.perf_counter() - t0) * 1000, 2)

        # --- Hybrid (RRF) ---
        t0 = time.perf_counter()
        result.hybrid = self._reciprocal_rank_fusion(
            [result.contextual, result.bm25, result.tfidf], top_k=top_k
        )
        result.latency_ms["hybrid"] = round((time.perf_counter() - t0) * 1000, 2)

        logger.debug(
            "Retrieval latencies — contextual: %.1fms  bm25: %.1fms  tfidf: %.1fms",
            result.latency_ms["contextual"],
            result.latency_ms["bm25"],
            result.latency_ms["tfidf"],
        )
        return result

    def get_context_text(
        self,
        result: RetrievalResult,
        method: RetrievalMethod = "hybrid",
    ) -> str:
        """
        Return retrieved chunk texts concatenated as a context string
        for LLM answer generation.
        """
        chunks = getattr(result, method, result.hybrid)
        parts = []
        for i, chunk in enumerate(chunks, 1):
            parts.append(
                f"[Source {i} | Page {chunk.get('page_number', '?')} | "
                f"Score: {chunk.get('score', 0):.3f}]\n{chunk['text']}"
            )
        return "\n\n---\n\n".join(parts)

    # ------------------------------------------------------------------ #
    # Reciprocal Rank Fusion                                               #
    # ------------------------------------------------------------------ #

    def _reciprocal_rank_fusion(
        self,
        ranked_lists: list[list[dict]],
        top_k: int,
    ) -> list[dict]:
        """
        Merge multiple ranked lists using RRF.
        RRF score = Σ 1 / (k + rank_i)

        Deduplicates by chunk_id and picks the highest-scoring text/metadata.
        """
        rrf_scores: dict[str, float] = {}
        chunk_data: dict[str, dict] = {}

        for ranked_list in ranked_lists:
            for rank, item in enumerate(ranked_list, start=1):
                cid = item["chunk_id"]
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self._RRF_K + rank)
                if cid not in chunk_data:
                    chunk_data[cid] = item

        sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]
        results = []
        for cid in sorted_ids:
            item = dict(chunk_data[cid])
            item["score"] = round(rrf_scores[cid], 6)
            results.append(item)

        return results
