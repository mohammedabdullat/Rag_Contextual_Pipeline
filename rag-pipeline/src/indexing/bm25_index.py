"""
BM25 Index — sparse keyword-based retrieval using rank_bm25.
"""
from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path
from typing import Optional

from rank_bm25 import BM25Okapi

from src.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)


class BM25Index:
    """
    Wraps BM25Okapi for fast sparse keyword retrieval over document chunks.

    Tokenization: lowercase + alphanumeric split (simple but effective).
    Persists index to disk for fast reload between runs.
    """

    _INDEX_FILENAME = "bm25_index.pkl"

    def __init__(self, persist_dir: Optional[str] = None) -> None:
        self._bm25: Optional[BM25Okapi] = None
        self._chunks: list[Chunk] = []
        self._persist_path = (
            Path(persist_dir) / self._INDEX_FILENAME if persist_dir else None
        )

    # ------------------------------------------------------------------ #
    # Indexing                                                             #
    # ------------------------------------------------------------------ #

    def build(self, chunks: list[Chunk], overwrite: bool = False) -> None:
        """Build or reload BM25 index from chunks."""
        if self._persist_path and self._persist_path.exists() and not overwrite:
            self._load()
            if self._chunks:
                logger.info("BM25 index loaded from disk (%d chunks)", len(self._chunks))
                return

        logger.info("Building BM25 index over %d chunks...", len(chunks))
        self._chunks = chunks
        tokenized = [self._tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(tokenized)

        if self._persist_path:
            self._save()

        logger.info("BM25 index built successfully")

    # ------------------------------------------------------------------ #
    # Retrieval                                                            #
    # ------------------------------------------------------------------ #

    def query(self, query_text: str, top_k: int = 5) -> list[dict]:
        """
        Returns top-K chunks by BM25 score.

        Each result dict: {chunk_id, text, score, page_number, strategy}
        """
        if self._bm25 is None:
            raise RuntimeError("BM25 index not built. Call build() first.")

        query_tokens = self._tokenize(query_text)
        scores = self._bm25.get_scores(query_tokens)

        # Get indices of top-K scores (descending)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            chunk = self._chunks[idx]
            results.append(
                {
                    "chunk_id": chunk.id,
                    "text": chunk.text,
                    "score": round(float(scores[idx]), 4),
                    "page_number": chunk.page_number,
                    "strategy": chunk.strategy,
                }
            )
        return results

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def _save(self) -> None:
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._persist_path, "wb") as f:
            pickle.dump({"bm25": self._bm25, "chunks": self._chunks}, f)
        logger.debug("BM25 index saved to %s", self._persist_path)

    def _load(self) -> None:
        with open(self._persist_path, "rb") as f:
            data = pickle.load(f)
        self._bm25 = data["bm25"]
        self._chunks = data["chunks"]

    # ------------------------------------------------------------------ #
    # Tokenizer                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple lowercase alphanumeric tokenizer."""
        return re.findall(r"[a-z0-9]+", text.lower())
