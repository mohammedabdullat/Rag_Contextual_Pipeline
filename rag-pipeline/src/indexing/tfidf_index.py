"""
TF-IDF Index — IR baseline using scikit-learn's TfidfVectorizer.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)


class TFIDFIndex:
    """
    TF-IDF retrieval index using scikit-learn.

    Vectorizes all chunk texts at build time, then scores queries
    via cosine similarity against the TF-IDF matrix.
    """

    _INDEX_FILENAME = "tfidf_index.pkl"

    def __init__(self, persist_dir: Optional[str] = None) -> None:
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._matrix = None  # sparse matrix: (n_chunks, vocab_size)
        self._chunks: list[Chunk] = []
        self._persist_path = (
            Path(persist_dir) / self._INDEX_FILENAME if persist_dir else None
        )

    # ------------------------------------------------------------------ #
    # Indexing                                                             #
    # ------------------------------------------------------------------ #

    def build(self, chunks: list[Chunk], overwrite: bool = False) -> None:
        """Build or reload TF-IDF index from chunks."""
        if self._persist_path and self._persist_path.exists() and not overwrite:
            self._load()
            if self._chunks:
                logger.info("TF-IDF index loaded from disk (%d chunks)", len(self._chunks))
                return

        logger.info("Building TF-IDF index over %d chunks...", len(chunks))
        self._chunks = chunks
        texts = [c.text for c in chunks]

        self._vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
        )
        self._matrix = self._vectorizer.fit_transform(texts)

        if self._persist_path:
            self._save()

        logger.info("TF-IDF index built. Vocab size: %d", len(self._vectorizer.vocabulary_))

    # ------------------------------------------------------------------ #
    # Retrieval                                                            #
    # ------------------------------------------------------------------ #

    def query(self, query_text: str, top_k: int = 5) -> list[dict]:
        """
        Returns top-K chunks by TF-IDF cosine similarity.

        Each result dict: {chunk_id, text, score, page_number, strategy}
        """
        if self._vectorizer is None or self._matrix is None:
            raise RuntimeError("TF-IDF index not built. Call build() first.")

        query_vec = self._vectorizer.transform([query_text])
        scores = cosine_similarity(query_vec, self._matrix).flatten()

        top_indices = np.argsort(scores)[::-1][:top_k]

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
            pickle.dump(
                {
                    "vectorizer": self._vectorizer,
                    "matrix": self._matrix,
                    "chunks": self._chunks,
                },
                f,
            )
        logger.debug("TF-IDF index saved to %s", self._persist_path)

    def _load(self) -> None:
        with open(self._persist_path, "rb") as f:
            data = pickle.load(f)
        self._vectorizer = data["vectorizer"]
        self._matrix = data["matrix"]
        self._chunks = data["chunks"]
