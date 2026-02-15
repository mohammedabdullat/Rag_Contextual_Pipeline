"""
Contextual Embedding Index — stores contextualized chunk embeddings in ChromaDB.
Uses OpenAI text-embedding-3-small (or any compatible model).
"""
from __future__ import annotations

import logging
from typing import Optional

import chromadb
import numpy as np
from openai import OpenAI
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from src.config import get_settings
from src.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)

COLLECTION_NAME = "contextual_rag"
BATCH_SIZE = 64  # OpenAI embedding batch limit


class EmbeddingIndex:
    """
    Manages a ChromaDB vector collection of contextualized chunk embeddings.

    Responsibilities:
      - Compute embeddings for each chunk's `contextualized_text`
      - Store in persistent ChromaDB collection
      - Query top-K nearest neighbors for a query string
    """

    def __init__(
        self,
        openai_client: Optional[OpenAI] = None,
        chroma_path: Optional[str] = None,
    ) -> None:
        settings = get_settings()
        self._client = openai_client or OpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        )
        self._model = SentenceTransformer("all-MiniLM-L6-v2")
        #self._embed_model = settings.openai_embedding_model
        chroma_path = chroma_path or settings.chroma_persist_dir

        self._chroma = chromadb.PersistentClient(path=chroma_path)
        self._collection = self._chroma.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("EmbeddingIndex initialized. Collection: %s", COLLECTION_NAME)

    # ------------------------------------------------------------------ #
    # Indexing                                                             #
    # ------------------------------------------------------------------ #

    def index_chunks(self, chunks: list[Chunk], overwrite: bool=True) -> None:
        """Embed and store chunks. Skip already-indexed IDs unless overwrite=True."""
        if overwrite:
            self._chroma.delete_collection(COLLECTION_NAME)
            self._collection = self._chroma.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )

        # Filter already-indexed chunks
        existing_ids = set(self._collection.get(include=[])["ids"])
        new_chunks = [c for c in chunks if c.id not in existing_ids]

        if not new_chunks:
            logger.info("All %d chunks already indexed. Skipping.", len(chunks))
            return

        logger.info("Indexing %d new chunks into ChromaDB...", len(new_chunks))

        for batch_start in tqdm(
            range(0, len(new_chunks), BATCH_SIZE), desc="Embedding chunks"
        ):
            batch = new_chunks[batch_start : batch_start + BATCH_SIZE]
            texts = [c.contextualized_text for c in batch]
            embeddings = self._embed(texts)

            self._collection.add(
                ids=[c.id for c in batch],
                embeddings=embeddings,
                documents=[c.text for c in batch],  # store raw text for display
                metadatas=[
                    {
                        "page_number": c.page_number,
                        "strategy": c.strategy,
                        "start_char": c.start_char,
                        "end_char": c.end_char,
                    }
                    for c in batch
                ],
            )

        logger.info("Indexed %d chunks successfully", len(new_chunks))

    # ------------------------------------------------------------------ #
    # Retrieval                                                            #
    # ------------------------------------------------------------------ #

    def query(self, query_text: str, top_k: int = 5) -> list[dict]:
        """
        Returns top-K chunks ordered by cosine similarity.

        Each result dict contains: {chunk_id, text, score, page_number, strategy}
        """
        query_embedding = self._embed([query_text])[0]
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._collection.count() or 1),
            include=["documents", "metadatas", "distances"],
        )

        output: list[dict] = []
        if not results["ids"] or not results["ids"][0]:
            return output

        for chunk_id, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append(
                {
                    "chunk_id": chunk_id,
                    "text": doc,
                    "score": round(1.0 - dist, 4),  # cosine similarity
                    "page_number": meta.get("page_number"),
                    "strategy": meta.get("strategy"),
                }
            )

        return output

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string (used by benchmarker)."""
        return self._embed([text])[0]

    # ------------------------------------------------------------------ #
    # Private                                                              #
    # ------------------------------------------------------------------ #

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Call OpenAI embedding API for a batch of texts."""
        '''
        response = self._client.embeddings.create(
            model=self._embed_model,
            input=texts,
        )
        return [item.embedding for item in response.data]
        '''
        """
        Replaces OpenAI API call with local model inference.
        """
        # .encode() returns numpy arrays; ChromaDB accepts lists of floats
        embeddings = self._model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()
