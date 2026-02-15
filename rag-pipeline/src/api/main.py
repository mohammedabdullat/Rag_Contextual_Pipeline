"""
FastAPI application — exposes the Contextual RAG Pipeline as a REST API.

Endpoints:
  GET  /health       — health check + index stats
  POST /query        — main query endpoint
  POST /index/build  — (re)build all indices from PDF

OpenAPI/Groq docs auto-available at: http://localhost:8000/docs
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.models import (
    HealthResponse,
    ChunkSource,
    LatencyBreakdown,
    QueryRequest,
    QueryResponse,
    SourcesBreakdown,
)
from src.config import get_settings
from src.contextual.context_generator import ContextGenerator
from src.generation.answer_generator import AnswerGenerator
from src.indexing.bm25_index import BM25Index
from src.indexing.embedding_index import EmbeddingIndex
from src.indexing.tfidf_index import TFIDFIndex
from src.ingestion.chunker import get_all_chunks
from src.ingestion.pdf_parser import PDFParser
from src.retrieval.retriever import Retriever

logger = logging.getLogger(__name__)
settings = get_settings()

# ──────────────────────────────────────────────────────────────────────────────
# App-level singletons (initialised at startup)
# ──────────────────────────────────────────────────────────────────────────────

_retriever: Optional[Retriever] = None
_generator: Optional[AnswerGenerator] = None
_embedding_index: Optional[EmbeddingIndex] = None

def _indices_exist() -> bool:
    """Check whether all three persisted indices are present on disk."""
    persist = Path(settings.chroma_persist_dir)
    chroma_ok = (persist / "chroma.sqlite3").exists()
    bm25_ok   = (persist / "bm25_index.pkl").exists()
    tfidf_ok  = (persist / "tfidf_index.pkl").exists()
    return chroma_ok and bm25_ok and tfidf_ok

def _load_pipeline() -> tuple[Retriever, int]:
    """
    Fast path — load already-built indices from disk without re-parsing
    the PDF or re-running the LLM context generator.

    Requires that _build_pipeline() (or scripts/build_index.py) was run
    at least once before.
    """
    global _retriever, _generator, _embedding_index

    logger.info("Loading pre-built indices from '%s'...", settings.chroma_persist_dir)

    # ChromaDB loads from its persist directory automatically
    embed_idx = EmbeddingIndex(chroma_path=settings.chroma_persist_dir)

    # BM25 + TF-IDF reload from their pickle files — pass empty chunks list;
    # build() detects the existing pickle and skips rebuilding.
    bm25_idx = BM25Index(persist_dir=settings.chroma_persist_dir)
    bm25_idx.build(chunks=[], overwrite=False)   # loads from bm25_index.pkl

    tfidf_idx = TFIDFIndex(persist_dir=settings.chroma_persist_dir)
    tfidf_idx.build(chunks=[], overwrite=False)  # loads from tfidf_index.pkl

    _embedding_index = embed_idx
    _retriever = Retriever(embed_idx, bm25_idx, tfidf_idx)
    _generator = AnswerGenerator()

    chunk_count = embed_idx._collection.count()
    logger.info("Pipeline loaded. %d chunks in ChromaDB.", chunk_count)
    return _retriever, chunk_count

def _build_pipeline(overwrite: bool = False) -> tuple[Retriever, int]:
    """Parse PDF, chunk, enrich with context, build all three indices."""
    global _retriever, _generator, _embedding_index

    # 1. Parse PDF
    parser = PDFParser(settings.pdf_path)
    pages = parser.parse()
    full_doc = parser.get_full_text()

    # 2. Chunk (both strategies; use fixed for indexing, semantic also stored)
    fixed_chunks, semantic_chunks = get_all_chunks(
        pages,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    all_chunks = fixed_chunks + semantic_chunks

    # 3. Contextual enrichment (Anthropic technique)
    ctx_gen = ContextGenerator()
    ctx_gen.enrich_chunks(fixed_chunks, full_doc)

    # 4. Build indices
    embed_idx = EmbeddingIndex(chroma_path=settings.chroma_persist_dir)
    embed_idx.index_chunks(fixed_chunks, overwrite=overwrite)

    bm25_idx = BM25Index(persist_dir=settings.chroma_persist_dir)
    bm25_idx.build(fixed_chunks, overwrite=overwrite)

    tfidf_idx = TFIDFIndex(persist_dir=settings.chroma_persist_dir)
    tfidf_idx.build(fixed_chunks, overwrite=overwrite)

    # 5. Compose retriever + generator
    _embedding_index = embed_idx
    _retriever = Retriever(embed_idx, bm25_idx, tfidf_idx)
    _generator = AnswerGenerator()

    logger.info("Pipeline ready. Total chunks: %d", len(fixed_chunks))
    return _retriever, len(fixed_chunks)


# ──────────────────────────────────────────────────────────────────────────────
# Lifespan
# ──────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    At startup:
      - If all indices already exist on disk → fast load (no LLM calls, no parsing)
      - If PDF exists but indices are missing  → full build
      - Otherwise → warn and wait for POST /index/build
    """
    if _indices_exist():
        try:
            _load_pipeline()
            logger.info("Startup: loaded pre-built indices.")
        except Exception as exc:
            logger.warning("Startup: index load failed, will try full build. %s", exc)
            _build_pipeline(overwrite=True)
    elif Path(settings.pdf_path).exists():
        try:
            _build_pipeline(overwrite=False)
        except Exception as exc:
            logger.warning("Pipeline startup build failed (non-fatal): %s", exc)
    else:
        logger.warning(
            "PDF not found at '%s'. Use POST /index/build after placing the PDF.",
            settings.pdf_path,
        )
    yield


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.app_title,
    version=settings.app_version,
    description=(
        "Contextual Retrieval-Augmented Generation Pipeline. "
        "Implements Anthropic's Contextual Retrieval technique with "
        "BM25, TF-IDF, and dense embedding baselines."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Health check — returns index status."""
    count = 0
    if _embedding_index is not None:
        try:
            count = _embedding_index._collection.count()
        except Exception:
            pass
    return HealthResponse(
        status="ok" if _retriever is not None else "not_indexed",
        version=settings.app_version,
        indexed_chunks=count,
    )


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query(request: QueryRequest):
    """
    Main RAG query endpoint.

    Retrieves top-K chunks using all three methods, generates an answer
    using the chosen method's context, and returns sources + latency breakdown.
    """
    if _retriever is None or _generator is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not ready. POST to /index/build first.",
        )

    t_total_start = time.perf_counter()

    # Retrieve
    retrieval_result = _retriever.retrieve(request.q, top_k=request.k)

    # Generate answer from chosen method's context
    context = _retriever.get_context_text(retrieval_result, method=request.method)
    answer = _generator.generate(request.q, context)

    total_ms = round((time.perf_counter() - t_total_start) * 1000, 2)

    def _to_sources(chunks: list[dict]) -> list[ChunkSource]:
        return [ChunkSource(**c) for c in chunks]

    return QueryResponse(
        query=request.q,
        answer=answer,
        sources=SourcesBreakdown(
            contextual=_to_sources(retrieval_result.contextual),
            bm25=_to_sources(retrieval_result.bm25),
            tfidf=_to_sources(retrieval_result.tfidf),
            hybrid=_to_sources(retrieval_result.hybrid),
        ),
        latency=LatencyBreakdown(
            contextual_ms=retrieval_result.latency_ms.get("contextual", 0),
            bm25_ms=retrieval_result.latency_ms.get("bm25", 0),
            tfidf_ms=retrieval_result.latency_ms.get("tfidf", 0),
            hybrid_ms=retrieval_result.latency_ms.get("hybrid", 0),
            total_ms=total_ms,
        ),
        retrieval_method_used=request.method,
    )


@app.post("/index/build", tags=["System"])
async def build_index(overwrite: bool = False):
    """
    (Re)build all indices from the configured PDF.

    Set overwrite=true to force a full rebuild even if indices exist.
    """
    pdf = Path(settings.pdf_path)
    if not pdf.exists():
        raise HTTPException(
            status_code=404,
            detail=f"PDF not found at '{settings.pdf_path}'. Place your PDF there first.",
        )
    try:
        _, chunk_count = _build_pipeline(overwrite=overwrite)
        return {"status": "success", "indexed_chunks": chunk_count}
    except Exception as exc:
        logger.exception("Index build failed")
        raise HTTPException(status_code=500, detail=str(exc))
