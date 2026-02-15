"""
Pydantic models for API request/response validation.
"""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────────────────────
# Request
# ──────────────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    q: str = Field(..., min_length=1, max_length=2000, description="Query string")
    k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    method: str = Field(
        default="hybrid",
        description="Retrieval method: contextual | bm25 | tfidf | hybrid",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"q": "What is the main contribution of this paper?", "k": 5, "method": "hybrid"}
            ]
        }
    }


# ──────────────────────────────────────────────────────────────────────────────
# Response
# ──────────────────────────────────────────────────────────────────────────────

class ChunkSource(BaseModel):
    chunk_id: str
    text: str
    score: float
    page_number: Optional[int] = None
    strategy: Optional[str] = None


class SourcesBreakdown(BaseModel):
    contextual: list[ChunkSource]
    bm25: list[ChunkSource]
    tfidf: list[ChunkSource]
    hybrid: list[ChunkSource]


class LatencyBreakdown(BaseModel):
    contextual_ms: float
    bm25_ms: float
    tfidf_ms: float
    hybrid_ms: float
    total_ms: float


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: SourcesBreakdown
    latency: LatencyBreakdown
    retrieval_method_used: str


# ──────────────────────────────────────────────────────────────────────────────
# Health check
# ──────────────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    indexed_chunks: int
