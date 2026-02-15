"""
Chunker — implements two chunking strategies over parsed PDF pages.

Strategy A: FixedSizeChunker  — token-window with overlap
Strategy B: SemanticChunker   — paragraph/sentence boundary aware
"""
from __future__ import annotations

import hashlib
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import nltk
import tiktoken

from src.ingestion.pdf_parser import PageContent

logger = logging.getLogger(__name__)

# Download NLTK sentence tokenizer data (silent if already present)
'''
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
'''

# ──────────────────────────────────────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """A single text chunk with provenance metadata."""
    id: str                   # Stable hash-based ID
    text: str                 # Raw chunk text
    contextualized_text: str  # Text with LLM-generated context prepended
    page_number: int          # Source page (1-indexed)
    start_char: int           # Char offset in page combined_text
    end_char: int             # Char offset in page combined_text
    strategy: str             # "fixed" or "semantic"
    metadata: dict = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        text: str,
        page_number: int,
        start_char: int,
        end_char: int,
        strategy: str,
        metadata: Optional[dict] = None,
    ) -> "Chunk":
        chunk_id = hashlib.sha256(
            f"{strategy}:{page_number}:{start_char}:{text[:64]}".encode()
        ).hexdigest()[:16]
        return cls(
            id=chunk_id,
            text=text,
            contextualized_text=text,  # Will be filled by ContextGenerator
            page_number=page_number,
            start_char=start_char,
            end_char=end_char,
            strategy=strategy,
            metadata=metadata or {},
        )


# ──────────────────────────────────────────────────────────────────────────────
# Abstract base
# ──────────────────────────────────────────────────────────────────────────────

class BaseChunker(ABC):
    """Abstract chunker — concrete classes implement `chunk_pages`."""

    @abstractmethod
    def chunk_pages(self, pages: list[PageContent]) -> list[Chunk]:
        ...

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        ...


# ──────────────────────────────────────────────────────────────────────────────
# Strategy A: Fixed-Size Token Chunker
# ──────────────────────────────────────────────────────────────────────────────

class FixedSizeChunker(BaseChunker):
    """
    Splits each page's text into fixed-size token windows with overlap.
    Uses tiktoken for accurate token counting.
    """

    def __init__(self, chunk_size: int = 1500, overlap: int = 200) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._enc = tiktoken.get_encoding("cl100k_base")

    @property
    def strategy_name(self) -> str:
        return "fixed"

    def chunk_pages(self, pages: list[PageContent]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for page in pages:
            text = page.combined_text
            if not text.strip():
                continue
            page_chunks = self._chunk_text(text, page.page_number)
            chunks.extend(page_chunks)
        logger.info("[FixedSizeChunker] Generated %d chunks", len(chunks))
        return chunks

    def _chunk_text(self, text: str, page_number: int) -> list[Chunk]:
        tokens = self._enc.encode(text)
        chunks: list[Chunk] = []
        step = max(1, self.chunk_size - self.overlap)

        for start_tok in range(0, len(tokens), step):
            end_tok = min(start_tok + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start_tok:end_tok]
            chunk_text = self._enc.decode(chunk_tokens)

            # Approximate char offsets
            start_char = len(self._enc.decode(tokens[:start_tok]))
            end_char = start_char + len(chunk_text)

            chunks.append(
                Chunk.build(
                    text=chunk_text,
                    page_number=page_number,
                    start_char=start_char,
                    end_char=end_char,
                    strategy=self.strategy_name,
                    metadata={"token_start": start_tok, "token_end": end_tok},
                )
            )
            if end_tok >= len(tokens):
                break

        return chunks


# ──────────────────────────────────────────────────────────────────────────────
# Strategy B: Semantic / Paragraph Chunker
# ──────────────────────────────────────────────────────────────────────────────

class SemanticChunker(BaseChunker):
    """
    Splits text on paragraph and sentence boundaries to preserve semantic units.

    Algorithm:
      1. Split page text into paragraphs (double-newline boundaries).
      2. Accumulate paragraphs until a token budget is exceeded.
      3. When budget exceeded, flush current buffer as a chunk.
      4. Overflow paragraphs carry over to maintain continuity.
    """

    def __init__(self, max_tokens: int = 400, min_tokens: int = 50) -> None:
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self._enc = tiktoken.get_encoding("cl100k_base")

    @property
    def strategy_name(self) -> str:
        return "semantic"

    def chunk_pages(self, pages: list[PageContent]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for page in pages:
            text = page.combined_text
            if not text.strip():
                continue
            page_chunks = self._chunk_text(text, page.page_number)
            chunks.extend(page_chunks)
        logger.info("[SemanticChunker] Generated %d chunks", len(chunks))
        return chunks

    def _chunk_text(self, text: str, page_number: int) -> list[Chunk]:
        paragraphs = self._split_paragraphs(text)
        chunks: list[Chunk] = []
        buffer: list[str] = []
        buffer_tokens = 0
        char_cursor = 0

        for para in paragraphs:
            para_tokens = len(self._enc.encode(para))

            # If single paragraph exceeds budget, split by sentences
            if para_tokens > self.max_tokens:
                sentences = nltk.sent_tokenize(para)
                for sent in sentences:
                    sent_tokens = len(self._enc.encode(sent))
                    if buffer_tokens + sent_tokens > self.max_tokens and buffer:
                        chunk_text = " ".join(buffer)
                        start_char = char_cursor
                        end_char = start_char + len(chunk_text)
                        chunks.append(
                            Chunk.build(
                                text=chunk_text,
                                page_number=page_number,
                                start_char=start_char,
                                end_char=end_char,
                                strategy=self.strategy_name,
                            )
                        )
                        char_cursor = end_char + 1
                        buffer = []
                        buffer_tokens = 0
                    buffer.append(sent)
                    buffer_tokens += sent_tokens
            else:
                if buffer_tokens + para_tokens > self.max_tokens and buffer:
                    chunk_text = "\n\n".join(buffer)
                    start_char = char_cursor
                    end_char = start_char + len(chunk_text)
                    chunks.append(
                        Chunk.build(
                            text=chunk_text,
                            page_number=page_number,
                            start_char=start_char,
                            end_char=end_char,
                            strategy=self.strategy_name,
                        )
                    )
                    char_cursor = end_char + 2
                    buffer = []
                    buffer_tokens = 0
                buffer.append(para)
                buffer_tokens += para_tokens

        # Flush remaining buffer
        if buffer:
            chunk_text = "\n\n".join(buffer)
            chunks.append(
                Chunk.build(
                    text=chunk_text,
                    page_number=page_number,
                    start_char=char_cursor,
                    end_char=char_cursor + len(chunk_text),
                    strategy=self.strategy_name,
                )
            )

        return chunks

    @staticmethod
    def _split_paragraphs(text: str) -> list[str]:
        """Split on blank lines; also treat headings as boundaries."""
        raw = re.split(r"\n\s*\n", text)
        paragraphs = []
        for block in raw:
            block = block.strip()
            if block:
                paragraphs.append(block)
        return paragraphs


# ──────────────────────────────────────────────────────────────────────────────
# Convenience factory
# ──────────────────────────────────────────────────────────────────────────────

def get_all_chunks(
    pages: list[PageContent],
    chunk_size: int = 1500,
    chunk_overlap: int = 200,
) -> tuple[list[Chunk], list[Chunk]]:
    """
    Returns (fixed_chunks, semantic_chunks) for a list of pages.
    Convenience function to run both strategies in one call.
    """
    fixed = FixedSizeChunker(chunk_size=chunk_size, overlap=chunk_overlap)
    semantic = SemanticChunker(max_tokens=chunk_size, min_tokens=30)
    return fixed.chunk_pages(pages), semantic.chunk_pages(pages)
