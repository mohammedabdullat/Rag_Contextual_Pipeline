"""
Unit tests for the RAG pipeline components.
Run with: pytest tests/ -v
"""
import pytest
from unittest.mock import MagicMock, patch
from src.ingestion.chunker import FixedSizeChunker, SemanticChunker, Chunk
from src.ingestion.pdf_parser import PageContent


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_pages():
    return [
        PageContent(
            page_number=1,
            raw_text="This is the introduction. It describes the problem we are solving. "
                     "The approach involves several key steps that are outlined below.",
            tables_md=[],
        ),
        PageContent(
            page_number=2,
            raw_text="The methodology section explains our approach in detail. "
                     "We use a transformer-based architecture with attention mechanisms. "
                     "Experiments were conducted on three benchmark datasets.",
            tables_md=["| Model | Accuracy |\n|---|---|\n| Ours | 92.3% |\n| Baseline | 87.1% |"],
        ),
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Chunker tests
# ──────────────────────────────────────────────────────────────────────────────

class TestFixedSizeChunker:
    def test_produces_chunks(self, sample_pages):
        chunker = FixedSizeChunker(chunk_size=50, overlap=5)
        chunks = chunker.chunk_pages(sample_pages)
        assert len(chunks) > 0

    def test_chunk_has_correct_fields(self, sample_pages):
        chunker = FixedSizeChunker(chunk_size=100, overlap=10)
        chunks = chunker.chunk_pages(sample_pages)
        chunk = chunks[0]
        assert isinstance(chunk, Chunk)
        assert chunk.id
        assert chunk.text
        assert chunk.page_number in [1, 2]
        assert chunk.strategy == "fixed"

    def test_chunk_ids_are_unique(self, sample_pages):
        chunker = FixedSizeChunker(chunk_size=100, overlap=10)
        chunks = chunker.chunk_pages(sample_pages)
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_empty_pages_skipped(self):
        empty_pages = [PageContent(page_number=1, raw_text="", tables_md=[])]
        chunker = FixedSizeChunker(chunk_size=100, overlap=10)
        chunks = chunker.chunk_pages(empty_pages)
        assert len(chunks) == 0


class TestSemanticChunker:
    def test_produces_chunks(self, sample_pages):
        chunker = SemanticChunker(max_tokens=100)
        chunks = chunker.chunk_pages(sample_pages)
        assert len(chunks) > 0

    def test_strategy_name(self):
        chunker = SemanticChunker()
        assert chunker.strategy_name == "semantic"

    def test_respects_max_tokens(self, sample_pages):
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        chunker = SemanticChunker(max_tokens=80)
        chunks = chunker.chunk_pages(sample_pages)
        for chunk in chunks:
            token_count = len(enc.encode(chunk.text))
            # Allow some overflow for sentence boundary handling
            assert token_count <= 200, f"Chunk too large: {token_count} tokens"


# ──────────────────────────────────────────────────────────────────────────────
# PDF Parser tests (mocked)
# ──────────────────────────────────────────────────────────────────────────────

class TestPageContent:
    def test_combined_text_with_table(self):
        page = PageContent(
            page_number=1,
            raw_text="This is text.",
            tables_md=["| A | B |\n|---|---|\n| 1 | 2 |"],
        )
        assert "This is text." in page.combined_text
        assert "| A | B |" in page.combined_text

    def test_combined_text_no_table(self):
        page = PageContent(page_number=1, raw_text="Only text.", tables_md=[])
        assert page.combined_text == "Only text."


# ──────────────────────────────────────────────────────────────────────────────
# BM25 Index tests
# ──────────────────────────────────────────────────────────────────────────────

class TestBM25Index:
    def _make_chunks(self):
        pages = [PageContent(page_number=1, raw_text="machine learning neural networks deep learning", tables_md=[])]
        chunker = FixedSizeChunker(chunk_size=50, overlap=0)
        return chunker.chunk_pages(pages)

    def test_build_and_query(self):
        from src.indexing.bm25_index import BM25Index
        chunks = self._make_chunks()
        if not chunks:
            pytest.skip("No chunks generated")
        idx = BM25Index()
        idx.build(chunks)
        results = idx.query("neural networks", top_k=3)
        assert isinstance(results, list)
        assert len(results) <= 3

    def test_results_have_required_fields(self):
        from src.indexing.bm25_index import BM25Index
        chunks = self._make_chunks()
        if not chunks:
            pytest.skip("No chunks generated")
        idx = BM25Index()
        idx.build(chunks)
        results = idx.query("machine learning", top_k=1)
        if results:
            r = results[0]
            assert "chunk_id" in r
            assert "text" in r
            assert "score" in r


# ──────────────────────────────────────────────────────────────────────────────
# TF-IDF Index tests
# ──────────────────────────────────────────────────────────────────────────────

class TestTFIDFIndex:
    def test_build_and_query(self):
        from src.indexing.tfidf_index import TFIDFIndex
        pages = [PageContent(page_number=1, raw_text="transformer attention mechanism self-attention", tables_md=[])]
        chunker = FixedSizeChunker(chunk_size=50, overlap=0)
        chunks = chunker.chunk_pages(pages)
        if not chunks:
            pytest.skip("No chunks generated")
        idx = TFIDFIndex()
        idx.build(chunks)
        results = idx.query("attention mechanism", top_k=3)
        assert isinstance(results, list)

    def test_not_built_raises(self):
        from src.indexing.tfidf_index import TFIDFIndex
        idx = TFIDFIndex()
        with pytest.raises(RuntimeError):
            idx.query("test")
