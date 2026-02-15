#!/usr/bin/env python3
"""
CLI script to build all RAG indices from a PDF.

Usage:
    python scripts/build_index.py --pdf data/paper.pdf [--overwrite]
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from src.config import get_settings
from src.contextual.context_generator import ContextGenerator
from src.indexing.bm25_index import BM25Index
from src.indexing.embedding_index import EmbeddingIndex
from src.indexing.tfidf_index import TFIDFIndex
from src.ingestion.chunker import get_all_chunks
from src.ingestion.pdf_parser import PDFParser

console = Console()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Build RAG indices from PDF")
    parser.add_argument("--pdf", default=None, help="Path to PDF file")
    parser.add_argument("--overwrite", action="store_true", help="Force rebuild of indices")
    args = parser.parse_args()

    settings = get_settings()
    pdf_path = args.pdf or settings.pdf_path
    print(f"Using PDF: {pdf_path}")

    if not Path(pdf_path).exists():
        console.print(f"[red]ERROR: PDF not found at {pdf_path}[/red]")
        sys.exit(1)
    
    console.rule("[bold blue]Contextual RAG Pipeline — Index Builder[/bold blue]")

    # Step 1: Parse PDF
    console.print("\n[bold]Step 1/4:[/bold] Parsing PDF...")
    pdf_parser = PDFParser(pdf_path)
    pages = pdf_parser.parse()
    full_doc = pdf_parser.get_full_text()
    console.print(f"  [green]✓[/green] Parsed {len(pages)} pages")

    # Step 2: Chunk
    console.print("\n[bold]Step 2/4:[/bold] Chunking document (2 strategies)...")
    fixed_chunks, semantic_chunks = get_all_chunks(
        pages, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap,
    )
    all_chunks = fixed_chunks + semantic_chunks
    console.print(f"  [green]✓[/green] Fixed-size: {len(fixed_chunks)}  |  Semantic: {len(semantic_chunks)}  |  Total: {len(all_chunks)}")

    # Step 3: Contextual enrichment (Anthropic technique)
    console.print("\n[bold]Step 3/4:[/bold] Enriching chunks with Anthropic-style context...")
    ctx_gen = ContextGenerator()
    ctx_gen.enrich_chunks(all_chunks, full_doc, show_progress=True)
    console.print("  [green]✓[/green] All chunks enriched")

    # Step 4: Build indices
    console.print("\n[bold]Step 4/4:[/bold] Building 3 indices...")
    embed_idx = EmbeddingIndex(chroma_path=settings.chroma_persist_dir)
    embed_idx.index_chunks(fixed_chunks, overwrite=args.overwrite)
    console.print("  [green]✓[/green] ChromaDB vector index")

    bm25_idx = BM25Index(persist_dir=settings.chroma_persist_dir)
    bm25_idx.build(fixed_chunks, overwrite=args.overwrite)
    console.print("  [green]✓[/green] BM25 index")

    tfidf_idx = TFIDFIndex(persist_dir=settings.chroma_persist_dir)
    tfidf_idx.build(fixed_chunks, overwrite=args.overwrite)
    console.print("  [green]✓[/green] TF-IDF index")

    console.rule("[bold green]Index build complete! Run: uvicorn src.api.main:app --reload[/bold green]")


if __name__ == "__main__":
    main()
