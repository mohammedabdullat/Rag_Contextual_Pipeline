#!/usr/bin/env python3
"""
CLI script to run the benchmark evaluation against ground truth QA pairs.

Usage:
    python scripts/run_benchmark.py [--ground-truth data/ground_truth.json]
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
from src.evaluation.benchmarker import Benchmarker
from src.generation.answer_generator import AnswerGenerator
from src.indexing.bm25_index import BM25Index
from src.indexing.embedding_index import EmbeddingIndex
from src.indexing.tfidf_index import TFIDFIndex
from src.ingestion.chunker import get_all_chunks
from src.ingestion.pdf_parser import PDFParser
from src.retrieval.retriever import Retriever

console = Console()
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Run RAG benchmark evaluation")
    parser.add_argument("--ground-truth", default=None, help="Path to ground_truth.json")
    parser.add_argument("--output", default="benchmark_report.md", help="Output markdown file")
    args = parser.parse_args()

    settings = get_settings()
    gt_path = args.ground_truth or settings.ground_truth_path

    if not Path(gt_path).exists():
        console.print(f"[red]Ground truth file not found: {gt_path}[/red]")
        sys.exit(1)

    console.rule("[bold blue]Contextual RAG Pipeline — Benchmark[/bold blue]")
    console.print(f"Loading indices from [cyan]{settings.chroma_persist_dir}[/cyan]...")

    # Load pipeline (assumes index already built)
    pdf_parser = PDFParser(settings.pdf_path)
    pages = pdf_parser.parse()
    full_doc = pdf_parser.get_full_text()
    fixed_chunks, semantic_chunks = get_all_chunks(pages)
    all_chunks = fixed_chunks + semantic_chunks

    ctx_gen = ContextGenerator()
    ctx_gen.enrich_chunks(all_chunks, full_doc, show_progress=False)

    embed_idx = EmbeddingIndex(chroma_path=settings.chroma_persist_dir)
    embed_idx.index_chunks(all_chunks)
    bm25_idx = BM25Index(persist_dir=settings.chroma_persist_dir)
    bm25_idx.build(all_chunks)
    tfidf_idx = TFIDFIndex(persist_dir=settings.chroma_persist_dir)
    tfidf_idx.build(all_chunks)

    retriever = Retriever(embed_idx, bm25_idx, tfidf_idx)
    generator = AnswerGenerator()

    console.print("Running benchmark evaluation...")
    benchmarker = Benchmarker(retriever, generator, embed_idx)
    metrics = benchmarker.run(gt_path)

    benchmarker.print_report(metrics)
    benchmarker.save_report(metrics, args.output)
    console.print(f"\n[green]✓ Report saved to [bold]{args.output}[/bold][/green]\n")


if __name__ == "__main__":
    main()
