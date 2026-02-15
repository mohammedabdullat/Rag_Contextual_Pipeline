"""
Benchmarker — evaluates the RAG pipeline against ground truth QA pairs.

Metrics:
  1. Average query latency per retrieval method
  2. Cosine similarity between generated answer embedding and ground truth embedding
  3. Recall@K (K=1,3,5) — does the correct page/chunk appear in top-K results?
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial.distance import cosine

from src.config import get_settings
from src.generation.answer_generator import AnswerGenerator
from src.indexing.embedding_index import EmbeddingIndex
from src.retrieval.retriever import Retriever

logger = logging.getLogger(__name__)


@dataclass
class QAPair:
    """One ground truth question-answer pair."""
    question: str
    answer: str
    page: int
    span_start: int
    span_end: int


@dataclass
class MethodMetrics:
    latencies_ms: list[float] = field(default_factory=list)
    cosine_sims: list[float] = field(default_factory=list)
    recall_at_1: list[bool] = field(default_factory=list)
    recall_at_3: list[bool] = field(default_factory=list)
    recall_at_5: list[bool] = field(default_factory=list)

    @property
    def avg_latency(self) -> float:
        return round(float(np.mean(self.latencies_ms)), 2) if self.latencies_ms else 0.0

    @property
    def avg_cosine_sim(self) -> float:
        return round(float(np.mean(self.cosine_sims)), 4) if self.cosine_sims else 0.0

    @property
    def recall_1(self) -> float:
        return round(float(np.mean(self.recall_at_1)), 4) if self.recall_at_1 else 0.0

    @property
    def recall_3(self) -> float:
        return round(float(np.mean(self.recall_at_3)), 4) if self.recall_at_3 else 0.0

    @property
    def recall_5(self) -> float:
        return round(float(np.mean(self.recall_at_5)), 4) if self.recall_at_5 else 0.0


class Benchmarker:
    """
    Runs all ground truth QA pairs through the RAG pipeline and computes metrics.

    Usage:
        bench = Benchmarker(retriever, generator, embed_index)
        report = bench.run("data/ground_truth.json")
        bench.print_report(report)
        bench.save_report(report, "benchmark_report.md")
    """

    def __init__(
        self,
        retriever: Retriever,
        generator: AnswerGenerator,
        embed_index: EmbeddingIndex,
    ) -> None:
        self._retriever = retriever
        self._generator = generator
        self._embed = embed_index

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def run(self, ground_truth_path: str) -> dict[str, MethodMetrics]:
        """Run full benchmark and return per-method metrics."""
        qa_pairs = self._load_ground_truth(ground_truth_path)
        logger.info("Running benchmark on %d QA pairs...", len(qa_pairs))

        methods = ["contextual", "bm25", "tfidf", "hybrid"]
        metrics: dict[str, MethodMetrics] = {m: MethodMetrics() for m in methods}

        for qa in qa_pairs:
            retrieval = self._retriever.retrieve(qa.question, top_k=5)
            gt_embedding = self._embed.embed_text(qa.answer)

            for method in methods:
                chunks = getattr(retrieval, method, [])
                latency = retrieval.latency_ms.get(method, 0.0)

                # Generate answer from this method's context
                context = self._retriever.get_context_text(retrieval, method=method)
                t0 = time.perf_counter()
                generated = self._generator.generate(qa.question, context)
                gen_latency = (time.perf_counter() - t0) * 1000

                total_latency = latency + gen_latency

                # Cosine similarity
                gen_embedding = self._embed.embed_text(generated)
                sim = 1.0 - cosine(gt_embedding, gen_embedding)
                metrics[method].cosine_sims.append(round(sim, 4))
                metrics[method].latencies_ms.append(round(total_latency, 2))

                # Recall@K — check if correct page appears in top-K results
                retrieved_pages = [c.get("page_number") for c in chunks]
                metrics[method].recall_at_1.append(qa.page in retrieved_pages[:1])
                metrics[method].recall_at_3.append(qa.page in retrieved_pages[:3])
                metrics[method].recall_at_5.append(qa.page in retrieved_pages[:5])

        return metrics

    def print_report(self, metrics: dict[str, MethodMetrics]) -> None:
        """Print a human-readable benchmark report to stdout."""
        print("\n" + "=" * 70)
        print("  BENCHMARK REPORT — Contextual RAG Pipeline")
        print("=" * 70)
        header = f"{'Method':<15} {'Avg Latency':>12} {'Cos Sim':>10} {'R@1':>8} {'R@3':>8} {'R@5':>8}"
        print(header)
        print("-" * 70)
        for method, m in metrics.items():
            print(
                f"{method:<15} {m.avg_latency:>10.1f}ms "
                f"{m.avg_cosine_sim:>10.4f} "
                f"{m.recall_1:>8.2%} "
                f"{m.recall_3:>8.2%} "
                f"{m.recall_5:>8.2%}"
            )
        print("=" * 70 + "\n")

    def save_report(self, metrics: dict[str, MethodMetrics], output_path: str) -> None:
        """Save benchmark report as a Markdown file."""
        lines = [
            "# Benchmark Report — Contextual RAG Pipeline",
            "",
            "## Methodology",
            "- **Latency**: Total time (retrieval + generation) per query",
            "- **Cosine Similarity**: Between generated answer embedding and ground truth embedding",
            "- **Recall@K**: % of queries where the correct page appeared in top-K results",
            "",
            "## Results",
            "",
            "| Method | Avg Latency | Cosine Sim | Recall@1 | Recall@3 | Recall@5 |",
            "|--------|------------|------------|----------|----------|----------|",
        ]
        for method, m in metrics.items():
            lines.append(
                f"| {method} | {m.avg_latency:.1f}ms | {m.avg_cosine_sim:.4f} "
                f"| {m.recall_1:.1%} | {m.recall_3:.1%} | {m.recall_5:.1%} |"
            )

        lines += [
            "",
            "## Notes",
            "- **Contextual** uses Anthropic-style context-prepended embeddings in ChromaDB",
            "- **BM25** uses sparse keyword matching (BM25Okapi)",
            "- **TF-IDF** uses cosine similarity over TF-IDF vectors",
            "- **Hybrid** uses Reciprocal Rank Fusion across all three methods",
        ]

        Path(output_path).write_text("\n".join(lines))
        logger.info("Benchmark report saved to %s", output_path)

    # ------------------------------------------------------------------ #
    # Private                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _load_ground_truth(path: str) -> list[QAPair]:
        data = json.loads(Path(path).read_text())
        pairs = []
        for item in data:
            pairs.append(
                QAPair(
                    question=item["question"],
                    answer=item["answer"],
                    page=item.get("page", 1),
                    span_start=item.get("span_start", 0),
                    span_end=item.get("span_end", 0),
                )
            )
        return pairs
