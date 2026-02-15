"""
PDF Parser — extracts text and table content from PDF files using PyMuPDF + pdfplumber.
Tables are serialized as Markdown so they can be embedded as natural text.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import pdfplumber

logger = logging.getLogger(__name__)


@dataclass
class PageContent:
    """Holds extracted content for a single PDF page."""
    page_number: int          # 1-indexed
    raw_text: str             # Plain text from the page
    tables_md: list[str]      # Each table serialized as Markdown
    combined_text: str = ""   # raw_text + tables merged (populated post-init)

    def __post_init__(self) -> None:
        table_block = "\n\n".join(self.tables_md)
        self.combined_text = (
            f"{self.raw_text}\n\n{table_block}".strip() if table_block else self.raw_text
        )


class PDFParser:
    """
    Parses a PDF into per-page content objects.

    Uses:
      - PyMuPDF  (fitz) for clean text extraction
      - pdfplumber for table extraction → Markdown
    """

    def __init__(self, pdf_path: str | Path) -> None:
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")
        self._pages: Optional[list[PageContent]] = None

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def parse(self) -> list[PageContent]:
        """Parse the entire PDF and return a list of PageContent objects."""
        if self._pages is not None:
            return self._pages

        logger.info("Parsing PDF: %s", self.pdf_path)
        text_by_page = self._extract_text()
        tables_by_page = self._extract_tables()

        pages: list[PageContent] = []
        for page_num, raw_text in text_by_page.items():
            tables = tables_by_page.get(page_num, [])
            pages.append(
                PageContent(
                    page_number=page_num,
                    raw_text=raw_text,
                    tables_md=tables,
                )
            )

        self._pages = sorted(pages, key=lambda p: p.page_number)
        logger.info("Parsed %d pages", len(self._pages))
        return self._pages

    def get_full_text(self) -> str:
        """Return the complete document text (all pages concatenated)."""
        pages = self.parse()
        return "\n\n".join(
            f"[Page {p.page_number}]\n{p.combined_text}" for p in pages
        )

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _extract_text(self) -> dict[int, str]:
        """Extract plain text per page using PyMuPDF."""
        text_map: dict[int, str] = {}
        with fitz.open(str(self.pdf_path)) as doc:
            for page in doc:
                page_num = page.number + 1  # 1-indexed
                text = page.get_text("text")
                text_map[page_num] = text.strip()
        return text_map

    def _extract_tables(self) -> dict[int, list[str]]:
        """Extract tables per page using pdfplumber, returning Markdown strings."""
        tables_map: dict[int, list[str]] = {}
        try:
            with pdfplumber.open(str(self.pdf_path)) as pdf:
                for i, page in enumerate(pdf.pages):
                    page_num = i + 1
                    raw_tables = page.extract_tables()
                    if raw_tables:
                        md_tables = [self._table_to_markdown(t) for t in raw_tables if t]
                        if md_tables:
                            tables_map[page_num] = md_tables
        except Exception as exc:
            logger.warning("Table extraction failed (non-fatal): %s", exc)
        return tables_map

    @staticmethod
    def _table_to_markdown(table: list[list]) -> str:
        """Convert a pdfplumber table (list of rows) to a Markdown table string."""
        if not table:
            return ""
        # Clean cells
        cleaned = [
            [str(cell).strip() if cell is not None else "" for cell in row]
            for row in table
        ]
        header = cleaned[0]
        separator = ["---"] * len(header)
        rows = cleaned[1:]

        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(separator) + " |",
        ]
        for row in rows:
            # Pad short rows
            while len(row) < len(header):
                row.append("")
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)
