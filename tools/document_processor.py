"""
Document Processor: PDF parsing with section/header-based chunking and table extraction.

Strategy
--------
* **Header detection**: span font-size compared to the page median — spans whose
  size is >= HEADING_RATIO x median (or whose font flags indicate bold) are treated
  as section headings.  Text between headings is collected into a single chunk.
* **Table extraction**: prefers PyMuPDF's native `page.find_tables()` (available
  since PyMuPDF ≥ 1.23).  Falls back to pdfplumber when PyMuPDF tables are absent
  or the library version is older.  Tables are converted to Markdown before
  embedding so their structure is preserved.
* Each chunk carries rich metadata: source filename, page number, section title,
  and type ("text" | "table").

Dependencies
------------
    pip install pymupdf pdfplumber
"""

from __future__ import annotations

import logging
import pathlib
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ── Constants

UPLOAD_DIR = pathlib.Path("uploads")

# Font-size ratio above page median to classify a span as a heading
HEADING_SIZE_RATIO: float = 1.15

# Minimum characters for a text chunk to be kept (avoids noise fragments)
DEFAULT_MIN_CHUNK_CHARS: int = 80


# ── Data model


@dataclass
class DocumentChunk:
    """One semantic unit extracted from a parsed document."""

    chunk_id: str       # unique identifier, e.g. "report.pdf::0012"
    document: str       # source filename
    page: int           # 1-based page number
    section: str        # enclosing section / heading title
    content: str        # raw text or Markdown table
    chunk_type: str     # "text" | "table"
    metadata: dict = field(default_factory=dict)


# ── Main class


class DocumentProcessor:
    """
    Parses financial PDFs into a flat list of :class:`DocumentChunk` objects.

    Usage::

        processor = DocumentProcessor()
        chunks = processor.process("annual_report.pdf")
    """

    def __init__(self, min_chunk_chars: int = DEFAULT_MIN_CHUNK_CHARS) -> None:
        self.min_chunk_chars = min_chunk_chars

    # ── Public API

    def process(self, filename: str) -> list[DocumentChunk]:
        """
        Parse *filename* (located in ``uploads/``) and return all chunks.

        Raises
        ------
        FileNotFoundError
            When the file is absent from the upload directory.
        ImportError
            When PyMuPDF is not installed.
        """
        safe_name = pathlib.Path(filename).name   # strip directory traversal
        file_path = UPLOAD_DIR / safe_name

        if not file_path.exists():
            raise FileNotFoundError(
                f"File '{safe_name}' not found. Please upload it first."
            )

        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError(
                "PyMuPDF is not installed. Run: pip install pymupdf"
            )

        fitz_doc = fitz.open(str(file_path))

        # Optional pdfplumber for table fallback
        plumber_doc = None
        try:
            import pdfplumber
            plumber_doc = pdfplumber.open(str(file_path))
        except Exception:
            pass  # table fallback simply won't fire

        try:
            chunks = self._extract_chunks(fitz_doc, plumber_doc, safe_name)
        finally:
            fitz_doc.close()
            if plumber_doc is not None:
                plumber_doc.close()

        logger.info(
            "Processed '%s': %d chunks (%d text, %d table).",
            safe_name,
            len(chunks),
            sum(1 for c in chunks if c.chunk_type == "text"),
            sum(1 for c in chunks if c.chunk_type == "table"),
        )
        return chunks

    # ── Internal helpers

    def _extract_chunks(
        self,
        fitz_doc,
        plumber_doc,
        filename: str,
    ) -> list[DocumentChunk]:
        """Walk every page and build an ordered list of DocumentChunk objects."""
        import fitz

        # ── Pass 1: collect all span font sizes for global median ────────────
        all_sizes: list[float] = []
        for page in fitz_doc:
            for block in page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]:
                if block["type"] == 0:  # text block
                    for line in block["lines"]:
                        for span in line["spans"]:
                            all_sizes.append(span["size"])

        median_size: float = (
            sorted(all_sizes)[len(all_sizes) // 2] if all_sizes else 12.0
        )

        # ── Pass 2: walk pages
        chunks: list[DocumentChunk] = []
        chunk_counter = 0

        current_section: str = "Preamble"
        text_buffer: list[str] = []
        section_page: int = 1

        def flush_text(page_num: int) -> None:
            nonlocal chunk_counter
            text = "\n".join(text_buffer).strip()
            text_buffer.clear()
            if len(text) < self.min_chunk_chars:
                return
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{filename}::{chunk_counter:04d}",
                    document=filename,
                    page=page_num,
                    section=current_section,
                    content=text,
                    chunk_type="text",
                )
            )
            chunk_counter += 1

        for page_num, fitz_page in enumerate(fitz_doc, start=1):
            plumber_page = (
                plumber_doc.pages[page_num - 1]
                if plumber_doc is not None
                else None
            )

            # ── Extract tables (before text so we know which bbox areas to skip)
            page_table_chunks, table_bboxes = self._extract_tables(
                fitz_page, plumber_page, filename, page_num,
                current_section, chunk_counter,
            )
            chunk_counter += len(page_table_chunks)
            chunks.extend(page_table_chunks)

            # ── Walk text spans, skip table bounding boxes
            page_dict = fitz_page.get_text(
                "dict", flags=fitz.TEXT_PRESERVE_WHITESPACE
            )
            for block in page_dict["blocks"]:
                if block["type"] != 0:
                    continue

                # Skip blocks that overlap with an extracted table region
                bx0, by0, bx1, by1 = block["bbox"]
                if any(
                    bx0 < tx1 and bx1 > tx0 and by0 < ty1 and by1 > ty0
                    for (tx0, ty0, tx1, ty1) in table_bboxes
                ):
                    continue

                for line in block["lines"]:
                    line_text = ""
                    is_heading = False
                    for span in line["spans"]:
                        line_text += span["text"]
                        if span["size"] >= median_size * HEADING_SIZE_RATIO or (
                            span["flags"] & (1 << 4)  # bold flag
                        ):
                            is_heading = True
                    line_text = line_text.strip()
                    if not line_text:
                        continue

                    if is_heading and len(line_text) > 3:
                        flush_text(section_page)
                        current_section = line_text
                        section_page = page_num
                    else:
                        text_buffer.append(line_text)

        flush_text(section_page)
        return chunks

    # ── Table extraction

    def _extract_tables(
        self,
        fitz_page,
        plumber_page,
        filename: str,
        page_num: int,
        section: str,
        chunk_counter_start: int,
    ) -> tuple[list[DocumentChunk], list[tuple]]:
        """
        Return (table_chunks, table_bboxes).

        *table_bboxes* is a list of (x0, y0, x1, y1) floats in page-space,
        used by the caller to skip overlapping text blocks.
        """
        table_chunks: list[DocumentChunk] = []
        table_bboxes: list[tuple] = []
        counter = chunk_counter_start

        # ── Strategy A: PyMuPDF native find_tables
        try:
            tabs = fitz_page.find_tables()
            for table in tabs:
                rows = table.extract()
                md = self._rows_to_markdown(rows)
                if not md:
                    continue
                table_bboxes.append(tuple(table.bbox))
                table_chunks.append(
                    DocumentChunk(
                        chunk_id=f"{filename}::{counter:04d}",
                        document=filename,
                        page=page_num,
                        section=section,
                        content=md,
                        chunk_type="table",
                        metadata={"extraction_method": "pymupdf"},
                    )
                )
                counter += 1
            if table_chunks:
                return table_chunks, table_bboxes
        except Exception:
            pass  # fall through to pdfplumber

        # ── Strategy B: pdfplumber fallback
        if plumber_page is not None:
            try:
                for i, table in enumerate(plumber_page.extract_tables() or []):
                    md = self._rows_to_markdown(table)
                    if not md:
                        continue
                    # pdfplumber returns bbox via table.bbox attr (if available)
                    bbox = getattr(table, "bbox", None)
                    if bbox:
                        table_bboxes.append(tuple(bbox))
                    table_chunks.append(
                        DocumentChunk(
                            chunk_id=f"{filename}::{counter:04d}",
                            document=filename,
                            page=page_num,
                            section=section,
                            content=md,
                            chunk_type="table",
                            metadata={"extraction_method": "pdfplumber"},
                        )
                    )
                    counter += 1
            except Exception:
                pass

        return table_chunks, table_bboxes

    # ── Markdown conversion

    @staticmethod
    def _rows_to_markdown(rows: list[list] | None) -> str:
        """Convert a list-of-rows table to a GitHub-flavoured Markdown table."""
        if not rows:
            return ""

        # Normalise: replace None/empty with empty string, stringify everything
        cleaned: list[list[str]] = [
            [str(cell).strip() if cell is not None else "" for cell in row]
            for row in rows
            if any(cell for cell in row)  # skip fully-empty rows
        ]
        if not cleaned:
            return ""

        col_count = max(len(r) for r in cleaned)
        # Pad all rows to the same width
        cleaned = [r + [""] * (col_count - len(r)) for r in cleaned]

        lines: list[str] = []
        # Header row
        lines.append("| " + " | ".join(cleaned[0]) + " |")
        # Separator
        lines.append("| " + " | ".join(["---"] * col_count) + " |")
        # Data rows
        for row in cleaned[1:]:
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)
