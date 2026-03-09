"""
RAG tools exposed as @function_tool for the Data Synthesis Agent.

Two tools are provided:

index_financial_document
    Parses a PDF, splits it into section-aware chunks (with tables preserved
    as Markdown), and feeds the chunks into the HybridSearchEngine so they
    are available for retrieval.

search_financial_documents
    Executes a hybrid BM25 + vector search over all indexed chunks and returns
    the top-k results re-ranked by Reciprocal Rank Fusion (RRF).

The DocumentProcessor and HybridSearchEngine are module-level singletons so
the index persists across multiple function_tool calls within a single server
or CLI process.
"""

from __future__ import annotations

import logging
import pathlib

from agents import function_tool
from utils.audit_logger import log_event
from tools.document_processor import DocumentProcessor
from tools.hybrid_search import HybridSearchEngine

logger = logging.getLogger(__name__)

# ── Module-level singletons ───────────────────────────────────────────────────
# Shared by all invocations within the same process lifetime.

_processor = DocumentProcessor()
_engine = HybridSearchEngine(embedding_model="all-MiniLM-L6-v2")
_indexed_docs: set[str] = set()   # tracks which filenames have been indexed


# ── Tools ─────────────────────────────────────────────────────────────────────


@function_tool
def index_financial_document(filename: str) -> dict:
    """
    Parse and index a financial PDF document for hybrid retrieval.

    This tool performs three steps:
    1. **Structured parsing** — extracts text using PyMuPDF and detects section
       headings via font-size heuristics (not fixed-size character windows).
    2. **Table preservation** — detects tabular regions and converts them to
       Markdown format so their structure is retained in the embedding.
    3. **Dual indexing** — builds a BM25 index for keyword recall and a dense
       vector index (sentence-transformers) for semantic recall.

    Call this tool *before* using search_financial_documents on a new file.
    Pass only the filename (e.g. 'annual_report.pdf'), not a full path.
    """
    log_event("DataAgent", "index_financial_document", {"filename": filename})

    safe_name = pathlib.Path(filename).name

    if safe_name in _indexed_docs:
        return {
            "status": "already_indexed",
            "filename": safe_name,
            "total_indexed_chunks": _engine.total_chunks,
            "message": (
                f"'{safe_name}' is already indexed. "
                "Call search_financial_documents to query it."
            ),
        }

    try:
        chunks = _processor.process(safe_name)
    except FileNotFoundError as exc:
        return {"error": str(exc)}
    except ImportError as exc:
        return {"error": str(exc)}
    except Exception as exc:
        logger.exception("Unexpected error while processing '%s'", safe_name)
        return {"error": f"Failed to process document: {exc}"}

    if not chunks:
        return {
            "status": "empty",
            "filename": safe_name,
            "message": "No content could be extracted from this document.",
        }

    try:
        _engine.index(chunks)
    except ImportError as exc:
        return {"error": str(exc)}
    except Exception as exc:
        logger.exception("Failed to build search index for '%s'", safe_name)
        return {"error": f"Indexing failed: {exc}"}

    _indexed_docs.add(safe_name)

    text_chunks = [c for c in chunks if c.chunk_type == "text"]
    table_chunks = [c for c in chunks if c.chunk_type == "table"]
    unique_sections = sorted({c.section for c in chunks})

    return {
        "status": "indexed",
        "filename": safe_name,
        "total_chunks": len(chunks),
        "text_chunks": len(text_chunks),
        "table_chunks": len(table_chunks),
        "sections_detected": unique_sections[:25],   # cap for token budget
        "total_indexed_chunks": _engine.total_chunks,
        "message": (
            f"Successfully indexed '{safe_name}'. "
            "You can now call search_financial_documents to query it."
        ),
    }


@function_tool
def search_financial_documents(query: str, top_k: int = 5) -> dict:
    """
    Search all indexed financial documents using Hybrid BM25 + Vector Search
    re-ranked by Reciprocal Rank Fusion (RRF).

    How retrieval works
    -------------------
    * **BM25 (keyword search)** — ensures exact matches for specific financial
      terms and regulations, e.g. "Roth IRA 5-year rule", "FDIC insurance
      limit", "Basel III Tier-1 capital ratio".
    * **Vector search (cosine similarity)** — captures semantic intent even
      when the exact words don't appear in the document, e.g. "client prefers
      steady growth" matches chunks about dividend or bond strategies.
    * **RRF fusion** — merges both ranked lists: a chunk ranking highly in
      *either* list surfaces to the top; one ranking highly in *both* receives
      the strongest combined boost.

    Parameters
    ----------
    query   : The search query (natural language or keywords).
    top_k   : Number of results to return (default 5, max 20).

    Always call index_financial_document first to load a PDF into the index.
    """
    log_event(
        "DataAgent",
        "search_financial_documents",
        {"query": query, "top_k": top_k},
    )

    top_k = max(1, min(int(top_k), 20))   # clamp to [1, 20]

    if not _indexed_docs:
        return {
            "error": (
                "No documents have been indexed yet. "
                "Call index_financial_document with a PDF filename first."
            ),
            "indexed_documents": [],
        }

    try:
        results = _engine.search(query, top_k=top_k)
    except ImportError as exc:
        return {"error": str(exc)}
    except Exception as exc:
        logger.exception("Search failed for query '%s'", query)
        return {"error": f"Search failed: {exc}"}

    if not results:
        return {
            "query": query,
            "indexed_documents": sorted(_indexed_docs),
            "results": [],
            "message": "No matching chunks found. Try rephrasing the query.",
        }

    return {
        "query": query,
        "indexed_documents": sorted(_indexed_docs),
        "total_results": len(results),
        "results": [
            {
                "rank": i + 1,
                "rrf_score": round(r.rrf_score, 6),
                "bm25_rank": r.bm25_rank,
                "vector_rank": r.vector_rank,
                "document": r.document,
                "page": r.page,
                "section": r.section,
                "chunk_type": r.chunk_type,    # "text" or "table"
                "content": r.content[:1500],   # cap for token budget
            }
            for i, r in enumerate(results)
        ],
    }
