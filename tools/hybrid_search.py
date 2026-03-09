"""
Hybrid Search Engine: BM25 (keyword) + Dense Vector Search fused with
Reciprocal Rank Fusion (RRF).

Architecture
------------
BM25 (via rank-bm25 / BM25Okapi)
    Exact-match recall for financial terminology — e.g. "Roth IRA 5-year rule",
    "Regulation D", "Basel III Tier-1 capital".

Vector Search (via sentence-transformers + cosine similarity)
    Semantic intent retrieval — e.g. "client wants steady income" maps to chunks
    about dividend portfolios even when those words don't appear verbatim.

Reciprocal Rank Fusion
    RRF score = Σ  1 / (k + rank_i)  (k = 60 by default).
    Ranks contributions from both lists; results that rank highly in either
    list surface to the top, while results that rank well in *both* lists
    receive the strongest boost — giving a natural balance of exact and
    semantic recall without requiring score normalisation across search types.

Dependencies
------------
    pip install rank-bm25 sentence-transformers numpy
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Data model ────────────────────────────────────────────────────────────────


@dataclass
class SearchResult:
    """One ranked result returned by :meth:`HybridSearchEngine.search`."""

    chunk_id: str
    document: str
    page: int
    section: str
    content: str
    chunk_type: str       # "text" | "table"
    bm25_rank: Optional[int]    # 1-based rank in the BM25 list  (None if absent)
    vector_rank: Optional[int]  # 1-based rank in the vector list (None if absent)
    rrf_score: float            # fused Reciprocal Rank Fusion score


# ── Engine ────────────────────────────────────────────────────────────────────


class HybridSearchEngine:
    """
    Maintains an in-memory index and answers queries with hybrid retrieval.

    Usage::

        engine = HybridSearchEngine()
        engine.index(chunks)                  # list[DocumentChunk]
        results = engine.search("query", 5)   # list[SearchResult]

    Calling ``index()`` multiple times is safe: new chunks are appended and
    only their embeddings are computed; the BM25 index is rebuilt from scratch
    (fast, pure-Python) over the full corpus.
    """

    # RRF constant — 60 is the standard empirical default
    RRF_K: int = 60

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2") -> None:
        self._embedding_model_name = embedding_model
        self._st_model: Optional[Any] = None           # SentenceTransformer, lazy-loaded
        self._chunks: list = []                        # DocumentChunk objects
        self._bm25: Optional[Any] = None               # BM25Okapi
        self._embeddings: Optional[np.ndarray] = None  # (N, D) float32, L2-normed

    # ── Indexing ──────────────────────────────────────────────────────────────

    def index(self, chunks: list) -> None:
        """
        Add *chunks* to the index and rebuild retrieval structures.

        Only the embeddings for the *new* chunks are computed; existing
        embeddings are reused to keep incremental indexing fast.
        """
        if not chunks:
            return

        # ── Compute embeddings for new chunks only ──────────────────────────
        model: Any = self._load_model()
        new_texts = [c.content for c in chunks]
        new_embs: np.ndarray = model.encode(
            new_texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)

        # Append chunks and embeddings
        self._chunks.extend(chunks)
        self._embeddings = (
            new_embs
            if self._embeddings is None
            else np.vstack([self._embeddings, new_embs])
        )

        # L2-normalise the full matrix (cheap, enables cosine via dot-product)
        assert self._embeddings is not None
        norms: np.ndarray = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        self._embeddings = self._embeddings / norms

        # ── Rebuild BM25 over full corpus (fast) ────────────────────────────
        self._build_bm25()

        logger.info(
            "HybridSearchEngine: indexed +%d chunks (%d total). "
            "BM25 rebuilt; vector store updated.",
            len(chunks),
            len(self._chunks),
        )

    def clear(self) -> None:
        """Remove all indexed chunks and reset both indices."""
        self._chunks.clear()
        self._bm25 = None
        self._embeddings = None

    @property
    def total_chunks(self) -> int:
        return len(self._chunks)

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """
        Hybrid BM25 + dense vector retrieval fused with Reciprocal Rank Fusion.

        Parameters
        ----------
        query:
            Natural-language or keyword query string.
        top_k:
            Number of final results to return (after fusion).

        Returns
        -------
        list[SearchResult]
            Sorted by descending ``rrf_score``.
        """
        if not self._chunks:
            return []

        candidate_k = min(top_k * 4, len(self._chunks))

        bm25_ranked = self._bm25_search(query, top_k=candidate_k)
        vector_ranked = self._vector_search(query, top_k=candidate_k)

        return self._rrf_fuse(bm25_ranked, vector_ranked)[:top_k]

    # ── BM25 ──────────────────────────────────────────────────────────────────

    def _build_bm25(self) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                "rank-bm25 is not installed. Run: pip install rank-bm25"
            )
        tokenised = [self._tokenise(c.content) for c in self._chunks]
        self._bm25 = BM25Okapi(tokenised)

    def _bm25_search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """
        Return ``[(chunk_index, bm25_score), …]`` sorted by descending score.
        """
        assert self._bm25 is not None, "_bm25 is not built — call index() first"
        tokens = self._tokenise(query)
        scores: np.ndarray = self._bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores.tolist()), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    # ── Vector search ─────────────────────────────────────────────────────────

    def _load_model(self) -> Any:
        if self._st_model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers is not installed. "
                    "Run: pip install sentence-transformers"
                )
            logger.info("Loading embedding model '%s'…", self._embedding_model_name)
            self._st_model = SentenceTransformer(self._embedding_model_name)
        return self._st_model

    def _vector_search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """
        Return ``[(chunk_index, cosine_similarity), …]`` sorted descending.
        """
        assert self._embeddings is not None, "_embeddings not built — call index() first"
        model: Any = self._load_model()
        q_emb: np.ndarray = model.encode([query], convert_to_numpy=True).astype(np.float32)
        # L2-normalise query vector
        q_norm: float = float(np.linalg.norm(q_emb))
        q_emb = q_emb / (q_norm if q_norm > 0.0 else 1.0)
        # Cosine similarity = dot product (both sides are L2-normed)
        sims: np.ndarray = (self._embeddings @ q_emb.T).squeeze()
        ranked = sorted(enumerate(sims.tolist()), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    # ── Reciprocal Rank Fusion ────────────────────────────────────────────────

    def _rrf_fuse(
        self,
        bm25_ranked: list[tuple[int, float]],
        vector_ranked: list[tuple[int, float]],
    ) -> list[SearchResult]:
        """
        Fuse two ranked lists using Reciprocal Rank Fusion.

        RRF(d) = Σ  1 / (k + rank(d, list_i))

        A document that ranks #1 in BM25 and #1 in vector search receives an
        RRF score of  2 / (60 + 1) ≈ 0.0328 — the maximum possible score for
        default k=60.
        """
        rrf_scores: dict[int, float] = defaultdict(float)

        for rank, (idx, _) in enumerate(bm25_ranked, start=1):
            rrf_scores[idx] += 1.0 / (self.RRF_K + rank)

        for rank, (idx, _) in enumerate(vector_ranked, start=1):
            rrf_scores[idx] += 1.0 / (self.RRF_K + rank)

        # Build rank-attribution maps for diagnostics
        bm25_rank_map = {idx: r for r, (idx, _) in enumerate(bm25_ranked, start=1)}
        vec_rank_map = {idx: r for r, (idx, _) in enumerate(vector_ranked, start=1)}

        sorted_entries = sorted(
            rrf_scores.items(), key=lambda kv: kv[1], reverse=True
        )

        results: list[SearchResult] = []
        for idx, rrf_score in sorted_entries:
            c = self._chunks[idx]
            results.append(
                SearchResult(
                    chunk_id=c.chunk_id,
                    document=c.document,
                    page=c.page,
                    section=c.section,
                    content=c.content,
                    chunk_type=c.chunk_type,
                    bm25_rank=bm25_rank_map.get(idx),
                    vector_rank=vec_rank_map.get(idx),
                    rrf_score=rrf_score,
                )
            )
        return results

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _tokenise(text: str) -> list[str]:
        """
        Lowercase, strip punctuation, split on whitespace.

        Financial abbreviations like "IRA", "FDIC", "AML" survive because only
        non-alphanumeric characters are removed; the resulting lowercase tokens
        ("ira", "fdic", "aml") still allow BM25 to match queries containing them.
        """
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return text.split()
