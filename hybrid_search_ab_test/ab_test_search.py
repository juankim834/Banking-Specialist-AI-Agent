"""
A/B Testing Pipeline: Hybrid Search vs BM25-Only Baseline
==========================================================

Tests whether HybridSearchEngine (BM25 + Dense Vector + RRF) outperforms
plain BM25 retrieval on a financial-domain RAG benchmark.

Metrics
-------
- MRR@k    : Mean Reciprocal Rank — rewards finding the right chunk early
- NDCG@k   : Normalized Discounted Cumulative Gain — graded relevance
- Recall@k : Fraction of relevant chunks recovered in top-k
- Precision@k : Fraction of top-k results that are relevant
- Latency  : Wall-clock retrieval time (ms), reported as mean ± std
- Statistical significance via Wilcoxon signed-rank test (paired, non-parametric)

Usage
-----
1. Prepare your test set (see EvalSample below).
2. Instantiate and index your HybridSearchEngine as normal.
3. Run:

    runner = ABTestRunner(engine, test_samples, top_k=5)
    report = runner.run()
    report.print_summary()
    report.save_json("results/ab_report.json")

Dependencies
------------
    pip install rank-bm25 sentence-transformers numpy scipy tqdm
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field, asdict
from typing import Optional
import numpy as np
from scipy.stats import wilcoxon  # type: ignore
from tqdm import tqdm

# Import your engine — adjust path as needed
from tools.hybrid_search import HybridSearchEngine, SearchResult

logger = logging.getLogger(__name__)


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class EvalSample:
    """
    One query with its ground-truth relevant chunk IDs and optional graded scores.

    Fields
    ------
    query_id      : Unique identifier for this query.
    query         : The natural-language or keyword query string.
    relevant_ids  : Set of chunk_ids that are considered relevant.
                    Even a single correct chunk is enough for Recall/MRR.
    relevance_scores : Optional dict mapping chunk_id → relevance grade (0–3).
                    Used for NDCG. If omitted, binary relevance is assumed (1 for
                    relevant, 0 otherwise).
    category      : Optional label (e.g. "keyword", "semantic", "hybrid_intent")
                    for per-category breakdown in the report.
    """
    query_id: str
    query: str
    relevant_ids: set[str]
    relevance_scores: dict[str, float] = field(default_factory=dict)
    category: str = "general"


@dataclass
class PerQueryResult:
    """Stores per-query metrics for both variants."""
    query_id: str
    query: str
    category: str

    # Control (BM25-only)
    ctrl_mrr: float
    ctrl_ndcg: float
    ctrl_recall: float
    ctrl_precision: float
    ctrl_latency_ms: float
    ctrl_top_ids: list[str]

    # Treatment (Hybrid)
    treat_mrr: float
    treat_ndcg: float
    treat_recall: float
    treat_precision: float
    treat_latency_ms: float
    treat_top_ids: list[str]


@dataclass
class ABReport:
    """Aggregated A/B test results with statistical significance."""
    top_k: int
    n_queries: int
    categories: dict[str, int]  # category → count

    # Aggregate metrics
    ctrl_mrr_mean: float
    treat_mrr_mean: float
    ctrl_ndcg_mean: float
    treat_ndcg_mean: float
    ctrl_recall_mean: float
    treat_recall_mean: float
    ctrl_precision_mean: float
    treat_precision_mean: float
    ctrl_latency_mean_ms: float
    treat_latency_mean_ms: float
    ctrl_latency_std_ms: float
    treat_latency_std_ms: float

    # Statistical significance (Wilcoxon signed-rank, two-sided)
    mrr_p_value: float
    ndcg_p_value: float
    recall_p_value: float

    # Relative lifts
    mrr_lift_pct: float
    ndcg_lift_pct: float
    recall_lift_pct: float

    # Per-query breakdown
    per_query: list[PerQueryResult] = field(default_factory=list)

    # Category breakdowns
    category_metrics: dict = field(default_factory=dict)

    def print_summary(self) -> None:
        sep = "─" * 64
        print(f"\n{'═' * 64}")
        print(f"  A/B TEST REPORT  |  top_k={self.top_k}  |  n={self.n_queries} queries")
        print(f"{'═' * 64}")
        print(f"\n{'Metric':<18} {'Control (BM25)':>14} {'Treatment (Hybrid)':>18} {'Lift':>8} {'p-value':>10}")
        print(sep)
        rows = [
            ("MRR@k",       self.ctrl_mrr_mean,       self.treat_mrr_mean,       self.mrr_lift_pct,    self.mrr_p_value),
            ("NDCG@k",      self.ctrl_ndcg_mean,      self.treat_ndcg_mean,      self.ndcg_lift_pct,   self.ndcg_p_value),
            ("Recall@k",    self.ctrl_recall_mean,    self.treat_recall_mean,    self.recall_lift_pct, self.recall_p_value),
            ("Precision@k", self.ctrl_precision_mean, self.treat_precision_mean, None,                 None),
        ]
        for name, ctrl, treat, lift, p in rows:
            lift_str = f"{lift:+.1f}%" if lift is not None else "  —"
            p_str = _p_label(p) if p is not None else "  —"
            print(f"{name:<18} {ctrl:>14.4f} {treat:>18.4f} {lift_str:>8} {p_str:>10}")

        print(sep)
        print(f"\n{'Latency (ms)':<18} {'mean':>6} {'± std':>9}")
        print(f"  Control (BM25)   {self.ctrl_latency_mean_ms:>6.1f} ± {self.ctrl_latency_std_ms:.1f}")
        print(f"  Treatment (Hyb)  {self.treat_latency_mean_ms:>6.1f} ± {self.treat_latency_std_ms:.1f}")

        if self.category_metrics:
            print(f"\n{'Category Breakdown':}")
            print(f"  {'Category':<16} {'n':>4}  {'BM25 MRR':>9}  {'Hybrid MRR':>10}  {'Lift':>7}")
            for cat, m in self.category_metrics.items():
                lift = _lift(m["ctrl_mrr"], m["treat_mrr"])
                print(f"  {cat:<16} {m['n']:>4}  {m['ctrl_mrr']:>9.4f}  {m['treat_mrr']:>10.4f}  {lift:>+7.1f}%")

        print(f"\n{'═' * 64}\n")

    def save_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)
        print(f"[ABReport] Saved → {path}")


# ── Control: BM25-only engine ──────────────────────────────────────────────────

class BM25OnlyEngine:
    """
    Stripped-down BM25-only retriever that shares the same chunk corpus as
    HybridSearchEngine. Used as the control variant in the A/B test.
    """

    def __init__(self, hybrid_engine: HybridSearchEngine) -> None:
        self._engine = hybrid_engine

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        """Run BM25 only, wrap in SearchResult objects to match the interface."""
        if not self._engine._chunks:
            return []
        ranked = self._engine._bm25_search(query, top_k=top_k)
        results = []
        for rank, (idx, score) in enumerate(ranked[:top_k], start=1):
            c = self._engine._chunks[idx]
            results.append(SearchResult(
                chunk_id=c.chunk_id,
                document=c.document,
                page=c.page,
                section=c.section,
                content=c.content,
                chunk_type=c.chunk_type,
                bm25_rank=rank,
                vector_rank=None,
                rrf_score=score,
            ))
        return results


# ── Runner ────────────────────────────────────────────────────────────────────

class ABTestRunner:
    """
    Orchestrates the A/B test between BM25-only (control) and Hybrid (treatment).

    Parameters
    ----------
    engine      : Fully indexed HybridSearchEngine instance.
    samples     : List of EvalSample test cases.
    top_k       : Retrieval depth for all metrics.
    warmup_runs : Number of throw-away queries before timing starts
                  (warms the embedding model's cache).
    """

    def __init__(
        self,
        engine: HybridSearchEngine,
        samples: list[EvalSample],
        top_k: int = 5,
        warmup_runs: int = 3,
    ) -> None:
        self.engine = engine
        self.control = BM25OnlyEngine(engine)
        self.samples = samples
        self.top_k = top_k
        self.warmup_runs = warmup_runs

    def run(self) -> ABReport:
        logger.info("A/B test started — %d queries, top_k=%d", len(self.samples), self.top_k)
        self._warmup()

        per_query: list[PerQueryResult] = []
        for sample in tqdm(self.samples, desc="Evaluating"):
            pq = self._evaluate_sample(sample)
            per_query.append(pq)

        return self._aggregate(per_query)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _warmup(self) -> None:
        warmup_q = "interest rate dividend yield"
        for _ in range(self.warmup_runs):
            self.engine.search(warmup_q, self.top_k)

    def _evaluate_sample(self, sample: EvalSample) -> PerQueryResult:
        k = self.top_k

        # ── Control (BM25-only)
        t0 = time.perf_counter()
        ctrl_results = self.control.search(sample.query, k)
        ctrl_lat = (time.perf_counter() - t0) * 1000

        ctrl_ids = [r.chunk_id for r in ctrl_results]
        ctrl_mrr       = _mrr(ctrl_ids, sample.relevant_ids)
        ctrl_ndcg      = _ndcg(ctrl_ids, sample.relevant_ids, sample.relevance_scores, k)
        ctrl_recall    = _recall(ctrl_ids, sample.relevant_ids)
        ctrl_precision = _precision(ctrl_ids, sample.relevant_ids)

        # ── Treatment (Hybrid)
        t0 = time.perf_counter()
        treat_results = self.engine.search(sample.query, k)
        treat_lat = (time.perf_counter() - t0) * 1000

        treat_ids = [r.chunk_id for r in treat_results]
        treat_mrr       = _mrr(treat_ids, sample.relevant_ids)
        treat_ndcg      = _ndcg(treat_ids, sample.relevant_ids, sample.relevance_scores, k)
        treat_recall    = _recall(treat_ids, sample.relevant_ids)
        treat_precision = _precision(treat_ids, sample.relevant_ids)

        return PerQueryResult(
            query_id=sample.query_id,
            query=sample.query,
            category=sample.category,
            ctrl_mrr=ctrl_mrr,
            ctrl_ndcg=ctrl_ndcg,
            ctrl_recall=ctrl_recall,
            ctrl_precision=ctrl_precision,
            ctrl_latency_ms=ctrl_lat,
            ctrl_top_ids=ctrl_ids,
            treat_mrr=treat_mrr,
            treat_ndcg=treat_ndcg,
            treat_recall=treat_recall,
            treat_precision=treat_precision,
            treat_latency_ms=treat_lat,
            treat_top_ids=treat_ids,
        )

    def _aggregate(self, per_query: list[PerQueryResult]) -> ABReport:
        ctrl_mrrs  = np.array([pq.ctrl_mrr  for pq in per_query])
        treat_mrrs = np.array([pq.treat_mrr for pq in per_query])

        ctrl_ndcgs  = np.array([pq.ctrl_ndcg  for pq in per_query])
        treat_ndcgs = np.array([pq.treat_ndcg for pq in per_query])

        ctrl_recalls  = np.array([pq.ctrl_recall  for pq in per_query])
        treat_recalls = np.array([pq.treat_recall for pq in per_query])

        ctrl_precs  = np.array([pq.ctrl_precision  for pq in per_query])
        treat_precs = np.array([pq.treat_precision for pq in per_query])

        ctrl_lats  = np.array([pq.ctrl_latency_ms  for pq in per_query])
        treat_lats = np.array([pq.treat_latency_ms for pq in per_query])

        def _wp(a: np.ndarray, b: np.ndarray) -> float:
            diff = b - a
            if np.all(diff == 0):
                return 1.0
            try:
                _, p = wilcoxon(diff, alternative="two-sided")
                return float(p)
            except Exception:
                return 1.0

        mrr_p    = _wp(ctrl_mrrs, treat_mrrs)
        ndcg_p   = _wp(ctrl_ndcgs, treat_ndcgs)
        recall_p = _wp(ctrl_recalls, treat_recalls)

        # Category breakdown
        cats: dict[str, dict] = {}
        for pq in per_query:
            c = pq.category
            if c not in cats:
                cats[c] = {"n": 0, "ctrl_mrr": 0.0, "treat_mrr": 0.0,
                           "ctrl_ndcg": 0.0, "treat_ndcg": 0.0}
            cats[c]["n"] += 1
            cats[c]["ctrl_mrr"]   += pq.ctrl_mrr
            cats[c]["treat_mrr"]  += pq.treat_mrr
            cats[c]["ctrl_ndcg"]  += pq.ctrl_ndcg
            cats[c]["treat_ndcg"] += pq.treat_ndcg
        for c, m in cats.items():
            n = m["n"]
            m["ctrl_mrr"]   /= n
            m["treat_mrr"]  /= n
            m["ctrl_ndcg"]  /= n
            m["treat_ndcg"] /= n

        return ABReport(
            top_k=self.top_k,
            n_queries=len(per_query),
            categories={c: m["n"] for c, m in cats.items()},
            ctrl_mrr_mean=float(ctrl_mrrs.mean()),
            treat_mrr_mean=float(treat_mrrs.mean()),
            ctrl_ndcg_mean=float(ctrl_ndcgs.mean()),
            treat_ndcg_mean=float(treat_ndcgs.mean()),
            ctrl_recall_mean=float(ctrl_recalls.mean()),
            treat_recall_mean=float(treat_recalls.mean()),
            ctrl_precision_mean=float(ctrl_precs.mean()),
            treat_precision_mean=float(treat_precs.mean()),
            ctrl_latency_mean_ms=float(ctrl_lats.mean()),
            treat_latency_mean_ms=float(treat_lats.mean()),
            ctrl_latency_std_ms=float(ctrl_lats.std()),
            treat_latency_std_ms=float(treat_lats.std()),
            mrr_p_value=mrr_p,
            ndcg_p_value=ndcg_p,
            recall_p_value=recall_p,
            mrr_lift_pct=_lift(float(ctrl_mrrs.mean()), float(treat_mrrs.mean())),
            ndcg_lift_pct=_lift(float(ctrl_ndcgs.mean()), float(treat_ndcgs.mean())),
            recall_lift_pct=_lift(float(ctrl_recalls.mean()), float(treat_recalls.mean())),
            per_query=per_query,
            category_metrics=cats,
        )


# ── Metric functions ──────────────────────────────────────────────────────────

def _mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / rank
    return 0.0


def _ndcg(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    relevance_scores: dict[str, float],
    k: int,
) -> float:
    def gain(rid: str) -> float:
        if relevance_scores:
            return relevance_scores.get(rid, 0.0)
        return 1.0 if rid in relevant_ids else 0.0

    dcg = sum(
        gain(rid) / math.log2(rank + 1)
        for rank, rid in enumerate(retrieved_ids[:k], start=1)
    )
    ideal_gains = sorted(
        (relevance_scores.get(rid, 1.0) for rid in relevant_ids),
        reverse=True,
    )
    idcg = sum(
        g / math.log2(rank + 1)
        for rank, g in enumerate(ideal_gains[:k], start=1)
    )
    return dcg / idcg if idcg > 0 else 0.0


def _recall(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    if not relevant_ids:
        return 0.0
    return len(set(retrieved_ids) & relevant_ids) / len(relevant_ids)


def _precision(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    if not retrieved_ids:
        return 0.0
    return len(set(retrieved_ids) & relevant_ids) / len(retrieved_ids)


def _lift(ctrl: float, treat: float) -> float:
    return ((treat - ctrl) / ctrl * 100) if ctrl > 0 else 0.0


def _p_label(p: float) -> str:
    if p < 0.001:
        return f"{p:.4f} ***"
    if p < 0.01:
        return f"{p:.4f} **"
    if p < 0.05:
        return f"{p:.4f} *"
    return f"{p:.4f}"


# ── Example test set (financial domain) ───────────────────────────────────────
# Replace chunk_ids with your actual indexed chunk IDs.

EXAMPLE_EVAL_SET: list[EvalSample] = [
    # --- Keyword-heavy queries (BM25 should shine) ---
    EvalSample(
        query_id="kw-001",
        query="Roth IRA 5-year rule",
        relevant_ids={"chunk_roth_5yr_rule", "chunk_roth_overview"},
        relevance_scores={"chunk_roth_5yr_rule": 3, "chunk_roth_overview": 1},
        category="keyword",
    ),
    EvalSample(
        query_id="kw-002",
        query="Regulation D exemption securities offering",
        relevant_ids={"chunk_reg_d_overview", "chunk_reg_d_limits"},
        category="keyword",
    ),
    EvalSample(
        query_id="kw-003",
        query="Basel III Tier-1 capital ratio",
        relevant_ids={"chunk_basel3_tier1"},
        category="keyword",
    ),
    EvalSample(
        query_id="kw-004",
        query="FDIC deposit insurance limit 2024",
        relevant_ids={"chunk_fdic_limits"},
        category="keyword",
    ),
    EvalSample(
        query_id="kw-005",
        query="SEP IRA contribution limits small business",
        relevant_ids={"chunk_sep_ira_contrib", "chunk_sep_ira_overview"},
        relevance_scores={"chunk_sep_ira_contrib": 3, "chunk_sep_ira_overview": 2},
        category="keyword",
    ),

    # --- Semantic queries (Vector should shine) ---
    EvalSample(
        query_id="sem-001",
        query="client wants steady income without market volatility",
        relevant_ids={"chunk_dividend_portfolio", "chunk_bond_ladder", "chunk_cd_strategy"},
        relevance_scores={"chunk_dividend_portfolio": 3, "chunk_bond_ladder": 2, "chunk_cd_strategy": 1},
        category="semantic",
    ),
    EvalSample(
        query_id="sem-002",
        query="how do I protect my savings from inflation",
        relevant_ids={"chunk_tips_bonds", "chunk_i_bonds", "chunk_reits"},
        category="semantic",
    ),
    EvalSample(
        query_id="sem-003",
        query="business owner planning for retirement tax efficiently",
        relevant_ids={"chunk_sep_ira_contrib", "chunk_solo_401k", "chunk_defined_benefit"},
        relevance_scores={"chunk_sep_ira_contrib": 2, "chunk_solo_401k": 3, "chunk_defined_benefit": 2},
        category="semantic",
    ),
    EvalSample(
        query_id="sem-004",
        query="young employee just starting to invest",
        relevant_ids={"chunk_401k_basics", "chunk_roth_overview", "chunk_compound_interest"},
        category="semantic",
    ),
    EvalSample(
        query_id="sem-005",
        query="reduce estate taxes for wealthy clients",
        relevant_ids={"chunk_irrevocable_trust", "chunk_gifting_strategy", "chunk_estate_tax"},
        category="semantic",
    ),

    # --- Hybrid intent (both methods contribute) ---
    EvalSample(
        query_id="hyb-001",
        query="401k early withdrawal penalty exceptions hardship",
        relevant_ids={"chunk_401k_early_withdrawal", "chunk_hardship_distribution"},
        relevance_scores={"chunk_401k_early_withdrawal": 3, "chunk_hardship_distribution": 2},
        category="hybrid_intent",
    ),
    EvalSample(
        query_id="hyb-002",
        query="can I contribute to both traditional and Roth IRA same year",
        relevant_ids={"chunk_ira_contribution_limits", "chunk_roth_overview"},
        category="hybrid_intent",
    ),
    EvalSample(
        query_id="hyb-003",
        query="AML suspicious activity reporting thresholds bank",
        relevant_ids={"chunk_aml_sar", "chunk_bsa_requirements"},
        category="hybrid_intent",
    ),
    EvalSample(
        query_id="hyb-004",
        query="required minimum distribution age rule change SECURE Act",
        relevant_ids={"chunk_rmd_rules", "chunk_secure_act_2"},
        relevance_scores={"chunk_rmd_rules": 3, "chunk_secure_act_2": 2},
        category="hybrid_intent",
    ),
    EvalSample(
        query_id="hyb-005",
        query="net unrealized appreciation NUA company stock 401k distribution",
        relevant_ids={"chunk_nua_strategy"},
        category="hybrid_intent",
    ),
]


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # ── 1. Build and index your engine (replace with your real chunks)
    engine = HybridSearchEngine(embedding_model="all-MiniLM-L6-v2")

    # TODO: Replace with real DocumentChunk objects from your pipeline:
    #   from your_pipeline import build_chunks
    #   chunks = build_chunks("uploads/p560b.pdf")
    #   engine.index(chunks)
    #
    # For demonstration we skip indexing; the runner will return empty results.
    # In practice you MUST index before running the test.

    # ── 2. Load eval set (replace with your curated samples)
    eval_set = EXAMPLE_EVAL_SET

    # ── 3. Run the A/B test
    runner = ABTestRunner(engine, eval_set, top_k=5, warmup_runs=3)
    report = runner.run()

    # ── 4. Print and save
    report.print_summary()
    os.makedirs("results", exist_ok=True)
    report.save_json("results/ab_report.json")