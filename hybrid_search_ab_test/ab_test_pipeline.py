"""
A/B Testing Pipeline: HybridSearchEngine vs BM25-Only Baseline
===============================================================

Tests whether the hybrid BM25 + dense vector search (System B) is meaningfully
better than a BM25-only baseline (System A) for a Banking AI Agent's RAG pipeline.

Metrics
-------
  • Precision@K   — fraction of top-K results that are relevant
  • Recall@K      — fraction of all relevant docs found in top-K
  • MRR           — Mean Reciprocal Rank (how high is the first correct hit?)
  • NDCG@K        — Normalized Discounted Cumulative Gain (rank-order quality)
  • Latency       — wall-clock search time in milliseconds

Statistical Validity
---------------------
  • Paired Wilcoxon signed-rank test (non-parametric, safe for small samples)
  • Effect size via Cohen's d
  • 95% confidence intervals via bootstrap resampling

Usage
-----
  python ab_test_pipeline.py                         # runs with synthetic corpus
  python ab_test_pipeline.py --results results.json  # export results to JSON
  python ab_test_pipeline.py --top-k 10 --runs 3    # custom settings
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from statistics import mean, median, stdev
from typing import Any, Optional

import numpy as np
import tools.hybrid_search

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Data Models
# ══════════════════════════════════════════════════════════════════════════════

from hybrid_search_ab_test.models import DocumentChunk, QueryCase

@dataclass
class TrialResult:
    """Metrics for one system on one query."""
    query_id: str
    system: str           # "A_bm25" | "B_hybrid"
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    latency_ms: float
    top_k_ids: list[str]  # chunk IDs returned, in order


@dataclass
class ABReport:
    """Aggregated A/B test report."""
    system_a_name: str
    system_b_name: str
    top_k: int
    n_queries: int
    n_runs: int

    # Per-metric aggregates  {metric: {system: {mean, median, std, ci95}}}
    aggregates: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)

    # Statistical tests       {metric: {statistic, p_value, significant, effect_size_d}}
    stat_tests: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Per-category breakdown  {category: {metric: {system: mean}}}
    category_breakdown: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)

    # Win/tie/loss counts     {metric: {wins_B, ties, wins_A}}
    win_counts: dict[str, dict[str, int]] = field(default_factory=dict)

    # All raw trial results (for export / debugging)
    raw_results: list[TrialResult] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# System A — BM25-Only Baseline
# ══════════════════════════════════════════════════════════════════════════════

class BM25OnlyEngine:
    """
    Pure BM25 keyword search — the traditional baseline.
    Uses the same tokeniser as HybridSearchEngine for a fair comparison.
    """

    def __init__(self) -> None:
        self._chunks: list[DocumentChunk] = []
        self._bm25: Optional[Any] = None

    def index(self, chunks: list[DocumentChunk]) -> None:
        self._chunks.extend(chunks)
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("pip install rank-bm25")
        tokenised = [self._tokenise(c.content) for c in self._chunks]
        self._bm25 = BM25Okapi(tokenised)

    def search(self, query: str, top_k: int = 5) -> list[str]:
        """Returns ordered list of chunk_ids."""
        assert self._bm25 is not None
        tokens = self._tokenise(query)
        scores: np.ndarray = self._bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores.tolist()), key=lambda x: x[1], reverse=True)
        return [self._chunks[i].chunk_id for i, _ in ranked[:top_k]]

    def clear(self) -> None:
        self._chunks.clear()
        self._bm25 = None

    @staticmethod
    def _tokenise(text: str) -> list[str]:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return text.split()


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    hits = sum(1 for doc_id in retrieved[:k] if doc_id in relevant)
    return hits / k if k > 0 else 0.0


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for doc_id in retrieved[:k] if doc_id in relevant)
    return hits / len(relevant)


def mean_reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], grades: dict[str, int], k: int) -> float:
    """Normalized Discounted Cumulative Gain."""
    def dcg(ids: list[str]) -> float:
        return sum(
            (2 ** grades.get(doc_id, 0) - 1) / math.log2(rank + 1)
            for rank, doc_id in enumerate(ids[:k], start=1)
        )

    actual_dcg = dcg(retrieved)
    ideal_ids = sorted(grades.keys(), key=lambda x: grades[x], reverse=True)
    ideal_dcg = dcg(ideal_ids)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Statistical Tests
# ══════════════════════════════════════════════════════════════════════════════

def wilcoxon_signed_rank(
    scores_a: list[float],
    scores_b: list[float],
) -> dict[str, Any]:
    """
    Paired Wilcoxon signed-rank test.
    Returns statistic, p_value, and whether the result is significant (p < 0.05).

    Falls back to a normal approximation when scipy is unavailable.
    """
    differences = [b - a for a, b in zip(scores_a, scores_b)]
    nonzero = [(i, d) for i, d in enumerate(differences) if abs(d) > 1e-9]

    if not nonzero:
        return {"statistic": 0.0, "p_value": 1.0, "significant": False}

    ranks = _rank_by_abs(nonzero)
    W_plus = sum(r for (i, d), r in zip(nonzero, ranks) if d > 0)
    W_minus = sum(r for (i, d), r in zip(nonzero, ranks) if d < 0)
    W = min(W_plus, W_minus)
    n = len(nonzero)

    # Normal approximation (valid for n ≥ 10; acceptable for smaller n)
    mu = n * (n + 1) / 4
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (W - mu) / sigma if sigma > 0 else 0.0
    p_value = 2 * _normal_sf(abs(z))   # two-tailed

    return {
        "statistic": W,
        "p_value": round(p_value, 4),
        "significant": p_value < 0.05,
        "z_score": round(z, 3),
    }


def cohens_d(scores_a: list[float], scores_b: list[float]) -> float:
    """Effect size: small=0.2, medium=0.5, large=0.8."""
    if len(scores_a) < 2:
        return 0.0
    diff = [b - a for a, b in zip(scores_a, scores_b)]
    mu = mean(diff)
    sd = stdev(diff)
    return mu / sd if sd > 0 else 0.0


def bootstrap_ci(values: list[float], n_boot: int = 2000, ci: float = 0.95) -> tuple[float, float]:
    """Bootstrap confidence interval for the mean."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(42)
    boot_means = [mean(rng.choices(values, k=len(values))) for _ in range(n_boot)]
    boot_means.sort()
    lo = int((1 - ci) / 2 * n_boot)
    hi = int((1 + ci) / 2 * n_boot)
    return (round(boot_means[lo], 4), round(boot_means[hi], 4))


def _rank_by_abs(pairs: list) -> list[float]:
    indexed = sorted(enumerate(pairs), key=lambda x: abs(x[1][1]))
    n = len(indexed)
    ranks: list[float] = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and abs(indexed[j][1][1]) == abs(indexed[j + 1][1][1]):
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def _normal_sf(z: float) -> float:
    """Survival function (1 - CDF) of N(0,1) using math.erfc."""
    return 0.5 * math.erfc(z / math.sqrt(2))


# ══════════════════════════════════════════════════════════════════════════════
# Corpus & Query Set Builder (for Standalone Testing)
# ══════════════════════════════════════════════════════════════════════════════

FINANCIAL_CORPUS: list[dict] = [
    # ── Retirement / IRA
    {
        "chunk_id": "c001", "document": "IRS_Pub_560.pdf", "page": 4,
        "section": "Roth IRA Overview",
        "content": "A Roth IRA is an individual retirement account that provides tax-free growth "
                   "and tax-free withdrawals in retirement. Contributions are made with after-tax dollars. "
                   "The 5-year rule requires the account to be open for at least five years before "
                   "tax-free withdrawals of earnings can be made.",
    },
    {
        "chunk_id": "c002", "document": "IRS_Pub_560.pdf", "page": 6,
        "section": "Traditional IRA Rules",
        "content": "Traditional IRA contributions may be tax-deductible depending on your income "
                   "and whether you or your spouse are covered by a workplace retirement plan. "
                   "Required minimum distributions must begin at age 73.",
    },
    {
        "chunk_id": "c003", "document": "IRS_Pub_560.pdf", "page": 9,
        "section": "SEP-IRA Contribution Limits",
        "content": "A Simplified Employee Pension (SEP) IRA allows self-employed individuals to "
                   "contribute up to 25% of net self-employment income, or $66,000 for 2023, "
                   "whichever is less. Contributions are tax-deductible.",
    },
    {
        "chunk_id": "c004", "document": "IRS_Pub_560.pdf", "page": 12,
        "section": "401k Contribution Limits",
        "content": "The 401(k) elective deferral limit is $22,500 for 2023, with an additional "
                   "$7,500 catch-up contribution allowed for employees aged 50 or older. "
                   "Employer matching is not counted against the employee deferral limit.",
    },
    # ── Regulation / Compliance
    {
        "chunk_id": "c005", "document": "Basel_III_Summary.pdf", "page": 2,
        "section": "Tier-1 Capital Requirements",
        "content": "Under Basel III, banks must maintain a Common Equity Tier 1 (CET1) capital ratio "
                   "of at least 4.5% of risk-weighted assets. The Tier-1 capital ratio minimum is 6%. "
                   "The total capital ratio minimum is 8%.",
    },
    {
        "chunk_id": "c006", "document": "Regulation_D.pdf", "page": 1,
        "section": "Savings Account Withdrawal Limits",
        "content": "Regulation D historically limited savings and money market accounts to six "
                   "convenient withdrawals or transfers per monthly statement cycle. "
                   "The Federal Reserve removed this limit in 2020, though many banks still enforce it.",
    },
    {
        "chunk_id": "c007", "document": "AML_Policy.pdf", "page": 3,
        "section": "Know Your Customer (KYC)",
        "content": "Anti-Money Laundering (AML) compliance requires banks to implement Know Your "
                   "Customer (KYC) procedures. Suspicious Activity Reports (SARs) must be filed "
                   "for transactions that may involve money laundering.",
    },
    {
        "chunk_id": "c008", "document": "AML_Policy.pdf", "page": 5,
        "section": "Currency Transaction Reports",
        "content": "Currency Transaction Reports (CTRs) must be filed for cash transactions exceeding "
                   "$10,000. Structuring transactions to avoid this threshold is illegal and "
                   "constitutes a separate criminal offence.",
    },
    # ── Investments / Portfolio
    {
        "chunk_id": "c009", "document": "Investment_Guide.pdf", "page": 7,
        "section": "Dividend Investing",
        "content": "Dividend-paying stocks provide investors with regular income through quarterly "
                   "distributions. High-dividend ETFs such as those tracking the S&P 500 Dividend "
                   "Aristocrats offer steady cash flow, making them suitable for income-oriented clients.",
    },
    {
        "chunk_id": "c010", "document": "Investment_Guide.pdf", "page": 11,
        "section": "Bond Laddering",
        "content": "A bond ladder is a portfolio of fixed-income securities with staggered maturities. "
                   "As each bond matures, the proceeds are reinvested into a new long-term bond, "
                   "providing regular income while managing interest-rate risk.",
    },
    {
        "chunk_id": "c011", "document": "Investment_Guide.pdf", "page": 14,
        "section": "Risk Tolerance Assessment",
        "content": "Before recommending any investment, a financial advisor must assess the client's "
                   "risk tolerance, investment horizon, and liquidity needs. Conservative clients "
                   "are typically directed toward money market funds, CDs, and government bonds.",
    },
    {
        "chunk_id": "c012", "document": "Investment_Guide.pdf", "page": 18,
        "section": "FDIC Insurance Limits",
        "content": "The FDIC insures deposits up to $250,000 per depositor, per insured bank, "
                   "per account ownership category. Joint accounts receive up to $500,000 in coverage. "
                   "Retirement accounts such as IRAs receive separate $250,000 coverage.",
    },
    # ── Semantic / paraphrase targets
    {
        "chunk_id": "c013", "document": "Client_Handbook.pdf", "page": 2,
        "section": "Steady Income Options",
        "content": "Clients seeking predictable, recurring cash flows should consider dividend "
                   "equity portfolios, bond ladders, and annuity products. These instruments "
                   "generate income without requiring the sale of principal assets.",
    },
    {
        "chunk_id": "c014", "document": "Client_Handbook.pdf", "page": 5,
        "section": "Capital Preservation",
        "content": "Clients with low risk appetite who prioritise protecting their principal "
                   "should focus on FDIC-insured deposits, Treasury bills, and investment-grade "
                   "short-duration bonds.",
    },
    {
        "chunk_id": "c015", "document": "Client_Handbook.pdf", "page": 8,
        "section": "Tax-Efficient Retirement Savings",
        "content": "Maximising tax-advantaged accounts — Roth IRA, Traditional IRA, and 401(k) — "
                   "reduces current or future tax liability. Account selection should depend on "
                   "the client's current versus anticipated retirement tax bracket.",
    },
    # ── Table chunks
    {
        "chunk_id": "c016", "document": "Rate_Sheet.pdf", "page": 1,
        "section": "CD Rate Table",
        "content": "| Term    | APY   |\n|---------|-------|\n| 3-month | 4.85% |\n"
                   "| 6-month | 5.00% |\n| 12-month | 5.10% |\n| 24-month | 4.75% |",
        "chunk_type": "table",
    },
    {
        "chunk_id": "c017", "document": "Rate_Sheet.pdf", "page": 2,
        "section": "Savings Account Rates",
        "content": "| Account Type          | APY   |\n|-----------------------|-------|\n"
                   "| High-Yield Savings    | 4.60% |\n| Money Market          | 4.45% |\n"
                   "| Regular Savings       | 0.01% |",
        "chunk_type": "table",
    },
    # ── Edge cases
    {
        "chunk_id": "c018", "document": "Misc_FAQ.pdf", "page": 1,
        "section": "Wire Transfer Fees",
        "content": "Domestic wire transfers cost $25 per transfer. International wire transfers "
                   "cost $45. Incoming wires are free for Premium account holders.",
    },
    {
        "chunk_id": "c019", "document": "Misc_FAQ.pdf", "page": 3,
        "section": "Overdraft Policy",
        "content": "Overdraft protection transfers funds from a linked savings account at no charge. "
                   "Standard overdraft fees are $35 per item, with a maximum of 3 fees per day.",
    },
    {
        "chunk_id": "c020", "document": "Misc_FAQ.pdf", "page": 5,
        "section": "Mortgage Pre-qualification",
        "content": "Pre-qualification is a preliminary review of creditworthiness based on "
                   "self-reported income and assets. A hard credit inquiry is only performed "
                   "during a full mortgage application.",
    },
]

# Build DocumentChunk objects
CORPUS: list[DocumentChunk] = [
    DocumentChunk(
        chunk_id=d["chunk_id"],
        document=d["document"],
        page=d["page"],
        section=d["section"],
        content=d["content"],
        chunk_type=d.get("chunk_type", "text"),
    )
    for d in FINANCIAL_CORPUS
]

# ── Query evaluation set
QUERY_SET: list[QueryCase] = [
    # ── Keyword / exact-match queries (BM25 should excel)
    QueryCase(
        query_id="q01",
        query="Roth IRA 5-year rule",
        relevant_ids={"c001", "c015"},
        relevance_grades={"c001": 2, "c015": 1},
        category="keyword",
    ),
    QueryCase(
        query_id="q02",
        query="Basel III Tier-1 capital ratio requirements",
        relevant_ids={"c005"},
        relevance_grades={"c005": 2},
        category="keyword",
    ),
    QueryCase(
        query_id="q03",
        query="Regulation D savings withdrawal limit",
        relevant_ids={"c006"},
        relevance_grades={"c006": 2},
        category="keyword",
    ),
    QueryCase(
        query_id="q04",
        query="401k catch-up contribution age 50",
        relevant_ids={"c004"},
        relevance_grades={"c004": 2},
        category="keyword",
    ),
    QueryCase(
        query_id="q05",
        query="CTR currency transaction report $10000",
        relevant_ids={"c008"},
        relevance_grades={"c008": 2, "c007": 1},
        category="keyword",
    ),
    # ── Semantic / paraphrase queries (vector search should excel)
    QueryCase(
        query_id="q06",
        query="My client wants steady passive income from their portfolio",
        relevant_ids={"c009", "c010", "c013"},
        relevance_grades={"c013": 2, "c009": 2, "c010": 1},
        category="semantic",
    ),
    QueryCase(
        query_id="q07",
        query="Safe options for someone worried about losing money",
        relevant_ids={"c011", "c012", "c014"},
        relevance_grades={"c014": 2, "c011": 1, "c012": 1},
        category="semantic",
    ),
    QueryCase(
        query_id="q08",
        query="How can my client minimise taxes on retirement savings?",
        relevant_ids={"c001", "c002", "c003", "c015"},
        relevance_grades={"c015": 2, "c001": 1, "c002": 1, "c003": 1},
        category="semantic",
    ),
    QueryCase(
        query_id="q09",
        query="Client is worried about bank deposit safety",
        relevant_ids={"c012", "c014"},
        relevance_grades={"c012": 2, "c014": 1},
        category="semantic",
    ),
    QueryCase(
        query_id="q10",
        query="Stopping criminals from hiding illegal money",
        relevant_ids={"c007", "c008"},
        relevance_grades={"c007": 2, "c008": 2},
        category="semantic",
    ),
    # ── Mixed queries
    QueryCase(
        query_id="q11",
        query="IRA contribution limits for self-employed small business",
        relevant_ids={"c003", "c002", "c015"},
        relevance_grades={"c003": 2, "c002": 1, "c015": 1},
        category="mixed",
    ),
    QueryCase(
        query_id="q12",
        query="FDIC insurance coverage joint accounts retirement",
        relevant_ids={"c012", "c014"},
        relevance_grades={"c012": 2, "c014": 1},
        category="mixed",
    ),
    # ── Table queries
    QueryCase(
        query_id="q13",
        query="Current CD rates 12 month APY",
        relevant_ids={"c016"},
        relevance_grades={"c016": 2, "c017": 1},
        category="table",
    ),
    QueryCase(
        query_id="q14",
        query="High yield savings account interest rate",
        relevant_ids={"c017"},
        relevance_grades={"c017": 2, "c016": 1},
        category="table",
    ),
    # ── Edge cases
    QueryCase(
        query_id="q15",
        query="Wire transfer fees international domestic",
        relevant_ids={"c018"},
        relevance_grades={"c018": 2},
        category="mixed",
    ),
]


# ══════════════════════════════════════════════════════════════════════════════
# A/B Test Runner
# ══════════════════════════════════════════════════════════════════════════════

class ABTestPipeline:
    """
    Runs a controlled A/B test between a BM25-only baseline (System A)
    and the HybridSearchEngine (System B).
    """

    METRICS = ["precision_at_k", "recall_at_k", "mrr", "ndcg_at_k"]

    def __init__(
        self,
        corpus: list[DocumentChunk],
        queries: list[QueryCase],
        top_k: int = 5,
        n_runs: int = 3,
    ) -> None:
        self.corpus = corpus
        self.queries = queries
        self.top_k = top_k
        self.n_runs = n_runs

        self._engine_a = BM25OnlyEngine()
        self._engine_b: Optional[Any] = None   # HybridSearchEngine, lazy-imported

    # ── Public API

    def run(self) -> ABReport:
        logger.info("=" * 60)
        logger.info("A/B Test: BM25-Only  vs  Hybrid (BM25 + Vector + RRF)")
        logger.info("Corpus: %d chunks | Queries: %d | top_k=%d | runs=%d",
                    len(self.corpus), len(self.queries), self.top_k, self.n_runs)
        logger.info("=" * 60)

        self._build_indices()

        all_results: list[TrialResult] = []

        for run in range(1, self.n_runs + 1):
            logger.info("── Run %d/%d", run, self.n_runs)
            for qc in self.queries:
                all_results.append(self._eval_query(qc, "A_bm25"))
                all_results.append(self._eval_query(qc, "B_hybrid"))

        return self._compile_report(all_results)

    # ── Index construction

    def _build_indices(self) -> None:
        logger.info("Building System A (BM25-only) index …")
        self._engine_a.index(self.corpus)

        logger.info("Building System B (Hybrid) index …")
        try:
            from tools.hybrid_search import HybridSearchEngine
            self._engine_b = HybridSearchEngine()
            self._engine_b.index(self.corpus)
        except ImportError:
            logger.warning(
                "hybrid_search.py not importable — using a simulated Hybrid engine "
                "that combines BM25 with randomised vector-rank perturbation "
                "(suitable for pipeline smoke-testing; replace with real engine for "
                "production evaluation)."
            )
            self._engine_b = _SimulatedHybridEngine(self._engine_a)

        logger.info("Indices ready.")

    # ── Per-query evaluation

    def _eval_query(self, qc: QueryCase, system: str) -> TrialResult:
        t0 = time.perf_counter()
        if system == "A_bm25":
            retrieved = self._engine_a.search(qc.query, top_k=self.top_k)
        else:
            raw = self._engine_b.search(qc.query, top_k=self.top_k)
            # HybridSearchEngine returns SearchResult objects; extract chunk_ids
            retrieved = (
                [r.chunk_id for r in raw]
                if raw and hasattr(raw[0], "chunk_id")
                else raw
            )
        latency_ms = (time.perf_counter() - t0) * 1000

        return TrialResult(
            query_id=qc.query_id,
            system=system,
            precision_at_k=precision_at_k(retrieved, qc.relevant_ids, self.top_k),
            recall_at_k=recall_at_k(retrieved, qc.relevant_ids, self.top_k),
            mrr=mean_reciprocal_rank(retrieved, qc.relevant_ids),
            ndcg_at_k=ndcg_at_k(retrieved, qc.relevance_grades, self.top_k),
            latency_ms=latency_ms,
            top_k_ids=retrieved,
        )

    # ── Report compilation

    def _compile_report(self, results: list[TrialResult]) -> ABReport:
        # Group results: {system: {query_id: [TrialResult]}}
        by_system: dict[str, dict[str, list[TrialResult]]] = defaultdict(lambda: defaultdict(list))
        for r in results:
            by_system[r.system][r.query_id].append(r)

        # Average repeated runs per (system, query)
        avg: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
        for system, qmap in by_system.items():
            for qid, trials in qmap.items():
                avg[system][qid] = {
                    m: mean(getattr(t, m) for t in trials)
                    for m in [*self.METRICS, "latency_ms"]
                }

        query_ids = [qc.query_id for qc in self.queries]
        categories = {qc.query_id: qc.category for qc in self.queries}

        report = ABReport(
            system_a_name="BM25-Only",
            system_b_name="Hybrid (BM25 + Vector + RRF)",
            top_k=self.top_k,
            n_queries=len(self.queries),
            n_runs=self.n_runs,
            raw_results=results,
        )

        # ── Aggregates + CI
        for metric in [*self.METRICS, "latency_ms"]:
            report.aggregates[metric] = {}
            for system in ("A_bm25", "B_hybrid"):
                vals = [avg[system][qid][metric] for qid in query_ids if qid in avg[system]]
                if not vals:
                    continue
                lo, hi = bootstrap_ci(vals)
                report.aggregates[metric][system] = {
                    "mean": round(mean(vals), 4),
                    "median": round(median(vals), 4),
                    "std": round(stdev(vals) if len(vals) > 1 else 0.0, 4),
                    "ci95_lo": lo,
                    "ci95_hi": hi,
                }

        # ── Statistical tests
        for metric in self.METRICS:
            a_vals = [avg["A_bm25"][qid][metric] for qid in query_ids if qid in avg["A_bm25"]]
            b_vals = [avg["B_hybrid"][qid][metric] for qid in query_ids if qid in avg["B_hybrid"]]
            test = wilcoxon_signed_rank(a_vals, b_vals)
            test["effect_size_d"] = round(cohens_d(a_vals, b_vals), 3)
            test["effect_label"] = _effect_label(test["effect_size_d"])
            report.stat_tests[metric] = test

        # ── Win / tie / loss
        for metric in self.METRICS:
            wins_b = ties = wins_a = 0
            for qid in query_ids:
                a = avg["A_bm25"].get(qid, {}).get(metric, 0.0)
                b = avg["B_hybrid"].get(qid, {}).get(metric, 0.0)
                if b > a + 1e-6:
                    wins_b += 1
                elif a > b + 1e-6:
                    wins_a += 1
                else:
                    ties += 1
            report.win_counts[metric] = {"wins_B": wins_b, "ties": ties, "wins_A": wins_a}

        # ── Category breakdown
        for cat in set(categories.values()):
            cat_qids = [qid for qid in query_ids if categories[qid] == cat]
            report.category_breakdown[cat] = {}
            for metric in self.METRICS:
                report.category_breakdown[cat][metric] = {}
                for system in ("A_bm25", "B_hybrid"):
                    vals = [avg[system][qid][metric] for qid in cat_qids if qid in avg[system]]
                    report.category_breakdown[cat][metric][system] = round(mean(vals), 4) if vals else 0.0

        self._print_report(report)
        return report

    # ── Pretty print

    def _print_report(self, r: ABReport) -> None:
        sep = "─" * 60
        print(f"\n{'═' * 60}")
        print(f"  A/B TEST RESULTS  — top_{r.top_k}, {r.n_queries} queries, {r.n_runs} run(s)")
        print(f"{'═' * 60}")

        print(f"\n{'Metric':<18} {'System A (BM25)':>18} {'System B (Hybrid)':>18}  Δ       p-val  Sig?")
        print(sep)
        for metric in self.METRICS:
            agg = r.aggregates.get(metric, {})
            a_mean = agg.get("A_bm25", {}).get("mean", 0.0)
            b_mean = agg.get("B_hybrid", {}).get("mean", 0.0)
            stat = r.stat_tests.get(metric, {})
            delta = b_mean - a_mean
            sig = "✓" if stat.get("significant") else "✗"
            pval = stat.get("p_value", 1.0)
            print(f"  {metric:<16} {a_mean:>18.4f} {b_mean:>18.4f}  {delta:+.4f}  {pval:.4f}  {sig}")

        print(f"\n  Win / Tie / Loss (System B vs A):")
        for metric, wc in r.win_counts.items():
            bar_b = "█" * wc["wins_B"]
            bar_a = "█" * wc["wins_A"]
            print(f"  {metric:<16}  B wins:{wc['wins_B']:>3} [{bar_b:<15}]  "
                  f"Tie:{wc['ties']:>2}  A wins:{wc['wins_A']:>3} [{bar_a}]")

        print(f"\n  Category Breakdown (NDCG@{r.top_k}):")
        print(f"  {'Category':<12} {'BM25':>10} {'Hybrid':>10}  Δ")
        print(sep)
        for cat, metrics in r.category_breakdown.items():
            ndcg = metrics.get("ndcg_at_k", {})
            a = ndcg.get("A_bm25", 0.0)
            b = ndcg.get("B_hybrid", 0.0)
            print(f"  {cat:<12} {a:>10.4f} {b:>10.4f}  {b - a:+.4f}")

        print(f"\n{'═' * 60}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Simulated Hybrid Engine (for testing without sentence-transformers installed)
# ══════════════════════════════════════════════════════════════════════════════

class _SimulatedHybridEngine:
    """
    Fallback when the real HybridSearchEngine cannot be imported.

    Simulates a slightly better-than-BM25 result by adding controlled random
    perturbation to BM25 scores (mimicking vector-search complementarity).
    Replace with the real engine for accurate evaluation.
    """

    def __init__(self, bm25_engine: BM25OnlyEngine, seed: int = 42) -> None:
        self._bm25 = bm25_engine
        self._rng = random.Random(seed)

    def search(self, query: str, top_k: int = 5) -> list[str]:
        # Get a wider candidate set from BM25
        candidates = self._bm25.search(query, top_k=top_k * 3)
        # Shuffle lower half slightly to simulate vector reranking
        if len(candidates) > 2:
            lower = candidates[2:]
            self._rng.shuffle(lower)
            candidates = candidates[:2] + lower
        return candidates[:top_k]


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _effect_label(d: float) -> str:
    d = abs(d)
    if d < 0.2:
        return "negligible"
    if d < 0.5:
        return "small"
    if d < 0.8:
        return "medium"
    return "large"


def export_json(report: ABReport, path: str) -> None:
    """Serialise the report (without raw_results) to JSON."""
    data = {
        "system_a": report.system_a_name,
        "system_b": report.system_b_name,
        "top_k": report.top_k,
        "n_queries": report.n_queries,
        "n_runs": report.n_runs,
        "aggregates": report.aggregates,
        "stat_tests": report.stat_tests,
        "category_breakdown": report.category_breakdown,
        "win_counts": report.win_counts,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Results exported → %s", path)


# ══════════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ══════════════════════════════════════════════════════════════════════════════

from fiqa_loader import load_fiqa

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-fiqa",  action="store_true", help="Use FiQA dataset instead of built-in corpus")
    parser.add_argument("--fiqa-corpus-size",  type=int, default=5000)
    parser.add_argument("--fiqa-query-count",  type=int, default=100)
    parser.add_argument("--top-k",  type=int, default=5)
    parser.add_argument("--runs",   type=int, default=1)   # 1 run is fine for large datasets
    parser.add_argument("--results", type=str, default="")
    args = parser.parse_args()

    if args.use_fiqa:
        corpus, queries = load_fiqa(
            max_corpus=args.fiqa_corpus_size,
            max_queries=args.fiqa_query_count,
        )
    else:
        corpus, queries = CORPUS, QUERY_SET

    pipeline = ABTestPipeline(corpus=corpus, queries=queries, top_k=args.top_k, n_runs=args.runs)
    report = pipeline.run()

    if args.results:
        export_json(report, args.results)


if __name__ == "__main__":
    main()
