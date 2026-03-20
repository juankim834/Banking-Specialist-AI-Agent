import json
import urllib.request
from hybrid_search_ab_test.models import DocumentChunk, QueryCase


def _get_parquet_urls(dataset: str) -> dict[str, list[str]]:
    """
    Ask the HuggingFace dataset-server for the real parquet URLs.
    Returns {config/split: [url, ...]}
    """
    api = f"https://datasets-server.huggingface.co/parquet?dataset={dataset}"
    req = urllib.request.Request(api, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())

    result: dict[str, list[str]] = {}
    for f in data.get("parquet_files", []):
        key = f"{f['config']}/{f['split']}"
        result.setdefault(key, []).append(f["url"])
    return result


def _load_parquet_rows(urls: list[str]) -> list[dict]:
    """Load one or more parquet files via load_dataset('parquet', ...)."""
    from datasets import load_dataset
    ds = load_dataset("parquet", data_files={"train": urls}, split="train")
    return list(ds)


def load_fiqa(max_corpus: int = 5000, max_queries: int = 100) -> tuple[list, list]:
    """
    Loads FiQA by:
      1. Asking the HF dataset-server API for verified parquet URLs  (no guessing)
      2. Loading them with load_dataset('parquet', ...)              (no loading scripts)

    Requires: pip install datasets pandas pyarrow
    """
    print("Fetching FiQA parquet URLs from HuggingFace API ...")
    fiqa_urls  = _get_parquet_urls("BeIR/fiqa")
    qrels_urls = _get_parquet_urls("BeIR/fiqa-qrels")

    # ── Debug: show what configs/splits are available
    print("  BeIR/fiqa splits found:      ", list(fiqa_urls.keys()))
    print("  BeIR/fiqa-qrels splits found:", list(qrels_urls.keys()))

    # ── Pick the right keys (corpus / queries / qrels)
    corpus_key  = next(k for k in fiqa_urls  if "corpus"  in k)
    queries_key = next(k for k in fiqa_urls  if "queries" in k)
    qrels_key   = next(k for k in qrels_urls if "test"    in k)

    print(f"  Loading corpus  from: {corpus_key}")
    print(f"  Loading queries from: {queries_key}")
    print(f"  Loading qrels   from: {qrels_key}")

    raw_corpus  = _load_parquet_rows(fiqa_urls[corpus_key])
    raw_queries = _load_parquet_rows(fiqa_urls[queries_key])
    raw_qrels   = _load_parquet_rows(qrels_urls[qrels_key])

    # ── Corpus  (_id, title, text)
    corpus = []
    for i, row in enumerate(raw_corpus):
        if i >= max_corpus:
            break
        corpus.append(DocumentChunk(
            chunk_id  = str(row["_id"]),
            document  = "fiqa_corpus",
            page      = 0,
            section   = str(row.get("title", "")),
            content   = str(row["text"]),
            chunk_type= "text",
        ))
    corpus_ids = {c.chunk_id for c in corpus}

    # ── Qrels  (query-id, corpus-id, score)
    qrels: dict[str, dict[str, int]] = {}
    for row in raw_qrels:
        qid   = str(row["query-id"])
        did   = str(row["corpus-id"])
        score = int(row["score"])
        if did not in corpus_ids:
            continue
        qrels.setdefault(qid, {})[did] = score

    # ── Queries  (_id, text)
    queries_map = {str(row["_id"]): str(row["text"]) for row in raw_queries}

    query_cases = []
    for qid, grade_map in qrels.items():
        if not grade_map:
            continue
        query_text = queries_map.get(qid)
        if not query_text:
            continue
        query_cases.append(QueryCase(
            query_id        = qid,
            query           = query_text,
            relevant_ids    = set(grade_map.keys()),
            relevance_grades= grade_map,
            category        = "fiqa",
        ))
        if len(query_cases) >= max_queries:
            break

    print(f"Loaded {len(corpus)} chunks, {len(query_cases)} queries from FiQA")
    return corpus, query_cases