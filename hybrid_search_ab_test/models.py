from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class DocumentChunk:
    chunk_id:   str
    document:   str
    page:       int
    section:    str
    content:    str
    chunk_type: str = "text"

@dataclass
class QueryCase:
    query_id:         str
    query:            str
    relevant_ids:     set[str]
    relevance_grades: dict[str, int]
    category:         str = "general"