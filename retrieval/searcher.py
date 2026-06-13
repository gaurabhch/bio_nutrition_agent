from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import psycopg2
import psycopg2.extras
from fastembed import TextEmbedding
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from config import (
    EMBEDDING_MODEL_FAST,
    NEON_TABLE_NAME,
    TOP_K_RETRIEVAL,
    TOP_K_CANDIDATES,
    AUTO_MERGE_RATIO,
    FINAL_CONTEXT_TOKENS,
    CATEGORY_TO_DOMAIN,
    DOMAIN_TO_DB_CATEGORIES,
)

W_VECTOR = 0.55
W_LEXICAL = 0.25
W_SECTION = 0.10
W_DOMAIN = 0.10

LEX_FLOOR = 0.05
LEX_PENALTY = 0.60

_STOP = frozenset({
    "the", "and", "for", "with", "this", "that", "are", "was", "has", "have",
    "can", "may", "your", "you", "its", "our", "from", "been", "will", "not",
    "also", "but", "such", "they", "their", "women", "woman",
    "index", "while", "which", "were", "had", "than",
})


def _tokenize(text_: str) -> set[str]:
    tokens = re.findall(r"[a-zA-Z0-9]+", (text_ or "").lower())
    return {t for t in tokens if len(t) >= 3 and t not in _STOP}


def _overlap(query_tokens: set[str], target_text: str) -> float:
    if not query_tokens:
        return 0.0
    return len(query_tokens & _tokenize(target_text or "")) / max(len(query_tokens), 1)


def _normalized_row_domain(row: dict) -> str:
    raw_category = row.get("category", "") or ""
    return CATEGORY_TO_DOMAIN.get(raw_category, "nutrition_general")


def _score_row(row: dict, query_tokens: set[str], routing_domain: str) -> tuple[float, dict]:
    vec = float(row.get("similarity", 0.0))
    lex = _overlap(query_tokens, row.get("chunk_text") or row.get("text", ""))
    sec = _overlap(query_tokens, row.get("field_type") or row.get("section_name", "") or row.get("topic", ""))
    row_domain = _normalized_row_domain(row)
    dom = 1.0 if routing_domain and row_domain == routing_domain else 0.0

    final = W_VECTOR * vec + W_LEXICAL * lex + W_SECTION * sec + W_DOMAIN * dom
    breakdown = {
        "vector_sim": round(vec, 4),
        "lexical_overlap": round(lex, 4),
        "section_match": round(sec, 4),
        "domain_prior": round(dom, 4),
        "lex_penalised": False,
        "final_score": round(final, 4),
    }
    return final, breakdown


def _apply_lex_floor(row: dict, query_tokens: set[str]) -> None:
    if not query_tokens:
        return
    lex = row.get("score_breakdown", {}).get("lexical_overlap", 0.0)
    if lex < LEX_FLOOR:
        penalised = row["final_score"] * LEX_PENALTY
        row["final_score"] = penalised
        row["score_breakdown"]["final_score"] = round(penalised, 4)
        row["score_breakdown"]["lex_penalised"] = True


def _rerank_and_dedup(rows: list[dict], query_tokens: set[str], routing_domain: str, top_k: int) -> list[dict]:
    for row in rows:
        final, breakdown = _score_row(row, query_tokens, routing_domain)
        row["final_score"] = final
        row["score_breakdown"] = breakdown

    for row in rows:
        _apply_lex_floor(row, query_tokens)

    best_per_cluster: dict[str, dict] = {}
    for row in rows:
        cid = row.get("cluster_id", "__unknown__")
        if cid not in best_per_cluster or row["final_score"] > best_per_cluster[cid]["final_score"]:
            best_per_cluster[cid] = row

    return sorted(best_per_cluster.values(), key=lambda r: r["final_score"], reverse=True)[:top_k]


@dataclass
class RetrievedContext:
    level: str
    chunk_id: str
    text: str
    cluster_id: str
    l1_id: str
    l1_text: str
    l2_id: str
    l2_text: str
    section_name: str
    category: str
    score: float
    token_count: int
    reference_sources: list[str] = field(default_factory=list)
    score_breakdown: dict = field(default_factory=dict)


class HierarchicalRetriever:
    def __init__(self) -> None:
        self._embedder = TextEmbedding(model_name=EMBEDDING_MODEL_FAST)
        self._conn_str = os.environ["DATABASE_URL_SYNC"]

    def retrieve_chunks(self, query: str, domain: Optional[str] = None, top_k: int = TOP_K_RETRIEVAL) -> list[RetrievedContext]:
        vector = self._embed(query)
        query_tokens = _tokenize(query)
        rows = self._search(vector, domain, top_k=TOP_K_CANDIDATES)
        reranked = _rerank_and_dedup(rows, query_tokens, domain or "", top_k)
        return self._auto_merge(reranked)

    def build_context_string(self, contexts: list[RetrievedContext], max_tokens: int = FINAL_CONTEXT_TOKENS) -> str:
        parts: list[str] = []
        total = 0
        for ctx in contexts:
            block = (
                f"[{ctx.level}] Cluster: {ctx.cluster_id} | "
                f"Section: {ctx.section_name} | "
                f"Domain: {ctx.category} | "
                f"Score: {ctx.score:.4f}\n"
                f"{ctx.text}"
            )
            words = len(block.split())
            if total + words > max_tokens:
                break
            parts.append(block)
            total += words
        return "\n\n---\n\n".join(parts)

    def _embed(self, query: str) -> list[float]:
        return list(self._embedder.embed([query]))[0].tolist()

    def _search(self, vector: list[float], domain: Optional[str], top_k: int) -> list[dict]:
        vec_str = "[" + ",".join(str(v) for v in vector) + "]"

        categories = DOMAIN_TO_DB_CATEGORIES.get(domain or "", [])
        if categories:
            where_clause = "WHERE category = ANY(%(categories)s)"
        else:
            where_clause = ""

        params: dict = {"vec": vec_str, "top_k": top_k}
        if categories:
            params["categories"] = categories

        sql = f"""
        SELECT
            COALESCE(CAST(id AS text), cluster_id) AS chunk_id,
            chunk_text AS text,
            chunk_text,
            cluster_id,
            category,
            topic AS section_name,
            COALESCE(array_length(string_to_array(chunk_text, ' '), 1), 0) AS token_count,
            cluster_id AS l2_id,
            chunk_text AS l2_text,
            cluster_id AS l1_id,
            chunk_text AS l1_text,
            reference_sources,
            1 - (embedding <=> %(vec)s::vector) AS similarity
        FROM {NEON_TABLE_NAME}
        {where_clause}
        ORDER BY embedding <=> %(vec)s::vector
        LIMIT %(top_k)s
        """

        conn = psycopg2.connect(self._conn_str)
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, params)
                rows = [dict(r) for r in cur.fetchall()]
        finally:
            conn.close()

        for r in rows:
            r.setdefault("chunk_text", r.get("text", ""))
            r.setdefault("field_type", r.get("section_name", r.get("topic", "")))
            r.setdefault("topic", r.get("category", ""))
        return rows

    def _auto_merge(self, rows: list[dict]) -> list[RetrievedContext]:
        l2_groups: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            l2_groups[row.get("l2_id", row.get("cluster_id", ""))].append(row)

        results: list[RetrievedContext] = []
        for _, group in l2_groups.items():
            best = max(group, key=lambda r: r.get("final_score", r.get("similarity", 0.0)))
            level = "L2" if len(group) >= AUTO_MERGE_RATIO else "L3"
            body = (best.get("l2_text") or best.get("text", "")) if level == "L2" else best.get("text", "")

            refs_raw = best.get("reference_sources") or []
            if isinstance(refs_raw, str):
                try:
                    refs = json.loads(refs_raw)
                    if not isinstance(refs, list):
                        refs = []
                except (json.JSONDecodeError, TypeError):
                    refs = []
            elif isinstance(refs_raw, list):
                refs = refs_raw
            else:
                refs = []

            results.append(
                RetrievedContext(
                    level=level,
                    chunk_id=best.get("chunk_id", ""),
                    text=body,
                    cluster_id=best.get("cluster_id", ""),
                    l1_id=best.get("l1_id", best.get("cluster_id", "")),
                    l1_text=best.get("l1_text", ""),
                    l2_id=best.get("l2_id", best.get("cluster_id", "")),
                    l2_text=best.get("l2_text", ""),
                    section_name=best.get("section_name", ""),
                    category=best.get("category", ""),
                    score=float(best.get("final_score", best.get("similarity", 0.0))),
                    token_count=int(best.get("token_count", 0)),
                    reference_sources=refs,
                    score_breakdown=best.get("score_breakdown", {}),
                )
            )

        results.sort(key=lambda c: c.score, reverse=True)
        return results


async def retrieve_chunks(
    query_embedding: list[float],
    domain: str,
    session: AsyncSession,
    top_k: int = TOP_K_RETRIEVAL,
    *,
    rewritten_query: str = "",
) -> list[dict]:
    vec_literal = "[" + ",".join(str(v) for v in query_embedding) + "]"

    categories = DOMAIN_TO_DB_CATEGORIES.get(domain or "", [])
    if categories:
        where_sql = "WHERE category = ANY(CAST(:categories AS text[]))"
    else:
        where_sql = ""

    sql = text(
        f"SELECT "
        f" chunk_text AS chunk_text, "
        f" chunk_text AS text, "
        f" topic AS field_type, "
        f" topic AS section_name, "
        f" reference_sources AS reference_sources, "
        f" cluster_id, "
        f" cluster_id AS l1_id, "
        f" chunk_text AS l1_text, "
        f" cluster_id AS l2_id, "
        f" chunk_text AS l2_text, "
        f" category AS topic, "
        f" category AS category, "
        f" COALESCE(CAST(id AS text), cluster_id) AS chunk_id, "
        f" COALESCE(array_length(string_to_array(chunk_text, ' '), 1), 0) AS token_count, "
        f" 1 - (embedding <=> CAST(:query_emb AS vector)) AS similarity "
        f"FROM {NEON_TABLE_NAME} "
        f"{where_sql} "
        f"ORDER BY embedding <=> CAST(:query_emb AS vector) "
        f"LIMIT :top_k"
    )

    params = {
        "query_emb": vec_literal,
        "top_k": TOP_K_CANDIDATES,
    }
    if categories:
        params["categories"] = categories

    result = await session.execute(sql, params)
    rows = [dict(row._mapping) for row in result.fetchall()]

    if not rows:
        return []

    query_tokens = _tokenize(rewritten_query) if rewritten_query else set()
    rows = _rerank_and_dedup(rows, query_tokens, domain, top_k)

    for row in rows:
        row["similarity"] = row.get("final_score", row.get("similarity", 0.0))

    return rows