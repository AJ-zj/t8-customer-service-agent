"""BM25 + Embedding + RRF + Reranker pipeline used by production Grounded QA.

The ranking constants and formulas mirror the validated retrieval baseline.
This module is the application runtime implementation; command-line and
experiment entry points are not imported here.
"""

import hashlib
import json
import math
import re
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import jieba
from rank_bm25 import BM25Okapi

from app.config import PROJECT_ROOT, get_siliconflow_api_key


CORPUS = PROJECT_ROOT / "data" / "processed" / "t8_pages.jsonl"
EMBEDDING_CACHE = PROJECT_ROOT / "tmp" / "t8_qwen3_embedding_0.6b_cache.json"
EMBEDDING_URL = "https://api.siliconflow.cn/v1/embeddings"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
RERANK_URL = "https://api.siliconflow.cn/v1/rerank"
RERANK_MODEL = "Qwen/Qwen3-Reranker-0.6B"
BM25_MIN_SCORE = 1.0
BM25_CANDIDATE_LIMIT = 10
EMBEDDING_CANDIDATE_LIMIT = 10
RRF_K = 60
RERANK_CANDIDATE_LIMIT = 10
RERANK_TOP_K = 3
EMBEDDING_BATCH_SIZE = 16


def load_corpus():
    raw = CORPUS.read_bytes()
    records = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError(f"知识库为空：{CORPUS}")
    return records, hashlib.sha256(raw).hexdigest()


def tokenize(text):
    return [
        word.strip().lower()
        for word in jieba.lcut(text.lower())
        if word.strip() and re.search(r"[\w\u4e00-\u9fff]", word)
    ]


def build_bm25_index(records):
    documents = [
        tokenize(" ".join(str(record.get(key, "")) for key in ("title", "section", "text")))
        for record in records
    ]
    return BM25Okapi(documents)


def rank_bm25(query, records, index):
    tokens = tokenize(query)
    scores = index.get_scores(tokens) if tokens else [0.0] * len(records)
    ordered = sorted(range(len(records)), key=lambda i: scores[i], reverse=True)
    ranks = {record_id: rank for rank, record_id in enumerate(ordered, 1)}
    candidates = [
        (record_id, ranks[record_id])
        for record_id in ordered
        if scores[record_id] >= BM25_MIN_SCORE
    ][:BM25_CANDIDATE_LIMIT]
    return scores, ranks, candidates


def embedding_text(record):
    return f"标题：{record.get('title', '')}\n章节：{record.get('section', '')}\n内容：{record.get('text', '')}"


def validate_vectors(vectors, expected_count):
    if not isinstance(vectors, list) or len(vectors) != expected_count:
        actual = len(vectors) if isinstance(vectors, list) else "无效"
        raise ValueError(f"向量数量不匹配：预期 {expected_count}，实际 {actual}")
    dimension = None
    checked = []
    for vector in vectors:
        if not isinstance(vector, list) or not vector:
            raise ValueError("接口返回了空或无效向量")
        current = [float(value) for value in vector]
        if dimension is None:
            dimension = len(current)
        if len(current) != dimension or not all(math.isfinite(value) for value in current):
            raise ValueError("接口返回的向量维度不一致或包含无效数值")
        checked.append(current)
    return checked


def request_embeddings(texts, api_key):
    vectors = []
    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[start:start + EMBEDDING_BATCH_SIZE]
        body = json.dumps({
            "model": EMBEDDING_MODEL,
            "input": batch,
            "encoding_format": "float",
        }).encode("utf-8")
        request = Request(EMBEDDING_URL, data=body, headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }, method="POST")
        try:
            with urlopen(request, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            raise RuntimeError(
                f"硅基流动 Embedding API 返回 HTTP {error.code}；未输出响应内容或密钥。"
            ) from None
        except (URLError, TimeoutError, OSError, json.JSONDecodeError):
            raise RuntimeError(
                "硅基流动 Embedding 请求失败或返回无效 JSON；请检查网络和服务状态。"
            ) from None

        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list) or len(data) != len(batch):
            raise RuntimeError(f"API 向量数量不匹配：本批预期 {len(batch)} 条。")
        ordered = [None] * len(batch)
        for item in data:
            index = item.get("index") if isinstance(item, dict) else None
            if type(index) is not int or not 0 <= index < len(batch) or ordered[index] is not None:
                raise RuntimeError("API 返回的向量索引缺失、重复或越界，已停止以防记录错位。")
            ordered[index] = item.get("embedding")
        if any(vector is None for vector in ordered):
            raise RuntimeError("API 响应缺少向量，已停止以防记录错位。")
        vectors.extend(ordered)
    return validate_vectors(vectors, len(texts))


def load_cached_document_vectors(records, corpus_hash):
    if not EMBEDDING_CACHE.is_file():
        raise RuntimeError(f"需要已有 Embedding 缓存：{EMBEDDING_CACHE}")
    try:
        saved = json.loads(EMBEDDING_CACHE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise RuntimeError("Embedding 缓存不可读取；请先按项目流程修复缓存。") from None
    if (
        not isinstance(saved, dict)
        or saved.get("model") != EMBEDDING_MODEL
        or saved.get("corpus_sha256") != corpus_hash
        or saved.get("record_count") != len(records)
    ):
        raise RuntimeError("Embedding 缓存与模型或知识库指纹不匹配；停止使用。")
    return validate_vectors(saved.get("vectors"), len(records))


def cosine_similarity(left, right):
    if len(left) != len(right):
        raise ValueError("查询向量与文档向量维度不一致。")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def rank_embedding(query_vector, records, document_vectors):
    if len(records) != len(document_vectors):
        raise ValueError("知识记录与文档向量数量不一致。")
    ordered = [
        (index, cosine_similarity(query_vector, vector))
        for index, vector in enumerate(document_vectors)
    ]
    ordered.sort(key=lambda item: item[1], reverse=True)
    ranks = {record_id: rank for rank, (record_id, _) in enumerate(ordered, 1)}
    candidates = [
        (record_id, ranks[record_id])
        for record_id, _ in ordered[:EMBEDDING_CANDIDATE_LIMIT]
    ]
    return ranks, candidates


def reciprocal_rank_fusion(bm25_candidates, embedding_candidates):
    rank_lists = [
        {record_id: rank for record_id, rank in candidates}
        for candidates in (bm25_candidates, embedding_candidates)
    ]
    candidate_ids = set().union(*(set(ranks) for ranks in rank_lists))
    fused = []
    for record_id in candidate_ids:
        score = sum(
            1 / (RRF_K + ranks[record_id])
            for ranks in rank_lists
            if record_id in ranks
        )
        fused.append((record_id, score, rank_lists[0].get(record_id), rank_lists[1].get(record_id)))
    return sorted(fused, key=lambda item: (-item[1], item[0]))


def rerank_document_text(record):
    return (
        f"页码：{record.get('pdf_page', '')}\n标题：{record.get('title', '')}\n"
        f"章节：{record.get('section', '')}\n内容：{record.get('text', '')}"
    )


def request_rerank(query, candidates, api_key):
    documents = [rerank_document_text(record) for _, record in candidates]
    body = json.dumps({
        "model": RERANK_MODEL,
        "query": query,
        "documents": documents,
        "top_n": len(documents),
        "return_documents": False,
    }).encode("utf-8")
    request = Request(RERANK_URL, data=body, headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }, method="POST")
    try:
        with urlopen(request, timeout=90) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        raise RuntimeError(
            f"Reranker API 返回 HTTP {error.code}；未输出响应内容或密钥。"
        ) from None
    except (URLError, TimeoutError, OSError, json.JSONDecodeError):
        raise RuntimeError("Reranker API 请求失败或返回无效 JSON；请检查网络和服务状态。") from None

    rows = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or len(rows) != len(candidates):
        raise RuntimeError(f"Reranker 返回结果数量不匹配：预期 {len(candidates)} 条。")
    ordered, seen = [], set()
    for row in rows:
        index = row.get("index") if isinstance(row, dict) else None
        if type(index) is not int or not 0 <= index < len(candidates) or index in seen:
            raise RuntimeError("Reranker 返回索引缺失、重复或越界，已停止防止知识错位。")
        score = float(row.get("relevance_score"))
        if not math.isfinite(score):
            raise RuntimeError("Reranker 返回了无效相关性分数。")
        seen.add(index)
        ordered.append((candidates[index][0], score))
    return sorted(ordered, key=lambda item: (-item[1], item[0]))


def create_pipeline():
    records, corpus_hash = load_corpus()
    # Both retrieval channels bind to the same parsed records and preserve index order.
    bm25_index = build_bm25_index(records)
    from app.retrieval.t8_vector_store import T8VectorStore
    vector_store = T8VectorStore(records, corpus_hash)
    api_key = get_siliconflow_api_key()
    # Keep the established six-item tuple consumed by the QA service and CLI.
    return None, None, records, bm25_index, vector_store, api_key


def retrieve(query, pipeline):
    _, _, records, bm25_index, vector_store, api_key = pipeline
    _, _, bm25_candidates = rank_bm25(query, records, bm25_index)
    dense_results = vector_store.search_dense(query, limit=EMBEDDING_CANDIDATE_LIMIT)
    embedding_candidates = [(record["record_index"], rank)
                            for rank, record in enumerate(dense_results, 1)]
    fused = reciprocal_rank_fusion(bm25_candidates, embedding_candidates)
    candidate_ids = [
        record_id for record_id, _, _, _ in fused[:RERANK_CANDIDATE_LIMIT]
    ]
    candidates = [(index, records[index]) for index in candidate_ids]
    if not candidates:
        raise RuntimeError("BM25 与 Embedding 均未提供候选知识。")
    reranked = request_rerank(query, candidates, api_key)
    return [
        (f"E{rank}", record_index, records[record_index])
        for rank, (record_index, _) in enumerate(reranked[:RERANK_TOP_K], 1)
    ]
