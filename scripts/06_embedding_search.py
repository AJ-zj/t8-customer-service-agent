"""使用硅基流动 Embedding API 检索石头 T8 说明书知识库。"""
import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data/processed/t8_pages.jsonl"
CACHE = ROOT / "tmp/t8_qwen3_embedding_0.6b_cache.json"
API_URL = "https://api.siliconflow.cn/v1/embeddings"
MODEL = "Qwen/Qwen3-Embedding-0.6B"
BATCH_SIZE = 16
TOP_K = 3

TEST_CASES = [
    {"query": "主刷缠了头发，怎么拆下来清理？", "page": 15,
     "section": "日常维护", "contains": ("软胶主刷", "拔出软胶主刷轴承", "清除软胶主刷")},
    {"query": "清扫时有异响怎么办？", "page": 23, "title": "清扫时有异响"},
    {"query": "WiFi 无法连接怎么办？", "page": 23, "title": "无法连接WiFi"},
    {"query": "如何重置 WiFi 并重新添加设备？", "page": 12,
     "section": "使用说明", "contains": ("重置 WiFi", "添加设备")},
]


def load_corpus():
    raw = CORPUS.read_bytes()
    records = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError(f"知识库为空：{CORPUS}")
    return records, hashlib.sha256(raw).hexdigest()


def embedding_text(record):
    return f"标题：{record.get('title', '')}\n章节：{record.get('section', '')}\n内容：{record.get('text', '')}"


def validate_vectors(vectors, expected_count):
    if not isinstance(vectors, list) or len(vectors) != expected_count:
        raise ValueError(f"向量数量不匹配：预期 {expected_count}，实际 {len(vectors) if isinstance(vectors, list) else '无效'}")
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
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        body = json.dumps({"model": MODEL, "input": batch, "encoding_format": "float"}).encode("utf-8")
        request = Request(API_URL, data=body, headers={
            "Authorization": f"Bearer {api_key}", "Content-Type": "application/json"
        }, method="POST")
        try:
            with urlopen(request, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            raise RuntimeError(f"硅基流动 API 返回 HTTP {error.code}；未输出响应内容或密钥。") from None
        except (URLError, TimeoutError, OSError, json.JSONDecodeError):
            raise RuntimeError("硅基流动 API 请求失败或返回了无效 JSON；请检查网络和服务状态。") from None

        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list) or len(data) != len(batch):
            raise RuntimeError(f"API 向量数量不匹配：本批预期 {len(batch)} 条。")
        ordered = [None] * len(batch)
        for item in data:
            index = item.get("index") if isinstance(item, dict) else None
            if not isinstance(index, int) or not 0 <= index < len(batch) or ordered[index] is not None:
                raise RuntimeError("API 返回的向量索引缺失、重复或越界，已停止以防记录错位。")
            ordered[index] = item.get("embedding")
        if any(vector is None for vector in ordered):
            raise RuntimeError("API 响应缺少向量，已停止以防记录错位。")
        vectors.extend(ordered)
    return validate_vectors(vectors, len(texts))


def load_document_vectors(records, corpus_hash, api_key):
    if CACHE.exists():
        try:
            saved = json.loads(CACHE.read_text(encoding="utf-8"))
            if (isinstance(saved, dict) and saved.get("model") == MODEL
                    and saved.get("corpus_sha256") == corpus_hash
                    and saved.get("record_count") == len(records)):
                return validate_vectors(saved.get("vectors"), len(records))
        except (OSError, TypeError, ValueError):
            pass  # 缓存损坏或不匹配时重新生成。

    vectors = request_embeddings([embedding_text(record) for record in records], api_key)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    temporary = CACHE.with_suffix(".tmp")
    cache_data = {"model": MODEL, "corpus_sha256": corpus_hash,
                  "record_count": len(records), "vectors": vectors}
    try:
        temporary.write_text(json.dumps(cache_data), encoding="utf-8")
        temporary.replace(CACHE)
    except OSError:
        raise RuntimeError("文档向量已生成，但本地缓存写入失败。") from None
    return vectors


def cosine_similarity(left, right):
    if len(left) != len(right):
        raise ValueError("查询向量与文档向量维度不一致。")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    return dot / (left_norm * right_norm)


def rank_records(query_vector, records, document_vectors):
    if len(records) != len(document_vectors):
        raise ValueError("知识记录与文档向量数量不一致。")
    ranked = [(i, cosine_similarity(query_vector, vector))
              for i, vector in enumerate(document_vectors)]
    return sorted(ranked, key=lambda item: item[1], reverse=True)


def summarize(text, limit=120):
    content = " ".join(text.split())
    return content if len(content) <= limit else content[:limit] + "..."


def show_results(query, ranked, records):
    print(f"问题：{query}")
    for rank, (index, score) in enumerate(ranked[:TOP_K], start=1):
        record = records[index]
        title = record.get("title") or record.get("section") or "（无标题）"
        metadata = {key: record.get(key) for key in ("model", "pdf_page", "section", "title")}
        print(f"\nTop {rank} | 余弦相似度：{score:.4f} | 第 {record.get('pdf_page')} 页 | {title}")
        print(f"Metadata：{metadata}")
        print(f"内容摘要：{summarize(record.get('text', ''))}")
    print("\n说明：相似度仅用于候选排序，不是置信度或正确性判断；请核对说明书原文。")


def is_test_target(record, case):
    if record.get("pdf_page") != case["page"]:
        return False
    if "title" in case:
        return record.get("title") == case["title"]
    return (record.get("section") == case["section"]
            and all(text in record.get("text", "") for text in case.get("contains", ())))


def run_tests(records, document_vectors, api_key):
    query_vectors = request_embeddings([case["query"] for case in TEST_CASES], api_key)
    if len(query_vectors) != len(TEST_CASES):
        raise RuntimeError("测试问题向量数量不匹配。")
    all_passed = True
    for case, query_vector in zip(TEST_CASES, query_vectors):
        ranked = rank_records(query_vector, records, document_vectors)
        targets = [i for i, record in enumerate(records) if is_test_target(record, case)]
        top3 = ranked[:TOP_K]
        top3_items = [
    f"第{records[i]['pdf_page']}页《{records[i].get('title') or records[i].get('section')}》"
    f" 相似度={score:.4f}"
    for i, score in top3
]
        if len(targets) != 1:
            all_passed = False
            print(f"[未通过] {case['query']} | 目标知识匹配数={len(targets)} | 实际排名无法确定 | Top-3={top3_items}")
            continue
        target_index = targets[0]
        target_rank = next(rank for rank, (index, _) in enumerate(ranked, start=1) if index == target_index)
        passed = target_rank <= TOP_K
        all_passed = all_passed and passed
        status = "通过" if passed else "未通过"
        title = case.get("title") or case.get("section")
        print(f"[{status}] {case['query']} | Top-3={top3_items} | 目标=第{case['page']}页《{title}》"
              f" | 实际排名=第{target_rank}名 | 是否进入Top-3={'是' if passed else '否'}")
    return all_passed


def get_api_key():
    load_dotenv(dotenv_path=ROOT / ".env", override=False)
    key = os.environ.get("SILICONFLOW_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "未配置 SILICONFLOW_API_KEY。请复制项目根目录 .env.example 为 .env，"
            "并在 .env 中填写密钥后重试。不要将 .env 提交到代码仓库。"
        )
    return key


def main():
    parser = argparse.ArgumentParser(description="T8 说明书 Embedding 语义检索")
    parser.add_argument("query", nargs="?", help="要检索的问题")
    parser.add_argument("--test", action="store_true", help="运行 4 道固定测试题")
    args = parser.parse_args()
    if args.test and args.query is not None:
        parser.error("--test 不需要额外查询文本")
    if not args.test and args.query is None:
        parser.error("请提供查询文本，或使用 --test")
    if args.query is not None and not args.query.strip():
        print("错误：查询内容不能为空。", file=sys.stderr)
        return 2

    try:
        api_key = get_api_key()
        records, corpus_hash = load_corpus()
        document_vectors = load_document_vectors(records, corpus_hash, api_key)
        if args.test:
            return 0 if run_tests(records, document_vectors, api_key) else 1
        query_vectors = request_embeddings([args.query], api_key)
        if len(query_vectors) != 1 or len(query_vectors[0]) != len(document_vectors[0]):
            raise RuntimeError("查询向量数量或维度与知识库向量不匹配。")
        show_results(args.query, rank_records(query_vectors[0], records, document_vectors), records)
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
