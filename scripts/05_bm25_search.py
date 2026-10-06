"""基于中文分词与 Okapi BM25 的独立检索基线。"""
import argparse
import json
import re
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data/processed/t8_pages.jsonl"
TOP_K = 3
MIN_SCORE = 1.0  # 小语料上的保守过滤阈值，不是相关概率。
TEST_CASES = [
    {
        "query": "主刷缠了头发，怎么拆下来清理？",
        "page": 15,
        "section": "日常维护",
        "contains": ("软胶主刷", "拔出软胶主刷轴承", "清除软胶主刷"),
    },
    {"query": "清扫时有异响怎么办？", "page": 23, "title": "清扫时有异响"},
    {"query": "WiFi 无法连接怎么办？", "page": 23, "title": "无法连接WiFi"},
    {
        "query": "如何重置 WiFi 并重新添加设备？",
        "page": 12,
        "section": "使用说明",
        "contains": ("重置 WiFi", "添加设备"),
    },
]


def tokenize(text):
    """使用 jieba 切词，并去掉空白和纯标点词。"""
    return [
        word.strip().lower()
        for word in jieba.lcut(text.lower())
        if word.strip() and re.search(r"[\w\u4e00-\u9fff]", word)
    ]


def build_index():
    """读取现有 JSONL，返回原始记录和 BM25 索引。"""
    records = [
        json.loads(line)
        for line in CORPUS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    documents = [
        tokenize(" ".join(str(record.get(key, "")) for key in ("title", "section", "text")))
        for record in records
    ]
    return records, BM25Okapi(documents)


def search(query, records, bm25):
    """返回达到最低分数的 Top-3 (BM25 分数, 原始知识记录)。"""
    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)
    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
    return [
        (float(score), records[index])
        for index, score in ranked
        if score >= MIN_SCORE
    ][:TOP_K]


def summarize(text, limit=120):
    content = " ".join(text.split())
    return content if len(content) <= limit else content[:limit] + "..."


def show_results(query, results):
    print(f"问题：{query}")
    if not results:
        print(f"未找到可靠结果：最高 BM25 分数低于阈值 {MIN_SCORE:.1f}。")
        print("可尝试补充说明书中的关键词；BM25 分数不代表答案正确性。")
        return

    for rank, (score, record) in enumerate(results, start=1):
        metadata = {key: record.get(key) for key in ("model", "pdf_page", "section", "title")}
        title = record.get("title") or record.get("section") or "（无标题）"
        print(f"\nTop {rank} | BM25 分数：{score:.4f}")
        print(f"标题：{title} | 页码：{record.get('pdf_page')}")
        print(f"Metadata：{metadata}")
        print(f"内容摘要：{summarize(record.get('text', ''))}")
    print("\n说明：BM25 分数是词项相关度排序值，不是置信度；请核对说明书原文。")


def is_test_target(record, case):
    if record.get("pdf_page") != case["page"]:
        return False
    if "title" in case:
        return record.get("title") == case["title"]
    return record.get("section") == case["section"] and all(
        phrase in record.get("text", "") for phrase in case.get("contains", ())
    )


def run_tests(records, bm25):
    all_passed = True
    for case in TEST_CASES:
        query = case["query"]
        target_indices = [i for i, record in enumerate(records) if is_test_target(record, case)]
        results = search(query, records, bm25)
        top3 = [record for _, record in results]
        if len(target_indices) != 1:
            all_passed = False
            print(f"[未通过] {query} | 目标知识在 JSONL 中匹配到 {len(target_indices)} 条，无法确定排名")
            continue

        target_index = target_indices[0]
        scores = bm25.get_scores(tokenize(query))
        ranked_indices = sorted(range(len(records)), key=lambda i: scores[i], reverse=True)
        actual_rank = ranked_indices.index(target_index) + 1
        target_score = float(scores[target_index])
        passed = any(is_test_target(record, case) for record in top3)
        all_passed = all_passed and passed
        status = "通过" if passed else "未通过"
        expected_name = case.get("title") or case.get("section")
        top3_names = [f"第{r['pdf_page']}页《{r.get('title') or r.get('section')}》" for r in top3]
        print(
            f"[{status}] {query} | 目标：第{case['page']}页《{expected_name}》"
            f" | 实际排名：第{actual_rank}名（BM25={target_score:.4f}）"
            f" | Top-3：{top3_names}"
        )
    return all_passed


def main():
    parser = argparse.ArgumentParser(description="查询 T8 说明书 BM25 基线")
    parser.add_argument("query", nargs="?", help="要检索的问题")
    parser.add_argument("--test", action="store_true", help="运行固定页码测试")
    args = parser.parse_args()
    if not args.test and not args.query:
        parser.error("请提供查询文本，或使用 --test")

    records, bm25 = build_index()
    if args.test:
        return 0 if run_tests(records, bm25) else 1
    show_results(args.query, search(args.query, records, bm25))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
