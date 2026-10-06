"""对 BM25/Embedding 联合候选执行 SiliconFlow Reranker 重排序实验。"""
import importlib.util
import json
import math
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
RERANK_URL = "https://api.siliconflow.cn/v1/rerank"
RERANK_MODEL = "Qwen/Qwen3-Reranker-0.6B"
TOP_K = 3
CANDIDATE_LIMIT = 10


def load_script(name, filename):
    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载检索模块：{filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_cached_vectors(embedding, count, corpus_hash):
    if not embedding.CACHE.is_file():
        raise RuntimeError(f"需要已有 Embedding 缓存：{embedding.CACHE}")
    saved = json.loads(embedding.CACHE.read_text(encoding="utf-8"))
    if (not isinstance(saved, dict) or saved.get("model") != embedding.MODEL
            or saved.get("corpus_sha256") != corpus_hash or saved.get("record_count") != count):
        raise RuntimeError("Embedding 缓存与模型或知识库指纹不匹配；停止实验，不重建缓存。")
    return embedding.validate_vectors(saved.get("vectors"), count)


def document_text(record):
    return (f"页码：{record.get('pdf_page', '')}\n标题：{record.get('title', '')}\n"
            f"章节：{record.get('section', '')}\n内容：{record.get('text', '')}")


def request_rerank(query, candidates, api_key):
    documents = [document_text(record) for _, record in candidates]
    body = json.dumps({"model": RERANK_MODEL, "query": query,
                       "documents": documents, "top_n": len(documents),
                       "return_documents": False}).encode("utf-8")
    request = Request(RERANK_URL, data=body, headers={
        "Authorization": f"Bearer {api_key}", "Content-Type": "application/json"
    }, method="POST")
    try:
        with urlopen(request, timeout=90) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        raise RuntimeError(f"Reranker API 返回 HTTP {error.code}；未输出响应内容或密钥。") from None
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
    return sorted(
    ordered,
    key=lambda item: (-item[1], item[0])
)


def record_label(record):
    title = record.get("title") or record.get("section") or "（无标题）"
    return f"第{record.get('pdf_page')}页《{title}》"


def evaluate_case(case, records, hybrid, bm25_index, document_vectors, query_vector, api_key):
    bm_scores, bm_ranks, bm_candidates = hybrid.rank_bm25(
        case["question"], records, bm25_index
    )
    em_ranks, em_candidates = hybrid.rank_embedding(query_vector, records, document_vectors)
    rrf_candidates = hybrid.reciprocal_rank_fusion(
        bm_candidates, em_candidates, k=hybrid.RRF_K
    )
    bm_ids = [index for index, _ in bm_candidates[:CANDIDATE_LIMIT]]
    em_ids = [index for index, _ in em_candidates[:CANDIDATE_LIMIT]]
    candidate_ids = list(dict.fromkeys(bm_ids + em_ids))
    candidates = [(index, records[index]) for index in candidate_ids]
    reranked = request_rerank(case["question"], candidates, api_key)
    top_ids = {
        "BM25": bm_ids[:TOP_K],
        "Embedding": em_ids[:TOP_K],
        "RRF": [index for index, _, _, _ in rrf_candidates[:TOP_K]],
        "Reranker": [index for index, _ in reranked[:TOP_K]],
    }
    targets = [item["record_index_zero_based"] for item in case["core_knowledge"]]
    passed, hits = {}, {}
    for method, ids in top_ids.items():
        hits[method] = [index in ids for index in targets]
        passed[method] = (all(hits[method]) if case["core_match_policy"] == "all_of"
                          else any(hits[method]))
    return {
        "case": case, "targets": targets, "bm_scores": bm_scores,
        "bm_ranks": bm_ranks, "bm_candidates": bm_candidates,
        "em_ranks": em_ranks, "em_candidates": em_candidates,
        "rrf_candidates": rrf_candidates, "candidate_ids": candidate_ids,
        "reranked": reranked, "top_ids": top_ids, "hits": hits, "passed": passed,
    }


def main():
    try:
        evaluator = load_script("t8_rerank_evaluation", "08_evaluate_retrieval.py")
        candidate, records, answerable, _ = evaluator.verify_inputs()
        hybrid = load_script("t8_rerank_hybrid", "07_hybrid_search.py")
        bm_records, bm_index = hybrid.bm25_module.build_index()
        em_records, corpus_hash = hybrid.embedding_module.load_corpus()
        if records != bm_records or records != em_records:
            raise RuntimeError("检索模块读取的知识记录或顺序不一致，停止实验。")
        if corpus_hash != candidate["source"]["corpus_sha256"]:
            raise RuntimeError("JSONL 指纹在校验后发生变化，停止实验。")
        if len(answerable) != 13:
            raise RuntimeError(f"有答案候选题应为13道，当前为{len(answerable)}道。")

        embedding = hybrid.embedding_module
        vectors = load_cached_vectors(embedding, len(records), corpus_hash)
        api_key = embedding.get_api_key()
        query_vectors = embedding.request_embeddings(
            [case["question"] for case in answerable], api_key
        )
        if len(query_vectors) != len(answerable):
            raise RuntimeError("查询向量数量不匹配，停止评测。")
        results = []
        for case, query_vector in zip(answerable, query_vectors):
            results.append(evaluate_case(
                case, records, hybrid, bm_index, vectors, query_vector, api_key
            ))
        actual_max = max(len(result["candidate_ids"]) for result in results)
        print(f"模型：{RERANK_MODEL} | 每题候选：BM25 Top-10 ∪ Embedding Top-10，"
              f"理论上最多20条，本批最大{actual_max}条。")
        print("文档 Embedding 使用本地缓存；BM25、Embedding、RRF 参数沿用现有实现。")
        return report(results, records)
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
        print(f"实验失败：{error}", file=sys.stderr)
        return 1


def report(results, records):
    methods = ("BM25", "Embedding", "RRF", "Reranker")
    print("\n13道有答案题逐题结果（✓通过，×未通过；多核心知识按 all_of/any_of 判定）：")
    print("题目 | BM25 | Embedding | RRF | Reranker")
    for result in results:
        marks = ["✓" if result["passed"][method] else "×" for method in methods]
        print(f"{result['case']['case_id']} | " + " | ".join(marks))

    print("\n四种方法汇总（当前13题）：")
    for method in methods:
        failures = [result["case"]["case_id"] for result in results
                    if not result["passed"][method]]
        print(f"  {method}: {len(results) - len(failures)}/{len(results)}；失败："
              f"{', '.join(failures) if failures else '无'}")

    print("\n重点案例：RRF 为重排前基线；Reranker 为联合候选重排后结果。")
    for result in results:
        case = result["case"]
        if case["case_id"] not in {"T8-C05", "T8-C10", "T8-C13"}:
            continue
        rrf_ranks = {index: rank for rank, (index, _, _, _) in
                     enumerate(result["rrf_candidates"], start=1)}
        rerank_ranks = {index: rank for rank, (index, _) in
                        enumerate(result["reranked"], start=1)}
        rerank_scores = dict(result["reranked"])
        print(f"\n{case['case_id']} | {case['question']}")
        for item, index in zip(case["core_knowledge"], result["targets"]):
            bm_rank = result["bm_ranks"].get(index)
            em_rank = result["em_ranks"].get(index)
            before = rrf_ranks.get(index)
            after = rerank_ranks.get(index)
            print(f"  核心知识 {item.get('pdf_location') or record_label(records[index])}："
                  f"BM25第{bm_rank if bm_rank is not None else '未排名'}名，"
                  f"Embedding第{em_rank if em_rank is not None else '未排名'}名，"
                  f"RRF第{before if before is not None else '未进候选'}名 → "
                  f"Reranker后第{after if after is not None else '未返回'}名；"
                  f"相关性分数={rerank_scores.get(index, float('nan')):.6f}")
        before_top = result["top_ids"]["RRF"]
        after_top = result["top_ids"]["Reranker"]
        print("  重排前 RRF Top-3：" + "；".join(record_label(records[i]) for i in before_top))
        print("  重排后 Top-3：" + "；".join(
            f"{record_label(records[i])}({rerank_scores[i]:.6f})" for i in after_top
        ))
        print(f"  题目判定：RRF={'通过' if result['passed']['RRF'] else '未通过'}；"
              f"Reranker={'通过' if result['passed']['Reranker'] else '未通过'}")
    print("\n说明：仅为13道候选题的阶段性诊断，不表示模型泛化能力或整体准确率；不改变默认配置。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
