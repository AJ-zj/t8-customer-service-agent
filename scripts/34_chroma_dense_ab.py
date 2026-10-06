"""Read-only old cosine vs Chroma A/B with fixed queries, vectors and downstream code."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.retrieval import t8_retrieval as retrieval
from app.services.qa_service import normalize_retrieval_query

RESULT_PATH = ROOT / "data/evaluation/t8_chroma_dense_ab_v1_results.json"
BAD_QUESTION = "请问石头 T8 的主刷怎么清理？"


def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate_query(query, query_vector, records, bm25_index, vectors, store, api_key, rerank_cache):
    _, bm25_ranks, bm25 = retrieval.rank_bm25(query, records, bm25_index)
    old_ranks, old_candidates = retrieval.rank_embedding(query_vector, records, vectors)
    new_rows = store.search_by_vector(query_vector, limit=10)
    new_full = store.search_by_vector(query_vector, limit=len(records))
    new_candidates = [(row["record_index"], rank) for rank, row in enumerate(new_rows, 1)]
    new_ranks = {row["record_index"]: rank for rank, row in enumerate(new_full, 1)}
    output = {"query": query, "bm25_ranks": bm25_ranks,
              "dense_top10_overlap": len(set(i for i, _ in old_candidates)
                                         & set(i for i, _ in new_candidates)) / 10,
              "dense_rank_differences": [
                  {"record_index": i, "old_rank": old_ranks[i], "chroma_rank": new_ranks[i]}
                  for i in range(len(records)) if old_ranks[i] != new_ranks[i]],
              "chroma_distances": {row["record_index"]: row["distance"] for row in new_full}}
    for name, ranks, candidates in (("old", old_ranks, old_candidates),
                                     ("chroma", new_ranks, new_candidates)):
        fused = retrieval.reciprocal_rank_fusion(bm25, candidates)
        ids = [i for i, *_ in fused[:retrieval.RERANK_CANDIDATE_LIMIT]]
        key = (query, tuple(ids))
        # Identical request inputs reuse one real response, eliminating API drift/cost.
        if key not in rerank_cache:
            rerank_cache[key] = retrieval.request_rerank(query, [(i, records[i]) for i in ids], api_key)
        output[name] = {"dense_ranks": ranks, "dense_candidates": candidates,
                        "fused": fused, "reranker_candidates": ids, "reranked": rerank_cache[key]}
    return output


def summary(cases, results, exp17):
    output = {}
    for method in ("old", "chroma"):
        mapped = {case["case_id"]: results[normalize_retrieval_query(case["question"])][method]
                  for case in cases}
        hit3, recall3, mrr = exp17.metrics(cases, mapped)
        hit1 = sum(exp17.is_hit(case, exp17.top_ids(mapped[case["case_id"]], 1)) for case in cases)
        output[method] = {"count": len(cases), "hit_at_1": hit1, "hit_at_3": hit3,
                          "recall_at_3": recall3, "mrr": mrr}
    changes, regressions, improvements = [], [], []
    overlaps = []
    for case in cases:
        result = results[normalize_retrieval_query(case["question"])]
        a, b = exp17.top_ids(result["old"], 3), exp17.top_ids(result["chroma"], 3)
        old_hit, new_hit = exp17.is_hit(case, a), exp17.is_hit(case, b)
        row = {"case_id": case["case_id"], "question": case["question"],
               "old_top3": a, "chroma_top3": b,
               "old_hit": old_hit, "chroma_hit": new_hit}
        if a != b:
            changes.append(row)
        if old_hit and not new_hit:
            regressions.append(row)
        if not old_hit and new_hit:
            improvements.append(row)
        overlaps.append(result["dense_top10_overlap"])
    output.update({"final_top3_regressions": regressions, "final_top3_improvements": improvements,
                   "top3_changed_cases": changes, "mean_dense_top10_overlap": sum(overlaps) / len(overlaps)})
    return output


def bad_case_detail(result):
    detail = {"question": BAD_QUESTION, "normalized_query": result["query"],
              "record_index": 14, "pdf_page": 15}
    for method in ("old", "chroma"):
        data = result[method]
        detail[method] = {
            "bm25_rank": result["bm25_ranks"].get(14),
            "dense_rank": data["dense_ranks"].get(14),
            "rrf_rank": next((rank for rank, row in enumerate(data["fused"], 1) if row[0] == 14), None),
            "reranker_rank": next((rank for rank, row in enumerate(data["reranked"], 1) if row[0] == 14), None),
            "top3": [i for i, _ in data["reranked"][:3]],
        }
    return detail


def main():
    if RESULT_PATH.exists():
        print("结果文件已存在，停止以避免无意重复真实 API 评测。")
        return 1
    results, rerank_cache = {}, {}
    try:
        evaluator = load_script("chroma_eval_inputs", "08_evaluate_retrieval.py")
        exp17 = load_script("chroma_eval_metrics", "17_retrieval_improvement_ablation.py")
        exp18 = load_script("chroma_eval_regression", "18_query_normalization_regression.py")
        candidate, records, answerable, unanswerable = evaluator.verify_inputs()
        expressions = exp18.validate_regression_set(candidate, records, answerable)
        _, _, live_records, bm25_index, store, api_key = retrieval.create_pipeline()
        if live_records != records:
            raise RuntimeError("corpus mismatch")
        vectors = retrieval.load_cached_document_vectors(records, candidate["source"]["corpus_sha256"])
        queries = list(dict.fromkeys(normalize_retrieval_query(c["question"])
                                    for c in answerable + unanswerable + expressions))
        if normalize_retrieval_query(BAD_QUESTION) not in queries:
            queries.append(normalize_retrieval_query(BAD_QUESTION))
        print(f"已真实重载持久化 collection：{store.validate_collection()}\n独立查询数：{len(queries)}", flush=True)
        query_vectors = retrieval.request_embeddings(queries, api_key)
        for number, (query, vector) in enumerate(zip(queries, query_vectors), 1):
            results[query] = evaluate_query(query, vector, records, bm25_index, vectors, store, api_key, rerank_cache)
            print(f"A/B {number}/{len(queries)} 完成；Dense Top10 overlap={results[query]['dense_top10_overlap']:.2f}", flush=True)
        candidate_summary = summary(answerable, results, exp17)
        expression_summary = summary(expressions, results, exp17)
        bad_case = bad_case_detail(results[normalize_retrieval_query(BAD_QUESTION)])
        passed = (not candidate_summary["final_top3_regressions"]
                  and not expression_summary["final_top3_regressions"]
                  and candidate_summary["chroma"]["hit_at_3"] >= candidate_summary["old"]["hit_at_3"]
                  and expression_summary["chroma"]["hit_at_3"] >= expression_summary["old"]["hit_at_3"]
                  and 14 in bad_case["chroma"]["top3"])
        output = {
            "source_corpus_sha256": candidate["source"]["corpus_sha256"],
            "dataset_sha256": {name: hashlib.sha256((ROOT / "data/evaluation" / name).read_bytes()).hexdigest()
                               for name in ("t8_retrieval_candidates_v1.json", "t8_query_normalization_regression_v1.json")},
            "parameters": {"embedding_model": retrieval.EMBEDDING_MODEL, "reranker_model": retrieval.RERANK_MODEL,
                           "bm25_limit": 10, "dense_limit": 10, "rrf_k": 60,
                           "reranker_candidate_limit": 10, "evidence_limit": 3},
            "metric_definition": "Hit 按既有 all_of/any_of；Recall@3: all_of核心命中比例, any_of任一命中为1；MRR首个核心记录倒数排名，候选外为0。",
            "candidate_13_answerable": candidate_summary,
            "expression_56": expression_summary, "bad_case": bad_case,
            "unanswerable_excluded": [{"case_id": c["case_id"], "question": c["question"],
                                       "old_top3": exp17.top_ids(results[normalize_retrieval_query(c["question"])]["old"], 3),
                                       "chroma_top3": exp17.top_ids(results[normalize_retrieval_query(c["question"])]["chroma"], 3)}
                                      for c in unanswerable],
            "real_embedding_queries": len(queries), "real_rerank_calls": len(rerank_cache),
            "rerank_reuse_policy": "相同 query 和有序候选共用一次真实结果；只改变 Dense 数据源。",
            "per_query": results, "result": "VECTOR_STORE_V1: " + ("PASS" if passed else "FAIL"),
        }
        RESULT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({key: output[key] for key in (
            "candidate_13_answerable", "expression_56", "bad_case", "unanswerable_excluded", "result")},
            ensure_ascii=False, indent=2))
        store.client.close()
        return 0 if passed else 1
    except Exception as error:
        print(f"A/B 未完成：{type(error).__name__}；未输出接口响应、密钥或其他敏感异常。", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
