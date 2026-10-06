"""只读对照 RRF 候选池上限 10、20、41 对 T8 检索的影响。"""
import importlib.util
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
LIMITS = (10, 20, 41)
TOP_K = 3
RRF_K = 60


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
        raise RuntimeError(f"需要现有 Embedding 缓存：{embedding.CACHE}")
    saved = json.loads(embedding.CACHE.read_text(encoding="utf-8"))
    if (not isinstance(saved, dict) or saved.get("model") != embedding.MODEL
            or saved.get("corpus_sha256") != corpus_hash or saved.get("record_count") != count):
        raise RuntimeError("Embedding 缓存格式、模型或知识库指纹不匹配；停止实验，不重建缓存。")
    return embedding.validate_vectors(saved.get("vectors"), count)


def record_label(record):
    title = record.get("title") or record.get("section") or "（无标题）"
    return f"第{record.get('pdf_page')}页《{title}》"


def evaluate_case(case, records, hybrid, bm25_index, document_vectors, query_vector):
    bm25, embedding = hybrid.bm25_module, hybrid.embedding_module
    tokens = bm25.tokenize(case["question"])
    scores = bm25_index.get_scores(tokens) if tokens else [0.0] * len(records)
    bm_order = sorted(range(len(records)), key=lambda i: scores[i], reverse=True)
    bm_ranks = {index: rank for rank, index in enumerate(bm_order, start=1)}
    bm_eligible = [(index, bm_ranks[index]) for index in bm_order if scores[index] >= bm25.MIN_SCORE]
    em_order = embedding.rank_records(query_vector, records, document_vectors)
    em_ranks = {index: rank for rank, (index, _) in enumerate(em_order, start=1)}
    targets = [item["record_index_zero_based"] for item in case["core_knowledge"]]
    by_limit = {}

    for limit in LIMITS:
        bm_candidates = bm_eligible[:limit]
        em_candidates = [(index, em_ranks[index]) for index, _ in em_order[:limit]]
        fused = hybrid.reciprocal_rank_fusion(bm_candidates, em_candidates, k=RRF_K)
        rrf_ranks = {index: rank for rank, (index, _, _, _) in enumerate(fused, start=1)}
        rrf_scores = {index: score for index, score, _, _ in fused}
        top_ids = [index for index, _, _, _ in fused[:TOP_K]]
        hits = [index in top_ids for index in targets]
        passed = all(hits) if case["core_match_policy"] == "all_of" else any(hits)
        by_limit[limit] = {
            "passed": passed, "hits": hits, "targets": targets,
            "bm_ranks": bm_ranks, "bm_scores": scores,
            "bm_candidates": bm_candidates, "em_ranks": em_ranks,
            "em_candidates": em_candidates, "fused": fused,
            "rrf_ranks": rrf_ranks, "rrf_scores": rrf_scores,
            "top_ids": top_ids,
        }
    return by_limit


def print_focus(case, records, results):
    print(f"\n{case['case_id']} | {case['question']} | 规则={case['core_match_policy']}")
    for limit in LIMITS:
        result = results[limit]
        print(f"  候选上限={limit} | RRF Top-3：" + "；".join(
            f"{record_label(records[index])}({result['rrf_scores'][index]:.6f})"
            for index in result["top_ids"]
        ))
        for item, index, hit in zip(case["core_knowledge"], result["targets"], result["hits"]):
            bm_pool = dict(result["bm_candidates"])
            em_pool = dict(result["em_candidates"])
            brank, erank = bm_pool.get(index), em_pool.get(index)
            bcontrib = 1 / (RRF_K + brank) if brank is not None else 0.0
            econtrib = 1 / (RRF_K + erank) if erank is not None else 0.0
            rr = result["rrf_ranks"].get(index)
            rscore = result["rrf_scores"].get(index, 0.0)
            print(f"    核心 {item.get('pdf_location') or record_label(records[index])}："
                  f"BM25全库第{result['bm_ranks'][index]}名/贡献{bcontrib:.6f}；"
                  f"Embedding全库第{result['em_ranks'][index]}名/贡献{econtrib:.6f}；"
                  f"RRF第{rr if rr is not None else '未进候选'}名/{rscore:.6f}；Top-3命中={'是' if hit else '否'}")
        print(f"    题目判定：{'通过' if result['passed'] else '未通过'}")


def main():
    try:
        evaluator = load_script("t8_ablation_evaluation", "08_evaluate_retrieval.py")
        candidate, records, answerable, _ = evaluator.verify_inputs()
        hybrid = load_script("t8_ablation_hybrid", "07_hybrid_search.py")
        bm_records, bm_index = hybrid.bm25_module.build_index()
        em_records, corpus_hash = hybrid.embedding_module.load_corpus()
        if records != bm_records or records != em_records:
            raise RuntimeError("检索模块读取的知识记录或顺序不一致，停止实验。")
        if corpus_hash != candidate["source"]["corpus_sha256"]:
            raise RuntimeError("JSONL 在校验后发生变化，停止实验。")

        embedding = hybrid.embedding_module
        vectors = load_cached_vectors(embedding, len(records), corpus_hash)
        api_key = embedding.get_api_key()
        queries = [case["question"] for case in answerable]
        query_vectors = embedding.request_embeddings(queries, api_key)
        if len(query_vectors) != len(answerable):
            raise RuntimeError("有答案问题的查询向量数量不匹配。")
        print(f"输入 SHA-256 与核心知识标注校验通过；{len(records)}条文档向量来自现有缓存。")
        print(f"固定参数：模型={embedding.MODEL}，BM25阈值={hybrid.bm25_module.MIN_SCORE}，RRF_K={RRF_K}，Top-K={TOP_K}。")

        all_results = {}
        for case, query_vector in zip(answerable, query_vectors):
            all_results[case["case_id"]] = evaluate_case(
                case, records, hybrid, bm_index, vectors, query_vector
            )
        focus_ids = {"T8-C05", "T8-C10", "T8-C13"}
        for case in answerable:
            if case["case_id"] in focus_ids:
                print_focus(case, records, all_results[case["case_id"]])

        print("\n候选上限对照（题目级 Hit@3，仅13道有答案题）：")
        print("上限 | 通过数 | 失败题目")
        for limit in LIMITS:
            failed = [case["case_id"] for case in answerable
                      if not all_results[case["case_id"]][limit]["passed"]]
            passed = len(answerable) - len(failed)
            print(f"{limit:>4} | {passed:>2}/{len(answerable)} | {', '.join(failed) if failed else '无'}")
        print("说明：这只是当前候选集上的诊断性对照；不代表泛化性能，也不改变线上默认候选上限。")
        return 0
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
        print(f"实验失败：{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
