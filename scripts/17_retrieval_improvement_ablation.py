"""只读对照主刷 Bad Case 的 query normalization 与 evidence Top-4。"""
import importlib.util
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
BAD_QUESTION = "请问石头 T8 的主刷怎么清理？"
BAD_TARGET = 14
TOP3 = 3
TOP4 = 4
VARIANTS = (
    ("A Baseline", False, TOP3),
    ("B Normalization", True, TOP3),
    ("C Top4", False, TOP4),
    ("D Norm + Top4", True, TOP4),
)


def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载现有模块：{filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_query(question):
    """确定性删除项目内低区分度词，不改动原始 question。"""
    retrieval_query = question
    for pattern in (r"请问", r"石头", r"(?i)T8", r"的"):
        retrieval_query = re.sub(pattern, "", retrieval_query)
    retrieval_query = re.sub(r"[，。！？、,.!?;；:：\s]+", " ", retrieval_query).strip()
    return retrieval_query or question


def record_label(index, records):
    record = records[index]
    title = record.get("title") or record.get("section") or "（无标题）"
    return f"record_index={index} 第{record.get('pdf_page')}页《{title}》"


def retrieve(query, query_vector, records, hybrid, reranker, bm25_index,
             document_vectors, api_key):
    """用现有检索模块完成单条检索，保留原始 JSONL 索引。"""
    bm_scores, bm_ranks, bm_candidates = hybrid.rank_bm25(query, records, bm25_index)
    em_ranks, em_candidates = hybrid.rank_embedding(query_vector, records, document_vectors)
    em_full = hybrid.embedding_module.rank_records(query_vector, records, document_vectors)
    em_scores = dict(em_full)
    fused = hybrid.reciprocal_rank_fusion(
        bm_candidates, em_candidates, k=hybrid.RRF_K
    )
    candidate_ids = list(dict.fromkeys(
        [index for index, _ in bm_candidates] + [index for index, _ in em_candidates]
    ))
    candidates = [(index, records[index]) for index in candidate_ids]
    reranked = reranker.request_rerank(query, candidates, api_key) if candidates else []
    return {
        "bm_scores": bm_scores, "bm_ranks": bm_ranks,
        "em_scores": em_scores, "em_ranks": em_ranks,
        "rrf": fused, "reranked": reranked, "candidate_ids": candidate_ids,
    }


def target_stage(result, name, target):
    if name == "BM25":
        rank, score = result["bm_ranks"].get(target), result["bm_scores"][target]
    elif name == "Embedding":
        rank, score = result["em_ranks"].get(target), result["em_scores"].get(target)
    elif name == "RRF":
        row = next(((rank, item[1]) for rank, item in enumerate(result["rrf"], 1)
                    if item[0] == target), None)
        rank, score = row if row else (None, None)
    else:
        row = next(((rank, item[1]) for rank, item in enumerate(result["reranked"], 1)
                    if item[0] == target), None)
        rank, score = row if row else (None, None)
    if rank is None or score is None:
        return "NOT_FOUND"
    return f"rank={rank}, score={score:.8f}"


def core_ids(case):
    return [item["record_index_zero_based"] for item in case["core_knowledge"]]


def is_hit(case, ids):
    targets = core_ids(case)
    found = [target in ids for target in targets]
    return all(found) if case["core_match_policy"] == "all_of" else any(found)


def recall_at_k(case, ids):
    targets = core_ids(case)
    found = [target in set(ids) for target in targets]
    if case["core_match_policy"] == "any_of":
        return float(any(found))
    return sum(found) / len(found) if found else 0.0


def top_ids(result, limit):
    return [index for index, _ in result["reranked"][:limit]]


def metrics(cases, results):
    hits, recalls, reciprocal_ranks = 0, [], []
    for case in cases:
        result = results[case["case_id"]]
        ids3 = top_ids(result, TOP3)
        hits += int(is_hit(case, ids3))
        recalls.append(recall_at_k(case, ids3))
        target_ranks = [rank for rank, (index, _) in enumerate(result["reranked"], 1)
                        if index in set(core_ids(case))]
        reciprocal_ranks.append(1 / min(target_ranks) if target_ranks else 0.0)
    return hits, sum(recalls) / len(cases), sum(reciprocal_ranks) / len(cases)


def describe_ids(ids, records):
    return "；".join(record_label(index, records) for index in ids) if ids else "（无结果）"


def print_bad_case(results, records):
    print("\nBad Case record 14 diagnostic")
    for name, normalized, evidence_limit in VARIANTS:
        query_variant = "B Normalization" if normalized else "A Baseline"
        result = results[("BAD", query_variant)]
        retrieval_query = normalize_query(BAD_QUESTION) if normalized else BAD_QUESTION
        print(f"\n{name}")
        print(f"  original_question: {BAD_QUESTION}")
        print(f"  retrieval_query: {retrieval_query}")
        print(f"  BM25: {target_stage(result, 'BM25', BAD_TARGET)}")
        print(f"  Embedding: {target_stage(result, 'Embedding', BAD_TARGET)}")
        print(f"  RRF: {target_stage(result, 'RRF', BAD_TARGET)}")
        print(f"  Reranker: {target_stage(result, 'Reranker', BAD_TARGET)}")
        print("  final_evidence:")
        evidence = result["reranked"][:evidence_limit]
        for rank, (index, score) in enumerate(evidence, 1):
            print(f"    rank={rank} record_index={index} pdf_page={records[index].get('pdf_page')} "
                  f"reranker_score={score:.8f}")
        print(f"  record_index=14 entered_final_evidence: "
              f"{any(index == BAD_TARGET for index, _ in evidence)}")


def print_case_changes(name, cases, records, baseline, current):
    improved, regressed = [], []
    for case in cases:
        case_id = case["case_id"]
        before_ids, after_ids = top_ids(baseline[case_id], TOP3), top_ids(current[case_id], TOP3)
        before_hit, after_hit = is_hit(case, before_ids), is_hit(case, after_ids)
        item = {
            "case_id": case_id,
            "question": case["question"],
            "expected_relevant_records": [record_label(i, records) for i in core_ids(case)],
            "Baseline Top3": describe_ids(before_ids, records),
            f"{name} Top3": describe_ids(after_ids, records),
        }
        if not before_hit and after_hit:
            improved.append(item)
        elif before_hit and not after_hit:
            regressed.append(item)
    print(f"\nimproved_cases ({name}):")
    print(improved if improved else "[]")
    print(f"\nregressed_cases ({name}):")
    print(regressed if regressed else "[]")


def main():
    try:
        evaluator = load_script("t8_ablation_eval", "08_evaluate_retrieval.py")
        candidate, records, cases, _ = evaluator.verify_inputs()
        if len(cases) != 13:
            raise RuntimeError(f"预期13道 answerable case，实际为{len(cases)}。")
        hybrid = load_script("t8_ablation_hybrid", "07_hybrid_search.py")
        reranker = load_script("t8_ablation_reranker", "11_rerank_experiment.py")
        bm_records, bm25_index = hybrid.bm25_module.build_index()
        live_records, corpus_hash = hybrid.embedding_module.load_corpus()
        if records != bm_records or records != live_records:
            raise RuntimeError("评测集、BM25 与 Embedding 的知识记录顺序不一致。")
        if corpus_hash != candidate["source"]["corpus_sha256"]:
            raise RuntimeError("知识库指纹与候选集不一致，停止实验。")

        embedding = hybrid.embedding_module
        vectors = reranker.load_cached_vectors(embedding, len(records), corpus_hash)
        api_key = embedding.get_api_key()
        jobs = []
        for case in cases:
            jobs.extend([
                (case["case_id"], "A Baseline", case["question"]),
                (case["case_id"], "B Normalization", normalize_query(case["question"])),
            ])
        jobs.extend([
            ("BAD", "A Baseline", BAD_QUESTION),
            ("BAD", "B Normalization", normalize_query(BAD_QUESTION)),
        ])
        query_vectors = embedding.request_embeddings([query for _, _, query in jobs], api_key)
        if len(query_vectors) != len(jobs):
            raise RuntimeError("查询向量数量与实验任务数不匹配。")

        results = {}
        for (case_id, name, query), vector in zip(jobs, query_vectors):
            results[(case_id, name)] = retrieve(
                query, vector, records, hybrid, reranker, bm25_index, vectors, api_key
            )
        # Case results are keyed separately from the independent fixed Bad Case.
        case_results = {
            name: {case["case_id"]: results[(case["case_id"], name)] for case in cases}
            for name in ("A Baseline", "B Normalization")
        }
        case_results["C Top4"] = case_results["A Baseline"]
        case_results["D Norm + Top4"] = case_results["B Normalization"]

        print(f"输入校验通过：{len(records)}条记录，{len(cases)}道 answerable case；"
              "PDF/JSONL 指纹及核心知识标注已核验。")
        print(f"文档向量来自现有缓存；BM25阈值={hybrid.bm25_module.MIN_SCORE}，"
              f"RRF k={hybrid.RRF_K}，BM25/Embedding候选上限={hybrid.CANDIDATE_LIMIT}；"
              "未调用聊天模型。")
        print("record_index=14 Bad Case 是脚本内独立 regression case，不加入候选集，也不计入13题指标。")
        print_bad_case(results, records)

        print("\n统一 answerable 回归指标（13题；Hit@3、Recall@3、MRR 为公平主比较）：")
        print("| Variant | Hit@3 | Recall@3 | MRR | Hit@4 |")
        print("|---|---:|---:|---:|---:|")
        scores = {}
        for name, _, evidence_limit in VARIANTS:
            scores[name] = metrics(cases, case_results[name])
            hit3, recall3, mrr = scores[name]
            hit4 = (sum(is_hit(case, top_ids(case_results[name][case["case_id"]], TOP4))
                        for case in cases) if evidence_limit == TOP4 else None)
            hit4_text = f"{hit4}/{len(cases)}" if hit4 is not None else "N/A"
            print(f"| {name} | {hit3}/{len(cases)} | {recall3:.4f} | {mrr:.4f} | {hit4_text} |")
        print("Recall@3定义：all_of按必需核心记录命中比例计；any_of按替代核心证据命中记1，否则0；对题目宏平均。")
        print("MRR使用Reranker完整联合候选排名中第一个核心记录的倒数排名，再对题目宏平均。")
        print("注意：C/D 的 Hit@4 仅为补充指标，不能与 Baseline Hit@3 直接比较并宣称提升。")

        baseline = case_results["A Baseline"]
        for name in ("B Normalization", "C Top4", "D Norm + Top4"):
            print_case_changes(name, cases, records, baseline, case_results[name])
        print("\nEXPERIMENT_RESULT")
        for name, normalized, evidence_limit in VARIANTS:
            key = "B Normalization" if normalized else "A Baseline"
            ranked = results[("BAD", key)]["reranked"]
            rank = next((position for position, (index, _) in enumerate(ranked, 1)
                         if index == BAD_TARGET), None)
            entered = any(index == BAD_TARGET for index, _ in ranked[:evidence_limit])
            print(f"Bad Case {name}: record_index=14 reranker_rank="
                  f"{rank if rank is not None else 'NOT_FOUND'}，进入 evidence={entered}。")
        print("Answerable Hit@3：" + "；".join(
            f"{name}={scores[name][0]}/{len(cases)}" for name, _, _ in VARIANTS
        ))
        print("以上为本次只读实验观察；未修改生产检索、Top3、候选集或数据库，不作上线建议。")
        return 0
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
        print(f"实验失败：{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
