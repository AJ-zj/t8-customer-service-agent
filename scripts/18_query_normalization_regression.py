"""只读评估礼貌词和产品词对 T8 检索稳定性的影响。"""
import importlib.util
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
REGRESSION_PATH = ROOT / "data/evaluation/t8_query_normalization_regression_v1.json"
VARIANT_TYPES = ("original", "polite", "product", "polite_product")
TOP_K = 3
BAD_QUESTION = "请问石头 T8 的主刷怎么清理？"
BAD_VARIANTS = {
    "original": "主刷怎么清理？",
    "polite": "请问主刷怎么清理？",
    "product": "石头 T8 的主刷怎么清理？",
    "polite_product": BAD_QUESTION,
}


def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载现有模块：{filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def add_polite(question):
    return question if "请问" in question else "请问" + question


def add_product(question):
    has_product = "石头" in question or re.search(r"T8", question, re.IGNORECASE)
    return question if has_product else "石头 T8 " + question


def expected_variants(question):
    product = add_product(question)
    return {
        "original": question,
        "polite": add_polite(question),
        "product": product,
        "polite_product": add_polite(product),
    }


def validate_regression_set(candidate, records, answerable):
    dataset = json.loads(REGRESSION_PATH.read_text(encoding="utf-8"))
    cases = dataset.get("cases")
    if (dataset.get("dataset_id") != "t8_query_normalization_regression_v1"
            or dataset.get("source_candidate_dataset") != candidate.get("dataset_id")
            or dataset.get("source_corpus_sha256") != candidate["source"]["corpus_sha256"]
            or not isinstance(cases, list) or dataset.get("case_count") != len(cases)):
        raise RuntimeError("Normalization regression 集元数据与当前候选集不匹配。")
    if len(answerable) != 13 or len(cases) != 56:
        raise RuntimeError(f"预期13个基础问题及56条回归表达，实际为{len(answerable)}及{len(cases)}。")
    ids = [case.get("case_id") for case in cases]
    if len(set(ids)) != len(ids):
        raise RuntimeError("回归集 case_id 重复。")

    source_by_id = {case["case_id"]: case for case in answerable}
    groups = defaultdict(list)
    for case in cases:
        groups[case.get("base_case_id")].append(case)
    if set(groups) != set(source_by_id) | {"REAL-MAIN-BRUSH"}:
        raise RuntimeError("回归集基础 case 分组与候选 answerable 集不一致。")
    for base_id, source_case in source_by_id.items():
        group = groups[base_id]
        if len(group) != 4 or {row.get("query_variant") for row in group} != set(VARIANT_TYPES):
            raise RuntimeError(f"{base_id} 必须恰有四种 query_variant。")
        expected = expected_variants(source_case["question"])
        for row in group:
            if (row.get("source") != "answerable_candidate"
                    or row.get("original_question") != source_case["question"]
                    or row.get("question") != expected[row["query_variant"]]
                    or row.get("answerability") != "answerable"
                    or row.get("core_match_policy") != source_case["core_match_policy"]
                    or row.get("core_knowledge") != source_case["core_knowledge"]):
                raise RuntimeError(f"{row.get('case_id')} 查询表达或继承的核心标注发生变化。")

    bad_group = groups["REAL-MAIN-BRUSH"]
    if len(bad_group) != 4:
        raise RuntimeError("真实 Bad Case 必须包含四种表达。")
    for row in bad_group:
        index = row["core_knowledge"][0]["record_index_zero_based"]
        if (row.get("source") != "real_bad_case" or index != 14
                or records[index].get("pdf_page") != 15
                or "软胶主刷" not in records[index].get("text", "")
                or row.get("original_question") != BAD_QUESTION
                or row.get("question") != BAD_VARIANTS.get(row.get("query_variant"))):
            raise RuntimeError("真实主刷 Bad Case 标注或表达与核验依据不一致。")
    return cases


def rank_map(result):
    return {index: rank for rank, (index, _) in enumerate(result["reranked"], 1)}


def relevant_case_rank(case, result):
    ranks = rank_map(result)
    values = [ranks.get(item["record_index_zero_based"])
              for item in case["core_knowledge"]]
    found = [rank for rank in values if rank is not None]
    if case["core_match_policy"] == "any_of":
        return min(found) if found else None
    return max(found) if found and len(found) == len(values) else None


def rank_text(case, result):
    ranks = rank_map(result)
    return ", ".join(
        f"{item['record_index_zero_based']}:{ranks.get(item['record_index_zero_based'], 'NOT_FOUND')}"
        for item in case["core_knowledge"]
    )


def record_name(index, records):
    record = records[index]
    title = record.get("title") or record.get("section") or "（无标题）"
    return f"record_index={index}/第{record.get('pdf_page')}页《{title}》"


def relevant_names(case, records):
    return "; ".join(record_name(item["record_index_zero_based"], records)
                      for item in case["core_knowledge"])


def hit(case, result, exp17, k=TOP_K):
    return exp17.is_hit(case, exp17.top_ids(result, k))


def metric_summary(cases, results, exp17):
    hit1 = sum(hit(case, results[case["case_id"]], exp17, 1) for case in cases)
    hit3, recall3, mrr = exp17.metrics(cases, results)
    ranks = [relevant_case_rank(case, results[case["case_id"]]) for case in cases]
    found = [rank for rank in ranks if rank is not None]
    return {
        "hit1": hit1, "hit3": hit3, "recall3": recall3, "mrr": mrr,
        "mean_rank": statistics.fmean(found) if found else None,
        "median_rank": statistics.median(found) if found else None,
        "rank_found": len(found),
    }


def print_paired_ranks(cases, records, baseline, normalized):
    print("\n成对 Reranker 排名（rank_delta = Baseline rank - Normalized rank）：")
    for case in cases:
        case_id = case["case_id"]
        base_rank = relevant_case_rank(case, baseline[case_id])
        norm_rank = relevant_case_rank(case, normalized[case_id])
        delta = base_rank - norm_rank if base_rank is not None and norm_rank is not None else "NOT_FOUND"
        base_text = base_rank if base_rank is not None else "NOT_FOUND"
        norm_text = norm_rank if norm_rank is not None else "NOT_FOUND"
        normalized_query = case.get("normalized_query")
        print(f"{case['base_case_id']} | {case['query_variant']} | "
              f"original_question={case['original_question']} | user_query={case['question']} | "
              f"retrieval_query={normalized_query} | relevant={relevant_names(case, records)} | "
              f"Baseline rank={base_text} [{rank_text(case, baseline[case_id])}] | "
              f"Normalized rank={norm_text} [{rank_text(case, normalized[case_id])}] | "
              f"rank_delta={delta}")


def classify_pairs(cases, records, baseline, normalized, exp17):
    improved, regressed, rank_up_same_hit, rank_down_same_hit = [], [], [], []
    deltas, top3_miss_to_hit, top3_hit_to_miss = [], 0, 0
    for case in cases:
        case_id = case["case_id"]
        before, after = baseline[case_id], normalized[case_id]
        before_rank, after_rank = relevant_case_rank(case, before), relevant_case_rank(case, after)
        before_hit, after_hit = hit(case, before, exp17), hit(case, after, exp17)
        delta = before_rank - after_rank if before_rank is not None and after_rank is not None else None
        if delta is not None:
            deltas.append(delta)
        if not before_hit and after_hit:
            top3_miss_to_hit += 1
        if before_hit and not after_hit:
            top3_hit_to_miss += 1
        details = {
            "base_case_id": case["base_case_id"],
            "query_variant": case["query_variant"],
            "original_question": case["original_question"],
            "question": case["question"],
            "retrieval_query": case["normalized_query"],
            "relevant_record": relevant_names(case, records),
            "Baseline_rank": before_rank if before_rank is not None else "NOT_FOUND",
            "Normalized_rank": after_rank if after_rank is not None else "NOT_FOUND",
            "rank_delta": delta if delta is not None else "NOT_FOUND",
            "Baseline_Hit@3": before_hit,
            "Normalized_Hit@3": after_hit,
        }
        if (not before_hit and after_hit) or (delta is not None and delta > 0):
            improved.append(details)
        if before_hit and not after_hit:
            regressed.append(details)
        if before_hit and after_hit and delta is not None and delta > 0:
            rank_up_same_hit.append(details)
        if before_hit and after_hit and delta is not None and delta < 0:
            rank_down_same_hit.append(details)
    counts = {
        "rank_improved": sum(delta > 0 for delta in deltas),
        "rank_unchanged": sum(delta == 0 for delta in deltas),
        "rank_regressed": sum(delta < 0 for delta in deltas),
        "rank_not_found_pairs": len(cases) - len(deltas),
        "top3_miss_to_hit": top3_miss_to_hit,
        "top3_hit_to_miss": top3_hit_to_miss,
    }
    return improved, regressed, rank_up_same_hit, rank_down_same_hit, counts


def rank_shift(values):
    if not values:
        return "n=0 (NOT_FOUND)", None
    mean = statistics.fmean(values)
    median = statistics.median(values)
    mean_abs = statistics.fmean(abs(value) for value in values)
    return f"n={len(values)}, mean={mean:+.3f}, median={median:+.3f}, mean_abs={mean_abs:.3f}", mean_abs


def print_product_stability(cases, baseline, normalized):
    by_group = defaultdict(dict)
    for case in cases:
        by_group[case["base_case_id"]][case["query_variant"]] = case
    print("\n产品词/礼貌词相对同组 original 的相关记录平均排名变化（变体 rank - original rank；正数表示变差）：")
    for variant in ("polite", "product", "polite_product"):
        shifts = {"Baseline": [], "Normalized": []}
        for group in by_group.values():
            if "original" not in group or variant not in group:
                continue
            original, decorated = group["original"], group[variant]
            for label, result_map in (("Baseline", baseline), ("Normalized", normalized)):
                orig_rank = relevant_case_rank(original, result_map[original["case_id"]])
                decorated_rank = relevant_case_rank(decorated, result_map[decorated["case_id"]])
                if orig_rank is not None and decorated_rank is not None:
                    shifts[label].append(decorated_rank - orig_rank)
        base_text, base_abs = rank_shift(shifts["Baseline"])
        norm_text, norm_abs = rank_shift(shifts["Normalized"])
        print(f"  {variant}: Baseline[{base_text}] | Normalized[{norm_text}] | "
              f"mean_abs_change_reduced={base_abs is not None and norm_abs is not None and norm_abs < base_abs}")


def print_case_lists(label, cases):
    print(f"\n{label}:")
    print(json.dumps(cases, ensure_ascii=False, indent=2) if cases else "[]")


def main():
    try:
        evaluator = load_script("t8_norm_eval", "08_evaluate_retrieval.py")
        candidate, records, answerable, _ = evaluator.verify_inputs()
        regression = validate_regression_set(candidate, records, answerable)
        exp17 = load_script("t8_norm_exp17", "17_retrieval_improvement_ablation.py")
        hybrid = load_script("t8_norm_hybrid", "07_hybrid_search.py")
        reranker = load_script("t8_norm_reranker", "11_rerank_experiment.py")
        bm_records, bm25_index = hybrid.bm25_module.build_index()
        live_records, corpus_hash = hybrid.embedding_module.load_corpus()
        if records != bm_records or records != live_records:
            raise RuntimeError("候选集、BM25 与 Embedding 的知识记录顺序不一致。")
        if corpus_hash != candidate["source"]["corpus_sha256"]:
            raise RuntimeError("当前知识库指纹与候选集不一致。")
        document_vectors = reranker.load_cached_vectors(
            hybrid.embedding_module, len(records), corpus_hash
        )
        api_key = hybrid.embedding_module.get_api_key()

        query_by_case = {}
        unique_queries = []
        for case in regression:
            normalized_query = exp17.normalize_query(case["question"])
            if not normalized_query.strip():
                raise RuntimeError(f"{case['case_id']} normalization 产生空查询。")
            case["normalized_query"] = normalized_query
            query_by_case[case["case_id"]] = (case["question"], normalized_query)
            unique_queries.extend((case["question"], normalized_query))
        unique_queries = list(dict.fromkeys(unique_queries))
        query_vectors = hybrid.embedding_module.request_embeddings(unique_queries, api_key)
        if len(query_vectors) != len(unique_queries):
            raise RuntimeError("唯一查询向量数量不匹配。")
        query_vector_map = dict(zip(unique_queries, query_vectors))

        print(f"输入核验通过：{len(answerable)}道 answerable 基础题，"
              f"回归集{len(regression)}条（52条派生表达+4条真实 Bad Case）。")
        print(f"文档向量读取现有缓存；本次唯一检索文本={len(unique_queries)}；"
              "BM25、Embedding、RRF、Reranker 均复用现有实现；未调用聊天模型。")
        retrieval_by_query = {}
        for query in unique_queries:
            retrieval_by_query[query] = exp17.retrieve(
                query, query_vector_map[query], records, hybrid, reranker,
                bm25_index, document_vectors, api_key
            )

        baseline, normalized = {}, {}
        for case in regression:
            raw_query, norm_query = query_by_case[case["case_id"]]
            baseline[case["case_id"]] = retrieval_by_query[raw_query]
            normalized[case["case_id"]] = retrieval_by_query[norm_query]
        print_paired_ranks(regression, records, baseline, normalized)

        summaries = {
            "Baseline": metric_summary(regression, baseline, exp17),
            "Normalized": metric_summary(regression, normalized, exp17),
        }
        print("\n总体指标（全部回归条目，含独立 Bad Case 表达）：")
        print("| Method | Hit@1 | Hit@3 | Recall@3 | MRR | Mean Relevant Rank | Median Relevant Rank | Rank found |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|")
        for name, values in summaries.items():
            mean = f"{values['mean_rank']:.3f}" if values["mean_rank"] is not None else "N/A"
            med = f"{values['median_rank']:.3f}" if values["median_rank"] is not None else "N/A"
            print(f"| {name} | {values['hit1']}/{len(regression)} | "
                  f"{values['hit3']}/{len(regression)} | {values['recall3']:.4f} | "
                  f"{values['mrr']:.4f} | {mean} | {med} | {values['rank_found']}/{len(regression)} |")

        print("\n按 query_variant 分组的 Hit@3：")
        print("| Variant Type | Baseline Hit@3 | Normalized Hit@3 |")
        print("|---|---:|---:|")
        for variant in VARIANT_TYPES:
            group = [case for case in regression
                     if case.get("source") == "answerable_candidate"
                     and case["query_variant"] == variant]
            base_metric = metric_summary(group, baseline, exp17)
            norm_metric = metric_summary(group, normalized, exp17)
            print(f"| {variant} | {base_metric['hit3']}/{len(group)} | "
                  f"{norm_metric['hit3']}/{len(group)} |")
        bad_group = [case for case in regression if case.get("source") == "real_bad_case"]
        base_bad = metric_summary(bad_group, baseline, exp17)
        norm_bad = metric_summary(bad_group, normalized, exp17)
        print(f"| real_bad_case | {base_bad['hit3']}/{len(bad_group)} | "
              f"{norm_bad['hit3']}/{len(bad_group)} |")

        improved, regressed, rank_up_same, rank_down_same, counts = classify_pairs(
            regression, records, baseline, normalized, exp17
        )
        print_product_stability(regression, baseline, normalized)
        print_case_lists("improved_cases", improved)
        print_case_lists("regressed_cases", regressed)
        print_case_lists("rank_improved_but_same_hit", rank_up_same)
        print_case_lists("rank_regressed_but_same_hit", rank_down_same)
        print("\n排名变化统计：" + json.dumps(counts, ensure_ascii=False))

        print("\nNORMALIZATION_REGRESSION_RESULT")
        result = {
            "total_cases": len(regression),
            "baseline_hit_at_3": f"{summaries['Baseline']['hit3']}/{len(regression)}",
            "normalized_hit_at_3": f"{summaries['Normalized']['hit3']}/{len(regression)}",
            "baseline_mrr": round(summaries["Baseline"]["mrr"], 4),
            "normalized_mrr": round(summaries["Normalized"]["mrr"], 4),
            "improved_count": len(improved),
            "regressed_count": len(regressed),
            "rank_improved_count": counts["rank_improved"],
            "rank_regressed_count": counts["rank_regressed"],
            "rank_unchanged_count": counts["rank_unchanged"],
            "rank_not_found_pairs": counts["rank_not_found_pairs"],
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("PRODUCTION_DECISION: NOT_MADE")
        return 0
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
        print(f"回归实验失败：{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
