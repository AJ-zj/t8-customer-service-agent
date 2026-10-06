"""只读比较保守 Query Normalization 与脚本17当前规则。"""
import importlib.util
import json
import re
import statistics
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
REGRESSION_PATH = ROOT / "data/evaluation/t8_query_normalization_regression_v1.json"
TOP_K = 3
BAD_VARIANTS = {
    "original": "主刷怎么清理？",
    "polite": "请问主刷怎么清理？",
    "product": "石头 T8 的主刷怎么清理？",
    "polite_product": "请问石头 T8 的主刷怎么清理？",
}


def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载现有模块：{filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def conservative_normalize(question):
    """仅删指定实体/开头礼貌词；不删“的”、其他虚词或标点。"""
    query = re.sub(r"^\s*请问", "", question, count=1)
    query = query.replace("石头", "")
    query = re.sub(r"T8", "", query, flags=re.IGNORECASE)
    query = re.sub(r"\s+", " ", query).strip()
    return query or question.strip() or question


def validate_inputs(evaluator, exp18):
    candidate, records, answerable, _ = evaluator.verify_inputs()
    regression = json.loads(REGRESSION_PATH.read_text(encoding="utf-8"))
    cases = regression.get("cases", [])
    exp18.validate_regression_set(candidate, records, answerable)
    if len(cases) != 56:
        raise RuntimeError(f"回归集应为56条，实际为{len(cases)}条。")
    return candidate, records, answerable, cases


def case_rank(case, result, exp18):
    return exp18.relevant_case_rank(case, result)


def case_hit(case, result, exp17):
    return exp17.is_hit(case, exp17.top_ids(result, TOP_K))


def summary(cases, results, exp17, exp18):
    if not cases:
        return {"hit1": 0, "hit3": 0, "recall3": 0.0, "mrr": 0.0,
                "mean_rank": None, "median_rank": None, "rank_found": 0}
    hit1 = sum(exp17.is_hit(case, exp17.top_ids(results[case["case_id"]], 1))
               for case in cases)
    hit3, recall3, mrr = exp17.metrics(cases, results)
    ranks = [case_rank(case, results[case["case_id"]], exp18) for case in cases]
    found = [rank for rank in ranks if rank is not None]
    return {
        "hit1": hit1, "hit3": hit3, "recall3": recall3, "mrr": mrr,
        "mean_rank": statistics.fmean(found) if found else None,
        "median_rank": statistics.median(found) if found else None,
        "rank_found": len(found),
    }


def case_detail(case, records, conservative, current, exp17, exp18):
    case_id = case["case_id"]
    b_rank = case_rank(case, conservative[case_id], exp18)
    c_rank = case_rank(case, current[case_id], exp18)
    b_hit, c_hit = case_hit(case, conservative[case_id], exp17), case_hit(case, current[case_id], exp17)
    return {
        "case_id": case_id,
        "query_variant": case["query_variant"],
        "original_question": case["original_question"],
        "conservative_retrieval_query": case["conservative_query"],
        "current_retrieval_query": case["current_query"],
        "relevant_record": exp18.relevant_names(case, records),
        "conservative_rank": b_rank if b_rank is not None else "NOT_FOUND",
        "current_rank": c_rank if c_rank is not None else "NOT_FOUND",
        "conservative_hit_at_3": b_hit,
        "current_hit_at_3": c_hit,
    }, b_rank, c_rank, b_hit, c_hit


def compare_conservative_current(cases, records, conservative, current, exp17, exp18):
    current_better, conservative_better, caused_regression = [], [], []
    same_result = 0
    differences = []
    for case in cases:
        detail, b_rank, c_rank, b_hit, c_hit = case_detail(
            case, records, conservative, current, exp17, exp18
        )
        if b_hit and not c_hit:
            caused_regression.append(detail)
        if b_rank is None and c_rank is not None:
            current_better.append(detail)
        elif c_rank is None and b_rank is not None:
            conservative_better.append(detail)
        elif b_rank is not None and c_rank is not None and c_rank < b_rank:
            current_better.append(detail)
        elif b_rank is not None and c_rank is not None and b_rank < c_rank:
            conservative_better.append(detail)
        elif b_rank == c_rank and b_hit == c_hit:
            same_result += 1
        else:
            differences.append(detail)
    return current_better, conservative_better, caused_regression, same_result, differences


def baseline_change_counts(cases, baseline, other, exp17, exp18):
    improved = regressed = miss_to_hit = hit_to_miss = 0
    for case in cases:
        before = baseline[case["case_id"]]
        after = other[case["case_id"]]
        before_rank, after_rank = case_rank(case, before, exp18), case_rank(case, after, exp18)
        before_hit, after_hit = case_hit(case, before, exp17), case_hit(case, after, exp17)
        if before_rank is not None and after_rank is not None:
            improved += int(after_rank < before_rank)
            regressed += int(after_rank > before_rank)
        miss_to_hit += int(not before_hit and after_hit)
        hit_to_miss += int(before_hit and not after_hit)
    return {"rank_improved": improved, "rank_regressed": regressed,
            "miss_to_hit": miss_to_hit, "hit_to_miss": hit_to_miss}


def print_case_list(title, rows):
    print(f"\n{title}:")
    print(json.dumps(rows, ensure_ascii=False, indent=2) if rows else "[]")


def de_special(cases, records, conservative, current, exp17, exp18):
    selected = [case for case in cases
                if case["question"].count("的") > case["current_query"].count("的")
                and case["question"].count("的") == case["conservative_query"].count("的")]
    b_summary = summary(selected, conservative, exp17, exp18)
    c_summary = summary(selected, current, exp17, exp18)
    b_better = c_better = same = 0
    for case in selected:
        b_rank = case_rank(case, conservative[case["case_id"]], exp18)
        c_rank = case_rank(case, current[case["case_id"]], exp18)
        if b_rank is None and c_rank is not None:
            c_better += 1
        elif c_rank is None and b_rank is not None:
            b_better += 1
        elif b_rank is not None and c_rank is not None:
            c_better += int(c_rank < b_rank)
            b_better += int(b_rank < c_rank)
            same += int(b_rank == c_rank)
        else:
            same += 1
    print("\n“的”专项（C 删除了“的”，B 保留）：")
    print(f"case_count={len(selected)}")
    print(f"Conservative Hit@3={b_summary['hit3']}/{len(selected)}；"
          f"Current Hit@3={c_summary['hit3']}/{len(selected)}")
    print(f"Conservative MRR={b_summary['mrr']:.4f}；Current MRR={c_summary['mrr']:.4f}")
    print(f"Conservative mean_rank={b_summary['mean_rank']}；Current mean_rank={c_summary['mean_rank']}")
    print(f"Current 更优={c_better}；Conservative 更优={b_better}；排名相同={same}")
    for case in selected:
        print(f"  {case['case_id']} | 保守={case['conservative_query']} | 当前={case['current_query']}")
    return selected, b_summary, c_summary, b_better, c_better, same


def print_bad_brush(cases, baseline, conservative, current, exp18):
    group = [case for case in cases if case.get("source") == "real_bad_case"]
    print("\n真实主刷 Bad Case 四种表达（Reranker relevant rank）：")
    for case in group:
        cid = case["case_id"]
        ranks = [case_rank(case, result[cid], exp18)
                 for result in (baseline, conservative, current)]
        display = [rank if rank is not None else "NOT_FOUND" for rank in ranks]
        print(f"{case['question']} | record_index=14 | Baseline={display[0]} | "
              f"Conservative={display[1]} | Current={display[2]}")


def main():
    try:
        evaluator = load_script("t8_rule_eval", "08_evaluate_retrieval.py")
        exp17 = load_script("t8_rule_exp17", "17_retrieval_improvement_ablation.py")
        exp18 = load_script("t8_rule_exp18", "18_query_normalization_regression.py")
        reranker = load_script("t8_rule_reranker", "11_rerank_experiment.py")
        candidate, records, answerable, cases = validate_inputs(evaluator, exp18)
        if len(cases) != 56:
            raise RuntimeError("本轮必须使用56条查询表达。")
        hybrid = load_script("t8_rule_hybrid", "07_hybrid_search.py")
        bm_records, bm25_index = hybrid.bm25_module.build_index()
        live_records, corpus_hash = hybrid.embedding_module.load_corpus()
        if records != bm_records or records != live_records:
            raise RuntimeError("候选集、BM25、Embedding 的知识记录顺序不一致。")
        if corpus_hash != candidate["source"]["corpus_sha256"]:
            raise RuntimeError("当前知识库指纹与候选集不一致。")
        vectors = reranker.load_cached_vectors(hybrid.embedding_module, len(records), corpus_hash)
        api_key = hybrid.embedding_module.get_api_key()

        jobs = {}
        for case in cases:
            query = case["question"]
            case["conservative_query"] = conservative_normalize(query)
            case["current_query"] = exp17.normalize_query(query)
            if not case["conservative_query"] or not case["current_query"]:
                raise RuntimeError(f"{case['case_id']} 规范化后查询为空。")
            jobs[query] = query
            jobs[case["conservative_query"]] = case["conservative_query"]
            jobs[case["current_query"]] = case["current_query"]
        unique_queries = list(jobs)
        query_vectors = hybrid.embedding_module.request_embeddings(unique_queries, api_key)
        if len(query_vectors) != len(unique_queries):
            raise RuntimeError("查询向量数量与唯一检索文本数量不一致。")
        vectors_by_query = dict(zip(unique_queries, query_vectors))
        print(f"输入核验通过：{len(answerable)}道基础题、{len(cases)}条回归表达；"
              f"本轮唯一 retrieval_query={len(unique_queries)}。")
        print("B规则：仅去开头‘请问’、‘石头’、‘T8’，再折叠/修剪空格；保留‘的’及标点。")
        print("C规则：直接调用脚本17的 normalize_query()；文档向量使用现有缓存，未调用聊天模型。")

        result_by_query = {}
        for query in unique_queries:
            result_by_query[query] = exp17.retrieve(
                query, vectors_by_query[query], records, hybrid, reranker,
                bm25_index, vectors, api_key
            )
        baseline = {case["case_id"]: result_by_query[case["question"]] for case in cases}
        conservative = {case["case_id"]: result_by_query[case["conservative_query"]] for case in cases}
        current = {case["case_id"]: result_by_query[case["current_query"]] for case in cases}

        summaries = {
            "A Baseline": summary(cases, baseline, exp17, exp18),
            "B Conservative": summary(cases, conservative, exp17, exp18),
            "C Current": summary(cases, current, exp17, exp18),
        }
        print("\nA/B/C 总体指标（56条；MRR与排名口径沿用脚本18）：")
        print("| Method | Hit@1 | Hit@3 | Recall@3 | MRR | Mean Relevant Rank | Median Relevant Rank |")
        print("|---|---:|---:|---:|---:|---:|---:|")
        for name, value in summaries.items():
            mean = f"{value['mean_rank']:.3f}" if value["mean_rank"] is not None else "N/A"
            median = f"{value['median_rank']:.3f}" if value["median_rank"] is not None else "N/A"
            print(f"| {name} | {value['hit1']}/{len(cases)} | {value['hit3']}/{len(cases)} | "
                  f"{value['recall3']:.4f} | {value['mrr']:.4f} | {mean} | {median} |")

        b_vs_a = baseline_change_counts(cases, baseline, conservative, exp17, exp18)
        c_vs_a = baseline_change_counts(cases, baseline, current, exp17, exp18)
        print("\n相对 Baseline 的排名/Top3变化：")
        print(f"Conservative: {json.dumps(b_vs_a, ensure_ascii=False)}")
        print(f"Current: {json.dumps(c_vs_a, ensure_ascii=False)}")

        better, conservative_better, caused_regression, same_count, residual = compare_conservative_current(
            cases, records, conservative, current, exp17, exp18
        )
        print_case_list("current_better_than_conservative", better)
        print_case_list("conservative_better_than_current", conservative_better)
        print_case_list("current_caused_regression (B Top3 hit → C miss)", caused_regression)
        print(f"\nsame_result count={same_count} (effective relevant rank and Hit@3 are equal)")
        if residual:
            print_case_list("same effective rank but differing hit-state / unresolved", residual)

        de_cases, de_b, de_c, de_b_better, de_c_better, de_same = de_special(
            cases, records, conservative, current, exp17, exp18
        )
        print_bad_brush(cases, baseline, conservative, current, exp18)

        b_summary, c_summary = summaries["B Conservative"], summaries["C Current"]
        print("\nRULE_ABLATION_RESULT")
        result = {
            "baseline_hit_at_3": f"{summaries['A Baseline']['hit3']}/{len(cases)}",
            "conservative_hit_at_3": f"{b_summary['hit3']}/{len(cases)}",
            "current_hit_at_3": f"{c_summary['hit3']}/{len(cases)}",
            "baseline_mrr": round(summaries["A Baseline"]["mrr"], 4),
            "conservative_mrr": round(b_summary["mrr"], 4),
            "current_mrr": round(c_summary["mrr"], 4),
            "conservative_regression_count": b_vs_a["hit_to_miss"],
            "current_regression_count": c_vs_a["hit_to_miss"],
            "conservative_vs_current_differences": {
                "current_better": len(better),
                "conservative_better": len(conservative_better),
                "same_result": same_count,
                "current_caused_regression": len(caused_regression),
            },
            "de_special_case_count": len(de_cases),
            "de_conservative_hit_at_3": f"{de_b['hit3']}/{len(de_cases)}",
            "de_current_hit_at_3": f"{de_c['hit3']}/{len(de_cases)}",
            "de_current_better": de_c_better,
            "de_conservative_better": de_b_better,
            "de_same_rank": de_same,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("PRODUCTION_DECISION: NOT_MADE")
        return 0
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
        print(f"规则消融失败：{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
