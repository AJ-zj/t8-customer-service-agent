"""Offline 48-case memory retrieval evaluation; no Agent, chat or database calls."""

import hashlib
import importlib.util
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Reuse the exact V2.4A instruction, thresholds, TopK, API and cosine ranking.
BASELINE_PATH = ROOT / "scripts/22_memory_retrieval_calibration.py"
spec = importlib.util.spec_from_file_location("memory_calibration_v1", BASELINE_PATH)
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)
DATASET = ROOT / "data/evaluation/long_term_memory_retrieval_v2.json"
V1_DATASET = ROOT / "data/evaluation/long_term_memory_retrieval_v1.json"
OUTPUT = ROOT / "data/evaluation/long_term_memory_retrieval_v2_results.json"
CATEGORIES = ("device_profile", "home_environment", "usage_preference")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load_dataset():
    raw = DATASET.read_bytes()
    dataset = json.loads(raw)
    v1_raw = V1_DATASET.read_bytes()
    v1 = json.loads(v1_raw)
    facts, cases = dataset["fact_bank"], dataset["cases"]
    if facts[:len(v1["fact_bank"])] != v1["fact_bank"]:
        raise ValueError("v1 fact bank changed in v2")
    by_id = {case["case_id"]: case for case in cases}
    if len(cases) != 48 or len(by_id) != 48:
        raise ValueError("expected 48 unique cases")
    for fact in facts:
        if fact["category"] not in CATEGORIES or not fact["text"].strip():
            raise ValueError("invalid fact bank")
    for case in cases:
        candidates, relevant = case["facts"], case["relevant_fact_indices"]
        if (not candidates or len(candidates) != len(set(candidates))
                or any(type(i) is not int or not 0 <= i < len(facts) for i in candidates)
                or len(relevant) != len(set(relevant))
                or any(i not in candidates for i in relevant)
                or case["answerability"] not in ("positive", "no_relevant")
                or bool(relevant) != (case["answerability"] == "positive")
                or not case["query"].strip() or not case["note"].strip()):
            raise ValueError(f"invalid annotation: {case['case_id']}")
        if relevant and any(facts[i]["category"] != case["category"] for i in relevant):
            raise ValueError(f"category mismatch: {case['case_id']}")
    counts = Counter(case["answerability"] for case in cases)
    if counts != {"positive": 30, "no_relevant": 18}:
        raise ValueError("expected 30 positive / 18 no-relevant")
    for old in v1["cases"]:
        current = by_id[old["id"]]
        for key in ("facts", "query", "relevant_fact_indices"):
            if old[key] != current[key]:
                raise ValueError(f"v1 label changed: {old['id']} {key}")
        if old["no_relevant"] != (current["answerability"] == "no_relevant"):
            raise ValueError(f"v1 label changed: {old['id']}")
    return dataset, digest(raw), digest(v1_raw)


def selection(ranking, threshold):
    return [(i, score) for i, score in ranking if score >= threshold][:baseline.TOP_K]


def evaluate(cases, rankings, threshold):
    hit1 = hit3 = recall = mrr = false_cases = injected = relevant_injected = 0
    category_hits = Counter()
    positive_counts = Counter(case["category"] for case in cases
                              if case["answerability"] == "positive")
    misses, false_ids = [], []
    for case in cases:
        chosen = selection(rankings[case["case_id"]], threshold)
        injected += len(chosen)
        relevant = set(case["relevant_fact_indices"])
        ranks = [rank for rank, (i, _) in enumerate(chosen, 1) if i in relevant]
        relevant_injected += len(ranks)
        if case["answerability"] == "no_relevant":
            false_cases += bool(chosen)
            if chosen:
                false_ids.append(case["case_id"])
            continue
        hit1 += bool(ranks and ranks[0] == 1)
        hit3 += bool(ranks)
        recall += len(ranks) / len(relevant)
        mrr += 1 / ranks[0] if ranks else 0
        category_hits[case["category"]] += bool(ranks)
        if not ranks:
            misses.append(case["case_id"])
    return {
        "threshold": threshold, "hit_at_1_count": hit1, "hit_at_3_count": hit3,
        "positive_cases": 30, "recall_at_3": recall / 30, "mrr": mrr / 30,
        "false_injection_case_count": false_cases, "no_relevant_cases": 18,
        "false_injection_rate": false_cases / 18,
        "avg_injected_facts": injected / len(cases), "injected_facts": injected,
        "injected_relevant_facts": relevant_injected,
        "injected_fact_precision": relevant_injected / injected if injected else None,
        "category_hit_at_3": {
            category: {"hit_cases": category_hits[category], "total": positive_counts[category],
                       "rate": category_hits[category] / positive_counts[category]}
            for category in CATEGORIES},
        "positive_misses": misses, "false_injection_cases": false_ids,
    }


def case_result(case, ranking, facts):
    relevant = set(case["relevant_fact_indices"])
    return {
        "ranking": [{"fact_index": i, "text": facts[i]["text"],
                     "category": facts[i]["category"], "similarity": score,
                     "rank": rank, "relevant": i in relevant}
                    for rank, (i, score) in enumerate(ranking, 1)],
        "thresholds": {
            f"{threshold:.2f}": {
                "injected_fact_indices": [i for i, _ in selection(ranking, threshold)],
                "hit_at_3": any(i in relevant for i, _ in selection(ranking, threshold))
                             if relevant else None,
            } for threshold in baseline.THRESHOLDS},
    }


def main():
    dataset, dataset_hash, v1_hash = load_dataset()
    facts, cases = dataset["fact_bank"], dataset["cases"]
    distribution = dict(Counter(case["category"] for case in cases
                                if case["answerability"] == "positive"))
    print(f"FIXED_DATASET: 48 cases, 30 positive, 18 no-relevant; {distribution}", flush=True)
    print(f"DATASET_SHA256={dataset_hash}", flush=True)
    key = baseline.get_siliconflow_api_key()
    fact_vectors = baseline.request_embeddings([fact["text"] for fact in facts], key)
    print(f"FACT_EMBEDDINGS_READY={len(fact_vectors)}", flush=True)
    queries = [case["query"] for case in cases]
    raw_vectors = baseline.request_embeddings(queries, key)
    print("RAW_QUERY_EMBEDDINGS_READY=48", flush=True)
    instructed_vectors = baseline.request_embeddings(
        [f"{baseline.INSTRUCTION}\n用户当前问题：{query}" for query in queries], key)
    print("INSTRUCTED_QUERY_EMBEDDINGS_READY=48", flush=True)
    if digest(DATASET.read_bytes()) != dataset_hash or digest(V1_DATASET.read_bytes()) != v1_hash:
        raise ValueError("dataset changed during experiment; results not saved")
    rankings = {
        mode: {case["case_id"]: baseline.rank_case(case, fact_vectors, vectors[position])
               for position, case in enumerate(cases)}
        for mode, vectors in (("raw", raw_vectors), ("instructed", instructed_vectors))}
    summaries = {mode: [evaluate(cases, rows, t) for t in baseline.THRESHOLDS]
                 for mode, rows in rankings.items()}
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": dataset_hash, "v1_dataset_sha256": v1_hash,
        "v1_script_sha256": digest(BASELINE_PATH.read_bytes()),
        "model": baseline.EMBEDDING_MODEL, "query_instruction": baseline.INSTRUCTION,
        "instructed_query_format": "{instruction}\n用户当前问题：{query}",
        "top_k": baseline.TOP_K, "thresholds": list(baseline.THRESHOLDS),
        "positive_distribution": distribution, "fact_bank": facts,
        "metric_definitions": {
            "hit_at_1": "positive cases whose first injected fact is relevant",
            "hit_at_3": "positive cases with at least one relevant fact injected",
            "recall_at_3": "mean fraction of relevant facts injected across positive cases",
            "mrr": "mean reciprocal first relevant rank among thresholded Top3 injections; 0 if absent; same as v1",
            "injected_fact_precision": "relevant injections / all injections across all 48 cases; null if none",
            "false_injection_rate": "no-relevant cases with any injection / 18",
        },
        "threshold_summary": summaries,
        "cases": [{**case, **{mode: case_result(case, rankings[mode][case["case_id"]], facts)
                              for mode in rankings}} for case in cases],
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for mode, rows in summaries.items():
        print(f"\nMODE={mode}")
        print("threshold | Hit1 | Hit3 | Recall3 | MRR | false/18 | false rate | precision | avg injected | device/home/preference Hit3")
        for row in rows:
            category = "/".join(f"{row['category_hit_at_3'][c]['hit_cases']}:{row['category_hit_at_3'][c]['total']}"
                                for c in CATEGORIES)
            precision = row["injected_fact_precision"]
            print(f"{row['threshold']:.2f} | {row['hit_at_1_count']}/30 | {row['hit_at_3_count']}/30 | "
                  f"{row['recall_at_3']:.4f} | {row['mrr']:.4f} | {row['false_injection_case_count']}/18 | "
                  f"{row['false_injection_rate']:.4f} | {precision:.4f} | {row['avg_injected_facts']:.4f} | {category}"
                  if precision is not None else f"{row['threshold']:.2f}: no injected facts; precision N/A")
        bad = next(case for case in report["cases"] if case["case_id"] == "M01")[mode]
        print("BAD_CASE=" + json.dumps(bad, ensure_ascii=False))
    print(f"RESULT_FILE={OUTPUT}")


if __name__ == "__main__":
    main()
