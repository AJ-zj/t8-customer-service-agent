"""Read-only raw-vs-instructed Qwen3 Embedding threshold sweep."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import PROJECT_ROOT, get_siliconflow_api_key
from app.memory.facts import cosine_similarity
from app.retrieval.t8_retrieval import EMBEDDING_MODEL, request_embeddings


DATASET = PROJECT_ROOT / "data" / "evaluation" / "long_term_memory_retrieval_v1.json"
THRESHOLDS = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75)
INSTRUCTION = "检索与当前用户问题有关的用户长期设备信息、家庭环境和使用偏好。"
TOP_K = 3


def load_dataset():
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    facts = dataset["fact_bank"]
    cases = dataset["cases"]
    if not 15 <= len(cases) <= 20 or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("评测题数量或 ID 无效。")
    for case in cases:
        candidates = case["facts"]
        relevant = case["relevant_fact_indices"]
        if (not candidates or len(candidates) != len(set(candidates))
                or any(type(index) is not int or not 0 <= index < len(facts) for index in candidates)
                or any(index not in candidates for index in relevant)
                or case["no_relevant"] != (len(relevant) == 0)):
            raise ValueError(f"评测标注无效：{case['id']}")
    bad = next(case for case in cases if case["id"] == "M01")
    if bad["facts"] != [0, 1] or bad["relevant_fact_indices"] != [0]:
        raise ValueError("真实地毯 Bad Case 标注发生变化。")
    return facts, cases


def rank_case(case, fact_vectors, query_vector):
    return sorted(
        ((index, cosine_similarity(query_vector, fact_vectors[index]))
         for index in case["facts"]),
        key=lambda row: (-row[1], row[0]),
    )


def evaluate(cases, rankings, threshold):
    positives = [case for case in cases if not case["no_relevant"]]
    negatives = [case for case in cases if case["no_relevant"]]
    hit1 = hit3 = recall = mrr = false_injection = injected = 0
    for case in cases:
        selected = [(index, score) for index, score in rankings[case["id"]]
                    if score >= threshold][:TOP_K]
        injected += len(selected)
        if case["no_relevant"]:
            false_injection += bool(selected)
            continue
        relevant = set(case["relevant_fact_indices"])
        ranks = [rank for rank, (index, _) in enumerate(selected, 1)
                 if index in relevant]
        hit1 += bool(ranks and ranks[0] == 1)
        hit3 += bool(ranks)
        recall += len(ranks) / len(relevant)
        mrr += 1 / ranks[0] if ranks else 0
    total = len(cases)
    return {
        "threshold": threshold,
        "hit_at_1": f"{hit1}/{len(positives)}",
        "hit_at_3": f"{hit3}/{len(positives)}",
        "recall_at_3": round(recall / len(positives), 4),
        "mrr": round(mrr / len(positives), 4),
        "false_injection": f"{false_injection}/{len(negatives)}",
        "false_injection_rate": round(false_injection / len(negatives), 4),
        "avg_injected_facts": round(injected / total, 4),
    }


def main():
    facts, cases = load_dataset()
    api_key = get_siliconflow_api_key()
    fact_vectors = request_embeddings([fact["text"] for fact in facts], api_key)
    raw_queries = [case["query"] for case in cases]
    instructed_queries = [f"{INSTRUCTION}\n用户当前问题：{query}"
                          for query in raw_queries]
    # Each fact is embedded once; each query is embedded once per variant.
    query_vectors = {
        "raw": request_embeddings(raw_queries, api_key),
        "instructed": request_embeddings(instructed_queries, api_key),
    }
    rankings = {
        mode: {case["id"]: rank_case(case, fact_vectors, vectors[position])
               for position, case in enumerate(cases)}
        for mode, vectors in query_vectors.items()
    }
    print(f"MODEL={EMBEDDING_MODEL} CASES={len(cases)} "
          f"POSITIVE={sum(not case['no_relevant'] for case in cases)} "
          f"NO_RELEVANT={sum(case['no_relevant'] for case in cases)}")
    print(f"QUERY_INSTRUCTION={INSTRUCTION}")
    for mode in ("raw", "instructed"):
        print(f"\nMODE={mode}")
        print("threshold | Hit@1 | Hit@3 | Recall@3 | MRR | false injection | rate | avg injected")
        for threshold in THRESHOLDS:
            item = evaluate(cases, rankings[mode], threshold)
            print(" | ".join(str(item[key]) for key in (
                "threshold", "hit_at_1", "hit_at_3", "recall_at_3", "mrr",
                "false_injection", "false_injection_rate", "avg_injected_facts")))
        print("BAD_CASE_SIMILARITY:")
        print(" ".join(f"{facts[index]['text']}={score:.6f}"
                       for index, score in rankings[mode]["M01"]))
        print("BAD_CASE_THRESHOLD_SWEEP:")
        for threshold in THRESHOLDS:
            selected = [facts[index]["text"] for index, score in rankings[mode]["M01"]
                        if score >= threshold][:TOP_K]
            print(f"{threshold:.2f}: {selected}")


if __name__ == "__main__":
    main()
