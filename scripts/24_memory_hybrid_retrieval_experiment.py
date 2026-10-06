"""Frozen Memory V2.4E experiment; reuse real dense scores, no network/DB writes."""

import hashlib
import importlib.util
import json
import math
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "memory_expansion", ROOT / "scripts/24_memory_retrieval_expansion.py")
expansion = importlib.util.module_from_spec(spec)
spec.loader.exec_module(expansion)
SOURCE = ROOT / "data/evaluation/long_term_memory_retrieval_v2_results.json"
OUTPUT = ROOT / "data/evaluation/long_term_memory_hybrid_results_v1.json"

# All grids are fixed before scoring. Every configuration applies globally.
LEXICAL_THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
ALPHAS = (0.50, 0.70, 0.85)
HYBRID_THRESHOLDS = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                     0.50, 0.55, 0.60, 0.65, 0.70, 0.75)
DUAL_DENSE_THRESHOLDS = (0.40, 0.50, 0.55, 0.60)
DUAL_LEXICAL_THRESHOLDS = (0.05, 0.10, 0.15, 0.20)
MAX_FALSE_INJECTION_CASES = 1


def bigrams(text):
    """Same NFKC/casefold/alphanumeric preprocessing for query and fact."""
    compact = "".join(char for char in unicodedata.normalize("NFKC", text).casefold()
                      if char.isalnum())
    if len(compact) < 2:
        return set(compact)
    return {compact[i:i + 2] for i in range(len(compact) - 1)}


def lexical_similarity(query, fact):
    left, right = bigrams(query), bigrams(fact)
    total = len(left) + len(right)
    return 2 * len(left & right) / total if total else 0.0


def rank(scores):
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))


def load_source():
    dataset, dataset_hash, _ = expansion.load_dataset()
    raw = SOURCE.read_bytes()
    source = json.loads(raw)
    if (source["dataset_sha256"] != dataset_hash
            or source["model"] != expansion.baseline.EMBEDDING_MODEL
            or source["query_instruction"] != expansion.baseline.INSTRUCTION
            or source["instructed_query_format"] != "{instruction}\n用户当前问题：{query}"
            or source["top_k"] != 3 or source["fact_bank"] != dataset["fact_bank"]):
        raise ValueError("stored dense source does not match frozen dataset/model/instruction")
    by_id = {case["case_id"]: case for case in source["cases"]}
    if len(by_id) != len(dataset["cases"]):
        raise ValueError("stored dense case count mismatch")
    dense = {}
    for case in dataset["cases"]:
        stored = by_id[case["case_id"]]
        for key in ("facts", "query", "relevant_fact_indices", "answerability", "category", "note"):
            if stored[key] != case[key]:
                raise ValueError(f"frozen case changed: {case['case_id']} {key}")
        rows = stored["instructed"]["ranking"]
        values = {row["fact_index"]: row["similarity"] for row in rows}
        if (len(values) != len(rows) or set(values) != set(case["facts"])
                or any(not math.isfinite(value) or not -1 <= value <= 1
                       for value in values.values())):
            raise ValueError(f"invalid dense scores: {case['case_id']}")
        dense[case["case_id"]] = values
    return dataset, dataset_hash, hashlib.sha256(raw).hexdigest(), dense


def evaluate_config(cases, rankings, threshold, config_id, method, **parameters):
    summary = expansion.evaluate(cases, rankings, threshold)
    summary.update(config_id=config_id, method=method, parameters=parameters)
    if method == "dual_gate":
        summary["threshold"] = None
    details = {}
    for case in cases:
        rows = [(i, score) for i, score in rankings[case["case_id"]] if score >= threshold]
        details[case["case_id"]] = {
            "admitted_fact_indices": [i for i, _ in rows],
            "injected_fact_indices": [i for i, _ in rows[:3]],
        }
    return {"summary": summary, "case_admissions": details}


def best(configs, constrained=True):
    eligible = [config for config in configs
                if not constrained or config["summary"]["false_injection_case_count"] <= 1]
    if not eligible:
        return None
    # Predeclared comparison order from the task; an empty injection has precision 0 here.
    def key(config):
        row = config["summary"]
        return (row["hit_at_3_count"], row["recall_at_3"],
                row["injected_fact_precision"] or 0,
                min(value["rate"] for value in row["category_hit_at_3"].values()))
    return max(eligible, key=key)


def main():
    dataset, dataset_hash, source_hash, dense = load_source()
    facts, cases = dataset["fact_bank"], dataset["cases"]
    lexical = {case["case_id"]: {i: lexical_similarity(case["query"], facts[i]["text"])
                               for i in case["facts"]} for case in cases}
    dense_rankings = {cid: rank(values) for cid, values in dense.items()}
    lexical_rankings = {cid: rank(values) for cid, values in lexical.items()}
    weighted = {alpha: {cid: {i: alpha * score + (1 - alpha) * lexical[cid][i]
                             for i, score in values.items()}
                        for cid, values in dense.items()} for alpha in ALPHAS}
    baseline = evaluate_config(cases, dense_rankings, 0.65, "dense@0.65", "dense")
    base = baseline["summary"]
    if (base["hit_at_3_count"] != 14 or base["false_injection_case_count"] != 1
            or not math.isclose(base["injected_fact_precision"], 0.7)):
        raise ValueError("constrained dense baseline is not the confirmed 14/30, 1/18, 70%")

    configs = []
    for threshold in LEXICAL_THRESHOLDS:
        configs.append(evaluate_config(cases, lexical_rankings, threshold,
                                       f"lexical@{threshold:.2f}", "lexical"))
    for alpha in ALPHAS:
        rankings = {cid: rank(values) for cid, values in weighted[alpha].items()}
        for threshold in HYBRID_THRESHOLDS:
            configs.append(evaluate_config(cases, rankings, threshold,
                f"weighted_a{alpha:.2f}@{threshold:.2f}", "weighted", alpha=alpha))
    for d in DUAL_DENSE_THRESHOLDS:
        for l in DUAL_LEXICAL_THRESHOLDS:
            # Absolute AND admission followed by dense ordering; no rank-only gate.
            rankings = {cid: [(i, score) for i, score in rows
                              if score >= d and lexical[cid][i] >= l]
                        for cid, rows in dense_rankings.items()}
            configs.append(evaluate_config(cases, rankings, -1,
                f"dual_d{d:.2f}_l{l:.2f}", "dual_gate", dense_threshold=d, lexical_threshold=l))

    selected = {
        "lexical_unconstrained": best([c for c in configs if c["summary"]["method"] == "lexical"], False),
        "lexical_constrained": best([c for c in configs if c["summary"]["method"] == "lexical"]),
        **{f"weighted_{alpha:.2f}": best([c for c in configs
           if c["summary"]["method"] == "weighted" and c["summary"]["parameters"]["alpha"] == alpha])
           for alpha in ALPHAS},
        "dual_constrained": best([c for c in configs if c["summary"]["method"] == "dual_gate"]),
        "best_constrained_hybrid": best([c for c in configs if c["summary"]["method"] in ("weighted", "dual_gate")]),
    }
    pairs = []
    for case in cases:
        cid = case["case_id"]
        dr = {i: n for n, (i, _) in enumerate(dense_rankings[cid], 1)}
        lr = {i: n for n, (i, _) in enumerate(lexical_rankings[cid], 1)}
        wr = {alpha: {i: n for n, (i, _) in enumerate(rank(weighted[alpha][cid]), 1)} for alpha in ALPHAS}
        pairs.append({**case, "fact_scores": [{
            "fact_index": i, "fact_text": facts[i]["text"], "relevant": i in case["relevant_fact_indices"],
            "dense_similarity": dense[cid][i], "dense_rank": dr[i],
            "lexical_similarity": lexical[cid][i], "lexical_rank": lr[i],
            "weighted": {f"{a:.2f}": {"score": weighted[a][cid][i], "rank": wr[a][i]} for a in ALPHAS},
        } for i in case["facts"]]})
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": dataset_hash, "dense_source_sha256": source_hash,
        "dense_source": str(SOURCE.relative_to(ROOT)), "model": expansion.baseline.EMBEDDING_MODEL,
        "instruction": expansion.baseline.INSTRUCTION,
        "dense_reused_without_api_call": True, "top_k": 3,
        "lexical_scorer": "set character bigram Dice; same NFKC/casefold/alphanumeric-only preprocessing for both texts; one-character fallback to unigram; no semantic rewriting",
        "fixed_grids": {"lexical_thresholds": list(LEXICAL_THRESHOLDS), "alphas": list(ALPHAS),
                        "weighted_thresholds": list(HYBRID_THRESHOLDS),
                        "dual_dense_thresholds": list(DUAL_DENSE_THRESHOLDS),
                        "dual_lexical_thresholds": list(DUAL_LEXICAL_THRESHOLDS)},
        "constraint": {"max_false_injection_cases": 1, "no_relevant_cases": 18,
                       "selection_order": ["Hit@3", "Recall@3", "precision", "minimum_category_hit_rate"]},
        "baseline": baseline, "configurations": configs,
        "best_configurations": {name: config["summary"] if config else None for name, config in selected.items()},
        "cases": pairs,
    }
    if (hashlib.sha256(expansion.DATASET.read_bytes()).hexdigest() != dataset_hash
            or hashlib.sha256(SOURCE.read_bytes()).hexdigest() != source_hash):
        raise ValueError("frozen input changed while evaluating; output not saved")
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Dense source reused; 48 frozen cases; fixed grids: lexical=10, weighted=36, dual=16")
    print("BASELINE=" + json.dumps(base, ensure_ascii=False))
    print("LEXICAL_ONLY_SWEEP:")
    for config in configs[:10]:
        print(json.dumps(config["summary"], ensure_ascii=False))
    print("BEST_CONFIGURATIONS=" + json.dumps(report["best_configurations"], ensure_ascii=False))
    for cid in ("M01", "N24", "N19", "M15", "M16", "M14"):
        pair = next(row for row in pairs if row["case_id"] == cid)
        print("DIAGNOSTIC=" + json.dumps(pair, ensure_ascii=False))
        for name, config in selected.items():
            if config is not None:
                print(name + ": " + json.dumps(config["case_admissions"][cid]))
    print(f"RESULT_FILE={OUTPUT}")


if __name__ == "__main__":
    main()
