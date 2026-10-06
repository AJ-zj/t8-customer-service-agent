"""One real classification per frozen case; no Agent, retrieval or database writes."""

import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.agent.llm_router import LLMIntentRouter, LLMRouterError
from app.agent.state import SceneIntent
from app.config import ROUTER_MODEL
from typing import get_args


def summarize(rows):
    labels = list(get_args(SceneIntent))
    matrix = {label: {prediction: 0 for prediction in labels + ["ERROR"]}
              for label in labels}
    for row in rows:
        matrix[row["expected_intent"]][row.get("intent", "ERROR")] += 1
    per_class = {}
    for label in labels:
        tp = matrix[label][label]
        predicted = sum(matrix[actual][label] for actual in labels)
        actual = sum(matrix[label].values())
        precision = tp / predicted if predicted else 0
        recall = tp / actual if actual else 0
        per_class[label] = {
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0,
        }
    latencies = sorted(row["elapsed_seconds"] for row in rows)
    count = len(rows)
    return {
        "total": count,
        "correct": sum(row.get("intent") == row["expected_intent"] for row in rows),
        "accuracy": sum(row.get("intent") == row["expected_intent"] for row in rows) / count,
        "contract_success": sum(row["status"] == "SUCCESS" for row in rows),
        "macro": {key: sum(value[key] for value in per_class.values()) / len(labels)
                  for key in ("precision", "recall", "f1")},
        "per_class": per_class, "confusion_matrix": matrix,
        "avg_latency": sum(latencies) / count,
        "p95_latency": latencies[math.ceil(.95 * count) - 1],
    }


def main():
    path = ROOT / "data/evaluation/scene_router_v1.json"
    source = path.read_bytes()
    rows = []
    router = LLMIntentRouter()
    destination = ROOT / "data/evaluation/scene_router_v1_results.json"
    for case in json.loads(source)["cases"]:
        start = time.perf_counter()
        row = dict(case)
        try:
            # Direct model classification: even explicit handoff cases count as real calls.
            decision = router.classify(case["question"])
            row.update(status="SUCCESS", intent=decision.intent, reason=decision.reason)
        except LLMRouterError:
            row.update(status="ERROR")
        row["elapsed_seconds"] = time.perf_counter() - start
        rows.append(row)
        destination.write_text(json.dumps({
            "model": ROUTER_MODEL, "dataset_sha256": hashlib.sha256(source).hexdigest(),
            "attempts": rows, "summary": summarize(rows),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(case["case_id"], row["status"], row.get("intent", "ERROR"),
              round(row["elapsed_seconds"], 3), flush=True)
    if path.read_bytes() != source:
        raise RuntimeError("Frozen evaluation dataset changed during evaluation.")
    print(json.dumps(summarize(rows), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
