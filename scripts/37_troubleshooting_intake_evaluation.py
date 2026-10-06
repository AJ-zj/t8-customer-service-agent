"""Frozen Intake evaluation only: no Agent, manual retrieval or database writes."""

import hashlib
import json
import math
from pathlib import Path
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.agent.troubleshooting import (
    TroubleshootingIntake, TroubleshootingIntakeError, TroubleshootingIntakeDecision,
    TROUBLESHOOTING_CLARIFICATION, build_intake_input,
)
from app.config import ROUTER_MODEL


class ObservedTransport:
    def __init__(self): self.failure = None

    def __call__(self, request, *, timeout):
        try:
            response = urlopen(request, timeout=timeout)
        except HTTPError:
            self.failure = "HTTP_ERROR"
            raise
        except TimeoutError:
            self.failure = "TIMEOUT"
            raise
        except URLError as error:
            self.failure = "TIMEOUT" if isinstance(error.reason, TimeoutError) else "NETWORK_ERROR"
            raise
        return response


def summarize(rows):
    labels = ["ready", "needs_clarification"]
    matrix = {label: {prediction: 0 for prediction in labels + ["ERROR"]} for label in labels}
    for row in rows: matrix[row["expected_status"]][row.get("predicted_status", "ERROR")] += 1
    metrics = {}
    for label in labels:
        tp = matrix[label][label]
        predicted = sum(matrix[gold][label] for gold in labels)
        actual = sum(matrix[label].values())
        precision, recall = tp / predicted if predicted else 0, tp / actual if actual else 0
        metrics[label] = {"precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0}
    latencies = sorted(row["elapsed_seconds"] for row in rows)
    return {"total": len(rows), "correct": sum(r.get("predicted_status") == r["expected_status"] for r in rows),
        "accuracy": sum(r.get("predicted_status") == r["expected_status"] for r in rows) / len(rows),
        "contract_success": sum(r["status"] == "SUCCESS" for r in rows), "per_class": metrics,
        "false_ready": matrix["needs_clarification"]["ready"],
        "false_clarify": matrix["ready"]["needs_clarification"],
        "fail_closed_ready_errors": matrix["ready"]["ERROR"],
        "confusion_matrix": matrix, "avg_latency": sum(latencies) / len(rows),
        "p95_latency": latencies[math.ceil(.95 * len(rows)) - 1]}


def main():
    path = ROOT / "data/evaluation/troubleshooting_intake_v1.json"
    source = path.read_bytes()
    dataset = json.loads(source)
    transport = ObservedTransport()
    intake = TroubleshootingIntake(transport=transport)
    result = {"model": ROUTER_MODEL, "temperature": 0,
              "dataset_sha256": hashlib.sha256(source).hexdigest(), "attempts": [], "multi_turn_results": []}
    destination = ROOT / "data/evaluation/troubleshooting_intake_v1_results.json"

    def save():
        destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    def call(case_id, question, expected, context=None):
        row = {"case_id": case_id, "question": question, "expected_status": expected,
               "input": build_intake_input(question, context)}
        transport.failure = None
        start = time.perf_counter()
        try:
            decision = intake.assess(question, conversation_context=context)
            row.update(status="SUCCESS", predicted_status=decision.status, decision=decision.model_dump())
        except TroubleshootingIntakeError:
            row.update(status="ERROR", failure_kind=transport.failure or "RESPONSE_OR_SCHEMA_ERROR",
                       effective_status="needs_clarification")
        row["elapsed_seconds"] = time.perf_counter() - start
        print(case_id, row["status"], row.get("predicted_status", "ERROR"),
              round(row["elapsed_seconds"], 3), flush=True)
        return row

    for case in dataset["cases"]:
        result["attempts"].append(call(case["case_id"], case["question"], case["expected_status"]))
        result["summary"] = summarize(result["attempts"])
        save()
    for case in dataset["multi_turn_cases"]:
        first = call(case["case_id"] + "-T1", case["turn1"], case["expected_turn1"])
        context = [{"role": "user", "content": case["turn1"]},
                   {"role": "assistant", "content": TROUBLESHOOTING_CLARIFICATION}]
        second = call(case["case_id"] + "-T2", case["turn2"], case["expected_turn2"], context)
        result["multi_turn_results"].append({"case_id": case["case_id"], "turn1": first, "turn2": second,
            "passed": first.get("predicted_status") == case["expected_turn1"]
                      and second.get("predicted_status") == case["expected_turn2"]})
        save()
    result["total_model_calls"] = len(result["attempts"]) + 2 * len(result["multi_turn_results"])
    result["session_isolation_input_check"] = build_intake_input("机器人无法正常工作怎么办？", [])
    if path.read_bytes() != source: raise RuntimeError("Frozen labels changed.")
    save()
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__": main()
