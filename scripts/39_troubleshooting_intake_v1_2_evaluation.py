"""Offline real-model evaluation for Intake V1.1; no Agent or database access."""
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
from app.agent.troubleshooting import TroubleshootingIntake, TroubleshootingIntakeError, build_intake_input
from app.agent.troubleshooting import TROUBLESHOOTING_CLARIFICATION
from app.config import ROUTER_MODEL


class ObservedTransport:
    def __init__(self):
        self.failure = None
        self.calls = 0
    def __call__(self, request, *, timeout):
        self.calls += 1
        try:
            return urlopen(request, timeout=timeout)
        except HTTPError:
            self.failure = "HTTP_ERROR"
            raise
        except TimeoutError:
            self.failure = "TIMEOUT"
            raise
        except URLError as exc:
            self.failure = "TIMEOUT" if isinstance(exc.reason, TimeoutError) else "NETWORK_ERROR"
            raise
        except OSError:
            self.failure = "NETWORK_ERROR"
            raise


def _metrics(rows):
    labels = ("ready", "needs_clarification")
    matrix = {g: {p: 0 for p in (*labels, "ERROR")} for g in labels}
    for row in rows:
        matrix[row["expected_status"]][row.get("predicted_status", "ERROR")] += 1
    per_class = {}
    for label in labels:
        tp = matrix[label][label]
        predicted = sum(matrix[g][label] for g in labels)
        actual = sum(matrix[label].values())
        precision = tp / predicted if predicted else 0.0
        recall = tp / actual if actual else 0.0
        per_class[label] = {"precision": precision, "recall": recall,
                            "f1": 2*precision*recall/(precision+recall) if precision+recall else 0.0}
    return {
        "total": len(rows),
        "correct": sum(r.get("predicted_status") == r["expected_status"] for r in rows),
        "accuracy": sum(r.get("predicted_status") == r["expected_status"] for r in rows)/len(rows),
        "per_class": per_class,
        "false_ready": matrix["needs_clarification"]["ready"],
        "false_clarify": matrix["ready"]["needs_clarification"],
        "contract": {
            "initial_success": sum(r["initial_contract_success"] is True for r in rows),
            "retry_success": sum(r["retry_contract_success"] is True for r in rows),
            "final_success": sum(r["final_contract_success"] is True for r in rows),
            "retry_count": sum(r["retry_count"] for r in rows),
            "deterministic_fast_path_count": sum(r.get("deterministic_fast_path", False) for r in rows),
            "llm_intake_call_count": sum(r.get("llm_intake_call_count", 0) for r in rows),
            "failure_types": {kind: sum(r.get("failure_kind") == kind for r in rows)
                              for kind in ("TIMEOUT", "HTTP_ERROR", "NETWORK_ERROR", "CONTRACT_ERROR")},
        },
        "confusion_matrix": matrix,
    }


def main():
    original_path = ROOT / "data/evaluation/troubleshooting_intake_v1.json"
    hard_path = ROOT / "data/evaluation/troubleshooting_intake_hard_v1.json"
    original_bytes, hard_bytes = original_path.read_bytes(), hard_path.read_bytes()
    original, hard = json.loads(original_bytes), json.loads(hard_bytes)
    transport = ObservedTransport()
    intake = TroubleshootingIntake(transport=transport)
    result = {"model": ROUTER_MODEL, "temperature": 0,
              "source_sha256": {"v1": hashlib.sha256(original_bytes).hexdigest(),
                                "hard": hashlib.sha256(hard_bytes).hexdigest()},
              "frozen_status_before_evaluation": original["status"],
              "single_turn": [], "multi_turn": [], "hard_set": []}
    out = ROOT / "data/evaluation/troubleshooting_intake_v1_2_results.json"

    def call(case_id, question, expected, context=None):
        transport.failure = None
        start_calls = transport.calls
        start = time.perf_counter()
        row = {"case_id": case_id, "question": question, "expected_status": expected,
               "input": build_intake_input(question, context)}
        try:
            assessment = intake.assess_with_metadata(question, conversation_context=context)
            row.update(status="SUCCESS", predicted_status=assessment.decision.status,
                       decision=assessment.decision.model_dump(), guard_rule=assessment.guard_rule,
                       initial_contract_success=assessment.initial_contract_success,
                       retry_contract_success=assessment.retry_contract_success,
                       final_contract_success=assessment.final_contract_success,
                       retry_count=assessment.retry_count,
                       deterministic_fast_path=assessment.deterministic_fast_path)
        except TroubleshootingIntakeError as exc:
            meta = exc.contract_metadata
            row.update(status="ERROR", predicted_status="needs_clarification",
                       failure_kind=transport.failure or "CONTRACT_ERROR",
                       initial_contract_success=meta["initial_contract_success"],
                       retry_contract_success=meta["retry_contract_success"],
                       final_contract_success=meta["final_contract_success"],
                       retry_count=meta["retry_count"], fail_closed=True,
                       deterministic_fast_path=False)
        row["http_calls"] = transport.calls-start_calls
        row["llm_intake_call_count"] = row["http_calls"]
        row["elapsed_seconds"] = time.perf_counter()-start
        return row

    def save():
        result["total_http_calls"] = transport.calls
        result["single_turn_summary"] = _metrics(result["single_turn"])
        result["hard_set_summary"] = _metrics(result["hard_set"]) if result["hard_set"] else None
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    for case in original["cases"]:
        result["single_turn"].append(call(case["case_id"], case["question"], case["expected_status"]))
        save()
        print("V1", case["case_id"], result["single_turn"][-1]["status"], flush=True)
    for case in original["multi_turn_cases"]:
        first = call(case["case_id"]+"-T1", case["turn1"], case["expected_turn1"])
        context = [{"role":"user", "content":case["turn1"]},
                   {"role":"assistant", "content":TROUBLESHOOTING_CLARIFICATION}]
        second = call(case["case_id"]+"-T2", case["turn2"], case["expected_turn2"], context)
        result["multi_turn"].append({"case_id":case["case_id"], "turn1":first, "turn2":second,
            "passed":first["predicted_status"]==case["expected_turn1"] and second["predicted_status"]==case["expected_turn2"]})
        save()
        print("MULTI", case["case_id"], "PASS" if result["multi_turn"][-1]["passed"] else "FAIL", flush=True)
    for case in hard["cases"]:
        result["hard_set"].append(call(case["case_id"], case["question"], case["expected_status"]))
        save()
        print("HARD", case["case_id"], result["hard_set"][-1]["status"], flush=True)
    result["total_http_calls"] = transport.calls
    result["multi_turn_passed"] = sum(row["passed"] for row in result["multi_turn"])
    result["session_isolation_context"] = build_intake_input("机器人无法正常工作怎么办？", [])
    if original_path.read_bytes() != original_bytes or hard_path.read_bytes() != hard_bytes:
        raise RuntimeError("Frozen evaluation labels changed during evaluation.")
    save()
    print(json.dumps({"single_turn": result["single_turn_summary"],
                      "multi_turn_passed":result["multi_turn_passed"],
                      "hard_set":result["hard_set_summary"],
                      "total_http_calls":transport.calls}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__": main()
