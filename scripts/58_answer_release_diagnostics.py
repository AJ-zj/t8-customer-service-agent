"""Offline diagnosis for Answer Release Policy V1.1.

This script reads frozen Golden and cached generation/release artifacts. It may
optionally replay the semantic gate on cached A/B drafts; it never writes to the
database and never calls the answer-generation model.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.answer_release_policy import AUTO_RELEASE_SCENES, PROGRAM_BLOCKED_STATUSES
from app.services.grounded_qa import ABSTENTION_PATTERN, audit_answer
from app.services.qa_service import normalize_retrieval_query
from app.services.semantic_grounding_gate import GATE_SYSTEM_PROMPT, SemanticGroundingGate

GOLDEN_PATH = ROOT / "data/evaluation/t8_rag_golden_v1.json"
GENERATION_PATH = ROOT / "data/evaluation/results/generation_eval_v1.json"
RELEASE_PATH = ROOT / "data/evaluation/results/answer_release_policy_v1.json"
OUTPUT_PATH = ROOT / "data/evaluation/results/answer_release_diagnostics_v1.json"
EXPECTED_GOLDEN_SHA256 = "547fd0f48197bc150640100fb07ba6287fee5d0f4439c1e96c2950c2167e9386"
E2E_SNAPSHOT_PATH = OUTPUT_PATH.parent / "answer_release_e2e_ab_snapshot.tmp.jsonl"

# Local diagnostic annotations only; these never participate in production
# audit/release decisions. Each must be justified by the saved draft/evidence.
SAFE_ABSTENTION_REVIEW: dict[str, dict] = {
    f"T8-G{number:03d}": {
        "category": "C_RETRIEVAL_HIT_AUDIT_MIXED_REFUSAL",
        "basis": "Manual inspection of the saved draft, reference and actual Evidence: the draft gives a cited answer to the question and separately limits an unconfirmed detail. Any abstention substring caused the original audit to classify the entire answer as refusal. This annotation does not certify all claims; mixed answers must still pass the semantic gate.",
    }
    for number in (2, 10, 12, 13, 18, 23, 26, 28, 30, 31, 36, 37, 39, 40, 41, 43, 47, 48, 49, 50)
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def evidence_records(evidence: list[dict]) -> list[tuple]:
    return [(item["evidence_id"], item["record_index"], item) for item in evidence]


def review_cause(case: dict) -> str:
    if case["audit_status"] in PROGRAM_BLOCKED_STATUSES:
        return "PROGRAM_BLOCKED"
    if case["intent"] not in AUTO_RELEASE_SCENES:
        return "SCENE_INELIGIBLE"
    if case["gate_status"] == "unsupported":
        return "SEMANTIC_GATE_UNSUPPORTED"
    if case["gate_status"] == "failed":
        return "SEMANTIC_GATE_FAILURE"
    return "OTHER"


def abstention_cause(case: dict) -> dict:
    if not case["gold_record_in_top3"]:
        return {"category": "A_RETRIEVAL_MISS", "basis": "No gold relevant record entered cached Top3."}
    reviewed = SAFE_ABSTENTION_REVIEW.get(case["id"])
    return reviewed or {
        "category": "PENDING_MANUAL_B_OR_C",
        "basis": "Gold record is present; inspect the full draft and evidence before assigning true refusal or audit misclassification.",
    }


def write_result(result: dict) -> None:
    temporary = OUTPUT_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(OUTPUT_PATH)


def replay_e2e_gate(result: dict, *, phase: str) -> None:
    if phase == "before_fix" and any(
        item.get("gate_system_prompt") != GATE_SYSTEM_PROMPT
        for item in result.get("e2e_ab_diagnostics", [])
    ):
        raise RuntimeError("Historical before-fix Gate verdicts are frozen; use after_fix for the current Prompt")
    if E2E_SNAPSHOT_PATH.exists():
        lines = E2E_SNAPSHOT_PATH.read_text(encoding="utf-8-sig").splitlines()
        snapshots = [json.loads(line) for line in lines if line.startswith("{")]
    else:
        # The diagnostic artifact retains the original input after the temporary
        # read-only DB export is removed. No DB access is needed for a replay.
        snapshots = [
            {**item["gate_input"], "audit_status": item["audit_status"]}
            for item in result.get("e2e_ab_diagnostics", [])
        ]
    expected = {"主刷多久清理一次？": [13], "主刷怎么清理？": [14]}
    if {row["question"] for row in snapshots} != set(expected):
        raise RuntimeError("Expected exact A/B snapshots from isolated V1 E2E")
    for row in snapshots:
        question = row["question"]
        evidence = row["evidence"]
        gate_input = {"question": question, "draft_answer": row["draft_answer"], "evidence": evidence}
        started = time.perf_counter()
        verdict = SemanticGroundingGate().evaluate(**gate_input)
        detail = {
            "case_id": "A" if question == "主刷多久清理一次？" else "B",
            "source": "Read-only snapshot of isolated V1 E2E qa_review_records; verdict is a diagnostic replay, not the unpersisted original Gate output.",
            "question": question,
            "intent": "maintenance",
            "retrieval_query": normalize_retrieval_query(question),
            "gold_relevant_record_indices": expected[question],
            "actual_evidence": evidence,
            "draft_answer": row["draft_answer"],
            "audit_status": row["audit_status"],
            "gate_input": gate_input,
            "gate_system_prompt": GATE_SYSTEM_PROMPT,
            "gate_structured_output": {
                "supported": verdict.status == "supported" if verdict.status != "failed" else None,
                "status": verdict.status,
                "reason_code": verdict.reason_code,
                "unsupported_claims": list(verdict.unsupported_claims),
            },
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "gate_input_sha256": hashlib.sha256(json.dumps(gate_input, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
        }
        key = "e2e_ab_diagnostics" if phase == "before_fix" else "e2e_ab_calibration_replays"
        result.setdefault(key, []).append(detail)
        write_result(result)
        print("E2E_GATE_REPLAY", json.dumps({
            "case_id": detail["case_id"], "question": question,
            "evidence_indices": [item["record_index"] for item in evidence],
            "gate_structured_output": detail["gate_structured_output"],
            "elapsed_seconds": detail["elapsed_seconds"],
        }, ensure_ascii=False), flush=True)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    golden_bytes = GOLDEN_PATH.read_bytes()
    golden_sha = hashlib.sha256(golden_bytes).hexdigest()
    golden = load_json(GOLDEN_PATH)
    generation = load_json(GENERATION_PATH)
    release = load_json(RELEASE_PATH)
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-e2e-gate", action="store_true", help="Replay only the Qwen gate on preserved A/B inputs; no generation or database write.")
    parser.add_argument("--inspect-safe", action="store_true", help="Print the local draft/evidence for the 21 original answerable abstentions.")
    parser.add_argument("--inspect-unanswerable", action="store_true")
    parser.add_argument("--gate-phase", choices=("before_fix", "after_fix"), default="after_fix")
    args = parser.parse_args()
    if golden_sha != EXPECTED_GOLDEN_SHA256:
        raise RuntimeError("Frozen Golden SHA-256 changed")
    if any(doc.get("golden_sha256") != golden_sha for doc in (generation, release)):
        raise RuntimeError("Cached result is not from the frozen Golden")
    if any(len(doc.get("cases", [])) != 60 for doc in (golden, generation, release)):
        raise RuntimeError("Expected exactly 60 cases in all three artifacts")
    golden_by_id = {case["id"]: case for case in golden["cases"]}
    release_by_id = {case["id"]: case for case in release["cases"]}
    details = []
    for cached in generation["cases"]:
        case_id = cached["id"]
        gold = golden_by_id[case_id]
        prior = release_by_id[case_id]
        for key in ("question", "answerable", "relevant_record_indices"):
            if cached.get(key) != gold.get(key) or prior.get(key) != gold.get(key):
                raise RuntimeError(f"Case {case_id} mapping differs for {key}")
        evidence = cached["retrieved_records"]
        retrieved = [item["record_index"] for item in evidence]
        if retrieved != prior["actual_evidence_indices"]:
            raise RuntimeError(f"Case {case_id} Evidence differs from cached Gate evaluation")
        draft = cached.get("answer") or ""
        current_audit = audit_answer(cached["question"], draft, evidence_records(evidence))
        item = {
            "id": case_id,
            "question": cached["question"],
            "answerable": cached["answerable"],
            "category": cached["category"],
            "intent": prior["scene"],
            "intent_source": "Golden category mapping used by script57; no live Router output is cached for these 60 cases.",
            "generation_status": "generated" if draft and cached.get("error") is None else "failed",
            "audit_status": prior["program_audit_status"],
            "current_audit_status": current_audit["status"],
            "draft_exists": bool(draft.strip()),
            "evidence_count": len(evidence),
            "gold_relevant_record_indices": gold["relevant_record_indices"],
            "actual_retrieved_record_indices": retrieved,
            "gold_record_in_top3": any(index in gold["relevant_record_indices"] for index in retrieved),
            "reference_answer": gold["reference_answer"],
            "draft_answer": draft,
            "actual_evidence": evidence,
            "release_status": prior["delivery_status"],
            "citation_status": prior["citation_status"],
            "gate_status": prior["gate_status"],
            "gate_reason_code": prior["gate_reason_code"],
            "original_gate_unsupported_claims": "Not persisted by script57; do not infer or invent them.",
            "abstention_pattern_matches": [match.group() for match in ABSTENTION_PATTERN.finditer(draft)],
        }
        if item["release_status"] == "review_required":
            item["review_required_cause"] = review_cause(item)
        if item["release_status"] == "safe_abstention" and item["answerable"]:
            item["safe_abstention_diagnosis"] = abstention_cause(item)
        details.append(item)
        if item["answerable"]:
            print(f"{case_id} intent={item['intent']} generated={item['generation_status']} audit={item['audit_status']} gold_hit={item['gold_record_in_top3']} release={item['release_status']}")
        if args.inspect_safe and item.get("safe_abstention_diagnosis"):
            print("SAFE_CASE", json.dumps({key: item[key] for key in (
                "id", "question", "gold_record_in_top3", "reference_answer",
                "draft_answer", "actual_retrieved_record_indices",
            )}, ensure_ascii=False), flush=True)
        if args.inspect_unanswerable and not item["answerable"]:
            print("UNANSWERABLE_CASE", json.dumps({key: item[key] for key in (
                "id", "question", "draft_answer", "citation_status", "audit_status",
            )}, ensure_ascii=False), flush=True)
    answerable = [case for case in details if case["answerable"]]
    safe = [case for case in answerable if case["release_status"] == "safe_abstention"]
    reviewed = [case for case in answerable if case["release_status"] == "review_required"]
    previous = load_json(OUTPUT_PATH) if OUTPUT_PATH.exists() else {}
    result = {
        "diagnostic_id": "answer_release_diagnostics_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "golden_sha256": golden_sha,
        "source": {"generation": str(GENERATION_PATH.relative_to(ROOT)), "release_v1": str(RELEASE_PATH.relative_to(ROOT)), "cached_generation_reused": True},
        "mapping_checks": {"all_60_ids_questions_labels_and_evidence_match": True, "mapping_errors": 0},
        "original_summary": release["summary"],
        "answerable_cases": answerable,
        "unanswerable_cases": [case for case in details if not case["answerable"]],
        "safe_abstention_cause_counts": dict(Counter(case["safe_abstention_diagnosis"]["category"] for case in safe)),
        "review_required_cause_counts": dict(Counter(case["review_required_cause"] for case in reviewed)),
        "audit_implementation_observation": "Original V1 ReleasePolicy selected safe_abstention solely from MODEL_ABSTAINED_NEEDS_REVIEW after program blockers; original audit_answer assigned it when any ABSTENTION_PATTERN matched anywhere in the draft.",
        "implementation_findings": {
            "audit_bug": "20 gold-hit drafts contain actual cited answers followed by local limitations; whole-draft substring classification erased those answers. Cited content is not proof of support, so corrected mixed classification must still go through the Gate.",
            "case_a_manual_evidence_check": {
                "frequency": "E1: 软胶主刷 每 2 周清理 每 6-12 个月; maintenance table explicitly introduced as recommendations.",
                "replacement_and_abnormal_wear": "E1 maintenance table and replacement note.",
                "cleaning_steps_and_safety": "E3 four disassembly/reassembly steps and both cleaning cautions.",
                "abnormal_noise": "E2 mentions possible brush entanglement and stopping before cleaning.",
                "finding": "All concrete draft claims are supported; the original-prompt replay rejected the explicitly stated cleaning frequency, demonstrating a false negative in table/semantic support.",
            },
            "case_b": "Original API was review_required but did not persist original Gate claims. Same saved draft/evidence replay was SUPPORTED. Original exact reason cannot be recovered and is not invented.",
        },
        "e2e_ab_diagnostics": previous.get("e2e_ab_diagnostics", []),
        "e2e_ab_calibration_replays": previous.get("e2e_ab_calibration_replays", []) if not (args.replay_e2e_gate and args.gate_phase == "after_fix") else [],
        "database_writes": 0,
        "answer_generation_calls": 0,
    }
    # Re-running the offline diagnostic must not discard the subsequent live
    # calibration and UI acceptance evidence stored in this same artifact.
    for key in (
        "calibration_summary", "real_e2e_after_fix",
        "live_ab_record_snapshots_after_fix", "react_ui_verification",
        "verification_summary",
    ):
        if key in previous:
            result[key] = previous[key]
    if args.replay_e2e_gate and args.gate_phase == "before_fix" and E2E_SNAPSHOT_PATH.exists():
        result["e2e_ab_diagnostics"] = []
    write_result(result)
    if args.replay_e2e_gate:
        replay_e2e_gate(result, phase=args.gate_phase)
    print("DIAGNOSTIC_SUMMARY", json.dumps({
        "answerable": len(answerable),
        "safe_abstention": len(safe),
        "safe_cause_counts": result["safe_abstention_cause_counts"],
        "review_required": len(reviewed),
        "review_cause_counts": result["review_required_cause_counts"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
