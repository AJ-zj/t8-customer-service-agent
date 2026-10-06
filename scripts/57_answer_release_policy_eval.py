"""Read-only Release Policy V1 evaluation over frozen Golden drafts and evidence.

This script reuses the already saved generation_eval_v1 drafts and retrieved
Evidence. It calls only the semantic grounding gate; it does not regenerate,
retrieve, or write to any database.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import SEMANTIC_GATE_MODEL  # noqa: E402
from app.services.answer_release_policy import (  # noqa: E402
    AnswerReleasePolicy,
    GateOutcome,
    PROGRAM_BLOCKED_STATUSES,
    SAFE_ABSTENTION_ANSWER,
)
from app.services.grounded_qa import audit_answer  # noqa: E402
from app.services.semantic_grounding_gate import SemanticGroundingGate  # noqa: E402
from app.services.static_knowledge_boundary import assess_static_knowledge_boundary  # noqa: E402


GOLDEN_PATH = ROOT / "data/evaluation/t8_rag_golden_v1.json"
GENERATION_PATH = ROOT / "data/evaluation/results/generation_eval_v1.json"
RESULT_PATH = ROOT / "data/evaluation/results/answer_release_policy_v1.json"
BOUNDARY_RESULT_PATH = ROOT / "data/evaluation/results/static_knowledge_boundary_v1.json"
SCENE_BY_CATEGORY = {"维护": "maintenance", "故障": "troubleshooting"}


def _records_from_case(case):
    records = []
    for rank, item in enumerate(case.get("retrieved_records", []), 1):
        records.append((
            item.get("evidence_id") or f"E{rank}",
            item["record_index"],
            {key: item.get(key) for key in ("pdf_page", "title", "section", "text")},
        ))
    return records


def _evidence_for_gate(records):
    return [
        {
            "evidence_id": evidence_id,
            "record_index": record_index,
            "pdf_page": record.get("pdf_page"),
            "title": record.get("title"),
            "section": record.get("section"),
            "text": record.get("text") or "",
        }
        for evidence_id, record_index, record in records
    ]


def _scene_for_case(case):
    return SCENE_BY_CATEGORY.get(case.get("category"), "consult_qa")


def _verification_basis(case, evidence, audit_status, decision, gate_status):
    """Keep an inspectable machine-assisted trace, never claim human truth."""
    return {
        "method": "machine_assisted_not_human_truth",
        "reference_answer": case.get("reference_answer"),
        "relevant_record_indices": case.get("relevant_record_indices", []),
        "actual_evidence": [
            {key: item.get(key) for key in (
                "evidence_id", "record_index", "pdf_page", "title", "section", "text"
            )}
            for item in evidence
        ],
        "draft_answer": case.get("answer"),
        "program_audit_status": audit_status,
        "citation_status": "VALID" if audit_status == "NEEDS_HUMAN_SEMANTIC_REVIEW" else None,
        "gate_status": gate_status,
        "gold_relevant_record_retrieved": any(
            item.get("record_index") in case.get("relevant_record_indices", [])
            for item in evidence
        ),
        "delivery_status": decision.status,
    }


def evaluate(*, gate=None, result_path=RESULT_PATH, boundary_result_path=BOUNDARY_RESULT_PATH):
    golden_bytes = GOLDEN_PATH.read_bytes()
    golden = json.loads(golden_bytes.decode("utf-8"))
    generation = json.loads(GENERATION_PATH.read_text(encoding="utf-8"))
    golden_sha = hashlib.sha256(golden_bytes).hexdigest()
    if golden_sha != "547fd0f48197bc150640100fb07ba6287fee5d0f4439c1e96c2950c2167e9386":
        raise RuntimeError("Frozen Golden SHA-256 changed")
    if generation.get("golden_sha256") != golden_sha:
        raise RuntimeError("generation_eval_v1 does not match the frozen Golden SHA-256")
    cases = generation.get("cases")
    if not isinstance(cases, list) or len(cases) != 60:
        raise RuntimeError("Expected exactly 60 cached generation cases")
    if len(golden.get("cases", [])) != 60:
        raise RuntimeError("Expected exactly 60 frozen Golden cases")
    boundary_by_id = {
        case["id"]: assess_static_knowledge_boundary(case["question"])
        for case in golden["cases"]
    }
    # Independent question-only output: no drafts or evidence influence the
    # boundary labels; keep the frozen answerability labels for false positives.
    boundary_result = {
        "golden_sha256": golden_sha,
        "model_calls": 0,
        "cases": [{
            "id": case["id"], "question": case["question"],
            "answerable": case["answerable"],
            "boundary_status": boundary_by_id[case["id"]].status,
            "boundary_reason": boundary_by_id[case["id"]].reason_code,
        } for case in golden["cases"]],
        "answerable_false_boundary_count": sum(
            case["answerable"] and boundary_by_id[case["id"]].status == "personal_runtime_unavailable"
            for case in golden["cases"]
        ),
    }
    _write_result(boundary_result, complete=True, path=boundary_result_path)
    gate = gate or SemanticGroundingGate()
    policy = AnswerReleasePolicy()
    result = {
        "evaluation_id": result_path.stem,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "golden_sha256": golden_sha,
        "source": {
            "generation_results": "data/evaluation/results/generation_eval_v1.json",
            "generation_model": generation.get("model"),
            "cached_generation_reused": True,
            "new_answer_generation_calls": 0,
        },
        "rules": {
            "eligible_scenes": ["consult_qa", "maintenance", "troubleshooting"],
            "program_blocked_statuses": sorted(PROGRAM_BLOCKED_STATUSES),
            "safe_abstention_status": "MODEL_ABSTAINED_NEEDS_REVIEW",
            "safe_abstention_answer": SAFE_ABSTENTION_ANSWER,
            "semantic_gate_model": SEMANTIC_GATE_MODEL,
            "live_auto_release_does_not_approve_or_publish": True,
        },
        "machine_assisted_verification": {
            "human_review_completed": False,
            "notice": "LLM Gate is not human ground truth; auto-release cases include reference, gold relevant index, actual Evidence, and draft for inspection.",
        },
        "gate_call_count": 0,
        "gate_failure_count": 0,
        "cases": [],
        "summary": {},
        "real_e2e": {"completed": False, "cases": []},
    }

    golden_by_id = {item["id"]: item for item in golden["cases"]}
    for case in cases:
        gold = golden_by_id.get(case["id"])
        if gold is None or case.get("question") != gold.get("question"):
            raise RuntimeError("Cached generation case does not match frozen Golden")
        records = _records_from_case(case)
        evidence = _evidence_for_gate(records)
        audit = audit_answer(case["question"], case.get("answer") or "", records)
        scene = _scene_for_case(case)
        boundary = boundary_by_id[case["id"]]
        pre = policy.decide(
            scene=scene,
            generation_status="generated" if case.get("answer") else "failed",
            audit_status=audit.get("status"),
            draft_answer=case.get("answer"),
            evidence=evidence,
            citation_status=audit.get("citation_status"),
            knowledge_boundary=boundary,
        )
        gate_outcome = None
        latency = 0.0
        if pre.reason_code == "SEMANTIC_GATE_REQUIRED":
            result["gate_call_count"] += 1
            started = time.perf_counter()
            gate_outcome = gate.evaluate(
                question=case["question"],
                draft_answer=case["answer"],
                evidence=evidence,
            )
            latency = round(time.perf_counter() - started, 3)
            if gate_outcome.status == "failed":
                result["gate_failure_count"] += 1
        decision = policy.decide(
            scene=scene,
            generation_status="generated" if case.get("answer") else "failed",
            audit_status=audit.get("status"),
            draft_answer=case.get("answer"),
            evidence=evidence,
            citation_status=audit.get("citation_status"),
            gate_outcome=gate_outcome,
            knowledge_boundary=boundary,
        )
        saved_case = {
            "id": case["id"],
            "question": case["question"],
            "answerable": bool(case["answerable"]),
            "category": case.get("category"),
            "scene": scene,
            "boundary_status": boundary.status,
            "boundary_reason": boundary.reason_code,
            "relevant_record_indices": case.get("relevant_record_indices", []),
            "actual_evidence_indices": [item.get("record_index") for item in evidence],
            "program_audit_status": audit.get("status"),
            "citation_status": audit.get("citation_status"),
            "gate_status": gate_outcome.status if gate_outcome else None,
            "gate_reason_code": gate_outcome.reason_code if gate_outcome else None,
            "gate_unsupported_claims": list(gate_outcome.unsupported_claims) if gate_outcome else [],
            "all_claims_supported": gate_outcome.all_claims_supported if gate_outcome else None,
            "core_answer_supported": gate_outcome.core_answer_supported if gate_outcome else None,
            "core_abstention_quote": gate_outcome.core_abstention_quote if gate_outcome else None,
            "gate_invoked": gate_outcome is not None,
            "gate_latency_seconds": latency,
            "delivery_status": decision.status,
            "release_reason_code": decision.reason_code,
            "draft_answer": case.get("answer"),
            "actual_evidence": evidence,
        }
        if decision.status == "auto_released":
            saved_case["verification_basis"] = _verification_basis(
                case, evidence, audit.get("status"), decision,
                gate_outcome.status if gate_outcome else None,
            )
        result["cases"].append(saved_case)
        print(
            f"{case['id']} answerable={bool(case['answerable'])} "
            f"audit={audit.get('status')} gate={saved_case['gate_status']} "
            f"delivery={decision.status} gate_s={latency:.3f}",
            flush=True,
        )
        # Persist progress atomically so an interrupted external call does not
        # discard completed, non-secret evaluation records.
        _write_result(result, complete=False, path=result_path)

    answerable = [item for item in result["cases"] if item["answerable"]]
    unanswerable = [item for item in result["cases"] if not item["answerable"]]
    auto_cases = [item for item in result["cases"] if item["delivery_status"] == "auto_released"]
    result["summary"] = {
        "total_cases": len(result["cases"]),
        "answerable": {
            "count": len(answerable),
            "auto_release_count": sum(item["delivery_status"] == "auto_released" for item in answerable),
            "review_required_count": sum(item["delivery_status"] == "review_required" for item in answerable),
            "safe_abstention_count": sum(item["delivery_status"] == "safe_abstention" for item in answerable),
        },
        "unanswerable": {
            "count": len(unanswerable),
            "safe_abstention_count": sum(item["delivery_status"] == "safe_abstention" for item in unanswerable),
            "review_required_count": sum(item["delivery_status"] == "review_required" for item in unanswerable),
            "incorrect_auto_release_count": sum(item["delivery_status"] == "auto_released" for item in unanswerable),
        },
        "auto_release_rate": len(auto_cases) / len(result["cases"]) if result["cases"] else 0,
        "safe_abstention_rate_unanswerable": (
            sum(item["delivery_status"] == "safe_abstention" for item in unanswerable) / len(unanswerable)
            if unanswerable else 0
        ),
        "incorrect_auto_release_count": sum(item["delivery_status"] == "auto_released" for item in unanswerable),
        "gate_call_count": result["gate_call_count"],
        "gate_failure_count": result["gate_failure_count"],
        "boundary_unavailable_count": sum(
            item["boundary_status"] == "personal_runtime_unavailable" for item in result["cases"]
        ),
        "answerable_false_boundary_count": boundary_result["answerable_false_boundary_count"],
        "semantic_unsupported_count": sum(item["gate_status"] == "unsupported" for item in result["cases"]),
        "review_required_count": sum(item["delivery_status"] == "review_required" for item in result["cases"]),
    }
    _write_result(result, complete=True, path=result_path)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    return result


def _write_result(result, *, complete, path=RESULT_PATH):
    output = dict(result)
    output["complete"] = complete
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.resolve().parent != RESULT_PATH.parent.resolve():
        parser.error("Output must be in data/evaluation/results")
    evaluate(result_path=output)
