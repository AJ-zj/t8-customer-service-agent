"""Deterministic live-answer release policy, separate from public publication."""

from dataclasses import dataclass
from typing import Literal

from app.services.static_knowledge_boundary import KnowledgeBoundaryDecision


AUTO_RELEASE_SCENES = frozenset({"consult_qa", "maintenance", "troubleshooting"})
PROGRAM_BLOCKED_STATUSES = frozenset({
    "PROGRAM_BLOCKED_INVALID_CITATION",
    "PROGRAM_BLOCKED_UNSUPPORTED_MONEY_CLAIM",
    "PROGRAM_BLOCKED_UNSUPPORTED_PERSONAL_RUNTIME",
    "PROGRAM_BLOCKED_MISSING_CITATION",
})
SAFE_ABSTENTION_STATUS = "MODEL_ABSTAINED_NEEDS_REVIEW"
SEMANTIC_REVIEW_STATUS = "NEEDS_HUMAN_SEMANTIC_REVIEW"
SAFE_ABSTENTION_ANSWER = "根据当前提供的说明书资料无法确认。"


@dataclass(frozen=True)
class ReleaseDecision:
    status: Literal["auto_released", "review_required", "safe_abstention"]
    reason_code: str


@dataclass(frozen=True)
class GateOutcome:
    status: Literal["supported", "unsupported", "failed"]
    reason_code: str
    all_claims_supported: bool | None = None
    core_answer_supported: bool | None = None
    core_abstention_quote: str | None = None


class AnswerReleasePolicy:
    """Decide only live delivery; this never changes review/publication state."""

    def decide(
        self,
        *,
        scene: str,
        generation_status: str | None,
        audit_status: str | None,
        draft_answer: str | None,
        evidence,
        citation_status: str | None,
        gate_outcome: GateOutcome | None = None,
        knowledge_boundary: KnowledgeBoundaryDecision | None = None,
    ) -> ReleaseDecision:
        if audit_status in PROGRAM_BLOCKED_STATUSES:
            return ReleaseDecision("review_required", "PROGRAM_AUDIT_BLOCKED")
        if scene not in AUTO_RELEASE_SCENES:
            return ReleaseDecision("review_required", "SCENE_NOT_ELIGIBLE")
        if generation_status != "generated":
            return ReleaseDecision("review_required", "AUDIT_NOT_RELEASE_CANDIDATE")
        if not isinstance(draft_answer, str) or not draft_answer.strip():
            return ReleaseDecision("review_required", "DRAFT_MISSING")
        if not isinstance(evidence, list) or not evidence:
            return ReleaseDecision("review_required", "EVIDENCE_MISSING")
        if audit_status not in {SAFE_ABSTENTION_STATUS, SEMANTIC_REVIEW_STATUS}:
            return ReleaseDecision("review_required", "AUDIT_NOT_RELEASE_CANDIDATE")
        if (knowledge_boundary is not None
                and knowledge_boundary.status == "personal_runtime_unavailable"):
            # Source capability is deterministic and cannot be rescued by a
            # positive semantic Gate, nor does it approve/publish the review.
            return ReleaseDecision("safe_abstention", "STATIC_KNOWLEDGE_RUNTIME_UNAVAILABLE")
        if audit_status == SAFE_ABSTENTION_STATUS:
            return ReleaseDecision("safe_abstention", "MODEL_ABSTAINED")
        if audit_status != SEMANTIC_REVIEW_STATUS:
            return ReleaseDecision("review_required", "AUDIT_NOT_RELEASE_CANDIDATE")
        if citation_status != "VALID":
            return ReleaseDecision("review_required", "CITATION_AUDIT_NOT_PASSED")
        if gate_outcome is None:
            return ReleaseDecision("review_required", "SEMANTIC_GATE_REQUIRED")
        if gate_outcome.status == "failed":
            return ReleaseDecision("review_required", "SEMANTIC_GATE_FAILED_CLOSED")
        claims_supported = gate_outcome.all_claims_supported
        core_supported = gate_outcome.core_answer_supported
        abstention_quote = gate_outcome.core_abstention_quote
        if (type(claims_supported) is not bool or type(core_supported) is not bool
                or not isinstance(abstention_quote, str)
                or gate_outcome.status not in {"supported", "unsupported"}
                or (gate_outcome.status == "supported") != (claims_supported and core_supported)
                or (core_supported and abstention_quote)
                or (abstention_quote and (
                    not abstention_quote.strip() or abstention_quote not in draft_answer
                ))):
            return ReleaseDecision("review_required", "SEMANTIC_GATE_FAILED_CLOSED")
        if claims_supported and core_supported:
            return ReleaseDecision("auto_released", "SEMANTICALLY_GROUNDED")
        if not core_supported and abstention_quote:
            # Return only the fixed abstention text, never ancillary draft facts.
            return ReleaseDecision("safe_abstention", "CORE_QUESTION_ABSTAINED")
        return ReleaseDecision("review_required", "SEMANTICALLY_UNSUPPORTED")
