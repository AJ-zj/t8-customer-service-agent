from datetime import datetime
from typing import Any, Literal

from app.agent.state import Intent
from pydantic import BaseModel, Field, field_validator


AuditStatus = Literal[
    "PROGRAM_BLOCKED_INVALID_CITATION",
    "PROGRAM_BLOCKED_UNSUPPORTED_MONEY_CLAIM",
    "PROGRAM_BLOCKED_UNSUPPORTED_PERSONAL_RUNTIME",
    "PROGRAM_BLOCKED_MISSING_CITATION",
    "MODEL_ABSTAINED_NEEDS_REVIEW",
    "NEEDS_HUMAN_SEMANTIC_REVIEW",
]


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)

    @field_validator("question", mode="before")
    @classmethod
    def trim_question(cls, value):
        if isinstance(value, str):
            return value.strip()
        return value


class AgentChatRequest(ChatRequest):
    # Local ownership label only; not an authenticated user identity.
    memory_owner_id: str | None = Field(
        default=None, min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$"
    )
    session_id: str | None = Field(
        default=None, min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$"
    )

    @field_validator("session_id", "memory_owner_id", mode="before")
    @classmethod
    def trim_session_id(cls, value):
        return value.strip() if isinstance(value, str) else value


class ChatResponse(BaseModel):
    generation_status: Literal["generated"]
    audit_status: AuditStatus
    answer: str | None = None
    published: bool = False
    safety_notice: str


class AgentChatResponse(BaseModel):
    intent: Intent
    answer: str | None = None
    delivery_status: Literal[
        "auto_released", "review_required", "safe_abstention",
        "clarification", "human_handoff",
    ] | None = None
    generation_status: Literal["generated"] | None = None
    audit_status: AuditStatus | None = None
    published: bool = False
    safety_notice: str | None = None


class ReviewListItem(BaseModel):
    id: int
    question: str
    audit_status: str
    review_status: str
    created_at: datetime
    has_draft: bool
    evidence_count: int
    publication_status: str = "unpublished"


class ReviewDetail(BaseModel):
    id: int
    question: str
    draft_answer: str | None
    audit_status: str
    review_status: str
    evidence: Any
    review_note: str | None
    created_at: datetime
    reviewed_at: datetime | None
    publication_status: str = "unpublished"
    published_at: datetime | None = None


class ReviewDecisionRequest(BaseModel):
    decision: Literal["approved", "rejected"]
    review_note: str = Field(..., min_length=1, max_length=1000)

    @field_validator("review_note", mode="before")
    @classmethod
    def trim_review_note(cls, value):
        if isinstance(value, str):
            return value.strip()
        return value


class ReviewDecisionResponse(BaseModel):
    id: int
    review_status: Literal["approved", "rejected"]
    review_note: str
    reviewed_at: datetime
    published: bool = False


class PublicationResponse(BaseModel):
    record_id: int
    review_status: Literal["approved"]
    publication_status: Literal["published"]
    published_at: datetime


class PublishedAnswerResponse(BaseModel):
    record_id: int
    question: str
    answer: str
    published: Literal[True] = True
    published_at: datetime
