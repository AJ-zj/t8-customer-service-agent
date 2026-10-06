"""ORM models used by the T8 internal review workflow."""

from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, Float, Integer, String, Text, UniqueConstraint, func, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base metadata for project ORM tables."""


class QAReviewRecord(Base):
    """A generated answer and its automated and human review state."""

    __tablename__ = "qa_review_records"
    __table_args__ = (
        CheckConstraint("publication_status IN ('unpublished', 'published')",
                        name="ck_review_publication_status"),
        CheckConstraint("publication_status != 'published' OR "
                        "(review_status = 'approved' AND published_at IS NOT NULL "
                        "AND draft_answer IS NOT NULL)",
                        name="ck_review_published_approved"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    draft_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    evidence: Mapped[list | dict] = mapped_column(JSONB, nullable=False)
    audit_status: Mapped[str] = mapped_column(String(64), nullable=False)
    review_status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="pending_review",
        server_default=text("'pending_review'"),
    )
    review_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    reviewed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    publication_status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="unpublished",
        server_default=text("'unpublished'"),
    )
    published_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class AgentMemoryFact(Base):
    """Candidate user fact; ownership is not an authenticated identity yet."""

    __tablename__ = "agent_memory_facts"
    __table_args__ = (
        UniqueConstraint("memory_owner_id", "fact_hash", name="uq_agent_memory_owner_fact_hash"),
        CheckConstraint("importance >= 0 AND importance <= 1", name="ck_memory_importance"),
        CheckConstraint("fact_category IS NULL OR fact_category IN "
                        "('device_profile', 'home_environment', 'usage_preference')",
                        name="ck_memory_fact_category"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    memory_owner_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    fact_text: Mapped[str] = mapped_column(Text, nullable=False)
    fact_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    embedding: Mapped[list] = mapped_column(JSONB, nullable=False)
    seen_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1,
                                            server_default=text("1"))
    # NULL category denotes historical/uncategorized facts; never infer it from text.
    fact_category: Mapped[str | None] = mapped_column(String(32), nullable=True)
    importance: Mapped[float] = mapped_column(Float, nullable=False, default=0.5,
                                              server_default=text("0.5"))
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False,
                                                 server_default=func.now())
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False,
                                                  server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False,
                                                  server_default=func.now(),
                                                  onupdate=func.now())
