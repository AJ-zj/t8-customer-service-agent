"""Internal read and decision operations for QA review records."""

from sqlalchemy import func, select, update

from app.models import QAReviewRecord


class ReviewServiceError(Exception):
    """Safe error raised when review records cannot be read."""


class ReviewRecordNotFound(Exception):
    """Raised when a review record ID does not exist."""


class ReviewDecisionConflict(Exception):
    """Raised when a record has already left pending_review."""


class ReviewPublicationConflict(Exception):
    """Only an approved, unpublished draft can be explicitly published."""


class ReviewRecordService:
    def __init__(self, session_factory=None):
        self._session_factory = session_factory

    def _new_session(self):
        try:
            if self._session_factory is None:
                from app.database import SessionLocal, configured_database_name

                if configured_database_name() != "roborock":
                    raise RuntimeError("Review records are restricted to roborock.")
                self._session_factory = SessionLocal
            return self._session_factory()
        except Exception:
            raise ReviewServiceError("审核记录暂时不可用，请稍后重试。") from None

    @staticmethod
    def _evidence_count(evidence):
        if isinstance(evidence, (list, dict)):
            return len(evidence)
        return 0

    def list_pending(self, limit):
        return self._list_records(limit, pending_only=True)

    def list_all(self, limit):
        return self._list_records(limit, pending_only=False)

    def _list_records(self, limit, *, pending_only):
        session = self._new_session()
        try:
            statement = (
                select(QAReviewRecord)
                .order_by(
                    QAReviewRecord.created_at.desc(),
                    QAReviewRecord.id.desc(),
                )
                .limit(limit)
            )
            if pending_only:
                statement = statement.where(QAReviewRecord.review_status == "pending_review")
            records = session.scalars(statement).all()
            return [
                {
                    "id": record.id,
                    "question": record.question,
                    "audit_status": record.audit_status,
                    "review_status": record.review_status,
                    "created_at": record.created_at,
                    "has_draft": record.draft_answer is not None,
                    "evidence_count": self._evidence_count(record.evidence),
                    "publication_status": getattr(record, "publication_status", None) or "unpublished",
                }
                for record in records
            ]
        except Exception:
            raise ReviewServiceError("审核记录暂时不可用，请稍后重试。") from None
        finally:
            self._close_session(session)

    def get_by_id(self, record_id):
        session = self._new_session()
        try:
            record = session.get(QAReviewRecord, record_id)
            if record is None:
                return None
            return {
                "id": record.id,
                "question": record.question,
                "draft_answer": record.draft_answer,
                "audit_status": record.audit_status,
                "review_status": record.review_status,
                "evidence": record.evidence,
                "review_note": record.review_note,
                "created_at": record.created_at,
                "reviewed_at": record.reviewed_at,
                "publication_status": getattr(record, "publication_status", None) or "unpublished",
                "published_at": getattr(record, "published_at", None),
            }
        except Exception:
            raise ReviewServiceError("审核记录暂时不可用，请稍后重试。") from None
        finally:
            self._close_session(session)

    def decide(self, record_id, decision, review_note):
        session = self._new_session()
        try:
            statement = (
                update(QAReviewRecord)
                .where(
                    QAReviewRecord.id == record_id,
                    QAReviewRecord.review_status == "pending_review",
                    QAReviewRecord.publication_status == "unpublished",
                )
                .values(
                    review_status=decision,
                    review_note=review_note,
                    reviewed_at=func.now(),
                )
                .returning(
                    QAReviewRecord.id,
                    QAReviewRecord.review_status,
                    QAReviewRecord.review_note,
                    QAReviewRecord.reviewed_at,
                )
            )
            updated = session.execute(statement).mappings().one_or_none()
            if updated is None:
                exists = session.scalar(
                    select(QAReviewRecord.id).where(QAReviewRecord.id == record_id)
                )
                session.rollback()
                if exists is None:
                    raise ReviewRecordNotFound
                raise ReviewDecisionConflict

            session.commit()
            return {
                "id": updated["id"],
                "review_status": updated["review_status"],
                "review_note": updated["review_note"],
                "reviewed_at": updated["reviewed_at"],
                "published": False,
            }
        except (ReviewRecordNotFound, ReviewDecisionConflict):
            raise
        except Exception:
            try:
                session.rollback()
            except Exception:
                pass
            raise ReviewServiceError("审核记录暂时不可用，请稍后重试。") from None
        finally:
            self._close_session(session)

    def publish(self, record_id):
        session = self._new_session()
        try:
            # The database checks both states inside UPDATE, including concurrent calls.
            statement = (
                update(QAReviewRecord)
                .where(
                    QAReviewRecord.id == record_id,
                    QAReviewRecord.review_status == "approved",
                    QAReviewRecord.publication_status == "unpublished",
                    QAReviewRecord.draft_answer.is_not(None),
                    func.length(func.trim(QAReviewRecord.draft_answer)) > 0,
                )
                .values(publication_status="published", published_at=func.now())
                .returning(QAReviewRecord.id, QAReviewRecord.review_status,
                           QAReviewRecord.publication_status, QAReviewRecord.published_at)
            )
            updated = session.execute(statement).mappings().one_or_none()
            if updated is None:
                exists = session.scalar(
                    select(QAReviewRecord.id).where(QAReviewRecord.id == record_id)
                )
                session.rollback()
                if exists is None:
                    raise ReviewRecordNotFound
                raise ReviewPublicationConflict
            result = {
                "record_id": updated["id"],
                "review_status": updated["review_status"],
                "publication_status": updated["publication_status"],
                "published_at": updated["published_at"],
            }
            session.commit()
            return result
        except (ReviewRecordNotFound, ReviewPublicationConflict):
            raise
        except Exception:
            try:
                session.rollback()
            except Exception:
                pass
            raise ReviewServiceError("审核记录暂时不可用，请稍后重试。") from None
        finally:
            self._close_session(session)

    def get_published(self, record_id):
        session = self._new_session()
        try:
            # Select only the public fields; never load evidence or review metadata.
            row = session.execute(
                select(QAReviewRecord.id, QAReviewRecord.question,
                       QAReviewRecord.draft_answer, QAReviewRecord.published_at)
                .where(QAReviewRecord.id == record_id,
                       QAReviewRecord.review_status == "approved",
                       QAReviewRecord.publication_status == "published",
                       QAReviewRecord.published_at.is_not(None),
                       QAReviewRecord.draft_answer.is_not(None))
            ).mappings().one_or_none()
            if row is None:
                return None
            return {"record_id": row["id"], "question": row["question"],
                    "answer": row["draft_answer"], "published": True,
                    "published_at": row["published_at"]}
        except Exception:
            raise ReviewServiceError("审核记录暂时不可用，请稍后重试。") from None
        finally:
            self._close_session(session)

    @staticmethod
    def _close_session(session):
        try:
            session.close()
        except Exception:
            # Cleanup must not replace a committed result or an existing error.
            pass
