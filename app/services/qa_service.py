import re
import threading

from app.services import grounded_qa
from app.models import QAReviewRecord


ALLOWED_AUDIT_STATUSES = {
    "PROGRAM_BLOCKED_INVALID_CITATION",
    "PROGRAM_BLOCKED_UNSUPPORTED_MONEY_CLAIM",
    "PROGRAM_BLOCKED_UNSUPPORTED_PERSONAL_RUNTIME",
    "PROGRAM_BLOCKED_MISSING_CITATION",
    "MODEL_ABSTAINED_NEEDS_REVIEW",
    "NEEDS_HUMAN_SEMANTIC_REVIEW",
}


class QAServiceError(Exception):
    """Safe service-level error; never contains upstream response details."""


def normalize_retrieval_query(question: str) -> str:
    """Normalize only the low-discrimination terms validated in offline ablation."""
    # 离线消融验证过的保守规则：只去除开头礼貌词和产品实体，不做通用停用词删除。
    query = re.sub(r"^\s*请问", "", question, count=1)
    query = query.replace("石头", "")
    query = re.sub(r"T8", "", query, flags=re.IGNORECASE)
    query = re.sub(r"\s+", " ", query).strip()
    return query or question.strip() or question


class GroundedQAService:
    def __init__(self, qa_module=None, session_factory=None):
        self._qa = qa_module
        self._pipeline = None
        self._session_factory = session_factory
        self._lock = threading.Lock()

    def _load_qa_module(self):
        return grounded_qa

    def _get_pipeline(self):
        if self._pipeline is None:
            with self._lock:
                if self._pipeline is None:
                    try:
                        if self._qa is None:
                            self._qa = self._load_qa_module()
                        self._pipeline = self._qa.create_pipeline()
                    except Exception:
                        raise QAServiceError("问答服务暂不可用，请稍后重试。") from None
        return self._pipeline

    def _get_session_factory(self):
        if self._session_factory is None:
            from app.database import SessionLocal, configured_database_name

            if configured_database_name() != "roborock":
                raise RuntimeError("Review records may only be written to roborock.")
            self._session_factory = SessionLocal
        return self._session_factory

    @staticmethod
    def _snapshot_evidence(top_records):
        return [
            {
                "evidence_id": evidence_id,
                "record_index": record_index,
                "pdf_page": record.get("pdf_page"),
                "title": record.get("title"),
                "section": record.get("section"),
                "text": record.get("text", ""),
            }
            for evidence_id, record_index, record in top_records
        ]

    def _save_review_record(
        self, question, draft_answer, evidence_snapshot, audit_status
    ):
        try:
            session = self._get_session_factory()()
        except Exception:
            raise QAServiceError("问答服务暂不可用，请稍后重试。") from None

        try:
            session.add(QAReviewRecord(
                question=question,
                draft_answer=draft_answer,
                evidence=evidence_snapshot,
                audit_status=audit_status,
            ))
            session.commit()
        except Exception:
            try:
                session.rollback()
            except Exception:
                pass
            raise QAServiceError("问答服务暂不可用，请稍后重试。") from None
        finally:
            try:
                session.close()
            except Exception:
                raise QAServiceError("问答服务暂不可用，请稍后重试。") from None

    def finalize_draft(self, question, draft_answer, top_records):
        """Audit supplied evidence and persist one draft without retrieving again."""
        try:
            audit = self._load_qa_module().audit_answer(
                question, draft_answer, top_records
            )
            audit_status = audit.get("status")
            if audit_status not in ALLOWED_AUDIT_STATUSES:
                raise QAServiceError("问答服务暂不可用，请稍后重试。")
            self._save_review_record(
                question=question,
                draft_answer=draft_answer,
                evidence_snapshot=self._snapshot_evidence(top_records),
                audit_status=audit_status,
            )
            # Internal metadata only. The API uses this to decide live delivery;
            # the public response schema does not expose citations or evidence.
            return {
                "generation_status": "generated",
                "audit_status": audit_status,
                "citation_status": audit.get("citation_status"),
            }
        except QAServiceError:
            raise
        except Exception:
            raise QAServiceError("问答服务暂不可用，请稍后重试。") from None

    def process_question(self, question):
        try:
            print("[1] 开始初始化 Pipeline", flush=True)
            pipeline = self._get_pipeline()

            print("[2] Pipeline 就绪，开始检索和重排序", flush=True)
            retrieval_query = normalize_retrieval_query(question)
            top_records = self._qa.retrieve(retrieval_query, pipeline)

            print("[3] 检索完成，开始调用回答模型", flush=True)
            raw_answer = self._qa.generate_answer(
                question, top_records, pipeline[-1]
            )

            print("[4] 模型回答完成，开始质量审核", flush=True)
            audit = self._qa.audit_answer(
                question, raw_answer, top_records
            )

            print("[5] 质量审核完成", flush=True)
            audit_status = audit.get("status")

            if audit_status not in ALLOWED_AUDIT_STATUSES:
                raise QAServiceError("问答服务暂不可用，请稍后重试。")

            self._save_review_record(
                question=question,
                draft_answer=raw_answer,
                evidence_snapshot=self._snapshot_evidence(top_records),
                audit_status=audit_status,
            )

            return {
                "generation_status": "generated",
                "audit_status": audit_status
            }

        except QAServiceError:
            raise
        except Exception:
            raise QAServiceError("问答服务暂不可用，请稍后重试。") from None
