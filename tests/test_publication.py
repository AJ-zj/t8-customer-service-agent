"""Publication regression using an isolated SQL database, never models/PostgreSQL."""

import os
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from sqlalchemy import CheckConstraint, Column, JSON, MetaData, Table, create_engine, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import sessionmaker

from app.main import create_app
from app.models import QAReviewRecord
from app.services.qa_service import GroundedQAService
from app.services.review_service import ReviewPublicationConflict, ReviewRecordService
from app.services.semantic_grounding_gate import SemanticGroundingOutcome


TOKEN = "publication-unit-token"


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.engine = create_engine(
            "sqlite:///" + self.temp.name.replace("\\", "/") + "/publication.db",
            connect_args={"check_same_thread": False, "timeout": 20},
        )
        # JSON substitutes for PostgreSQL JSONB only in this isolated test database.
        metadata = MetaData()
        columns = [Column(c.name, JSON() if isinstance(c.type, JSONB) else c.type,
                          primary_key=c.primary_key, nullable=c.nullable,
                          server_default=c.server_default)
                   for c in QAReviewRecord.__table__.columns]
        constraints = [CheckConstraint(str(c.sqltext), name=c.name)
                       for c in QAReviewRecord.__table__.constraints
                       if isinstance(c, CheckConstraint)]
        Table("qa_review_records", metadata, *columns, *constraints)
        metadata.create_all(self.engine)
        self.factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.service = ReviewRecordService(self.factory)
        self.auth = patch.dict(os.environ, {"REVIEW_ADMIN_TOKEN": TOKEN})
        self.auth.start()
        self.headers = {"X-Review-Token": TOKEN}
        self.graph = Mock()
        self.client = TestClient(create_app(
            qa_service=Mock(), review_service=self.service, agent_graph=self.graph,
        ))

    def tearDown(self):
        self.client.close()
        self.auth.stop()
        self.engine.dispose()
        self.temp.cleanup()

    def add_record(self, status="pending_review", draft="已核对的答案 [E1]"):
        with self.factory() as session:
            record = QAReviewRecord(question="主刷怎么清理？", draft_answer=draft,
                                    evidence=[{"text": "private evidence"}],
                                    audit_status="NEEDS_HUMAN_SEMANTIC_REVIEW",
                                    review_status=status, review_note="private note")
            session.add(record)
            session.commit()
            return record.id

    def publish(self, rid):
        return self.client.post(f"/internal/reviews/{rid}/publish", headers=self.headers)

    def test_pending_cannot_publish(self):
        self.assertEqual(self.publish(self.add_record()).status_code, 409)

    def test_rejected_cannot_publish(self):
        self.assertEqual(self.publish(self.add_record("rejected")).status_code, 409)

    def test_approved_can_publish_and_timestamp_is_saved(self):
        rid = self.add_record("approved")
        response = self.publish(rid)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(set(response.json()), {
            "record_id", "review_status", "publication_status", "published_at"})
        with self.factory() as session:
            record = session.get(QAReviewRecord, rid)
            self.assertEqual(record.publication_status, "published")
            self.assertIsNotNone(record.published_at)
            self.assertEqual(record.review_note, "private note")
            self.assertEqual(record.evidence, [{"text": "private evidence"}])

    def test_unpublished_states_publicly_look_missing(self):
        for status in ("pending_review", "approved", "rejected"):
            with self.subTest(status=status):
                rid = self.add_record(status)
                response = self.client.get(f"/api/published/{rid}")
                self.assertEqual(response.status_code, 404)
                self.assertEqual(response.json(), {"detail": "已发布答案不存在。"})
                self.assertNotIn("答案", response.json().get("answer", ""))

    def test_published_public_response_is_allowlisted(self):
        rid = self.add_record("approved")
        self.publish(rid)
        response = self.client.get(f"/api/published/{rid}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(set(response.json()), {
            "record_id", "question", "answer", "published", "published_at"})
        self.assertEqual(response.json()["answer"], "已核对的答案 [E1]")
        self.assertTrue(response.json()["published"])
        self.assertNotIn("private", response.text)

    def test_duplicate_publish_conflicts_without_changing_time(self):
        rid = self.add_record("approved")
        first = self.publish(rid).json()
        self.assertEqual(self.publish(rid).status_code, 409)
        self.assertEqual(self.client.get(f"/api/published/{rid}").json()["published_at"],
                         first["published_at"])

    def test_missing_publish_is_404(self):
        self.assertEqual(self.publish(9999).status_code, 404)

    def test_missing_public_read_is_404(self):
        self.assertEqual(self.client.get("/api/published/9999").status_code, 404)

    def test_missing_and_wrong_tokens_are_401(self):
        rid = self.add_record("approved")
        for headers in ({}, {"X-Review-Token": "wrong"}):
            self.assertEqual(self.client.post(
                f"/internal/reviews/{rid}/publish", headers=headers).status_code, 401)
        with self.factory() as session:
            self.assertEqual(session.get(QAReviewRecord, rid).publication_status, "unpublished")

    def test_unconfigured_token_is_401(self):
        rid = self.add_record("approved")
        with patch.dict(os.environ, {"REVIEW_ADMIN_TOKEN": ""}):
            self.assertEqual(self.publish(rid).status_code, 401)

    def test_published_record_cannot_be_rejected(self):
        rid = self.add_record("approved")
        self.publish(rid)
        response = self.client.post(f"/internal/reviews/{rid}/decision",
                                    headers=self.headers,
                                    json={"decision": "rejected", "review_note": "不可撤回"})
        self.assertEqual(response.status_code, 409)
        with self.factory() as session:
            self.assertEqual(session.get(QAReviewRecord, rid).review_status, "approved")

    def test_decision_never_automatically_publishes(self):
        rid = self.add_record()
        response = self.client.post(f"/internal/reviews/{rid}/decision",
                                    headers=self.headers,
                                    json={"decision": "approved", "review_note": "核对完成"})
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["published"])
        self.assertEqual(self.client.get(f"/api/published/{rid}").status_code, 404)

    def test_agent_review_insert_defaults_to_unpublished(self):
        qa = GroundedQAService(session_factory=self.factory)
        qa._save_review_record("测试新生成", "待审核草稿", [],
                               "NEEDS_HUMAN_SEMANTIC_REVIEW")
        with self.factory() as session:
            row = session.scalar(select(QAReviewRecord))
            self.assertEqual(row.review_status, "pending_review")
            self.assertEqual(row.publication_status, "unpublished")
            self.assertIsNone(row.published_at)

    def test_agent_api_safety_response_unchanged(self):
        self.graph.invoke.return_value = {
            "intent": "consult_qa", "answer": "private draft",
            "evidence": [{"text": "private evidence"}],
            "qa_result": {"generation_status": "generated",
                          "audit_status": "NEEDS_HUMAN_SEMANTIC_REVIEW"},
        }
        response = self.client.post("/api/agent/chat", json={"question": "主刷怎么清理？"})
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json()["answer"])
        self.assertFalse(response.json()["published"])
        self.assertNotIn("private", response.text)

    def test_live_auto_release_stays_private_until_explicit_publication(self):
        evidence = [{"evidence_id": "E1", "record_index": 14, "pdf_page": 15,
                     "title": "日常维护", "section": "软胶主刷",
                     "text": "软胶主刷应拆下清理后装回。"}]
        draft = "按说明书拆下软胶主刷清理后装回。[E1]"
        qa = GroundedQAService(session_factory=self.factory)

        class FinalizingGraph:
            def invoke(_self, payload):
                audit = qa.finalize_draft(
                    payload["question"], draft,
                    [("E1", 14, {k: evidence[0][k] for k in ("pdf_page", "title", "section", "text")})],
                )
                return {"intent": "maintenance", "draft_answer": draft,
                        "evidence": evidence, "qa_result": audit}

        gate = Mock()
        gate.evaluate.return_value = SemanticGroundingOutcome(
            status="supported", reason_code="SUPPORTED",
            all_claims_supported=True, core_answer_supported=True, core_abstention_quote="",
        )
        client = TestClient(create_app(
            qa_service=qa, review_service=self.service,
            agent_graph=FinalizingGraph(), semantic_grounding_gate=gate,
        ))
        live = client.post("/api/agent/chat", json={"question": "主刷怎么清理？"})
        self.assertEqual(live.status_code, 200)
        self.assertEqual(live.json()["answer"], draft)
        self.assertEqual(live.json()["delivery_status"], "auto_released")
        self.assertFalse(live.json()["published"])
        with self.factory() as session:
            row = session.scalar(select(QAReviewRecord))
            record_id = row.id
            self.assertEqual(row.review_status, "pending_review")
            self.assertEqual(row.publication_status, "unpublished")
        self.assertEqual(client.get(f"/api/published/{record_id}").status_code, 404)
        approved = client.post(
            f"/internal/reviews/{record_id}/decision", headers=self.headers,
            json={"decision": "approved", "review_note": "核对完成"},
        )
        self.assertEqual(approved.status_code, 200)
        self.assertEqual(client.get(f"/api/published/{record_id}").status_code, 404)
        self.assertEqual(client.post(
            f"/internal/reviews/{record_id}/publish", headers=self.headers,
        ).status_code, 200)
        public = client.get(f"/api/published/{record_id}")
        self.assertEqual(public.status_code, 200)
        self.assertEqual(public.json()["answer"], draft)
        client.close()

    def test_two_real_sql_transactions_only_publish_once(self):
        rid = self.add_record("approved")
        barrier = threading.Barrier(2)

        def submit():
            barrier.wait(timeout=5)
            try:
                return self.service.publish(rid)
            except ReviewPublicationConflict:
                return "conflict"

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: submit(), range(2)))
        self.assertEqual(results.count("conflict"), 1)
        success = next(r for r in results if isinstance(r, dict))
        with self.factory() as session:
            row = session.get(QAReviewRecord, rid)
            self.assertEqual(row.publication_status, "published")
            self.assertEqual(row.published_at, success["published_at"])

    def test_publish_error_rolls_back_closes_and_sanitizes(self):
        for stage in ("execute", "commit"):
            with self.subTest(stage=stage):
                session = Mock()
                session.execute.return_value.mappings.return_value.one_or_none.return_value = {
                    "id": 1, "review_status": "approved", "publication_status": "published",
                    "published_at": __import__("datetime").datetime.now(),
                }
                getattr(session, stage).side_effect = RuntimeError("password=secret SQL private")
                session.close.side_effect = RuntimeError("cleanup secret")
                service = ReviewRecordService(lambda: session)
                client = TestClient(create_app(qa_service=Mock(), agent_graph=Mock(),
                                              review_service=service))
                response = client.post("/internal/reviews/1/publish", headers=self.headers)
                self.assertEqual(response.status_code, 503)
                self.assertNotIn("secret", response.text)
                session.rollback.assert_called_once()
                session.close.assert_called_once()

    def test_committed_publish_survives_close_failure(self):
        session = Mock()
        session.execute.return_value.mappings.return_value.one_or_none.return_value = {
            "id": 1, "review_status": "approved", "publication_status": "published",
            "published_at": datetime(2026, 10, 6, 4, 0, tzinfo=timezone.utc),
        }
        session.close.side_effect = RuntimeError("cleanup secret")
        with TestClient(create_app(qa_service=Mock(), agent_graph=Mock(),
                                  review_service=ReviewRecordService(lambda: session))) as client:
            response = client.post("/internal/reviews/1/publish", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "record_id": 1, "review_status": "approved", "publication_status": "published",
            "published_at": "2026-10-06T04:00:00Z",
        })
        session.commit.assert_called_once()
        session.rollback.assert_not_called()
        session.close.assert_called_once()

    def test_successful_public_read_survives_close_failure(self):
        session = Mock()
        session.execute.return_value.mappings.return_value.one_or_none.return_value = {
            "id": 1, "question": "主刷怎么清理？", "draft_answer": "正式发布的答案",
            "published_at": datetime(2026, 10, 6, 4, 0, tzinfo=timezone.utc),
        }
        session.close.side_effect = RuntimeError("cleanup secret")
        with TestClient(create_app(qa_service=Mock(), agent_graph=Mock(),
                                  review_service=ReviewRecordService(lambda: session))) as client:
            response = client.get("/api/published/1")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "record_id": 1, "question": "主刷怎么清理？", "answer": "正式发布的答案",
            "published": True, "published_at": "2026-10-06T04:00:00Z",
        })
        session.commit.assert_not_called()
        session.close.assert_called_once()

    def test_business_errors_survive_close_failure(self):
        for exists, expected_http in ((None, 404), (1, 409)):
            with self.subTest(expected_http=expected_http):
                session = Mock()
                session.execute.return_value.mappings.return_value.one_or_none.return_value = None
                session.scalar.return_value = exists
                session.close.side_effect = RuntimeError("cleanup secret")
                with TestClient(create_app(qa_service=Mock(), agent_graph=Mock(),
                                          review_service=ReviewRecordService(lambda: session))) as client:
                    response = client.post("/internal/reviews/1/publish", headers=self.headers)
                self.assertEqual(response.status_code, expected_http)
                self.assertNotIn("secret", response.text)
                session.rollback.assert_called_once()
                session.commit.assert_not_called()
                session.close.assert_called_once()

    def test_public_error_closes_and_sanitizes(self):
        session = Mock()
        session.execute.side_effect = RuntimeError("password=secret SELECT private")
        client = TestClient(create_app(qa_service=Mock(), agent_graph=Mock(),
                                      review_service=ReviewRecordService(lambda: session)))
        response = client.get("/api/published/1")
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("secret", response.text)
        session.close.assert_called_once()

    def test_missing_draft_cannot_publish(self):
        for draft in (None, "", "   "):
            with self.subTest(draft=draft):
                self.assertEqual(self.publish(self.add_record("approved", draft)).status_code, 409)

    def test_database_constraint_rejects_published_rejected(self):
        from sqlalchemy import update
        from sqlalchemy.exc import IntegrityError
        rid = self.add_record("approved")
        self.publish(rid)
        with self.factory() as session:
            with self.assertRaises(IntegrityError):
                session.execute(update(QAReviewRecord).where(QAReviewRecord.id == rid)
                                .values(review_status="rejected"))
            session.rollback()


if __name__ == "__main__":
    unittest.main()
