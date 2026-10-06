import os
import threading
import unittest
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from sqlalchemy.sql.operators import desc_op

from app.main import create_app
from app.services.review_service import (
    ReviewDecisionConflict,
    ReviewRecordService,
)


TEST_TOKEN = "unit-test-review-token"
EVIDENCE = [
    {
        "evidence_id": "E1",
        "record_index": 14,
        "pdf_page": 15,
        "title": "日常维护",
        "section": "主刷清理",
        "text": "证据正文不应出现在列表中。",
    }
]


def make_record(record_id=5):
    return type("ReviewRecordStub", (), {
        "id": record_id,
        "question": "主刷缠了头发，怎么拆下来清理？",
        "draft_answer": "仅供审核的内部草稿。",
        "audit_status": "NEEDS_HUMAN_SEMANTIC_REVIEW",
        "review_status": "pending_review",
        "evidence": EVIDENCE,
        "review_note": None,
        "created_at": datetime(2026, 10, 5, 8, 30, tzinfo=timezone.utc),
        "reviewed_at": None,
    })()


def make_review_service(rows=None, detail=None, query_error=None):
    session = Mock()
    session.scalars.return_value.all.return_value = rows or []
    session.get.return_value = detail
    if query_error:
        session.scalars.side_effect = query_error
        session.get.side_effect = query_error
    factory = Mock(return_value=session)
    return ReviewRecordService(factory), factory, session


def make_decision_service(updated=None, exists=5, execute_error=None):
    session = Mock()
    result = Mock()
    result.mappings.return_value.one_or_none.return_value = updated
    session.execute.return_value = result
    session.scalar.return_value = exists
    if execute_error:
        session.execute.side_effect = execute_error
    factory = Mock(return_value=session)
    return ReviewRecordService(factory), factory, session


class MappingResult:
    def __init__(self, row):
        self.row = row

    def mappings(self):
        return self

    def one_or_none(self):
        return self.row


class AtomicDecisionStore:
    """Mock database applying the pending-state predicate atomically."""

    def __init__(self):
        self.status = "pending_review"
        self.review_note = None
        self.lock = threading.Lock()

    def session(self):
        return AtomicDecisionSession(self)


class AtomicDecisionSession:
    def __init__(self, store):
        self.store = store

    def execute(self, statement):
        values = {
            getattr(key, "key", key): getattr(value, "value", value)
            for key, value in statement._values.items()
        }
        clauses = statement._where_criteria
        self.has_pending_guard = any(
            getattr(getattr(clause, "right", None), "value", None)
            == "pending_review"
            for clause in clauses
        )
        if not self.has_pending_guard:
            raise AssertionError("conditional pending_review update is required")

        with self.store.lock:
            if self.store.status != "pending_review":
                return MappingResult(None)
            self.store.status = values["review_status"]
            self.store.review_note = values["review_note"]
            return MappingResult({
                "id": 5,
                "review_status": self.store.status,
                "review_note": self.store.review_note,
                "reviewed_at": datetime.now(timezone.utc),
            })

    def scalar(self, statement):
        return 5

    def commit(self):
        pass

    def rollback(self):
        pass

    def close(self):
        pass


class InternalReviewApiTests(unittest.TestCase):
    def client_for(self, review_service):
        return TestClient(create_app(review_service=review_service))

    def test_valid_token_lists_pending_summaries_only(self):
        service, factory, session = make_review_service(rows=[make_record()])
        with patch.dict(os.environ, {"REVIEW_ADMIN_TOKEN": TEST_TOKEN}):
            response = self.client_for(service).get(
                "/internal/reviews",
                headers={"X-Review-Token": TEST_TOKEN},
            )

        self.assertEqual(response.status_code, 200)
        item = response.json()[0]
        self.assertEqual(set(item), {
            "id", "question", "audit_status", "review_status",
            "created_at", "has_draft", "evidence_count", "publication_status",
        })
        self.assertEqual(item["id"], 5)
        self.assertTrue(item["has_draft"])
        self.assertEqual(item["evidence_count"], 1)
        statement = session.scalars.call_args.args[0]
        self.assertEqual(statement._where_criteria[0].right.value, "pending_review")
        self.assertEqual(statement._limit_clause.value, 20)
        ordering = statement._order_by_clauses
        self.assertEqual(
            [item.element.key for item in ordering], ["created_at", "id"]
        )
        self.assertTrue(all(item.modifier is desc_op for item in ordering))
        factory.assert_called_once_with()
        session.close.assert_called_once_with()
        session.add.assert_not_called()
        session.commit.assert_not_called()

    def test_list_never_returns_draft_or_full_evidence(self):
        service, _, _ = make_review_service(rows=[make_record()])
        with patch.dict(os.environ, {"REVIEW_ADMIN_TOKEN": TEST_TOKEN}):
            response = self.client_for(service).get(
                "/internal/reviews", headers={"X-Review-Token": TEST_TOKEN}
            )

        self.assertEqual(response.status_code, 200)
        item = response.json()[0]
        self.assertNotIn("draft_answer", item)
        self.assertNotIn("evidence", item)
        self.assertNotIn("仅供审核的内部草稿", response.text)
        self.assertNotIn("证据正文不应出现在列表中", response.text)

    def test_valid_token_returns_full_detail_from_database_snapshot(self):
        detail = make_record()
        service, factory, session = make_review_service(detail=detail)
        with patch.dict(os.environ, {"REVIEW_ADMIN_TOKEN": TEST_TOKEN}):
            response = self.client_for(service).get(
                "/internal/reviews/5",
                headers={"X-Review-Token": TEST_TOKEN},
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["draft_answer"], detail.draft_answer)
        self.assertEqual(body["evidence"], EVIDENCE)
        self.assertEqual(body["created_at"], "2026-10-05T08:30:00Z")
        session.get.assert_called_once()
        factory.assert_called_once_with()
        session.close.assert_called_once_with()
        session.add.assert_not_called()
        session.commit.assert_not_called()

    def test_wrong_and_missing_tokens_are_rejected_before_database_access(self):
        service, factory, _ = make_review_service(rows=[make_record()])
        with patch.dict(os.environ, {"REVIEW_ADMIN_TOKEN": TEST_TOKEN}):
            client = self.client_for(service)
            wrong = client.get(
                "/internal/reviews", headers={"X-Review-Token": "wrong-token"}
            )
            missing = client.get("/internal/reviews/5")

        self.assertEqual(wrong.status_code, 401)
        self.assertEqual(missing.status_code, 401)
        factory.assert_not_called()

    def test_unconfigured_server_token_fails_closed(self):
        service, factory, _ = make_review_service(rows=[make_record()])
        with patch.dict(os.environ, {"REVIEW_ADMIN_TOKEN": ""}):
            response = self.client_for(service).get(
                "/internal/reviews/5",
                headers={"X-Review-Token": TEST_TOKEN},
            )

        self.assertEqual(response.status_code, 401)
        factory.assert_not_called()

    def test_missing_record_returns_404(self):
        service, _, session = make_review_service(detail=None)
        with patch.dict(os.environ, {"REVIEW_ADMIN_TOKEN": TEST_TOKEN}):
            response = self.client_for(service).get(
                "/internal/reviews/999",
                headers={"X-Review-Token": TEST_TOKEN},
            )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json(), {"detail": "审核记录不存在。"})
        session.close.assert_called_once_with()

    def test_database_error_closes_session_and_does_not_leak_details(self):
        service, _, session = make_review_service(
            query_error=RuntimeError("password=secret; SELECT * FROM private")
        )
        with patch.dict(os.environ, {"REVIEW_ADMIN_TOKEN": TEST_TOKEN}):
            response = self.client_for(service).get(
                "/internal/reviews",
                headers={"X-Review-Token": TEST_TOKEN},
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json(), {"detail": "审核记录暂时不可用，请稍后重试。"}
        )
        self.assertNotIn("secret", response.text)
        self.assertNotIn("SELECT", response.text)
        session.close.assert_called_once_with()

    def test_limit_is_bounded(self):
        service, factory, _ = make_review_service(rows=[])
        with patch.dict(os.environ, {"REVIEW_ADMIN_TOKEN": TEST_TOKEN}):
            response = self.client_for(service).get(
                "/internal/reviews?limit=51",
                headers={"X-Review-Token": TEST_TOKEN},
            )

        self.assertEqual(response.status_code, 422)
        factory.assert_not_called()

    def test_valid_token_approves_pending_record_and_trims_note(self):
        reviewed_at = datetime(2026, 10, 5, 9, 0, tzinfo=timezone.utc)
        updated = {
            "id": 5,
            "review_status": "approved",
            "review_note": "已核对说明书证据，回答准确。",
            "reviewed_at": reviewed_at,
        }
        service, _, session = make_decision_service(updated=updated)
        with patch.dict(os.environ, {"REVIEW_ADMIN_TOKEN": TEST_TOKEN}):
            response = self.client_for(service).post(
                "/internal/reviews/5/decision",
                headers={"X-Review-Token": TEST_TOKEN},
                json={
                    "decision": "approved",
                    "review_note": "  已核对说明书证据，回答准确。  ",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["review_status"], "approved")
        self.assertEqual(
            response.json()["review_note"], "已核对说明书证据，回答准确。"
        )
        self.assertEqual(response.json()["reviewed_at"], "2026-10-05T09:00:00Z")
        self.assertFalse(response.json()["published"])
        statement = session.execute.call_args.args[0]
        update_values = {
            getattr(key, "key", key): getattr(value, "value", value)
            for key, value in statement._values.items()
        }
        self.assertEqual(
            set(update_values), {"review_status", "review_note", "reviewed_at"}
        )
        self.assertEqual(update_values["review_status"], "approved")
        self.assertEqual(update_values["review_note"], "已核对说明书证据，回答准确。")
        self.assertEqual(str(update_values["reviewed_at"]), "now()")
        self.assertEqual(
            statement._where_criteria[1].right.value, "pending_review"
        )
        session.commit.assert_called_once_with()
        session.rollback.assert_not_called()
        session.close.assert_called_once_with()

    def test_valid_token_rejects_pending_record(self):
        updated = {
            "id": 5,
            "review_status": "rejected",
            "review_note": "证据不足，需人工重写。",
            "reviewed_at": datetime(2026, 10, 5, 9, 0, tzinfo=timezone.utc),
        }
        service, _, session = make_decision_service(updated=updated)
        with patch.dict(os.environ, {"REVIEW_ADMIN_TOKEN": TEST_TOKEN}):
            response = self.client_for(service).post(
                "/internal/reviews/5/decision",
                headers={"X-Review-Token": TEST_TOKEN},
                json={
                    "decision": "rejected",
                    "review_note": "证据不足，需人工重写。",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["review_status"], "rejected")
        self.assertFalse(response.json()["published"])
        session.commit.assert_called_once_with()
        session.close.assert_called_once_with()

    def test_invalid_decision_and_empty_review_note_are_rejected(self):
        service, factory, _ = make_decision_service()
        with patch.dict(os.environ, {"REVIEW_ADMIN_TOKEN": TEST_TOKEN}):
            client = self.client_for(service)
            bad_decision = client.post(
                "/internal/reviews/5/decision",
                headers={"X-Review-Token": TEST_TOKEN},
                json={"decision": "published", "review_note": "原因"},
            )
            empty_note = client.post(
                "/internal/reviews/5/decision",
                headers={"X-Review-Token": TEST_TOKEN},
                json={"decision": "approved", "review_note": "  "},
            )
            oversized_note = client.post(
                "/internal/reviews/5/decision",
                headers={"X-Review-Token": TEST_TOKEN},
                json={"decision": "approved", "review_note": "x" * 1001},
            )

        self.assertEqual(bad_decision.status_code, 422)
        self.assertEqual(empty_note.status_code, 422)
        self.assertEqual(oversized_note.status_code, 422)
        factory.assert_not_called()

    def test_decision_missing_or_wrong_token_is_rejected(self):
        service, factory, _ = make_decision_service(updated={})
        with patch.dict(os.environ, {"REVIEW_ADMIN_TOKEN": TEST_TOKEN}):
            client = self.client_for(service)
            wrong = client.post(
                "/internal/reviews/5/decision",
                headers={"X-Review-Token": "wrong-token"},
                json={"decision": "approved", "review_note": "核验完成"},
            )
            missing = client.post(
                "/internal/reviews/5/decision",
                json={"decision": "approved", "review_note": "核验完成"},
            )

        self.assertEqual(wrong.status_code, 401)
        self.assertEqual(missing.status_code, 401)
        factory.assert_not_called()

    def test_decision_returns_404_when_record_does_not_exist(self):
        service, _, session = make_decision_service(updated=None, exists=None)
        with patch.dict(os.environ, {"REVIEW_ADMIN_TOKEN": TEST_TOKEN}):
            response = self.client_for(service).post(
                "/internal/reviews/999/decision",
                headers={"X-Review-Token": TEST_TOKEN},
                json={"decision": "approved", "review_note": "核验完成"},
            )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json(), {"detail": "审核记录不存在。"})
        session.rollback.assert_called_once_with()
        session.commit.assert_not_called()
        session.close.assert_called_once_with()

    def test_decision_returns_409_when_record_is_already_reviewed(self):
        service, _, session = make_decision_service(updated=None, exists=5)
        with patch.dict(os.environ, {"REVIEW_ADMIN_TOKEN": TEST_TOKEN}):
            response = self.client_for(service).post(
                "/internal/reviews/5/decision",
                headers={"X-Review-Token": TEST_TOKEN},
                json={"decision": "rejected", "review_note": "重复审核"},
            )

        self.assertEqual(response.status_code, 409)
        session.rollback.assert_called_once_with()
        session.commit.assert_not_called()
        session.close.assert_called_once_with()

    def test_concurrent_decisions_only_allow_one_transition(self):
        store = AtomicDecisionStore()
        service = ReviewRecordService(store.session)

        def submit(decision):
            try:
                result = service.decide(5, decision, f"decision={decision}")
                return result["review_status"]
            except ReviewDecisionConflict:
                return "conflict"

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(submit, ["approved", "rejected"]))

        self.assertEqual(results.count("conflict"), 1)
        self.assertEqual(sum(item in {"approved", "rejected"} for item in results), 1)
        self.assertIn(store.status, {"approved", "rejected"})

    def test_decision_database_error_rolls_back_closes_and_is_sanitized(self):
        service, _, session = make_decision_service(
            execute_error=RuntimeError("password=secret; UPDATE private")
        )
        with patch.dict(os.environ, {"REVIEW_ADMIN_TOKEN": TEST_TOKEN}):
            response = self.client_for(service).post(
                "/internal/reviews/5/decision",
                headers={"X-Review-Token": TEST_TOKEN},
                json={"decision": "approved", "review_note": "核验完成"},
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json(), {"detail": "审核记录暂时不可用，请稍后重试。"}
        )
        self.assertNotIn("secret", response.text)
        self.assertNotIn("UPDATE", response.text)
        session.rollback.assert_called_once_with()
        session.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
