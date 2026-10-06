import unittest
from unittest.mock import Mock

from fastapi.testclient import TestClient

from app.main import create_app
from app.models import QAReviewRecord
from app.services import grounded_qa
from app.services.qa_service import GroundedQAService, QAServiceError


class FakeAPIService:
    def __init__(self, status="NEEDS_HUMAN_SEMANTIC_REVIEW", error=None):
        self.status = status
        self.error = error
        self.questions = []

    def process_question(self, question):
        self.questions.append(question)
        if self.error:
            raise self.error
        return {"generation_status": "generated", "audit_status": self.status}


class FakeQAModule:
    def __init__(self, failure=None, top_records=None):
        self.pipeline_builds = 0
        self.failure = failure
        self.top_records = top_records or [
            ("E1", 0, {
                "pdf_page": 15,
                "title": "日常维护",
                "section": "主刷清理",
                "text": "测试用的本轮检索证据",
            })
        ]
        self.retrieve_count = 0
        self.retrieved_question = None
        self.generated_question = None
        self.audited_question = None
        self.generated_records = None
        self.audited_records = None

    def create_pipeline(self):
        self.pipeline_builds += 1
        return ("cached-pipeline",)

    def retrieve(self, question, pipeline):
        self.retrieve_count += 1
        self.retrieved_question = question
        return self.top_records

    def generate_answer(self, question, top_records, api_key):
        if self.failure:
            raise RuntimeError("secret key and raw upstream body")
        self.generated_question = question
        self.generated_records = top_records
        return "raw generated answer [E1]"

    def audit_answer(self, question, answer, top_records):
        self.audited_question = question
        self.audited_records = top_records
        return {"status": "NEEDS_HUMAN_SEMANTIC_REVIEW", "raw_answer": answer}


def make_session_factory():
    sessions = []

    def new_session():
        session = Mock()
        sessions.append(session)
        return session

    return Mock(side_effect=new_session), sessions


class ChatEndpointTests(unittest.TestCase):
    def test_production_qa_module_is_loaded_from_app(self):
        service = GroundedQAService()
        self.assertIs(service._load_qa_module(), grounded_qa)
        self.assertEqual(grounded_qa.__name__, "app.services.grounded_qa")

    def test_health_and_trimmed_question(self):
        service = FakeAPIService()
        client = TestClient(create_app(service))
        self.assertEqual(client.get("/health").json(), {"status": "ok"})
        response = client.post("/api/chat", json={"question": "  帮我看看  "})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(service.questions, ["帮我看看"])

    def test_missing_empty_and_oversized_question_are_rejected(self):
        client = TestClient(create_app(FakeAPIService()))
        self.assertEqual(client.post("/api/chat", json={}).status_code, 422)
        self.assertEqual(client.post("/api/chat", json={"question": "   "}).status_code, 422)
        self.assertEqual(client.post("/api/chat", json={"question": " x " * 501}).status_code, 422)

    def test_500_characters_after_trimming_are_allowed(self):
        service = FakeAPIService()
        response = TestClient(create_app(service)).post(
            "/api/chat", json={"question": "  " + "x" * 500 + "  "}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(service.questions[0]), 500)

    def test_generated_but_pending_answer_is_never_published(self):
        response = TestClient(create_app(FakeAPIService())).post(
            "/api/chat", json={"question": "问题"}
        )
        body = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(body["generation_status"], "generated")
        self.assertEqual(body["audit_status"], "NEEDS_HUMAN_SEMANTIC_REVIEW")
        self.assertIsNone(body["answer"])
        self.assertFalse(body["published"])
        self.assertNotIn("raw_answer", body)
        self.assertIn("需人工审核", body["safety_notice"])

    def test_program_block_is_returned_without_model_text(self):
        service = FakeAPIService(status="PROGRAM_BLOCKED_UNSUPPORTED_MONEY_CLAIM")
        body = TestClient(create_app(service)).post(
            "/api/chat", json={"question": "价格"}
        ).json()
        self.assertEqual(body["audit_status"], "PROGRAM_BLOCKED_UNSUPPORTED_MONEY_CLAIM")
        self.assertIsNone(body["answer"])
        self.assertFalse(body["published"])
        self.assertNotIn("raw generated answer", str(body))

    def test_model_abstention_is_pending_not_published(self):
        service = FakeAPIService(status="MODEL_ABSTAINED_NEEDS_REVIEW")
        body = TestClient(create_app(service)).post(
            "/api/chat", json={"question": "当前售价"}
        ).json()
        self.assertEqual(body["generation_status"], "generated")
        self.assertEqual(body["audit_status"], "MODEL_ABSTAINED_NEEDS_REVIEW")
        self.assertIsNone(body["answer"])
        self.assertFalse(body["published"])

    def test_service_exception_is_generic_and_does_not_leak_details(self):
        service = FakeAPIService(error=QAServiceError("API_KEY=secret user question raw answer"))
        response = TestClient(create_app(service)).post(
            "/api/chat", json={"question": "private question"}
        )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {"detail": "问答服务暂不可用，请稍后重试。"})
        self.assertNotIn("secret", response.text)
        self.assertNotIn("private question", response.text)

    def test_grounded_qa_pipeline_is_built_once_and_raw_answer_not_returned(self):
        module = FakeQAModule()
        session_factory, sessions = make_session_factory()
        service = GroundedQAService(
            qa_module=module, session_factory=session_factory
        )
        first = service.process_question("问题一")
        second = service.process_question("问题二")
        self.assertEqual(module.pipeline_builds, 1)
        self.assertEqual(session_factory.call_count, 2)
        self.assertEqual(len(sessions), 2)
        self.assertIsNot(sessions[0], sessions[1])
        self.assertEqual(first["audit_status"], "NEEDS_HUMAN_SEMANTIC_REVIEW")
        self.assertEqual(second["generation_status"], "generated")
        self.assertNotIn("raw_answer", first)

    def test_review_record_saves_the_same_evidence_used_by_generation(self):
        evidence = [
            ("E1", 12, {
                "pdf_page": 12,
                "title": "使用说明",
                "section": "WiFi 设置",
                "text": "重置 WiFi 并重新添加设备的说明。",
            }),
            ("E2", 22, {
                "pdf_page": 22,
                "title": "清洁保养",
                "section": "滤网",
                "text": "滤网清理说明。",
            }),
        ]
        module = FakeQAModule(top_records=evidence)
        session_factory, sessions = make_session_factory()
        service = GroundedQAService(
            qa_module=module, session_factory=session_factory
        )

        result = service.process_question("如何重置 WiFi？")

        self.assertEqual(module.retrieve_count, 1)
        self.assertIs(module.generated_records, evidence)
        self.assertIs(module.audited_records, evidence)
        record = sessions[0].add.call_args.args[0]
        self.assertIsInstance(record, QAReviewRecord)
        self.assertEqual(record.question, "如何重置 WiFi？")
        self.assertEqual(record.draft_answer, "raw generated answer [E1]")
        self.assertEqual(record.audit_status, "NEEDS_HUMAN_SEMANTIC_REVIEW")
        self.assertEqual(record.evidence, [
            {
                "evidence_id": "E1",
                "record_index": 12,
                "pdf_page": 12,
                "title": "使用说明",
                "section": "WiFi 设置",
                "text": "重置 WiFi 并重新添加设备的说明。",
            },
            {
                "evidence_id": "E2",
                "record_index": 22,
                "pdf_page": 22,
                "title": "清洁保养",
                "section": "滤网",
                "text": "滤网清理说明。",
            },
        ])
        sessions[0].commit.assert_called_once_with()
        sessions[0].rollback.assert_not_called()
        sessions[0].close.assert_called_once_with()
        self.assertNotIn("draft_answer", result)
        self.assertNotIn("raw_answer", result)

    def test_normalized_query_is_retrieval_only_and_original_is_preserved(self):
        module = FakeQAModule()
        session_factory, sessions = make_session_factory()
        service = GroundedQAService(
            qa_module=module, session_factory=session_factory
        )
        original_question = "请问石头 T8 的主刷怎么清理？"

        service.process_question(original_question)

        self.assertEqual(module.retrieved_question, "的主刷怎么清理？")
        self.assertEqual(module.generated_question, original_question)
        self.assertEqual(module.audited_question, original_question)
        saved = sessions[0].add.call_args.args[0]
        self.assertEqual(saved.question, original_question)

    def test_database_commit_failure_rolls_back_closes_and_is_sanitized(self):
        module = FakeQAModule()
        session = Mock()
        session.commit.side_effect = RuntimeError(
            "password=private database details"
        )
        session_factory = Mock(return_value=session)
        service = GroundedQAService(
            qa_module=module, session_factory=session_factory
        )

        response = TestClient(create_app(service)).post(
            "/api/chat", json={"question": "问题"}
        )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json(), {"detail": "问答服务暂不可用，请稍后重试。"}
        )
        self.assertNotIn("password", response.text)
        session.rollback.assert_called_once_with()
        session.close.assert_called_once_with()

    def test_chat_saves_draft_without_returning_it_to_client(self):
        module = FakeQAModule()
        session_factory, sessions = make_session_factory()
        service = GroundedQAService(
            qa_module=module, session_factory=session_factory
        )
        response = TestClient(create_app(service)).post(
            "/api/chat", json={"question": "清理主刷"}
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIsNone(body["answer"])
        self.assertFalse(body["published"])
        self.assertNotIn("raw generated answer", response.text)
        self.assertNotIn("draft_answer", response.text)
        saved = sessions[0].add.call_args.args[0]
        self.assertEqual(saved.draft_answer, "raw generated answer [E1]")

    def test_upstream_exception_is_sanitized_by_service(self):
        service = GroundedQAService(qa_module=FakeQAModule(failure=True))
        with self.assertRaises(QAServiceError) as raised:
            service.process_question("secret user question")
        self.assertEqual(str(raised.exception), "问答服务暂不可用，请稍后重试。")
        self.assertNotIn("secret", str(raised.exception))


if __name__ == "__main__":
    unittest.main()

