import unittest
from unittest.mock import Mock

from fastapi.testclient import TestClient

from app.agent.graph import build_agent
from app.agent.llm_router import IntentDecision
from app.main import create_app
from app.services.qa_service import QAServiceError


class FakeQAService:
    def __init__(self, error=None):
        self.error = error
        self.questions = []

    def process_question(self, question):
        self.questions.append(question)
        if self.error:
            raise self.error
        # Deliberately include a draft-like value to verify response filtering.
        return {
            "generation_status": "generated",
            "audit_status": "NEEDS_HUMAN_SEMANTIC_REVIEW",
            "draft_answer": "internal secret draft",
        }


class AgentAPIEndpointTests(unittest.TestCase):
    def make_client(self, qa_service, intent="consult_qa"):
        llm_router = Mock()
        llm_router.classify.return_value = IntentDecision(
            intent=intent, reason="测试分类理由"
        )
        agent_graph = build_agent(qa_service, llm_router)
        return TestClient(
            create_app(qa_service=qa_service, agent_graph=agent_graph)
        ), llm_router

    def test_knowledge_qa_calls_agent_and_never_returns_draft(self):
        qa_service = FakeQAService()
        client, llm_router = self.make_client(qa_service)

        response = client.post(
            "/api/agent/chat", json={"question": "主刷怎么清理？"}
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["intent"], "consult_qa")
        self.assertEqual(body["generation_status"], "generated")
        self.assertEqual(body["audit_status"], "NEEDS_HUMAN_SEMANTIC_REVIEW")
        self.assertIsNone(body["answer"])
        self.assertFalse(body["published"])
        self.assertEqual(qa_service.questions, ["主刷怎么清理？"])
        llm_router.classify.assert_called_once_with("主刷怎么清理？")
        self.assertNotIn("draft_answer", body)
        self.assertNotIn("internal secret draft", response.text)

    def test_clarify_does_not_call_rag(self):
        qa_service = Mock()
        client, _llm_router = self.make_client(qa_service, intent="clarify")
        response = client.post(
            "/api/agent/chat", json={"question": "它坏了怎么办？"}
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["intent"], "clarify")
        self.assertEqual(
            body["answer"],
            "请再描述一下具体异常现象，例如是否有语音提示、错误码，或者出现了什么行为异常。",
        )
        self.assertFalse(body["published"])
        qa_service.process_question.assert_not_called()

    def test_human_handoff_does_not_call_rag(self):
        qa_service = Mock()
        client, llm_router = self.make_client(qa_service)
        response = client.post(
            "/api/agent/chat", json={"question": "我要人工客服"}
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["intent"], "human_handoff")
        self.assertEqual(body["answer"], "已为你转入人工处理流程。")
        self.assertFalse(body["published"])
        qa_service.process_question.assert_not_called()
        llm_router.classify.assert_not_called()

    def test_empty_or_missing_question_is_rejected(self):
        client, _llm_router = self.make_client(Mock())

        self.assertEqual(client.post("/api/agent/chat", json={}).status_code, 422)
        self.assertEqual(
            client.post("/api/agent/chat", json={"question": "   "}).status_code,
            422,
        )

    def test_legacy_chat_endpoint_keeps_its_safe_response(self):
        qa_service = FakeQAService()
        client, _llm_router = self.make_client(qa_service)
        response = client.post(
            "/api/chat", json={"question": "主刷怎么清理？"}
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIsNone(body["answer"])
        self.assertFalse(body["published"])
        self.assertNotIn("draft_answer", body)
        self.assertNotIn("internal secret draft", response.text)

    def test_agent_qa_error_is_sanitized(self):
        qa_service = FakeQAService(
            error=RuntimeError("secret token, SQL and upstream response")
        )
        client, _llm_router = self.make_client(qa_service)
        response = client.post(
            "/api/agent/chat", json={"question": "主刷怎么清理？"}
        )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json(), {"detail": "问答服务暂不可用，请稍后重试。"}
        )
        self.assertNotIn("secret", response.text)
        self.assertNotIn("SQL", response.text)


if __name__ == "__main__":
    unittest.main()
