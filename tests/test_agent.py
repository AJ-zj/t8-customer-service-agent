import json
import unittest
from unittest.mock import Mock

from app.agent.graph import CLARIFY_ANSWER, HANDOFF_ANSWER, build_agent
from app.agent.llm_router import IntentDecision, LLMIntentRouter
from app.agent.router import ALLOWED_INTENTS, classify_intent
from app.config import ROUTER_MODEL


class FakeHTTPResponse:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.payload


def client_router_with_content(content, status=200, transport=None):
    payload = json.dumps({
        "choices": [{"message": {"content": content}}]
    }).encode("utf-8")

    def default_transport(_request, timeout):
        return FakeHTTPResponse(payload, status=status)

    return LLMIntentRouter(
        api_key_provider=lambda: "test-key-never-logged",
        settings_provider=lambda: (
            "https://api.siliconflow.cn/v1/chat/completions",
            ROUTER_MODEL,
        ),
        transport=transport or default_transport,
    )


class AgentGraphTests(unittest.TestCase):
    def test_llm_client_parses_structured_intent_with_existing_chat_settings(self):
        captured = {}
        content = json.dumps(
            {"intent": "consult_qa", "reason": "产品知识问题"},
            ensure_ascii=False,
        )
        response_payload = json.dumps({
            "choices": [{"message": {"content": content}}]
        }).encode("utf-8")

        def transport(request, timeout):
            captured["url"] = request.full_url
            captured["body"] = json.loads(request.data.decode("utf-8"))
            captured["timeout"] = timeout
            return FakeHTTPResponse(response_payload)

        router = LLMIntentRouter(
            api_key_provider=lambda: "test-key",
            transport=transport,
        )

        decision = router.classify("主刷怎么清理？")

        self.assertEqual(decision.intent, "consult_qa")
        self.assertEqual(decision.reason, "产品知识问题")
        self.assertEqual(captured["url"], "https://api.siliconflow.cn/v1/chat/completions")
        self.assertEqual(captured["body"]["model"], ROUTER_MODEL)
        self.assertEqual(
            json.loads(captured["body"]["messages"][1]["content"]),
            {"question": "主刷怎么清理？"},
        )
        self.assertEqual(captured["timeout"], 25.0)

    def test_knowledge_question_routes_to_existing_qa_service(self):
        qa_service = Mock()
        qa_service.process_question.return_value = {
            "generation_status": "generated",
            "audit_status": "NEEDS_HUMAN_SEMANTIC_REVIEW",
        }
        llm_router = Mock()
        llm_router.classify.return_value = IntentDecision(
            intent="consult_qa", reason="产品知识问题"
        )
        agent = build_agent(qa_service, llm_router)

        result = agent.invoke({"question": "主刷怎么清理？"})

        self.assertEqual(result["intent"], "consult_qa")
        self.assertIsNone(result["answer"])
        self.assertEqual(result["qa_result"]["generation_status"], "generated")
        self.assertNotIn("draft_answer", result["qa_result"])
        self.assertEqual(result["route_reason"], "产品知识问题")
        qa_service.process_question.assert_called_once_with("主刷怎么清理？")
        llm_router.classify.assert_called_once_with("主刷怎么清理？")

    def test_vague_failure_routes_to_clarify_without_rag(self):
        qa_service = Mock()
        llm_router = Mock()
        llm_router.classify.return_value = IntentDecision(
            intent="clarify", reason="故障描述信息不足"
        )
        result = build_agent(qa_service, llm_router).invoke(
            {"question": "它坏了怎么办？"}
        )

        self.assertEqual(result["intent"], "clarify")
        self.assertEqual(result["answer"], CLARIFY_ANSWER)
        self.assertEqual(result["route_reason"], "故障描述信息不足")
        qa_service.process_question.assert_not_called()

    def test_explicit_human_request_routes_to_handoff_without_rag(self):
        qa_service = Mock()
        llm_router = Mock()
        result = build_agent(qa_service, llm_router).invoke(
            {"question": "我要人工客服"}
        )

        self.assertEqual(result["intent"], "human_handoff")
        self.assertEqual(result["answer"], HANDOFF_ANSWER)
        qa_service.process_question.assert_not_called()
        llm_router.classify.assert_not_called()

    def test_invalid_intent_falls_back_to_rule_router(self):
        qa_service = Mock()
        qa_service.process_question.return_value = {
            "generation_status": "generated",
            "audit_status": "NEEDS_HUMAN_SEMANTIC_REVIEW",
        }
        llm_router = Mock()
        llm_router.classify.return_value = {
            "intent": "shopping",
            "reason": "非法业务意图",
        }

        result = build_agent(qa_service, llm_router).invoke(
            {"question": "主刷怎么清理？"}
        )

        expected_intent, expected_reason = classify_intent("主刷怎么清理？")
        self.assertEqual(result["intent"], expected_intent)
        self.assertEqual(result["route_reason"], expected_reason)
        qa_service.process_question.assert_called_once_with("主刷怎么清理？")

    def test_llm_exception_falls_back_to_rule_router(self):
        qa_service = Mock()
        llm_router = Mock()
        llm_router.classify.side_effect = TimeoutError("private transport detail")

        result = build_agent(qa_service, llm_router).invoke(
            {"question": "它坏了怎么办？"}
        )

        self.assertEqual(result["intent"], "clarify")
        self.assertIn("模糊", result["route_reason"])
        qa_service.process_question.assert_not_called()

    def test_unparseable_llm_content_falls_back_to_rule_router(self):
        qa_service = Mock()
        router = client_router_with_content("not JSON")

        result = build_agent(qa_service, router).invoke(
            {"question": "它坏了怎么办？"}
        )

        self.assertEqual(result["intent"], "clarify")
        self.assertIn("模糊", result["route_reason"])
        qa_service.process_question.assert_not_called()

    def test_api_timeout_status_empty_and_invalid_schema_fall_back(self):
        qa_service = Mock()
        cases = [
            (
                "timeout",
                lambda _request, **_kwargs: (_ for _ in ()).throw(TimeoutError()),
            ),
            ("non_200", lambda _request, **_kwargs: FakeHTTPResponse(b"{}", 503)),
            (
                "empty_content",
                None,
            ),
            (
                "invalid_schema",
                None,
            ),
        ]
        for name, transport in cases:
            with self.subTest(failure=name):
                if name == "empty_content":
                    router = client_router_with_content("")
                elif name == "invalid_schema":
                    router = client_router_with_content(
                        '{"intent":"shopping","reason":"非法意图"}'
                    )
                else:
                    router = client_router_with_content("", transport=transport)
                result = build_agent(qa_service, router).invoke(
                    {"question": "它坏了怎么办？"}
                )
                self.assertEqual(result["intent"], "clarify")
                self.assertIn("模糊", result["route_reason"])
        qa_service.process_question.assert_not_called()

    def test_intent_decision_schema_only_accepts_allowed_intents(self):
        valid_intents = ("consult_qa", "clarify", "human_handoff")
        for intent in valid_intents:
            with self.subTest(intent=intent):
                decision = IntentDecision(intent=intent, reason="简短分类理由")
                self.assertIn(decision.intent, ALLOWED_INTENTS)

        with self.assertRaises(Exception):
            IntentDecision(intent="shopping", reason="非法意图")

    def test_router_only_emits_supported_intents(self):
        questions = (
            "主刷怎么清理？",
            "它坏了怎么办？",
            "我要人工客服",
            "WiFi 怎么重置？",
        )
        for question in questions:
            with self.subTest(question=question):
                intent, _reason = classify_intent(question)
                self.assertIn(intent, ALLOWED_INTENTS)


if __name__ == "__main__":
    unittest.main()
