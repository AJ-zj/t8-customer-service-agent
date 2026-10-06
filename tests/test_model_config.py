import json
import unittest
from unittest.mock import patch

from app import config
from app.agent.llm_router import IntentDecision, LLMIntentRouter


class FakeResponse:
    status = 200

    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.payload


class ModelConfigurationTests(unittest.TestCase):
    def test_config_has_separate_router_chat_and_backup_models(self):
        self.assertEqual(config.ROUTER_MODEL, "Qwen/Qwen2.5-7B-Instruct")
        self.assertEqual(config.CHAT_MODEL, "deepseek-ai/DeepSeek-V4-Flash")
        self.assertEqual(config.BACKUP_CHAT_MODEL, "Qwen/Qwen3.8-27B")

    def test_router_default_request_uses_router_model(self):
        captured = {}
        content = json.dumps({"intent": "consult_qa", "reason": "产品问题"})
        payload = json.dumps({
            "choices": [{"message": {"content": content}}]
        }).encode("utf-8")

        def transport(request, timeout):
            captured["body"] = json.loads(request.data.decode("utf-8"))
            captured["timeout"] = timeout
            return FakeResponse(payload)

        decision = LLMIntentRouter(
            api_key_provider=lambda: "unit-test-key",
            transport=transport,
        ).classify("主刷怎么清理？")

        self.assertIsInstance(decision, IntentDecision)
        self.assertEqual(captured["body"]["model"], config.ROUTER_MODEL)
        self.assertNotEqual(captured["body"]["model"], config.CHAT_MODEL)

    def test_grounded_qa_generation_uses_chat_model(self):
        from app.services import grounded_qa as qa
        captured = {}
        payload = json.dumps({
            "choices": [{"message": {"content": "说明书回答 [E1]"}}]
        }).encode("utf-8")

        def fake_urlopen(request, timeout):
            captured["body"] = json.loads(request.data.decode("utf-8"))
            captured["timeout"] = timeout
            return FakeResponse(payload)

        evidence = [("E1", 14, {
            "pdf_page": 15,
            "title": "",
            "section": "日常维护",
            "text": "软胶主刷清理步骤。",
        })]
        with patch.object(qa, "urlopen", side_effect=fake_urlopen):
            answer = qa.generate_answer("主刷怎么清理？", evidence, "unit-test-key")

        self.assertEqual(answer, "说明书回答 [E1]")
        self.assertEqual(captured["body"]["model"], config.CHAT_MODEL)
        self.assertNotEqual(captured["body"]["model"], config.ROUTER_MODEL)
        self.assertEqual(captured["body"]["temperature"], 0.1)
        self.assertEqual(captured["body"]["max_tokens"], qa.MAX_TOKENS)


if __name__ == "__main__":
    unittest.main()
