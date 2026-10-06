import json
import unittest
from unittest.mock import Mock

from app.agent.graph import build_intent_router
from app.agent.llm_router import LLMRouterError, ROUTER_SYSTEM_PROMPT
from app.agent.ambiguity import needs_router_clarification
from app.agent.router import classify_intent
from test_agent import client_router_with_content


class SceneRouterV2Tests(unittest.TestCase):
    def test_production_guard_corrects_valid_but_generic_fault_decision(self):
        router = client_router_with_content('{"intent":"troubleshooting","reason":"故障"}')
        self.assertEqual(router.classify("它坏了怎么办？").intent, "clarify")

    def test_guard_never_repairs_invalid_model_contract(self):
        from app.agent.llm_router import LLMRouterError
        for content in ("troubleshooting", '{"intent":"troubleshooting","reason":"故障","extra":1}'):
            with self.subTest(content=content), self.assertRaises(LLMRouterError):
                client_router_with_content(content).classify("它坏了怎么办？")

    def test_guard_matches_only_short_generic_ambiguity(self):
        for question in ("这个怎么弄？", "它坏了怎么办？", "机器人好像有问题，那怎么办？",
                         "不对劲，帮我看看。", "那个用不了。", "怎么解决？"):
            with self.subTest(question=question):
                self.assertTrue(needs_router_clarification(question))

    def test_guard_preserves_explicit_business_requests(self):
        for question in ("机器人一直卡住怎么办？", "无法回充怎么办？", "机器漏水怎么办？",
                         "出现 E4。", "扫着扫着停了。", "主刷不转了怎么办？",
                         "主刷多久清理一次？", "滤网怎么清理？", "帮我生成维护清单",
                         "我家有地毯和宠物，怎么设置？", "如何重置 WiFi 并重新添加设备？",
                         "我要人工客服", "T8 支持哪些功能？"):
            with self.subTest(question=question):
                self.assertFalse(needs_router_clarification(question))

    def test_guard_does_not_override_recent_user_context(self):
        self.assertFalse(needs_router_clarification("这个怎么弄？", [
            {"role": "user", "content": "如何重置 WiFi？"}]))

    def test_guard_does_not_use_generic_assistant_notice_as_user_context(self):
        self.assertTrue(needs_router_clarification("那个用不了。", [
            {"role": "assistant", "content": "需人工审核，未发布。"}]))

    def test_guard_is_not_substring_keyword_routing(self):
        self.assertFalse(needs_router_clarification("这个怎么弄？我想生成维护报告。"))
        self.assertFalse(needs_router_clarification("机器人不对劲，清扫时有异响怎么办？"))

    def test_prompt_distinguishes_routing_ambiguity_from_fault_followup(self):
        for rule in ("Router clarify", "故障排查内部追问", "业务对象、操作目标或具体异常症状",
                     "故障信息还不完整不自动等于 clarify", "即使没有错误码或语音提示仍选择 troubleshooting"):
            self.assertIn(rule, ROUTER_SYSTEM_PROMPT)

    def test_prompt_contains_small_contrastive_pairs(self):
        for example in ("机器人一直卡住怎么办？→ troubleshooting", "机器人有问题怎么办？→ clarify",
                        "扫着扫着停了。→ troubleshooting", "它不正常，怎么办？→ clarify",
                        "主刷不转了怎么办？→ troubleshooting", "这个怎么处理？→ clarify",
                        "出现 E4。→ troubleshooting"):
            self.assertIn(example, ROUTER_SYSTEM_PROMPT)

    def test_valid_structured_decisions_are_not_overridden_by_keywords(self):
        for question, intent in (("机器人一直卡住怎么办？", "troubleshooting"),
                                 ("机器人有问题怎么办？", "clarify"),
                                 ("主刷多久清理一次？", "maintenance")):
            with self.subTest(question=question):
                router = client_router_with_content(json.dumps({"intent": intent, "reason": "分类"}))
                self.assertEqual(build_intent_router(router)({"question": question})["intent"], intent)

    def test_handoff_remains_hard_bypass(self):
        classifier = Mock()
        result = build_intent_router(classifier)({"question": "我要人工客服"})
        self.assertEqual(result["intent"], "human_handoff")
        classifier.classify.assert_not_called()

    def test_router_failure_uses_ambiguity_guard_for_generic_questions(self):
        for question in ("机器好像坏了。", "它坏了怎么办？", "怎么解决？"):
            with self.subTest(question=question):
                classifier = Mock(side_effect=LLMRouterError("sanitized router failure"))
                result = build_intent_router(classifier)({"question": question})
                self.assertEqual(result["intent"], "clarify")
                self.assertIn("模糊", result["route_reason"])
                classifier.classify.assert_called_once()

    def test_router_failure_does_not_guard_explicit_business_requests(self):
        for question in (
            "机器人一直卡住怎么办？",
            "无法回充",
            "出现 E4",
            "主刷多久清理一次？",
            "帮我生成维护清单",
            "我家有地毯和宠物怎么设置？",
            "T8 支持哪些功能？",
        ):
            with self.subTest(question=question):
                classifier = Mock(side_effect=LLMRouterError("sanitized router failure"))
                result = build_intent_router(classifier)({"question": question})
                expected_intent, expected_reason = classify_intent(question)
                self.assertEqual(result, {
                    "intent": expected_intent,
                    "route_reason": expected_reason,
                })

    def test_router_failure_respects_recent_user_context(self):
        classifier = Mock(side_effect=LLMRouterError("sanitized router failure"))
        question = "这个多久一次？"
        context = [{"role": "user", "content": "主刷怎么清理？"}]
        self.assertFalse(needs_router_clarification(question, context))
        result = build_intent_router(classifier)({
            "question": question,
            "conversation_context": context,
        })
        expected_intent, expected_reason = classify_intent(question)
        self.assertEqual(result, {
            "intent": expected_intent,
            "route_reason": expected_reason,
        })
        self.assertNotEqual(result["intent"], "clarify")
        classifier.classify.assert_called_once_with(question, conversation_context=context)

    def test_model_input_still_only_uses_current_and_recent_user_context(self):
        classifier = Mock()
        classifier.classify.return_value = {"intent": "maintenance", "reason": "维护周期"}
        context = [{"role": "user", "content": "主刷怎么清理？"}]
        build_intent_router(classifier)({"question": "那它多久清理一次？",
            "conversation_context": context, "long_term_memory": ["私有事实"], "evidence": [{"text": "证据"}]})
        classifier.classify.assert_called_once_with("那它多久清理一次？", conversation_context=context)
