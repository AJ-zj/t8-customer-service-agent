import json
import unittest
from typing import get_args
from unittest.mock import Mock

from fastapi.testclient import TestClient

from app.agent.llm_router import IntentDecision
from app.agent.react_graph import build_react_agent
from app.agent.scene_policy import BUSINESS_SCENES, SCENE_POLICIES, get_scene_policy
from app.agent.state import SceneIntent
from app.agent.tool_calling import ToolCallingError
from app.main import create_app
from test_react_agent import make_graph, tool_decision
import test_react_agent


class ScenePolicyTests(unittest.TestCase):
    def test_seven_strict_intents_without_knowledge_qa(self):
        self.assertEqual(set(get_args(SceneIntent)), BUSINESS_SCENES | {"clarify", "human_handoff"})
        for intent in get_args(SceneIntent):
            self.assertEqual(IntentDecision(intent=intent, reason="分类").intent, intent)
        for bad in ({"intent": "knowledge_qa", "reason": "旧类别"},
                    {"intent": "maintenance", "reason": "分类", "extra": True}):
            with self.assertRaises(ValueError):
                IntentDecision.model_validate(bad)

    def test_policy_registry_has_only_readonly_tools(self):
        for intent, policy in SCENE_POLICIES.items():
            self.assertIs(get_scene_policy(intent), policy)
            self.assertEqual(policy.allowed_tools, ("search_t8_manual",))
            self.assertTrue(policy.system_instruction)
            self.assertTrue(policy.output_mode)
        self.assertTrue(get_scene_policy("troubleshooting").allow_clarification)
        with self.assertRaises(KeyError):
            get_scene_policy("clarify")

    def test_all_business_scenes_inject_policy_and_finalize_once(self):
        for intent in BUSINESS_SCENES:
            with self.subTest(intent=intent):
                graph, llm, search, qa = make_graph(
                    [tool_decision(), {"content": "说明书依据。[E1]"}], intent=intent)
                state = graph.invoke({"question": "主刷怎么清理？"})
                self.assertEqual(state["intent"], intent)
                self.assertIn(get_scene_policy(intent).system_instruction,
                              llm.complete.call_args_list[0].args[0][0]["content"])
                self.assertEqual(llm.complete.call_args.kwargs["allowed_tools"],
                                 ("search_t8_manual",))
                search.invoke.assert_called_once()
                qa.finalize_draft.assert_called_once()
                qa.process_question.assert_not_called()

    def test_runtime_rejects_disallowed_tool_even_with_mock_llm(self):
        decision = tool_decision()
        decision["assistant_message"]["tool_calls"][0]["function"]["name"] = "t8_knowledge_qa"
        graph, _, search, qa = make_graph([decision], intent="maintenance")
        with self.assertRaises(ToolCallingError):
            graph.invoke({"question": "主刷怎么清理？"})
        search.invoke.assert_not_called()
        qa.finalize_draft.assert_not_called()

    def test_transport_rejects_tool_outside_injected_whitelist(self):
        calls = tool_decision()["assistant_message"]["tool_calls"]
        client = test_react_agent.NativeToolCallClientTests.client_with_calls(calls)
        with self.assertRaises(ToolCallingError):
            client.complete([], allowed_tools=())

    def test_handoff_bypasses_model_and_policy(self):
        router, llm, search, qa = Mock(), Mock(), Mock(), Mock()
        graph = build_react_agent(llm_router=router, tool_llm=llm, search_tool=search, qa_service=qa)
        state = graph.invoke({"question": "我要人工客服。"})
        self.assertEqual(state["intent"], "human_handoff")
        self.assertNotIn("scene_policy", state)
        router.classify.assert_not_called()
        llm.complete.assert_not_called()
        search.invoke.assert_not_called()
        qa.finalize_draft.assert_not_called()

    def test_insufficient_fault_can_route_to_clarify_without_tools(self):
        graph, llm, search, qa = make_graph([], intent="clarify")
        state = graph.invoke({"question": "它不对劲。"})
        self.assertEqual(state["intent"], "clarify")
        self.assertNotIn("scene_policy", state)
        llm.complete.assert_not_called()
        search.invoke.assert_not_called()
        qa.finalize_draft.assert_not_called()

    def test_scene_specific_output_constraints(self):
        self.assertIn("维护项目", get_scene_policy("maintenance_report").system_instruction)
        self.assertIn("不可信用户数据", get_scene_policy("configuration_recommendation").system_instruction)
        self.assertIn("不得制造错误码", get_scene_policy("troubleshooting").system_instruction)
        self.assertIn("不编造耗材寿命", get_scene_policy("maintenance").system_instruction)

    def test_all_scene_api_responses_hide_internal_data(self):
        for intent in BUSINESS_SCENES:
            with self.subTest(intent=intent):
                graph, _, _, _ = make_graph(
                    [tool_decision(), {"content": "内部草稿。[E1]"}], intent=intent)
                response = TestClient(create_app(agent_graph=graph)).post(
                    "/api/agent/chat", json={"question": "主刷怎么清理？"})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["intent"], intent)
                self.assertIsNone(response.json()["answer"])
                self.assertFalse(response.json()["published"])
                for secret in ("内部草稿", "draft_answer", "record_index", "tool_call_id", "evidence"):
                    self.assertNotIn(secret, response.text)

    def test_frozen_eval_distribution(self):
        from collections import Counter
        from pathlib import Path
        rows = json.loads((Path(__file__).resolve().parents[1] /
                           "data/evaluation/scene_router_v1.json").read_text(encoding="utf-8"))["cases"]
        self.assertEqual(len(rows), 70)
        self.assertEqual(Counter(row["expected_intent"] for row in rows), {
            "consult_qa": 12, "troubleshooting": 12, "maintenance": 12,
            "maintenance_report": 10, "configuration_recommendation": 12,
            "clarify": 6, "human_handoff": 6})
