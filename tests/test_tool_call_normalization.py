import json
import unittest
from unittest.mock import Mock

from app.agent.react_graph import MAX_MANUAL_TOOL_CALLS, build_react_agent
from app.agent.llm_router import IntentDecision
from app.agent.tool_calling import ManualToolCallingLLM, ToolCallingError
from test_react_agent import multi_tool_decision, EVIDENCE


def client(calls):
    responses = [
        {"content": None, "tool_calls": calls},
        {"content": "维护项目：软胶主刷。当前建议：清理主刷。依据：[E1]。注意事项：按说明书操作。[E1]"},
    ]
    captured = []

    class Response:
        def __init__(self, message):
            self.message = message

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": self.message}]}).encode()

    def transport(request, **_):
        captured.append(json.loads(request.data))
        return Response(responses.pop(0))

    return ManualToolCallingLLM(api_key_provider=lambda: "test-key", transport=transport), captured


def calls(count):
    return multi_tool_decision(*[f"维护项目 {i}" for i in range(count)])["assistant_message"]["tool_calls"]


class ToolCallNormalizationTests(unittest.TestCase):
    def test_single_call_retains_id_arguments_and_protocol(self):
        original = calls(1)
        llm, _ = client(original)
        result = llm.complete([])
        self.assertEqual(result["original_tool_call_count"], 1)
        self.assertEqual(result["normalized_tool_call_count"], 1)
        self.assertEqual(result["assistant_message"]["tool_calls"], original)
        self.assertEqual(result["tool_calls"][0]["question"], "维护项目 0")

    def test_two_calls_normalize_both_internal_and_assistant_lists(self):
        original = calls(2)
        llm, _ = client(original)
        result = llm.complete([])
        self.assertEqual(result["original_tool_call_count"], 2)
        self.assertEqual(len(result["tool_calls"]), 1)
        self.assertEqual(result["assistant_message"]["tool_calls"], original[:1])

    def test_three_calls_normalize_without_increasing_budget(self):
        original = calls(3)
        llm, _ = client(original)
        result = llm.complete([])
        self.assertEqual(result["original_tool_call_count"], 3)
        self.assertEqual(result["normalized_tool_call_count"], 1)
        self.assertEqual(result["assistant_message"]["tool_calls"], original[:1])
        self.assertEqual(MAX_MANUAL_TOOL_CALLS, 2)

    def test_invalid_third_tool_is_not_hidden_by_normalization(self):
        original = calls(3)
        original[2]["function"]["name"] = "t8_knowledge_qa"
        llm, _ = client(original)
        with self.assertRaises(ToolCallingError):
            llm.complete([])

    def test_invalid_third_schema_is_not_hidden_by_normalization(self):
        for args in ('{"question":"   "}', '{"question":null}', '{"question":' ):
            with self.subTest(arguments=args):
                original = calls(3)
                original[2]["function"]["arguments"] = args
                llm, _ = client(original)
                with self.assertRaises(ToolCallingError):
                    llm.complete([])

    def test_invalid_duplicate_id_is_not_hidden(self):
        original = calls(3)
        original[2]["id"] = original[0]["id"]
        llm, _ = client(original)
        with self.assertRaises(ToolCallingError):
            llm.complete([])

    def test_normalized_three_call_response_executes_one_search_and_matching_observation(self):
        llm, captured = client(calls(3))
        router, search, qa = Mock(), Mock(), Mock()
        router.classify.return_value = IntentDecision(intent="maintenance_report", reason="维护报告")
        search.name = "search_t8_manual"
        search.invoke.return_value = {"retrieval_query": "原始维护问题", "evidence": EVIDENCE}
        qa.finalize_draft.return_value = {"generation_status": "generated",
                                        "audit_status": "NEEDS_HUMAN_SEMANTIC_REVIEW"}
        state = build_react_agent(llm_router=router, tool_llm=llm, search_tool=search,
                                  qa_service=qa).invoke({"question": "帮我生成日常维护清单。"})
        search.invoke.assert_called_once_with({"question": "帮我生成日常维护清单。"})
        self.assertEqual(state["tool_call_count"], 1)
        messages = captured[1]["messages"]
        assistant = [m for m in messages if m["role"] == "assistant"][0]
        observations = [m for m in messages if m["role"] == "tool"]
        self.assertEqual(len(assistant["tool_calls"]), 1)
        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0]["tool_call_id"], assistant["tool_calls"][0]["id"])
        self.assertEqual(json.loads(observations[0]["content"])["evidence"], state["evidence"])
        self.assertIn("维护项目", state["draft_answer"])
        qa.finalize_draft.assert_called_once()
        qa.process_question.assert_not_called()
