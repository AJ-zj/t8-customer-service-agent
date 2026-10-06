import json
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from app.agent.llm_router import IntentDecision
from app.agent.react_graph import build_react_agent
from app.agent.tool_calling import AGENT_SYSTEM_PROMPT, ManualToolCallingLLM, ToolCallingError
from app.config import CHAT_MODEL
from app.main import create_app
from app.services.grounded_qa import SYSTEM_PROMPT
from app.services.qa_service import GroundedQAService


QUESTION = "主刷怎么清理？"
EVIDENCE = [{
    "evidence_id": "E1", "record_index": 14, "pdf_page": 15,
    "title": "日常维护", "section": "软胶主刷", "text": "拆下软胶主刷并清理。",
}]


def tool_decision(question=QUESTION, call_id="call-1"):
    call = {
        "id": call_id, "type": "function",
        "function": {"name": "search_t8_manual", "arguments": json.dumps({"question": question})},
    }
    return {
        "tool_calls": [{"id": call_id, "question": question}],
        "assistant_message": {"role": "assistant", "content": None, "tool_calls": [call]},
    }


def multi_tool_decision(*questions):
    decisions = [tool_decision(question, f"call-{index}")
                 for index, question in enumerate(questions, 1)]
    return {
        "tool_calls": [item["tool_calls"][0] for item in decisions],
        "assistant_message": {
            "role": "assistant", "content": None,
            "tool_calls": [item["assistant_message"]["tool_calls"][0]
                           for item in decisions],
        },
    }


def fake_router(intent="consult_qa"):
    router = Mock()
    router.classify.return_value = IntentDecision(intent=intent, reason="测试分类")
    return router


def make_graph(decisions, *, intent="consult_qa", qa_service=None):
    intake = Mock()
    intake.assess.return_value = {
        "status": "ready", "symptom_summary": "明确症状", "error_code": None,
        "missing_information": [], "clarifying_question": None,
    }
    llm = Mock()
    llm.complete.side_effect = decisions
    search = Mock()
    search.name = "search_t8_manual"
    search.invoke.return_value = {"retrieval_query": QUESTION, "evidence": EVIDENCE}
    qa = qa_service or Mock()
    if qa_service is None:
        qa.finalize_draft.return_value = {
            "generation_status": "generated", "audit_status": "NEEDS_HUMAN_SEMANTIC_REVIEW",
        }
    graph = build_react_agent(
        qa_service=qa, llm_router=fake_router(intent), tool_llm=llm,
        search_tool=search,
        troubleshooting_intake=intake,
    )
    return graph, llm, search, qa


class ReactAgentTests(unittest.TestCase):
    def test_two_calls_in_one_response_share_one_retrieval_and_snapshot(self):
        draft = "说明书没有明确清理周期。[E1]"
        graph, llm, search, qa = make_graph([
            multi_tool_decision("主刷清理方法", "滤网怎么洗？"), {"content": draft},
        ])
        context = [{"role": "user", "content": QUESTION}]
        state = graph.invoke({"question": "那它多久清理一次？", "conversation_context": context})
        search.invoke.assert_called_once_with({"question": f"{QUESTION} 那它多久清理一次？"})
        self.assertEqual(state["tool_call_count"], 2)
        tool_messages = [item for item in state["messages"] if item["role"] == "tool"]
        self.assertEqual([item["tool_call_id"] for item in tool_messages], ["call-1", "call-2"])
        self.assertEqual(tool_messages[0]["content"], tool_messages[1]["content"])
        self.assertEqual(json.loads(tool_messages[0]["content"]), state["manual_observation"])
        self.assertEqual(state["manual_observation"]["evidence"], state["evidence"])
        self.assertEqual(llm.complete.call_count, 2)
        self.assertFalse(llm.complete.call_args.kwargs["allow_tool"])
        self.assertEqual(state["draft_answer"], draft)
        qa.finalize_draft.assert_called_once()
        self.assertEqual(qa.finalize_draft.call_args.args[2][0][1], 14)
        qa.process_question.assert_not_called()

    def test_three_calls_in_one_response_execute_no_retrieval(self):
        graph, llm, search, qa = make_graph([
            multi_tool_decision("a", "b", "c"),
        ])
        state = graph.invoke({"question": QUESTION})
        self.assertEqual(state.get("tool_call_count", 0), 0)
        self.assertIn("无法确认", state["draft_answer"])
        self.assertFalse(any(item["role"] == "tool" for item in state["messages"]))
        search.invoke.assert_not_called()
        llm.complete.assert_called_once()
        qa.finalize_draft.assert_called_once()

    def test_tool_action_observation_draft_and_one_finalization(self):
        draft = "请拆下软胶主刷并清理。[E1]"
        graph, llm, search, qa = make_graph([
            tool_decision(), {"content": draft},
        ])

        state = graph.invoke({"question": QUESTION})

        self.assertEqual(state["intent"], "consult_qa")
        self.assertEqual(state["tool_call_count"], 1)
        self.assertIsNone(state["answer"])
        self.assertEqual(state["draft_answer"], draft)
        self.assertEqual(state["evidence"], EVIDENCE)
        search.invoke.assert_called_once_with({"question": QUESTION})
        self.assertEqual(llm.complete.call_count, 2)
        observation = llm.complete.call_args.args[0][-1]
        self.assertEqual(observation["role"], "tool")
        self.assertEqual(observation["tool_call_id"], "call-1")
        self.assertEqual(json.loads(observation["content"])["evidence"], EVIDENCE)
        self.assertEqual(qa.finalize_draft.call_count, 1)
        args = qa.finalize_draft.call_args.args
        self.assertEqual(args[0:2], (QUESTION, draft))
        self.assertEqual(args[2][0][0:2], ("E1", 14))
        self.assertEqual(args[2][0][2]["text"], EVIDENCE[0]["text"])
        qa.process_question.assert_not_called()

    def test_two_tool_calls_are_the_limit_and_third_call_falls_back(self):
        graph, llm, search, qa = make_graph([
            tool_decision(call_id="one"), tool_decision(call_id="two"),
            tool_decision(call_id="three"),
        ])

        state = graph.invoke({"question": QUESTION})

        self.assertEqual(state["tool_call_count"], 2)
        search.invoke.assert_called_once_with({"question": QUESTION})
        self.assertEqual(llm.complete.call_count, 3)
        self.assertFalse(llm.complete.call_args.kwargs["allow_tool"])
        first_observation = llm.complete.call_args_list[1].args[0][-1]
        second_observation = llm.complete.call_args_list[2].args[0][-1]
        self.assertEqual(first_observation["tool_call_id"], "one")
        self.assertEqual(second_observation["tool_call_id"], "two")
        self.assertEqual(first_observation["content"], second_observation["content"])
        self.assertEqual(state["manual_observation"]["evidence"], state["evidence"])
        self.assertIn("无法确认", state["draft_answer"])
        qa.finalize_draft.assert_called_once()
        qa.process_question.assert_not_called()

    def test_second_tool_request_replays_first_snapshot_without_retrieval(self):
        graph, llm, search, qa = make_graph([
            tool_decision(call_id="first"), tool_decision(call_id="second"),
            {"content": "按说明清理软胶主刷。[E1]"},
        ])
        changed_evidence = [{**EVIDENCE[0], "record_index": 99, "text": "另一份知识"}]
        search.invoke.side_effect = [
            {"retrieval_query": QUESTION, "evidence": EVIDENCE},
            {"retrieval_query": QUESTION, "evidence": changed_evidence},
        ]

        state = graph.invoke({"question": QUESTION})

        search.invoke.assert_called_once_with({"question": QUESTION})
        self.assertEqual(llm.complete.call_count, 3)
        tool_messages = [message for message in state["messages"] if message["role"] == "tool"]
        self.assertEqual([message["tool_call_id"] for message in tool_messages], ["first", "second"])
        self.assertEqual(tool_messages[0]["content"], tool_messages[1]["content"])
        self.assertEqual(state["manual_observation"]["evidence"], EVIDENCE)
        self.assertEqual(state["evidence"], EVIDENCE)
        self.assertEqual(qa.finalize_draft.call_args.args[2][0][1], 14)
        qa.finalize_draft.assert_called_once()
        qa.process_question.assert_not_called()

    def test_no_manual_search_cannot_become_unsupported_product_answer(self):
        graph, _llm, search, qa = make_graph([{"content": "主刷这样做即可。[E1]"}])
        state = graph.invoke({"question": QUESTION})
        self.assertIn("无法确认", state["draft_answer"])
        self.assertEqual(state.get("evidence", []), [])
        search.invoke.assert_not_called()
        self.assertEqual(qa.finalize_draft.call_args.args[2], [])

    def test_rephrased_tool_argument_still_searches_original_question(self):
        graph, llm, search, qa = make_graph([
            tool_decision(question="主刷清理方法"), {"content": "请清理主刷。[E1]"},
        ])
        state = graph.invoke({"question": QUESTION})
        search.invoke.assert_called_once_with({"question": QUESTION})
        self.assertEqual(state["draft_answer"], "请清理主刷。[E1]")
        self.assertEqual(llm.complete.call_args.args[0][-1]["role"], "tool")
        qa.finalize_draft.assert_called_once()

    def test_unrelated_tool_argument_still_searches_original_question(self):
        graph, _llm, search, qa = make_graph([
            tool_decision(question="滤网怎么清洗？"), {"content": "请清理主刷。[E1]"},
        ])
        state = graph.invoke({"question": QUESTION})
        search.invoke.assert_called_once_with({"question": QUESTION})
        self.assertEqual(state["tool_call_count"], 1)
        qa.finalize_draft.assert_called_once()

    def test_clarify_skips_tool_model_and_review(self):
        graph, llm, search, qa = make_graph([], intent="clarify")
        state = graph.invoke({"question": "它坏了怎么办？"})
        self.assertEqual(state["intent"], "clarify")
        self.assertIn("具体异常现象", state["answer"])
        llm.complete.assert_not_called()
        search.invoke.assert_not_called()
        qa.finalize_draft.assert_not_called()

    def test_handoff_skips_tool_model_and_review(self):
        graph, llm, search, qa = make_graph([])
        state = graph.invoke({"question": "我要人工客服"})
        self.assertEqual(state["intent"], "human_handoff")
        self.assertIn("人工处理", state["answer"])
        llm.complete.assert_not_called()
        search.invoke.assert_not_called()
        qa.finalize_draft.assert_not_called()

    def test_existing_audit_and_one_database_commit_use_tool_evidence(self):
        session = Mock()
        service = GroundedQAService(session_factory=Mock(return_value=session))
        graph, _llm, search, _qa = make_graph([
            tool_decision(), {"content": "请清理软胶主刷。[E1]"},
        ], qa_service=service)
        def read_only_search(_arguments):
            session.add.assert_not_called()
            session.commit.assert_not_called()
            return {"retrieval_query": QUESTION, "evidence": EVIDENCE}

        search.invoke.side_effect = read_only_search
        with patch("app.services.grounded_qa.audit_answer", wraps=__import__(
            "app.services.grounded_qa", fromlist=["audit_answer"]
        ).audit_answer) as audit, patch.object(service, "process_question") as process:
            state = graph.invoke({"question": QUESTION})

        self.assertEqual(state["qa_result"]["audit_status"], "NEEDS_HUMAN_SEMANTIC_REVIEW")
        audit.assert_called_once()
        self.assertEqual(audit.call_args.args[2][0][1], 14)
        session.add.assert_called_once()
        session.commit.assert_called_once()
        session.close.assert_called_once()
        record = session.add.call_args.args[0]
        self.assertEqual(record.evidence, EVIDENCE)
        self.assertEqual(record.draft_answer, "请清理软胶主刷。[E1]")
        process.assert_not_called()
        search.invoke.assert_called_once()

    def test_legacy_agent_api_still_hides_draft_and_evidence(self):
        graph, _, _, _ = make_graph([tool_decision(), {"content": "内部草稿 [E1]"}])
        # This graph is injectable for safety checking; create_app's default remains V2.
        response = TestClient(create_app(agent_graph=graph)).post(
            "/api/agent/chat", json={"question": QUESTION}
        )
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json()["answer"])
        self.assertFalse(response.json()["published"])
        self.assertNotIn("draft_answer", response.text)
        self.assertNotIn("内部草稿", response.text)
        self.assertNotIn("软胶主刷", response.text)


class NativeToolCallClientTests(unittest.TestCase):
    @staticmethod
    def client_with_calls(calls):
        response = json.dumps({"choices": [{"message": {
            "content": None, "tool_calls": calls,
        }}]}).encode()

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return response

        return ManualToolCallingLLM(
            api_key_provider=lambda: "mock-key",
            transport=lambda *_args, **_kwargs: FakeResponse(),
        )

    def test_two_native_calls_are_all_validated_then_normalized(self):
        calls = multi_tool_decision("主刷清理方法", "滤网怎么洗？")["assistant_message"]["tool_calls"]
        decision = self.client_with_calls(calls).complete([{"role": "user", "content": QUESTION}])
        self.assertEqual([call["question"] for call in decision["tool_calls"]],
                         ["主刷清理方法"])
        self.assertEqual(decision["assistant_message"]["tool_calls"], calls[:1])
        self.assertEqual(decision["original_tool_call_count"], 2)
        self.assertEqual(decision["normalized_tool_call_count"], 1)

    def test_invalid_second_call_name_rejects_entire_response(self):
        calls = multi_tool_decision("主刷清理方法", "滤网怎么洗？")["assistant_message"]["tool_calls"]
        calls[1]["function"]["name"] = "t8_knowledge_qa"
        with self.assertRaises(ToolCallingError):
            self.client_with_calls(calls).complete([{"role": "user", "content": QUESTION}])

    def test_invalid_second_call_arguments_rejects_entire_response(self):
        calls = multi_tool_decision("主刷清理方法", "滤网怎么洗？")["assistant_message"]["tool_calls"]
        for arguments in ('{"question":', '{"question": "   "}'):
            with self.subTest(arguments=arguments):
                calls[1]["function"]["arguments"] = arguments
                with self.assertRaises(ToolCallingError):
                    self.client_with_calls(calls).complete([{"role": "user", "content": QUESTION}])

    def test_v3_prompt_uses_tool_evidence_and_preserves_safety_rules(self):
        self.assertIn("search_t8_manual", AGENT_SYSTEM_PROMPT)
        self.assertIn('role="tool"', AGENT_SYSTEM_PROMPT)
        self.assertIn("本轮唯一允许使用", AGENT_SYSTEM_PROMPT)
        self.assertNotIn("只允许依据用户消息提供的本轮证据", AGENT_SYSTEM_PROMPT)
        self.assertIn("根据当前提供的说明书资料无法确认", AGENT_SYSTEM_PROMPT)
        for citation in ("[E1]", "[E2]", "[E3]"):
            self.assertIn(citation, AGENT_SYSTEM_PROMPT)
        self.assertIn("不得发明 Evidence 编号", AGENT_SYSTEM_PROMPT)
        self.assertIn("不输出 Chain of Thought", AGENT_SYSTEM_PROMPT)

    def test_traditional_grounded_qa_prompt_remains_separate(self):
        self.assertIn("只允许依据用户消息提供的本轮证据回答", SYSTEM_PROMPT)
        self.assertNotEqual(AGENT_SYSTEM_PROMPT, SYSTEM_PROMPT)

    def test_openai_tool_call_schema_and_main_model(self):
        captured = {}
        response = json.dumps({"choices": [{"message": {
            "content": None, "tool_calls": [tool_decision()["assistant_message"]["tool_calls"][0]],
        }}]}).encode()

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return response

        def transport(request, timeout):
            captured["body"] = json.loads(request.data.decode())
            return FakeResponse()

        client = ManualToolCallingLLM(api_key_provider=lambda: "mock-key", transport=transport)
        decision = client.complete([{"role": "user", "content": QUESTION}])
        self.assertEqual(decision["tool_calls"][0]["question"], QUESTION)
        self.assertEqual(captured["body"]["model"], CHAT_MODEL)
        self.assertEqual(captured["body"]["tools"][0]["function"]["name"], "search_t8_manual")
        self.assertEqual(captured["body"]["tool_choice"], "auto")
        self.assertEqual(len(captured["body"]["tools"]), 1)

    def test_invalid_tool_response_is_sanitized(self):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return b'{"choices":[{"message":{"tool_calls":[{"id":"x","type":"function","function":{"name":"t8_knowledge_qa","arguments":"{}"}}]}}]}'

        client = ManualToolCallingLLM(api_key_provider=lambda: "secret-key", transport=lambda *_a, **_k: FakeResponse())
        with self.assertRaises(ToolCallingError) as raised:
            client.complete([{"role": "user", "content": QUESTION}])
        self.assertNotIn("secret-key", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
