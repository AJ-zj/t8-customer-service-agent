import json
import unittest
from unittest.mock import Mock

from fastapi.testclient import TestClient

from app.agent.llm_router import IntentDecision, LLMIntentRouter, ROUTER_SYSTEM_PROMPT
from app.agent.memory import SessionMemoryStore
from app.agent.react_graph import build_react_agent
from app.main import create_app


QUESTION = "主刷怎么清理？"
FOLLOWUP = "那它多久清理一次？"


def knowledge_state():
    return {
        "intent": "consult_qa",
        "answer": None,
        "qa_result": {
            "generation_status": "generated",
            "audit_status": "NEEDS_HUMAN_SEMANTIC_REVIEW",
        },
        "draft_answer": "内部未审核 Draft：主刷每周清理。",
        "manual_observation": {"evidence": [{"text": "内部 Tool Evidence 原文"}]},
        "messages": [{"role": "tool", "content": "内部 Tool 协议消息"}],
    }


class MemoryStoreTests(unittest.TestCase):
    def test_sliding_window_keeps_only_six_recent_visible_messages(self):
        store = SessionMemoryStore()
        for index in range(4):
            store.append_exchange("s1", f"用户{index}", f"可见回复{index}")
        messages = store.get("s1")
        self.assertEqual(len(messages), 6)
        self.assertEqual(messages[0], {"role": "user", "content": "用户1"})
        self.assertEqual(messages[-1], {"role": "assistant", "content": "可见回复3"})
        messages[0]["content"] = "外部修改"
        self.assertEqual(store.get("s1")[0]["content"], "用户1")

    def test_session_limit_evicts_least_recently_used_session(self):
        store = SessionMemoryStore(max_sessions=2)
        store.append_exchange("A", "问题A", "回复A")
        store.append_exchange("B", "问题B", "回复B")
        store.get("A")
        store.append_exchange("C", "问题C", "回复C")
        self.assertEqual(store.get("B"), [])
        self.assertEqual(store.get("A")[0]["content"], "问题A")
        self.assertEqual(store.get("C")[0]["content"], "问题C")


class AgentMemoryApiTests(unittest.TestCase):
    def make_client(self):
        graph = Mock()
        graph.invoke.return_value = knowledge_state()
        store = SessionMemoryStore()
        client = TestClient(create_app(qa_service=Mock(), agent_graph=graph, memory_store=store))
        return client, graph, store

    def test_missing_session_id_keeps_single_turn_graph_input(self):
        client, graph, store = self.make_client()
        store.session_guard = Mock(side_effect=AssertionError("anonymous request took a session lock"))
        response = client.post("/api/agent/chat", json={"question": QUESTION})
        self.assertEqual(response.status_code, 200)
        graph.invoke.assert_called_once_with({"question": QUESTION})
        store.session_guard.assert_not_called()
        self.assertEqual(store.get("any-session"), [])

    def test_same_session_exposes_only_previous_visible_exchange(self):
        client, graph, store = self.make_client()
        client.post("/api/agent/chat", json={"question": QUESTION, "session_id": "abc123"})
        response = client.post("/api/agent/chat", json={"question": FOLLOWUP, "session_id": "abc123"})
        self.assertEqual(response.status_code, 200)
        second_input = graph.invoke.call_args_list[1].args[0]
        self.assertEqual(second_input["question"], FOLLOWUP)
        self.assertEqual(second_input["conversation_context"][0],
                         {"role": "user", "content": QUESTION})
        self.assertIn("人工审核", second_input["conversation_context"][1]["content"])
        self.assertEqual(len(store.get("abc123")), 4)

    def test_sessions_are_isolated_and_anonymous_request_has_no_history(self):
        client, graph, store = self.make_client()
        client.post("/api/agent/chat", json={"question": "A 的问题", "session_id": "A"})
        client.post("/api/agent/chat", json={"question": "B 的问题", "session_id": "B"})
        client.post("/api/agent/chat", json={"question": "A 后续", "session_id": "A"})
        client.post("/api/agent/chat", json={"question": "无会话问题"})
        self.assertNotIn("conversation_context", graph.invoke.call_args_list[1].args[0])
        self.assertEqual(graph.invoke.call_args_list[2].args[0]["conversation_context"][0]["content"], "A 的问题")
        self.assertEqual(graph.invoke.call_args_list[3].args[0], {"question": "无会话问题"})
        self.assertEqual(store.get("B")[0]["content"], "B 的问题")

    def test_internal_draft_tool_observation_and_protocol_never_enter_memory(self):
        client, graph, store = self.make_client()
        response = client.post("/api/agent/chat", json={"question": QUESTION, "session_id": "safe"})
        history = store.get("safe")
        joined = json.dumps(history, ensure_ascii=False)
        self.assertEqual(len(history), 2)
        self.assertNotIn("内部未审核 Draft", joined)
        self.assertNotIn("内部 Tool Evidence", joined)
        self.assertNotIn("内部 Tool 协议", joined)
        self.assertNotIn("tool_call", joined)
        self.assertNotIn("Chain of Thought", joined)
        self.assertIsNone(response.json()["answer"])
        self.assertFalse(response.json()["published"])
        self.assertEqual(history[1]["content"], response.json()["safety_notice"])

    def test_clarify_and_handoff_store_only_visible_messages(self):
        client, graph, store = self.make_client()
        graph.invoke.side_effect = [
            {"intent": "clarify", "answer": "请描述具体异常。"},
            {"intent": "human_handoff", "answer": "已为你转入人工处理流程。"},
        ]
        client.post("/api/agent/chat", json={"question": "它坏了怎么办？", "session_id": "flow"})
        client.post("/api/agent/chat", json={"question": "我要人工客服", "session_id": "flow"})
        history = store.get("flow")
        self.assertEqual([row["content"] for row in history], [
            "它坏了怎么办？", "请描述具体异常。", "我要人工客服", "已为你转入人工处理流程。",
        ])

    def test_invalid_session_ids_return_explicit_validation_error(self):
        client, graph, store = self.make_client()
        for bad in ("", "   ", "has space", "@bad", "x" * 65, 123):
            with self.subTest(value=bad):
                response = client.post("/api/agent/chat", json={
                    "question": QUESTION, "session_id": bad,
                })
                self.assertEqual(response.status_code, 422)
                self.assertIn("session_id", response.json()["detail"])
        graph.invoke.assert_not_called()

    def test_failed_request_does_not_append_partial_history(self):
        client, graph, store = self.make_client()
        graph.invoke.side_effect = RuntimeError("internal failure")
        response = client.post("/api/agent/chat", json={
            "question": QUESTION, "session_id": "failed",
        })
        self.assertEqual(response.status_code, 503)
        self.assertEqual(store.get("failed"), [])


class AgentContextWiringTests(unittest.TestCase):
    def test_router_prompt_distinguishes_resolved_reference_from_missing_fault_details(self):
        self.assertIn("不得仅因为当前问题使用", ROUTER_SYSTEM_PROMPT)
        self.assertIn("代词就判定为 clarify", ROUTER_SYSTEM_PROMPT)
        self.assertIn("用户已经描述具体异常现象", ROUTER_SYSTEM_PROMPT)
        self.assertIn("故障信息还不完整不自动等于 clarify", ROUTER_SYSTEM_PROMPT)
        self.assertIn("历史用户问题只用于意图分类，不是说明书证据", ROUTER_SYSTEM_PROMPT)

    def test_router_and_agent_llm_receive_recent_context_without_real_calls(self):
        context = [
            {"role": "user", "content": QUESTION},
            {"role": "assistant", "content": "回答待人工审核，尚未发布。"},
        ]
        router = Mock()
        router.classify.return_value = IntentDecision(
            intent="consult_qa", reason="近期问题讨论主刷，当前为承接提问。"
        )
        llm = Mock()
        llm.complete.return_value = {"content": "无证据直接回答不安全。"}
        qa = Mock()
        qa.finalize_draft.return_value = {
            "generation_status": "generated", "audit_status": "MODEL_ABSTAINED_NEEDS_REVIEW",
        }
        search = Mock()
        graph = build_react_agent(
            qa_service=qa, llm_router=router, tool_llm=llm, search_tool=search
        )
        result = graph.invoke({"question": FOLLOWUP, "conversation_context": context})
        router.classify.assert_called_once_with(FOLLOWUP, conversation_context=context)
        messages = llm.complete.call_args.args[0]
        self.assertEqual(messages[1:3], context)
        self.assertEqual(messages[-1], {"role": "user", "content": FOLLOWUP})
        self.assertEqual(result["intent"], "consult_qa")
        self.assertIn("无法确认", result["draft_answer"])
        search.invoke.assert_not_called()

    def test_llm_router_payload_uses_latest_user_question_without_safety_notice(self):
        captured = {}

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return b'{"choices":[{"message":{"content":"{\\"intent\\":\\"consult_qa\\",\\"reason\\":\\"context\\"}"}}]}'

        def transport(request, timeout):
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return Response()

        router = LLMIntentRouter(api_key_provider=lambda: "mock-key", transport=transport)
        context = [
            {"role": "user", "content": QUESTION},
            {"role": "assistant", "content": "模型已生成回答，但语义依据未自动核验……"},
        ]
        result = router.classify(FOLLOWUP, conversation_context=context)
        self.assertEqual(result.intent, "consult_qa")
        user_payload = json.loads(captured["body"]["messages"][1]["content"])
        self.assertEqual(user_payload, {
            "current_question": FOLLOWUP,
            "latest_user_question": QUESTION,
            "contextual_question": f"{QUESTION} {FOLLOWUP}",
        })

    def test_llm_router_can_still_choose_clarify_with_vague_history(self):
        captured = {}

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return b'{"choices":[{"message":{"content":"{\\"intent\\":\\"clarify\\",\\"reason\\":\\"information missing\\"}"}}]}'

        def transport(request, timeout):
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return Response()

        router = LLMIntentRouter(api_key_provider=lambda: "mock-key", transport=transport)
        decision = router.classify("那怎么办？", conversation_context=[
            {"role": "user", "content": "机器人好像有问题"},
        ])
        self.assertEqual(decision.intent, "clarify")
        payload = json.loads(captured["body"]["messages"][1]["content"])
        self.assertEqual(payload["latest_user_question"], "机器人好像有问题")

    def test_llm_router_without_history_sends_only_current_question(self):
        captured = {}

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return b'{"choices":[{"message":{"content":"{\\"intent\\":\\"consult_qa\\",\\"reason\\":\\"clear\\"}"}}]}'

        def transport(request, timeout):
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return Response()

        router = LLMIntentRouter(api_key_provider=lambda: "mock-key", transport=transport)
        self.assertEqual(router.classify(QUESTION).intent, "consult_qa")
        self.assertEqual(json.loads(captured["body"]["messages"][1]["content"]), {
            "question": QUESTION,
        })


if __name__ == "__main__":
    unittest.main()
