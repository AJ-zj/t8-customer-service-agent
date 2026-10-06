"""Contextual retrieval and per-session request serialization, all with mocks."""

import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock

from fastapi.testclient import TestClient

from app.agent.contextual_query import build_contextual_retrieval_question
from app.agent.memory import SessionMemoryStore
from app.main import create_app
from tests.test_react_agent import make_graph, tool_decision


QUESTION = "主刷怎么清理？"
FOLLOWUP = "那它多久清理一次？"


class ContextualRetrievalTests(unittest.TestCase):
    def test_latest_user_only_and_bounded_length(self):
        context = [
            {"role": "user", "content": "滤网怎么洗？"},
            {"role": "assistant", "content": "回答待审核。"},
            {"role": "user", "content": QUESTION},
            {"role": "assistant", "content": "回答待审核。"},
        ]
        self.assertEqual(
            build_contextual_retrieval_question(FOLLOWUP, context),
            f"{QUESTION} {FOLLOWUP}",
        )
        self.assertEqual(build_contextual_retrieval_question(FOLLOWUP, []), FOLLOWUP)
        self.assertEqual(build_contextual_retrieval_question(FOLLOWUP, [
            {"role": "user", "content": "x" * 500},
        ]), FOLLOWUP)

    def test_model_argument_cannot_replace_contextual_tool_input(self):
        context = [
            {"role": "user", "content": QUESTION},
            {"role": "assistant", "content": "回答待审核。"},
        ]
        for model_argument in ("主刷保养周期", "滤网怎么清洗？"):
            with self.subTest(model_argument=model_argument):
                graph, llm, search, qa = make_graph([
                    tool_decision(question=model_argument),
                    {"content": "请参考本轮证据。[E1]"},
                ])
                result = graph.invoke({
                    "question": FOLLOWUP, "conversation_context": context,
                })
                self.assertEqual(result["intent"], "consult_qa")
                self.assertEqual(llm.complete.call_args_list[0].args[0][1:3], context)
                search.invoke.assert_called_once_with({
                    "question": f"{QUESTION} {FOLLOWUP}",
                })
                qa.finalize_draft.assert_called_once()


class BlockingGraph:
    def __init__(self):
        self.first_entered = Event()
        self.release_first = Event()
        self.second_entered = Event()
        self.other_entered = Event()
        self.calls = []
        self.lock = Lock()

    def invoke(self, state):
        with self.lock:
            self.calls.append(state.copy())
        question = state["question"]
        if question == "first":
            self.first_entered.set()
            if not self.release_first.wait(5):
                raise TimeoutError("test did not release first request")
        elif question == "second":
            self.second_entered.set()
        elif question == "other":
            self.other_entered.set()
        return {"intent": "clarify", "answer": "请描述具体异常。"}


class SessionConcurrencyTests(unittest.TestCase):
    @staticmethod
    def make_client():
        graph = BlockingGraph()
        memory = SessionMemoryStore()
        client = TestClient(create_app(
            qa_service=object(), agent_graph=graph, memory_store=memory,
        ))
        return client, graph, memory

    def test_same_session_second_request_waits_for_first_exchange(self):
        client, graph, memory = self.make_client()
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(client.post, "/api/agent/chat", json={
                "question": "first", "session_id": "same",
            })
            self.assertTrue(graph.first_entered.wait(3))
            second = pool.submit(client.post, "/api/agent/chat", json={
                "question": "second", "session_id": "same",
            })
            try:
                self.assertFalse(graph.second_entered.wait(0.2))
                self.assertEqual(len(graph.calls), 1)
            finally:
                graph.release_first.set()
            self.assertEqual(first.result(timeout=5).status_code, 200)
            self.assertEqual(second.result(timeout=5).status_code, 200)
        self.assertTrue(graph.second_entered.is_set())
        self.assertEqual(graph.calls[1]["conversation_context"], [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "请描述具体异常。"},
        ])
        self.assertEqual(memory._guards, {})

    def test_different_session_runs_while_first_is_blocked(self):
        client, graph, memory = self.make_client()
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(client.post, "/api/agent/chat", json={
                "question": "first", "session_id": "A",
            })
            self.assertTrue(graph.first_entered.wait(3))
            other = pool.submit(client.post, "/api/agent/chat", json={
                "question": "other", "session_id": "B",
            })
            try:
                self.assertTrue(graph.other_entered.wait(3))
                self.assertEqual(other.result(timeout=3).status_code, 200)
            finally:
                graph.release_first.set()
            self.assertEqual(first.result(timeout=5).status_code, 200)
        self.assertEqual(memory._guards, {})
        self.assertEqual(memory.get("B")[0]["content"], "other")

    def test_guard_entries_do_not_accumulate_for_random_ids(self):
        memory = SessionMemoryStore()
        for index in range(120):
            with memory.session_guard(f"id-{index}"):
                memory.append_exchange(f"id-{index}", "问题", "可见回复")
        self.assertEqual(memory._guards, {})
        self.assertEqual(len(memory._sessions), 100)


if __name__ == "__main__":
    unittest.main()
