import json
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from app.agent.llm_router import IntentDecision
from app.agent.react_graph import build_react_agent
from app.agent.tool_calling import ToolCallingError
from app.main import create_app
from app.agent.memory import SessionMemoryStore
from app.services.semantic_grounding_gate import SemanticGroundingOutcome
from app.services.qa_service import GroundedQAService


QUESTION = "主刷怎么清理？"
EVIDENCE = [{
    "evidence_id": "E1",
    "record_index": 14,
    "pdf_page": 15,
    "title": "日常维护",
    "section": "软胶主刷",
    "text": "软胶主刷应拆下清理后装回。",
}]


def make_tool_decision(call_id="call-1"):
    call = {
        "id": call_id,
        "type": "function",
        "function": {
            "name": "search_t8_manual",
            "arguments": json.dumps({"question": "主刷清理方法"}),
        },
    }
    return {
        "tool_calls": [{"id": call_id, "question": "主刷清理方法"}],
        "assistant_message": {"role": "assistant", "content": None, "tool_calls": [call]},
    }


def make_client(intent="consult_qa", decisions=None):
    session = Mock()
    service = GroundedQAService(session_factory=Mock(return_value=session))
    router = Mock()
    router.classify.return_value = IntentDecision(intent=intent, reason="测试分类")
    llm = Mock()
    llm.complete.side_effect = decisions or []
    search = Mock()
    search.name = "search_t8_manual"
    search.invoke.return_value = {"retrieval_query": QUESTION, "evidence": EVIDENCE}
    graph = build_react_agent(
        qa_service=service, llm_router=router, tool_llm=llm, search_tool=search
    )
    gate = Mock()
    gate.evaluate.return_value = SemanticGroundingOutcome(
        status="failed", reason_code="UPSTREAM_ERROR"
    )
    memory = SessionMemoryStore()
    return (
        TestClient(create_app(qa_service=service, agent_graph=graph,
                              semantic_grounding_gate=gate, memory_store=memory)),
        service, session, llm, search, gate, memory,
    )


class AgentV3ApiTests(unittest.TestCase):
    def test_auto_release_returns_answer_keeps_published_false_and_stores_visible_memory(self):
        draft = "请按说明书拆下软胶主刷清理后装回。[E1]"
        client, service, session, llm, search, gate, memory = make_client(
            decisions=[make_tool_decision(), {"content": draft}]
        )
        gate.evaluate.return_value = SemanticGroundingOutcome(
            status="supported", reason_code="SUPPORTED",
            all_claims_supported=True, core_answer_supported=True, core_abstention_quote="",
        )
        response = client.post("/api/agent/chat", json={
            "question": QUESTION, "session_id": "release-test",
        })
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["delivery_status"], "auto_released")
        self.assertEqual(body["answer"], draft)
        self.assertFalse(body["published"])
        self.assertEqual(session.add.call_count, 1)
        self.assertEqual(session.commit.call_count, 1)
        self.assertEqual(memory.get("release-test"), [
            {"role": "user", "content": QUESTION},
            {"role": "assistant", "content": draft},
        ])
        self.assertNotIn(EVIDENCE[0]["text"], response.text)

    def test_review_required_memory_stores_only_visible_notice(self):
        client, service, session, llm, search, gate, memory = make_client(
            decisions=[make_tool_decision(), {"content": "内部草稿：不确定。[E1]"}]
        )
        response = client.post("/api/agent/chat", json={
            "question": QUESTION, "session_id": "review-test",
        })
        self.assertEqual(response.json()["delivery_status"], "review_required")
        self.assertIsNone(response.json()["answer"])
        history = memory.get("review-test")
        self.assertEqual(history[0], {"role": "user", "content": QUESTION})
        self.assertEqual(history[1]["content"], response.json()["safety_notice"])
        self.assertNotIn("内部草稿", history[1]["content"])
        self.assertEqual(session.commit.call_count, 1)

    def test_two_calls_in_one_response_commit_once_and_hide_draft(self):
        first = make_tool_decision("call-A")
        second = make_tool_decision("call-B")
        decision = {
            "tool_calls": first["tool_calls"] + second["tool_calls"],
            "assistant_message": {
                "role": "assistant", "content": None,
                "tool_calls": first["assistant_message"]["tool_calls"]
                + second["assistant_message"]["tool_calls"],
            },
        }
        client, service, session, llm, search, gate, memory = make_client(
            decisions=[decision, {"content": "内部草稿：清理主刷。[E1]"}]
        )
        with patch.object(service, "process_question") as old_process:
            response = client.post("/api/agent/chat", json={"question": QUESTION})
        self.assertEqual(response.status_code, 200)
        search.invoke.assert_called_once_with({"question": QUESTION})
        self.assertFalse(llm.complete.call_args.kwargs["allow_tool"])
        session.add.assert_called_once()
        session.commit.assert_called_once()
        self.assertEqual(session.add.call_args.args[0].evidence, EVIDENCE)
        self.assertIsNone(response.json()["answer"])
        self.assertNotIn("内部草稿", response.text)
        old_process.assert_not_called()

    def test_create_app_defaults_to_v3_graph(self):
        service = Mock()
        with patch("app.main.build_react_agent") as builder:
            create_app(qa_service=service)
        builder.assert_called_once_with(qa_service=service)

    def test_custom_graph_injection_still_works(self):
        custom_graph = Mock()
        custom_graph.invoke.return_value = {
            "intent": "clarify", "answer": "请描述具体异常。"
        }
        with patch("app.main.build_react_agent") as builder:
            response = TestClient(create_app(
                qa_service=Mock(), agent_graph=custom_graph
            )).post("/api/agent/chat", json={"question": QUESTION})
        builder.assert_not_called()
        custom_graph.invoke.assert_called_once_with({"question": QUESTION})
        self.assertEqual(response.json()["answer"], "请描述具体异常。")

    def test_knowledge_qa_uses_tool_audits_and_commits_once_without_leaks(self):
        draft = "内部草稿：清理软胶主刷。[E1]"
        client, service, session, llm, search, gate, memory = make_client(
            decisions=[make_tool_decision(), {"content": draft}]
        )
        with patch.object(service, "process_question") as old_process:
            response = client.post("/api/agent/chat", json={"question": QUESTION})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["intent"], "consult_qa")
        self.assertEqual(body["generation_status"], "generated")
        self.assertEqual(body["audit_status"], "NEEDS_HUMAN_SEMANTIC_REVIEW")
        self.assertIsNone(body["answer"])
        self.assertFalse(body["published"])
        self.assertNotIn("draft_answer", response.text)
        self.assertNotIn("内部草稿", response.text)
        self.assertNotIn(EVIDENCE[0]["text"], response.text)
        self.assertNotIn("tool_call_id", response.text)
        self.assertNotIn("messages", response.text)
        search.invoke.assert_called_once_with({"question": QUESTION})
        self.assertEqual(llm.complete.call_count, 2)
        session.add.assert_called_once()
        session.commit.assert_called_once()
        session.close.assert_called_once()
        saved = session.add.call_args.args[0]
        self.assertEqual(saved.question, QUESTION)
        self.assertEqual(saved.draft_answer, draft)
        self.assertEqual(saved.evidence, EVIDENCE)
        old_process.assert_not_called()

    def test_invalid_e9_citation_is_program_blocked_before_gate(self):
        client, service, session, llm, search, gate, memory = make_client(
            decisions=[make_tool_decision(), {"content": "主刷应每周清理。[E9]"}]
        )
        response = client.post("/api/agent/chat", json={"question": QUESTION})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["audit_status"], "PROGRAM_BLOCKED_INVALID_CITATION")
        self.assertEqual(body["delivery_status"], "review_required")
        self.assertIsNone(body["answer"])
        gate.evaluate.assert_not_called()
        session.add.assert_called_once()
        session.commit.assert_called_once()

    def test_clarify_bypasses_tool_and_database(self):
        client, service, session, llm, search, gate, memory = make_client(intent="clarify")
        response = client.post("/api/agent/chat", json={"question": "它坏了怎么办？"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["intent"], "clarify")
        self.assertIn("具体异常现象", response.json()["answer"])
        llm.complete.assert_not_called()
        search.invoke.assert_not_called()
        session.add.assert_not_called()
        session.commit.assert_not_called()

    def test_handoff_bypasses_tool_and_database(self):
        client, service, session, llm, search, gate, memory = make_client()
        response = client.post("/api/agent/chat", json={"question": "我要人工客服"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["intent"], "human_handoff")
        self.assertIn("人工处理", response.json()["answer"])
        llm.complete.assert_not_called()
        search.invoke.assert_not_called()
        session.add.assert_not_called()
        session.commit.assert_not_called()

    def test_tool_calling_error_is_sanitized_without_partial_record(self):
        client, service, session, llm, search, gate, memory = make_client(
            decisions=[ToolCallingError("secret key and upstream response")]
        )
        response = client.post("/api/agent/chat", json={"question": QUESTION})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {"detail": "问答服务暂不可用，请稍后重试。"})
        self.assertNotIn("secret", response.text)
        self.assertNotIn("upstream", response.text)
        search.invoke.assert_not_called()
        session.add.assert_not_called()
        session.commit.assert_not_called()

    def test_second_tool_call_still_writes_only_after_final_draft(self):
        client, service, session, llm, search, gate, memory = make_client(decisions=[
            make_tool_decision("call-1"), make_tool_decision("call-2"),
            {"content": "请清理软胶主刷。[E1]"},
        ])

        def read_only_tool(_arguments):
            session.add.assert_not_called()
            session.commit.assert_not_called()
            return {"retrieval_query": QUESTION, "evidence": EVIDENCE}

        search.invoke.side_effect = read_only_tool
        with patch.object(service, "process_question") as old_process:
            response = client.post("/api/agent/chat", json={"question": QUESTION})
        self.assertEqual(response.status_code, 200)
        search.invoke.assert_called_once_with({"question": QUESTION})
        self.assertEqual(llm.complete.call_count, 3)
        first_observation = llm.complete.call_args_list[1].args[0][-1]
        second_observation = llm.complete.call_args_list[2].args[0][-1]
        self.assertEqual(first_observation["tool_call_id"], "call-1")
        self.assertEqual(second_observation["tool_call_id"], "call-2")
        self.assertEqual(first_observation["content"], second_observation["content"])
        session.add.assert_called_once()
        session.commit.assert_called_once()
        self.assertEqual(session.add.call_args.args[0].evidence, EVIDENCE)
        old_process.assert_not_called()
        self.assertIsNone(response.json()["answer"])

    def test_second_model_error_after_tool_creates_no_record(self):
        client, service, session, llm, search, gate, memory = make_client(decisions=[
            make_tool_decision(), ToolCallingError("secret second response"),
        ])
        response = client.post("/api/agent/chat", json={"question": QUESTION})
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("secret", response.text)
        search.invoke.assert_called_once()
        session.add.assert_not_called()
        session.commit.assert_not_called()

    def test_tool_error_is_sanitized_without_partial_record(self):
        client, service, session, llm, search, gate, memory = make_client(
            decisions=[make_tool_decision()]
        )
        search.invoke.side_effect = RuntimeError("secret retrieval body")
        response = client.post("/api/agent/chat", json={"question": QUESTION})
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("secret", response.text)
        session.add.assert_not_called()
        session.commit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
