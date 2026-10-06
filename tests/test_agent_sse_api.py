import json
import time
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from app.agent.llm_router import IntentDecision
from app.agent.memory import SessionMemoryStore
from app.agent.react_graph import build_react_agent
from app.main import create_app
from app.services.answer_release_policy import GateOutcome, SAFE_ABSTENTION_ANSWER
from app.services.qa_service import GroundedQAService


QUESTION = "主刷怎么清理？"
EVIDENCE = [{"evidence_id": "E1", "record_index": 14, "pdf_page": 15,
             "title": "日常维护", "section": "软胶主刷", "text": "证据正文不能出现在 SSE。"}]
DRAFT = "按说明书拆下软胶主刷清理后装回。[E1]"
NOTICE = "模型已生成回答，但语义依据未自动核验；需人工审核，未作为正式答案发布。"


def parse_events(text):
    events = []
    for frame in text.replace("\r\n", "\n").split("\n\n"):
        if not frame or frame.startswith(":"):
            continue
        name = next((line[7:] for line in frame.splitlines() if line.startswith("event: ")), None)
        data = next((line[6:] for line in frame.splitlines() if line.startswith("data: ")), None)
        if name and data:
            events.append((name, json.loads(data)))
    return events


def mock_app(*, state=None, gate_outcome=None, graph=None, memory=None, gate=None):
    graph = graph or Mock()
    graph.invoke.return_value = state or {
        "intent": "maintenance", "draft_answer": DRAFT, "evidence": EVIDENCE,
        "qa_result": {"generation_status": "generated", "audit_status": "NEEDS_HUMAN_SEMANTIC_REVIEW",
                      "citation_status": "VALID"},
    }
    gate = gate or Mock()
    gate.evaluate.return_value = gate_outcome or GateOutcome(
        "supported", "SUPPORTED", True, True, ""
    )
    memory = memory or SessionMemoryStore()
    return TestClient(create_app(qa_service=Mock(), agent_graph=graph,
                                 semantic_grounding_gate=gate, memory_store=memory)), graph, gate, memory


class AgentSseApiTests(unittest.TestCase):
    def test_stream_content_type_and_status_order(self):
        client, _, _, _ = mock_app()
        response = client.post("/api/agent/chat/stream", json={"question": QUESTION})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("text/event-stream"))
        names = [name for name, _ in parse_events(response.text)]
        self.assertEqual(names[:2], ["status", "status"])
        self.assertEqual(names[2], "answer_start")
        self.assertEqual(names[-1], "done")
        stages = [data.get("stage") for name, data in parse_events(response.text) if name == "status"]
        self.assertEqual(stages, ["received", "processing"])

    def test_auto_release_delta_exactly_matches_shared_json_orchestration(self):
        json_client, _, _, _ = mock_app()
        sse_client, graph, _, _ = mock_app()
        json_response = json_client.post("/api/agent/chat", json={"question": QUESTION}).json()
        sse_response = sse_client.post("/api/agent/chat/stream", json={"question": QUESTION})
        events = parse_events(sse_response.text)
        deltas = "".join(data["text"] for name, data in events if name == "answer_delta")
        self.assertEqual(deltas, json_response["answer"])
        self.assertEqual(json_response["delivery_status"], "auto_released")
        done = next(data for name, data in events if name == "done")
        self.assertNotIn("answer", done)
        self.assertFalse(done["published"])
        self.assertEqual(graph.invoke.call_count, 1)

    def test_safe_abstention_streams_only_the_fixed_answer(self):
        client, _, gate, _ = mock_app()
        response = client.post("/api/agent/chat/stream", json={"question": "我家机器人现在电量是多少？"})
        events = parse_events(response.text)
        deltas = "".join(data["text"] for name, data in events if name == "answer_delta")
        self.assertEqual(deltas, SAFE_ABSTENTION_ANSWER)
        self.assertEqual(next(data["delivery_status"] for name, data in events if name == "answer_start"),
                         "safe_abstention")
        gate.evaluate.assert_not_called()

    def test_review_required_has_notice_and_no_answer_delta_or_private_payload(self):
        memory = SessionMemoryStore()
        client, _, _, _ = mock_app(gate_outcome=GateOutcome("failed", "SECRET_GATE_REASON"), memory=memory)
        response = client.post("/api/agent/chat/stream", json={"question": QUESTION, "session_id": "review-sse"})
        events = parse_events(response.text)
        self.assertIn("review_required", [name for name, _ in events])
        review = next(data for name, data in events if name == "review_required")
        self.assertEqual(review["message"], NOTICE)
        self.assertNotIn("answer_delta", [name for name, _ in events])
        for forbidden in ("evidence", "draft_answer", "record_index", "SECRET_GATE_REASON", EVIDENCE[0]["text"]):
            self.assertNotIn(forbidden, response.text)
        self.assertEqual(memory.get("review-sse"), [
            {"role": "user", "content": QUESTION}, {"role": "assistant", "content": NOTICE},
        ])

    def test_clarification_streams_the_existing_visible_message(self):
        text = "请补充具体异常现象。"
        client, _, _, _ = mock_app(state={"intent": "clarify", "answer": text})
        response = client.post("/api/agent/chat/stream", json={"question": "它坏了怎么办？"})
        events = parse_events(response.text)
        self.assertEqual("".join(d["text"] for n, d in events if n == "answer_delta"), text)
        self.assertEqual(next(d["delivery_status"] for n, d in events if n == "answer_start"), "clarification")

    def test_handoff_streams_the_graph_message(self):
        text = "已为你转入人工处理流程。"
        client, _, _, _ = mock_app(state={"intent": "human_handoff", "answer": text})
        response = client.post("/api/agent/chat/stream", json={"question": "我要人工客服"})
        events = parse_events(response.text)
        self.assertEqual("".join(d["text"] for n, d in events if n == "answer_delta"), text)
        self.assertEqual(next(d["delivery_status"] for n, d in events if n == "answer_start"), "human_handoff")

    def test_worker_error_becomes_sanitized_sse_error_after_http_start(self):
        graph = Mock()
        graph.invoke.side_effect = RuntimeError("API key, DSN, traceback and upstream response")
        client, _, _, _ = mock_app(graph=graph)
        response = client.post("/api/agent/chat/stream", json={"question": QUESTION})
        self.assertEqual(response.status_code, 200)
        events = parse_events(response.text)
        self.assertEqual(events[-1], ("error", {"code": "SERVICE_UNAVAILABLE", "message": "问答服务暂不可用，请稍后重试。"}))
        for secret in ("API key", "DSN", "traceback", "upstream response"):
            self.assertNotIn(secret, response.text)

    def test_stream_validation_happens_before_sse_response(self):
        client, graph, _, _ = mock_app()
        response = client.post("/api/agent/chat/stream", json={"question": "问题", "session_id": "bad id"})
        self.assertEqual(response.status_code, 422)
        self.assertFalse(response.headers.get("content-type", "").startswith("text/event-stream"))
        graph.invoke.assert_not_called()

    def test_stream_never_emits_evidence_draft_or_gate_metadata(self):
        client, _, _, _ = mock_app()
        response = client.post("/api/agent/chat/stream", json={"question": QUESTION})
        text = response.text
        for forbidden in ("draft_answer", "evidence", "record_index", "pdf_page", "tool_call_id",
                          "messages", "core_answer_supported", "unsupported_claims"):
            self.assertNotIn(forbidden, text)
        done = next(data for name, data in parse_events(text) if name == "done")
        self.assertNotIn("answer", done)

    def test_long_operation_sends_heartbeats_without_user_payload(self):
        graph = Mock()
        def slow_invoke(_):
            time.sleep(0.04)
            return {"intent": "clarify", "answer": "请补充信息。"}
        graph.invoke.side_effect = slow_invoke
        client, _, _, _ = mock_app(graph=graph)
        with patch("app.main.SSE_HEARTBEAT_SECONDS", 0.01):
            response = client.post("/api/agent/chat/stream", json={"question": QUESTION})
        self.assertIn(": ping\n\n", response.text)
        self.assertNotIn(QUESTION, response.text)

    def test_streaming_knowledge_request_commits_once_and_stores_only_final_visible_answer(self):
        session = Mock()
        qa = GroundedQAService(session_factory=Mock(return_value=session))
        router = Mock()
        router.classify.return_value = IntentDecision(intent="maintenance", reason="test")
        tool_llm = Mock()
        tool_llm.complete.side_effect = [
            {"tool_calls": [{"id": "id-1", "question": QUESTION}],
             "assistant_message": {"role": "assistant", "content": None, "tool_calls": [{
                 "id": "id-1", "type": "function", "function": {"name": "search_t8_manual",
                 "arguments": json.dumps({"question": QUESTION})}}]}},
            {"content": DRAFT},
        ]
        search = Mock(); search.name = "search_t8_manual"
        search.invoke.return_value = {"retrieval_query": QUESTION, "evidence": EVIDENCE}
        graph = build_react_agent(qa_service=qa, llm_router=router, tool_llm=tool_llm, search_tool=search)
        gate = Mock(); gate.evaluate.return_value = GateOutcome("supported", "SUPPORTED", True, True, "")
        memory = SessionMemoryStore()
        client = TestClient(create_app(qa_service=qa, agent_graph=graph,
                                      semantic_grounding_gate=gate, memory_store=memory))
        response = client.post("/api/agent/chat/stream", json={"question": QUESTION, "session_id": "sse-memory"})
        self.assertEqual(response.status_code, 200)
        search.invoke.assert_called_once()
        session.add.assert_called_once()
        session.commit.assert_called_once()
        events = parse_events(response.text)
        public_answer = "".join(d["text"] for n, d in events if n == "answer_delta")
        self.assertEqual(public_answer, DRAFT)
        self.assertEqual(memory.get("sse-memory"), [
            {"role": "user", "content": QUESTION}, {"role": "assistant", "content": DRAFT},
        ])


if __name__ == "__main__":
    unittest.main()
