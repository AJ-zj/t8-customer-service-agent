"""Long-term storage stays enabled while automatic injection defaults off."""

import os
from pathlib import Path
import runpy
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from app.agent.memory import SessionMemoryStore
from app.main import create_app
from app.memory.runtime import LongTermMemoryRuntime
from app.memory.schemas import ExtractedMemoryFact, MemoryExtractionResult
from app.memory.service import MemoryFactService
from tests.test_long_term_memory_integration import knowledge_state
from tests.test_react_agent import make_graph, tool_decision


class LongTermMemoryFeatureFlagTests(unittest.TestCase):
    def test_configuration_defaults_off_and_only_explicit_true_enables(self):
        config_path = Path(__file__).resolve().parents[1] / "app" / "config.py"
        values = [(None, False), ("", False), ("false", False), ("1", False),
                  ("yes", False), ("invalid", False), ("true", True),
                  ("  TrUe  ", True)]
        for value, expected in values:
            with self.subTest(value=value), patch.dict(os.environ, {}, clear=True), \
                    patch("dotenv.load_dotenv"):
                if value is not None:
                    os.environ["ENABLE_LONG_TERM_MEMORY_INJECTION"] = value
                configuration = runpy.run_path(str(config_path))
                self.assertIs(configuration["ENABLE_LONG_TERM_MEMORY_INJECTION"], expected)

    @patch("app.main.ENABLE_LONG_TERM_MEMORY_INJECTION", False)
    def test_disabled_injection_still_extracts_and_saves_original_filtered_fact(self):
        question = "我家有地毯，主刷怎么清理？"
        extractor = Mock()
        extractor.extract_candidates.return_value = MemoryExtractionResult(facts=[
            ExtractedMemoryFact(fact_text="用户有地毯还养猫",
                category="home_environment", source_quote="我家有地毯"),
        ])
        query_embedding = Mock()
        session_factory = Mock()
        fact_service = MemoryFactService(session_factory=session_factory,
                                         embedding_provider=query_embedding)
        runtime = LongTermMemoryRuntime(fact_service=fact_service, extractor=extractor)
        graph = Mock()
        graph.invoke.return_value = knowledge_state()
        with patch.object(fact_service, "save_fact") as save_fact, \
                patch.object(fact_service, "search_facts", wraps=fact_service.search_facts) as search:
            client = TestClient(create_app(agent_graph=graph, qa_service=Mock(),
                long_term_memory_runtime=runtime))
            response = client.post("/api/agent/chat", json={
                "question": question, "memory_owner_id": "owner_a"})
        self.assertEqual(response.status_code, 200)
        graph.invoke.assert_called_once_with({"question": question})
        extractor.extract_candidates.assert_called_once_with(question)
        save_fact.assert_called_once_with("owner_a", "我家有地毯", fact_category="home_environment")
        search.assert_not_called()
        query_embedding.assert_not_called()
        session_factory.assert_not_called()
        self.assertIsNone(response.json()["answer"])
        self.assertFalse(response.json()["published"])

    @patch("app.main.ENABLE_LONG_TERM_MEMORY_INJECTION", False)
    def test_disabled_injection_never_reaches_actual_reasoner_messages(self):
        secret_fact = "我家隔壁房间铺了木地板"
        runtime = Mock()
        runtime.retrieve_for_agent.return_value = [secret_fact]
        graph, llm, search, qa = make_graph([
            tool_decision(), {"content": "内部未审核草稿。[E1]"},
        ])
        client = TestClient(create_app(agent_graph=graph, qa_service=Mock(),
                                      long_term_memory_runtime=runtime))
        response = client.post("/api/agent/chat", json={
            "question": "主刷怎么清理？", "memory_owner_id": "owner_a"})
        self.assertEqual(response.status_code, 200)
        messages = llm.complete.call_args_list[0].args[0]
        self.assertEqual([item["role"] for item in messages], ["system", "user"])
        self.assertFalse(any(secret_fact in item["content"] for item in messages))
        self.assertFalse(any("<user_long_term_memory>" in item["content"]
                             for item in messages if item["role"] != "system"))
        runtime.retrieve_for_agent.assert_not_called()
        runtime.process_successful_user_message.assert_called_once_with(
            "owner_a", "主刷怎么清理？")
        search.invoke.assert_called_once_with({"question": "主刷怎么清理？"})
        qa.finalize_draft.assert_called_once()
        for private_value in (secret_fact, "内部未审核草稿", "draft_answer", "record_index"):
            self.assertNotIn(private_value, response.text)
        self.assertIsNone(response.json()["answer"])
        self.assertFalse(response.json()["published"])

    @patch("app.main.ENABLE_LONG_TERM_MEMORY_INJECTION", False)
    def test_disabled_injection_preserves_short_term_context_and_visible_messages(self):
        runtime = Mock()
        runtime.retrieve_for_agent.return_value = ["我家有地毯"]
        graph = Mock()
        graph.invoke.return_value = {**knowledge_state(),
            "draft_answer": "禁止进入短期记忆的内部草稿", "evidence": [{"text": "内部证据"}]}
        store = SessionMemoryStore()
        client = TestClient(create_app(agent_graph=graph, qa_service=Mock(),
                memory_store=store, long_term_memory_runtime=runtime))
        first = client.post("/api/agent/chat", json={
            "question": "主刷怎么清理？", "session_id": "short_a", "memory_owner_id": "owner_a"})
        second = client.post("/api/agent/chat", json={
            "question": "那它多久清理一次？", "session_id": "short_a", "memory_owner_id": "owner_a"})
        self.assertEqual((first.status_code, second.status_code), (200, 200))
        expected_first = [{"role": "user", "content": "主刷怎么清理？"},
                          {"role": "assistant", "content": first.json()["safety_notice"]}]
        self.assertEqual(graph.invoke.call_args_list[1].args[0], {
            "question": "那它多久清理一次？", "conversation_context": expected_first})
        self.assertEqual(store.get("short_a"), expected_first + [
            {"role": "user", "content": "那它多久清理一次？"},
            {"role": "assistant", "content": second.json()["safety_notice"]}])
        self.assertEqual(store.get("short_b"), [])
        runtime.retrieve_for_agent.assert_not_called()
        self.assertEqual(runtime.process_successful_user_message.call_count, 2)

    def test_no_owner_skips_runtime_creation_for_both_flag_values(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled), \
                    patch("app.main.ENABLE_LONG_TERM_MEMORY_INJECTION", enabled), \
                    patch("app.main.LongTermMemoryRuntime") as runtime_factory:
                graph = Mock()
                graph.invoke.return_value = knowledge_state()
                client = TestClient(create_app(agent_graph=graph, qa_service=Mock()))
                response = client.post("/api/agent/chat", json={"question": "主刷怎么清理？"})
                self.assertEqual(response.status_code, 200)
                graph.invoke.assert_called_once_with({"question": "主刷怎么清理？"})
                runtime_factory.assert_not_called()

    def test_agent_failure_does_not_extract_or_save_for_both_flag_values(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled), \
                    patch("app.main.ENABLE_LONG_TERM_MEMORY_INJECTION", enabled):
                graph = Mock()
                graph.invoke.side_effect = RuntimeError("internal secret")
                runtime = Mock()
                runtime.retrieve_for_agent.return_value = []
                client = TestClient(create_app(agent_graph=graph, qa_service=Mock(),
                                              long_term_memory_runtime=runtime))
                response = client.post("/api/agent/chat", json={
                    "question": "我家有地毯", "memory_owner_id": "owner_a"})
                self.assertEqual(response.status_code, 503)
                self.assertNotIn("internal secret", response.text)
                runtime.process_successful_user_message.assert_not_called()
                if not enabled:
                    runtime.retrieve_for_agent.assert_not_called()

    @patch("app.main.ENABLE_LONG_TERM_MEMORY_INJECTION", False)
    def test_disabled_injection_storage_failures_still_degrade_safely(self):
        for failure_point in ("extract_candidates", "save_fact"):
            with self.subTest(failure_point=failure_point):
                extractor = Mock()
                extractor.extract_candidates.return_value = MemoryExtractionResult(facts=[
                    ExtractedMemoryFact(fact_text="用户家中有地毯", category="home_environment",
                                        source_quote="我家有地毯"),
                ])
                facts = Mock()
                target = extractor if failure_point == "extract_candidates" else facts
                getattr(target, failure_point).side_effect = RuntimeError("storage secret")
                runtime = LongTermMemoryRuntime(fact_service=facts, extractor=extractor)
                graph = Mock()
                graph.invoke.return_value = knowledge_state()
                client = TestClient(create_app(agent_graph=graph, qa_service=Mock(),
                                              long_term_memory_runtime=runtime))
                response = client.post("/api/agent/chat", json={
                    "question": "我家有地毯", "memory_owner_id": "owner_a"})
                self.assertEqual(response.status_code, 200)
                self.assertIsNone(response.json()["answer"])
                self.assertFalse(response.json()["published"])
                self.assertNotIn("storage secret", response.text)
                facts.search_facts.assert_not_called()


if __name__ == "__main__":
    unittest.main()
