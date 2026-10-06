"""Source-capability rules and release integration; no model or real DB."""

import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import Mock

from fastapi.testclient import TestClient

from app.main import create_app
from app.services.answer_release_policy import AnswerReleasePolicy, GateOutcome, SAFE_ABSTENTION_ANSWER
from app.services.static_knowledge_boundary import assess_static_knowledge_boundary


EVIDENCE = [{"evidence_id": "E1", "record_index": 3, "pdf_page": 4,
             "text": "电源指示灯白色表示电量≥20%，红色表示电量＜20%。"}]
DRAFT = "电源指示灯白色表示电量≥20%，红色表示电量＜20%。[E1]"
SUPPORTED = GateOutcome("supported", "SUPPORTED", True, True, "")


class StaticKnowledgeBoundaryTests(unittest.TestCase):
    def assert_boundary(self, questions, status):
        for question in questions:
            with self.subTest(question=question):
                self.assertEqual(assess_static_knowledge_boundary(question).status, status)

    def test_battery_values_require_a_device_instance_reading(self):
        self.assert_boundary(("我家机器人现在电量是多少？", "这台机器人当前还剩多少电？",
                              "我的机器人此刻电量百分之几？"), "personal_runtime_unavailable")

    def test_location_requires_live_device_observation(self):
        self.assert_boundary(("机器人现在在哪里？", "我的机器人现在在哪个房间？",
                              "这台设备的当前位置是什么？"), "personal_runtime_unavailable")

    def test_charging_state_requires_live_device_observation(self):
        self.assert_boundary(("它现在是否正在充电？", "机器人是否在充电？",
                              "这台设备充电状态是什么？"), "personal_runtime_unavailable")

    def test_network_state_requires_instance_connectivity_data(self):
        self.assert_boundary(("我家机器人当前有没有联网？", "它现在是否在线？",
                              "它当前是否连接WiFi？"), "personal_runtime_unavailable")

    def test_progress_requires_the_actual_cleaning_task(self):
        self.assert_boundary(("现在清扫到哪了？", "这台机器人当前清扫进度是多少？"),
                             "personal_runtime_unavailable")

    def test_actual_device_state_requires_instance_telemetry(self):
        self.assert_boundary(("它当前是什么状态？", "我的机器人现在故障状态是什么？"),
                             "personal_runtime_unavailable")

    def test_duration_requires_an_actual_run_not_product_endurance(self):
        self.assert_boundary(("昨晚机器人实际清扫了多久？", "上次运行了几分钟？",
                              "我家T8上次实际运行了几小时？"), "personal_runtime_unavailable")

    def test_history_requires_private_device_records(self):
        self.assert_boundary(("我家机器人上一次错误日志是什么？", "我的机器昨晚清扫记录有哪些？"),
                             "personal_runtime_unavailable")

    def test_all_required_method_and_product_negatives_remain_eligible(self):
        self.assert_boundary((
            "怎么看机器人当前电量？", "如何查看剩余电量？", "现在应该怎么清理主刷？",
            "机器人没电了怎么办？", "电量低时指示灯是什么颜色？", "T8 的续航时间是多少？",
            "如何判断机器人是否在充电？", "怎么看上次清扫记录？",
            "如何查看当前清扫状态？", "当前电量低时应该怎么办？",
            "机器人充满电可以运行几小时？",
        ), "within_static_manual")

    def test_time_words_and_interrogatives_alone_do_not_block(self):
        self.assert_boundary(("现在应该选什么模式？", "剩余电量低时指示灯是什么颜色？",
                              "主刷多久清理一次？", "T8支持哪些功能？", "有哪些耗材？"),
                             "within_static_manual")

    def test_runtime_clause_is_not_rescued_by_a_separate_method_clause(self):
        self.assert_boundary(("我家机器人当前电量是多少，主刷怎么清理？",
                              "怎么看清扫记录？我家机器人昨晚清扫了多久？"),
                             "personal_runtime_unavailable")

    def test_explicit_connectors_split_independent_compound_requests(self):
        cases = (
            ("我家机器人现在电量是多少以及怎么看电量状态",
             "personal_runtime_unavailable"),
            ("怎么看电量状态以及电量低了怎么办", "within_static_manual"),
            ("我家机器人现在在哪里，同时怎么看清扫地图",
             "personal_runtime_unavailable"),
            ("怎么看机器人当前电量是多少", "within_static_manual"),
            ("现在怎么清理主刷", "within_static_manual"),
            ("我家机器人当前还剩多少电", "personal_runtime_unavailable"),
            ("昨晚实际清扫多久，另外怎么看清扫记录",
             "personal_runtime_unavailable"),
            ("怎么看上次清扫记录", "within_static_manual"),
        )
        for question, expected_status in cases:
            with self.subTest(question=question):
                self.assertEqual(assess_static_knowledge_boundary(question).status,
                                 expected_status)

    def test_ambiguous_haiyou_splits_only_before_a_clear_new_request(self):
        cases = (
            ("我家机器人现在还有多少电？", "personal_runtime_unavailable"),
            ("这台机器人还有百分之几电？", "personal_runtime_unavailable"),
            ("机器人现在还有多久能工作？", "personal_runtime_unavailable"),
            ("怎么看机器人还有多少电？", "within_static_manual"),
            ("怎么看电量状态还有我家机器人现在在哪里？",
             "personal_runtime_unavailable"),
            ("怎么看电量状态还有怎么查看清扫记录？", "within_static_manual"),
        )
        for question, expected_status in cases:
            with self.subTest(question=question):
                self.assertEqual(assess_static_knowledge_boundary(question).status,
                                 expected_status)

    def test_deterministic_and_whitespace_tolerant(self):
        a = assess_static_knowledge_boundary("  这台机器人 当前还剩 多少电？  ")
        b = assess_static_knowledge_boundary("这台机器人当前还剩多少电？")
        self.assertEqual(a, b)
        self.assertEqual(a.reason_code, "PERSONAL_BATTERY_VALUE")

    def test_frozen_answerable_golden_has_zero_false_boundary(self):
        path = Path(__file__).resolve().parents[1] / "data/evaluation/t8_rag_golden_v1.json"
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                         "547fd0f48197bc150640100fb07ba6287fee5d0f4439c1e96c2950c2167e9386")
        cases = json.loads(path.read_text(encoding="utf-8"))["cases"]
        answerable = [case for case in cases if case["answerable"]]
        self.assertEqual(len(answerable), 50)
        self.assert_boundary([case["question"] for case in answerable], "within_static_manual")


class StaticBoundaryReleaseTests(unittest.TestCase):
    def client(self, *, intent="consult_qa", audit="NEEDS_HUMAN_SEMANTIC_REVIEW", draft=DRAFT,
               outcome=SUPPORTED):
        graph, gate, service = Mock(), Mock(), Mock()
        graph.invoke.return_value = {
            "intent": intent, "troubleshooting_status": "ready",
            "draft_answer": draft, "evidence": EVIDENCE,
            "qa_result": {"generation_status": "generated", "audit_status": audit,
                          "citation_status": "VALID"},
        }
        gate.evaluate.return_value = outcome
        return TestClient(create_app(qa_service=service, agent_graph=graph,
                                     semantic_grounding_gate=gate)), graph, gate, service

    def test_g054_api_returns_only_fixed_abstention_and_never_calls_gate(self):
        client, graph, gate, service = self.client(
            draft="无法确认实时电量。" + DRAFT)
        with client:
            response = client.post("/api/agent/chat", json={"question": "我家机器人现在电量是多少？"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["delivery_status"], "safe_abstention")
        self.assertEqual(body["answer"], SAFE_ABSTENTION_ANSWER)
        self.assertFalse(body["published"])
        for private in ("指示灯", "draft_answer", "evidence", "knowledge_boundary", "reason_code"):
            self.assertNotIn(private, response.text)
        gate.evaluate.assert_not_called()
        graph.invoke.assert_called_once_with({"question": "我家机器人现在电量是多少？"})
        service.process_question.assert_not_called()

    def test_positive_gate_cannot_override_boundary(self):
        decision = AnswerReleasePolicy().decide(
            scene="consult_qa", generation_status="generated", audit_status="NEEDS_HUMAN_SEMANTIC_REVIEW",
            draft_answer=DRAFT, evidence=EVIDENCE, citation_status="VALID", gate_outcome=SUPPORTED,
            knowledge_boundary=assess_static_knowledge_boundary("这台机器人当前还剩多少电？"),
        )
        self.assertEqual((decision.status, decision.reason_code),
                         ("safe_abstention", "STATIC_KNOWLEDGE_RUNTIME_UNAVAILABLE"))

    def test_runtime_draft_without_refusal_still_uses_source_boundary(self):
        client, _, gate, _ = self.client()
        with client:
            body = client.post("/api/agent/chat", json={"question": "它现在电量多少？"}).json()
        self.assertEqual(body["answer"], SAFE_ABSTENTION_ANSWER)
        gate.evaluate.assert_not_called()

    def test_how_to_battery_uses_real_release_orchestration_and_gate(self):
        client, _, gate, _ = self.client()
        with client:
            response = client.post("/api/agent/chat", json={"question": "怎么看机器人当前电量？"})
        self.assertEqual(response.json()["delivery_status"], "auto_released")
        self.assertFalse(response.json()["published"])
        gate.evaluate.assert_called_once_with(question="怎么看机器人当前电量？",
                                               draft_answer=DRAFT, evidence=EVIDENCE)

    def test_natural_now_method_still_reaches_gate(self):
        client, _, gate, _ = self.client(intent="maintenance")
        with client:
            body = client.post("/api/agent/chat", json={"question": "现在应该怎么清理主刷？"}).json()
        self.assertEqual(body["delivery_status"], "auto_released")
        gate.evaluate.assert_called_once()

    def test_low_battery_troubleshooting_is_not_boundary_abstention(self):
        client, _, gate, _ = self.client(intent="troubleshooting")
        with client:
            body = client.post("/api/agent/chat", json={"question": "机器人没电了怎么办？"}).json()
        self.assertEqual(body["delivery_status"], "auto_released")
        gate.evaluate.assert_called_once()

    def test_answer_level_program_blockers_remain_first(self):
        for audit in ("PROGRAM_BLOCKED_INVALID_CITATION", "PROGRAM_BLOCKED_UNSUPPORTED_MONEY_CLAIM",
                      "PROGRAM_BLOCKED_UNSUPPORTED_PERSONAL_RUNTIME", "PROGRAM_BLOCKED_MISSING_CITATION"):
            with self.subTest(audit=audit):
                client, _, gate, _ = self.client(audit=audit)
                with client:
                    body = client.post("/api/agent/chat", json={"question": "我家机器人现在电量是多少？"}).json()
                self.assertEqual(body["delivery_status"], "review_required")
                self.assertIsNone(body["answer"])
                gate.evaluate.assert_not_called()

    def test_boundary_does_not_rescue_empty_or_failed_inputs(self):
        inputs = dict(scene="consult_qa", generation_status="generated", audit_status="NEEDS_HUMAN_SEMANTIC_REVIEW",
                      draft_answer=DRAFT, evidence=EVIDENCE, citation_status="VALID",
                      knowledge_boundary=assess_static_knowledge_boundary("它现在电量多少？"))
        for changes in ({"draft_answer": ""}, {"evidence": []}, {"generation_status": "failed"},
                        {"audit_status": "UNKNOWN"}):
            with self.subTest(changes=changes):
                self.assertEqual(AnswerReleasePolicy().decide(**{**inputs, **changes}).status, "review_required")

    def test_boundary_does_not_expand_complex_scene_release(self):
        for scene in ("maintenance_report", "configuration_recommendation"):
            with self.subTest(scene=scene):
                client, _, gate, _ = self.client(intent=scene)
                with client:
                    body = client.post("/api/agent/chat", json={"question": "它现在电量多少？"}).json()
                self.assertEqual(body["delivery_status"], "review_required")
                gate.evaluate.assert_not_called()

    def test_method_question_with_failed_gate_remains_review_required(self):
        client, _, gate, _ = self.client(outcome=GateOutcome("failed", "UPSTREAM_ERROR"))
        with client:
            body = client.post("/api/agent/chat", json={"question": "如何查看剩余电量？"}).json()
        self.assertEqual(body["delivery_status"], "review_required")
        self.assertIsNone(body["answer"])
        gate.evaluate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
