"""Core-question release contract tests; no paid models or real database."""

import json
from pathlib import Path
import unittest
from unittest.mock import Mock

from fastapi.testclient import TestClient

from app.main import create_app
from app.services.answer_release_policy import AnswerReleasePolicy, GateOutcome, SAFE_ABSTENTION_ANSWER
from app.services.grounded_qa import audit_answer
from app.services.semantic_grounding_gate import SemanticGroundingGate, GATE_SYSTEM_PROMPT


EVIDENCE = [{"evidence_id": "E1", "record_index": 13, "pdf_page": 14,
             "text": "软胶主刷 每2周清理。电源指示灯白色：电量≥20%；红色：电量＜20%。"}]


def payload(*, claims=True, core=True, quote="", unsupported=()):
    return {
        "all_claims_supported": claims, "core_answer_supported": core,
        "core_abstention_quote": quote,
        "reason_code": "SUPPORTED" if claims and core else (
            "UNSUPPORTED_FACT" if not claims else "CORE_QUESTION_UNSUPPORTED"),
        "unsupported_claims": list(unsupported),
    }


class Response:
    status = 200

    def __init__(self, content):
        self.body = json.dumps({"choices": [{"message": {"content": json.dumps(content)}}]}).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.body


class CoreQuestionSupportTests(unittest.TestCase):
    def gate(self, data, *, error=None):
        calls = []

        def transport(request, timeout):
            calls.append(json.loads(request.data))
            if error:
                raise error
            return Response(data)

        return SemanticGroundingGate(api_key_provider=lambda: "unit-test-key", transport=transport), calls

    def inputs(self, question, draft, evidence=EVIDENCE):
        records = [(row["evidence_id"], row["record_index"], row) for row in evidence]
        audit = audit_answer(question, draft, records)
        return dict(scene="maintenance", generation_status="generated",
                    audit_status=audit["status"], draft_answer=draft, evidence=evidence,
                    citation_status=audit["citation_status"])

    def decide(self, question, draft, data, evidence=EVIDENCE):
        inputs = self.inputs(question, draft, evidence)
        gate, calls = self.gate(data)
        outcome = gate.evaluate(question=question, draft_answer=draft, evidence=evidence)
        return outcome, AnswerReleasePolicy().decide(**inputs, gate_outcome=outcome), calls

    def test_all_claims_and_core_supported_are_both_required(self):
        outcome, decision, _ = self.decide("主刷多久清理一次？", "主刷每两周清理一次。[E1]", payload())
        self.assertTrue(outcome.all_claims_supported)
        self.assertTrue(outcome.core_answer_supported)
        self.assertEqual(decision.status, "auto_released")

    def test_g054_explicit_refusal_and_ancillary_facts_deliver_only_safe_text(self):
        root = Path(__file__).resolve().parents[1]
        g054 = next(case for case in json.loads((root / "data/evaluation/results/generation_eval_v1.json").read_text(encoding="utf-8"))["cases"] if case["id"] == "T8-G054")
        quote = g054["answer"].split("。", 1)[0] + "。"
        gate, calls = self.gate(payload(core=False, quote=quote))
        inputs = self.inputs(g054["question"], g054["answer"], g054["retrieved_records"])
        self.assertEqual(inputs["audit_status"], "NEEDS_HUMAN_SEMANTIC_REVIEW")
        graph = Mock()
        graph.invoke.return_value = {
            "intent": "consult_qa", "draft_answer": g054["answer"], "evidence": g054["retrieved_records"],
            "qa_result": {"generation_status": "generated", "audit_status": inputs["audit_status"],
                          "citation_status": inputs["citation_status"]},
        }
        with TestClient(create_app(qa_service=Mock(), agent_graph=graph, semantic_grounding_gate=gate)) as client:
            response = client.post("/api/agent/chat", json={"question": g054["question"]})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["delivery_status"], "safe_abstention")
        self.assertEqual(response.json()["answer"], SAFE_ABSTENTION_ANSWER)
        self.assertFalse(response.json()["published"])
        self.assertNotIn("指示灯", response.text)
        for private in ("core_answer_supported", "core_abstention_quote", "unsupported_claims", "evidence"):
            self.assertNotIn(private, response.text)
        # V1.1.3 classifies this runtime query before the semantic Gate;
        # the old Gate cannot rescue G054 with supported ancillary facts.
        self.assertEqual(len(calls), 0)

    def test_current_battery_without_refusal_requires_review(self):
        outcome, decision, _ = self.decide("我家机器人现在电量是多少？", "电量低时指示灯为红色。[E1]",
                                          payload(core=False))
        self.assertTrue(outcome.all_claims_supported)
        self.assertFalse(outcome.core_answer_supported)
        self.assertEqual(decision.status, "review_required")

    def test_how_to_check_battery_can_be_supported(self):
        outcome, decision, _ = self.decide("怎么看机器人电量状态？", "指示灯白色表示电量≥20%，红色表示电量＜20%。[E1]", payload())
        self.assertTrue(outcome.core_answer_supported)
        self.assertEqual(decision.status, "auto_released")

    def test_unknown_code_with_explicit_core_refusal_is_safe(self):
        quote = "说明书无法确认错误码401的含义。"
        _, decision, _ = self.decide("错误码401是什么意思？", quote + "主刷每两周清理一次。[E1]",
                                     payload(core=False, quote=quote))
        self.assertEqual(decision.status, "safe_abstention")

    def test_both_mixed_orders_reach_gate_and_can_answer_core(self):
        for draft in (
            "主刷每两周清理一次。[E1] 说明书未明确是否每天检查。",
            "说明书未明确是否每天检查，但主刷每两周清理一次。[E1]",
        ):
            with self.subTest(draft=draft):
                inputs = self.inputs("主刷多久清理一次？", draft)
                self.assertEqual(AnswerReleasePolicy().decide(**inputs).reason_code, "SEMANTIC_GATE_REQUIRED")
                outcome, decision, _ = self.decide("主刷多久清理一次？", draft, payload())
                self.assertTrue(outcome.core_answer_supported)
                self.assertEqual(decision.status, "auto_released")

    def test_now_word_in_a_how_to_question_is_not_a_program_blacklist(self):
        _, decision, _ = self.decide("现在应该怎么清理主刷？", "主刷应拆下来清理。[E1]", payload())
        self.assertEqual(decision.status, "auto_released")

    def test_supported_core_does_not_rescue_unsupported_ancillary_claim(self):
        _, decision, _ = self.decide("主刷多久清理一次？", "主刷每两周清理一次；滤网可高温烘干。[E1]",
                                     payload(claims=False, unsupported=["滤网可高温烘干"]))
        self.assertEqual(decision.status, "review_required")

    def test_fabricated_core_abstention_quote_fails_closed(self):
        outcome, decision, calls = self.decide("当前电量多少？", "电量低时指示灯为红色。[E1]",
                                               payload(core=False, quote="无法确认当前电量。"))
        self.assertEqual((outcome.status, outcome.reason_code), ("failed", "INVALID_RESPONSE"))
        self.assertEqual(decision.status, "review_required")
        self.assertEqual(len(calls), 2)

    def test_core_supported_and_core_refusal_is_contradictory(self):
        outcome, decision, _ = self.decide("主刷多久清理一次？", "无法确认；主刷每两周清理一次。[E1]",
                                          payload(quote="无法确认"))
        self.assertEqual(outcome.status, "failed")
        self.assertEqual(decision.status, "review_required")

    def test_legacy_single_supported_bool_schema_is_not_accepted(self):
        outcome, decision, _ = self.decide("主刷怎么清理？", "主刷拆下清理。[E1]",
                                          {"supported": True, "reason_code": "SUPPORTED", "unsupported_claims": []})
        self.assertEqual(outcome.status, "failed")
        self.assertEqual(decision.status, "review_required")

    def test_legacy_positive_outcome_without_both_flags_fails_closed(self):
        inputs = self.inputs("主刷怎么清理？", "主刷拆下清理。[E1]")
        decision = AnswerReleasePolicy().decide(**inputs, gate_outcome=GateOutcome("supported", "SUPPORTED"))
        self.assertEqual(decision.status, "review_required")

    def test_inconsistent_outcome_status_cannot_auto_release(self):
        inputs = self.inputs("当前电量多少？", "无法确认当前电量。指示灯可显示状态。[E1]")
        decision = AnswerReleasePolicy().decide(**inputs, gate_outcome=GateOutcome(
            "supported", "SUPPORTED", True, False, "无法确认当前电量。"))
        self.assertEqual(decision.status, "review_required")

    def test_all_ten_unanswerable_cannot_release_from_ancillary_support_alone(self):
        path = Path(__file__).resolve().parents[1] / "data/evaluation/results/generation_eval_v1.json"
        cases = [case for case in json.loads(path.read_text(encoding="utf-8"))["cases"] if not case["answerable"]]
        self.assertEqual(len(cases), 10)
        for case in cases:
            with self.subTest(case=case["id"]):
                quote = case["answer"].split("。", 1)[0] + "。"
                _, decision, _ = self.decide(case["question"], case["answer"],
                                            payload(core=False, quote=quote), case["retrieved_records"])
                self.assertEqual(decision.status, "safe_abstention")

    def test_empty_inputs_never_become_safe_abstention(self):
        inputs = self.inputs("当前电量多少？", "无法确认。")
        inputs["audit_status"] = "MODEL_ABSTAINED_NEEDS_REVIEW"
        for overrides in ({"draft_answer": " "}, {"evidence": []}, {"generation_status": "failed"}):
            with self.subTest(overrides=overrides):
                self.assertEqual(AnswerReleasePolicy().decide(**{**inputs, **overrides}).status, "review_required")

    def test_program_blockers_precede_core_abstention(self):
        inputs = self.inputs("当前电量多少？", "无法确认。指示灯为红色。[E1]")
        for status in ("PROGRAM_BLOCKED_INVALID_CITATION", "PROGRAM_BLOCKED_UNSUPPORTED_MONEY_CLAIM",
                       "PROGRAM_BLOCKED_UNSUPPORTED_PERSONAL_RUNTIME", "PROGRAM_BLOCKED_MISSING_CITATION"):
            with self.subTest(status=status):
                decision = AnswerReleasePolicy().decide(**{**inputs, "audit_status": status},
                    gate_outcome=GateOutcome("unsupported", "CORE_QUESTION_UNSUPPORTED", True, False, "无法确认。"))
                self.assertEqual((decision.status, decision.reason_code), ("review_required", "PROGRAM_AUDIT_BLOCKED"))

    def test_complex_scenes_do_not_use_abstention_to_bypass_review(self):
        inputs = self.inputs("当前电量多少？", "无法确认。")
        for scene in ("maintenance_report", "configuration_recommendation"):
            with self.subTest(scene=scene):
                self.assertEqual(AnswerReleasePolicy().decide(**{**inputs, "scene": scene}).status, "review_required")

    def test_gate_timeout_has_unknown_semantic_flags_and_fails_closed(self):
        gate, calls = self.gate({}, error=TimeoutError("private detail"))
        inputs = self.inputs("主刷怎么清理？", "主刷拆下清理。[E1]")
        outcome = gate.evaluate(question="主刷怎么清理？", draft_answer=inputs["draft_answer"], evidence=EVIDENCE)
        self.assertIsNone(outcome.core_answer_supported)
        self.assertIsNone(outcome.all_claims_supported)
        self.assertEqual(AnswerReleasePolicy().decide(**inputs, gate_outcome=outcome).status, "review_required")
        self.assertEqual(len(calls), 1)

    def test_prompt_separates_core_request_from_supported_facts(self):
        for rule in ("核心信息需求", "实时电量", "查看方法", "附加的通用产品事实", "局部限制",
                     "core_answer_supported", "all_claims_supported", "core_abstention_quote"):
            self.assertIn(rule, GATE_SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
