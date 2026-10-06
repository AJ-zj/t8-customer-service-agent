import json
from pathlib import Path
import unittest
from unittest.mock import Mock
from urllib.error import HTTPError

from fastapi.testclient import TestClient

from app.main import create_app
from app.services.answer_release_policy import (
    AnswerReleasePolicy,
    GateOutcome,
    SAFE_ABSTENTION_ANSWER,
)
from app.services.semantic_grounding_gate import (
    SemanticGroundingGate,
    SemanticGroundingOutcome,
    GATE_SYSTEM_PROMPT,
)


EVIDENCE = [{
    "evidence_id": "E1", "record_index": 14, "pdf_page": 15,
    "title": "日常维护", "section": "软胶主刷",
    "text": "软胶主刷应拆下清理后装回。",
}]
DRAFT = "按说明书拆下软胶主刷清理后装回。[E1]"
QA_RESULT = {
    "generation_status": "generated",
    "audit_status": "NEEDS_HUMAN_SEMANTIC_REVIEW",
    "citation_status": "VALID",
}


class _Response:
    status = 200

    def __init__(self, content):
        self.payload = json.dumps({"choices": [{"message": {"content": content}}]}).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.payload


def _valid_gate(supported=True):
    return json.dumps({
        "all_claims_supported": supported,
        "core_answer_supported": supported,
        "core_abstention_quote": "",
        "reason_code": "SUPPORTED" if supported else "UNSUPPORTED_FACT",
        "unsupported_claims": [] if supported else ["存在未被证据支持的陈述"],
    }, ensure_ascii=False)


def _api(state, *, gate=None, memory=None):
    graph = Mock()
    graph.invoke.return_value = state
    if gate is None:
        gate = Mock()
        gate.evaluate.return_value = SemanticGroundingOutcome(
            status="supported", reason_code="SUPPORTED",
            all_claims_supported=True, core_answer_supported=True, core_abstention_quote="",
        )
    app = create_app(
        qa_service=Mock(), agent_graph=graph, semantic_grounding_gate=gate,
        memory_store=memory,
    )
    return TestClient(app), graph, gate


def _state(scene="maintenance", *, audit=None, draft=DRAFT, evidence=EVIDENCE):
    return {
        "intent": scene,
        "draft_answer": draft,
        "evidence": evidence,
        "qa_result": {**QA_RESULT, **(audit or {})},
    }


class AnswerReleasePolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy = AnswerReleasePolicy()
        self.base = dict(
            scene="maintenance", generation_status="generated",
            audit_status="NEEDS_HUMAN_SEMANTIC_REVIEW", draft_answer=DRAFT,
            evidence=EVIDENCE, citation_status="VALID",
        )

    def test_eligible_supported_answer_is_auto_released(self):
        result = self.policy.decide(**self.base, gate_outcome=GateOutcome("supported", "SUPPORTED", True, True, ""))
        self.assertEqual((result.status, result.reason_code), ("auto_released", "SEMANTICALLY_GROUNDED"))

    def test_program_blockers_take_priority_over_gate(self):
        for status in (
            "PROGRAM_BLOCKED_INVALID_CITATION", "PROGRAM_BLOCKED_UNSUPPORTED_MONEY_CLAIM",
            "PROGRAM_BLOCKED_UNSUPPORTED_PERSONAL_RUNTIME", "PROGRAM_BLOCKED_MISSING_CITATION",
        ):
            with self.subTest(status=status):
                result = self.policy.decide(**{**self.base, "audit_status": status},
                                            gate_outcome=GateOutcome("supported", "SUPPORTED", True, True, ""))
                self.assertEqual(result.status, "review_required")
                self.assertEqual(result.reason_code, "PROGRAM_AUDIT_BLOCKED")

    def test_model_abstention_uses_safe_delivery(self):
        result = self.policy.decide(**{**self.base, "audit_status": "MODEL_ABSTAINED_NEEDS_REVIEW"})
        self.assertEqual(result.status, "safe_abstention")

    def test_missing_citation_evidence_or_draft_never_reaches_gate(self):
        for overrides in ({"citation_status": "MISSING"}, {"evidence": []}, {"draft_answer": " "}):
            with self.subTest(overrides=overrides):
                result = self.policy.decide(**{**self.base, **overrides})
                self.assertEqual(result.status, "review_required")
                self.assertNotEqual(result.reason_code, "SEMANTIC_GATE_REQUIRED")

    def test_only_three_simple_scenes_are_eligible(self):
        for scene in ("maintenance_report", "configuration_recommendation"):
            with self.subTest(scene=scene):
                result = self.policy.decide(**{**self.base, "scene": scene},
                                            gate_outcome=GateOutcome("supported", "SUPPORTED", True, True, ""))
                self.assertEqual(result.status, "review_required")
                self.assertEqual(result.reason_code, "SCENE_NOT_ELIGIBLE")

    def test_unsupported_and_failed_gate_fail_closed(self):
        for outcome, expected in (
            (GateOutcome("unsupported", "UNSUPPORTED_FACT", False, False, ""), "SEMANTICALLY_UNSUPPORTED"),
            (GateOutcome("failed", "GATE_UNAVAILABLE"), "SEMANTIC_GATE_FAILED_CLOSED"),
        ):
            with self.subTest(outcome=outcome):
                result = self.policy.decide(**self.base, gate_outcome=outcome)
                self.assertEqual(result.status, "review_required")
                self.assertEqual(result.reason_code, expected)

    def test_published_state_is_not_part_of_live_release_decision(self):
        result = self.policy.decide(**self.base, gate_outcome=GateOutcome("supported", "SUPPORTED", True, True, ""))
        self.assertEqual(result.status, "auto_released")
        # Publication is deliberately not an input to this policy.
        self.assertNotIn("published", result.__dict__)


class AnswerReleaseApiTests(unittest.TestCase):
    def test_auto_release_returns_answer_but_not_published(self):
        client, _, gate = _api(_state())
        body = client.post("/api/agent/chat", json={"question": "主刷多久清理一次？"}).json()
        self.assertEqual(body["delivery_status"], "auto_released")
        self.assertEqual(body["answer"], DRAFT)
        self.assertFalse(body["published"])
        self.assertEqual(set(body), {"intent", "answer", "delivery_status", "generation_status", "audit_status", "published"})
        gate.evaluate.assert_called_once_with(question="主刷多久清理一次？", draft_answer=DRAFT, evidence=EVIDENCE)

    def test_standard_abstention_is_safe_visible_text(self):
        state = _state(audit={"audit_status": "MODEL_ABSTAINED_NEEDS_REVIEW", "citation_status": "MISSING"})
        client, _, gate = _api(state)
        body = client.post("/api/agent/chat", json={"question": "错误码401是什么意思？"}).json()
        self.assertEqual(body["delivery_status"], "safe_abstention")
        self.assertEqual(body["answer"], SAFE_ABSTENTION_ANSWER)
        self.assertFalse(body["published"])
        gate.evaluate.assert_not_called()

    def test_program_citation_block_cannot_be_rescued_by_gate(self):
        client, _, gate = _api(_state(audit={"audit_status": "PROGRAM_BLOCKED_INVALID_CITATION"}))
        body = client.post("/api/agent/chat", json={"question": "主刷怎么清理？"}).json()
        self.assertEqual(body["delivery_status"], "review_required")
        self.assertIsNone(body["answer"])
        self.assertFalse(body["published"])
        gate.evaluate.assert_not_called()

    def test_money_block_cannot_be_rescued_by_gate(self):
        client, _, gate = _api(_state(audit={"audit_status": "PROGRAM_BLOCKED_UNSUPPORTED_MONEY_CLAIM"}))
        body = client.post("/api/agent/chat", json={"question": "维修多少钱？"}).json()
        self.assertEqual(body["delivery_status"], "review_required")
        self.assertIsNone(body["answer"])
        gate.evaluate.assert_not_called()

    def test_missing_citation_block_is_review_required(self):
        client, _, gate = _api(_state(audit={"audit_status": "PROGRAM_BLOCKED_MISSING_CITATION"}))
        body = client.post("/api/agent/chat", json={"question": "主刷怎么清理？"}).json()
        self.assertEqual(body["delivery_status"], "review_required")
        self.assertIsNone(body["answer"])
        gate.evaluate.assert_not_called()

    def test_semantic_unsupported_answer_is_not_returned(self):
        gate = Mock()
        gate.evaluate.return_value = SemanticGroundingOutcome(
            status="unsupported", reason_code="UNSUPPORTED_FACT",
            all_claims_supported=False, core_answer_supported=True, core_abstention_quote="",
            unsupported_claims=("内部断言",),
        )
        client, _, _ = _api(_state(), gate=gate)
        body = client.post("/api/agent/chat", json={"question": "主刷怎么清理？"}).json()
        self.assertEqual(body["delivery_status"], "review_required")
        self.assertIsNone(body["answer"])
        self.assertNotIn("unsupported_claims", str(body))
        self.assertNotIn("内部断言", str(body))

    def test_gate_timeout_is_safe_review_response_not_503(self):
        gate = Mock()
        gate.evaluate.side_effect = TimeoutError("private upstream detail")
        client, _, _ = _api(_state(), gate=gate)
        response = client.post("/api/agent/chat", json={"question": "主刷怎么清理？"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["delivery_status"], "review_required")
        self.assertIsNone(response.json()["answer"])
        self.assertNotIn("private", response.text)

    def test_maintenance_report_never_calls_gate(self):
        client, _, gate = _api(_state("maintenance_report"))
        body = client.post("/api/agent/chat", json={"question": "做维护报告"}).json()
        self.assertEqual(body["delivery_status"], "review_required")
        self.assertIsNone(body["answer"])
        gate.evaluate.assert_not_called()

    def test_configuration_recommendation_never_calls_gate(self):
        client, _, gate = _api(_state("configuration_recommendation"))
        body = client.post("/api/agent/chat", json={"question": "我家有宠物和地毯，怎么设置？"}).json()
        self.assertEqual(body["delivery_status"], "review_required")
        self.assertIsNone(body["answer"])
        gate.evaluate.assert_not_called()

    def test_clarification_and_handoff_keep_visible_response(self):
        for intent, answer, status in (
            ("clarify", "请补充异常现象。", "clarification"),
            ("human_handoff", "已为你转入人工处理流程。", "human_handoff"),
        ):
            with self.subTest(intent=intent):
                client, _, gate = _api({"intent": intent, "answer": answer})
                body = client.post("/api/agent/chat", json={"question": "测试"}).json()
                self.assertEqual(body["answer"], answer)
                self.assertEqual(body["delivery_status"], status)
                gate.evaluate.assert_not_called()

    def test_response_never_contains_evidence_or_gate_internals(self):
        gate = Mock()
        gate.evaluate.return_value = SemanticGroundingOutcome(
            status="unsupported", reason_code="UNSUPPORTED_FACT",
            all_claims_supported=False, core_answer_supported=True, core_abstention_quote="",
            unsupported_claims=("private evidence claim",),
        )
        client, _, _ = _api(_state(), gate=gate)
        response = client.post("/api/agent/chat", json={"question": "主刷清理"})
        for hidden in ("draft_answer", "evidence", "private evidence claim", "unsupported_claims", "messages"):
            self.assertNotIn(hidden, response.text)
        self.assertNotIn(EVIDENCE[0]["text"], response.text)


class SemanticGroundingGateTests(unittest.TestCase):
    def gate(self, contents, *, transport=None):
        queue = iter(contents)
        calls = []

        def default_transport(request, timeout):
            calls.append((json.loads(request.data.decode("utf-8")), timeout))
            return _Response(next(queue))

        gate = SemanticGroundingGate(
            api_key_provider=lambda: "test-secret",
            transport=transport or default_transport,
        )
        return gate, calls

    def evaluate(self, gate):
        return gate.evaluate(question="主刷怎么清理？", draft_answer=DRAFT, evidence=EVIDENCE)

    def test_valid_supported_schema_uses_qwen_and_temperature_zero(self):
        gate, calls = self.gate([_valid_gate()])
        result = self.evaluate(gate)
        self.assertEqual(result.status, "supported")
        self.assertEqual(calls[0][0]["model"], "Qwen/Qwen2.5-7B-Instruct")
        self.assertEqual(calls[0][0]["temperature"], 0)
        self.assertEqual(calls[0][0]["response_format"]["type"], "json_schema")
        self.assertTrue(calls[0][0]["response_format"]["json_schema"]["strict"])
        self.assertFalse(calls[0][0]["response_format"]["json_schema"]["schema"]["additionalProperties"])

    def test_valid_unsupported_schema_is_not_released(self):
        gate, _ = self.gate([_valid_gate(False)])
        self.assertEqual(self.evaluate(gate).status, "unsupported")

    def test_schema_failure_retries_exactly_once(self):
        gate, calls = self.gate([
            json.dumps({"all_claims_supported": True, "core_answer_supported": True,
                        "core_abstention_quote": "", "reason_code": "SUPPORTED", "unsupported_claims": [], "extra": 1}),
            _valid_gate(),
        ])
        self.assertEqual(self.evaluate(gate).status, "supported")
        self.assertEqual(len(calls), 2)

    def test_second_schema_failure_fails_closed(self):
        invalid = json.dumps({"all_claims_supported": True, "core_answer_supported": True,
                              "core_abstention_quote": "", "reason_code": "SUPPORTED", "unsupported_claims": [], "why": "bad"})
        gate, calls = self.gate([invalid, invalid])
        result = self.evaluate(gate)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.reason_code, "INVALID_RESPONSE")
        self.assertEqual(len(calls), 2)

    def test_invalid_json_has_one_schema_retry(self):
        gate, calls = self.gate(["not json", _valid_gate()])
        self.assertEqual(self.evaluate(gate).status, "supported")
        self.assertEqual(len(calls), 2)

    def test_transport_timeout_does_not_retry(self):
        calls = []

        def transport(*_args, **_kwargs):
            calls.append(1)
            raise TimeoutError("sensitive transport detail")

        gate = SemanticGroundingGate(api_key_provider=lambda: "test-secret", transport=transport)
        result = self.evaluate(gate)
        self.assertEqual(result.status, "failed")
        self.assertEqual(calls, [1])
        self.assertNotIn("sensitive", result.model_dump_json())

    def test_empty_inputs_fail_without_api_call(self):
        calls = []
        gate = SemanticGroundingGate(api_key_provider=lambda: "test-secret", transport=lambda *_a, **_k: calls.append(1))
        self.assertEqual(gate.evaluate(question="q", draft_answer="d", evidence=[]).status, "failed")
        self.assertEqual(calls, [])

    def test_http_error_does_not_expose_body_or_retry(self):
        calls = []

        def transport(*_args, **_kwargs):
            calls.append(1)
            raise HTTPError("https://example.invalid", 503, "upstream", {}, None)

        gate = SemanticGroundingGate(api_key_provider=lambda: "test-secret", transport=transport)
        outcome = self.evaluate(gate)
        self.assertEqual(outcome.status, "failed")
        self.assertEqual(calls, [1])
        self.assertNotIn("upstream", outcome.model_dump_json())

    def test_case_a_maintenance_table_is_sent_without_gold_or_reference(self):
        evidence = [{"evidence_id": "E1", "text": "部件 维护频率 更换频率\n软胶主刷 每 2 周清理 每 6-12 个月"}]
        draft = "软胶主刷建议每两周清理一次。[E1]"
        gate, calls = self.gate([_valid_gate()])
        result = gate.evaluate(question="主刷多久清理一次？", draft_answer=draft, evidence=evidence)
        self.assertEqual(result.status, "supported")
        sent = json.loads(calls[0][0]["messages"][1]["content"])
        self.assertEqual(sent, {"question": "主刷多久清理一次？", "draft_answer": draft, "evidence": evidence})
        self.assertEqual(set(sent), {"question", "draft_answer", "evidence"})
        self.assertIn("表格的列标题与对应部件行", calls[0][0]["messages"][0]["content"])

    def test_faithful_paraphrase_is_eligible_for_supported_gate_verdict(self):
        gate, calls = self.gate([_valid_gate()])
        draft = "请按说明书操作：用柔软的干布擦净主轮。[E1]"
        result = gate.evaluate(question="主轮怎么清理？", draft_answer=draft,
                               evidence=[{"evidence_id": "E1", "text": "用柔软干布擦拭清理主轮。"}])
        self.assertEqual(result.status, "supported")
        self.assertIn("忠实改述、概括和同义表达可以受支持", calls[0][0]["messages"][0]["content"])

    def test_unsupported_fact_verdict_blocks_a_specific_new_claim(self):
        claim = "滤网可以使用洗洁精清洗"
        gate, _ = self.gate([json.dumps({"all_claims_supported": False, "core_answer_supported": True,
                                      "core_abstention_quote": "", "reason_code": "UNSUPPORTED_FACT", "unsupported_claims": [claim]})])
        result = gate.evaluate(question="滤网怎么洗？", draft_answer=claim + "。[E1]",
                               evidence=[{"evidence_id": "E1", "text": "只用清水，不添加洗涤剂。"}])
        self.assertEqual(result.status, "unsupported")
        self.assertEqual(result.unsupported_claims, (claim,))

    def test_mixed_supported_and_unsupported_fact_does_not_release(self):
        gate, _ = self.gate([_valid_gate(False)])
        result = gate.evaluate(question="滤网怎么洗？",
                               draft_answer="清水清洗，然后使用高温烘干。[E1]",
                               evidence=[{"evidence_id": "E1", "text": "请用清水清洗滤网。"}])
        self.assertEqual(result.status, "unsupported")
        self.assertIn("支持和不支持的事实混在一起", GATE_SYSTEM_PROMPT)

    def test_citation_heading_and_politeness_are_not_claims_in_prompt(self):
        for rule in ("礼貌语", "结构标题", "引用标记 [E1]/[E2]/[E3]", "用户问题列为 unsupported_claims"):
            self.assertIn(rule, GATE_SYSTEM_PROMPT)
        self.assertIn("具体频率、时间、步骤、功能、错误码含义、参数和安全限制都必须有依据", GATE_SYSTEM_PROMPT)

    def test_nonstandard_reason_or_contradictory_verdict_fails_closed(self):
        for payload in (
            {"all_claims_supported": False, "core_answer_supported": False, "core_abstention_quote": "", "reason_code": "free text", "unsupported_claims": []},
            {"all_claims_supported": True, "core_answer_supported": True, "core_abstention_quote": "", "reason_code": "SUPPORTED", "unsupported_claims": ["unsupported"]},
            {"all_claims_supported": False, "core_answer_supported": True, "core_abstention_quote": "", "reason_code": "UNSUPPORTED_FACT", "unsupported_claims": []},
        ):
            with self.subTest(payload=payload):
                gate, calls = self.gate([json.dumps(payload)] * 2)
                outcome = self.evaluate(gate)
                self.assertEqual((outcome.status, outcome.reason_code), ("failed", "INVALID_RESPONSE"))
                self.assertEqual(len(calls), 2)

    def test_empty_draft_and_evidence_have_standard_codes_without_transport(self):
        gate, calls = self.gate([])
        self.assertEqual(gate.evaluate(question="q", draft_answer=" ", evidence=EVIDENCE).reason_code, "EMPTY_DRAFT")
        self.assertEqual(gate.evaluate(question="q", draft_answer=DRAFT, evidence=[]).reason_code, "EMPTY_EVIDENCE")
        self.assertEqual(calls, [])


class CachedUnanswerableSafetyTests(unittest.TestCase):
    def test_all_ten_frozen_unanswerable_drafts_cannot_release_without_semantic_verification(self):
        from app.services.grounded_qa import audit_answer

        path = Path(__file__).resolve().parents[1] / "data/evaluation/results/generation_eval_v1.json"
        cases = [case for case in json.loads(path.read_text(encoding="utf-8"))["cases"] if not case["answerable"]]
        self.assertEqual(len(cases), 10)
        for case in cases:
            with self.subTest(case=case["id"]):
                evidence = case["retrieved_records"]
                records = [(item["evidence_id"], item["record_index"], item) for item in evidence]
                audit = audit_answer(case["question"], case["answer"], records)
                inputs = dict(
                    scene="consult_qa", generation_status="generated", audit_status=audit["status"],
                    draft_answer=case["answer"], evidence=evidence, citation_status=audit["citation_status"],
                )
                policy = AnswerReleasePolicy()
                decision = policy.decide(**inputs)
                # Mixed responses may contain unrelated manual facts after an
                # opening refusal. They require the Gate, never Audit-only release.
                self.assertIn(decision.status, {"safe_abstention", "review_required"})
                if audit["status"] == "MODEL_ABSTAINED_NEEDS_REVIEW":
                    self.assertEqual(decision.status, "safe_abstention")
                    self.assertEqual(policy.decide(**inputs, gate_outcome=GateOutcome(
                        "supported", "SUPPORTED", True, True, "")).status, "safe_abstention")
                else:
                    self.assertEqual(decision.reason_code, "SEMANTIC_GATE_REQUIRED")
                    self.assertEqual(policy.decide(**inputs, gate_outcome=GateOutcome(
                        "unsupported", "INSUFFICIENT_EVIDENCE", True, False, "")).status, "review_required")


if __name__ == "__main__":
    unittest.main()
