import json
import unittest
from unittest.mock import Mock

from fastapi.testclient import TestClient
from app.agent.llm_router import IntentDecision, LLMRouterError
from app.agent.ambiguity import needs_router_clarification
from app.agent.memory import SessionMemoryStore
from app.agent.react_graph import build_react_agent
from app.agent.troubleshooting import (
    TroubleshootingIntake, TroubleshootingIntakeError, TroubleshootingIntakeDecision,
    TROUBLESHOOTING_CLARIFICATION, build_intake_input, safe_clarification_decision,
)
from app.config import ROUTER_MODEL
from app.main import create_app
from app.services.qa_service import GroundedQAService
from test_react_agent import EVIDENCE, tool_decision


def ready(summary="清扫时中途停止", code=None):
    return TroubleshootingIntakeDecision(status="ready", symptom_summary=summary,
        error_code=code, missing_information=[], clarifying_question=None)


def intake_client(content=None, error=None):
    captured = {}
    class Response:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *_): return False
        def read(self):
            return json.dumps({"choices": [{"message": {"content": content}}]}).encode()
    def transport(request, *, timeout):
        captured.update(body=json.loads(request.data), timeout=timeout)
        if error: raise error
        return Response()
    return TroubleshootingIntake(api_key_provider=lambda: "unit-secret", transport=transport), captured


def queued_intake(contents):
    calls = []
    class Response:
        status = 200
        def __init__(self, value): self.value = value
        def __enter__(self): return self
        def __exit__(self, *_): return False
        def read(self): return json.dumps({"choices": [{"message": {"content": self.value}}]}).encode()
    def transport(request, *, timeout):
        calls.append(json.loads(request.data))
        value = contents.pop(0)
        if isinstance(value, Exception): raise value
        return Response(value)
    return TroubleshootingIntake(api_key_provider=lambda: "unit-secret", transport=transport), calls


def workflow(decisions, intent="troubleshooting"):
    intake, router, llm, search, session = Mock(), Mock(), Mock(), Mock(), Mock()
    intake.assess.side_effect = decisions
    router.classify.return_value = IntentDecision(intent=intent, reason="测试场景")
    llm.complete.side_effect = [tool_decision(question="模型参数不能改变问题"),
                                {"content": "按本轮说明书进行排查。[E1]"}]
    search.name = "search_t8_manual"
    search.invoke.return_value = {"retrieval_query": "原问题", "evidence": EVIDENCE}
    service = GroundedQAService(session_factory=lambda: session)
    graph = build_react_agent(qa_service=service, llm_router=router, tool_llm=llm,
                              search_tool=search, troubleshooting_intake=intake)
    memory = SessionMemoryStore()
    api = TestClient(create_app(qa_service=service, agent_graph=graph, memory_store=memory))
    return api, graph, memory, intake, router, llm, search, session


class TroubleshootingIntakeTests(unittest.TestCase):
    def test_specific_symptom_contract_ready_without_error_code(self):
        client, captured = intake_client(ready().model_dump_json())
        self.assertEqual(client.assess("扫着扫着停了。").status, "ready")
        self.assertEqual(captured["body"]["model"], ROUTER_MODEL)
        self.assertEqual(captured["body"]["temperature"], 0)

    def test_error_code_contract_ready(self):
        client, _ = intake_client(ready("提示 E4", "E4").model_dump_json())
        self.assertEqual(client.assess("出现 E4。").error_code, "E4")

    def test_generic_malfunction_contract_needs_clarification(self):
        client, _ = intake_client(safe_clarification_decision().model_dump_json())
        self.assertEqual(client.assess("机器人无法正常工作怎么办？").status, "needs_clarification")

    def test_invalid_json_enum_extra_and_inconsistent_status_are_rejected(self):
        bad = ['ready', '```json\n{}\n```', '{"status":"maybe"}',
               json.dumps({**ready().model_dump(), "extra": 1}),
               json.dumps({**ready().model_dump(), "symptom_summary": " "})]
        for content in bad:
            with self.subTest(content=content), self.assertRaises(TroubleshootingIntakeError):
                intake_client(content)[0].assess("机器人无法正常工作。")

    def test_timeout_is_sanitized(self):
        client, _ = intake_client(error=TimeoutError("unit-secret upstream body"))
        with self.assertRaises(TroubleshootingIntakeError) as raised:
            client.assess("机器人无法正常工作。")
        self.assertNotIn("unit-secret", str(raised.exception))

    def test_only_latest_user_context_enters_intake(self):
        context = [{"role": "user", "content": "更早问题"},
                   {"role": "user", "content": "机器人无法正常工作怎么办？"},
                   {"role": "assistant", "content": "内部字段不能作为事实"}]
        self.assertEqual(build_intake_input("扫着扫着停了。", context), {
            "current_question": "扫着扫着停了。", "latest_user_question": "机器人无法正常工作怎么办？",
            "contextual_question": "机器人无法正常工作怎么办？ 扫着扫着停了。"})

    def test_hallucinated_error_code_is_rejected(self):
        client, _ = intake_client(ready("故障", "E99").model_dump_json())
        with self.assertRaises(TroubleshootingIntakeError):
            client.assess("主刷不转了。")

    def test_initial_valid_ready_has_no_retry(self):
        client, calls = queued_intake([ready().model_dump_json()])
        result = client.assess_with_metadata("机器人一直卡住")
        self.assertEqual(result.decision.status, "ready")
        self.assertEqual((result.initial_contract_success, result.retry_contract_success,
                          result.final_contract_success, result.retry_count), (True, None, True, 0))
        self.assertEqual(len(calls), 1)

    def test_invalid_json_gets_one_strict_retry(self):
        client, calls = queued_intake(["{oops", ready().model_dump_json()])
        result = client.assess_with_metadata("机器人一直卡住")
        self.assertEqual((result.initial_contract_success, result.retry_contract_success,
                          result.final_contract_success, result.retry_count), (False, True, True, 1))
        self.assertIn("上一响应不符合严格 JSON Schema", calls[1]["messages"][-1]["content"])

    def test_invalid_schema_gets_one_strict_retry(self):
        client, calls = queued_intake(['{"status":"ready"}', ready().model_dump_json()])
        self.assertEqual(client.assess("机器人一直卡住").status, "ready")
        self.assertEqual(len(calls), 2)

    def test_retry_still_invalid_fails_closed_with_metadata(self):
        client, calls = queued_intake(["NONE", "```json```"])
        with self.assertRaises(TroubleshootingIntakeError) as raised:
            client.assess("机器人无法正常工作")
        self.assertEqual(raised.exception.contract_metadata, {
            "initial_contract_success": False, "retry_contract_success": False,
            "final_contract_success": False, "retry_count": 1})
        self.assertEqual(len(calls), 2)

    def test_retry_is_bounded_to_one_even_when_more_responses_are_queued(self):
        client, calls = queued_intake(["bad", "still bad", ready().model_dump_json()])
        with self.assertRaises(TroubleshootingIntakeError): client.assess("机器人一直卡住")
        self.assertEqual(len(calls), 2)

    def test_explicit_error_code_promotes_model_clarification(self):
        client, calls = queued_intake([])
        result = client.assess_with_metadata("提示 E4")
        self.assertEqual(result.decision.status, "ready")
        self.assertEqual(result.decision.error_code, "E4")
        self.assertEqual(result.decision.symptom_summary, "提示 E4")
        self.assertEqual(result.guard_rule, "ERROR_CODE_FAST_PATH")
        self.assertTrue(result.deterministic_fast_path)
        self.assertEqual(calls, [])

    def test_recharge_and_brush_signals_promote_only_clarifications(self):
        for question in ("它回不了充电座", "主刷不转了"):
            with self.subTest(question=question):
                result = intake_client(safe_clarification_decision().model_dump_json())[0].assess_with_metadata(question)
                self.assertEqual(result.decision.status, "ready")
                self.assertIn(result.decision.symptom_summary, question)

    def test_generic_malfunction_never_promoted(self):
        for question in ("机器好像坏了", "机器人有问题", "运行异常", "机器不正常", "机器用不了", "T8 故障了"):
            with self.subTest(question=question):
                result = intake_client(safe_clarification_decision().model_dump_json())[0].assess_with_metadata(question)
                self.assertEqual(result.decision.status, "needs_clarification")
                self.assertIsNone(result.guard_rule)

    def test_latest_user_error_code_promotes_but_assistant_text_does_not(self):
        context = [{"role": "user", "content": "机器人出问题了"},
                   {"role": "assistant", "content": "你提到了 E4"}]
        result = intake_client(safe_clarification_decision().model_dump_json())[0].assess_with_metadata("提示 E4", context)
        self.assertEqual(result.decision.status, "ready")
        assistant_only = [{"role": "assistant", "content": "错误码 E4"}]
        actual, calls = queued_intake([safe_clarification_decision().model_dump_json()])
        self.assertEqual(actual.assess_with_metadata("机器人有问题", assistant_only).decision.status,
                         "needs_clarification")
        self.assertEqual(len(calls), 1)

    def test_guard_does_not_turn_ready_into_clarification(self):
        result = intake_client(ready("明确卡住").model_dump_json())[0].assess_with_metadata("机器人一直卡住")
        self.assertEqual(result.decision.status, "ready")
        self.assertIsNone(result.guard_rule)

    def test_explicit_symptom_guard_preserves_false_ready_zero_regressions(self):
        generic = ("机器坏了", "机器人有问题", "运行异常", "不正常", "用不了", "故障了")
        for question in generic:
            with self.subTest(question=question):
                assessment = intake_client(safe_clarification_decision().model_dump_json())[0].assess_with_metadata(question)
                self.assertNotEqual(assessment.decision.status, "ready")
        negative = intake_client(safe_clarification_decision().model_dump_json())[0].assess_with_metadata("没有漏水")
        self.assertEqual(negative.decision.status, "needs_clarification")

    def test_explicit_symptom_guard_does_not_fabricate_summary(self):
        result = intake_client(safe_clarification_decision().model_dump_json())[0].assess_with_metadata("扫着扫着会自己停下来")
        self.assertEqual(result.decision.symptom_summary, "扫着扫着会自己停下来")

    def test_contract_retry_transport_failure_stays_fail_closed(self):
        client, calls = queued_intake(["not json", TimeoutError("private response")])
        with self.assertRaises(TroubleshootingIntakeError) as raised: client.assess("机器人运行异常")
        self.assertNotIn("private", str(raised.exception))
        self.assertEqual(raised.exception.contract_metadata["retry_count"], 1)
        self.assertEqual(len(calls), 2)

    def test_code_and_common_fault_phrases_are_explicit_signals(self):
        for question, expected in (("错误码 5", "错误码 5"), ("机器一直漏水", "漏水"),
                                   ("放到充电座不充电", "放到充电座不充电"),
                                   ("WiFi 无法连接", "WiFi 无法连接")):
            with self.subTest(question=question):
                assessment = intake_client(safe_clarification_decision().model_dump_json())[0].assess_with_metadata(question)
                self.assertEqual(assessment.decision.status, "ready")
                self.assertEqual(assessment.decision.symptom_summary, expected)

    def test_previous_user_code_or_symptom_is_used_not_assistant_clarification(self):
        context = [{"role": "user", "content": "机器人出问题了"},
                   {"role": "assistant", "content": "请再描述具体异常"},
                   {"role": "user", "content": "提示 E4"}]
        assessment = intake_client(safe_clarification_decision().model_dump_json())[0].assess_with_metadata("然后呢？", context)
        self.assertEqual(assessment.decision.status, "ready")
        self.assertEqual(assessment.decision.error_code, "E4")
        self.assertEqual(assessment.decision.symptom_summary, "提示 E4")

    def test_retry_success_metadata_is_not_reported_as_initial_success(self):
        client, _ = queued_intake(["{", ready("明确卡住").model_dump_json()])
        result = client.assess_with_metadata("机器人一直卡住")
        self.assertFalse(result.initial_contract_success)
        self.assertTrue(result.retry_contract_success)
        self.assertTrue(result.final_contract_success)
        self.assertEqual(result.retry_count, 1)

    def test_e2_current_code_fast_path_and_case_normalization(self):
        for question in ("提示 E2", "显示 e2", "错误码 E2"):
            with self.subTest(question=question):
                intake, calls = queued_intake([])
                result = intake.assess_with_metadata(question)
                self.assertEqual(result.decision.status, "ready")
                self.assertEqual(result.decision.error_code, "E2")
                self.assertTrue(result.deterministic_fast_path)
                self.assertIsNone(result.initial_contract_success)
                self.assertEqual(calls, [])

    def test_generic_and_non_code_alphanumeric_inputs_still_call_model(self):
        for question in ("机器坏了", "有问题", "不正常", "T8", "P3", "abcE4x", "比如 E4"):
            with self.subTest(question=question):
                intake, calls = queued_intake([safe_clarification_decision().model_dump_json()])
                result = intake.assess_with_metadata(question)
                self.assertFalse(result.deterministic_fast_path)
                self.assertEqual(result.decision.status, "needs_clarification")
                self.assertEqual(len(calls), 1)

    def test_latest_user_error_code_fast_path_ignores_assistant_text(self):
        context = [{"role":"user", "content":"机器人出问题了"},
                   {"role":"assistant", "content":"请描述具体异常"},
                   {"role":"user", "content":"提示 E4"}]
        intake, calls = queued_intake([])
        result = intake.assess_with_metadata("然后呢？", context)
        self.assertEqual(result.decision.error_code, "E4")
        self.assertEqual(result.decision.symptom_summary, "提示 E4")
        self.assertEqual(calls, [])

    def test_api_fast_path_skips_intake_llm_but_retrieves_and_reviews_once(self):
        api, _, _, intake_mock, _, _, search, session = workflow([safe_clarification_decision()])
        actual, calls = queued_intake([])
        intake_mock.assess.side_effect = lambda question, conversation_context=None: actual.assess(question, conversation_context)
        response = api.post("/api/agent/chat", json={"question":"提示 E4"})
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json()["answer"])
        self.assertFalse(response.json()["published"])
        intake_mock.assess.assert_called_once()
        self.assertEqual(calls, [])
        search.invoke.assert_called_once()
        session.add.assert_called_once()
        session.commit.assert_called_once()

    def test_e4_followup_fast_path_restores_ready_in_same_session(self):
        api, _, memory, intake_mock, _, _, search, session = workflow([safe_clarification_decision()])
        actual, calls = queued_intake([])
        intake_mock.assess.side_effect = lambda question, conversation_context=None: (
            safe_clarification_decision() if question == "机器人出问题了。"
            else actual.assess(question, conversation_context))
        sid = "fault-e4-fast-path"
        first = api.post("/api/agent/chat", json={"question":"机器人出问题了。", "session_id":sid})
        second = api.post("/api/agent/chat", json={"question":"提示 E4", "session_id":sid})
        self.assertEqual(first.json()["answer"], TROUBLESHOOTING_CLARIFICATION)
        self.assertEqual(second.status_code, 200)
        self.assertIsNone(second.json()["answer"])
        self.assertEqual(calls, [])
        self.assertEqual(search.invoke.call_count, 1)
        self.assertEqual(session.commit.call_count, 1)
        self.assertEqual(search.invoke.call_args.args[0], {"question":"机器人出问题了。 提示 E4"})
        self.assertEqual(len(memory.get(sid)), 4)

    def test_clean_session_ambiguity_guard_is_true_and_clarification_bypasses_tool(self):
        self.assertTrue(needs_router_clarification("机器好像坏了。", []))
        api, _, _, intake, _, llm, search, session = workflow([], intent="clarify")
        response = api.post("/api/agent/chat", json={"question":"机器好像坏了。", "session_id":"clean-broken"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["intent"], "clarify")
        intake.assess.assert_not_called()
        llm.complete.assert_not_called()
        search.invoke.assert_not_called()
        session.commit.assert_not_called()

    def test_api_guarded_e4_uses_normal_ready_retrieval_and_single_review(self):
        api, _, _, intake_mock, _, _, search, session = workflow([safe_clarification_decision()])
        actual, _ = intake_client(safe_clarification_decision().model_dump_json())
        intake_mock.assess.side_effect = lambda question, conversation_context=None: actual.assess(question, conversation_context)
        response = api.post("/api/agent/chat", json={"question": "提示 E4"})
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json()["answer"])
        search.invoke.assert_called_once()
        session.commit.assert_called_once()


class TroubleshootingWorkflowTests(unittest.TestCase):
    def test_router_failure_generic_case_clarifies_without_tool_or_review(self):
        api, _, _, intake, router, llm, search, session = workflow(
            [safe_clarification_decision()]
        )
        router.classify.side_effect = LLMRouterError("sensitive upstream detail")

        response = api.post(
            "/api/agent/chat", json={"question": "机器好像坏了。"}
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "intent": "clarify",
            "answer": "请再描述一下具体异常现象，例如是否有语音提示、错误码，或者出现了什么行为异常。",
            "delivery_status": "clarification",
            "published": False,
        })
        self.assertNotIn("sensitive", response.text)
        router.classify.assert_called_once_with("机器好像坏了。")
        intake.assess.assert_not_called()
        llm.complete.assert_not_called()
        search.invoke.assert_not_called()
        session.add.assert_not_called()
        session.commit.assert_not_called()

    def test_clarification_is_visible_without_generation_or_review(self):
        api, _, _, intake, _, llm, search, session = workflow([safe_clarification_decision()])
        response = api.post("/api/agent/chat", json={"question": "机器人无法正常工作怎么办？"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"intent": "troubleshooting",
                                          "answer": TROUBLESHOOTING_CLARIFICATION,
                                          "delivery_status": "clarification", "published": False})
        intake.assess.assert_called_once()
        llm.complete.assert_not_called()
        search.invoke.assert_not_called()
        session.add.assert_not_called()
        session.commit.assert_not_called()

    def test_clarification_saves_only_visible_exchange_to_memory(self):
        api, _, memory, _, _, _, _, _ = workflow([safe_clarification_decision()])
        question = "机器人无法正常工作怎么办？"
        api.post("/api/agent/chat", json={"question": question, "session_id": "fault-a"})
        self.assertEqual(memory.get("fault-a"), [{"role": "user", "content": question},
            {"role": "assistant", "content": TROUBLESHOOTING_CLARIFICATION}])

    def test_intake_invalid_contract_fails_closed(self):
        api, _, _, _, _, llm, search, session = workflow([{"status": "ready"}])
        response = api.post("/api/agent/chat", json={"question": "机器人无法正常工作。"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], TROUBLESHOOTING_CLARIFICATION)
        llm.complete.assert_not_called()
        search.invoke.assert_not_called()
        session.commit.assert_not_called()

    def test_intake_timeout_fails_closed(self):
        api, _, _, _, _, llm, search, session = workflow([TimeoutError("secret token")])
        response = api.post("/api/agent/chat", json={"question": "机器运行异常。"})
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("secret", response.text)
        llm.complete.assert_not_called()
        search.invoke.assert_not_called()
        session.commit.assert_not_called()

    def test_followup_uses_history_and_restores_ready_path(self):
        api, _, memory, intake, router, _, search, session = workflow([safe_clarification_decision(), ready()])
        first, second = "机器人无法正常工作怎么办？", "扫着扫着会自己停下来。"
        api.post("/api/agent/chat", json={"question": first, "session_id": "recover"})
        response = api.post("/api/agent/chat", json={"question": second, "session_id": "recover"})
        self.assertEqual(response.status_code, 200)
        context = [{"role": "user", "content": first},
                   {"role": "assistant", "content": TROUBLESHOOTING_CLARIFICATION}]
        intake.assess.assert_called_with(second, conversation_context=context)
        router.classify.assert_called_with(second, conversation_context=context)
        search.invoke.assert_called_once_with({"question": f"{first} {second}"})
        session.commit.assert_called_once()
        self.assertEqual(len(memory.get("recover")), 4)

    def test_ready_path_retrieves_and_finalizes_once_and_hides_draft(self):
        api, _, memory, _, _, _, search, session = workflow([ready("一直卡住")])
        response = api.post("/api/agent/chat", json={"question": "机器人一直卡住怎么办？", "session_id": "ready"})
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json()["answer"])
        self.assertFalse(response.json()["published"])
        search.invoke.assert_called_once()
        session.add.assert_called_once()
        session.commit.assert_called_once()
        for field in ("symptom_summary", "error_code", "evidence", "draft_answer", "messages"):
            self.assertNotIn(field, response.json())
        self.assertNotIn("按本轮说明书", json.dumps(memory.get("ready"), ensure_ascii=False))

    def test_distinct_sessions_do_not_share_fault_context(self):
        api, _, memory, intake, _, _, _, _ = workflow([safe_clarification_decision(), safe_clarification_decision()])
        api.post("/api/agent/chat", json={"question": "T8 运行异常。", "session_id": "a"})
        api.post("/api/agent/chat", json={"question": "机器人无法正常工作。", "session_id": "b"})
        intake.assess.assert_called_with("机器人无法正常工作。", conversation_context=None)
        self.assertNotEqual(memory.get("a"), memory.get("b"))

    def test_other_four_business_scenes_skip_intake(self):
        for intent in ("consult_qa", "maintenance", "maintenance_report", "configuration_recommendation"):
            with self.subTest(intent=intent):
                api, _, _, intake, _, _, search, session = workflow([], intent=intent)
                response = api.post("/api/agent/chat", json={"question": "明确产品问题"})
                self.assertEqual(response.status_code, 200)
                intake.assess.assert_not_called()
                search.invoke.assert_called_once()
                session.commit.assert_called_once()

    def test_explicit_handoff_skips_intake_tool_and_review(self):
        api, _, _, intake, router, llm, search, session = workflow([])
        response = api.post("/api/agent/chat", json={"question": "我要人工客服"})
        self.assertEqual(response.json()["intent"], "human_handoff")
        for method in (intake.assess, router.classify, llm.complete, search.invoke, session.commit):
            method.assert_not_called()

    def test_unreviewed_model_clarifying_text_is_not_exposed(self):
        decision = safe_clarification_decision().model_copy(update={"clarifying_question": "请拆机维修并支付999元。"})
        api, _, _, _, _, _, _, _ = workflow([decision])
        response = api.post("/api/agent/chat", json={"question": "T8 故障了。"})
        self.assertEqual(response.json()["answer"], TROUBLESHOOTING_CLARIFICATION)
        self.assertNotIn("999", response.text)
