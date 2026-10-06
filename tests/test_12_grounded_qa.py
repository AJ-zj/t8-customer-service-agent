import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from app.services import grounded_qa as qa


def evidence():
    return [
        ("E1", 17, {"pdf_page": 17, "text": "请用清水清洗滤网，不要添加洗涤剂。"}),
        ("E2", 18, {"pdf_page": 18, "text": "晾晒滤网至干透后装回使用。"}),
        ("E3", 24, {"pdf_page": 24, "text": "T8 三包凭证和保修期限。"}),
    ]


class GroundedAnswerAuditTests(unittest.TestCase):
    def test_chat_url_is_plain_api_url(self):
        self.assertEqual(qa.CHAT_URL, "https://api.siliconflow.cn/v1/chat/completions")
        self.assertFalse(qa.CHAT_URL.startswith("["))

    def test_ascii_and_chinese_citation_brackets_are_recognized(self):
        result = qa.audit_answer("滤网怎么清洗？", "清水清洗 [E1]，晾干后装回【E2】。", evidence())
        self.assertEqual(result["citation_status"], "VALID")
        self.assertEqual(result["citation_ids"], ["E1", "E2"])
        self.assertEqual(result["status"], "NEEDS_HUMAN_SEMANTIC_REVIEW")
        self.assertEqual(result["semantic_support"], "NOT_VERIFIED")

    def test_nonexistent_citation_is_blocked(self):
        result = qa.audit_answer("问题", "说明书写了这个操作 [E4]。", evidence())
        self.assertEqual(result["status"], "PROGRAM_BLOCKED_INVALID_CITATION")
        self.assertEqual(result["invalid_citations"], ["E4"])

    def test_irrelevant_but_legal_citation_requires_human_review(self):
        result = qa.audit_answer("T8 官方价格是多少？", "滤网请用清水清洗 [E1]。", evidence())
        self.assertEqual(result["citation_status"], "VALID")
        self.assertEqual(result["status"], "NEEDS_HUMAN_SEMANTIC_REVIEW")

    def test_unsupported_feature_claim_is_not_auto_approved(self):
        result = qa.audit_answer("T8 有什么功能？", "T8 支持自动抬升拖布 [E3]。", evidence())
        self.assertEqual(result["status"], "NEEDS_HUMAN_SEMANTIC_REVIEW")
        self.assertEqual(result["semantic_support"], "NOT_VERIFIED")

    def test_abstention_followed_by_unsupported_price_is_blocked(self):
        result = qa.audit_answer(
            "现在官方卖多少钱？", "无法确认，不过现在售价是 2499 元 [E3]。", evidence()
        )
        self.assertEqual(result["status"], "PROGRAM_BLOCKED_UNSUPPORTED_MONEY_CLAIM")
        chinese_amount = qa.audit_answer(
            "现在官方卖多少钱？", "无法确认，但价格是人民币两千元 [E3]。", evidence()
        )
        self.assertEqual(chinese_amount["status"], "PROGRAM_BLOCKED_UNSUPPORTED_MONEY_CLAIM")

    def test_invented_personal_runtime_is_blocked(self):
        result = qa.audit_answer(
            "我家 T8 昨晚清扫用了多久？",
            "我无法确认，但你家 T8 昨晚清扫用了 68 分钟 [E1]。", evidence(),
        )
        self.assertEqual(result["status"], "PROGRAM_BLOCKED_UNSUPPORTED_PERSONAL_RUNTIME")

    def test_refusal_marker_alone_is_not_a_safe_pass(self):
        result = qa.audit_answer("现在官方卖多少钱？", "无法确认当前售价。", evidence())
        self.assertEqual(result["status"], "MODEL_ABSTAINED_NEEDS_REVIEW")
        self.assertEqual(result["semantic_support"], "NOT_VERIFIED")
        self.assertEqual(result["citation_status"], "MISSING")

    def test_uncited_assertion_is_blocked_and_raw_output_is_retained(self):
        answer = "机器人支持自动集尘。"
        result = qa.audit_answer("T8 有什么功能？", answer, evidence())
        self.assertEqual(result["status"], "PROGRAM_BLOCKED_MISSING_CITATION")
        self.assertEqual(result["raw_answer"], answer)

    def test_terminal_labels_raw_output_as_unverified(self):
        answer = "滤网要用清水清洗 [E1]。"
        output = io.StringIO()
        with redirect_stdout(output):
            result = qa.show_audit_result("滤网怎么洗？", answer, evidence())
        rendered = output.getvalue()
        self.assertEqual(result["status"], "NEEDS_HUMAN_SEMANTIC_REVIEW")
        self.assertIn("模型原始输出（仅供本地审查", rendered)
        self.assertIn(answer, rendered)
        self.assertNotIn("最终回答：", rendered)
    def test_blocked_answer_is_not_reported_success(self):
        blocked = {
            "status": "PROGRAM_BLOCKED_INVALID_CITATION"
        }

        with patch.object(qa, "answer_one", return_value=blocked):
            result = qa.run_query_safely("测试问题", None)

        self.assertFalse(result)

    def test_cited_answer_with_local_abstention_is_not_a_full_refusal(self):
        result = qa.audit_answer("滤网怎么洗？", "请用清水清洗滤网 [E1]。其他细节无法确认。", evidence())
        self.assertEqual(result["status"], "NEEDS_HUMAN_SEMANTIC_REVIEW")
        self.assertEqual(result["semantic_support"], "NOT_VERIFIED")

    def test_citation_after_sentence_then_local_abstention_is_mixed(self):
        result = qa.audit_answer("滤网怎么洗？", "请用清水清洗滤网。[E1]\n晾晒时长无法确认。", evidence())
        self.assertEqual(result["status"], "NEEDS_HUMAN_SEMANTIC_REVIEW")

    def test_opening_refusal_with_unrelated_cited_facts_requires_semantic_gate(self):
        result = qa.audit_answer("现在电量多少？", "无法确认当前电量。滤网应晾干再装回 [E2]。", evidence())
        self.assertEqual(result["status"], "NEEDS_HUMAN_SEMANTIC_REVIEW")
        self.assertEqual(result["semantic_support"], "NOT_VERIFIED")
        from app.services.answer_release_policy import AnswerReleasePolicy, GateOutcome

        inputs = dict(scene="consult_qa", generation_status="generated", audit_status=result["status"],
                      draft_answer=result["raw_answer"], evidence=[row[2] for row in evidence()],
                      citation_status=result["citation_status"])
        policy = AnswerReleasePolicy()
        self.assertEqual(policy.decide(**inputs).reason_code, "SEMANTIC_GATE_REQUIRED")
        self.assertEqual(policy.decide(**inputs, gate_outcome=GateOutcome(
            "unsupported", "INSUFFICIENT_EVIDENCE")).status, "review_required")

    def test_citation_within_unfinished_refusal_does_not_make_an_answer(self):
        result = qa.audit_answer("多少钱？", "根据说明书 [E1]，无法确认当前价格。", evidence())
        self.assertEqual(result["status"], "MODEL_ABSTAINED_NEEDS_REVIEW")

    def test_mixed_answer_does_not_rescue_unsupported_money(self):
        result = qa.audit_answer("滤网怎么洗？", "清水清洗 [E1]。其他无法确认，但配件售价299元。", evidence())
        self.assertEqual(result["status"], "PROGRAM_BLOCKED_UNSUPPORTED_MONEY_CLAIM")


class MixedAnswerOrderRegressionTests(unittest.TestCase):
    def audit(self, answer):
        return qa.audit_answer("主刷多久清理一次？", answer, [
            ("E1", 13, {"pdf_page": 14, "text": "软胶主刷 每2周清理。"}),
        ])

    def assert_mixed(self, answer):
        result = self.audit(answer)
        self.assertTrue(qa.ABSTENTION_PATTERN.search(answer))
        self.assertTrue(qa._has_cited_answer_content(answer, {"E1"}))
        self.assertEqual(result["status"], "NEEDS_HUMAN_SEMANTIC_REVIEW")
        self.assertEqual(result["semantic_support"], "NOT_VERIFIED")
        self.assertTrue(any("局部无法确认" in reason for reason in result["reasons"]))

    def test_answer_before_caveat(self):
        self.assert_mixed("主刷每两周清理一次。[E1] 说明书未明确是否每天检查。")

    def test_caveat_before_answer(self):
        for answer in (
            "说明书未明确是否每天检查，但主刷每两周清理一次。[E1]",
            "说明书未明确是否每天检查，\n但主刷每两周清理一次。【E1】",
        ):
            with self.subTest(answer=answer):
                self.assert_mixed(answer)

    def test_pure_abstention(self):
        result = self.audit("根据当前提供的说明书资料无法确认。")
        self.assertEqual(result["status"], "MODEL_ABSTAINED_NEEDS_REVIEW")

    def test_cited_abstention_is_not_an_answer(self):
        for answer in (
            "说明书未提及该错误码含义。[E1]",
            "根据说明书无法确认该错误码含义。[E1]",
            "根据说明书 [E1]，无法确认该错误码含义。",
            "当前资料不足。【E1】",
        ):
            with self.subTest(answer=answer):
                self.assertFalse(qa._has_cited_answer_content(answer, {"E1"}))
                self.assertEqual(self.audit(answer)["status"], "MODEL_ABSTAINED_NEEDS_REVIEW")

    def test_price_caveat_with_cited_answer_is_mixed_but_unverified(self):
        self.assert_mixed("无法确认具体价格；主刷每两周清理一次。[E1]")

    def test_normal_cited_answer_remains_unverified(self):
        result = self.audit("主刷每两周清理一次。[E1]")
        self.assertEqual(result["status"], "NEEDS_HUMAN_SEMANTIC_REVIEW")
        self.assertEqual(result["semantic_support"], "NOT_VERIFIED")

    def test_normal_uncited_answer_is_blocked(self):
        result = self.audit("主刷每两周清理一次。")
        self.assertEqual(result["status"], "PROGRAM_BLOCKED_MISSING_CITATION")

    def test_invalid_citation_takes_priority_over_mixed_classification(self):
        answer = "说明书未明确是否每天检查，但主刷每两周清理一次。[E4]"
        self.assertFalse(qa._has_cited_answer_content(answer, {"E1"}))
        self.assertEqual(self.audit(answer)["status"], "PROGRAM_BLOCKED_INVALID_CITATION")


if __name__ == "__main__":
    unittest.main()
