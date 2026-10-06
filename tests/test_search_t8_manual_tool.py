import unittest
from unittest.mock import Mock, patch

from pydantic import ValidationError

from app.agent.tools import (
    SEARCH_TOOL_NAME,
    build_search_t8_manual_tool,
)
from app.services.qa_service import normalize_retrieval_query


def fake_retrieval():
    retrieval = Mock()
    retrieval.create_pipeline.return_value = object()
    retrieval.retrieve.return_value = [
        (f"E{rank}", index, {
            "pdf_page": page,
            "title": "",
            "section": "日常维护",
            "text": f"第{page}页原文",
        })
        for rank, (index, page) in enumerate(((14, 15), (27, 23), (15, 16), (2, 3)), 1)
    ]
    return retrieval


class SearchT8ManualToolTests(unittest.TestCase):
    def test_original_question_is_normalized_for_production_retrieval(self):
        retrieval = fake_retrieval()
        original = "请问石头 T8 的主刷怎么清理？"
        with patch(
            "app.services.qa_service.normalize_retrieval_query",
            wraps=normalize_retrieval_query,
        ) as normalization:
            tool = build_search_t8_manual_tool(retrieval)
            result = tool.invoke({"question": original})

        normalization.assert_called_once_with(original)
        retrieval.create_pipeline.assert_called_once_with()
        retrieval.retrieve.assert_called_once_with("的主刷怎么清理？", retrieval.create_pipeline.return_value)
        self.assertEqual(result["retrieval_query"], "的主刷怎么清理？")

    def test_evidence_is_structured_and_limited_to_three(self):
        retrieval = fake_retrieval()
        result = build_search_t8_manual_tool(retrieval).invoke({"question": "主刷怎么清理？"})

        self.assertEqual([item["record_index"] for item in result["evidence"]], [14, 27, 15])
        self.assertEqual([item["evidence_id"] for item in result["evidence"]], ["E1", "E2", "E3"])
        self.assertEqual(result["evidence"][0], {
            "evidence_id": "E1", "record_index": 14, "pdf_page": 15,
            "title": "", "section": "日常维护", "text": "第15页原文",
        })
        self.assertEqual(set(result), {"retrieval_query", "evidence"})

    def test_repeated_calls_reuse_pipeline_without_qa_or_database_side_effects(self):
        retrieval = fake_retrieval()
        tool = build_search_t8_manual_tool(retrieval)
        with (
            patch("app.services.qa_service.GroundedQAService.process_question") as process,
            patch("app.services.grounded_qa.generate_answer") as generate,
            patch("app.services.grounded_qa.audit_answer") as audit,
            patch("app.database.SessionLocal") as session_factory,
        ):
            tool.invoke({"question": "主刷怎么清理？"})
            tool.invoke({"question": "滤网怎么清理？"})
        self.assertEqual(retrieval.retrieve.call_count, 2)
        retrieval.create_pipeline.assert_called_once_with()
        process.assert_not_called()
        generate.assert_not_called()
        audit.assert_not_called()
        session_factory.assert_not_called()

    def test_name_and_question_length_validation(self):
        tool = build_search_t8_manual_tool(fake_retrieval())
        self.assertEqual(tool.name, SEARCH_TOOL_NAME)
        self.assertEqual(set(tool.args), {"question"})
        with self.assertRaises(ValidationError):
            tool.invoke({"question": "   "})
        with self.assertRaises(ValidationError):
            tool.invoke({"question": "x" * 501})


if __name__ == "__main__":
    unittest.main()
