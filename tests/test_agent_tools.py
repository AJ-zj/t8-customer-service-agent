import unittest
from unittest.mock import Mock

from app.agent.tools import (
    SAFE_TOOL_NOTICE,
    TOOL_DESCRIPTION,
    TOOL_NAME,
    build_t8_knowledge_qa_tool,
)


class T8KnowledgeQAToolTests(unittest.TestCase):
    def make_tool(self, qa_service):
        qa_service.process_question.return_value = {
            "generation_status": "generated",
            "audit_status": "NEEDS_HUMAN_SEMANTIC_REVIEW",
            "draft_answer": "private draft must not escape",
            "evidence": [{"text": "private evidence must not escape"}],
        }
        return build_t8_knowledge_qa_tool(qa_service)

    def test_tool_calls_grounded_qa_service_once(self):
        qa_service = Mock()
        tool = self.make_tool(qa_service)

        tool.invoke({"question": "主刷怎么清理？"})

        qa_service.process_question.assert_called_once()

    def test_tool_passes_original_question_unchanged(self):
        qa_service = Mock()
        tool = self.make_tool(qa_service)
        original_question = "请问石头 T8 的主刷怎么清理？"

        tool.invoke({"question": original_question})

        qa_service.process_question.assert_called_once_with(original_question)

    def test_tool_output_filters_draft_and_evidence(self):
        tool = self.make_tool(Mock())

        result = tool.invoke({"question": "主刷怎么清理？"})

        self.assertEqual(result["generation_status"], "generated")
        self.assertEqual(result["audit_status"], "NEEDS_HUMAN_SEMANTIC_REVIEW")
        self.assertFalse(result["published"])
        self.assertEqual(result["safety_notice"], SAFE_TOOL_NOTICE)
        self.assertNotIn("draft_answer", result)
        self.assertNotIn("evidence", result)
        self.assertNotIn("private draft", str(result))
        self.assertNotIn("private evidence", str(result))

    def test_tool_has_explicit_name_description_and_question_schema(self):
        tool = self.make_tool(Mock())

        self.assertEqual(tool.name, TOOL_NAME)
        self.assertIn("石头 T8 官方说明书", tool.description)
        self.assertIn("不联网", TOOL_DESCRIPTION)
        self.assertIn("question", tool.args)
        self.assertEqual(set(tool.args), {"question"})


if __name__ == "__main__":
    unittest.main()
