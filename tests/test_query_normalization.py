import unittest

from app.services.qa_service import normalize_retrieval_query


class ConservativeQueryNormalizationTests(unittest.TestCase):
    def test_plain_query_is_preserved(self):
        self.assertEqual(
            normalize_retrieval_query("主刷怎么清理？"),
            "主刷怎么清理？",
        )

    def test_removes_leading_polite_expression(self):
        self.assertEqual(
            normalize_retrieval_query("请问主刷怎么清理？"),
            "主刷怎么清理？",
        )

    def test_removes_brand_and_model_but_preserves_de(self):
        self.assertEqual(
            normalize_retrieval_query("石头 T8 的主刷怎么清理？"),
            "的主刷怎么清理？",
        )

    def test_combined_case_matches_experiment_rule(self):
        self.assertEqual(
            normalize_retrieval_query("请问石头 T8 的主刷怎么清理？"),
            "的主刷怎么清理？",
        )

    def test_determiner_is_not_removed_without_product_entity(self):
        question = "机器上的键有什么作用？"
        self.assertEqual(normalize_retrieval_query(question), question)

    def test_empty_normalized_result_falls_back_to_original_question(self):
        question = "请问石头 T8"
        result = normalize_retrieval_query(question)
        self.assertEqual(result, question)
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
