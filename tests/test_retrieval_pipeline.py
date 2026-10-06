import unittest
from unittest.mock import Mock, patch

from app.retrieval import t8_retrieval


class ProductionRetrievalPipelineTests(unittest.TestCase):
    def test_rrf_selects_reranker_pool_and_evidence_remains_top_three(self):
        records = [
            {"pdf_page": index + 1, "title": f"record-{index}", "text": "test"}
            for index in range(20)
        ]
        bm25_candidates = [(index, index + 1) for index in range(10)]
        embedding_candidates = [(index + 10, index + 1) for index in range(10)]
        captured = {}
        vector_store = Mock()
        vector_store.search_dense.return_value = [
            {"record_index": index} for index, _ in embedding_candidates
        ]

        def fake_rerank(_query, candidates, _api_key):
            captured["record_ids"] = [record_id for record_id, _ in candidates]
            return [
                (record_id, float(len(candidates) - rank))
                for rank, (record_id, _) in enumerate(candidates)
            ]

        pipeline = (
            None,
            None,
            records,
            object(),
            vector_store,
            "test-only-key",
        )
        with (
            patch.object(
                t8_retrieval,
                "rank_bm25",
                return_value=([0.0] * len(records), {}, bm25_candidates),
            ),
            patch.object(
                t8_retrieval,
                "reciprocal_rank_fusion",
                wraps=t8_retrieval.reciprocal_rank_fusion,
            ) as fusion,
            patch.object(t8_retrieval, "request_rerank", side_effect=fake_rerank),
        ):
            evidence = t8_retrieval.retrieve("test query", pipeline)

        self.assertEqual(t8_retrieval.BM25_CANDIDATE_LIMIT, 10)
        self.assertEqual(t8_retrieval.EMBEDDING_CANDIDATE_LIMIT, 10)
        self.assertEqual(t8_retrieval.RRF_K, 60)
        self.assertEqual(t8_retrieval.RERANK_CANDIDATE_LIMIT, 10)
        self.assertEqual(t8_retrieval.RERANK_TOP_K, 3)
        fusion.assert_called_once_with(bm25_candidates, embedding_candidates)
        vector_store.search_dense.assert_called_once_with("test query", limit=10)
        self.assertEqual(len(captured["record_ids"]), 10)
        # With equal ranks in disjoint lists, RRF ties are resolved by record index.
        self.assertEqual(captured["record_ids"], [0, 10, 1, 11, 2, 12, 3, 13, 4, 14])
        self.assertEqual(len(evidence), 3)
        self.assertEqual([item[1] for item in evidence], [0, 10, 1])
        self.assertEqual([item[0] for item in evidence], ["E1", "E2", "E3"])


if __name__ == "__main__":
    unittest.main()
