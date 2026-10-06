"""Real local Chroma tests; Qwen requests are mocked and production storage is untouched."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from app.retrieval import t8_retrieval as retrieval
from app.retrieval.t8_vector_store import (
    COLLECTION_NAME, T8VectorStore, VectorStoreError,
)


class T8VectorStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "chroma"
        self.records = [{"pdf_page": i + 1, "title": f"标题{i}",
                         "section": "使用说明", "text": f"知识正文{i}"}
                        for i in range(41)]
        self.vectors = [[1.0 if j == i else 0.01 for j in range(43)] for i in range(41)]
        self.store = T8VectorStore.build_index(self.records, "test-corpus-hash", self.vectors, self.path)

    def tearDown(self):
        self.store.client.close()
        self.temp.cleanup()

    def test_build_count_and_dimension(self):
        stats = self.store.validate_collection()
        self.assertEqual(stats["source_record_count"], 41)
        self.assertEqual(stats["stored_record_count"], 41)
        self.assertEqual(stats["embedding_dimension"], 43)
        self.assertEqual(stats["collection_name"], COLLECTION_NAME)

    def test_repeat_build_does_not_duplicate(self):
        self.store.client.close()
        self.store = T8VectorStore.build_index(self.records, "test-corpus-hash", self.vectors, self.path)
        self.assertEqual(self.store.collection.count(), 41)
        self.assertEqual(set(self.store.collection.get(include=[])["ids"]),
                         {f"t8_{i}" for i in range(41)})

    def test_metadata_and_documents_are_exact(self):
        row = self.store.collection.get(ids=["t8_14"], include=["documents", "metadatas"])
        self.assertEqual(row["documents"], ["知识正文14"])
        self.assertEqual(row["metadatas"], [{"record_index": 14, "pdf_page": 15,
                                             "title": "标题14", "section": "使用说明"}])
        self.assertIsNone(self.store.collection._embedding_function)

    def test_missing_string_metadata_is_safe(self):
        self.records[0]["title"] = None
        self.records[0].pop("section")
        self.store.client.close()
        self.store = T8VectorStore.build_index(self.records, "changed-hash", self.vectors, self.path)
        row = self.store.collection.get(ids=["t8_0"], include=["metadatas"])
        self.assertEqual(row["metadatas"][0]["title"], "")
        self.assertEqual(row["metadatas"][0]["section"], "")

    def test_top_k_and_record_alignment(self):
        rows = self.store.search_by_vector(self.vectors[7], limit=10)
        self.assertEqual(len(rows), 10)
        self.assertEqual(rows[0]["record_index"], 7)
        self.assertEqual(len({r["record_index"] for r in rows}), 10)
        for row in rows:
            expected = self.records[row["record_index"]]
            self.assertEqual(row["text"], expected["text"])
            self.assertEqual(row["pdf_page"], expected["pdf_page"])
            self.assertAlmostEqual(row["dense_score"], 1 - row["distance"])

    def test_dense_question_uses_only_existing_qwen_provider(self):
        with (patch.object(retrieval, "request_embeddings", return_value=[self.vectors[14]]) as embed,
              patch("app.retrieval.t8_vector_store.get_siliconflow_api_key", return_value="test-key")):
            rows = self.store.search_dense("的主刷怎么清理？", limit=10)
        embed.assert_called_once_with(["的主刷怎么清理？"], "test-key")
        self.assertEqual(rows[0]["record_index"], 14)

    def test_empty_query_and_invalid_limits_do_not_embed(self):
        with patch.object(retrieval, "request_embeddings") as embed:
            for query in ("", "  ", None):
                with self.subTest(query=query), self.assertRaises(ValueError):
                    self.store.search_dense(query)
            for limit in (0, -1, True, 1.5, "10", 42):
                with self.subTest(limit=limit), self.assertRaises(ValueError):
                    self.store.search_dense("主刷", limit)
            embed.assert_not_called()

    def test_missing_collection_fails_explicitly(self):
        with self.assertRaises(VectorStoreError):
            T8VectorStore(self.records, "test-corpus-hash", Path(self.temp.name) / "empty")

    def test_corpus_fingerprint_mismatch_fails(self):
        with self.assertRaises(VectorStoreError):
            T8VectorStore(self.records, "wrong-hash", self.path)

    def test_changed_text_is_rejected_even_with_same_fingerprint(self):
        self.records[0]["text"] = "changed text"
        with self.assertRaises(VectorStoreError):
            self.store.validate_collection()

    def test_missing_metadata_is_rejected(self):
        self.store.collection.update(ids=["t8_0"], metadatas=[{"record_index": 99}])
        with self.assertRaises(VectorStoreError):
            self.store.validate_collection()

    def test_duplicate_record_index_is_rejected(self):
        self.store.collection.update(ids=["t8_1"], metadatas=[{"record_index": 0}])
        with self.assertRaises(VectorStoreError):
            self.store.validate_collection()

    def test_count_change_fails_without_fallback(self):
        self.store.collection.delete(ids=["t8_40"])
        with self.assertRaises(VectorStoreError):
            self.store.search_by_vector(self.vectors[0])

    def test_invalid_document_vectors_never_write(self):
        for vectors in ([], [[float("nan")]] * 41, [[float("inf")]] * 41,
                        [[]] * 41, [[0.0]] * 41, [[1.0]] * 40 + [[1.0, 2.0]]):
            with self.subTest(vectors_type=str(type(vectors))):
                with patch("app.retrieval.t8_vector_store._client") as client:
                    with self.assertRaises(VectorStoreError):
                        T8VectorStore.build_index(self.records, "hash", vectors, self.path)
                    client.assert_not_called()

    def test_invalid_query_vector_fails(self):
        for vector in ([], [float("nan")], [float("inf")], [1.0], [0.0] * 43):
            with self.subTest(vector_size=len(vector)), self.assertRaises(VectorStoreError):
                self.store.search_by_vector(vector)

    def test_backend_query_error_is_safe(self):
        with patch.object(self.store.collection, "query", side_effect=RuntimeError("private password")):
            with self.assertRaises(VectorStoreError) as raised:
                self.store.search_by_vector(self.vectors[0])
        self.assertNotIn("password", str(raised.exception))

    def test_reload_uses_persisted_vectors_without_document_embedding(self):
        self.store.client.close()
        with patch.object(retrieval, "request_embeddings") as embed:
            self.store = T8VectorStore(self.records, "test-corpus-hash", self.path)
            self.assertEqual(self.store.collection.count(), 41)
            self.assertEqual(self.store.search_by_vector(self.vectors[14])[0]["record_index"], 14)
            embed.assert_not_called()

    def test_pipeline_loads_chroma_instead_of_cache(self):
        with (patch.object(retrieval, "load_corpus", return_value=(self.records, "test-corpus-hash")),
              patch.object(retrieval, "build_bm25_index", return_value=Mock()),
              patch.object(retrieval, "get_siliconflow_api_key", return_value="test-key"),
              patch.object(retrieval, "load_cached_document_vectors") as cache,
              patch("app.retrieval.t8_vector_store.T8VectorStore", return_value=self.store) as store):
            pipeline = retrieval.create_pipeline()
        store.assert_called_once_with(self.records, "test-corpus-hash")
        cache.assert_not_called()
        self.assertIs(pipeline[4], self.store)


if __name__ == "__main__":
    unittest.main()
